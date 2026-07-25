"""
predict.py - analyzes a local repo and predicts the technical debt per file

Uses model v3: tabular features (Radon + AST + relative git activity) combined
with PCA-reduced code embeddings (see docs/decisions/004-embeddings-tradeoff.md).
"""

import argparse
import sys
import os
import shutil
import stat
import tempfile
import re
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from git import Repo
from git.exc import InvalidGitRepositoryError, NoSuchPathError
from sentence_transformers import SentenceTransformer

from feature_lib import (
    features_ats,
    features_radon,
    feature_git_raw,
    normalization_relative_to_repo,
    embedding_extract_functions,
)

SRC_DIR = Path(__file__).resolve().parent
MODELS_DIR = SRC_DIR.parent / "models"
REPORTS_DIR = SRC_DIR.parent / "temp_reports"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Oficial Artifacts: v3 (embeddings + PCA)
feat_list = joblib.load(MODELS_DIR / "features_v3.pkl")
MODEL = joblib.load(MODELS_DIR / "random_forest_v3.pkl")
PCA_MODEL = joblib.load(MODELS_DIR / "pca_v3.pkl")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

CONFIG = {
    "skip_dirs": ["test", ".venv"],
}


def validate_repo(path: str) -> tuple[Path, Repo]:
    repo_path = Path(path).resolve()

    if not repo_path.exists():
        print(f"Error: the path '{repo_path}' does not exist.")
        sys.exit(1)

    try:
        repo_git = Repo(repo_path)
    except (InvalidGitRepositoryError, NoSuchPathError):
        print(f"Error: '{repo_path}' is not a valid git repository.")
        sys.exit(1)

    return repo_path, repo_git


def find_python_files(repo_path: Path) -> list[str]:
    return [
        str(f.relative_to(repo_path)) for f in repo_path.rglob("*.py")
        if not any(part in CONFIG["skip_dirs"] for part in f.parts)
    ]


def print_summary(repo_path: Path, files: list[str]):
    print("\n" + "=" * 60)
    print(f"  FILES FOUND IN {repo_path.name}")
    print("=" * 60)
    print(f"  Total: {len(files)} .py files")
    print("=" * 60)

    for file in files[:15]:
        print(f"  {file}")

    if len(files) > 15:
        print(f"  ... and {len(files) - 15} more")

    print("=" * 60)


def feature_extraction(path, files, repo_git, repo_name):
    resultados = {}
    todas_las_funciones = []
    indice_por_archivo = []

    for archivo in files:
        try:
            with open(path / archivo, "r", encoding="utf-8") as f:
                feat_file = f.read()

                features = {}
                features.update(features_radon(feat_file))
                features.update(features_ats(feat_file))
                features.update(feature_git_raw(archivo, repo_git))
                features["repo"] = repo_name

                # Ruta normalizada — evita el desajuste backslash/forward-slash
                # entre este dict y las claves de embeddings_por_archivo.
                clave = f"{repo_name}/{archivo}".replace("\\", "/")
                resultados[clave] = features

                funciones_archivo = embedding_extract_functions(feat_file)
                indice_por_archivo.append((clave, len(funciones_archivo)))
                todas_las_funciones.extend(funciones_archivo)

        except Exception as e:
            print(f"corrupted file: {archivo} , {e}")
            continue

    return resultados, todas_las_funciones, indice_por_archivo


def generar_embeddings_reducidos(todas_las_funciones, indice_por_archivo):
    """Genera embeddings por función (batched), hace mean pooling por archivo.
    NO aplica PCA aquí — eso ocurre en full_report, sobre la matriz completa."""
    if todas_las_funciones:
        embeddings_funciones = EMBED_MODEL.encode(todas_las_funciones, batch_size=64)
    else:
        embeddings_funciones = np.empty((0, 384))

    embeddings_por_archivo = {}
    cursor = 0
    for clave, cantidad in indice_por_archivo:
        if cantidad == 0:
            embeddings_por_archivo[clave] = np.zeros(384)
            continue
        bloque = embeddings_funciones[cursor: cursor + cantidad]
        embeddings_por_archivo[clave] = bloque.mean(axis=0)
        cursor += cantidad

    return embeddings_por_archivo


def full_report(results: dict, embeddings_por_archivo: dict) -> pd.DataFrame:
    """Combina features tabulares (normalizados relativo al repo) con embeddings
    reducidos por PCA (ya ajustado en entrenamiento — aquí solo .transform())."""
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "file"

    df = normalization_relative_to_repo(df, repo_col="repo")

    X_tabular = df[feat_list].values
    matriz_embeddings = np.array([embeddings_por_archivo[clave] for clave in df.index])
    X_embeddings_reducido = PCA_MODEL.transform(matriz_embeddings)  # transform, NUNCA fit_transform

    X_final = np.hstack([X_tabular, X_embeddings_reducido])
    df.insert(0, "danger", MODEL.predict_proba(X_final)[:, 1].round(3))

    return df.sort_values("danger", ascending=False).reset_index()


def save_csv_report(report: pd.DataFrame, repo_name: str, destination: Path = REPORTS_DIR) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    clean_name = re.sub(r"[^A-Za-z0-9._-]", "_", repo_name) or "repo"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = destination / f"report_{clean_name}_{timestamp}.csv"
    report.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def _delete_tree(path: Path):
    def _remove_readonly(func, path, _exc_info):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    shutil.rmtree(path, onerror=_remove_readonly)


def fetch_and_clone_repo(max_attempts: int = 5):
    try:
        from repo_search import get_random_repo
    except EnvironmentError as e:
        print(f"Error: {e}")
        return None

    tried_urls = set()

    for attempt in range(1, max_attempts + 1):
        repo_info = get_random_repo()
        if repo_info is None:
            print(f"  [{attempt}/{max_attempts}] No repo was obtained. Retrying...")
            continue

        clone_url = repo_info["clone_url"]
        if clone_url in tried_urls:
            continue
        tried_urls.add(clone_url)

        tmp_dir = Path(tempfile.mkdtemp(prefix="techdebt_"))
        dest = tmp_dir / repo_info["name"]

        try:
            print(f"  [{attempt}/{max_attempts}] Cloning {repo_info['full_name']} ...")
            repo_git = Repo.clone_from(clone_url, dest)
        except Exception as e:
            print(f"    Clone failed: {e}")
            _delete_tree(tmp_dir)
            continue

        files = find_python_files(dest)
        if not files:
            print("    The repo has no usable .py files. Trying another one...")
            repo_git.close()
            _delete_tree(tmp_dir)
            continue

        return tmp_dir, dest, repo_git, repo_info["name"], files

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyzes a repository and predicts the technical debt per file. "
                    "Without arguments, it looks for and clones a random GitHub repo."
    )
    parser.add_argument(
        "repo_path", nargs="?", default=None,
        help="Path to a local repo (optional). If omitted, a random GitHub repo "
             "is looked up and cloned.",
    )
    args = parser.parse_args()

    tmp_dir = None
    repo_git = None
    try:
        if args.repo_path:
            repo_path, repo_git = validate_repo(args.repo_path)
            repo_name = repo_path.name
            files = find_python_files(repo_path)
            if not files:
                print(f"No .py files were found in '{repo_path}'.")
                return
        else:
            print("Looking for a random repo on GitHub...")
            result = fetch_and_clone_repo()
            if result is None:
                print("Could not obtain a valid GitHub repo after several attempts.")
                return
            tmp_dir, repo_path, repo_git, repo_name, files = result

        print_summary(repo_path, files)

        results, functions_list, index_list = feature_extraction(
            repo_path, files, repo_git, repo_name
        )
        print(f"\nFeatures extracted from {len(results)} files.")

        if not results:
            print("Could not extract features from any file.")
            return

        embeddings_por_archivo = generar_embeddings_reducidos(functions_list, index_list)
        report = full_report(results, embeddings_por_archivo)

        print("\n" + "=" * 60)
        print("  RISK + RAW VALUES (highest to lowest risk)")
        print("=" * 60)
        print(report.to_string(index=False))

        csv_path = save_csv_report(report, repo_name)
        print(f"\nReport saved to: {csv_path}")
    finally:
        if repo_git is not None:
            repo_git.close()
        if tmp_dir is not None:
            _delete_tree(tmp_dir)
            print("\nTemporary repo deleted.")


if __name__ == "__main__":
    main()