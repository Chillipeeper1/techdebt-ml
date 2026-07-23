"""
predict.py - analyzes a local repo and predicts the technical debt per file
"""

import argparse
import sys
import os
import shutil
import stat
import tempfile
from pathlib import Path
import joblib
from git import Repo
from git.exc import InvalidGitRepositoryError, NoSuchPathError
from datetime import datetime
import re
import pandas as pd
from feature_lib import features_ats, features_radon, feature_git_raw, normalization_relative_to_repo

SRC_DIR = Path(__file__).resolve().parent
MODELS_DIR = SRC_DIR.parent / "models"
REPORTES_DIR = SRC_DIR.parent / "reportes_temp"

# Permite importar repo_search.py (mismo directorio) sin importar el CWD.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

feat_list = joblib.load(MODELS_DIR / "features_v2.pkl")
MODEL = joblib.load(MODELS_DIR / "random_forest_v2.pkl")

CONFIG = {
    "skip_dirs": ["test", ".venv"],
}


def validar_repo(ruta: str) -> tuple[Path, Repo]:
    repo_path = Path(ruta).resolve()

    if not repo_path.exists():
        print(f"Error: la ruta '{repo_path}' no existe.")
        sys.exit(1)

    try:
        repo_git = Repo(repo_path)
    except (InvalidGitRepositoryError, NoSuchPathError):
        print(f"Error: '{repo_path}' no es un repositorio git válido.")
        sys.exit(1)

    return repo_path, repo_git


def buscar_archivos_py(ruta_repo: Path) -> list[str]:
    return [
        str(f.relative_to(ruta_repo)) for f in ruta_repo.rglob("*.py")
        if not any(part in CONFIG["skip_dirs"] for part in f.parts)
    ]


def imprimir_resumen(ruta_repo: Path, archivos: list[str]):
    print("\n" + "="*60)
    print(f"  ARCHIVOS ENCONTRADOS EN {ruta_repo.name}")
    print("="*60)
    print(f"  Total: {len(archivos)} archivos .py")
    print("="*60)

    for archivo in archivos[:15]:
        print(f"  {archivo}")

    if len(archivos) > 15:
        print(f"  ... y {len(archivos) - 15} más")

    print("="*60)

def feature_extraction(path, files, repo_git, repo_name):
    results = {}
    for file in files:
        author_list = set()
        churn_rate_total = 0
        try:
            with open(path / file, 'r', encoding='utf-8') as f:

                feat_file = f.read()
                features = {}
                features.update(features_radon(feat_file))
                features.update(features_ats(feat_file))
                features.update(feature_git_raw(file,repo_git))
                features["repo"] = repo_name

                # Feat list for csv
                results[f"{repo_name}/{file}"] = features
        except Exception as e:
            print(f"corrupted file: {file} , {e}")
            continue

    return results


def prediction(results:dict) -> pd.DataFrame:
    rows = []
    for file, features in results.items():
        X = pd.DataFrame([features])[feat_list]
        prob = MODEL.predict_proba(X)[:, 1][0]

        rows.append({
            "files" : file,
            "danger":round(prob,3)
        })
    return pd.DataFrame(rows).sort_values("danger",ascending=False)


def reporte_completo(results: dict) -> pd.DataFrame:
    """Tabla con los valores crudos de cada archivo más el riesgo predicho,
    ordenada del más riesgoso al menos riesgoso."""
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "file"
    df = normalization_relative_to_repo(df, repo_col="repo")
    # Riesgo (probabilidad de deuda técnica) por archivo.
    X = df[feat_list]
    df.insert(0, "danger", MODEL.predict_proba(X)[:, 1].round(3))

    return df.sort_values("danger", ascending=False).reset_index()


def guardar_reporte_csv(reporte: pd.DataFrame, nombre_repo: str, destino: Path = REPORTES_DIR) -> Path:
    """Guarda el reporte como CSV en la carpeta de reportes y devuelve la ruta
    del archivo generado. El nombre lleva timestamp para no pisar ejecuciones
    anteriores del mismo repo."""
    destino.mkdir(parents=True, exist_ok=True)

    # El nombre del repo viene de GitHub, así que se limpia para usarlo como archivo.
    nombre_limpio = re.sub(r"[^A-Za-z0-9._-]", "_", nombre_repo) or "repo"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_csv = destino / f"reporte_{nombre_limpio}_{timestamp}.csv"

    reporte.to_csv(ruta_csv, index=False, encoding="utf-8")
    return ruta_csv


def _borrar_arbol(ruta: Path):
    """Borra un árbol de directorios. En Windows los objetos de .git quedan como
    solo-lectura, así que se les quita ese flag antes de reintentar el borrado."""
    def _quitar_readonly(func, path, _exc_info):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    shutil.rmtree(ruta, onerror=_quitar_readonly)


def obtener_y_clonar_repo(max_intentos: int = 5):
    """Busca un repo aleatorio en GitHub, lo clona en una carpeta temporal y
    valida que tenga archivos .py. Reintenta con otro repo si el clon falla o el
    repo no tiene código Python utilizable. Devuelve
    (tmp_dir, ruta_repo, repo_git, nombre_repo, archivos) o None."""
    try:
        from repo_search import obtener_repo_aleatorio
    except EnvironmentError as e:
        print(f"Error: {e}")
        return None

    urls_probadas = set()

    for intento in range(1, max_intentos + 1):
        repo_info = obtener_repo_aleatorio()
        if repo_info is None:
            print(f"  [{intento}/{max_intentos}] No se obtuvo ningún repo. Reintentando...")
            continue

        clone_url = repo_info["clone_url"]
        if clone_url in urls_probadas:
            continue
        urls_probadas.add(clone_url)

        tmp_dir = Path(tempfile.mkdtemp(prefix="techdebt_"))
        dest = tmp_dir / repo_info["name"]

        try:
            print(f"  [{intento}/{max_intentos}] Clonando {repo_info['full_name']} ...")
            repo_git = Repo.clone_from(clone_url, dest)
        except Exception as e:
            print(f"    Falló el clon: {e}")
            _borrar_arbol(tmp_dir)
            continue

        archivos = buscar_archivos_py(dest)
        if not archivos:
            print("    El repo no tiene archivos .py utilizables. Probando con otro...")
            repo_git.close()
            _borrar_arbol(tmp_dir)
            continue

        return tmp_dir, dest, repo_git, repo_info["name"], archivos

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Analiza un repositorio y predice la deuda técnica por archivo. "
                    "Sin argumentos, busca y clona un repo aleatorio de GitHub."
    )
    parser.add_argument(
        "repo_path",
        nargs="?",
        default=None,
        help="Ruta a un repo local (opcional). Si se omite, se busca y clona un "
             "repo aleatorio de GitHub.",
    )
    args = parser.parse_args()

    tmp_dir = None
    repo_git = None
    try:
        if args.repo_path:
            ruta_repo, repo_git = validar_repo(args.repo_path)
            nombre_repo = ruta_repo.name
            archivos = buscar_archivos_py(ruta_repo)
            if not archivos:
                print(f"No se encontraron archivos .py en '{ruta_repo}'.")
                return
        else:
            print("Buscando un repo aleatorio en GitHub...")
            resultado = obtener_y_clonar_repo()
            if resultado is None:
                print("No se pudo obtener un repo válido de GitHub tras varios intentos.")
                return
            tmp_dir, ruta_repo, repo_git, nombre_repo, archivos = resultado

        imprimir_resumen(ruta_repo, archivos)

        resultados = feature_extraction(ruta_repo, archivos, repo_git, nombre_repo)
        print(f"\nFeatures extraídas de {len(resultados)} archivos.")

        if not resultados:
            print("No se pudieron extraer features de ningún archivo.")
            return

        reporte = reporte_completo(resultados)
        print("\n" + "=" * 60)
        print("  RIESGO + VALORES CRUDOS (mayor a menor riesgo)")
        print("=" * 60)
        print(reporte.to_string(index=False))

        ruta_csv = guardar_reporte_csv(reporte, nombre_repo)
        print(f"\nReporte guardado en: {ruta_csv}")
    finally:
        if repo_git is not None:
            repo_git.close()
        if tmp_dir is not None:
            _borrar_arbol(tmp_dir)
            print("\nRepo temporal eliminado.")


if __name__ == "__main__":
    main()

