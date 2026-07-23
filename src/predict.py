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
REPORTS_DIR = SRC_DIR.parent / "temp_reports"

# Allows importing repo_search.py (same directory) regardless of the CWD.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

feat_list = joblib.load(MODELS_DIR / "features_v2.pkl")
MODEL = joblib.load(MODELS_DIR / "random_forest_v2.pkl")

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
    print("\n" + "="*60)
    print(f"  FILES FOUND IN {repo_path.name}")
    print("="*60)
    print(f"  Total: {len(files)} .py files")
    print("="*60)

    for file in files[:15]:
        print(f"  {file}")

    if len(files) > 15:
        print(f"  ... and {len(files) - 15} more")

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


def full_report(results: dict) -> pd.DataFrame:
    """Table with the raw values of each file plus the predicted risk,
    sorted from the riskiest to the least risky."""
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "file"
    df = normalization_relative_to_repo(df, repo_col="repo")
    # Risk (technical debt probability) per file.
    X = df[feat_list]
    df.insert(0, "danger", MODEL.predict_proba(X)[:, 1].round(3))

    return df.sort_values("danger", ascending=False).reset_index()


def save_csv_report(report: pd.DataFrame, repo_name: str, destination: Path = REPORTS_DIR) -> Path:
    """Saves the report as a CSV in the reports folder and returns the path of
    the generated file. The name carries a timestamp so it does not overwrite
    previous runs on the same repo."""
    destination.mkdir(parents=True, exist_ok=True)

    # The repo name comes from GitHub, so it is sanitized before using it as a filename.
    clean_name = re.sub(r"[^A-Za-z0-9._-]", "_", repo_name) or "repo"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = destination / f"report_{clean_name}_{timestamp}.csv"

    report.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def _delete_tree(path: Path):
    """Deletes a directory tree. On Windows the .git objects are left as
    read-only, so that flag is removed before retrying the deletion."""
    def _remove_readonly(func, path, _exc_info):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    shutil.rmtree(path, onerror=_remove_readonly)


def fetch_and_clone_repo(max_attempts: int = 5):
    """Looks for a random repo on GitHub, clones it into a temporary folder and
    validates that it has .py files. Retries with another repo if the clone fails
    or the repo has no usable Python code. Returns
    (tmp_dir, repo_path, repo_git, repo_name, files) or None."""
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
        "repo_path",
        nargs="?",
        default=None,
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

        results = feature_extraction(repo_path, files, repo_git, repo_name)
        print(f"\nFeatures extracted from {len(results)} files.")

        if not results:
            print("Could not extract features from any file.")
            return

        report = full_report(results)
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
