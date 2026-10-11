"""
fast_git.py - same output as feature_lib.feature_git_raw, but fast on big repos.

feature_git_raw calls commit.stats (one `git diff` per commit) for every commit of
every file, so a commit touching 50 files is diffed 50 times. On a repo with
thousands of commits that takes many minutes. Here:

  * churn: ONE `git log --numstat` over the whole history, diffing each commit
    against its first parent (or the empty tree for root commits) with renames
    off, which is exactly what GitPython's commit.stats does.
  * per-file history: one `git log -- <file>` per file, which applies the same
    history simplification as repo.iter_commits(paths=file).

Raw author names (%an, no mailmap) and the committer-local date (%cI) match
commit.author.name and commit.committed_datetime.
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path


def _git(repo_path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout


def churn_by_commit(repo_path: Path) -> dict[str, dict[str, int]]:
    out = _git(
        repo_path, "log", "HEAD", "--no-renames", "--diff-merges=first-parent",
        "--numstat", "--format=%x00%H",
    )
    churn = {}
    for block in out.split("\x00")[1:]:
        lines = block.strip("\n").split("\n")
        files = {}
        for line in lines[1:]:
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            insertions, deletions, path = parts
            # GitPython counts binary files ("-") as 0
            files[path] = (0 if insertions == "-" else int(insertions)) + (0 if deletions == "-" else int(deletions))
        churn[lines[0]] = files
    return churn


def file_history(repo_path: Path, file: str) -> list[tuple[str, str, str]]:
    out = _git(repo_path, "log", "HEAD", "--format=%H%x09%cI%x09%an", "--", file)
    return [tuple(line.split("\t", 2)) for line in out.splitlines() if line]


def git_features(repo_path: Path, files: list[str], workers: int = 4) -> dict[str, dict]:
    churn = churn_by_commit(repo_path)
    posix_files = [f.replace("\\", "/") for f in files]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        histories = list(pool.map(lambda f: file_history(repo_path, f), posix_files))

    features = {}
    for file, posix, history in zip(files, posix_files, histories):
        authors = set()
        churn_total = 0
        for sha, _, author in history:
            authors.add(author)
            churn_total += churn.get(sha, {}).get(posix, 0)

        features[file] = {
            "commit_count": len(history),
            "author_count": len(authors),
            "days_since_change": (date.today() - date.fromisoformat(history[0][1][:10])).days if history else 0,
            "churn_rate": churn_total,
        }
    return features
