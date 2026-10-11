"""
build_examples.py - pre-computes the reports for the demo's example repos,
so clicking an example in the Space is instant. Run before deploying:

    python space/build_examples.py
"""

import tempfile
import time
from pathlib import Path

import app


def main():
    app.EXAMPLES_DIR.mkdir(exist_ok=True)
    for url in app.EXAMPLE_REPOS:
        owner, name = app.parse_github_url(url)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="techdebt_") as tmp:
            app.clone_repo(owner, name, Path(tmp) / name)
            report, notes = app.analyze_repo(Path(tmp) / name, name, lambda *a, **k: None)
        out = app.example_cache_path(owner, name)
        report.to_csv(out, index=False)
        print(f"{owner}/{name}: {len(report)} files in {time.monotonic() - started:.0f}s -> {out.name} {notes}")


if __name__ == "__main__":
    main()
