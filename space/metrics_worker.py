"""
metrics_worker.py - computes the per-file code features in a separate process.

Radon can hang on some files (e.g. a 7k-line test file in fastapi never finishes
in radon.raw.analyze), and a stuck thread can't be killed. app.py runs this
worker as a subprocess so it can kill it when a file takes too long.

Protocol: argv[1] = repo path, stdin = one relative file path per line,
stdout = one JSON object per file, in the same order.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for candidate in (HERE / "src", HERE.parent / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
        break

from feature_lib import embedding_extract_functions, features_ats, features_radon


def main():
    repo_path = Path(sys.argv[1])
    for file in sys.stdin.read().splitlines():
        try:
            code = (repo_path / file).read_text(encoding="utf-8")
            features = {}
            features.update(features_radon(code))
            features.update(features_ats(code))
            out = {"file": file, "features": features, "functions": embedding_extract_functions(code)}
        except Exception as e:
            out = {"file": file, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(out, default=float), flush=True)


if __name__ == "__main__":
    main()
