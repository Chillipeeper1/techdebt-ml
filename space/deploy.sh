#!/usr/bin/env bash
# deploy.sh - assembles the Space (app + src/ + models/) and uploads it to Hugging Face
# as a ZeroGPU Space (free for personal accounts older than 30 days, max 2 per account).
#
# Usage:  hf auth login            (once, with a token that has write access)
#         python space/build_examples.py   (when the model or the examples change)
#         ./space/deploy.sh [hf-user/space-name]
set -euo pipefail

SPACE_ID="${1:-ChilliPeep/techdebt-ml}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp "$REPO_ROOT"/space/{app.py,fast_git.py,metrics_worker.py,requirements.txt,README.md} "$STAGE"/
cp -r "$REPO_ROOT"/space/examples "$STAGE"/
mkdir -p "$STAGE"/src "$STAGE"/models
cp "$REPO_ROOT"/src/{predict.py,feature_lib.py} "$STAGE"/src/
cp "$REPO_ROOT"/models/{random_forest_v3.pkl,pca_v3.pkl,features_v3.pkl} "$STAGE"/models/

hf repos create "$SPACE_ID" --repo-type space --space-sdk gradio --flavor zero-a10g --exist-ok
hf upload "$SPACE_ID" "$STAGE" . --repo-type space --commit-message "Deploy demo"

echo "Done: https://huggingface.co/spaces/$SPACE_ID"
