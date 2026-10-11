"""
app.py - Gradio demo for TechDebt ML (Hugging Face Spaces, ZeroGPU)

Clones a public GitHub repo, runs the same v3 pipeline as src/predict.py
(tabular features + code embeddings) and shows the per-file risk report.

Only the embeddings run on the GPU. If the visitor's ZeroGPU quota is used up,
they are computed on CPU instead (same model, a few seconds slower).
"""

import json
import os
import re
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# src/ lives next to this file on the Space, and one level up in the GitHub repo
HERE = Path(__file__).resolve().parent
for candidate in (HERE / "src", HERE.parent / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
        break

try:
    import spaces  # must be imported before torch on ZeroGPU
except ImportError:  # running locally without the HF `spaces` package
    spaces = None

import gradio as gr
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

import predict  # loads the v3 model, PCA and the CPU embedding model once at startup
from fast_git import git_features

EXAMPLES_DIR = HERE / "examples"
GITHUB_URL_RE = re.compile(r"^https?://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")

MAX_FILES = 150          # larger repos are cut down to their biggest .py files
CLONE_TIMEOUT_S = 90
PER_FILE_TIMEOUT_S = 10  # radon can hang on some huge files

# ZeroGPU wants models placed on cuda at import time; the CPU copy in predict stays as fallback
GPU_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2").to("cuda") if torch.cuda.is_available() else None

# column in the report -> header shown in the UI
SHOWN_COLUMNS = {
    "file": "file",
    "danger": "risk",
    "sloc": "SLOC",
    "cyclomatic_complexity": "avg CC",
    "max_complexity": "max CC",
    "maintainability_index": "MI",
    "max_nesting_depth": "nesting",
    "num_functions": "funcs",
    "commit_count": "commits×",
    "author_count": "authors×",
    "churn_rate": "churn×",
    "days_since_change": "age×",
}

EXAMPLE_REPOS = [
    "https://github.com/psf/requests",
    "https://github.com/pallets/click",
    "https://github.com/Chillipeeper1/techdebt-ml",
]


def parse_github_url(url: str) -> tuple[str, str]:
    match = GITHUB_URL_RE.match(url.strip())
    if not match:
        raise gr.Error("Please paste a public GitHub repo URL, e.g. https://github.com/psf/requests")
    return match.group(1), match.group(2)


def clone_repo(owner: str, name: str, dest: Path):
    # Full history is required: commit count, authors and churn are model features.
    try:
        subprocess.run(
            ["git", "clone", "--quiet", "--single-branch", "--no-tags",
             f"https://github.com/{owner}/{name}.git", str(dest)],
            check=True, capture_output=True, text=True, timeout=CLONE_TIMEOUT_S,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except subprocess.TimeoutExpired:
        raise gr.Error(f"Cloning took more than {CLONE_TIMEOUT_S}s. Try a smaller repo.")
    except subprocess.CalledProcessError:
        raise gr.Error(f"Could not clone {owner}/{name}. Is it public and does it exist?")


def pick_files(repo_path: Path) -> tuple[list[str], int]:
    files = predict.find_python_files(repo_path)
    total = len(files)
    if total > MAX_FILES:
        files = sorted(files, key=lambda f: (repo_path / f).stat().st_size, reverse=True)[:MAX_FILES]
    return files, total


def code_metrics(repo_path: Path, files: list[str], progress) -> tuple[dict, list[str]]:
    """Runs metrics_worker.py over the files, killing and restarting it when a
    file takes more than PER_FILE_TIMEOUT_S. Returns ({file: output}, skipped)."""
    outputs, skipped = {}, []
    pending = list(files)
    while pending:
        proc = subprocess.Popen(
            [sys.executable, str(HERE / "metrics_worker.py"), str(repo_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        proc.stdin.write("\n".join(pending).encode())
        proc.stdin.close()
        buffer = b""
        try:
            while pending:
                if b"\n" not in buffer:
                    ready, _, _ = select.select([proc.stdout], [], [], PER_FILE_TIMEOUT_S)
                    chunk = os.read(proc.stdout.fileno(), 1 << 20) if ready else b""
                    if not chunk:  # timed out or worker died on pending[0]
                        skipped.append(pending.pop(0))
                        break
                    buffer += chunk
                    continue
                line, buffer = buffer.split(b"\n", 1)
                outputs[pending.pop(0)] = json.loads(line)
                progress(0.2 + 0.5 * len(outputs) / len(files), desc=f"Code metrics ({len(outputs)}/{len(files)})")
        finally:
            proc.kill()
            proc.wait()
            proc.stdout.close()
    return outputs, skipped


def extract_features(repo_path: Path, files: list[str], repo_name: str, progress):
    """Same features as predict.feature_extraction, but with git activity from
    fast_git and code metrics from a killable worker process."""
    progress(0.1, desc="Reading git history")
    git_feats = git_features(repo_path, files)
    metrics, skipped = code_metrics(repo_path, files, progress)

    results, functions, index = {}, [], []
    for file in files:
        out = metrics.get(file)
        if out is None or "error" in out:
            continue
        features = {**out["features"], **git_feats[file], "repo": repo_name}
        key = f"{repo_name}/{file}".replace("\\", "/")
        results[key] = features
        index.append((key, len(out["functions"])))
        functions.extend(out["functions"])
    return results, functions, index, skipped


def file_embeddings(functions: list[str], index: list, model) -> dict:
    """Same as predict.generar_embeddings_reducidos, with the model as a parameter:
    encodes every function, then mean-pools them per file (zeros if none)."""
    encoded = model.encode(functions, batch_size=64) if functions else np.empty((0, 384))
    per_file, cursor = {}, 0
    for key, count in index:
        per_file[key] = encoded[cursor: cursor + count].mean(axis=0) if count else np.zeros(384)
        cursor += count
    return per_file


def _embed_on_gpu(functions, index):
    return file_embeddings(functions, index, GPU_EMBED_MODEL)


if spaces is not None and GPU_EMBED_MODEL is not None:
    _embed_on_gpu = spaces.GPU(duration=30)(_embed_on_gpu)


def embed(functions: list[str], index: list) -> tuple[dict, str]:
    if GPU_EMBED_MODEL is not None:
        try:
            return _embed_on_gpu(functions, index), "GPU"
        except Exception as e:  # typically the visitor's daily ZeroGPU quota
            print(f"GPU embedding failed, falling back to CPU: {e}")
    return file_embeddings(functions, index, predict.EMBED_MODEL), "CPU"


def analyze_repo(repo_path: Path, repo_name: str, progress) -> tuple[pd.DataFrame, list[str]]:
    files, total = pick_files(repo_path)
    if not files:
        raise gr.Error("No .py files found. TechDebt ML only supports Python codebases.")

    notes = []
    if total > len(files):
        notes.append(f"The repo has {total} .py files; only the {len(files)} largest were analyzed.")

    results, functions, index, skipped = extract_features(repo_path, files, repo_name, progress)
    if skipped:
        notes.append(f"Skipped {len(skipped)} file(s) that took over {PER_FILE_TIMEOUT_S}s to parse: "
                     + ", ".join(f"`{f}`" for f in skipped[:5]))
    if not results:
        raise gr.Error("Could not extract features from any file.")

    progress(0.75, desc=f"Embedding {len(functions)} functions and predicting")
    embeddings, device = embed(functions, index)
    if device == "CPU" and GPU_EMBED_MODEL is not None:
        notes.append("GPU quota used up, embeddings were computed on CPU.")
    report = predict.full_report(results, embeddings)
    report["file"] = report["file"].str.removeprefix(f"{repo_name}/")
    return report, notes


def example_cache_path(owner: str, name: str) -> Path:
    return EXAMPLES_DIR / f"{owner}__{name}.csv"


def build_outputs(report: pd.DataFrame, owner: str, name: str, notes: list[str], elapsed: float | None):
    high = int((report["danger"] >= 0.5).sum())
    summary = [
        f"### {owner}/{name}",
        f"**{len(report)}** files analyzed · **{high}** with risk ≥ 0.5 · "
        f"median risk **{report['danger'].median():.2f}**",
    ]
    if elapsed is None:
        summary.append("_Pre-computed example (instant). Any other URL is analyzed live._")
    else:
        summary.append(f"_Analyzed live in {elapsed:.0f}s._")
    summary += [f"- {n}" for n in notes]

    top = report.head(15)[["file", "danger"]].iloc[::-1]

    csv_path = Path(tempfile.mkdtemp(prefix="techdebt_report_")) / f"techdebt_{owner}_{name}.csv"
    report.to_csv(csv_path, index=False)

    shown = report[list(SHOWN_COLUMNS)].round(2).rename(columns=SHOWN_COLUMNS)
    return "\n\n".join(summary), top, shown, str(csv_path)


def run(url: str, progress=gr.Progress()):
    owner, name = parse_github_url(url)

    cached = example_cache_path(owner, name)
    if cached.exists():
        return build_outputs(pd.read_csv(cached), owner, name, [], None)

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="techdebt_") as tmp:
        progress(0, desc=f"Cloning {owner}/{name}")
        clone_repo(owner, name, Path(tmp) / name)
        report, notes = analyze_repo(Path(tmp) / name, name, progress)

    return build_outputs(report, owner, name, notes, time.monotonic() - started)


DESCRIPTION = """
# TechDebt ML

Predicts, **per file**, the probability that a Python file carries technical debt.
It combines code metrics (Radon + AST), the file's git activity relative to its own repo,
and embeddings of its most complex functions (`all-MiniLM-L6-v2`), fed to a Random Forest
trained on 2,503 files from 22 public repos.

Paste a **public GitHub repo** with Python code. Live analysis clones the full history,
so it takes from ~10s to a couple of minutes depending on the repo.
[Source code](https://github.com/Chillipeeper1/techdebt-ml) ·
[Known limitations](https://github.com/Chillipeeper1/techdebt-ml/tree/main/docs/decisions)
"""

with gr.Blocks(title="TechDebt ML") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        url = gr.Textbox(label="GitHub repo URL", placeholder="https://github.com/owner/repo", scale=4)
        button = gr.Button("Analyze", variant="primary", scale=1)
    gr.Examples(EXAMPLE_REPOS, inputs=url, label="Try one (pre-computed, instant)")

    summary = gr.Markdown()
    chart = gr.BarPlot(x="file", y="danger", label="Top 15 riskiest files",
                       y_lim=[0, 1], x_label_angle=-45, sort="-y", height=380)
    table = gr.Dataframe(label="All analyzed files (highest risk first)")
    gr.Markdown(
        "CC = cyclomatic complexity · MI = maintainability index · "
        "**×** = relative to the repo's median file (2.0 = twice the typical file). "
        "The CSV has every feature the model uses."
    )
    download = gr.File(label="Full report (CSV)")

    gr.Markdown(
        "A high score is a prompt to look closer, not a verdict. "
        "Files in folders named `test` are skipped, same as the CLI."
    )

    outputs = [summary, chart, table, download]
    button.click(run, inputs=url, outputs=outputs)
    url.submit(run, inputs=url, outputs=outputs)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
