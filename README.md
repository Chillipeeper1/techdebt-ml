# TechDebt ML

You don't know what technical debt is? Do you consider yourself a vibe-coder? If so, you are probably in the right place.

TechDebt ML is a machine learning project that aims to detect technical debt in your codebase.

Even so, this project was mainly created to help vibe-coders detect technical debt in their codebase. The truth is that
nobody can detect it in a 100% accurate way; there are just hundreds of files that would need to be reviewed by a senior
developer to be sure if they have technical debt or not, and even then, the senior developer could be wrong. So this
project is just a tool to help you detect technical debt in your codebase, but it is not a silver bullet.

**Currently supports Python codebases only.** Files in other languages are ignored.

## How it works

TechDebt ML analyzes your codebase file by file and extracts three families of features: code-quality metrics
(complexity, maintainability, size, via Radon), structural metrics (imports, nesting depth, tests, via Python's
`ast`), and git activity (commits, authors, code churn, normalized relative to each repo). It then predicts, per
file, the probability that it carries technical debt. Check the [training notebook](notebooks/techdebt_model.ipynb)
for the full feature list and training process.

The model was trained on 2,503 files from 22 public repositories (2,002 for training, 501 held out for testing,
stratified split).

## Dataset

The full training dataset — features, labels, preprocessing steps, and source repo attribution — is published
separately on Hugging Face:

**[huggingface.co/datasets/ChilliPeep/techdebt-ml-dataset](https://huggingface.co/datasets/ChilliPeep/techdebt-ml-dataset)**

Use it if you want to explore the data, retrain the model yourself, or build something else on top of it without
having to re-scrape and re-analyze 22 repositories from scratch.

## Installation

```bash
git clone <this-repo-url>
cd techdebt-ml
pip install -r requirements.txt
```

## Usage

**Analyze a local repo:**

```bash
python src/predict.py /path/to/your/repo
```

**Or let it pick a random public repo for you** (requires a `GITHUB_TOKEN` in a `.env` file — see `.env.example`):

```bash
python src/predict.py
```

Either way, you'll get a report in your terminal and a timestamped CSV (`report_<repo>_<date>.csv`) with every
analyzed file, sorted from highest to lowest predicted risk.

## Known limitations

This project documents its own blind spots as it finds them; see [`docs/descisions/`](docs/descisions/) for the full
history. The short version:

- **[001](docs/descisions/001-data-leakage.md):** an early version of the label leaked into the features. Fixed.
- **[002](docs/descisions/002-limitations-temporal-cut.md):** features and label are both computed from a file's full
  git history, without a strict "past predicts future" time split. Known limitation, not yet addressed.
- **[003](docs/descisions/003-model-bias-git-activity.md):** the model initially over-relied on raw git activity
  (commits, authors) rather than actual code complexity, especially on repos more active than the training set.
  Largely mitigated by normalizing activity features relative to each repo, though complex-but-rarely-touched files
  can still score lower than they probably should.

If a file gets a surprisingly low or high score, take it as a prompt to look closer — not as a verdict.

## Contributing

This is an early-stage, actively-evolving project. Issues, PRs, and — especially — reports of files where the model
got it wrong are welcome.

## License

Code is released under the [MIT License](LICENSE). The training dataset is released separately under CC-BY 4.0 —
see the [Hugging Face dataset card](https://huggingface.co/datasets/ChilliPeep/techdebt-ml-dataset) for details.
