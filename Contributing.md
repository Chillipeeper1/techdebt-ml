# Contributing to TechDebt ML

Thanks for considering contributing. This project is early-stage and actively evolving — there's
real room to shape how it works, not just polish the edges.

## Setup

```bash
git clone <this-repo-url>
cd techdebt-ml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your own `GITHUB_TOKEN` (needed for `repo_search.py` and
for `predict.py`'s random-repo mode). Generate one at
[github.com/settings/tokens](https://github.com/settings/tokens) — `public_repo` scope is enough.

## Running the tests

```bash
pytest tests/ -v
```

`tests/test_feature_lib.py` covers `feature_lib.py` — the module shared between the training
pipeline and `predict.py`. If you're changing anything in there, run the tests before and after.

## Project structure

```
src/
  feature_lib.py       # shared feature-calculation logic — the single source of truth
  feature_extraction.py  # training pipeline: reads repos_candidatos.csv, builds the dataset
  predict.py            # analyzes any repo, predicts risk per file
  repo_search.py         # finds candidate repos on GitHub
notebooks/
  techdebt_model.ipynb   # model training, evaluation, and experiments
docs/decisions/          # numbered logs of real findings, bugs, and design trade-offs
models/                  # trained model artifacts (v3 is current)
```

## A rule that matters more than it looks

`feature_lib.py` exists specifically so the training pipeline and `predict.py` calculate every
feature the exact same way. Several real bugs in this project's history came from that
guarantee slipping — a fix applied in one script but not the other. If you're adding or changing
a feature, it almost certainly belongs in `feature_lib.py`, imported by both, not duplicated.

## Proposing changes

1. Fork the repo, create a branch (`git checkout -b fix/thing`)
2. Make your change, and add or update a test in `tests/test_feature_lib.py` if it touches
   feature logic
3. Run `pytest tests/` and make sure the training notebook still runs end to end if you touched
   the pipeline
4. Open a PR describing what changed and why

## If your change affects model behavior

If your PR changes what features go into the model, how they're calculated, or how the model is
trained, please include a before/after comparison — score, recall on the high-debt class, and
(if relevant) the correlation analysis used in `docs/decisions/003` and `004`. This project's
history shows that a change that looks like a pure improvement locally can hide a trade-off
(see `004` — an accuracy drop that was still worth making). We'd rather see the full picture than
just the number that looks best.

For anything non-trivial, consider adding a numbered entry to `docs/decisions/` following the
existing format (context, what was tried, what was found, what was decided). It doesn't need to
be long — the goal is that someone six months from now can understand why a decision was made
without re-deriving it.

## Good first issues

Look for issues tagged `good first issue` or `help wanted` — a few are already up covering things
like multi-language support, dataset expansion, and the temporal-split limitation from
`docs/decisions/002`. Feel free to open an issue yourself if you spot something.

## Reporting a bad prediction

If the model gets a file obviously wrong — flags something trivial as high risk, or misses
something genuinely messy — please open an issue with the file (or a minimal reproduction) and
the score it got. Real examples like this are some of the most useful feedback this project can
get.