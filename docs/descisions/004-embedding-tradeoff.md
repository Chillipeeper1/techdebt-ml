# 004 — Code embeddings via PCA: adopted despite a local recall trade-off

**Date:** 2026-07-25
**State:** Adopted as the official model (v3)
**Related:** 003-git-activity-bias.md

## Context

Even after the relative git-activity normalization in 003, cyclomatic_complexity remained
weakly correlated with predicted risk (0.16), while maintainability_index and git-activity
features continued to dominate. The hypothesis: raw numeric metrics don't capture *what the
code actually does* a semantic embedding of the code itself might.

## Approach

For each file, the most complex functions (via `cc_visit`, reusing the existing Radon call)
were extracted and encoded with `all-MiniLM-L6-v2` (`sentence-transformers`), batched for
performance. Per-function embeddings were mean-pooled into one 384-dim vector per file.

384 raw dimensions would have overwhelmed the 17 existing tabular features, so PCA was used
to reduce them before combining. The variance curve showed no clear "elbow":

| Components | Cumulative variance |
|------------|---------------------|
| 5          | 31.2%               |
| 10         | 41.2%               |
| 15         | 48.7%               |
| 20         | 54.2%               |
| 30         | 62.3%               |
| 50         | 72.5%               |

**20 components** were chosen, comparable in count to the 17 tabular features, avoiding the
same imbalance problem the embeddings were meant to help with, not repeat.

PCA was fit only on the training split; `.transform()` (never `.fit_transform()`) is used on
test data and on any new repo analyzed by `predict.py`, to avoid the same train/test leakage
category flagged as a methodological note in `003`.

## Results

Correlation of raw features with predicted risk across external repos never seen in training:

| Feature                 | v1 (absolute) | v2 (relative) | v3 (embeddings + PCA) |
|-------------------------|---------------|---------------|-----------------------|
| `cyclomatic_complexity` | 0.13          | 0.16          | **0.28**              |
| `max_complexity`        | 0.24          | 0.33          | **0.37**              |
| `max_nesting_depth`     | 0.20          | 0.38          | **0.46**              |
| `maintainability_index` | -0.41         | -0.58         | **-0.60**             |
| `author_count`          | 0.73          | 0.28          | 0.48                  |
| `commit_count`          | 0.73          | 0.61          | 0.55                  |
| `dependency_count`      | 0.56          | N/A           | 0.62                  |
| `loc`                   | 0.63          | 0.54          | 0.60                  |

Local test-set metrics (same 22-repo dataset, held-out split):

|                  | v1     | v2     | v3     |
|------------------|--------|--------|--------|
| Score            | 0.8842 | 0.8743 | 0.8683 |
| Recall (label 1) | 0.79   | 0.76   | 0.75   |

## Decision

**v3 is adopted as the official model**, despite scoring slightly lower on the local test set
than v1 and v2. The reasoning: local score/recall reflect performance on the same 22 repos the
model was trained on, and part of v1's and v2's higher local numbers came from exactly the
biases being corrected (raw and relative git-activity shortcuts). The external correlation
results are the more meaningful signal for a tool meant to generalize to unseen repos, and on
that signal, v3 is a clear improvement on the target limitation: `cyclomatic_complexity`
correlation nearly doubled (0.16 → 0.28), and `maintainability_index` strengthened further
(-0.58 → -0.60).

This is treated as a deliberate trade-off — roughly 4 points of local recall spent to reduce
reliance on git-activity shortcuts and increase reliance on actual code structure — not as an
unqualified win.

## New limitation introduced

`dependency_count`, `loc`, and `sloc` correlations rose in v3 (e.g. `loc`: 0.54 → 0.60),
undoing part of the improvement v2 had made on file-size reliance. The likely cause: mean-pooled
embeddings over a file's functions probably encode "how much code there is" alongside "how
complex it is" — longer files have more functions contributing to the pooled vector, which may
reintroduce a raw-size signal the embeddings weren't meant to carry. This wasn't a targeted
problem going into this experiment and wasn't fully solved; it's now a known trade-off of v3,
not a blocking issue.

## Next steps

- Update `predict.py` and README to point to `random_forest_v3.pkl` / `pca_v3.pkl` /
  `features_v3.pkl` by default.
- If the size-coupling above becomes a problem in practice, consider embedding a fixed-size
  code sample (e.g. always the single most complex function, not a pooled average) instead of
  mean-pooling across all functions — a promising direction for a future v4, not undertaken now.