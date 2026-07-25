---
license: cc-by-4.0
task_categories:
  - tabular-classification
tags:
  - technical-debt
  - code-quality
  - software-engineering
size_categories:
  - 1K<n<10K
---

# TechDebt ML Dataset

Features extracted from 22 public Python repositories, used to train a Random Forest
classifier that predicts technical debt risk per file. Companion dataset to the
[TechDebt ML](https://github.com/Chillipeeper1/techdebt-ml) project.

## Content

Each row is one Python file. Columns include:

- **Code-quality metrics** (via [Radon](https://radon.readthedocs.io/)): `loc`, `sloc`,
  `comment_ratio`, `blank_ratio`, `cyclomatic_complexity`, `max_complexity`,
  `maintainability_index`, `halstead_effort`, `num_functions`, `avg_function_len`
- **Structural metrics** (via Python's `ast`): `dependency_count`, `max_nesting_depth`,
  `is_tested`
- **Git activity** (normalized relative to each file's own repo — see Preprocessing below):
  `commit_count`, `author_count`, `churn_rate`, `days_since_change`
- **Metadata**: `file name`, `repo`
- **`label`**: 1 if the file had 2+ bug-fix commits in its git history, 0 otherwise
- `bug_fix_commits`, `refactor_commits`: kept for reference only — **do not use these as
  model features**, they were excluded from training due to label leakage (see below)

2,503 rows total. Label balance: 62% (0) / 38% (1).

## Preprocessing applied

Git-activity features (`commit_count`, `author_count`, `churn_rate`, `days_since_change`)
were normalized relative to each file's own repository (divided by that repo's median),
since their absolute values are not comparable across repos with different overall
activity levels.

Three columns were additionally clipped at the 99th percentile to control extreme outliers:
- `max_complexity`: clipped at 42.98
- `churn_rate`: clipped at 27.05
- `commit_count`: clipped at 20.97

## Known limitations

This dataset inherits every limitation already documented in the source project:

- **Label leakage (resolved):** an early version leaked the label into the features via
  `bug_fix_commits`/`refactor_commits`. [Details](https://github.com/Chillipeeper1/techdebt-ml/blob/main/docs/decisions/001-data-leakage.md)
- **No temporal split:** features and label are both computed from a file's entire git
  history, without a strict past-predicts-future split. [Details](https://github.com/Chillipeeper1/techdebt-ml/blob/main/docs/decisions/002-limitations-temporal-cut.md)
- **Git-activity bias (partially mitigated):** earlier model versions over-relied on commit
  and author counts rather than actual code complexity. Relative normalization improved
  this substantially but did not fully resolve it. [Details](https://github.com/Chillipeeper1/techdebt-ml/blob/main/docs/decisions/003-model-bias-git-activity.md)
- **Small-sample repos:** three of the 22 source repos have very few files (`flow`: 2,
  `Discord-OTP-Forcer`: 11, `ibet-Network`: 12). Per-repo medians used for git-activity
  normalization are less reliable for these repos. They were kept rather than excluded,
  given the small number of files affected.
- **Median computed pre-split:** per-repo medians for normalization were computed over the
  full dataset (train + test combined) rather than fit only on the training partition.
  Likely low impact, but a known methodological gap.

## Source repositories

See [`source_repos.csv`](./source_repos.csv) for the full list of the 22 public GitHub
repositories this dataset was derived from, with their URLs. Only aggregate numeric metrics
are included here — no source code, commit messages, or author names are redistributed.

## License

The derived metrics in this dataset are released under CC-BY 4.0. The original source code
of each repository remains under its own respective license.
