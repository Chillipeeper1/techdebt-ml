# 003 - Model bias toward git-activity features (partially resolved)

**Date**: 7-23-2026  
**State**: Improved, limitation partially resolved  
**Related**: [001 - Data leakage](001-data-leakage.md) 

## Context 

After validating predict.py end-to-end, three external repos never seen during training 
(luxtronik, zulip-hermes-integration, certbot_dns_duckdns) were evaluated. The risk
scores followed an unintuitive pattern: files with healthy code-quality metrics
(low cyclomatic complexity, high maintainability index) received high risk scores, 
with at least one generally complex file receiving a low risk score. 

## Diagnosis 
A Pearson correlation was computed between each raw feature and predicted danger across
100 files from three external repos: 

| Feature                 | Correlation with danger (v1) |
|-------------------------|------------------------------|
| `author_count`          | 0.73                         |
| `commit_count`          | 0.73                         |
| `churn_rate`            | 0.69                         |
| `cyclomatic_complexity` | 0.13                         |
| `maintainability_index` | -0.41                        |

Git activity features (author_count, commit_count, churn_rate) dominated the model's
behavior far more than actual code quality metrics. Two concrete examples confirmed this: 

- `luxtronik/number_entities_predefined.py`: cyclomatic complexity 0, maintainability index
  80.6 (healthy) — scored **0.97** risk, driven by high `commit_count` and `author_count`.
- `zulip-hermes-integration/zulip/logger.py` : the highest cyclomatic complexity in its repo
  (12) — scored only **0.19** risk, because it had low git activity (`commit_count=1`).

The root cause: commit_count, author_count, churn_rate, and days_since_change were all computed as
absolute values, which are not comparable across repos with different overall activity levels. A repo
with a large, active contributor community has naturally higher absolute values than moderately active repos
used for training, the model to over-index on activity for any repo more active its training 
distribution. 

## Decision
Instead of removing git activity features entirely (which carry real predictive signal, as shown
by their high correlation), each of the four affected features was recalculated relative to its own repo, using
the median of the feature within the repo as the denominator:  

commit_count_relative = commit_count_file / median(commit_count, same repo)  

This makes the features comparable across repos of any activity level, since the reference
point is internal to each repo rather than tied to the absolute scale seen during training. 
Outliers in the resulting ratios (e.g., churn_rate up to 458x the repo median) were
clipped to the 99th percentile, the same treatment already applied to max_complexity. 

## Results 
The same correlation analysis was repeated on 665 files from 7 new external repos, using the
retrained model (v2) with relative features:

| Feature                 | v1 (absolute) | v2 (relative) |
|-------------------------|---------------|---------------|
| `author_count`          | 0.73          | **0.28**      |
| `commit_count`          | 0.73          | 0.61          |
| `churn_rate`            | 0.69          | 0.53          |
| `maintainability_index` | -0.41         | **-0.58**     |
| `cyclomatic_complexity` | 0.13          | 0.16          |

maintainability_index: a genuine code-quality metric is now the single strongest
correlate with predicted risk, ahead of any individual git-activity feature. author_count's
dominance dropped sharply. Local test-set metrics on the training repos stayed close to v1
(score 0.874 vs. 0.884, recall on the high-debt class 0.76 vs. 0.79) — expected, since relative
normalization does not change file ordering within a repo, only across repos.

## Remaining limitation

cyclomatic_complexity correlation barely moved (0.13 → 0.16). Files with genuinely high
structural complexity but low relative git activity (rarely touched, stable, or simply
avoided by contributors) still receive low risk scores e.g. prism/research/features.py
(cyclomatic complexity 5.9, max_complexity 23) scored 0.04. The model still under-weighs
raw code complexity when it isn't accompanied by above-median git activity within its repo.

This is accepted as a known limitation for the current version rather than a blocking issue:
fixing it would likely require re-weighting or re-engineering how complexity features are
combined with activity features, which is a larger redesign left for a future iteration.

## Methodological note

normalization_relative_to_repo() was applied to the full dataset before the train/test
split, meaning each repo's median was computed using files from both sets combined. The
practical impact is likely small (a median over file counts is far less sensitive than a
direct label leak), but it is the same category of issue as fitting a scaler before splitting.
Worth correcting in a future dataset regeneration, not urgent today.