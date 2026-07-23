# 001 - Data leakage in initial label. 

**Date**: 7-8-2026
**State**: known limitation

## Context 
the initial label for the dataset was generated using a conditional filter on the count_bug feature. [count_bug: bug_fix_commits]
is a feature derived from the git analytics extraction of the number of bugs in the project. 

## Detected Issue. 
The model selected for the project was a Random Forest Classifier. The model was trained on the dataset with the initial
label. During the evaluation of the model, it was observed that the model was able to predict the initial label with 100% accuracy. 
This raised concerns about the whole architecture of the model and the dataset. 
## Diagnosis
The main issue was that the initial label was generated using a feature that was also present in the dataset, which led to 
a data leakage problem. The model was able to learn from the count_bug feature and predict the initial label without any 
real understanding of the underlying data. This was confirmed by inspecting the feature importance of the model, wich 
showed that the count_bug feature had a very high importance score over the other features in the dataset. But count_bug 
was not hoarding the 100% of the importance but the 59% of the importance score, and reviewing the other highest importance 
features: commit_count (11.7%), refactor_commits (6.8%) and author_count (6%). The data leakage was not caused directly by 
count_bug but by the combination of the features that were used to generate the initial label, specifically the combination 
of count_bug and refactor_commits since both features share information about the number of bugs in the file. The dataset 
had 2503 rows (Total dataset: 2503 rows (2002 train / 501 test, stratified split).), and the Random Forest Classifier was 
initially trained with 100 trees, so it was easy for each tree to memorize the exact combination of features that led to 
the initial label.

## Decision
The decision taken was to remove any column sharing information with the initial label. The features removed were bug_fix_commits
and refactor_commits since those were the ones sharing the exact same information.

## Results
After removing the columns that shared information with the initial label, the model was retrained and evaluated. The new
model achieved 0.842% accuracy on the test set, not perfect, but honest. The confusion matrix on the test set was
[[291, 20], [40, 150]]. The model evaluated 190 files with real high technical debt and failed to predict 40 of them,
giving a recall of 0.79 for that class.

**Note:** this run also includes an outlier clip on max_complexity (capped at the 99th percentile, the same treatment 
already applied to churn_rate). Comparing against the unclipped version, the score moved from 0.8822 to 0.8802 and
one file shifted from true positive to false negative, a marginal difference. This is consistent with Random Forest's
general robustness to isolated outliers, since its decision splits depend on relative thresholds rather than the exact
magnitude of extreme values. The clip was kept regardless, since it remains the more defensible choice even when its
impact on this particular metric is small.

**Note 2:** after building the predict.py script, I found some important issues with some features. blank_ratio was forced to
be set as 0 for files with no comments, but blank_ratio is defined as the ratio of blank lines to total lines. After fixing 
this bug and get again the features and retraining the model, the score moved from 0.882 to 0.884, reducing false positives from 
20 to 18, but the recall remained 0.79; the fix didn't improve the deteccion of highly dangerous files.

## Limitations and next steps. 
The confusion matrix shows that the model is still not perfect, and it's problematic because of the false negatives; a file 
with real high technical debt marked as 0 can silently cause problems downstream. But a recall of 0.79 is a good start, the model 
was trained with a small dataset, and implementing a robust solution would require a larger dataset. The next steps would 
be to increase the dataset size and to explore solutions to improve the model in general. 

| Metric            | With leakage | Without leakage |
|-------------------|--------------|-----------------|
| Score             | 1.00         | 0.88            |
| Recall (class 1)  | N/A          | 0.79            |
| Features in X     | 21           | 19              |


