# 002 — Known limitation: no temporal cut

**Date:** 7-11-2026
**State:** known limitation, pending
**Related:** 001-data-leakage.md

## Context

commit_count is a feature derived from the git history extraction of the **total** number of commits across the whole
life of the file, including bug-fix commits. This is a problem because the model is supposed to predict technical debt
in a file before it accumulates that debt, but commit_count as currently calculated already includes commits that
happened after the file was confirmed to have technical debt. The feature and the label are drawn from overlapping
history, even though they no longer share the exact same commit-message keywords as in 001-data-leakage.md.

## Possible solution

The main problem with commit_count is that it aggregates commits from the entire file history in one number. We need
to split each file's history into two non-overlapping windows: a "before" group and an "after" group. A reasonable
starting point is the first 70% of a file's commits for "before" and the last 30% for "after," with a minimum of 10
commits per file to discard files with too little history to split reliably.

All features would be calculated using only the "before" group, and the label would be calculated using only the
"after" group. Since these two windows don't overlap, the direct mathematical dependency between features and label
disappears, but the predictive relationship stays intact, because what happened earlier in a file's life can still
say something real about what happens later. This is a **temporal separation**, not statistical independence; if the
label were truly independent of the features, the model would have nothing to learn from them.

One consequence worth noting: this approach would let us bring back `bug_fix_commits` and `refactor_commits` as
features, as long as they're computed only from the "before" window. They would no longer be contaminated, since they
would represent past history predicting future outcomes rather than the same event counted twice.

A detail that needs an explicit rule: commits landing exactly on the cutoff point must be assigned consistently to one
side only. By convention, a commit at the boundary should be counted as part of the "before" group, never both; otherwise
a small version of the same leakage problem could creep back in.

## Why wasn't this addressed now?

The dataset, as noted in 001-data-leakage.md, is small. If the repo search script pulls in a repo with a low commit
count, the "after" group for that file's label won't be reliable, and the file (or the whole repo) would have to be
discarded. Applying this solution to the current dataset would shrink the number of usable rows drastically, and the
model likely wouldn't have enough data left to learn anything meaningful.

The model trained with the leakage removed is already a solid MVP. The next step is to grow the dataset before
attempting the temporal cut. Specifically, once the project reaches 50+ candidate repos (roughly double the current
count), this solution should be revisited, since that's the point where losing files to the cut is less likely to
cripple the dataset.



*(I know it sounds like a paradox; a technical debt detector that has its own technical debt. But at least I'm
documenting it. Document your technical debts, pls.)*