import pandas as pd
import pytest
from radon.complexity import cc_visit
from git import Repo, Actor

from feature_lib import (
    features_radon,
    features_ats,
    feature_git_raw,
    normalization_relative_to_repo,
)


def test_blank_ratio_does_not_depend_on_having_comments():
    code = "x = 1\n\n\ny = 2\n"  # no comments, with blank lines

    result = features_radon(code)

    assert result["comment_ratio"] == 0  # correct, there are no comments
    assert result["blank_ratio"] > 0     # the original bug returned 0 here


def test_comment_ratio_counts_real_comments():
    code = "# comment 1\n# comment 2\nx = 1\ny = 2\n"

    result = features_radon(code)

    assert result["comment_ratio"] > 0


def test_max_complexity_is_the_max_not_the_sum():
    code = """
def simple():
    return 1

def with_conditionals(x):
    if x > 0:
        if x > 10:
            return 1
    return 0

def another_simple():
    return 2
"""
    real_complexities = [f.complexity for f in cc_visit(code)]
    real_max = max(real_complexities)
    total_sum = sum(real_complexities)

    result = features_radon(code)

    assert result["max_complexity"] == real_max
    assert result["max_complexity"] != total_sum or real_max == total_sum


def test_empty_file_does_not_crash():
    code = "x = 1\n"

    result = features_radon(code)

    assert result["num_functions"] == 0
    assert result["max_complexity"] == 0
    assert result["avg_function_len"] == 0


def test_dependency_count_counts_imports():
    code = "import os\nimport sys\nfrom pathlib import Path\n\nx = 1\n"

    result = features_ats(code)

    assert result["dependency_count"] == 3


def test_max_nesting_depth_detects_real_nesting():
    code = """
def function():
    if True:
        for i in range(10):
            if i > 5:
                pass
"""
    result = features_ats(code)

    # function(1) -> if(2) -> for(3) -> if(4)
    assert result["max_nesting_depth"] == 4


def test_is_tested_detects_test_functions():
    code_with_test = "def test_something():\n    assert True\n"
    code_without_test = "def normal_function():\n    return 1\n"

    assert features_ats(code_with_test)["is_tested"] == 1
    assert features_ats(code_without_test)["is_tested"] == 0


@pytest.fixture
def test_repo(tmp_path):
    repo = Repo.init(tmp_path)

    file = tmp_path / "example.py"
    file.write_text("x = 1\n")
    repo.index.add(["example.py"])
    repo.index.commit(
        "first commit",
        author=Actor("Author One", "one@example.com"),
    )

    file.write_text("x = 1\ny = 2\n")
    repo.index.add(["example.py"])
    repo.index.commit(
        "second commit",
        author=Actor("Author Two", "two@example.com"),
    )

    return repo

def test_feature_git_raw_counts_commits_and_authors(test_repo):
    result = feature_git_raw("example.py", test_repo)

    assert result["commit_count"] == 2
    assert result["author_count"] == 2
    assert result["churn_rate"] > 0
    assert result["days_since_change"] >= 0


def test_feature_git_raw_never_committed_file(test_repo):
    result = feature_git_raw("does_not_exist.py", test_repo)

    assert result["commit_count"] == 0
    assert result["days_since_change"] == 0


def test_relative_normalization_computes_correct_ratio():
    df = pd.DataFrame({
        "repo": ["a", "a", "a", "b", "b"],
        "commit_count": [10, 20, 30, 5, 15],
        "author_count": [1, 1, 1, 1, 1],
        "churn_rate": [1, 1, 1, 1, 1],
        "days_since_change": [1, 1, 1, 1, 1],
    })

    result = normalization_relative_to_repo(df.copy(), repo_col="repo")

    assert result.loc[0, "commit_count"] == pytest.approx(0.5)
    assert result.loc[1, "commit_count"] == pytest.approx(1.0)
    assert result.loc[2, "commit_count"] == pytest.approx(1.5)


def test_relative_normalization_does_not_divide_by_zero():
    df = pd.DataFrame({
        "repo": ["a", "a"],
        "commit_count": [0, 0],
        "author_count": [1, 1],
        "churn_rate": [0, 0],
        "days_since_change": [1, 1],
    })

    result = normalization_relative_to_repo(df.copy(), repo_col="repo")

    assert not result["commit_count"].isin([float("inf")]).any()
    assert result["commit_count"].notna().all()
