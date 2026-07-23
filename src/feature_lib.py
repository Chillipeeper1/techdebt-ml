import ast
from radon.complexity import cc_visit
from radon import metrics, raw
from datetime import date
import numpy as np

class MyAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.dependency_counts = 0
        self.max_nesting_depth = 0
        self.current_depth = 0
        self.is_tested = 0

    def visit_Import(self, node):
        self.dependency_counts += 1
        self.generic_visit(node)
    def visit_ImportFrom(self, node):
        self.dependency_counts += 1
        self.generic_visit(node)

    def visit_If(self, node):
        self.current_depth += 1

        if self.current_depth > self.max_nesting_depth:
            self.max_nesting_depth = self.current_depth
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_For(self, node):
        self.current_depth += 1

        if self.current_depth > self.max_nesting_depth:
            self.max_nesting_depth = self.current_depth
        self.generic_visit(node)
        self.current_depth -= 1
    def visit_While(self, node):
        self.current_depth += 1

        if self.current_depth > self.max_nesting_depth:
            self.max_nesting_depth = self.current_depth
        self.generic_visit(node)
        self.current_depth -= 1
    def visit_Try(self, node):
        self.current_depth += 1

        if self.current_depth > self.max_nesting_depth:
            self.max_nesting_depth = self.current_depth
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_FunctionDef(self, node):
        self.current_depth += 1
        if self.current_depth > self.max_nesting_depth:
            self.max_nesting_depth = self.current_depth

        if node.name.startswith("test_"):
            self.is_tested = 1
        self.generic_visit(node)
        self.current_depth -= 1


    def visit_ClassDef(self, node):
        self.current_depth += 1

        if self.current_depth > self.max_nesting_depth:
            self.max_nesting_depth = self.current_depth
        if node.name.startswith("Test"):
            self.is_tested = 1
        self.generic_visit(node)
        self.current_depth -= 1


def features_radon(file):
    result_raw_loc = raw.analyze(file).loc  # feat
    result_raw_sloc = raw.analyze(file).sloc  # feat
    result_raw_comments = raw.analyze(file).comments
    result_raw_multi = raw.analyze(file).multi
    result_raw_blank = raw.analyze(file).blank

    comment_ratio = (
        (result_raw_comments + result_raw_multi) / result_raw_loc
        if result_raw_comments > 0 else 0
    )
    blank_ratio = result_raw_blank / result_raw_loc if result_raw_loc > 0 else 0
    # Cyclomatic complexity
    result_cc = cc_visit(file)
    if len(result_cc) > 0:

        sum_func_lines = 0
        complexities = [score.complexity for score in result_cc]
        max_complexity = max(complexities)  # feat
        average_cc = sum(complexities) / len(complexities)  # feat
        for score in result_cc:
            func_lines = score.endline - score.lineno + 1
            sum_func_lines += func_lines

        num_functions = len(result_cc)  # feat

        avg_function_len = sum_func_lines / num_functions  # feat
    else:
        num_functions = 0
        avg_function_len = 0
        average_cc = 0
        max_complexity = 0

    # Maintainability Index
    result_mi = metrics.mi_visit(file, multi=True)  # feat

    # Halstead metrics (effort specifically)
    result_hl = metrics.h_visit(file).total.effort  # feat

    return {
        "loc": result_raw_loc,
        "sloc": result_raw_sloc,
        "comment_ratio": comment_ratio,
        "blank_ratio": blank_ratio,
        "cyclomatic_complexity": average_cc,
        "max_complexity": max_complexity,
        "maintainability_index": result_mi,
        "halstead_effort": result_hl,
        "num_functions": num_functions,
        "avg_function_len": avg_function_len,
    }

def features_ats(file):
    # ast dependency counts and max nesting depth
    result_ast = ast.parse(file)
    analyze = MyAnalyzer()
    analyze.visit(result_ast)
    max_nesting_depth = analyze.max_nesting_depth  # feat
    dependency_counts = analyze.dependency_counts #feat
    nesting_depth = analyze.current_depth
    is_tested = analyze.is_tested  # feat

    return {
        "max_nesting_depth": max_nesting_depth,
        "is_tested": is_tested,
        "dependency_count": dependency_counts
    }



def feature_git_raw(file, repo):
    author_list = set()
    churn_rate_total = 0

    file_commits = list(repo.iter_commits(paths=file))
    commit_count = len(file_commits)

    for commit in file_commits:
        author_list.add(commit.author.name)
        churn_rate = commit.stats.files
        change_file = file.replace("\\", "/")
        if change_file in churn_rate:
            churn_rate_total += (
                    churn_rate[change_file]['deletions']
                    + churn_rate[change_file]['insertions']
            )

    author_count = len(author_list)

    if commit_count > 0:
        days_since_change = (date.today() - file_commits[0].committed_datetime.date()).days
    else:
        days_since_change = 0

    return {
        "commit_count": commit_count,
        "author_count": author_count,
        "days_since_change": days_since_change,
        "churn_rate": churn_rate_total,
    }

def normalization_relative_to_repo(df_features, repo_col="repo"):
    relative_cols = ["commit_count", "author_count", "churn_rate", "days_since_change"]
    for col in relative_cols:
        medians = df_features.groupby(repo_col)[col].transform("median")
        medians = medians.replace(0, 1)
        df_features[col] = df_features[col] / medians
    return df_features