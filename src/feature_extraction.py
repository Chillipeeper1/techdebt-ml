import ast
from ast import NodeVisitor
from itertools import count
from pathlib import Path
from git import Repo
import radon
from radon import raw
from radon.complexity import cc_rank,cc_visit
from radon import metrics
from datetime import date
import csv

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


def featureExtraction():
    # Extract a single file only to test
    #with open("./wicked-garden/scripts/_integration_resolver.py", 'r' ,encoding="utf-8") as f:
        #test_code = f.read()
    # Now extract all the file in the test repo directory excluding test files and .venv
    repos = Path("../cloned_repos")
    feat_list = {}
    repos.mkdir(exist_ok=True)
    with open("../data/raw/repos_candidatos.csv", mode="r", encoding="utf-8") as csv_file:
        read_csv = csv.DictReader(csv_file)

        #Everything starts here, it reads every single repo from the list repos_candidatos
        for row in read_csv:
            repo_url = row["clone_url"]
            repo_name = repo_url.split("/")[-1].replace(".git","")
            dest_route = repos / repo_name

            if not dest_route.exists():
                repo_git = Repo.clone_from(repo_url, dest_route)
            else:
                repo_git = Repo(dest_route)

            skip_dirs = ["test",".venv"]
            files = [
                str(f.relative_to(dest_route)) for f in dest_route.rglob('*.py')
                if not any(part in skip_dirs for part in f.parts)
            ]
            # Commit hist count of a test file

            #test_repo = Repo("./wicked-garden")
            #file_commits = list(test_repo.iter_commits(paths="./scripts/_agents.py" ))
            #count = len(file_commits)
            #print(f"total commits of the test file: {count}")

            # Commit hist count of all .py files

            #repo_commits = list(Repo("./wicked-garden").iter_commits(paths=files))

            # radon LOC, SLOC, comment_ratio, blank_ratio

            for file in files:
                author_list = set()
                count_bug = 0
                count_refactor = 0
                churn_rate_total = 0
                try:
                    with open(dest_route / file, 'r', encoding='utf-8') as f:

                        feat_file = f.read()
                        #TODO: read every single file of the repo and get its metrics with radon
                        result_raw = raw.analyze(feat_file)
                        result_raw_loc = raw.analyze(feat_file).loc #feat
                        result_raw_sloc = raw.analyze(feat_file).sloc #feat
                        result_raw_comments = raw.analyze(feat_file).comments
                        result_raw_multi = raw.analyze(feat_file).multi
                        result_raw_blank = raw.analyze(feat_file).blank

                        #Operations with raw metrics
                        if result_raw_comments > 0:
                            total_comments = result_raw_comments + result_raw_multi
                            comment_ratio = total_comments / result_raw_loc #feat
                            blank_ratio = result_raw_blank / result_raw_loc #feat
                        else:
                            comment_ratio = 0
                            blank_ratio = 0

                        # Cyclomatic complexity
                        result_cc = cc_visit(feat_file)

                        if len(result_cc) > 0:

                            complexity_score = 0
                            sum_func_lines = 0
                            highest_complexity = []
                            for score in result_cc:
                                complexity_score += score.complexity
                                highest_complexity.append(complexity_score)
                                func_lines = score.endline - score.lineno + 1
                                sum_func_lines += func_lines

                            num_functions = len(result_cc) #feat

                            avg_function_len = sum_func_lines / num_functions #feat
                            average_cc = complexity_score / num_functions  # feat
                            max_complexity = max(highest_complexity) #feat
                        else:
                            num_functions = 0
                            avg_function_len = 0
                            average_cc = 0
                            max_complexity = 0
                        # Maintainability Index
                        result_mi = metrics.mi_visit(feat_file, multi=True) #feat

                        #Halstead metrics (effort specifically)
                        result_hl = metrics.h_visit(feat_file).total.effort #feat

                        #ast dependency counts and max nesting depth
                        result_ast = ast.parse(feat_file)
                        analyze = MyAnalyzer()
                        analyze.visit(result_ast)
                        max_nesting_depth = analyze.max_nesting_depth #feat
                        dependency_counts = analyze.dependency_counts
                        nesting_depth = analyze.current_depth
                        is_tested = analyze.is_tested #feat
                        #print(analyze.dependency_counts, analyze.max_nesting_depth)

                        # Git analytics
                        file_commits = list(repo_git.iter_commits(paths=file))
                        commit_count = len(file_commits) #feat
                        for commit in file_commits:
                            author_list.add(commit.author.name)
                            bug_fix_commits = commit.message
                            bug_fix_commits_lower = bug_fix_commits.lower()
                            if "bug" in bug_fix_commits_lower or "fix" in bug_fix_commits_lower or "error" in bug_fix_commits_lower or "patch" in bug_fix_commits_lower:
                                count_bug += 1 #feat
                            if "refactor" in bug_fix_commits_lower or "cleanup" in bug_fix_commits_lower or "rewrite" in bug_fix_commits_lower:
                                count_refactor += 1 #feat

                            churn_rate = commit.stats.files
                            change_file = file.replace("\\", "/")
                            if change_file in churn_rate:
                                churn_rate_deletions = churn_rate[change_file]['deletions']
                                churn_rate_insertions = churn_rate[change_file]['insertions']
                                churn_rate_total += (churn_rate_insertions + churn_rate_deletions) #feat

                        author_count = len(author_list) #feat
                        if commit_count > 0:
                            days_since_change = (date.today() - file_commits[0].committed_datetime.date()).days #feat
                        else:
                            days_since_change = 0


                        #Labeling
                        label = 0
                        if count_bug >= 2:
                            label = 1 #feat

                        #Feat list for csv
                        feat_list[f"{repo_name}/{file}"] = {"loc" : result_raw_loc,
                                           "sloc":result_raw_sloc,
                                           "comment_ratio":comment_ratio,
                                           "blank_ratio":blank_ratio,
                                           "cyclomatic_complexity":average_cc,
                                           "max_complexity":max_complexity,
                                           "maintainability_index": result_mi,
                                           "halstead_effort":result_hl,
                                           "num_functions":num_functions,
                                           "avg_function_len": avg_function_len,
                                           "commit_count":commit_count,
                                           "author_count":author_count,
                                           "days_since_change":days_since_change,
                                           "bug_fix_commits":count_bug,
                                           "churn_rate":churn_rate_total,
                                           "refactor_commits":count_refactor,
                                           "dependency_count": dependency_counts,
                                           "max_nesting_depth":max_nesting_depth,
                                           "is_tested": is_tested,
                                           "label":label,
                                           }
                except Exception as e:
                    print(f"corrupted file: {file} , {e}")
                    continue

    #csv creation
    header = list(list(feat_list.values())[0].keys())
    header.insert(0, "file name")
    with open("../data/proccessed/feature_extraction.csv", mode='w', newline='') as csv_file:
        file_write = csv.DictWriter(csv_file, fieldnames=header)
        file_write.writeheader()
        for key, value in feat_list.items():
            value["file name"] = key
            file_write.writerow(value)
    #TODO: adapt to every single repo in repos_candidatos.csv
    # Commit hist count of all .py files

    # repo_commits = list(Repo("./wicked-garden").iter_commits(paths=files))
    #count = len(repo_commits)
    #print(count)


    #result_cc = cc_visit(test_code)
    #print(result_cc)

featureExtraction()