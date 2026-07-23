from pathlib import Path
from git import Repo
import csv
from feature_lib import MyAnalyzer, features_radon, features_ats, feature_git_raw, normalization_relative_to_repo
import pandas as pd


def featureExtraction():
    repos = Path("../cloned_repos")
    feat_list = {}
    repos.mkdir(exist_ok=True)
    with open("../data/raw/candidate_repos.csv", mode="r", encoding="utf-8") as csv_file:
        read_csv = csv.DictReader(csv_file)
        #Everything starts here, it reads every single repo from the list candidate_repos
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
            for file in files:
                author_list = set()
                count_bug = 0
                count_refactor = 0
                churn_rate_total = 0
                try:
                    with open(dest_route / file, 'r', encoding='utf-8') as f:

                        feat_file = f.read()
                        features = {}
                        #radon metrics computing
                        features.update(features_radon(feat_file))
                        #ats dependency counts and max nesting depth
                        features.update(features_ats(feat_file))
                        # git analytics
                        features.update(feature_git_raw(file, repo_git))
                        # bug count
                        file_commits = list(repo_git.iter_commits(paths=file))
                        commit_count = len(file_commits)
                        for commit in file_commits:
                            bug_fix_commits = commit.message
                            bug_fix_commits_lower = bug_fix_commits.lower()
                            if "bug" in bug_fix_commits_lower or "fix" in bug_fix_commits_lower or "error" in bug_fix_commits_lower or "patch" in bug_fix_commits_lower:
                                count_bug += 1 #feat
                            if "refactor" in bug_fix_commits_lower or "cleanup" in bug_fix_commits_lower or "rewrite" in bug_fix_commits_lower:
                                count_refactor += 1 #feat

                        features["bug_fix_commits"] = count_bug
                        features["refactor_commits"] = count_refactor
                        features["repo"] = repo_name
                        #Labeling
                        label = 0
                        if count_bug >= 2:
                            label = 1 #feat
                        features["label"] = label
                        #Feat list for csv
                        feat_list[f"{repo_name}/{file}"] = features
                except Exception as e:
                    print(f"corrupted file: {file} , {e}")
                    continue

    #csv creation
    header = list(list(feat_list.values())[0].keys())
    header.insert(0, "file name")
    df = pd.DataFrame.from_dict(feat_list, orient="index")
    df.index.name = "file name"
    df = normalization_relative_to_repo(df, repo_col="repo")
    df.to_csv("../data/proccessed/feature_extraction_V3.csv")
featureExtraction()