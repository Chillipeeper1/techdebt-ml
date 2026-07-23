from pathlib import Path
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR.parent / "data"

candidates = pd.read_csv(DATA_DIR / "raw" / "candidate_repos.csv")
final_dataset = pd.read_csv(DATA_DIR / "proccessed" / "repo_features_dataset_final_v2.csv")

used_repos = final_dataset["repo"].unique()
source_repos = candidates[candidates["name"].str.split("/").str[-1].isin(used_repos)]
source_repos = source_repos[["name", "url"]].drop_duplicates()

source_repos.to_csv(SRC_DIR / "source_repos.csv", index=False)
print(source_repos)
