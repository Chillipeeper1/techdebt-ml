from huggingface_hub import HfApi
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
api = HfApi()
repo_id = "ChilliPeep/techdebt-ml-dataset"

api.upload_file(
    path_or_fileobj=DATA_DIR / "proccessed" / "repo_features_dataset_final_v2.csv",
    path_in_repo="dataset.csv",
    repo_id=repo_id,
    repo_type="dataset",
)
api.upload_file(
    path_or_fileobj=ROOT_DIR / "docs" / "README_hf.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="dataset",
)
api.upload_file(
    path_or_fileobj=SRC_DIR / "source_repos.csv",
    path_in_repo="source_repos.csv",
    repo_id=repo_id,
    repo_type="dataset",
)