"""
TechDebt ML — Repository search
===============================
Searches for Python repos on GitHub, filters the ones that have
enough documented bugs, and saves the result to a CSV.

Usage:
    pip install requests python-dotenv
    python repo_search.py

Requirements:
    A .env file with: GITHUB_TOKEN=ghp_yourTokenHere
"""

import os
import time
import csv
import random
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")

if not TOKEN:
    raise EnvironmentError(
        "GITHUB_TOKEN was not found.\n"
        "Create a .env file with: GITHUB_TOKEN=ghp_yourTokenHere"
    )

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "min_stars": 5,
    "max_stars": 500,
    "max_size_kb": 8000,     # small repos (~3 MB maximum)
    "min_closed_bugs": 5,    # minimum of closed issues labeled "bug"
    "pages": 10,             # search pages (30 repos per page)
    "request_pause": 1,      # seconds between requests (respects rate limit)
    "output_file": "candidate_repos.csv"
}

# ─── Functions ────────────────────────────────────────────────────────────────

def get(url, params=None):
    """GET request with rate limit handling."""
    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code == 403:
        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
        wait = max(0, reset_time - int(time.time())) + 5
        print(f"  Rate limit reached. Waiting {wait}s...")
        time.sleep(wait)
        response = requests.get(url, headers=HEADERS, params=params)

    return response


def search_repos():
    """Searches for Python repos using the CONFIG filters."""
    print("Searching for repositories on GitHub...")
    repos = []

    for page in range(1, CONFIG["pages"] + 1):
        print(f"  Page {page}/{CONFIG['pages']}...", end=" ")

        params = {
            "q": (
                f"language:python "
                f"stars:{CONFIG['min_stars']}..{CONFIG['max_stars']} "
                f"size:<{CONFIG['max_size_kb']} "
                f"is:public"
            ),
            "sort": "updated",
            "per_page": 30,
            "page": page
        }

        response = get("https://api.github.com/search/repositories", params)

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.json().get('message')}")
            break

        items = response.json().get("items", [])
        repos.extend(items)
        print(f"{len(items)} repos found")

        time.sleep(CONFIG["request_pause"])

    print(f"\nTotal found: {len(repos)} repos\n")
    return repos


def count_closed_bugs(repo_fullname):
    """Counts closed issues labeled 'bug' (excludes PRs)."""
    params = {
        "state": "closed",
        "labels": "bug",
        "per_page": 100
    }

    response = get(
        f"https://api.github.com/repos/{repo_fullname}/issues",
        params
    )

    if response.status_code != 200:
        return 0

    issues = response.json()
    # GitHub mixes issues and PRs — we filter only real issues
    only_issues = [i for i in issues if "pull_request" not in i]
    return len(only_issues)


def get_repo_details(repo):
    """Extracts the relevant fields of a repo for the CSV."""
    return {
        "name": repo["full_name"],
        "description": (repo.get("description") or "")[:100],
        "stars": repo["stargazers_count"],
        "forks": repo["forks_count"],
        "open_issues": repo["open_issues_count"],
        "size_kb": repo["size"],
        "created": repo["created_at"][:10],
        "updated": repo["updated_at"][:10],
        "url": repo["html_url"],
        "clone_url": repo["clone_url"],
        "closed_bugs": 0  # filled in during filtering
    }


def filter_by_bugs(repos_raw):
    """Checks the closed bugs of each repo and filters the ones meeting the minimum."""
    print("Checking closed bugs per repo...")
    candidates = []
    total = len(repos_raw)

    for i, repo in enumerate(repos_raw):
        name = repo["full_name"]
        print(f"  [{i+1}/{total}] {name}...", end=" ")

        n_bugs = count_closed_bugs(name)
        print(f"{n_bugs} closed bugs")

        if n_bugs >= CONFIG["min_closed_bugs"]:
            details = get_repo_details(repo)
            details["closed_bugs"] = n_bugs
            candidates.append(details)

        time.sleep(CONFIG["request_pause"])

    return candidates


def save_csv(candidates):
    """Saves the candidate repos to a CSV sorted by bugs."""
    if not candidates:
        print("\nNo repos with enough bugs were found.")
        return

    sorted_candidates = sorted(
        candidates,
        key=lambda r: r["closed_bugs"],
        reverse=True
    )

    file = CONFIG["output_file"]
    columns = [
        "name", "closed_bugs", "stars", "forks",
        "open_issues", "size_kb", "created",
        "updated", "description", "url", "clone_url"
    ]

    with open(file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(sorted_candidates)

    print(f"\nSaved to: {file}")
    return file


def print_summary(candidates):
    """Shows a summary of the repos found."""
    if not candidates:
        return

    print("\n" + "="*60)
    print(f"  CANDIDATE REPOS FOR TECHDEBT ML")
    print("="*60)
    print(f"  Total: {len(candidates)} repos")
    print(f"  Criterion: >= {CONFIG['min_closed_bugs']} closed bugs")
    print("="*60)

    sorted_candidates = sorted(candidates, key=lambda r: r["closed_bugs"], reverse=True)

    for repo in sorted_candidates[:15]:  # shows the top 15
        print(
            f"  {repo['closed_bugs']:>3} bugs | "
            f"{repo['stars']:>3} stars | "
            f"{repo['name']}"
        )

    if len(sorted_candidates) > 15:
        print(f"  ... and {len(sorted_candidates) - 15} more in the CSV")

    print("="*60)


def get_random_repo():
    """Returns a random Python repo matching the base CONFIG query.

    Picks a random page (the search API exposes up to 1000 results:
    10 pages of 100) and a random repo from that page. Returns the repo dict
    exactly as the GitHub API delivers it, or None if none was obtained.
    It does not apply the closed-bugs filter: only the base query.
    """
    per_page = 100
    max_pages = 10  # 10 * 100 = 1000, the maximum the search API exposes

    query = (
        f"language:python "
        f"stars:{CONFIG['min_stars']}..{CONFIG['max_stars']} "
        f"size:<{CONFIG['max_size_kb']} "
        f"is:public"
    )

    def _request(page):
        params = {
            "q": query,
            "sort": "updated",
            "per_page": per_page,
            "page": page,
        }
        return get("https://api.github.com/search/repositories", params)

    response = _request(random.randint(1, max_pages))
    if response.status_code != 200:
        print(f"  Error {response.status_code}: {response.json().get('message')}")
        return None

    data = response.json()
    items = data.get("items", [])

    # If the page fell outside the real range of results, retry within the range.
    if not items:
        total = data.get("total_count", 0)
        if total == 0:
            return None
        real_pages = max(1, min(max_pages, -(-total // per_page)))
        response = _request(random.randint(1, real_pages))
        if response.status_code != 200:
            return None
        items = response.json().get("items", [])

    if not items:
        return None

    return random.choice(items)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\nTechDebt ML — Repository search")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}\n")

    repos_raw = search_repos()

    if not repos_raw:
        print("No repos were found. Check your token or the filters.")
        return

    candidates = filter_by_bugs(repos_raw)
    save_csv(candidates)
    print_summary(candidates)

    print(f"\nEnd: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Next step: clone the repos from the CSV and extract features with Radon + GitPython\n")


if __name__ == "__main__":
    main()
