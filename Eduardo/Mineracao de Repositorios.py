from pathlib import Path
import pandas as pd
import os
import urllib
from pydriller import Repository
import requests

project_root_dir = Path(__file__).parent.parent
dataset_dir = os.path.join(project_root_dir, "datasets")
dataset_file = os.path.join(dataset_dir, "NICHE.csv")
metainfo_dir = os.path.join(project_root_dir, "Datasets", "metainfo")

if not os.path.exists(metainfo_dir):
    os.makedirs(metainfo_dir)

def check_connectivity(reference='http://google.com'):
    try:
        urllib.request.urlopen(reference)
        return True
    except urllib.request.URLError:
        return False

def get_all_projects():
    df = pd.read_csv(dataset_file)
    projects = df["GitHub Repo"].tolist()
    engineered_ml_projects = df[df["Engineered ML Project"] == 1]["GitHub Repo"].tolist()
    return projects, engineered_ml_projects, df

def get_metainfo_from_commits(repo: Repository):
    commits = []
    try:
        for commit in repo.traverse_commits():
            try:
                record = {
                    'commit_hash': commit.hash,
                    'author_name': commit.author.name,
                    'author_date': commit.author_date,
                    'committer_name': commit.committer.name,
                    'committer_date': commit.committer_date,
                }
                commits.append(record)
            except Exception as e:
                print(f'Problem reading commit {commit.hash}: {e}')
                continue
    except Exception as e:
        print(f'Problem traversing commits in repository: {e}')
        raise e

    return commits

def check_if_exists(project_name):
    return os.path.exists(os.path.join(metainfo_dir, project_name + ".csv"))

def calculate_date_range(commits):
    if not commits:
        return None, None
    dates = [commit['author_date'] for commit in commits]
    min_date = min(dates)
    max_date = max(dates)
    return min_date, max_date

if __name__ == '__main__':
    projects, engineered_ml_projects, df = get_all_projects()
    failed_projects = []

    for project in projects:
        if project in failed_projects:
            print(f"Skipping {project} because it previously failed.")
            continue

        print(f"Analyzing project: {project}")

        if check_if_exists(project):
            print(f"Data for {project} already exists. Skipping.")
            continue

        url = f"https://github.com/{project}"

        try:
            response = requests.get(url)
            if response.status_code == 404:
                print(f"Repository {project} not found. Skipping.")
                failed_projects.append(project)
                continue

            repo = Repository(url)
            commits = get_metainfo_from_commits(repo)
            min_date, max_date = calculate_date_range(commits)

            df_commits = pd.DataFrame(commits)
            df_project_info = df[df["GitHub Repo"] == project]
            if not df_project_info.empty:
                engineered_ml_project = df_project_info["Engineered ML Project"].values[0]
            else:
                engineered_ml_project = None

            df_commits['project_name'] = project
            df_commits['date_range_start'] = min_date
            df_commits['date_range_end'] = max_date
            df_commits['Engineered_ML_Project'] = engineered_ml_project
            safe_project_name = project.replace('/', '_')
            project_metainfo_dir = os.path.join(metainfo_dir, safe_project_name)
            if not os.path.exists(project_metainfo_dir):
                os.makedirs(project_metainfo_dir)

            df_commits.to_csv(os.path.join(project_metainfo_dir, f"{safe_project_name}.csv"), index=False)

            print(f"Data for {project}:")
            print(f"Date Range: {min_date} to {max_date}")
            print(f"Engineered ML Project: {engineered_ml_project}")

        except Exception as e:
            print(f"Problem analyzing repository {project}: {e}")
            failed_projects.append(project)
            continue
