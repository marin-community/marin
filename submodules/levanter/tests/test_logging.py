import pathlib

import pytest
from git import InvalidGitRepositoryError, NoSuchPathError, Repo

from levanter.tracker.helpers import infer_experiment_git_root


def test_infer_experiment_git_root():
    # make sure this test is running in a git repo
    try:
        repo = Repo(pathlib.Path(__file__), search_parent_directories=True)
    except (InvalidGitRepositoryError, NoSuchPathError):
        pytest.skip("test not running in a git repo")

    root = infer_experiment_git_root()

    # ensure that 1) this is a git root and 2) this source file is underneath
    assert root is not None
    assert pathlib.Path(root).exists()
    repo = Repo(root)
    assert repo.working_dir == root
    assert pathlib.Path(__file__).is_relative_to(root), f"{__file__} is not relative to {root}"
