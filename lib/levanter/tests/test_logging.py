# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import pathlib

import jax
import pytest
from git import InvalidGitRepositoryError, NoSuchPathError, Repo

from levanter.tracker.helpers import infer_experiment_git_root
from levanter.utils.logging import init_logging


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


@pytest.mark.parametrize(
    ("process_count", "process_index", "expected_name"),
    [(1, 0, "run.log"), (2, 1, "run.p1.log")],
)
def test_init_logging_file_name_per_process(tmp_path, monkeypatch, process_count, process_index, expected_name):
    monkeypatch.setattr(jax, "process_count", lambda: process_count)
    monkeypatch.setattr(jax, "process_index", lambda: process_index)
    root = logging.getLogger()
    handlers_before = list(root.handlers)

    init_logging(tmp_path, "run")
    try:
        logging.getLogger("levanter").info("hello from this process")
    finally:
        for handler in root.handlers[:]:
            if handler not in handlers_before:
                root.removeHandler(handler)
                handler.close()

    assert [p.name for p in tmp_path.iterdir()] == [expected_name]
    assert "hello from this process" in (tmp_path / expected_name).read_text()
