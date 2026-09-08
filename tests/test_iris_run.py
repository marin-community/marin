# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess

from marin.run import iris_run
from marin.run.iris_run import _should_stage


def test_working_dir_include_overrides_parent_exclusion():
    relative_path = "experiments/domain_phase_mix/exploratory/review/manifest.json"

    assert not _should_stage(relative_path, ["experiments/domain_phase_mix/exploratory"], [])
    assert _should_stage(
        relative_path,
        ["experiments/domain_phase_mix/exploratory"],
        ["experiments/domain_phase_mix/exploratory/review"],
    )
    assert not _should_stage(
        "experiments/domain_phase_mix/exploratory/other/large.csv",
        ["experiments/domain_phase_mix/exploratory"],
        ["experiments/domain_phase_mix/exploratory/review"],
    )


def test_working_dir_include_adds_gitignored_directory(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("ignored/\n")
    (tmp_path / "tracked.txt").write_text("tracked\n")
    subprocess.run(["git", "add", ".gitignore", "tracked.txt"], cwd=tmp_path, check=True)

    review_dir = tmp_path / "ignored/review"
    review_dir.mkdir(parents=True)
    (review_dir / "manifest.json").write_text("{}\n")
    other_dir = tmp_path / "ignored/other"
    other_dir.mkdir(parents=True)
    (other_dir / "large.csv").write_text("excluded\n")

    monkeypatch.setattr(iris_run, "_iris_revision_date", lambda _: "")
    staged = iris_run._create_filtered_workspace(tmp_path, ["ignored"], ["ignored/review"])
    try:
        assert (staged / "tracked.txt").is_file()
        assert (staged / "ignored/review/manifest.json").is_file()
        assert not (staged / "ignored/other/large.csv").exists()
    finally:
        shutil.rmtree(staged)
