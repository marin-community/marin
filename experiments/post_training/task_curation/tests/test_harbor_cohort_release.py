# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The cohort release stage publishes the whole converted population, strictly."""

import json

import pyarrow.parquet as pq
import pytest

from experiments.post_training.task_curation.harbor_cohort_release import (
    RECEIPTS_FILENAME,
    release_repo,
    source_slug,
    write_conversion_artifacts,
)


def test_release_repo_names_are_deterministic_and_private_safe():
    assert source_slug("TaskTrove:laion__nemotron-gym-identity-following-v4") == (
        "laion__nemotron-gym-identity-following-v4"
    )
    repo = release_repo("TaskTrove:laion__nemotron-gym-identity-following-v4", "20261124a")
    assert repo.startswith("open-athena/") and repo.endswith("-taskspec-20261124a")
    assert repo == release_repo("TaskTrove:laion__nemotron-gym-identity-following-v4", "20261124a")
    # Two sources sharing a long slug still name distinct repos through the id digest.
    other = release_repo("TaskTrove:laion__nemotron-gym-identity-following-v4@repaired", "20261124a")
    assert other != repo


def test_artifacts_carry_the_whole_population_and_refuse_a_shrunken_one(tmp_path):
    """Any rejected row fails the run: a shrunken population silently changes the census contract."""
    rows = [
        {"path": f"row-{index}", "task_spec": json.dumps({"id": f"t{index}"}), "task_sha256": f"{index:064d}"}
        for index in range(4)
    ]
    receipts = [{"task_id": f"t{index}", "protocol": "harbor-cohort-taskspec-v1"} for index in range(4)]
    counts = {"input_rows": 4, "converted_rows": 4}
    manifest = write_conversion_artifacts(rows, receipts, counts, output_root=str(tmp_path), source_id="Marin:fixture")
    assert manifest["task_specs_sha256"]
    assert manifest["rejections"] == {}
    written = pq.read_table(tmp_path / "tasks.parquet").to_pylist()
    assert [row["path"] for row in written] == [f"row-{index}" for index in range(4)]
    ledger = [json.loads(line) for line in (tmp_path / RECEIPTS_FILENAME).read_text().splitlines()]
    assert len(ledger) == 4
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest

    with pytest.raises(ValueError, match="must convert every input row"):
        write_conversion_artifacts(
            rows[:3],
            receipts[:3],
            {"input_rows": 4, "converted_rows": 3},
            output_root=str(tmp_path / "second"),
            source_id="Marin:fixture",
        )
