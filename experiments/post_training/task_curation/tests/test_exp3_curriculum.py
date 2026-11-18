# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify adopted release digests, row counts, and provenance."""

import hashlib
import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.post_training.task_curation.datasets.long_loop_rl import AdoptedFile, AdoptedRelease, verify_release


def test_verify_release_admits_the_recorded_composition(tmp_path):
    config = _local_release_config(tmp_path)
    composition = verify_release(config)
    assert composition["train.parquet"].rows == 2
    assert composition["train.parquet"].source_counts == {"alpha": 1, "beta": 1}
    tampered = replace(config, members=tuple(replace(member, sha256="0" * 64) for member in config.members))
    with pytest.raises(ValueError, match="does not match the pinned"):
        verify_release(tampered)
    miscounted = replace(
        config,
        members=tuple(
            replace(member, source_counts={"alpha": 2, "beta": 0}) if member.source_counts else member
            for member in config.members
        ),
    )
    with pytest.raises(ValueError, match="provenance counts"):
        verify_release(miscounted)


def _local_release_config(tmp_path):
    """A two-row local release shaped like the pinned ones: a task_spec payload column."""
    rows = [
        {"task_spec": json.dumps({"source": {"dataset": "alpha"}, "id": "a1"})},
        {"task_spec": json.dumps({"source": {"dataset": "beta"}, "id": "b1"})},
    ]
    release = tmp_path / "release"
    release.mkdir()
    path = release / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return AdoptedRelease(
        name="test-adopted-release",
        version="1",
        url=str(release),
        manifest_sha256="0" * 64,
        members=(AdoptedFile("train.parquet", sha256, 2, {"alpha": 1, "beta": 1}),),
    )
