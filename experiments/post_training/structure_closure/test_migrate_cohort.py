# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archived exact-token tasks retain their private grading contract after migration."""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rolloutengine.models import TaskRolloutSpec
from taskcompendium.grader import grader_config
from taskcompendium.models import TaskResource
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.structure_closure.migrate_cohort import migrate_cohort

FIXTURE = Path(__file__).parents[1] / "task_curation/tests/fixtures/structure_closure.json"


def test_archived_cohort_preserves_token_prompt_and_private_grading(tmp_path):
    row = json.loads(FIXTURE.read_text())
    source = tmp_path / "source"
    source.mkdir()
    parquet = source / "validation.parquet"
    pq.write_table(pa.Table.from_pylist([row]), parquet)
    digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
    output = tmp_path / "converted"
    migrate_cohort(source=str(source), output=str(output), hashes={"validation": digest})
    [record] = pq.read_table(output / "validation.parquet").to_pylist()
    migrated = TaskRolloutSpec.model_validate_json(record["task_rollout_spec"])
    native = json.loads(row["lowered_task_spec"])
    [resource] = native["task"]["resources"]["verifier"]
    original = json.loads(resource_bytes(TaskResource.model_validate(resource)))
    assert grader_config(migrated.task) == original
    assert migrated.task.context.model_dump(mode="json") == native["task"]["context"]
    assert migrated.harness == "structure_closure"
    assert migrated.task_machine is None and migrated.verifier_machine is None
    assert migrated.limits.max_turns == native["session"]["max_turns"]
    assert migrated.limits.model_turn_timeout == native["session"]["model_turn_timeout"]

    # A changed archive at the same locator must not produce a different training cohort silently.
    pq.write_table(pa.Table.from_pylist([row, row]), parquet)
    with pytest.raises(ValueError, match="hash differs"):
        migrate_cohort(source=str(source), output=str(tmp_path / "changed"), hashes={"validation": digest})
    assert not (tmp_path / "changed/validation.parquet").exists()
