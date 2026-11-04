# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import pyarrow.parquet as pq
from shellbox.machine import CommandSemantics
from taskcompendium.examples.protocol_probes import probe_check
from taskcompendium.models import SCHEMA_VERSION, TaskSpec, VerifyitGrader
from taskcompendium.runtime.local import LOCK_FILE

from experiments.post_training.inventory.protocol_probes import PROBES_FILENAME, write_probe_tasks
from experiments.post_training.task_curation.inventory import TASK_SPEC_COLUMN


def test_probe_rows_round_trip_with_a_shellsim_task_machine_and_a_lock_graded_answer_file(tmp_path):
    grader_root = tmp_path / "env"
    write_probe_tasks(str(tmp_path), str(grader_root), variants=2, schema_version=SCHEMA_VERSION)

    rows = pq.read_table(tmp_path / PROBES_FILENAME).column(TASK_SPEC_COLUMN).to_pylist()
    tasks = [TaskSpec.model_validate_json(row) for row in rows]
    checks = Counter(probe_check(task) for task in tasks)
    assert checks == {"required_file": 2, "tool_execution": 2, "thinking_disabled": 2, "url_loading": 2}
    assert len({task.id for task in tasks}) == len(tasks)

    [file_task] = [task for task in tasks if task.id == "required_file-0"]
    assert file_task.environment_requirements.command_semantics == CommandSemantics.SHELL_SIMULATOR
    assert isinstance(file_task.grader, VerifyitGrader)
    assert file_task.grader.environment is not None
    assert file_task.grader.environment.packages_lock == str(grader_root / LOCK_FILE)
