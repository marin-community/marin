# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from taskcompendium.convert.answers import exact_answer_task
from taskcompendium.harbor.runtime import write_source_runtime
from taskcompendium.models import Source
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import inline_resource


@pytest.mark.parametrize("role,section", [("task_machine", "environment"), ("verifier_machine", "verifier.environment")])
def test_source_role_budget_rejects_conflicting_original_configuration(tmp_path, role, section):
    task = exact_answer_task(
        RawRow("conflict", Source(dataset="fixture", revision="pin", row="0", importer_revision="1"), {}),
        prompt="Return 42.",
        answers=("42",),
        ignore_case=False,
    )
    task = task.model_copy(
        update={
            "resources": task.resources.model_copy(
                update={
                    "oracle": (inline_resource("task.toml", f"[{section}]\ncpus = 1\n".encode()),),
                }
            )
        }
    )
    runtime = {role: {"cpus": 2}}
    with pytest.raises(ValueError, match="conflicts"):
        write_source_runtime(task, runtime, tmp_path)
    assert not (tmp_path / "source-runtime.json").exists()
