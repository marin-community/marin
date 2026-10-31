# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a published verifier in the real local sandbox; no model GPUs are used."""

import base64
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from huggingface_hub import hf_hub_download
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.models import TaskRolloutSpec
from rolloutengine.test_support import ReplayModel, engine
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.machine import UnsupportedMachineSpec
from taskcompendium.grading_result import Outcome

from experiments.evaluation.mini_policy_runtime.profile import DATASET, REVISION
from experiments.evaluation.mini_policy_runtime.repair_release import repaired_row, wheel_requirements

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
@pytest.mark.timeout(900)
@pytest.mark.parametrize("benchmark", ["math500", "ifbench"])
async def test_published_package_grades_in_bubblewrap(benchmark):
    path = hf_hub_download(DATASET, "data/validation.parquet", repo_type="dataset", revision=REVISION)
    row = next(r for r in pq.read_table(path).to_pylist() if r["benchmark"] == benchmark)
    original = TaskRolloutSpec.model_validate_json(row["task_rollout_spec"])
    payload = next(resource for resource in original.task.resources.verifier if resource.path == "config.json")
    config = json.loads(base64.b64decode(payload.source.content_base64))
    answer = (
        "\\boxed{" + config["spec"]["doc"]["answer"] + "}"
        if benchmark == "math500"
        else "Please whisper today whisper now"
    )
    factory = LocalMachineFactory(hash_seed="20261030")
    model = ReplayModel([{"role": "assistant", "content": answer}])
    runner = engine(model, {"local": factory}, harnesses={original.harness: ShellHarness(command_timeout=None)})
    with pytest.raises(UnsupportedMachineSpec):
        await runner.run_task(original)
    wheels = wheel_requirements(os.environ["MINI840_GRADING_WHEELS_REVISION"])
    repaired = TaskRolloutSpec.model_validate_json(repaired_row(row, wheels)["task_rollout_spec"])
    result = await runner.run_task(repaired)
    receipt = {
        "task_id": original.task.id,
        "reward": result.grade.reward,
        "status": str(result.grade.status),
        "grade_error": result.grade.error,
        "diagnostics": result.grade.diagnostics,
        "control": "fixed reference response, not an evaluated model score",
    }
    (Path(os.environ["IRIS_OUTPUT_DIR"]) / f"{benchmark}-grader-control.json").write_text(json.dumps(receipt, indent=2))
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 1.0), receipt
    if benchmark == "ifbench":
        failing = engine(
            ReplayModel([{"role": "assistant", "content": "The constraint was ignored."}]),
            {"local": factory},
            harnesses={original.harness: ShellHarness(command_timeout=None)},
        )
        incorrect = await failing.run_task(repaired)
        assert (incorrect.grade.status, incorrect.grade.reward) == (Outcome.GRADED, 0.0), incorrect.grade
