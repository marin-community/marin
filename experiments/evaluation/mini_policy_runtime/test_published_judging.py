# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the published OlympiadBench fallback against a live co-hosted judge."""

import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from aiohttp import ClientSession, ClientTimeout, web
from aiohttp.test_utils import TestServer
from huggingface_hub import hf_hub_download
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.models import TaskRolloutSpec
from rolloutengine.test_support import ReplayModel, engine
from shellbox.backends.local.machine import LocalMachineFactory
from taskcompendium.grading_result import Outcome

from experiments.evaluation.mini_policy_runtime.profile import DATASET, SOURCE_REVISION
from experiments.evaluation.mini_policy_runtime.repair_release import repaired_row, wheel_requirements

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
@pytest.mark.timeout(900)
async def test_published_olympiad_package_calls_live_judge(monkeypatch):
    path = hf_hub_download(DATASET, "data/validation.parquet", repo_type="dataset", revision=SOURCE_REVISION)
    row = next(row for row in pq.read_table(path).to_pylist() if row["benchmark"] == "olympiadbench")
    wheels = wheel_requirements(os.environ["MINI840_GRADING_WHEELS_REVISION"])
    task = TaskRolloutSpec.model_validate_json(repaired_row(row, wheels)["task_rollout_spec"])
    # A nonempty, incorrect extraction enters the shared grader's LLM fallback.
    model = ReplayModel([{"role": "assistant", "content": r"\boxed{99999999999999999999999999999999}"}])
    runner = engine(
        model,
        {"local": LocalMachineFactory(hash_seed="20261030")},
        harnesses={task.harness: ShellHarness(command_timeout=None)},
    )
    upstream = os.environ["JUDGE_BASE_URL"]
    requests = []

    async def forward(request):
        payload = await request.json()
        async with ClientSession(timeout=ClientTimeout(total=300)) as session:
            async with session.post(
                upstream.rstrip("/") + "/chat/completions",
                json=payload,
                headers={"Authorization": request.headers["Authorization"]},
            ) as response:
                body = await response.read()
                requests.append({"model": payload["model"], "status": response.status})
                return web.Response(body=body, status=response.status, content_type="application/json")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", forward)
    async with TestServer(app) as proxy:
        monkeypatch.setenv("JUDGE_BASE_URL", str(proxy.make_url("/v1")))
        result = await runner.run_task(task)
    receipt = {
        "task_id": task.task.id,
        "reward": result.grade.reward,
        "status": str(result.grade.status),
        "grade_error": result.grade.error,
        "judge": "MiniMaxAI/MiniMax-M3-MXFP8",
        "judge_requests": requests,
        "control": "deliberately incorrect fixed response; not a candidate model score",
    }
    (Path(os.environ["IRIS_OUTPUT_DIR"]) / "live-judge-control.json").write_text(json.dumps(receipt, indent=2))
    assert requests and all(r["model"] == receipt["judge"] and r["status"] == 200 for r in requests)
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 0.0), result.grade.error
