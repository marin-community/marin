# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from cloud.iris.launch_config import load_launch_config
from harbor.models.job.config import JobConfig
from omegaconf import OmegaConf
from skyrl_train.callbacks.base import CallbackHandler, TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import HarborEvaluationCallback, create_callbacks_from_config
from skyrl_train.evaluation_pool import EvaluationPool

config, changed = (load_launch_config(path) for path in map(Path, sys.argv[1:3]))
try:
    load_launch_config(Path(sys.argv[3]))
except ValueError as error:
    assert "Harbor output tokens must leave room for input" in str(error), str(error)
else:
    raise AssertionError("Harbor accepted a reply cap equal to the serving window")

windows = []
for launch in (config, changed):
    window = launch.skyrl.generator.engine_init_kwargs.max_model_len
    windows.append(window)
    harbor = next(row for row in launch.skyrl.trainer.callbacks if row.type == "harbor_evaluation")
    for cohort in harbor.cohorts:
        job = JobConfig.model_validate(OmegaConf.to_container(cohort.job, resolve=True))
        model_info = job.agents[0].kwargs["model_info"]
        assert model_info["max_input_tokens"] == window, model_info
        assert model_info.get("max_output_tokens", 0) < window, (model_info, window)
        agent = job.agents[0]
        verifier_budget = job.verifier.override_timeout_sec or job.verifier.max_timeout_sec
        assert verifier_budget is not None
        agent_budget = agent.override_setup_timeout_sec + agent.override_timeout_sec
        assert agent_budget + verifier_budget < job.trial_attempt_timeout_sec
        attempt_budget = job.trial_attempt_timeout_sec + job.trial_cleanup_grace_sec
        assert 2 * attempt_budget + job.retry.min_wait_sec < harbor.timeout
assert windows[0] != windows[1]

changed_harbor = next(row for row in changed.skyrl.trainer.callbacks if row.type == "harbor_evaluation")
changed_harbor.cohorts[0].job.environment.kwargs.image_build = "unrecognized-build"
try:
    create_callbacks_from_config(changed.skyrl)
except ValueError as error:
    assert "unrecognized-build" in str(error), str(error)
else:
    raise AssertionError("Harbor accepted an unknown Daytona image build")

selected = []
for index, (native, cohorts) in enumerate(((99, (27, 27, 27)), (98, (30, 30, 30)), (100, (26, 30, 30)))):
    directory = Path(sys.argv[1]).parent / f"coverage-{index}"
    config.skyrl.trainer.ckpt_path = str(directory)
    metrics = {"eval/all/num_attempted": 100, "eval/all/num_scored": native, "eval/all/avg_verifier_score": 0.5}
    harbor = next(row for row in config.skyrl.trainer.callbacks if row.type == "harbor_evaluation")
    for cohort, scored in zip(harbor.cohorts, cohorts, strict=True):
        prefix = f"eval/harbor/{cohort.name}/"
        metrics.update({prefix + "num_attempted": 30, prefix + "num_scored": scored, prefix + "reward_sum": scored / 2})
    handler = CallbackHandler(create_callbacks_from_config(config.skyrl))
    state = TrainerState(global_step=0, epoch=0, total_steps=100, num_steps_per_epoch=100)
    asyncio.run(
        handler.call_event_async(
            "on_evaluate", state, TrainerControl(), metrics=metrics, trainer=SimpleNamespace(cfg=config.skyrl)
        )
    )
    selected.append((directory / "best_checkpoint.json").exists())


async def check_runtime_handoff():
    callback = next(
        item for item in create_callbacks_from_config(config.skyrl) if isinstance(item, HarborEvaluationCallback)
    )
    trainer = SimpleNamespace(
        cfg=config.skyrl,
        inference_engine_client=SimpleNamespace(
            enable_http_endpoint=True,
            model_name="policy",
            http_endpoint_host="127.0.0.1",
            http_endpoint_port=8000,
        ),
    )
    original_exec = asyncio.create_subprocess_exec
    expected = None

    async def start_worker(*args, **kwargs):
        worker = Path(__file__).with_name("native_harbor_worker.py")
        return await original_exec(sys.executable, str(worker), args[-1], json.dumps(expected), **kwargs)

    asyncio.create_subprocess_exec = start_worker
    try:
        endpoint = "http://127.0.0.1:4318/iris-capability/v1/telemetry"
        os.environ.update(
            SKYRL_RUN_ID=str(config.run.id),
            SKYRL_EXECUTION_UID="iris:uid-native",
            IRIS_TASK_ID="/test/native/0:1",
        )
        for enabled in (True, False):
            if enabled:
                os.environ["SKYRL_TELEMETRY_ENDPOINT"] = endpoint
            else:
                os.environ.pop("SKYRL_TELEMETRY_ENDPOINT", None)
            expected = (
                {
                    "endpoint": endpoint,
                    "root_run_uid": str(config.run.id),
                    "execution_uid": "iris:uid-native",
                    "serving_job_id": "/test/native",
                }
                if enabled
                else None
            )
            async with EvaluationPool(3) as pool:
                metrics = await callback.collect_evaluation(state, trainer=trainer, pool=pool)
            assert all(
                metrics[f"eval/harbor/{cohort['name']}/num_scored"] == len(cohort["job"]["tasks"])
                for cohort in callback.cohorts
            ), metrics
    finally:
        asyncio.create_subprocess_exec = original_exec


os.environ["DAYTONA_EVAL_API_KEY"] = "test-eval-credential"
asyncio.run(check_runtime_handoff())
print(json.dumps(selected))
