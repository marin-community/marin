"""Probe generation overlaps requests without changing the frozen sample bank."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from skyrl_train.mismatch_probe.collect import ProbeCollector
from skyrl_train.mismatch_probe.protocol import request_seed
from skyrl_train.task_inputs import PreferenceInput, TrajectoryMetadata


class _PromptDataset(list):
    @staticmethod
    def collate_fn(rows):
        return rows


class _ControlledRunner:
    """A rollout provider whose completions and failures the test controls."""

    def __init__(self):
        self.started = asyncio.Queue()
        self.pending = {}
        self.requests = {}
        self.active = set()
        self.peak_active = 0
        self.sessions = []

    async def start_eval_session(self, **kwargs):
        self.sessions.append("start")

    async def stop_eval_session(self):
        assert not self.active, "provider requests must drain before the frozen session closes"
        self.sessions.append("stop")

    async def run(self, request):
        assert self.sessions == ["start"]
        identity = request.trajectory_ids[0]
        sample_id = f"{identity.instance_id}:{identity.repetition_id}"
        self.requests[sample_id] = request
        self.active.add(sample_id)
        self.peak_active = max(self.peak_active, len(self.active))
        completion = asyncio.get_running_loop().create_future()
        self.pending[sample_id] = completion
        await self.started.put(sample_id)
        try:
            await completion
            token = 10 * int(identity.instance_id.removeprefix("prompt-")) + identity.repetition_id + 1
            return {
                "prompt_token_ids": list(request.prompts),
                "response_ids": [[token]],
                "rewards": [1.0],
                "loss_masks": [[1]],
                "rollout_logprobs": [np.array([-float(token)], dtype=np.float32)],
                "rollout_routed_experts": [np.array([[[token, token + 1]]], dtype=np.uint16)],
            }
        finally:
            self.active.remove(sample_id)


def _trainer(runner, *, runner_limit=8, serving_dp=2, max_num_seqs=1):
    config = OmegaConf.create(
        {
            "trainer": {
                "mismatch_probe": {
                    "updates": 0,
                    "archive_uri": "unused",
                    "seed": 42,
                    "prompts": {"count": 2, "samples_per_prompt": 3},
                },
                "algorithm": {"tis_lcs_alert_threshold": 0.0},
            },
            "trajectory_runner": {"max_concurrent_tasks": runner_limit},
            "generator": {
                "backend": "vllm",
                "num_inference_engines": 1,
                "inference_engine_data_parallel_size": serving_dp,
                "max_num_seqs": max_num_seqs,
                "sampling_params": {
                    "temperature": 1.0,
                    "max_generate_length": 16,
                    "top_p": 1.0,
                    "top_k": -1,
                    "min_p": 0.0,
                    "logprobs": 0,
                },
            },
        }
    )
    rows = _PromptDataset(
        {
            "uid": f"prompt-{index}",
            "input": PreferenceInput([100 + index], {}, TrajectoryMetadata(f"prompt-{index}", "math")),
        }
        for index in range(2)
    )
    return SimpleNamespace(cfg=config, global_step=17, eval_dataset=rows, trajectory_runner=runner)


@pytest.mark.parametrize("capacity", [{"runner_limit": 2, "serving_dp": 3}, {"runner_limit": 8, "serving_dp": 2}])
def test_probe_generation_is_bounded_and_preserves_ordered_seeded_rows(capacity):
    async def scenario():
        runner = _ControlledRunner()
        trainer = _trainer(runner, **capacity)
        collection = asyncio.create_task(ProbeCollector(trainer.cfg)._generate(trainer))
        async with asyncio.timeout(5):
            # Keep the first request blocked while every other request finishes.
            # A serial implementation cannot admit the second request here.
            first = await runner.started.get()
            for _ in range(5):
                admitted = await runner.started.get()
                runner.pending[admitted].set_result(None)
            runner.pending[first].set_result(None)
            samples = await collection

        expected_ids = [f"prompt-{prompt}:{repetition}" for prompt in range(2) for repetition in range(3)]
        assert runner.peak_active == 2
        assert runner.sessions == ["start", "stop"]
        assert samples.sample_ids == expected_ids
        assert samples.prompt_ids == ["prompt-0"] * 3 + ["prompt-1"] * 3
        expected_seeds = [
            request_seed(42, f"prompt-{prompt}", repetition) for prompt in range(2) for repetition in range(3)
        ]
        assert samples.seeds == expected_seeds
        assert [runner.requests[key].sampling_params["seed"] for key in expected_ids] == expected_seeds
        assert [runner.requests[key].trajectory_ids[0].repetition_id for key in expected_ids] == [0, 1, 2] * 2
        assert samples.trajectory["response_ids"] == [[1], [2], [3], [11], [12], [13]]
        assert samples.trajectory["prompt_token_ids"] == [[100]] * 3 + [[101]] * 3
        assert [row.tolist() for row in samples.trajectory["rollout_logprobs"]] == [
            [-1.0],
            [-2.0],
            [-3.0],
            [-11.0],
            [-12.0],
            [-13.0],
        ]
        assert [row[0, 0, 0] for row in samples.trajectory["rollout_routed_experts"]] == [1, 2, 3, 11, 12, 13]

    asyncio.run(scenario())


@pytest.mark.parametrize("cause", ["provider_failure", "caller_cancelled"])
def test_probe_generation_drains_cancelled_requests_before_closing_session(cause):
    async def scenario():
        runner = _ControlledRunner()
        trainer = _trainer(runner)
        collection = asyncio.create_task(ProbeCollector(trainer.cfg)._generate(trainer))
        async with asyncio.timeout(5):
            first = await runner.started.get()
            other = await runner.started.get()
            if cause == "provider_failure":
                runner.pending[first].set_exception(ValueError("rollout failed"))
                with pytest.raises(ExceptionGroup) as error:
                    await collection
                assert isinstance(error.value.exceptions[0], ValueError)
            else:
                collection.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await collection

        assert runner.pending[other].cancelled()
        assert all(completion.done() for completion in runner.pending.values())
        assert not runner.active
        assert runner.sessions == ["start", "stop"]

    asyncio.run(scenario())
