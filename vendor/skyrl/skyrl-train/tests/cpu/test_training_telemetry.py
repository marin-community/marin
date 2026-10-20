"""Telemetry the rollout-buffer training loop delivers to Finelog over two training steps."""

import asyncio
import collections
import functools
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from skyrl_train import trainer as trainer_module
from skyrl_train.distributed.dispatch import ActorInfo, MeshRank
from skyrl_train.rollouts.context import TrainingContext
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingOutputBatch
from skyrl_train.timing_observability import STEP_WALL_PHASES
from tests.cpu.trajectory_runners.fixture_runner import FixtureRunner
from tests.cpu.util import example_dummy_config
from tests.cpu.task_specs import task_input, text_task, lowered_task

# Record names and attribute values the async RL dashboard reads (marin
# infra/grafana/src/async_rl_observability.py).
DASHBOARD_EVENTS = {
    "lifecycle",
    "async_phase_window",
    "rollout_call",
    "consumed_staleness",
    "weight_sync_completed",
}
DASHBOARD_METRICS = {
    "policy_step",
    "work_completed",
    "phase_duration_seconds",
    "rollout_wait_seconds",
    "rollout_waits",
    "rollout_queue_depth",
    "rollout_capacity",
    "rollout_buffer_dwell_seconds",
    "rollout_staleness_steps",
    "rollout_groups",
    "rollout_group_tokens",
    "event_loop_lag_seconds",
    "training_metric_value",
}
PHASE_WINDOWS = {"rollout_wait", "training", "weight_sync"}
ROLLOUT_CALL_BODY = {"call_id", "started_unix_ms", "finished_unix_ms", "response_tokens"}
ROLLOUT_PHASES = {"rollout_call", "rollout_finalize", "rollout_retain", "rollout_call_residual"}
DISPATCH_WAITS = {"prompt", "slot", "enqueue"}
DISPOSITIONS = {"consumed", "fully_masked", "insufficient_reward_spread"}
WORK_KINDS = {"consumed_sample", "consumed_response_token", "consumed_loss_token"}
CONSUMED_STALENESS_BODY = {"staleness", "groups", "sequences", "response_tokens"}
PERFORMANCE_METRICS = {
    "async/performance/core_seconds",
    "async/performance/buffer_wait_fraction",
    "async/performance/training_fraction",
    "async/performance/weight_sync_fraction",
    "async/performance/loss_tokens_per_configured_policy_gpu_second",
}
MISMATCH_METRICS = {
    "policy/mismatch/pooled/log_ratio_abs_mean",
    "policy/mismatch/pooled/ess_fraction",
    "policy/mismatch/staleness0/log_ratio_abs_p999",
    "policy/mismatch/pooled/pos_first256/log_ratio_abs_mean",
}

# Rows the scripted runner turns into each disposition.
PROMPT_UIDS = ["masked", "uniform", "a", "b", "c", "d"]
RESPONSE = [5, 6, 7]


class PromptRows(torch.utils.data.Dataset):
    def __len__(self):
        return len(PROMPT_UIDS)

    def __getitem__(self, index):
        uid = self.uid(index)
        return {"input": task_input(lowered_task(text_task(uid, prompt=uid))), "uid": uid}

    def uid(self, index):
        return PROMPT_UIDS[index]

    def collate_fn(self, batch):
        return batch


class ScriptedRunner(FixtureRunner):
    """Return a two-sample group whose rewards and masks the prompt's uid selects."""

    def __init__(self):
        self.calls = collections.Counter()

    async def _run(self, input_batch):
        uid = input_batch.trajectory_ids[0].instance_id
        self.calls[uid] += 1
        await asyncio.sleep(0)
        output = {
            "prompt_token_ids": [[1, 2], [1, 2]],
            "response_ids": [RESPONSE, RESPONSE],
            "rewards": [0.5, 0.5] if uid == "uniform" else [0.0, 1.0],
            "loss_masks": [[0] * 3] * 2 if uid == "masked" else [[1] * 3] * 2,
            "stop_reasons": ["stop", "length"],
            "rollout_metrics": {},
            "rollout_logprobs": [
                np.asarray([-1.0, -0.5, -0.25], dtype=np.float32),
                np.asarray([-1.0, -0.5, -0.25], dtype=np.float32),
            ],
        }
        return output


class FakePolicyGroup:
    actor_infos = [ActorInfo(handle=None, rank=MeshRank(dp=0, sp=0, tp=0, pp=0, world_size=1, dp_size=1, pp_size=1))]

    def async_run_ray_method(self, dispatch, method, *args, data=None, **kwargs):
        if method != "forward":
            return []
        logprobs = torch.full((data["sequences"].shape[0], data.metadata["response_length"]), -0.75)
        return [TrainingOutputBatch({"output": logprobs})]


class FakeEngines:
    def __init__(self):
        self.generation_paused_event = Event()

    async def pause_generation(self):
        self.generation_paused_event.set()

    async def resume_generation(self, policy_step):
        self.generation_paused_event.clear()


class FakeTracker:
    def __init__(self):
        self.logs = []

    def log(self, metrics, step, commit=True):
        self.logs.append((dict(metrics), step))


def _config(max_staleness_steps: int):
    cfg = example_dummy_config()
    OmegaConf.update(
        cfg,
        "trainer",
        {
            "train_batch_size": 2,
            "policy_mini_batch_size": 2,
            "max_steps": 2,
            "ckpt_interval": -1,
            "hf_save_interval": -1,
            "eval_interval": -1,
            "placement": {"colocate_all": False},
            "rollout_buffer": {"max_staleness_steps": max_staleness_steps, "max_in_flight": None},
            "algorithm": {
                "use_kl_loss": False,
                "off_policy_correction": "none",
                "dynamic_sampling": {"type": "filter"},
            },
        },
    )
    # Retention off gives the trainer its in-process disabled sink instead of a Ray actor.
    OmegaConf.update(cfg, "generator", {"n_samples_per_prompt": 2, "trajectory_retention": {"enabled": False}})
    OmegaConf.update(cfg, "data", {"shuffle": False})
    return cfg


async def _train_two_steps(monkeypatch, max_staleness_steps: int) -> RayPPOTrainer:
    cfg = _config(max_staleness_steps)
    runner = ScriptedRunner()
    dataset = PromptRows()
    trainer = RayPPOTrainer(
        cfg=cfg,
        tracker=FakeTracker(),
        tokenizer=None,
        train_dataset=dataset,
        eval_dataset=None,
        inference_engine_client=FakeEngines(),
        trajectory_runner=runner,
        context=TrainingContext.from_config(cfg, dataset, runner),
        models=(FakePolicyGroup(), None, None),
    )
    trainer.tokenizer = SimpleNamespace(decode=str, pad_token_id=0)

    async def no_op(*args, **kwargs):
        pass

    for name in (
        "_startup_trajectory_runner",
        "sync_policy_weights_to_inference_engines",
        "_drain_policy_event_loops",
        "_finalize_training",
        "shutdown",
    ):
        monkeypatch.setattr(trainer, name, no_op)
    monkeypatch.setattr(trainer, "init_weight_sync_state", lambda: None)
    monkeypatch.setattr(trainer, "train_critic_and_policy", lambda data: {"policy_loss": 0.0})
    # The fake policy returns its outputs directly instead of Ray object refs.
    monkeypatch.setattr(trainer_module, "ray", SimpleNamespace(get=lambda refs: refs))
    monkeypatch.setattr(
        trainer_module,
        "monitor_event_loop_lag",
        functools.partial(trainer_module.monitor_event_loop_lag, interval=0.001),
    )
    try:
        await trainer.train()
    finally:
        await trainer.context.close()
    return trainer


@pytest.mark.asyncio
async def test_two_steps_deliver_every_record_the_dashboard_reads(ray_module, delivered_telemetry, monkeypatch):
    # Async (staleness 1) exercises every record the sync configuration emits; tiny_training covers sync.
    trainer = await _train_two_steps(monkeypatch, max_staleness_steps=1)
    rows = delivered_telemetry.flush()

    # The launch environment names the training type, and every record carries it.
    assert {row["resource"]["training_type"] for row in rows} == {"async"}
    assert DASHBOARD_EVENTS | DASHBOARD_METRICS <= {row["name"] for row in rows}
    windows = delivered_telemetry.select("async_phase_window", outcome="success")
    assert {row["attributes"]["phase"] for row in windows} == PHASE_WINDOWS
    assert all(row["body"]["finished_unix_ms"] >= row["body"]["started_unix_ms"] for row in windows)
    calls = delivered_telemetry.select("rollout_call", outcome="success")
    assert all(set(row["body"]) == ROLLOUT_CALL_BODY for row in calls)
    assert {row["body"]["response_tokens"] for row in calls} == {2 * len(RESPONSE)}
    phases = {
        row["attributes"]["phase"] for row in delivered_telemetry.select("phase_duration_seconds", root="rollout_call")
    }
    assert phases == ROLLOUT_PHASES
    assert DISPATCH_WAITS <= {row["attributes"]["wait"] for row in delivered_telemetry.select("rollout_waits")}
    assert DISPOSITIONS <= {row["attributes"]["disposition"] for row in delivered_telemetry.select("rollout_groups")}
    assert delivered_telemetry.values("rollout_groups", disposition="consumed") == [1.0] * 4
    assert len(delivered_telemetry.values("rollout_buffer_dwell_seconds", disposition="consumed")) == 4
    consumed = delivered_telemetry.select("consumed_staleness")
    assert all(set(row["body"]) == CONSUMED_STALENESS_BODY for row in consumed)
    assert sum(row["body"]["groups"] for row in consumed) == 4
    assert WORK_KINDS <= {row["attributes"]["work_kind"] for row in delivered_telemetry.select("work_completed")}
    assert delivered_telemetry.values("policy_step") == [1.0, 2.0]
    assert [row["body"]["model_version_step"] for row in delivered_telemetry.select("weight_sync_completed")] == [
        0,
        1,
        2,
    ]
    assert delivered_telemetry.select("event_loop_lag_seconds")
    metrics = {row["attributes"]["metric"] for row in delivered_telemetry.select("training_metric_value")}
    assert PERFORMANCE_METRICS | MISMATCH_METRICS | {"consumed/length_stop_fraction"} <= metrics
    step_logs = [(payload, step) for payload, step in trainer.tracker.logs if "timing/step" in payload]
    assert [step for _, step in step_logs] == [1, 2]
    for payload, _ in step_logs:
        assert sum(payload[f"timing/step_wall/{phase}"] for phase in STEP_WALL_PHASES) == pytest.approx(
            payload["timing/step"], abs=0.01
        )
        assert all(f"timing/step_wall_overrun/{phase}" in payload for phase in STEP_WALL_PHASES)
