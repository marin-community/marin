import asyncio
from threading import Event
from types import SimpleNamespace

import pytest

from skyrl_train.config.utils import get_default_config
from skyrl_train.rollouts.buffer import BatchPolicy, RolloutBufferConfig, RolloutGroup
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.trajectory_runners.base import TrajectoryID
from skyrl_train.trajectory_selection import BestOfNTrajectorySelector


def _group(uid: str, policy_step: int, *, rewards: list[float] | None = None) -> RolloutGroup:
    batch = {
        "prompt_token_ids": [[1], [1]],
        "response_ids": [[2], [3]],
        "rewards": rewards or [0.0, 1.0],
        "loss_masks": [[1], [1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_metrics": {},
        "rollout_logprobs": None,
        "trajectory_ids": [TrajectoryID(instance_id=uid, repetition_id=index) for index in range(2)],
    }
    return RolloutGroup(batch, uid, policy_step, {"uid": uid})


@pytest.mark.parametrize(
    ("reason", "paused_on_entry", "expected"),
    [
        ("initial", False, ["pause", "sync", ("resume", 5)]),
        ("initial", True, ["sync", ("resume", 5)]),
        ("training_step", False, ["pause", "sync", ("resume", 5)]),
        ("training_step", True, None),
    ],
)
@pytest.mark.parametrize("offload_enabled", [False, True])
def test_weight_sync_respects_pause_and_optimizer_offload_policy(reason, paused_on_entry, expected, offload_enabled):
    trainer = object.__new__(RayPPOTrainer)
    trainer.cfg = get_default_config()
    trainer.cfg.trainer.offload_optimizer_during_rollouts = offload_enabled
    trainer.colocate_all = False
    trainer.global_step = 4
    trainer.all_startup_timings = {}
    trainer.all_timings = {}
    events = []

    class Policy:
        optimizer_on_gpu = True

        def offload_to_cpu(self, *, offload_optimizer, offload_model):
            assert offload_optimizer and not offload_model
            self.optimizer_on_gpu = False
            events.append("offload")

    class Engine:
        generation_paused_event = Event()

        async def pause_generation(self):
            if self.generation_paused_event.is_set():
                raise RuntimeError("Generation is already paused")
            self.generation_paused_event.set()
            events.append("pause")

        async def resume_generation(self, policy_step):
            assert self.generation_paused_event.is_set()
            assert trainer.policy_model.optimizer_on_gpu != offload_enabled
            self.generation_paused_event.clear()
            events.append(("resume", policy_step))

    async def sync_weights():
        events.append("sync")

    trainer.policy_model = Policy()
    trainer.inference_engine_client = Engine()
    if paused_on_entry:
        trainer.inference_engine_client.generation_paused_event.set()
    trainer.sync_policy_weights_to_inference_engines = sync_weights

    if expected is None:
        with pytest.raises(RuntimeError, match="Generation is already paused"):
            asyncio.run(trainer._sync_policy_for_rollouts(reason=reason))
        assert events == []
        return

    asyncio.run(trainer._sync_policy_for_rollouts(reason=reason))

    assert trainer.policy_model.optimizer_on_gpu != offload_enabled
    expected_events = expected.copy()
    if offload_enabled:
        expected_events.insert(expected_events.index("sync"), "offload")
    assert events == expected_events
    assert not trainer.inference_engine_client.generation_paused_event.is_set()


def test_rollout_batch_conversion_reports_staleness_and_stage_timings(monkeypatch):
    trainer = object.__new__(RayPPOTrainer)
    trainer.context = SimpleNamespace(config=RolloutBufferConfig(2, None, 2, BatchPolicy.FULL_BATCH, None, None))
    trainer.cfg = SimpleNamespace(trainer=SimpleNamespace(algorithm=get_default_config().trainer.algorithm))
    trainer.cfg.trainer.algorithm.off_policy_correction = "none"
    trainer.global_step = 10
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer.tokenizer = SimpleNamespace(decode=lambda response: str(response))
    now = [0.0]
    monkeypatch.setattr("skyrl_train.utils.utils.time", SimpleNamespace(monotonic=lambda: now[0]))

    def postprocess(batch, uids):
        now[0] += 7.0
        return batch

    def select(batch, uids):
        now[0] += 11.0
        return batch, uids

    def convert(batch, uids, *, rollout_staleness):
        now[0] += 3.0
        return {"rewards": batch["rewards"], "uids": uids, "rollout_staleness": rollout_staleness}

    trainer.postprocess_trajectory_batch = postprocess
    trainer.select_trajectories = select
    trainer.convert_to_training_input = convert

    result = trainer.convert_rollout_groups_to_training_input([_group("fresh", 10), _group("stale", 8)])

    assert result == {
        "rewards": [0.0, 1.0, 0.0, 1.0],
        "uids": ["fresh", "fresh", "stale", "stale"],
        "rollout_staleness": [0, 0, 2, 2],
    }
    assert trainer.all_metrics["async/staleness_max"] == 2
    assert trainer.all_metrics["async/staleness_ratio"] == 0.5
    assert trainer.all_timings == {
        "assemble_generation_group_mini_batch": 0.0,
        "postprocess_trajectory_batch": 18.0,
        "convert_to_training_input": 3.0,
    }


def test_rollout_batch_conversion_records_domain_reward_metrics():
    trainer = object.__new__(RayPPOTrainer)
    trainer.context = SimpleNamespace(config=RolloutBufferConfig(3, None, 0, BatchPolicy.FULL_BATCH, None, None))
    trainer.cfg = get_default_config()
    trainer.cfg.trainer.algorithm.policy_loss_type = "regular"
    trainer.cfg.trainer.algorithm.off_policy_correction = "none"
    trainer.cfg.generator.n_samples_per_prompt = 2
    trainer.global_step = 0
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer.tokenizer = SimpleNamespace(decode=str)
    trainer.select_trajectories = lambda batch, uids: (batch, uids)
    trainer.convert_to_training_input = lambda batch, uids, *, rollout_staleness: batch
    math = _group("math", 0, rewards=[0.2, 0.6])
    tools = _group("tools", 0, rewards=[0.6, 1.0])
    missing = _group("missing", 0, rewards=[0.4, 0.6])
    math.trajectory_batch["data_sources"] = ["math", "math"]
    tools.trajectory_batch["data_sources"] = ["tools", "tools"]

    trainer.convert_rollout_groups_to_training_input([math, tools, missing])

    assert trainer.all_metrics["reward/domain/math/avg_raw_reward"] == pytest.approx(0.4)
    assert trainer.all_metrics["reward/domain/tools/avg_raw_reward"] == pytest.approx(0.8)
    assert trainer.all_metrics["reward/domain/_missing/avg_raw_reward"] == pytest.approx(0.5)


class _RecordingDistillationRuntime:
    def __init__(self):
        self.submitted = []

    async def submit_before_batch_assembly(self, trajectory_batch):
        self.submitted.append(trajectory_batch)
        return object()


def test_teacher_scores_only_the_rows_the_learner_selects():
    trainer = object.__new__(RayPPOTrainer)
    runtime = _RecordingDistillationRuntime()
    trainer._distillation_runtime = runtime
    trainer._distillation_tickets = {}
    trainer.trajectory_selector = BestOfNTrajectorySelector(2)

    asyncio.run(trainer._submit_admitted_groups_for_teacher_scoring([_group("best", 10, rewards=[0.25, 0.75])]))

    (submitted,) = runtime.submitted
    assert submitted["response_ids"] == [[3]]
    assert submitted["trajectory_ids"][0].repetition_id == 1
    assert set(trainer._distillation_tickets) == {"best"}
