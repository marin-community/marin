# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover rollout spend and close sampling feedback when the policy skips training."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from datasets import Dataset
from rigging.telemetry.serialization import EventBody, event_fields
from rolloutengine.models import CUMULATIVE_GROSS_ROLLOUT_TOKENS_METRIC, GROSS_ROLLOUT_TOKENS_METRIC
from skyrlrecipe.declared import PromptAcquisition

from skyrl_train.callbacks.base import TrainerControl
from skyrl_train.callbacks.builtin import DefaultCallbackHandler
from skyrl_train.config.rollout_validation import validate_rollout_launch
from skyrl_train.config.utils import get_default_config
from skyrl_train.contextual_sampling import PromptFeedback, PromptThompsonOrder, PublicPrompt
from skyrl_train.dataset.tasks import TASK_EVALUATION_IDENTITY
from skyrl_train.probe_mix import INITIAL_PROBABILITIES, DevelopmentProbeMix, panel_identity
from skyrl_train.rollouts.buffer import AdmittedRollout, GroupDisposition, SamplingGroupResult
from skyrl_train.rollouts.context import RolloutBatchMetadata, TrainingContext, TrainingContextState
from skyrl_train.rollouts.loader import PromptLoaderState
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils.trainer_utils import ResumeMode


class CheckpointPolicy:
    def async_run_ray_method(self, _dispatch, operation, *, ckpt_dir, **kwargs):
        path = Path(ckpt_dir) / "rank_0.pt"
        if operation == "save_checkpoint":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"policy")
        elif operation == "load_checkpoint":
            assert path.read_bytes() == b"policy"
        else:
            raise AssertionError(operation)
        return []


@pytest.fixture
def budget_trainer(tmp_path, monkeypatch):
    monkeypatch.setattr("skyrl_train.trainer.ray.get", lambda refs: refs)

    def make(*, budget=100, milestones=(), mode=ResumeMode.LATEST, resume_path=None, reset_step=False, eval_interval=-1):
        cfg = get_default_config()
        cfg.trainer.ckpt_path = str(tmp_path)
        cfg.trainer.ckpt_interval = 1
        cfg.trainer.max_ckpts_to_keep = -1
        cfg.trainer.rollout_buffer.max_staleness_steps = 0
        cfg.trainer.rollout_token_budget = budget
        cfg.trainer.rollout_token_eval_milestones = list(milestones)
        cfg.trainer.eval_interval = eval_interval
        cfg.trainer.offload_optimizer_during_rollouts = False
        cfg.trainer.resume_mode = mode.value
        cfg.trainer.resume_path = resume_path
        cfg.trainer.reset_global_step_on_resume = reset_step
        cfg.generator.inference_stats_interval = 0
        cfg.data.val_data = ["holdout"]
        validate_rollout_launch(cfg)

        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.cfg = cfg
        trainer.policy_model = CheckpointPolicy()
        trainer.tokenizer = None
        rollout = TrainingContextState(PromptLoaderState({}, []), [], None)
        trainer.context = SimpleNamespace(
            state_dict=AsyncMock(return_value=rollout),
            load_state_dict=AsyncMock(),
            start=Mock(side_effect=AssertionError("a completed budget must not dispatch rollouts")),
        )
        trainer.all_metrics = {}
        trainer.all_timings = {}
        trainer.all_startup_timings = {}
        trainer._ftpo_stopped = False
        trainer._domain_balancer = None
        trainer.distillation_scored_tokens_total = 0
        trainer.rollout_gross_response_tokens_total = 0
        trainer._last_saved_step = None
        trainer._last_evaluated_step = None
        trainer._restored_rollout_state = None
        trainer._resume_checkpoint = None
        trainer.resume_mode = mode
        trainer.colocate_all = False
        trainer.global_step = 0
        trainer.total_training_steps = trainer.num_steps_per_epoch = 4
        trainer.eval_dataset = ["holdout"]
        trainer.callback_handler = DefaultCallbackHandler(cfg)
        trainer._control = TrainerControl()
        trainer.tracker = Mock()
        trainer.speculative_decoding = None
        trainer._init_weight_sync = AsyncMock()
        trainer._sync_policy_for_rollouts = AsyncMock()
        trainer._evaluate_and_log = AsyncMock()
        trainer._training_metrics_enabled = False
        return trainer

    return make


@pytest.mark.asyncio
async def test_budget_retry_counts_each_checkpointed_update_and_runs_terminal_evaluation(budget_trainer):
    first = budget_trainer()
    first.global_step = 1
    first.all_metrics[GROSS_ROLLOUT_TOKENS_METRIC] = 60
    await first._run_step_end_callbacks(first._create_trainer_state(epoch=0))
    await first._drain_checkpoint_upload()
    first._evaluate_and_log.assert_not_awaited()
    with pytest.raises(ValueError, match="same budget"):
        budget_trainer(budget=105).load_checkpoints()

    second = budget_trainer(reset_step=True, eval_interval=2)
    second.global_step, _ = second.load_checkpoints()
    assert second.global_step == 1
    second.global_step = 2
    second.all_metrics[GROSS_ROLLOUT_TOKENS_METRIC] = 50
    await second._run_step_end_callbacks(second._create_trainer_state(epoch=0))
    assert second.all_metrics[CUMULATIVE_GROSS_ROLLOUT_TOKENS_METRIC] == 110
    second._evaluate_and_log.assert_not_awaited()
    second._evaluate_and_log.side_effect = OSError("evaluation interrupted")
    with pytest.raises(OSError, match="evaluation interrupted"):
        await second._finalize_training(completed_step=2, epoch=0)

    retry = budget_trainer()
    await retry._train_loop()

    assert retry.global_step == 2
    assert retry.rollout_gross_response_tokens_total == 110
    retry.context.load_state_dict.assert_awaited_once()
    retry.context.start.assert_not_called()
    retry._evaluate_and_log.assert_awaited_once()
    assert retry._evaluate_and_log.await_args.args[0].global_step == 2
    assert retry._evaluate_and_log.await_args.args[0].is_last_step


@pytest.mark.asyncio
async def test_budget_recovers_an_interrupted_milestone_evaluation(budget_trainer):
    first = budget_trainer(milestones=(40, 50))
    first.global_step = 1
    first.all_metrics[GROSS_ROLLOUT_TOKENS_METRIC] = 65
    first._evaluate_and_log.side_effect = OSError("evaluation interrupted")
    with pytest.raises(OSError, match="evaluation interrupted"):
        await first._run_step_end_callbacks(first._create_trainer_state(epoch=0))
    # Both milestones crossed in one step share a single snapshot.
    first._evaluate_and_log.assert_awaited_once()
    await first._drain_checkpoint_upload()

    async def evaluate_with_timer(_state, timer, **_kwargs):
        with timer:
            pass

    retry = budget_trainer(milestones=(40, 50))
    retry._evaluate_and_log.side_effect = evaluate_with_timer
    retry.context.start.side_effect = RuntimeError("reached new rollout work")
    with pytest.raises(RuntimeError, match="reached new rollout work"):
        await retry._train_loop()
    retry._evaluate_and_log.assert_awaited_once()
    assert retry._evaluate_and_log.await_args.args[0].global_step == 1


@pytest.mark.asyncio
async def test_budget_requires_compatible_history_or_a_new_stage(budget_trainer):
    older = budget_trainer(budget=None)
    older.global_step = 1
    await older.save_checkpoints()
    assert budget_trainer(budget=None).load_checkpoints()[0] == 1
    with pytest.raises(ValueError, match="same budget.*valid saved gross"):
        budget_trainer().load_checkpoints()

    checkpoint = Path(older.cfg.trainer.ckpt_path) / "global_step_1"
    new_stage = budget_trainer(mode=ResumeMode.FROM_PATH, resume_path=str(checkpoint), reset_step=True)
    assert new_stage.load_checkpoints()[0] == 0
    assert new_stage.rollout_gross_response_tokens_total == 0

    old_zero = budget_trainer(budget=None)
    await old_zero.save_checkpoints()
    checkpoint = Path(old_zero.cfg.trainer.ckpt_path) / "global_step_0"
    assert budget_trainer(mode=ResumeMode.FROM_PATH, resume_path=str(checkpoint)).load_checkpoints()[0] == 0


@pytest.mark.asyncio
async def test_empty_ftpo_batch_closes_paid_sampling_feedback(monkeypatch):
    order = PromptThompsonOrder(
        [PublicPrompt("prompt", "math", "What is twelve plus eighteen?")],
        set(),
        {},
        acquisition=PromptAcquisition.COST_AWARE,
        seed=1,
        window_size=1,
        decay=0.95,
        epsilon=0.05,
        reference_tokens=100,
    )
    order.assign("prompt")
    context = TrainingContext.__new__(TrainingContext)
    context.loader = SimpleNamespace(order=order)
    events = []
    monkeypatch.setattr("skyrl_train.rollouts.context.record_event", lambda name, body: events.append((name, body)))
    group = AdmittedRollout(0, "prompt", 1, 2, 100)
    result = SamplingGroupResult("lease", 1, 1, 1, 2, (0.0, 1.0))
    disposition = GroupDisposition("prompt", "consumed", 100, None, result)
    metadata = RolloutBatchMetadata(1, (group,), {}, True, (disposition,))
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.all_timings = {}
    trainer.all_metrics = {}
    trainer.context = context
    batch = TrainingInputBatch(
        loss_mask=torch.zeros((2, 1), dtype=torch.bool),
        ftpo_weights=torch.zeros((2, 1)),
        ftpo_chosen_mask=torch.zeros((2, 1, 1), dtype=torch.bool),
    )

    assert await trainer._run_training(batch, sampling_batch=metadata) == {"policy_update_steps": 0.0}
    assert trainer.all_metrics["sampling/groups"] == 1
    assert trainer.all_metrics["sampling/generated_tokens"] == 100
    assert trainer.all_metrics["sampling/optimizer_groups"] == trainer.all_metrics["sampling/useful_groups"] == 0
    assert [(name, data["feedback_batch"]) for name, data in events] == [("sampling_feedback_observed", 1)]
    with pytest.raises(ValueError, match="only once"):
        order.finish_step(2, [PromptFeedback("lease", "prompt", 1, 2, (0.0, 1.0), 100, None)])


def configure_development_probe(trainer):
    digests = tuple(f"{i:064x}" for i in range(4))
    reserved = tuple(f"{i:064x}" for i in range(4, 8))
    identity = panel_identity(digests)
    trainer.cfg.data.sampling.kind = PromptAcquisition.WITHIN_SOURCE_PASS_COST_AWARE.value
    trainer.cfg.data.sampling.epsilon = 1.0
    trainer.cfg.trainer.train_batch_size = 16
    trainer.cfg.generator.eval_n_samples_per_prompt = 1
    trainer.cfg.trainer.callbacks = [
        {"type": "checkpoint", "save_steps": 1},
        {
            "type": "evaluation",
            "eval_steps": 0,
            "eval_before_train": True,
            "additional_evaluations": {
                "confirmation": {"schedule": "final", "data_files": ["confirmation"], "n_samples_per_prompt": 1}
            },
        },
        {
            "type": "development_probe_mix", "expected_tasks": 4, "panel_sha256": identity, "policy": "adaptive",
            "confirmation_tasks": 4, "confirmation_sha256": panel_identity(reserved),
        },
    ]
    validate_rollout_launch(trainer.cfg)
    trainer.callback_handler = DefaultCallbackHandler(trainer.cfg)
    trainer.rollout_gross_response_tokens_last_update = 0

    def dataset(identities):
        return SimpleNamespace(
            dataframe=Dataset.from_dict({TASK_EVALUATION_IDENTITY: [json.dumps({"task_digest": d}) for d in identities]})
        )

    trainer.eval_dataset = dataset(digests)
    trainer.additional_eval_dataloaders = {"confirmation": SimpleNamespace(dataset=dataset(reserved))}
    prompts = [
        PublicPrompt(str(index), "easy" if index < 128 else "intermediate", f"Reverse the letters of word {index}.")
        for index in range(256)
    ]
    order = PromptThompsonOrder(
        prompts,
        set(),
        {},
        acquisition=PromptAcquisition.WITHIN_SOURCE_PASS_COST_AWARE,
        seed=2,
        window_size=16,
        decay=0.95,
        epsilon=1,
        reference_tokens=100,
        development_mix=DevelopmentProbeMix(seed=2, panel_sha256=identity, policy="adaptive"),
    )

    def load(state):
        order.load_state_dict(state.loader.order)

    trainer.context.loader = SimpleNamespace(order=order)
    trainer.context.state_dict = AsyncMock(side_effect=lambda: TrainingContextState(
        PromptLoaderState(order.state_dict(), []), [], None
    ))
    trainer.context.load_state_dict = AsyncMock(side_effect=load)
    return prompts


async def emit_probe(trainer, state, accuracy, *, scored=4):
    await trainer.callback_handler.call_event_async(
        "on_evaluate",
        state,
        TrainerControl(),
        trainer=trainer,
        metrics={
            "eval/all/num_attempted": 4,
            "eval/all/num_scored": scored,
            "eval/all/avg_verifier_score": accuracy,
            "eval/all/response_tokens": 100,
            "eval/confirmation/all/avg_verifier_score": 1 - accuracy,
        },
    )


@pytest.mark.asyncio
async def test_development_probe_replays_boundary_before_next_publish(budget_trainer, monkeypatch):
    total, milestones = 4_194_304, (1_048_576, 2_097_152, 3_145_728)
    events = []

    def capture_decision(name, body: EventBody, **_kwargs):
        if name == "development_probe_mix_decision":
            events.append((name, event_fields(body, budget=4096)))

    monkeypatch.setattr("skyrl_train.telemetry.telemetry.event", capture_decision)
    original = budget_trainer(budget=total, milestones=milestones)
    prompts = configure_development_probe(original)
    original_order = original.context.loader.order
    state = original._create_trainer_state(epoch=0)
    await original.callback_handler.call_event_async("on_train_begin", state, TrainerControl(), trainer=original)
    await emit_probe(original, state, 0.25)
    first_rows = [original_order.next_index() for _ in range(16)]
    assert sum(prompts[index].source == "easy" for index in first_rows) == 8

    async def interrupt_after_probe(state, _timer, **_kwargs):
        await emit_probe(original, state, 0.44)
        raise OSError("interrupted after development feedback")

    original.global_step = 1
    original.all_metrics[GROSS_ROLLOUT_TOKENS_METRIC] = 1_100_000
    original._evaluate_and_log.side_effect = interrupt_after_probe
    with pytest.raises(OSError, match="interrupted after development feedback"):
        await original._run_step_end_callbacks(original._create_trainer_state(epoch=0))
    await original._drain_checkpoint_upload()
    uninterrupted_rows = [original_order.next_index() for _ in range(16)]
    assert sum(prompts[index].source == "easy" for index in uninterrupted_rows) == 10
    assert not set(first_rows).intersection(uninterrupted_rows)
    assert json.loads(events[0][1]["action_probabilities"]) == list(INITIAL_PROBABILITIES)
    assert json.loads(events[1][1]["action_probabilities"]) == pytest.approx((0.214, 0.572, 0.214), abs=0.001)

    retry = budget_trainer(budget=total, milestones=milestones)
    configure_development_probe(retry)
    order = retry.context.loader.order

    async def rerun_probe(state, timer, **_kwargs):
        with timer:
            await emit_probe(retry, state, 0.44)
            await emit_probe(retry, state, 0.44)

    async def consume_next_window(step):
        assert step == 2
        assert [order.next_index() for _ in range(16)] == uninterrupted_rows
        raise RuntimeError("reached next rollout window")

    retry.context.start.side_effect = None
    retry._evaluate_and_log.side_effect = rerun_probe
    retry._publish = AsyncMock(side_effect=consume_next_window)
    with pytest.raises(RuntimeError, match="reached next rollout window"):
        await retry._train_loop()
    assert retry.rollout_gross_response_tokens_total == 1_100_000
    assert events[1] == events[2]
    assert len(events) == 3

    for step, spend, score in ((2, 1_070_000, 0.53), (3, 1_045_000, 0.62)):
        retry.global_step = step
        retry.rollout_gross_response_tokens_total += spend
        retry.rollout_gross_response_tokens_last_update = spend
        await emit_probe(retry, retry._create_trainer_state(epoch=0), score)
    decisions = [data for _, data in events]
    assert [d["boundary"] for d in decisions] == [0, 1, 1, 2, 3]
    assert [d["block_gross_training_tokens"] for d in decisions] == [0, 1_100_000, 1_100_000, 1_070_000, 1_045_000]
    assert decisions[-1]["transfer_gain_per_100k_train_tokens"] == pytest.approx(0.9 / 1.045)

    retry.global_step = 4
    retry.rollout_gross_response_tokens_total = total
    state = retry._create_trainer_state(epoch=0)
    await retry.callback_handler.call_event_async("on_train_end", state, TrainerControl(), trainer=retry)
    await emit_probe(retry, state, 0.0, scored=0)
    assert len(events) == 5


@pytest.mark.asyncio
async def test_development_probe_pins_both_panels_and_rejects_partial_feedback(budget_trainer):
    trainer = budget_trainer(budget=100, milestones=(25, 50, 75))
    configure_development_probe(trainer)
    valid = trainer.additional_eval_dataloaders["confirmation"].dataset
    trainer.additional_eval_dataloaders["confirmation"].dataset = trainer.eval_dataset
    state = trainer._create_trainer_state(epoch=0)
    with pytest.raises(ValueError, match="does not match its pinned task digests"):
        await trainer.callback_handler.call_event_async("on_train_begin", state, TrainerControl(), trainer=trainer)
    trainer.additional_eval_dataloaders["confirmation"].dataset = valid
    await trainer.callback_handler.call_event_async("on_train_begin", state, TrainerControl(), trainer=trainer)
    with pytest.raises(ValueError, match="complete verifier coverage"):
        await emit_probe(trainer, state, 0.25, scored=3)
    order = trainer.context.loader.order
    with pytest.raises(ValueError, match="baseline must complete"):
        order.next_index()
    await emit_probe(trainer, state, 0.25)
    assert 0 <= order.next_index() < 256


def test_mild_tilt_common_draw_and_randomized_shadow_across_different_update_steps():
    adaptive = DevelopmentProbeMix(seed=137, panel_sha256="test", policy="adaptive")
    randomized = DevelopmentProbeMix(seed=137, panel_sha256="test", policy="randomized")
    for boundary, (accuracy, gross) in enumerate(((0.25, 0), (0.36, 1_100_000), (0.28, 2_170_000))):
        args = dict(boundary=boundary, accuracy=accuracy, gross_tokens=gross, development_tokens=700)
        a = adaptive.observe(step=boundary * 17, **args)
        r = randomized.observe(step=boundary * 19, **args)
        assert a.uniform_draw == r.uniform_draw
        assert r.action_probabilities == INITIAL_PROBABILITIES
    assert a.action_probabilities == pytest.approx((0.443, 0.447, 0.111), abs=0.001)
    assert r.adaptive_action_probabilities == pytest.approx((0.438, 0.451, 0.111), abs=0.001)
    assert (a.easy_quota, r.easy_quota) == (8, 10)
