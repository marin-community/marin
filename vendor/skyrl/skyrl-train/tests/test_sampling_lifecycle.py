# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover rollout spend and close sampling feedback when the policy skips training."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from rolloutengine.models import CUMULATIVE_GROSS_ROLLOUT_TOKENS_METRIC, GROSS_ROLLOUT_TOKENS_METRIC
from skyrlrecipe.declared import PromptAcquisition

from skyrl_train.callbacks.base import TrainerControl
from skyrl_train.callbacks.builtin import DefaultCallbackHandler
from skyrl_train.config.rollout_validation import validate_rollout_launch
from skyrl_train.config.utils import get_default_config
from skyrl_train.contextual_sampling import PromptFeedback, PromptThompsonOrder, PublicPrompt
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
