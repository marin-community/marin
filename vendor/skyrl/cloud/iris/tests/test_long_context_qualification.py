"""Qualification uses real unpadded lengths, action masks and finite learner evidence."""

import pytest
import torch
from unittest.mock import AsyncMock

from skyrl_train.callbacks.base import CallbackHandler, TrainerControl
from skyrl_train.callbacks.builtin import HFModelSaveCallback
from skyrl_train.config.utils import get_default_config

from skyrl_train.long_context_qualification import (
    LongContextQualification,
    QualificationStatus,
    qualify_learner_batch,
    qualify_optimizer_status,
)
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch


def batch():
    return {'attention_mask': torch.tensor([[0, 0, 1, 1, 1, 1]]),
            'response_mask': torch.tensor([[1, 1, 1]]), 'loss_mask': torch.tensor([[1, 0, 1]]),
            'rollout_logprobs': torch.tensor([[-1., 0., -2.]])}


def test_padding_is_not_natural_context_and_bad_action_evidence_fails():
    evidence = qualify_learner_batch(batch(), minimum_tokens=4, maximum_tokens=8)
    assert evidence['longest_unpadded_tokens'] == 4
    assert not qualify_learner_batch(batch(), minimum_tokens=5, maximum_tokens=8)['qualifying_length']
    broken = batch()
    broken['rollout_logprobs'][0, 0] = torch.nan
    with pytest.raises(ValueError, match='finite behavior'):
        qualify_learner_batch(broken, minimum_tokens=4, maximum_tokens=8)
    broken = batch()
    broken['attention_mask'] = torch.tensor([[1, 0, 1, 1, 1, 1]])
    with pytest.raises(ValueError, match='position gaps'):
        qualify_learner_batch(broken, minimum_tokens=4, maximum_tokens=8)
    qualify_optimizer_status({'policy_update_steps': 1, 'raw_grad_norm': 0.1})
    with pytest.raises(ValueError, match='optimizer update'):
        qualify_optimizer_status({'policy_update_steps': 0})
    with pytest.raises(ValueError, match='Nonfinite'):
        qualify_optimizer_status({'policy_update_steps': 1, 'raw_grad_norm': float('inf')})


def test_short_first_batch_does_not_prevent_a_later_qualifying_update():
    qualification = LongContextQualification(minimum_tokens=5, maximum_tokens=8, maximum_steps=5)
    assert qualification.observe(batch(), step=1) is None
    assert qualification.status is QualificationStatus.PENDING
    later = batch()
    later['attention_mask'] = torch.ones((1, 6), dtype=torch.long)
    evidence = qualification.observe(later, step=2)
    assert evidence['longest_unpadded_tokens'] == 6
    assert evidence['step'] == 2
    # Only successful optimizer/save/restore completion qualifies this update.
    assert qualification.status is QualificationStatus.PENDING
    qualification.qualified_step = 2
    assert qualification.status is QualificationStatus.PASSED


def test_short_batches_exhaust_the_budget_without_a_pre_forward_exception():
    qualification = LongContextQualification(minimum_tokens=5, maximum_tokens=8, maximum_steps=5)
    for step in range(1, 6):
        assert qualification.observe(batch(), step=step) is None
    assert qualification.status is QualificationStatus.EXHAUSTED
    assert qualification.observed_steps == 5
    assert qualification.longest_observed_tokens == 4
    assert qualification.qualified_step is None


@pytest.mark.asyncio
async def test_trainer_updates_short_batch_then_qualifies_later_long_batch(tmp_path):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = get_default_config()
    trainer.cfg.trainer.export_path = str(tmp_path)
    trainer.cfg.trainer.dump_training_order = False
    trainer.cfg.trainer.dump_data_batch = True
    trainer._long_context_qualification = LongContextQualification(5, 8, 5)
    trainer.distillation_plan = None
    trainer.all_timings = {}
    trainer.all_metrics = {}
    trainer._drain_policy_event_loops = AsyncMock()
    trainer.fwd_logprobs_values_reward = lambda data: data
    trainer.compute_advantages_and_returns = lambda data: data
    trainer.finalize_advantages_for_training = lambda data: data
    trainer.dump_data = lambda data, file_name: None
    updates = []

    def optimizer(data):
        updates.append(trainer.global_step)
        return {"policy_update_steps": 1, "raw_grad_norm": 0.1}

    trainer.train_critic_and_policy = optimizer
    trainer._qualify_checkpoint_restore = AsyncMock()
    for step in (1, 2):
        trainer.global_step = step
        data = batch()
        if step == 2:
            data['attention_mask'] = torch.ones((1, 6), dtype=torch.long)
        result = await trainer._run_training(TrainingInputBatch(data))
        assert result['policy_update_steps'] == 1
    assert updates == [1, 2]
    trainer._qualify_checkpoint_restore.assert_awaited_once()
    assert trainer._qualify_checkpoint_restore.call_args.args[1]['step'] == 2
    assert trainer._long_context_qualification.status is QualificationStatus.PASSED


@pytest.mark.asyncio
async def test_unqualified_completion_preserves_checkpoint_and_blocks_export(tmp_path):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = get_default_config()
    trainer.cfg.trainer.export_path = str(tmp_path)
    trainer._long_context_qualification = LongContextQualification(5, 8, 5)
    for step in range(1, 6):
        trainer._long_context_qualification.observe(batch(), step=step)
    trainer.global_step = 5
    trainer.total_training_steps = 5
    trainer.num_steps_per_epoch = 5
    trainer.all_timings = {}
    trainer.all_metrics = {}
    trainer.eval_dataset = None
    trainer.colocate_all = False
    trainer._last_saved_step = 5
    trainer._control = TrainerControl()
    trainer.callback_handler = CallbackHandler([HFModelSaveCallback(save_steps=-1)])
    trainer._drain_checkpoint_upload = AsyncMock()
    trainer.handle_hf_export = lambda: pytest.fail("An unqualified run must not export a model")
    checkpoint = tmp_path / 'global_step_5'
    checkpoint.mkdir()
    (checkpoint / 'trainer_state.pt').write_bytes(b'committed step 5')
    with pytest.raises(ValueError, match='5 completed updates; native checkpoints retained'):
        await trainer._finalize_training(completed_step=5, epoch=0)
    assert (checkpoint / 'trainer_state.pt').read_bytes() == b'committed step 5'
    assert (tmp_path / 'long-context-qualification-progress.json').exists()
