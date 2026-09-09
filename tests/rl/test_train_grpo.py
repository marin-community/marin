# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
from functools import partial

import equinox as eqx
import haliax as hax
import jax
import jmp
import numpy as np
import optax
import pytest
from levanter.checkpoint import CheckpointerConfig, save_checkpoint
from levanter.distributed import DistributedConfig
from levanter.grpo import GrpoConfig, KlGradient
from levanter.grpo_model import grpo_model_loss
from levanter.models.llama import LlamaConfig
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.tracker.tracker import NoopConfig
from levanter.trainer import Trainer, TrainerConfig
from marin.rl.grpo_artifact import GoldenRollout, write_golden_rollout
from marin.rl.train_grpo import OfflineGrpoConfig, main, prepare_grpo_example, score_grpo_batch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig as HfLlamaConfig
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast


def _rollout():
    mask = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=np.int32)
    return GoldenRollout(
        sequences=np.array([[0, 2, 3, 4, 0], [2, 3, 5, 0, 0], [0, 4, 6, 7, 8], [3, 4, 9, 0, 0]]),
        attention_mask=np.array([[0, 1, 1, 1, 0], [1, 1, 1, 0, 0], [0, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
        response_mask=mask,
        loss_mask=mask,
        rewards=np.array([[0, 1, 0], [0, 0, 0], [0, 0, 2], [-1, 0, 0]], dtype=np.float32),
        advantages=np.zeros((4, 3), dtype=np.float32),
        old_logprobs=np.full((4, 3), -9.0, dtype=np.float32),
        behavior_logprobs=np.full((4, 3), -10.0, dtype=np.float32),
        group_ids=np.array([0, 0, 1, 1]),
        objective_partition_ids=np.array([0, 1, 0, 1]),
    )


def _model():
    return LlamaConfig(
        max_seq_len=5,
        hidden_dim=8,
        intermediate_dim=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        gradient_checkpointing=False,
    ).build(hax.Axis("vocab", 16), key=jax.random.PRNGKey(7))


def _trainer(tmp_path, microbatch, checkpoint=None):
    config = TrainerConfig(
        id="offline-grpo-test",
        train_batch_size=4,
        per_device_parallelism=microbatch,
        num_train_steps=2,
        tracker=NoopConfig(),
        require_accelerator=False,
        mp=jmp.get_policy("f32"),
        log_dir=str(tmp_path),
        log_jaxprs=False,
        log_xla_hlo=False,
        load_checkpoint=checkpoint is not None,
        load_checkpoint_path=checkpoint,
    )
    loss = partial(
        grpo_model_loss,
        config=GrpoConfig(0.2, 0.2, 0.0, KlGradient.DETACHED),
        accumulation_steps=4 // microbatch,
        block_size=8,
    )
    return Trainer(config, optax.adam(1e-3), loss, add_default_hooks=False)


def _arrays(tree):
    return [np.asarray(x).copy() for x in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


@pytest.mark.timeout(180)
def test_offline_grpo_optimizer_update_preserves_objective_across_microbatches(tmp_path):
    batch = prepare_grpo_example(_rollout(), normalize_by_std=True)
    initial = _arrays(_model())
    outcomes = []
    for microbatch in (4, 2):
        with _trainer(tmp_path, microbatch) as trainer:
            state = trainer.initial_state(jax.random.PRNGKey(17), model=_model())
            scored = score_grpo_batch(trainer, state.model, batch, block_size=8)
            assert not np.allclose(scored.old_logprobs.array, _rollout().old_logprobs)
            result = trainer.train_step(state, scored)
            outcomes.append((float(result.loss), _arrays(result.state.model), _arrays(result.state.opt_state)))
    assert any(not np.array_equal(a, b) for a, b in zip(initial, outcomes[0][1], strict=True))
    np.testing.assert_allclose(outcomes[0][0], outcomes[1][0], atol=1e-6, rtol=1e-5)
    for full, micro in zip(outcomes[0][1] + outcomes[0][2], outcomes[1][1] + outcomes[1][2], strict=True):
        np.testing.assert_allclose(full, micro, atol=1e-6, rtol=1e-5)


@pytest.mark.timeout(180)
def test_offline_grpo_checkpoint_resume_reproduces_second_optimizer_step(tmp_path):
    batch = prepare_grpo_example(_rollout(), normalize_by_std=True)
    checkpoint = str(tmp_path / "step-1")
    with _trainer(tmp_path, 2) as trainer:
        state = trainer.initial_state(jax.random.PRNGKey(17), model=_model())
        scored = score_grpo_batch(trainer, state.model, batch, block_size=8)
        state = trainer.train_step(state, scored).state
        first_step = _arrays((state.model, state.opt_state, state.training_key, state.step))
        save_checkpoint(state, int(state.step), checkpoint, is_temporary=False)
        scored = score_grpo_batch(trainer, state.model, batch, block_size=8)
        uninterrupted = trainer.train_step(state, scored)
        expected = _arrays(
            (
                uninterrupted.state.model,
                uninterrupted.state.opt_state,
                uninterrupted.state.training_key,
                uninterrupted.state.step,
            )
        )
    with _trainer(tmp_path, 2, checkpoint) as trainer:
        # A different seed makes failure to restore the training RNG observable.
        restored = trainer.initial_state(jax.random.PRNGKey(999), model=_model())
        for saved, loaded in zip(
            first_step, _arrays((restored.model, restored.opt_state, restored.training_key, restored.step)), strict=True
        ):
            np.testing.assert_array_equal(saved, loaded)
        scored = score_grpo_batch(trainer, restored.model, batch, block_size=8)
        resumed = trainer.train_step(restored, scored)
        actual = _arrays((resumed.state.model, resumed.state.opt_state, resumed.state.training_key, resumed.state.step))
        assert int(resumed.state.step) == 2
        for direct, loaded in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(direct, loaded)
        np.testing.assert_array_equal(uninterrupted.loss, resumed.loss)


@pytest.mark.timeout(180)
def test_offline_grpo_main_rejects_rewritten_capture_on_continuation(tmp_path):
    model_path = str(tmp_path / "initial")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"token{i}": i for i in range(16)}, unk_token="token0")),
        unk_token="token0",
        pad_token="token0",
    )
    tokenizer.save_pretrained(model_path)
    model = LlamaForCausalLM(
        HfLlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=5,
        )
    )
    model.save_pretrained(model_path)
    captures = [str(tmp_path / f"capture{i}.npz") for i in range(2)]
    for uri in captures:
        write_golden_rollout(uri, _rollout(), {"tokenizer": model_path})
    config = OfflineGrpoConfig(
        captures=captures,
        initial_model=model_path,
        tokenizer=model_path,
        hf_save_path=str(tmp_path / "export"),
        stop_after=1,
        model=LlamaConfig(),
        trainer=TrainerConfig(
            id="capture-contract",
            train_batch_size=4,
            per_device_parallelism=2,
            num_train_steps=2,
            tracker=JsonFileTrackerConfig(output_path=str(tmp_path / "metrics")),
            require_accelerator=False,
            distributed=DistributedConfig(initialize_jax_distributed=False),
            mp=jmp.get_policy("f32"),
            log_dir=str(tmp_path / "logs"),
            log_jaxprs=False,
            log_xla_hlo=False,
            checkpointer=CheckpointerConfig(base_path=str(tmp_path / "checkpoints")),
        ),
        vocab_block_size=8,
    )
    main(config)
    with (tmp_path / "metrics" / "eval_results.json").open() as source:
        metrics = json.load(source)
    assert metrics["train/ppo_ratio_exact_unit_fraction"] == 1.0
    assert np.isfinite(metrics["train/log_ratio_abs_max"])
    changed = dataclasses.replace(_rollout(), rewards=_rollout().rewards + 0.25 * _rollout().response_mask)
    write_golden_rollout(captures[1], changed, {"tokenizer": model_path})
    with pytest.raises(ValueError, match="same capture stream"):
        main(dataclasses.replace(config, stop_after=None))
