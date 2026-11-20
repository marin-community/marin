# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import datetime
import json
import math
import os
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from transformers import AutoTokenizer
from chex import assert_trees_all_close

from haliax import Axis
from haliax.quantization import QuantizationConfig

import levanter.main.eval_lm as eval_lm
import levanter.main.train_lm as train_lm
from levanter.testing import tiny_corpus
from levanter.adaptor import LoraAdaptorConfig
from levanter.checkpoint import CheckpointerConfig, latest_checkpoint_path, load_checkpoint_metadata
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.layers.attention import AttentionBackend
from levanter.models.llama_native_lm import NativeLlamaLmConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.trainer_state import trainables_only
from levanter.testing.helpers import arrays_only
from levanter.utils.mesh import MeshConfig


def _array_leaves(tree):
    return jax.tree_util.tree_leaves(arrays_only(tree))


def _assert_training_recorded(output_path: str) -> dict:
    """Load the JsonFileTracker record and assert training produced finite metrics.

    The smoke tests run ``train_lm.main`` end to end; the only stable observable
    effect is the metrics persisted by the tracker on ``finish()``. Asserting on
    them catches silent no-ops (no step ever logged) and NaN/inf loss blowups.
    """
    with open(os.path.join(output_path, "eval_results.json")) as f:
        metrics = json.load(f)
    assert metrics["parameter_count"] > 0
    assert "train/loss" in metrics, "per-step logging hook never fired"
    assert math.isfinite(metrics["train/loss"])
    return metrics


@pytest.mark.tpu
@pytest.mark.timeout(240)
def test_train_lm(local_gpt2_tokenizer_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_corpus.construct_small_data_cache(tmpdir, tokenizer=local_gpt2_tokenizer_path)
        config = train_lm.TrainLmConfig(
            data=data_config,
            hf_save_path=os.path.join(tmpdir, "hf"),
            hf_save_steps=3,
            hf_save_chat_template="{% for m in messages %}{{ m['role'] }}:{{ m['content'] }};{% endfor %}",
            model=train_lm.LlamaConfig(
                reference_checkpoint=local_gpt2_tokenizer_path,
                num_layers=2,
                num_heads=2,
                num_kv_heads=2,
                max_seq_len=64,
                hidden_dim=32,
                attn_backend=None,  # use default for platform
            ),
            trainer=train_lm.TrainerConfig(
                id="template-export",
                num_train_steps=2,
                train_batch_size=len(jax.devices()),
                max_eval_batches=1,
                tracker=JsonFileTrackerConfig(output_path=tmpdir),
                checkpointer=CheckpointerConfig(base_path=os.path.join(tmpdir, "checkpoints")),
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
            ),
        )
        train_lm.main(config)
        _assert_training_recorded(tmpdir)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(tmpdir, "hf", "template-export", "step-1"))
        assert tokenizer.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=False) == "user:hello;"


def test_train_lm_fp8():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_corpus.construct_small_data_cache(tmpdir)
        config = train_lm.TrainLmConfig(
            data=data_config,
            model=train_lm.LlamaConfig(
                num_layers=2,
                num_heads=2,
                num_kv_heads=2,
                max_seq_len=64,
                hidden_dim=32,
                attn_backend=None,  # use default for platform
            ),
            trainer=train_lm.TrainerConfig(
                quantization=QuantizationConfig(fp8=True),
                num_train_steps=2,
                train_batch_size=len(jax.devices()),
                max_eval_batches=1,
                tracker=JsonFileTrackerConfig(output_path=tmpdir),
                checkpointer=CheckpointerConfig(base_path=os.path.join(tmpdir, "checkpoints")),
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
            ),
        )
        train_lm.main(config)
        _assert_training_recorded(tmpdir)


def test_train_lm_with_lora_adapter():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_corpus.construct_small_data_cache(tmpdir)
        config = train_lm.TrainLmConfig(
            data=data_config,
            model=train_lm.LlamaConfig(
                num_layers=2,
                num_heads=2,
                num_kv_heads=2,
                max_seq_len=64,
                hidden_dim=32,
                attn_backend=None,
            ),
            trainer=train_lm.TrainerConfig(
                num_train_steps=2,
                train_batch_size=len(jax.devices()),
                max_eval_batches=1,
                tracker=JsonFileTrackerConfig(output_path=tmpdir),
                checkpointer=CheckpointerConfig(base_path=os.path.join(tmpdir, "checkpoints")),
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
            ),
            adapter=LoraAdaptorConfig(r=4),
        )
        train_lm.main(config)
        _assert_training_recorded(tmpdir)


def test_restore_lm_model_from_partial_checkpoint_recovers_base_model():
    config = train_lm.LlamaConfig(
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=16,
        hidden_dim=16,
        attn_backend=None,
    )
    Vocab = Axis("vocab", 32)
    base_key, wrong_base_key, adapter_key, wrong_adapter_key = jrandom.split(jrandom.PRNGKey(0), 4)

    adapter = LoraAdaptorConfig(r=4, a_init_mode="random")
    trained_model = adapter.apply(config.build(Vocab, key=base_key), key=adapter_key)
    wrong_resume_skeleton = adapter.apply(config.build(Vocab, key=wrong_base_key), key=wrong_adapter_key)
    correct_source_skeleton = adapter.apply(config.build(Vocab, key=base_key), key=wrong_adapter_key)
    trainable_filter = adapter.trainable_filter(trained_model)

    checkpointed_trainables = trainables_only(trained_model, trainable_filter)
    wrong_resumed_model = eqx.combine(checkpointed_trainables, wrong_resume_skeleton)
    restored_model = train_lm._restore_lm_model_from_partial_checkpoint(
        wrong_resumed_model,
        correct_source_skeleton,
        trainable_filter,
    )

    assert_trees_all_close(_array_leaves(restored_model), _array_leaves(trained_model))


def test_train_lm_direct_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 128
        seq_len = 64
        data = []
        for i in range(8):
            tokens = jnp.full((seq_len,), i % vocab_size, dtype=jnp.int32)
            data.append(GrugLmExample.causal(tokens))
        dataset = ListAsyncDataset(data)

        component = DirectDatasetComponent(datasets={"train": dataset})
        data_config = LmDataConfig(components={"direct": component}, vocab_size=vocab_size, tokenizer="passthrough")

        config = train_lm.TrainLmConfig(
            data=data_config,
            model=train_lm.LlamaConfig(
                num_layers=2,
                num_heads=2,
                num_kv_heads=2,
                max_seq_len=seq_len,
                hidden_dim=32,
                attn_backend=None,
            ),
            trainer=train_lm.TrainerConfig(
                num_train_steps=2,
                train_batch_size=len(jax.devices()),
                max_eval_batches=1,
                tracker=JsonFileTrackerConfig(output_path=tmpdir),
                checkpointer=CheckpointerConfig(base_path=os.path.join(tmpdir, "checkpoints")),
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
            ),
        )
        train_lm.main(config)
        _assert_training_recorded(tmpdir)


@pytest.mark.multidevice
def test_train_lm_native_rounds_odd_vocab_for_tensor_parallelism_and_resumes(tmp_path):
    if len(jax.devices()) != 4:
        pytest.skip("Requires four devices for DP2/TP2 placement")

    vocab_size = 31
    seq_len = 16
    examples = [GrugLmExample.causal((jnp.arange(seq_len) + i) % vocab_size) for i in range(12)]
    component = DirectDatasetComponent(datasets={"train": ListAsyncDataset(examples)})
    config = train_lm.TrainLmConfig(
        data=LmDataConfig(components={"direct": component}, vocab_size=vocab_size, tokenizer="passthrough"),
        model=NativeLlamaLmConfig(
            hidden_dim=16,
            intermediate_dim=24,
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            max_seq_len=seq_len,
            attn_backend=AttentionBackend.VANILLA,
        ),
        trainer=train_lm.TrainerConfig(
            id="native-odd-vocab",
            num_train_steps=1,
            train_batch_size=4,
            mesh=MeshConfig(axes={"data": 2, "model": 2}),
            tracker=JsonFileTrackerConfig(output_path=str(tmp_path)),
            checkpointer=CheckpointerConfig(base_path=str(tmp_path / "checkpoints"), append_run_id_to_base_path=False),
            require_accelerator=False,
            distributed=DistributedConfig(initialize_jax_distributed=False),
            log_dir=tmp_path / "logs",
        ),
    )
    train_lm.main(config)
    first = latest_checkpoint_path(tmp_path / "checkpoints")
    metadata = load_checkpoint_metadata(first)
    assert config.model.checkpoint_vocab_size(metadata) == 32
    assert len(config.data.the_tokenizer) == vocab_size

    resumed = dataclasses.replace(
        config,
        trainer=dataclasses.replace(
            config.trainer, num_train_steps=2, load_checkpoint=True, load_checkpoint_path=first
        ),
    )
    train_lm.main(resumed)
    latest = latest_checkpoint_path(tmp_path / "checkpoints")
    metadata_after_resume = load_checkpoint_metadata(latest)
    assert metadata_after_resume["step"] == metadata["step"] + 1
    assert config.model.checkpoint_vocab_size(metadata_after_resume) == 32
    _assert_training_recorded(str(tmp_path))

    eval_output = tmp_path / "evaluation"
    eval_lm.main(
        eval_lm.EvalLmConfig(
            checkpoint_path=latest,
            trainer=dataclasses.replace(
                resumed.trainer,
                tracker=JsonFileTrackerConfig(output_path=str(eval_output)),
                max_eval_batches=1,
            ),
            data=resumed.data,
            model=resumed.model,
            max_eval_length=seq_len,
            eval_on_train=True,
        )
    )
    metrics = json.loads((eval_output / "eval_results.json").read_text())
    assert math.isfinite(metrics["eval/loss"])


def _frozen_lm_config(tmpdir, data_config, *, out, seed, init_from=None, ckpt_base=None):
    """A tiny one-step ``TrainLmConfig`` with LR 0 (weights frozen) and a fixed data order.

    Freezing the weights (LR 0) and pinning ``data_seed`` makes the single logged ``train/loss`` a
    deterministic function of the initial model and the data alone — so two runs log the same loss
    iff they start from the same weights, regardless of the trainer ``seed`` that would otherwise
    change the random init.
    """
    # Always base the checkpointer under tmpdir (never the default relative "checkpoints/", which would
    # pollute the cwd). The step policy saves every step so the run leaves a discoverable checkpoint;
    # append_run_id_to_base_path=False keeps it at a stable <base>/step-0 address for init_from.
    checkpointer = CheckpointerConfig(
        base_path=ckpt_base or os.path.join(out, "ckpts"),
        save_interval=datetime.timedelta(minutes=30),
        keep=[dict(every=1)],
        append_run_id_to_base_path=False,
    )
    return train_lm.TrainLmConfig(
        data=data_config,
        model=train_lm.LlamaConfig(
            num_layers=2, num_heads=2, num_kv_heads=2, max_seq_len=64, hidden_dim=32, attn_backend=None
        ),
        optimizer=AdamConfig(learning_rate=0.0),
        data_seed=0,
        initialize_model_from_checkpoint_path=init_from,
        trainer=train_lm.TrainerConfig(
            num_train_steps=1,
            train_batch_size=len(jax.devices()),
            max_eval_batches=1,
            tracker=JsonFileTrackerConfig(output_path=out),
            require_accelerator=False,
            seed=seed,
            checkpointer=checkpointer,
            distributed=DistributedConfig(initialize_jax_distributed=False),
        ),
    )


def _logged_loss(output_path: str) -> float:
    with open(os.path.join(output_path, "eval_results.json")) as f:
        return json.load(f)["train/loss"]


def test_train_lm_initialize_model_from_checkpoint():
    """initialize_model_from_checkpoint_path restores the model weights (fresh optimizer, step 0).

    Run once from scratch (seed 0) saving a checkpoint, then again initializing weights-only from it
    with a different trainer seed. With LR 0 and a fixed data order, the second run logs the first
    run's loss iff the weights were actually loaded; a control run from scratch with the second seed
    logs a different loss, proving the assertion is sensitive to which weights the model started from.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_corpus.construct_small_data_cache(tmpdir)

        ref_base = os.path.join(tmpdir, "ref_ckpts")
        out_ref = os.path.join(tmpdir, "ref")
        train_lm.main(_frozen_lm_config(tmpdir, data_config, out=out_ref, seed=0, ckpt_base=ref_base))
        loss_ref = _logged_loss(out_ref)

        out_init = os.path.join(tmpdir, "init")
        train_lm.main(
            _frozen_lm_config(tmpdir, data_config, out=out_init, seed=999, init_from=latest_checkpoint_path(ref_base))
        )
        loss_init = _logged_loss(out_init)

        out_ctrl = os.path.join(tmpdir, "ctrl")
        train_lm.main(_frozen_lm_config(tmpdir, data_config, out=out_ctrl, seed=999))
        loss_ctrl = _logged_loss(out_ctrl)

        assert loss_init == pytest.approx(loss_ref, abs=1e-4), "weights were not restored from the checkpoint"
        assert abs(loss_ctrl - loss_ref) > 1e-4, "control run coincidentally matched; test is not sensitive"


def test_train_lm_rejects_multiple_init_sources():
    """The three weight-init sources are mutually exclusive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_corpus.construct_small_data_cache(tmpdir)
        config = train_lm.TrainLmConfig(
            data=data_config,
            model=train_lm.LlamaConfig(num_layers=2, num_heads=2, num_kv_heads=2, max_seq_len=64, hidden_dim=32),
            initialize_from_checkpoint_path="/some/full/checkpoint",
            initialize_model_from_checkpoint_path="/some/model/checkpoint",
            trainer=train_lm.TrainerConfig(
                num_train_steps=1,
                train_batch_size=len(jax.devices()),
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
            ),
        )
        with pytest.raises(ValueError, match="at most one of"):
            train_lm.main(config)


def test_train_lm_rejects_weights_only_init_with_trainer_initialize_from():
    """trainer.initialize_from is a full-state resume; combined with the weights-only field it would
    restore step > 0 and silently skip the weights-only init, so reject the combination."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_corpus.construct_small_data_cache(tmpdir)
        config = train_lm.TrainLmConfig(
            data=data_config,
            model=train_lm.LlamaConfig(num_layers=2, num_heads=2, num_kv_heads=2, max_seq_len=64, hidden_dim=32),
            initialize_model_from_checkpoint_path="/some/model/checkpoint",
            trainer=train_lm.TrainerConfig(
                num_train_steps=1,
                train_batch_size=len(jax.devices()),
                initialize_from="/some/full/checkpoint",
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
            ),
        )
        with pytest.raises(ValueError, match="trainer.initialize_from"):
            train_lm.main(config)
