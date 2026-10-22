# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from types import SimpleNamespace

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.callbacks import _iris_status
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.eval import _default_lm_eval_loss_fn
from levanter.main.train_lm import TrainLmConfig, main
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmExample
from levanter.models.loss import maybe_fused_next_token_loss
from levanter.models.snowball import SnowballConfig
from levanter.optim.config import AdamConfig
from levanter.testing.cpu_devices import run_on_cpu_devices
from levanter.tracker import get_tracker
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from safetensors.numpy import load_file
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from experiments.post_training.agentic_sweep.callbacks import EpochTracker, EpochTrackerConfig, SweepExportsConfig
from experiments.post_training.agentic_sweep.export import ExportConfig, export_checkpoint
from experiments.post_training.agentic_sweep.validate import OFFLINE_MESH, OfflineValidationConfig, evaluate_checkpoint


def test_offline_snowball_loss_uses_matching_explicit_batch_layout():
    run_on_cpu_devices(
        "import runpy; runpy.run_path('tests/experiment/test_agentic_sweep.py')['_check_offline_snowball_layout']()",
        device_count=8,
    )


def _check_offline_snowball_layout():
    trainer = TrainerConfig(
        mesh=OFFLINE_MESH,
        use_explicit_mesh_axes=True,
        train_batch_size=8,
        per_device_parallelism=1,
    )
    model_config = SnowballConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=32,
        shared_expert_intermediate_dim=32,
        num_experts=8,
        num_experts_per_token=1,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        attention_implementation="reference",
    )
    batch = hax.Axis("batch", 8)
    position = hax.Axis("position", 8)
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        model = model_config.build(hax.Axis("vocab", 32), key=jax.random.PRNGKey(0))
        tokens = jnp.broadcast_to(jnp.arange(8), (8, 8))
        example = hax.vmap(LmExample.causal, batch)(hax.named(tokens, (batch, position)))
        example = hax.shard(example)

        @hax.named_jit(axis_resources=trainer.compute_axis_mapping)
        def compute(model, example):
            return _default_lm_eval_loss_fn(model, example, EvalBatch=batch, mp=None)

        loss, weights, token_ids = compute(model, example)
    assert np.isfinite(np.asarray(loss)).all()
    np.testing.assert_array_equal(token_ids, np.tile([1, 2, 3, 4, 5, 6, 7, 0], (8, 1)))
    assert float(weights.sum()) == 56


def test_unequal_assistant_masks_use_global_token_mean_and_gradient():
    batch = hax.Axis("batch", 16)
    position = hax.Axis("position", 8)
    embedding = hax.Axis("embedding", 11)
    vocab = hax.Axis("vocab", 11)
    tokens = jnp.arange(batch.size * position.size).reshape(batch.size, position.size) % vocab.size
    lengths = jnp.arange(batch.size) % position.size
    weights = (jnp.arange(position.size)[None, :] < lengths[:, None]).astype(jnp.float32)
    logits = jax.random.normal(jax.random.PRNGKey(42), (batch.size, position.size, vocab.size))
    targets = jnp.roll(tokens, -1, axis=1)

    def reference(values):
        losses = -jnp.take_along_axis(jax.nn.log_softmax(values), targets[..., None], axis=-1)[..., 0]
        return jnp.sum(losses * weights) / weights.sum()

    def actual(values):
        return maybe_fused_next_token_loss(
            position,
            embedding,
            vocab,
            hax.named(values, (batch, position, embedding)),
            hax.named(jnp.eye(vocab.size), (vocab, embedding)),
            hax.named(tokens, (batch, position)),
            loss_weight=hax.named(weights, (batch, position)),
        ).scalar()

    expected_loss, expected_gradient = jax.value_and_grad(reference)(logits)
    loss, gradient = jax.value_and_grad(actual)(logits)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(gradient)[np.asarray(weights) == 0], 0)
    losses = -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[..., None], axis=-1)[..., 0]
    sequence_mean = jnp.mean(jnp.sum(losses * weights, axis=1) / jnp.maximum(lengths, 1))
    assert abs(float(loss - sequence_mean)) > 0.01


def test_epoch_statistics_weight_tokens_without_validation(tmp_path):
    tracker = EpochTracker(str(tmp_path), steps_per_epoch=4, tokens_per_epoch=10)
    for step, (tokens, loss) in enumerate([(1, 2.0), (2, 3.0), (3, 4.0), (4, 5.0)]):
        tracker.log({"train/loss_sum": tokens * loss, "train/supervised_tokens": tokens}, step=step)
    rows = [json.loads(line) for line in (tmp_path / "epoch-metrics.jsonl").read_text().splitlines()]
    assert rows == [
        {
            "epoch": 1,
            "completed_updates": 4,
            "checkpoint_step": 3,
            "train_loss": 4.0,
            "supervised_tokens": 10,
            "manifest_supervised_tokens": 10,
            "supervised_token_difference": 0,
        }
    ]


@pytest.mark.timeout(180)
def test_epoch_checkpoints_offline_validation_and_final_export(tmp_path, monkeypatch):
    status_reports = []
    client = SimpleNamespace(
        report_task_status_text=lambda *args: status_reports.append((get_tracker("wandb").name, args))
    )
    monkeypatch.setattr(_iris_status, "get_iris_ctx", lambda: SimpleNamespace(client=client))
    monkeypatch.setattr(_iris_status, "get_job_info", lambda: SimpleNamespace(task_id="/test/0", attempt_id=0))
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel(
                {"<pad>": 0, "<eos>": 1, **{f"t{index}": index for index in range(2, 32)}},
                unk_token="<pad>",
            )
        ),
        pad_token="<pad>",
        eos_token="<eos>",
    )
    tokenizer.save_pretrained(tokenizer_path)
    examples = [GrugLmExample.causal(jnp.arange(8, dtype=jnp.int32) + 2) for _ in range(8)]
    data = LmDataConfig(
        components={"train": DirectDatasetComponent(datasets={"train": ListAsyncDataset(examples)})},
        tokenizer=str(tokenizer_path),
        shuffle=False,
    )
    validation = LmDataConfig(
        components={
            "tmax": DirectDatasetComponent(datasets={"validation": ListAsyncDataset(examples[:2])}),
            "swesmith": DirectDatasetComponent(datasets={"validation": ListAsyncDataset(examples[:4])}),
        },
        tokenizer=str(tokenizer_path),
        shuffle=False,
    )
    checkpoint_path = tmp_path / "checkpoints"
    config = TrainLmConfig(
        data=data,
        train_seq_len=8,
        model=LlamaConfig(
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            max_seq_len=64,
            hidden_dim=16,
            intermediate_dim=32,
            attn_backend=None,
            reference_checkpoint=str(tokenizer_path),
        ),
        optimizer=AdamConfig(learning_rate=1e-6, weight_decay=0, warmup=0, lr_schedule="constant"),
        trainer=TrainerConfig(
            id="epoch-integration",
            train_batch_size=2,
            per_device_parallelism=2,
            per_device_eval_parallelism=2,
            num_train_steps=12,
            load_checkpoint=False,
            tracker=(
                WandbConfig(mode="offline", project="sweep-integration", save_code=False),
                EpochTrackerConfig(str(tmp_path), 4, 56),
                JsonLoggerConfig(),
                JsonFileTrackerConfig(output_path=str(tmp_path)),
            ),
            checkpointer=CheckpointerConfig(
                base_path=str(checkpoint_path),
                append_run_id_to_base_path=False,
                keep=[],
                save_interval=None,
            ),
            require_accelerator=False,
            log_jaxprs=False,
            log_xla_hlo=False,
            distributed=DistributedConfig(initialize_jax_distributed=False),
        ),
        adapter=SweepExportsConfig(validation, 4, 8),
        hf_save_path=str(tmp_path / "hf"),
    )
    main(config)
    rows = [json.loads(line) for line in (tmp_path / "epoch-metrics.jsonl").read_text().splitlines()]
    assert [row["completed_updates"] for row in rows] == [4, 8, 12]
    assert all("validation_loss" not in row for row in rows)
    assert sorted(path.name for path in checkpoint_path.glob("step-*")) == ["step-11", "step-3", "step-7"]
    export = tmp_path / "hf" / "step-11"
    assert (export / "config.json").exists()
    assert (export / "tokenizer.json").exists()
    assert list(export.glob("*.safetensors"))
    assert status_reports
    assert all(report[0] == "wandb" for report in status_reports)
    assert any("10" in str(report) for report in status_reports)
    result = evaluate_checkpoint(
        OfflineValidationConfig(
            checkpoint_path=str(checkpoint_path / "step-11"),
            output_path=str(tmp_path / "offline-validation.json"),
            model=config.model,
            data=validation,
            trainer=config.trainer,
            sequence_length=8,
        )
    )
    metrics = result["metrics"]
    assert metrics["eval/tmax/tokens"] == 14
    assert metrics["eval/swesmith/tokens"] == 28
    assert metrics["eval/tokens"] == 42
    assert np.isfinite(metrics["eval/loss"])
    np.testing.assert_allclose(
        metrics["eval/loss"],
        (2 * metrics["eval/tmax/loss"] + 4 * metrics["eval/swesmith/loss"]) / 6,
        rtol=1e-5,
    )
    assert json.loads((tmp_path / "offline-validation.json").read_text()) == result
    offline_export = tmp_path / "offline-hf"
    export_checkpoint(
        ExportConfig(
            trainer=config.trainer,
            model=config.model,
            reference_checkpoint=str(export),
            tokenizer=str(tokenizer_path),
            checkpoint_path=str(checkpoint_path / "step-11"),
            output_path=str(offline_export),
            eos_token_ids=[1],
        )
    )
    assert (offline_export / "tokenizer.json").read_bytes() == (export / "tokenizer.json").read_bytes()
    assert json.loads((offline_export / "generation_config.json").read_text())["eos_token_id"] == [1]
    for shard in export.glob("*.safetensors"):
        assert (offline_export / shard.name).read_bytes() == shard.read_bytes()
    resumed_path = tmp_path / "resumed"
    main(
        replace(
            config,
            trainer=replace(
                config.trainer,
                id="epoch-resume-integration",
                load_checkpoint=True,
                load_checkpoint_path=str(checkpoint_path / "step-3"),
                tracker=(
                    WandbConfig(mode="offline", project="sweep-integration", save_code=False),
                    EpochTrackerConfig(str(resumed_path), 4, 56, start_step=4),
                    JsonLoggerConfig(),
                ),
                checkpointer=replace(config.trainer.checkpointer, base_path=str(resumed_path / "checkpoints")),
            ),
            hf_save_path=str(resumed_path / "hf"),
        )
    )
    resumed_rows = [json.loads(line) for line in (resumed_path / "epoch-metrics.jsonl").read_text().splitlines()]
    assert [row["completed_updates"] for row in resumed_rows] == [8, 12]
    np.testing.assert_allclose([row["train_loss"] for row in resumed_rows], [row["train_loss"] for row in rows[1:]])
    for shard in export.glob("*.safetensors"):
        expected = load_file(str(shard))
        restored = load_file(str(resumed_path / "hf" / "step-11" / shard.name))
        assert expected.keys() == restored.keys()
        for name in expected:
            np.testing.assert_array_equal(expected[name], restored[name])
