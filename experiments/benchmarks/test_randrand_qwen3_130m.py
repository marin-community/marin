# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Benchmark geometry and actual training/evaluation behavior on synthetic data."""

import dataclasses
import json

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from levanter.checkpoint import CheckpointerConfig, latest_checkpoint_path
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.layers.attention import AttentionBackend
from levanter.main.train_lm import main as train_main
from levanter.optim.randrand import parameter_layout
from levanter.optim.randrand_muon_hybrid import MuonBlendConfig, RandRANDMuonHybridConfig
from levanter.tracker.json_file import JsonFileTrackerConfig

from experiments.benchmarks.randrand_qwen3_130m import Recipe, build_config, model_config


@pytest.mark.parametrize("recipe", list(Recipe))
def test_130m_routes_preserve_whole_attention_matrices(recipe):
    config = model_config(recipe)
    model = eqx.filter_eval_shape(config.build, hax.Axis("vocab", 128256), key=jax.random.PRNGKey(0))
    layouts, _ = parameter_layout(model)
    counts = {kind: 0 for kind in ("matrix", "embed", "vector", "head")}
    matrices = []
    for layout in layouts:
        counts[layout.kind] += np.prod(layout.ordered_shape)
        if layout.kind == "matrix":
            matrices.extend([(layout.rows, layout.columns)] * layout.batch_size)
    vector_count = 13568 if recipe != Recipe.PR6301 else 7424
    assert counts == dict(matrix=22806528, embed=65667072, vector=vector_count, head=65667072)
    assert matrices.count((512, 512)) == 24
    assert matrices.count((1792, 512)) == 12
    assert matrices.count((512, 1792)) == 6


@pytest.mark.timeout(180)
@pytest.mark.parametrize("recipe", list(Recipe))
def test_training_saves_final_off_interval_validation_and_checkpoint(tmp_path, recipe):
    config = build_config(
        recipe=recipe,
        train_cache=str(tmp_path / "unused-train"),
        c4_cache=str(tmp_path / "unused-val"),
        output_path=str(tmp_path / "checkpoints"),
        run_id="randrand-test",
    )
    examples = ListAsyncDataset([GrugLmExample.causal((jnp.arange(8, dtype=jnp.int32) + i) % 32) for i in range(8)])
    config.data = LmDataConfig(
        tokenizer="passthrough",
        vocab_size=32,
        cache_dir=None,
        shuffle=False,
        components={"paloma/c4_en": DirectDatasetComponent(datasets={"train": examples, "validation": examples})},
        train_weights={"paloma/c4_en": 1.0},
    )
    config.model = dataclasses.replace(
        config.model,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=8,
        gradient_checkpointing=False,
        attn_backend=AttentionBackend.VANILLA,
    )
    config.train_seq_len = 8
    config.optimizer = dataclasses.replace(
        config.optimizer,
        rank=2,
        rank_sketch=3,
        pair_k=1,
        min_dim_for_deflation=1,
        randrand_warmup=2,
        basis_every=2,
        warmup=2,
    )
    if isinstance(config.optimizer, RandRANDMuonHybridConfig):
        config.optimizer = dataclasses.replace(
            config.optimizer, switch=MuonBlendConfig(min_step=1, threshold=1.0, ramp=2)
        )
    config.trainer = dataclasses.replace(
        config.trainer,
        num_train_steps=3,
        steps_per_eval=2,
        train_batch_size=len(jax.devices()),
        per_device_parallelism=1,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        require_accelerator=False,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        tracker=JsonFileTrackerConfig(output_path=str(tmp_path)),
        log_dir=tmp_path / "logs",
        log_jaxprs=False,
        log_xla_hlo=False,
        checkpointer=CheckpointerConfig(base_path=str(tmp_path / "checkpoint"), append_run_id_to_base_path=False),
    )
    train_main(config)
    metrics = json.loads((tmp_path / "eval_results.json").read_text())
    if recipe in (Recipe.RANDRAND_MUON_HYBRID, Recipe.SPECTRAL_RANDRAND_MUON_HYBRID):
        assert 0 < metrics["optim/switch/blend"] <= 1
        assert np.isfinite(metrics["optim/switch/cosine_ema"])
    assert np.isfinite(metrics["train/loss"])
    assert np.isfinite(metrics["eval/paloma/c4_en/loss"])
    eval_records = [
        json.loads(line) for line in (tmp_path / "checkpoint" / "eval_metrics.jsonl").read_text().splitlines()
    ]
    assert eval_records[-1]["step"] == 2  # Final update is not on the two-update evaluation interval.
    assert np.isfinite(eval_records[-1]["eval/paloma/c4_en/loss"])
    checkpoint = latest_checkpoint_path(str(tmp_path / "checkpoint"))
    assert checkpoint is not None
    metadata = json.loads((tmp_path / "checkpoint" / checkpoint.split("/")[-1] / "metadata.json").read_text())
    assert metadata["step"] == 2  # Trainer logs zero-based steps; three updates completed.
