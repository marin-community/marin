# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
from levanter.grug.attention import AttentionMask

from experiments.grug.moe_hero_fsdp import launch, model, train


def test_build_hero_run_uses_run_id_argument(monkeypatch):
    monkeypatch.setenv("RUN_ID", "ignored-environment-run")

    step = launch.build_hero_run(
        run_id="cli-run",
        dp_racks=1,
        num_steps=1,
        version="2026.08.01",
    )

    assert step.name == "grug/cli-run"


def test_run_grug_applies_xla_command_buffer_default_and_keeps_override(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_latency_hiding_scheduler=true")
    config = SimpleNamespace(
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="test-run")),
        resources=object(),
        processes_per_task=1,
        run_mode=train.GrugRunMode.DEFAULT,
    )

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

        assert os.environ["XLA_FLAGS"].split() == [
            "--xla_gpu_enable_latency_hiding_scheduler=true",
            train.XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG,
        ]

        explicit_flags = "--xla_gpu_enable_command_buffer=FUSION"
        monkeypatch.setenv("XLA_FLAGS", explicit_flags)
        train.run_grug(config)

        assert os.environ["XLA_FLAGS"] == explicit_flags


def test_layer_attention_masks_preserve_thd_segment_metadata():
    mask = AttentionMask.causal().with_segment_ids(
        jnp.array([[0, 0, 1, 1, -1, -1]], dtype=jnp.int32),
        max_segments=3,
    )

    short_mask, long_mask = model._layer_attention_masks(mask, sliding_window=512)

    assert short_mask.thd_segment_metadata is mask.thd_segment_metadata
    assert long_mask.thd_segment_metadata is mask.thd_segment_metadata
    assert short_mask.segment_ids is mask.segment_ids
    assert long_mask.segment_ids is mask.segment_ids
    assert short_mask.sliding_window == 512
    assert long_mask.sliding_window is None
