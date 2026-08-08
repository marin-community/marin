# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import pytest
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask

from experiments.grug.moe_hero_fsdp import launch, model, train

EOS_ID = 128_001


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


def test_thd_base_mask_derives_metadata_from_training_mask():
    """The training mask carries segment ids but no THD metadata, so it must be derived.

    ``GrugLmExample.causal`` builds segment ids from an EOS cumsum without passing
    ``max_segments``, which is exactly the shape the native SM100 kernel rejects.
    """
    tokens = jnp.array([[5, 7, EOS_ID, 9, 3, EOS_ID, 4, 4]], dtype=jnp.int32)
    training_mask = GrugLmExample.causal(tokens=tokens[0], eos_id=EOS_ID).attn_mask
    assert training_mask.thd_segment_metadata is None, "precondition: conversion carries no metadata"
    assert training_mask.segment_ids is not None

    base_mask = model._thd_base_mask(training_mask.segment_ids, None, max_segments=4)

    metadata = base_mask.thd_segment_metadata
    assert metadata is not None
    assert int(metadata.num_segments.reshape(-1)[0]) == 3
    # Documents end after each EOS, so the packed lengths are [3, 3, 2] padded to max_segments.
    assert metadata.segment_lengths.reshape(-1)[:3].tolist() == [3, 3, 2]


def test_thd_base_mask_requires_segment_ids():
    with pytest.raises(NotImplementedError, match="packed segment ids"):
        model._thd_base_mask(None, None, max_segments=4)
