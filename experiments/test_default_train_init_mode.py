# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plumbing tests: SimpleTrainConfig.checkpoint_init_mode must reach TrainLmConfig via default_train.

Layer 2 of the flat-LR fix test coverage. Layer 1 lives in
``lib/levanter/tests/test_checkpoint.py`` (raw load semantics under
``CheckpointInitMode.MODEL_ONLY`` vs ``FULL_STATE``).

The regression this guards against: a future refactor of ``default_train``
(``experiments/defaults.py``) forgets to forward ``checkpoint_init_mode`` into
``TrainLmConfig``, so a caller setting ``MODEL_ONLY`` silently falls back to
``FULL_STATE``. Raw-load tests alone don't catch that because the plumbing
bypasses them.
"""

import pytest

from levanter.main.train_lm import CheckpointInitMode, TrainLmConfig

from experiments.simple_train_config import SimpleTrainConfig


def test_simple_train_config_default_checkpoint_init_mode_is_full_state():
    """SimpleTrainConfig's default preserves today's behavior so unaudited callers don't regress."""
    from fray.cluster import ResourceConfig

    cfg = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu("v4-8"),
        train_batch_size=1,
        num_train_steps=1,
        learning_rate=1e-3,
    )
    assert cfg.checkpoint_init_mode is CheckpointInitMode.FULL_STATE


def test_train_lm_config_default_checkpoint_init_mode_is_full_state():
    """TrainLmConfig's Levanter-side default matches SimpleTrainConfig's Marin-side default."""
    assert TrainLmConfig().checkpoint_init_mode is CheckpointInitMode.FULL_STATE


def test_default_train_propagates_model_only_mode():
    """End-to-end: a SimpleTrainConfig with MODEL_ONLY set must produce a TrainLmConfig with MODEL_ONLY.

    Uses the Delphi midtrain experiment as a real caller (it explicitly sets
    ``checkpoint_init_mode=CheckpointInitMode.MODEL_ONLY``). If that experiment
    stops exercising the plumbing path, adjust this test to build a direct
    ``default_train`` call with a stub SimpleTrainConfig instead.
    """
    import experiments.exp_delphi_math_10b_midtrain as delphi

    assert len(delphi.runs) == len(delphi.BASES) * len(delphi.LR_FACTORS)
    for step in delphi.runs:
        # step.config is TrainLmOnPodConfig; step.config.train_config is TrainLmConfig.
        inner = step.config.train_config
        assert inner.checkpoint_init_mode is CheckpointInitMode.MODEL_ONLY, (
            f"run {step.name!r} lost its MODEL_ONLY selection on the way to TrainLmConfig "
            f"— default_train may not be forwarding checkpoint_init_mode"
        )
        # Sanity: the path to load is also set, otherwise the mode has no effect.
        assert inner.initialize_from_checkpoint_path is not None, (
            f"run {step.name!r} has MODEL_ONLY but no initialize_from_checkpoint_path "
            f"— the mode is dead code without a path"
        )


def test_delphi_midtrain_child_resources_follow_coordinator_region(monkeypatch):
    import experiments.exp_delphi_math_10b_midtrain as delphi

    monkeypatch.setattr(delphi, "MIDTRAIN_TRAIN_REGION", None)
    monkeypatch.setattr(delphi, "marin_region", lambda: "us-east5")

    resources = delphi._midtrain_tpu_resources("v5p-64")

    assert resources.regions == ["us-east5"]


def test_delphi_midtrain_train_region_override_accepts_zone(monkeypatch):
    import experiments.exp_delphi_math_10b_midtrain as delphi

    monkeypatch.setattr(delphi, "MIDTRAIN_TRAIN_REGION", "us-central1-a")

    resources = delphi._midtrain_tpu_resources("v5p-64")

    assert resources.regions == ["us-central1"]


def test_delphi_midtrain_explicit_bad_region_fails(monkeypatch):
    import experiments.exp_delphi_math_10b_midtrain as delphi

    monkeypatch.setattr(delphi, "MIDTRAIN_TRAIN_REGION", "us-central2")

    with pytest.raises(ValueError, match="Delphi midtraining must run"):
        delphi._midtrain_tpu_resources("v5p-64")


def test_delphi_midtrain_ignores_local_non_v5p_region(monkeypatch):
    import experiments.exp_delphi_math_10b_midtrain as delphi

    monkeypatch.setattr(delphi, "MIDTRAIN_TRAIN_REGION", None)
    monkeypatch.setattr(delphi, "marin_region", lambda: "us-central2")

    resources = delphi._midtrain_tpu_resources("v5p-64")

    assert resources.regions is None
