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


def test_delphi_resume_output_path_sets_all_run_identity():
    import experiments.exp_delphi_math_10b_midtrain as delphi

    output_path = "gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.67-ecbd27"

    assert delphi._resume_run_id_from_output_path(output_path) == "delphi-1e21-p67m33-9p25b-lr0.67-ecbd27"
    assert delphi._resume_identity_env_vars(output_path) == {
        "RUN_ID": "delphi-1e21-p67m33-9p25b-lr0.67-ecbd27",
        "WANDB_RUN_ID": "delphi-1e21-p67m33-9p25b-lr0.67-ecbd27",
        "WANDB_RESUME": "allow",
    }


def test_delphi_resume_env_requires_min_step_unless_empty_resume():
    import experiments.exp_delphi_math_10b_midtrain as delphi

    output_path = "gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.67-ecbd27"

    with pytest.raises(ValueError, match="requires MIDTRAIN_EXPECT_RESUME_MIN_STEP"):
        delphi._validate_resume_env_contract(
            resume_output_path=output_path,
            expected_min_step=None,
            allow_empty=False,
        )

    delphi._validate_resume_env_contract(
        resume_output_path=output_path,
        expected_min_step=2600,
        allow_empty=False,
    )
    delphi._validate_resume_env_contract(
        resume_output_path=output_path,
        expected_min_step=None,
        allow_empty=True,
    )


def test_delphi_resume_output_path_must_match_selected_cell(monkeypatch):
    import experiments.exp_delphi_math_10b_midtrain as delphi

    monkeypatch.setattr(delphi, "_MIDTRAIN_MIX_NAME", delphi.PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME)
    token_budget = delphi._token_budget_for_base(delphi.BASES["1e21-v5"])

    delphi._validate_resume_output_path_matches_run(
        "gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.67-ecbd27",
        base_tag="1e21-v5",
        lr_factor=0.67,
        token_budget=token_budget,
    )

    with pytest.raises(ValueError, match="does not match selected Delphi midtraining run"):
        delphi._validate_resume_output_path_matches_run(
            "gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.5-114e49",
            base_tag="1e21-v5",
            lr_factor=0.67,
            token_budget=token_budget,
        )


def test_delphi_resume_checkpoint_preflight_rejects_low_step(monkeypatch):
    import experiments.exp_delphi_math_10b_midtrain as delphi

    output_path = "gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.67-ecbd27"
    monkeypatch.setattr(
        delphi,
        "_discover_latest_resume_checkpoint",
        lambda path: f"{path}/checkpoints/step-1013",
    )

    with pytest.raises(ValueError, match="below MIDTRAIN_EXPECT_RESUME_MIN_STEP=2600"):
        delphi._verify_resume_checkpoint_namespace(
            output_path,
            expected_min_step=2600,
            allow_empty=False,
        )


def test_delphi_resume_checkpoint_preflight_requires_checkpoint_unless_explicitly_empty(monkeypatch):
    import experiments.exp_delphi_math_10b_midtrain as delphi

    output_path = "gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.67-ecbd27"
    monkeypatch.setattr(delphi, "_discover_latest_resume_checkpoint", lambda path: None)

    with pytest.raises(FileNotFoundError, match="No checkpoint found for resume output path"):
        delphi._verify_resume_checkpoint_namespace(
            output_path,
            expected_min_step=None,
            allow_empty=False,
        )

    assert (
        delphi._verify_resume_checkpoint_namespace(
            output_path,
            expected_min_step=None,
            allow_empty=True,
        )
        is None
    )
