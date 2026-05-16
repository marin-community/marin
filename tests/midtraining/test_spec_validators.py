# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validators tied to incident-ledger entries from the redesign doc."""

import dataclasses
from unittest.mock import patch

import pytest
from marin.midtraining.modes import (
    CheckpointSourceKind,
    CooldownMode,
    CptInit,
    CptMode,
)
from marin.midtraining.spec import MidtrainSpec, resolve_midtrain_spec, validate_midtrain_spec

from tests.midtraining._fixtures import (
    FAKE_1E21,
    FAKE_1E22,
    make_cooldown_spec,
    make_cpt_spec,
    make_data_manifest,
    make_model_config,
)


def _resolve(spec: MidtrainSpec, *, region: str = "us-east5"):
    manifest = make_data_manifest(region=region)
    with patch("marin.midtraining.spec.load_data_manifest", return_value=manifest):
        return resolve_midtrain_spec(spec)


def _resolve_and_validate(spec: MidtrainSpec, *, region: str = "us-east5") -> None:
    validate_midtrain_spec(_resolve(spec, region=region))


def test_banned_v5_isoflop_path_rejected():
    bad = dataclasses.replace(
        FAKE_1E21,
        gcs_run_root="gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5",
    )
    spec = make_cpt_spec(base=bad)
    with pytest.raises(ValueError, match="banned substring"):
        _resolve_and_validate(spec)


def test_cpt_requires_optimizer_reset():
    with pytest.raises(ValueError, match="reset_optimizer"):
        CptInit(
            source_kind=CheckpointSourceKind.NATIVE_LEVANTER,
            registry_model=FAKE_1E21,
            reset_optimizer=False,
        )


def test_cooldown_requires_original_pretrain_num_train_steps():
    spec = make_cooldown_spec()
    resolved = _resolve(spec)
    assert resolved.num_train_steps == FAKE_1E22.num_train_steps


def test_cooldown_stop_step_override_cannot_exceed_base_steps():
    spec = make_cooldown_spec(stop_step_override=FAKE_1E22.num_train_steps + 1)
    with pytest.raises(ValueError, match=r"exceeds base\.num_train_steps"):
        _resolve_and_validate(spec)


def test_cooldown_resume_step_must_be_before_base_end():
    bad = make_cooldown_spec(resume_step=FAKE_1E22.num_train_steps)
    with pytest.raises(ValueError, match="must be <"):
        _resolve_and_validate(bad)


def test_modes_are_disjoint_by_construction():
    """Mode is one object, so 'mixing CPT and cooldown' is impossible by construction."""
    assert isinstance(make_cpt_spec().mode, CptMode)
    assert isinstance(make_cooldown_spec().mode, CooldownMode)


def test_compute_regions_must_collapse_to_output_region():
    spec = make_cpt_spec(extra_compute_kwargs={"regions": ("us-east5", "us-central1")})
    with pytest.raises(ValueError, match="must equal"):
        _resolve_and_validate(spec)


def test_data_manifest_region_must_match_run_region():
    spec = make_cpt_spec(data_manifest_uri="gs://marin-us-central1/midtrain-manifests/data/p33m67/abc.json")
    with pytest.raises(ValueError, match="does not match run output region"):
        _resolve_and_validate(spec, region="us-east5")


def test_data_manifest_requires_bos_sample():
    from marin.midtraining.data_manifest import DataCacheComponent, DataCacheManifest
    from marin.midtraining.tokenizers import LLAMA3_TOKENIZER

    bos_missing = DataCacheComponent(
        logical_name="x",
        cache_path="gs://marin-us-east5/tokenized/x",
        cache_digest="sha256:test",
        tokenizer=LLAMA3_TOKENIZER,
        bos_sample=(),
    )
    with pytest.raises(ValueError, match="has no bos_sample"):
        DataCacheManifest(
            mix_name="m",
            mix_spec_digest="sha256:m",
            region="us-east5",
            components=(bos_missing,),
            weights={"x": 1.0},
            seq_len=4096,
        )


def test_cooldown_eval_basis_is_remaining_steps():
    """Eval cadence basis is mode-derived — no user knob anymore."""
    from marin.midtraining.levanter_config import _eval_basis_steps

    spec = make_cooldown_spec(resume_step=30_000)
    resolved = _resolve(spec)
    assert _eval_basis_steps(resolved) == FAKE_1E22.num_train_steps - 30_000


def test_cpt_eval_basis_is_total_cpt_steps():
    from marin.midtraining.levanter_config import _eval_basis_steps

    spec = make_cpt_spec()
    resolved = _resolve(spec)
    assert _eval_basis_steps(resolved) == resolved.num_train_steps


def test_model_config_must_match_base_shape():
    bad_model_config = {**make_model_config(FAKE_1E21), "hidden_dim": 9999}
    spec = make_cpt_spec(model_config_override=bad_model_config)
    with pytest.raises(ValueError, match=r"does not match base\.hidden_dim"):
        _resolve_and_validate(spec)


def test_checkpoint_override_requires_meaningful_reason():
    from marin.midtraining.modes import CheckpointOverride
    from marin.midtraining.tokenizers import LLAMA3_TOKENIZER

    with pytest.raises(ValueError, match="meaningful explanation"):
        CheckpointOverride(
            checkpoint_path="gs://marin-us-east5/x/checkpoints/step-1",
            reason="why",
            run_name_suffix="-override",
            expected_hidden_dim=2560,
            expected_num_layers=26,
            expected_seq_len=4096,
            expected_tokenizer=LLAMA3_TOKENIZER,
        )


def test_cpt_init_rejects_unpinned_hf_revision():
    with pytest.raises(ValueError, match="pinned hf_revision"):
        CptInit(
            source_kind=CheckpointSourceKind.HF_WEIGHTS,
            hf_repo="marin-community/delphi-1e21-3.4Bparams-46.3Btokens",
            hf_revision="main",
        )


def test_resume_settings_reject_inconsistent_flags():
    """allow_empty_resume + expected_min_step > 0 is impossible by construction."""
    with pytest.raises(ValueError, match="incompatible"):
        make_cpt_spec(expected_min_step=100).__class__(
            **{**dataclasses.asdict(make_cpt_spec(expected_min_step=100)), "allow_empty_resume": True}
        )


def test_cpt_init_demands_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one"):
        CptInit(
            source_kind=CheckpointSourceKind.NATIVE_LEVANTER,
            registry_model=FAKE_1E21,
            hf_repo="marin-community/delphi-1e21-3.4Bparams-46.3Btokens",
            hf_revision="ca7b0e7c0a6b9ea8e3a4bbe847efa8b53f793902",
        )
