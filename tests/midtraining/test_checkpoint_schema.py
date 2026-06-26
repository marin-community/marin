# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OCDBT key-schema fail-closed guards.

Background: the 2026-05-27 silent-type-degradation bug class produced
checkpoints that looked complete by file-presence checks but were missing
Qwen3 q_norm/k_norm arrays in their OCDBT kvstore. These tests pin the
behaviour of the fail-closed helpers in
``marin.midtraining.checkpoint_schema`` so the same failure mode can never
recur silently in this codebase.
"""

from unittest.mock import patch

import pytest
from marin.midtraining.checkpoint_schema import (
    assert_checkpoint_complete_for_model_type,
    assert_qwen3_qk_norm_present,
    default_list_checkpoint_keys,
)
from marin.midtraining.preflight import fake_gcs, preflight
from marin.midtraining.spec import MidtrainSpec, resolve_midtrain_spec

from tests.midtraining._fixtures import (
    FAKE_1E22,
    make_cooldown_spec,
    make_data_manifest,
    make_model_config,
)


def _qwen3_ish_keys(num_layers: int) -> tuple[str, ...]:
    """Synthesize the OCDBT key list a healthy Qwen3 checkpoint would expose.

    Real Levanter checkpoints surface chunk + metadata keys per leaf; we
    only care about substring presence here, so a single key per leaf is
    sufficient for substring-match assertions.
    """
    keys = ["root.metadata"]
    for layer in range(num_layers):
        keys.append(f"model.layers.{layer}.self_attn.q_proj.weight")
        keys.append(f"model.layers.{layer}.self_attn.k_proj.weight")
        keys.append(f"model.layers.{layer}.self_attn.q_norm.weight")
        keys.append(f"model.layers.{layer}.self_attn.k_norm.weight")
    return tuple(keys)


def _llama_degraded_keys(num_layers: int) -> tuple[str, ...]:
    """Same checkpoint after silent type degradation: no q_norm/k_norm."""
    keys = ["root.metadata"]
    for layer in range(num_layers):
        keys.append(f"model.layers.{layer}.self_attn.q_proj.weight")
        keys.append(f"model.layers.{layer}.self_attn.k_proj.weight")
    return tuple(keys)


# ---------------------------------------------------------------------------
# assert_qwen3_qk_norm_present
# ---------------------------------------------------------------------------


def test_qwen3_qk_norm_present_passes_for_healthy_checkpoint():
    assert_qwen3_qk_norm_present(
        "gs://fake/healthy",
        num_layers=11,
        list_keys=lambda _: _qwen3_ish_keys(11),
    )


def test_qwen3_qk_norm_present_fails_when_both_missing():
    with pytest.raises(ValueError, match="missing Qwen3 QK-norm arrays") as exc:
        assert_qwen3_qk_norm_present(
            "gs://fake/llama-degraded",
            num_layers=11,
            list_keys=lambda _: _llama_degraded_keys(11),
        )
    # Error must explicitly name both missing markers AND link to the helper.
    msg = str(exc.value)
    assert "q_norm" in msg and "k_norm" in msg
    assert "scripts/materialize_delphi_prefix_checkpoint.py" in msg
    assert "5afac0bdf" in msg


def test_qwen3_qk_norm_present_fails_when_only_q_norm_missing():
    keys = [k for k in _qwen3_ish_keys(11) if "q_norm" not in k]
    with pytest.raises(ValueError, match="missing Qwen3 QK-norm arrays") as exc:
        assert_qwen3_qk_norm_present("gs://fake/x", num_layers=11, list_keys=lambda _: tuple(keys))
    assert "q_norm" in str(exc.value)
    assert "k_norm" not in str(exc.value).split("missing Qwen3 QK-norm arrays")[1].split(".")[0]


def test_qwen3_qk_norm_present_rejects_zero_layers():
    with pytest.raises(ValueError, match="num_layers must be positive"):
        assert_qwen3_qk_norm_present("gs://fake/x", num_layers=0, list_keys=lambda _: ())


# ---------------------------------------------------------------------------
# assert_checkpoint_complete_for_model_type
# ---------------------------------------------------------------------------


def test_dispatcher_invokes_qwen3_check_for_qwen3():
    with pytest.raises(ValueError, match="missing Qwen3 QK-norm arrays"):
        assert_checkpoint_complete_for_model_type(
            "gs://fake/bad",
            model_type="qwen3",
            num_layers=11,
            list_keys=lambda _: _llama_degraded_keys(11),
        )


def test_dispatcher_does_not_invoke_qwen3_check_for_llama():
    sentinel: list[str] = []

    def fake_list(_):
        sentinel.append("called")
        return ()

    # Llama is not in _MODEL_TYPES_WITH_QK_NORM; the dispatcher should skip
    # the class-specific check, so list_keys must not be called.
    assert_checkpoint_complete_for_model_type(
        "gs://fake/llama",
        model_type="llama",
        num_layers=11,
        list_keys=fake_list,
    )
    assert sentinel == []


def test_dispatcher_rejects_empty_model_type():
    with pytest.raises(ValueError, match="non-empty model_type"):
        assert_checkpoint_complete_for_model_type(
            "gs://fake/x",
            model_type="",
            num_layers=11,
            list_keys=lambda _: (),
        )


# ---------------------------------------------------------------------------
# default_list_checkpoint_keys: surface tensorstore errors loudly
# ---------------------------------------------------------------------------


def test_default_list_checkpoint_keys_surfaces_unreachable_path():
    # Real tensorstore would raise; we patch it to simulate the failure
    # path and confirm we re-raise as RuntimeError with our message.
    with patch("tensorstore.KvStore") as mock_kvstore:
        mock_kvstore.open.side_effect = RuntimeError("simulated open failure")
        with pytest.raises(RuntimeError, match="Could not list OCDBT keys"):
            default_list_checkpoint_keys("gs://fake/unreachable")


# ---------------------------------------------------------------------------
# Preflight integration
# ---------------------------------------------------------------------------


def _resolve(spec: MidtrainSpec, region: str = "us-east5"):
    manifest = make_data_manifest(region=region)
    with patch("marin.midtraining.spec.load_data_manifest", return_value=manifest):
        return resolve_midtrain_spec(spec)


def _qwen3_cooldown_spec(region: str = "us-east5", resume_step: int = 30_000) -> MidtrainSpec:
    """A cooldown spec whose model_config declares ``type: qwen3``."""
    spec = make_cooldown_spec(region=region, resume_step=resume_step)
    qwen3_cfg = {**make_model_config(FAKE_1E22), "type": "qwen3"}
    return type(spec)(
        **{**{f.name: getattr(spec, f.name) for f in spec.__dataclass_fields__.values()}, "model_config": qwen3_cfg}
    )


def test_preflight_blocks_qwen3_cooldown_when_staged_ckpt_missing_qk_norm():
    spec = _qwen3_cooldown_spec(region="us-central2")  # avoid cross-region failure
    resolved = _resolve(spec, region="us-central2")
    staged = spec.mode.resume.staged_checkpoint_path
    exists, list_ = fake_gcs(
        spec.data_manifest_uri,
        staged,
        f"{staged}/manifest.ocdbt",
        f"{staged}/metadata.json",
        f"{staged}/d",
    )
    report = preflight(
        resolved,
        exists=exists,
        list_=list_,
        list_ocdbt_keys=lambda _: _llama_degraded_keys(FAKE_1E22.num_layers),
    )
    assert not report.ok
    assert any("missing Qwen3 QK-norm arrays" in f for f in report.failures), report.failures


def test_preflight_passes_qwen3_cooldown_when_staged_ckpt_has_qk_norm():
    spec = _qwen3_cooldown_spec(region="us-central2")
    resolved = _resolve(spec, region="us-central2")
    staged = spec.mode.resume.staged_checkpoint_path
    exists, list_ = fake_gcs(
        spec.data_manifest_uri,
        staged,
        f"{staged}/manifest.ocdbt",
        f"{staged}/metadata.json",
        f"{staged}/d",
    )
    report = preflight(
        resolved,
        exists=exists,
        list_=list_,
        list_ocdbt_keys=lambda _: _qwen3_ish_keys(FAKE_1E22.num_layers),
    )
    # Schema check should pass; other failures (if any) come from elsewhere
    # in preflight — explicitly assert no QK-norm-related failure surfaced.
    assert all("missing Qwen3 QK-norm arrays" not in f for f in report.failures), report.failures


def test_preflight_skips_schema_check_when_artifacts_missing():
    # If manifest.ocdbt / metadata.json / d/ are absent, the artifact
    # failure is the primary signal; we should not also produce a confusing
    # secondary OCDBT-listing error.
    spec = _qwen3_cooldown_spec(region="us-central2")
    resolved = _resolve(spec, region="us-central2")
    staged = spec.mode.resume.staged_checkpoint_path
    exists, list_ = fake_gcs(spec.data_manifest_uri, staged)  # staged exists but no artifacts

    sentinel: list[str] = []

    def fake_list(_):
        sentinel.append("called")
        return ()

    report = preflight(
        resolved,
        exists=exists,
        list_=list_,
        list_ocdbt_keys=fake_list,
    )
    assert not report.ok
    assert sentinel == [], "list_ocdbt_keys should not be called when artifacts are missing"
    assert any("missing manifest.ocdbt" in f for f in report.failures)


def test_preflight_fails_when_cooldown_model_config_has_no_type():
    spec = make_cooldown_spec(region="us-central2")
    # Forcibly strip the 'type' discriminator after construction to simulate
    # an old launcher that forgot to set it. spec validation will catch this
    # first; preflight's secondary guard is defense-in-depth.
    no_type_cfg = {k: v for k, v in spec.model_config.items() if k != "type"}
    spec = type(spec)(
        **{**{f.name: getattr(spec, f.name) for f in spec.__dataclass_fields__.values()}, "model_config": no_type_cfg}
    )
    resolved = _resolve(spec, region="us-central2")
    staged = spec.mode.resume.staged_checkpoint_path
    exists, list_ = fake_gcs(
        spec.data_manifest_uri,
        staged,
        f"{staged}/manifest.ocdbt",
        f"{staged}/metadata.json",
        f"{staged}/d",
    )
    report = preflight(
        resolved,
        exists=exists,
        list_=list_,
        list_ocdbt_keys=lambda _: _qwen3_ish_keys(FAKE_1E22.num_layers),
    )
    assert not report.ok
    assert any("missing the 'type' discriminator" in f for f in report.failures), report.failures
