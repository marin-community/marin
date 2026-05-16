# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preflight + staging tests using callable injection (no FakeGcsClient class)."""

from unittest.mock import patch

import pytest
from marin.midtraining.preflight import (
    CrossRegionCopyPolicy,
    fake_gcs,
    preflight,
    stage_cooldown_checkpoint,
)
from marin.midtraining.spec import MidtrainSpec, resolve_midtrain_spec

from tests.midtraining._fixtures import (
    FAKE_1E21,
    make_cooldown_spec,
    make_cpt_spec,
    make_data_manifest,
)


def _resolve(spec: MidtrainSpec, region: str = "us-east5"):
    manifest = make_data_manifest(region=region)
    with patch("marin.midtraining.spec.load_data_manifest", return_value=manifest):
        return resolve_midtrain_spec(spec)


def test_fresh_cpt_refuses_existing_checkpoint_namespace():
    spec = make_cpt_spec()
    resolved = _resolve(spec)
    exists, list_ = fake_gcs(
        spec.data_manifest_uri,
        FAKE_1E21.verified_checkpoint_path,
        f"{spec.run.permanent_checkpoints_uri}/step-100",
    )
    report = preflight(resolved, exists=exists, list_=list_)
    assert not report.ok
    assert any("permanent checkpoints already exist" in f for f in report.failures)


def test_fresh_cpt_refuses_existing_manifest():
    spec = make_cpt_spec()
    resolved = _resolve(spec)
    exists, list_ = fake_gcs(
        spec.data_manifest_uri,
        FAKE_1E21.verified_checkpoint_path,
        spec.run.manifest_uri,
    )
    report = preflight(resolved, exists=exists, list_=list_)
    assert not report.ok
    assert any("manifest already at" in f for f in report.failures)


def test_resume_requires_checkpoint_floor():
    spec = make_cpt_spec(expected_min_step=4_500)
    resolved = _resolve(spec)
    exists, list_ = fake_gcs(
        spec.data_manifest_uri,
        FAKE_1E21.verified_checkpoint_path,
        f"{spec.run.permanent_checkpoints_uri}/step-1000",
    )
    report = preflight(resolved, exists=exists, list_=list_)
    assert not report.ok
    assert any("below expected_min_step" in f for f in report.failures)


def test_resume_accepts_above_floor():
    spec = make_cpt_spec(expected_min_step=1_000)
    resolved = _resolve(spec)
    exists, list_ = fake_gcs(
        spec.data_manifest_uri,
        FAKE_1E21.verified_checkpoint_path,
        f"{spec.run.permanent_checkpoints_uri}/step-1500",
    )
    report = preflight(resolved, exists=exists, list_=list_)
    assert report.ok, report.failures


def test_cooldown_refuses_without_staged_checkpoint():
    spec = make_cooldown_spec()
    resolved = _resolve(spec)
    exists, list_ = fake_gcs(spec.data_manifest_uri)
    report = preflight(resolved, exists=exists, list_=list_)
    assert not report.ok
    assert any("Cooldown staged checkpoint not found" in f for f in report.failures)


def test_cooldown_refuses_cross_region_without_explicit_authorization():
    spec = make_cooldown_spec(region="us-east5")
    resolved = _resolve(spec, region="us-east5")
    staged = spec.mode.resume.staged_checkpoint_path
    exists, list_ = fake_gcs(
        spec.data_manifest_uri,
        staged,
        f"{staged}/manifest.ocdbt",
        f"{staged}/metadata.json",
        f"{staged}/d",
    )
    report = preflight(resolved, cross_region_copy=CrossRegionCopyPolicy(), exists=exists, list_=list_)
    assert not report.ok
    assert any("Cooldown source region" in f for f in report.failures)


def test_cooldown_accepts_cross_region_with_explicit_authorization():
    spec = make_cooldown_spec(region="us-east5")
    resolved = _resolve(spec, region="us-east5")
    staged = spec.mode.resume.staged_checkpoint_path
    exists, list_ = fake_gcs(
        spec.data_manifest_uri,
        staged,
        f"{staged}/manifest.ocdbt",
        f"{staged}/metadata.json",
        f"{staged}/d",
    )
    policy = CrossRegionCopyPolicy(allowed=True, budget_gb=200, reason="cooldown stage from us-central2 base")
    report = preflight(resolved, cross_region_copy=policy, exists=exists, list_=list_)
    assert all("Cooldown source region" not in f for f in report.failures), report.failures


def test_stage_cooldown_refuses_cross_region_without_policy():
    spec = make_cooldown_spec()
    exists, _ = fake_gcs()
    with pytest.raises(ValueError, match="Refusing cross-region cooldown stage"):
        stage_cooldown_checkpoint(spec, exists=exists, dry_run=True)


def test_stage_cooldown_dry_run_reports_intent_when_authorized():
    spec = make_cooldown_spec()
    exists, _ = fake_gcs()
    record = stage_cooldown_checkpoint(
        spec,
        cross_region_copy=CrossRegionCopyPolicy(allowed=True, budget_gb=200, reason="cooldown stage from base"),
        exists=exists,
        dry_run=True,
    )
    assert record.cross_region_copy
    assert record.bytes_copied == 0
