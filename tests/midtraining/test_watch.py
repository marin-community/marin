# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from marin.midtraining.spec import resolve_midtrain_spec
from marin.midtraining.watch import evaluate_startup

from experiments.midtrain_specs.delphi_small_cpt_k020 import build_spec
from tests.midtraining._fixtures import (
    make_cooldown_spec,
    make_cpt_spec,
    make_data_manifest,
)


def _resolve(spec):
    manifest = make_data_manifest(region=spec.run.output_region)
    with patch("marin.midtraining.spec.load_data_manifest", return_value=manifest):
        return resolve_midtrain_spec(spec)


def test_cpt_startup_proof_passes_on_expected_lines():
    resolved = _resolve(make_cpt_spec())
    log = [
        "Using output path gs://marin-us-east5/checkpoints/cpt-cell-a001",
        "Using run ID cpt-cell-a001",
        "No checkpoints found in [...]",
        "Loading checkpoint from gs://marin-us-central2/...",
        "checkpoint_init_mode=model_only",
    ]
    proof = evaluate_startup(resolved, log)
    assert proof.healthy, proof


def test_hf_cpt_startup_proof_passes_on_hf_init_line():
    resolved = resolve_midtrain_spec(build_spec(base_key="3e18", mix="p33m67", lr_factor=0.5))
    log = [
        "Using output path gs://marin-us-east5/checkpoints/delphi-3e18-p33m67-k0p20-lr50-a001",
        "Using run ID delphi-3e18-p33m67-k0p20-lr50-a001",
        "No checkpoints found in [...]",
        "No training checkpoint found. Initializing model from HF checkpoint 'marin-community/delphi-3e18'",
    ]
    proof = evaluate_startup(resolved, log)
    assert proof.healthy, proof


def test_cpt_resume_startup_proof_uses_resume_lines():
    resolved = _resolve(make_cpt_spec(expected_min_step=1_000))
    log = [
        "Using output path gs://marin-us-east5/checkpoints/cpt-cell-a001",
        "Using run ID cpt-cell-a001",
        "Discovered latest checkpoint at gs://marin-us-east5/checkpoints/cpt-cell-a001/checkpoints/step-1500",
        "Loading checkpoint from gs://marin-us-east5/checkpoints/cpt-cell-a001/checkpoints/step-1500",
        "Resuming training from step 1500",
    ]
    proof = evaluate_startup(resolved, log)
    assert proof.healthy, proof
    assert proof.detected_step == 1_500


def test_cooldown_startup_refuses_cpt_lines():
    resolved = _resolve(make_cooldown_spec())
    log = [
        "Using output path gs://marin-us-east5/checkpoints/cd-step30000-a001",
        "Using run ID cd-step30000-a001",
        "No checkpoints found in [...]",
    ]
    proof = evaluate_startup(resolved, log)
    assert not proof.healthy
    assert "No checkpoints found" in proof.forbidden_seen


def test_cooldown_startup_passes_on_resume_line():
    resolved = _resolve(make_cooldown_spec(resume_step=30_000))
    log = [
        "Using output path gs://marin-us-east5/checkpoints/cd-step30000-a001",
        "Using run ID cd-step30000-a001",
        "Discovered latest checkpoint at gs://marin-us-east5/checkpoints/cd-step30000-a001/checkpoints/step-30000",
        "Loading checkpoint from gs://marin-us-east5/checkpoints/cd-step30000-a001/checkpoints/step-30000",
        "Resuming training from step 30000",
    ]
    proof = evaluate_startup(resolved, log)
    assert proof.healthy, proof
    assert proof.detected_step == 30_000


def test_wandb_step_regression_marks_unhealthy():
    resolved = _resolve(make_cpt_spec())
    log = [
        "Using output path ...",
        "Using run ID cpt-cell-a001",
        "No checkpoints found",
        "Loading checkpoint from ...",
        "checkpoint_init_mode=model_only",
        "Step 1 is less than current W&B step 4768",
    ]
    proof = evaluate_startup(resolved, log)
    assert not proof.healthy
    assert "W&B step regression" in proof.forbidden_seen
