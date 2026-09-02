# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from levanter.main.train_lm import TrainLmConfig
from marin.execution.lazy import materialized_config
from marin.training.training import TrainLmOnPodConfig

from experiments.domain_phase_mix import launch_starcoder_wsd80_coupled_onset_refinement_confirmation as launcher
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_starcoder_wsd80_coupled_onset_refinement_confirmation_20260901 as design,
)


def test_design_reproduces_frozen_manifest() -> None:
    manifest = json.loads(gzip.decompress(design.OUTPUT_PATH.read_bytes()))
    claimed_hash = manifest.pop("design_sha256")

    assert design.lr_design.canonical_sha256(manifest) == claimed_hash == launcher.EXPECTED_DESIGN_SHA256
    assert design.build_payload() == manifest


def test_wandb_panel_tag_respects_runtime_limit() -> None:
    assert 0 < len(launcher.CENTRAL2_V4_DEPLOYMENT.panel_tag) <= launcher.MAX_WANDB_TAG_LENGTH


def test_adaptive_and_confirmatory_inventories_are_disjoint() -> None:
    payload, rows = launcher.load_design()
    adaptive = [row for row in rows if row.stage == "bayesian_refinement_discovery"]
    confirmation = [row for row in rows if row.stage == "fresh_confirmation"]

    assert len(adaptive) == 24
    assert len(confirmation) == 72
    assert {row.data_seed for row in adaptive} == {payload["discovery_seed"]}
    assert {row.data_seed for row in confirmation} == set(payload["confirmation_seeds"])
    assert payload["discovery_seed"] not in payload["confirmation_seeds"]
    assert all(row.acquisition is not None and row.selection_class == "eligible_untied" for row in adaptive)
    assert all(row.acquisition is None for row in confirmation)
    assert all(row.boundary_step == row.optimizer["decay_onset_step"] for row in rows)
    assert "paired cross-arm differences" in payload["confirmation_contract"]["E1_primary"]["test"]
    assert "never drop, replace, or reselect" in payload["completeness_contract"]["failure_rule"]
    assert payload["checkpoint_contract"]["all_rows"] == "terminal permanent checkpoint only"

    for arm in payload["bayesian_refinement"]["arms"].values():
        assert arm["summary"]["local_basin_observations"] >= design.MINIMUM_LOCAL_POINTS
        assert max(row["predicted_sd_bpb"] for row in arm["selected"]) < 0.005


def test_materialized_configs_preserve_policy_and_seed(monkeypatch: Any) -> None:
    def load_cache(_cls: type[launcher.lr_only.TokenizedCache], path: str) -> launcher.lr_only.TokenizedCache:
        cache = launcher.lr_only.TokenizedCache(path=path)
        cache.__dict__["record"] = SimpleNamespace(
            config={
                "format": {"text_key": "text"},
                "tags": [],
                "tokenizer": "meta-llama/Meta-Llama-3.1-8B",
            },
            source=None,
        )
        return cache

    monkeypatch.setattr(launcher.lr_only.TokenizedCache, "raw_load", classmethod(load_cache))
    _, all_rows = launcher.load_design()
    selected = []
    for arm_id in sorted(launcher.EXPECTED_ARM_IDS):
        for stage in launcher.EXPECTED_STAGE_COUNTS:
            selected.append(next(row for row in all_rows if row.arm_id == arm_id and row.stage == stage))
    rows, steps = launcher.build_training_steps(selected_runs=frozenset(row.run_name for row in selected))

    for row, step in zip(rows, steps, strict=True):
        pod_config = materialized_config(step, launcher.CENTRAL2_V4_DEPLOYMENT.marin_prefix)
        assert isinstance(pod_config, TrainLmOnPodConfig)
        train_config = pod_config.train_config
        assert isinstance(train_config, TrainLmConfig)
        weights = train_config.data.train_weights
        assert isinstance(weights, list)
        assert [boundary for boundary, _ in weights] == [0, row.boundary_step]
        assert train_config.optimizer.decay == row.total_steps - row.boundary_step
        assert weights[0][1]["dolma/starcoder"] == row.phase_0_starcoder
        assert weights[1][1]["dolma/starcoder"] == row.phase_1_starcoder
        assert train_config.data_seed == row.data_seed
        assert train_config.trainer.seed == row.trainer_seed
        assert train_config.trainer.checkpointer.keep == launcher._keep_policy(row)

    assert launcher.audit_materialized_runtime_configs(rows, steps) == 6


def test_release_freeze_and_load_round_trip(tmp_path: Path) -> None:
    review_path = tmp_path / "review.md"
    review_path.write_text("No blockers.\nAPPROVE\n")
    deployment = replace(launcher.CENTRAL2_V4_DEPLOYMENT, output_dir=tmp_path, cc_review_path=review_path)

    release = launcher._freeze_release(deployment)

    assert launcher._load_release(deployment) == release
