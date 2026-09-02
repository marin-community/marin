# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from levanter.main.train_lm import TrainLmConfig
from marin.execution.lazy import materialized_config
from marin.training.training import TrainLmOnPodConfig

from experiments.domain_phase_mix import launch_starcoder_wsd80_coupled_onset_dense_surfaces as launcher
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_starcoder_wsd80_coupled_onset_dense_surfaces_20260830 as design,
)


def test_design_reproduces_frozen_manifest() -> None:
    manifest = json.loads(gzip.decompress(design.OUTPUT_PATH.read_bytes()))
    claimed_hash = manifest.pop("design_sha256")

    assert design.lr_only_design.canonical_sha256(manifest) == claimed_hash == launcher.EXPECTED_DESIGN_SHA256
    assert design.build_payload() == manifest


def test_coupled_arms_preserve_common_exposure_geometry() -> None:
    payload, rows = launcher.load_design()

    assert len(rows) == 375
    assert payload["stage_counts"] == {"surface_discovery": 375}
    assert {row.data_seed for row in rows} == {payload["discovery_seed"]}
    assert all(row.boundary_step == row.optimizer["decay_onset_step"] for row in rows)
    assert {row.arm_id: row.boundary_step for row in rows} == {
        "coupled_0p60": 16_944,
        "coupled_0p80": 22_608,
        "coupled_0p90": 25_424,
    }

    source_coordinates = {coordinate["coordinate_id"]: coordinate for coordinate in payload["coordinates"]}
    for coordinate_id in source_coordinates:
        coordinate_rows = [row for row in rows if row.coordinate_id == coordinate_id]
        assert len(coordinate_rows) == 3
        assert len({row.aggregate_starcoder for row in coordinate_rows}) == 1
        assert len({row.normalized_fiber_position for row in coordinate_rows}) == 1
        for row in coordinate_rows:
            assert 0.0 <= row.phase_0_starcoder <= 1.0
            assert 0.0 <= row.phase_1_starcoder <= 1.0
            realized_aggregate = (
                row.realized_onset_fraction * row.phase_0_starcoder
                + (1.0 - row.realized_onset_fraction) * row.phase_1_starcoder
            )
            np.testing.assert_allclose(
                realized_aggregate,
                row.aggregate_starcoder,
                atol=1e-12,
            )

        reference = next(row for row in coordinate_rows if row.arm_id == "coupled_0p80")
        source = source_coordinates[coordinate_id]
        np.testing.assert_allclose(reference.phase_0_starcoder, source["phase_0_starcoder"], atol=1e-12)
        np.testing.assert_allclose(reference.phase_1_starcoder, source["phase_1_starcoder"], atol=1e-12)


def test_materialized_configs_couple_data_and_optimizer_onsets(monkeypatch: Any) -> None:
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
    selected = tuple(row for row in all_rows if row.coordinate_id == "c020")
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
        assert row.boundary_step == row.optimizer["decay_onset_step"]
        assert weights[0][1]["dolma/starcoder"] == row.phase_0_starcoder
        assert weights[1][1]["dolma/starcoder"] == row.phase_1_starcoder
        assert train_config.data_seed == row.data_seed
        assert train_config.trainer.seed == row.trainer_seed

    assert launcher.audit_materialized_runtime_configs(rows, steps) == 3


def test_release_freeze_and_load_round_trip(tmp_path: Path) -> None:
    deployment = replace(launcher.CENTRAL2_V4_DEPLOYMENT, output_dir=tmp_path)

    release = launcher._freeze_release(deployment)

    assert launcher._load_release(deployment) == release
