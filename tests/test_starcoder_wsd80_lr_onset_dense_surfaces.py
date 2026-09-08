# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from levanter.main.train_lm import TrainLmConfig
from marin.execution.lazy import materialized_config
from marin.training.training import TrainLmOnPodConfig

from experiments.domain_phase_mix import launch_starcoder_wsd80_lr_onset_dense_surfaces as launcher
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_starcoder_wsd80_lr_onset_dense_surfaces_20260825 as design,
)


def test_design_reproduces_frozen_manifest() -> None:
    manifest = json.loads(gzip.decompress(design.OUTPUT_PATH.read_bytes()))
    claimed_hash = manifest.pop("design_sha256")

    assert design.canonical_sha256(manifest) == claimed_hash == launcher.EXPECTED_DESIGN_SHA256
    assert design.build_payload() == manifest


def test_design_preserves_dense_coverage_and_disjoint_confirmation_seeds() -> None:
    payload, rows = launcher.load_design()
    discovery = [row for row in rows if row.stage == "surface_discovery"]
    spine = [row for row in rows if row.stage == "primary_spine"]
    replication = [row for row in rows if row.stage == "replay_replication"]

    assert len(rows) == 644
    assert len(discovery) == 500
    assert len(spine) == 80
    assert len(replication) == 64
    assert {row.data_seed for row in discovery} == {payload["discovery_seed"]}
    assert {row.data_seed for row in spine} == set(payload["confirmation_seeds"])
    assert {row.data_seed for row in replication} == set(payload["confirmation_seeds"])
    assert payload["discovery_seed"] not in payload["confirmation_seeds"]
    assert not set(payload["confirmation_seeds"]) & set(payload["historical_confirmation"]["seeds"])

    for arm_id in launcher.EXPECTED_MAIN_ARM_IDS:
        arm_rows = [row for row in discovery if row.arm_id == arm_id]
        assert len(arm_rows) == 125
        assert len({(row.phase_0_starcoder, row.phase_1_starcoder) for row in arm_rows}) == 125
    assert not [row for row in discovery if row.arm_role == "lr_integral_sensitivity_only"]
    assert {row.support_id for row in discovery} == {"m100"}
    assert {row.support_id for row in replication} == {"m200"}


def test_schedule_treatment_moves_lr_but_not_phase_boundary() -> None:
    _, rows = launcher.load_design(selected_stage="surface_discovery")
    representatives = {row.arm_id: row for row in rows if row.coordinate_id == "c000"}
    earliest_onset = representatives["decay_0p60"].optimizer["decay_onset_step"]
    historical = launcher._schedule_vector(representatives["decay_0p80"])

    for arm_id, row in representatives.items():
        observed = launcher._schedule_vector(row)
        assert row.boundary_step == launcher.EXPECTED_BOUNDARY_STEP
        np.testing.assert_array_equal(observed[:earliest_onset], historical[:earliest_onset])
        if arm_id != "decay_0p80":
            assert not np.array_equal(observed, historical)

    _, spine_rows = launcher.load_design(
        selected_stage="primary_spine",
        selected_runs=frozenset(
            {
                "lrod_sp_m100_0p60_c109_s0831",
                "lrod_sp_m100_0p80_area_match_0p60_c109_s0831",
            }
        ),
    )
    profiles = {row.arm_id: row.optimizer for row in spine_rows}
    assert profiles["decay_0p80_area_match_0p60"]["normalized_lr_integral"] == pytest.approx(
        profiles["decay_0p60"]["normalized_lr_integral"], abs=1e-6
    )
    assert spine_rows[1].peak_lr_multiplier != 1.0
    assert spine_rows[1].arm_role == "lr_integral_sensitivity_only"

    no_decay = launcher._schedule_vector(representatives["no_decay"])
    np.testing.assert_array_equal(
        no_decay[launcher.EXPECTED_WARMUP_STEPS :],
        np.full_like(no_decay[launcher.EXPECTED_WARMUP_STEPS :], no_decay[launcher.EXPECTED_WARMUP_STEPS]),
    )
    phase_1_integrals = [
        representatives[arm_id].optimizer["normalized_phase_1_lr_integral"]
        for arm_id in ("decay_0p60", "decay_0p80", "decay_0p90", "no_decay")
    ]
    assert phase_1_integrals == sorted(phase_1_integrals)


def test_parent_schedule_audit_uses_cpu_with_auto_discovery_and_tpu_advertised() -> None:
    script = """
import jax

from experiments.domain_phase_mix import launch_starcoder_wsd80_lr_onset_dense_surfaces as launcher

launcher._configure_parent_jax()
_, rows = launcher.load_design(selected_stage="surface_discovery")
schedule = launcher._schedule_vector(rows[0])
assert schedule.shape == (launcher.EXPECTED_TOTAL_STEPS,)
print(jax.default_backend(), schedule.shape[0])
"""
    environments = []
    auto_discovery = dict(os.environ)
    auto_discovery.pop("JAX_PLATFORMS", None)
    environments.append(auto_discovery)
    environments.append({**os.environ, "JAX_PLATFORMS": "tpu"})

    for environment in environments:
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == f"cpu {launcher.EXPECTED_TOTAL_STEPS}"


def test_materialized_optimizer_arms_share_exact_data_contract(monkeypatch: Any) -> None:
    def load_cache(_cls: type[launcher.TokenizedCache], path: str) -> launcher.TokenizedCache:
        cache = launcher.TokenizedCache(path=path)
        cache.__dict__["record"] = SimpleNamespace(
            config={
                "format": {"text_key": "text"},
                "tags": [],
                "tokenizer": "meta-llama/Meta-Llama-3.1-8B",
            },
            source=None,
        )
        return cache

    monkeypatch.setattr(launcher.TokenizedCache, "raw_load", classmethod(load_cache))
    _, all_rows = launcher.load_design(selected_stage="primary_spine")
    selected = tuple(
        row for row in all_rows if row.coordinate_id == "c109" and row.data_seed == design.CONFIRMATION_SEEDS[0]
    )
    run_names = frozenset(row.run_name for row in selected)
    rows, steps = launcher.build_training_steps(selected_stage="primary_spine", selected_runs=run_names)
    contracts = []
    schedules = {}
    for row, step in zip(rows, steps, strict=True):
        pod_config = materialized_config(step, launcher.CENTRAL1_V5P_DEPLOYMENT.marin_prefix)
        assert isinstance(pod_config, TrainLmOnPodConfig)
        train_config = pod_config.train_config
        assert isinstance(train_config, TrainLmConfig)
        contracts.append(launcher._data_contract(train_config))
        schedules[row.arm_id] = (
            train_config.optimizer.learning_rate,
            train_config.optimizer.decay,
            train_config.optimizer.min_lr_ratio,
        )
        assert train_config.data.train_holdout_sequences is None
        assert train_config.data.max_train_batches == {"dolma/starcoder": row.starcoder_support_batches}
        assert train_config.trainer.num_train_steps == row.total_steps
        assert train_config.optimizer.warmup == launcher.EXPECTED_WARMUP_STEPS
        train_weights = train_config.data.train_weights
        assert isinstance(train_weights, list)
        assert [boundary for boundary, _ in train_weights] == [0, launcher.EXPECTED_BOUNDARY_STEP]
        assert train_config.trainer.checkpointer.keep == launcher._keep_policy(row)
        assert isinstance(train_config.trainer.checkpointer.save_interval, timedelta)
        assert train_config.trainer.checkpointer.save_interval == launcher.TEMPORARY_CHECKPOINT_INTERVAL

    assert all(contract == contracts[0] for contract in contracts[1:])
    assert len(set(schedules.values())) == len(launcher.EXPECTED_ARM_IDS)
    assert launcher.audit_materialized_runtime_configs(rows, steps) == len(rows)


def test_primary_and_replication_fixed_coordinates_remain_wrapped_and_eligible() -> None:
    payload, rows = launcher.load_design()
    contracts = payload["analysis_contract"]
    expected = {
        ("m100", contracts["p1_primary"]["tied_coordinate_id"]): "tied",
        ("m100", contracts["p1_primary"]["untied_coordinate_id"]): "eligible_untied",
        ("m200", contracts["replication"]["tied_coordinate_id"]): "tied",
        ("m200", contracts["replication"]["untied_coordinate_id"]): "eligible_untied",
    }
    for identity, role in expected.items():
        matching = [row for row in rows if (row.support_id, row.coordinate_id) == identity]
        assert matching
        assert {row.selection_class for row in matching} == {role}
        assert all(row.starcoder_support_wraps for row in matching)

    provenance = {
        (row["support_id"], row["policy_class"]): row["coordinate_id"]
        for row in payload["historical_confirmation"]["fixed_policy_provenance"]
    }
    assert provenance == {
        ("m100", "tied"): "c109",
        ("m100", "untied"): "c020",
        ("m200", "tied"): "c079",
        ("m200", "untied"): "c011",
    }


def test_release_freeze_requires_passing_review(tmp_path: Path) -> None:
    blocked_review = tmp_path / "review.md"
    blocked_review.write_text("VERDICT: BLOCK\n", encoding="utf-8")
    deployment = replace(
        launcher.CENTRAL2_V4_DEPLOYMENT,
        output_dir=tmp_path,
        cc_review_path=blocked_review,
    )

    with pytest.raises(ValueError, match="VERDICT: PASS"):
        launcher._freeze_release(deployment)


def test_cache_object_contract_excludes_only_bookkeeping(monkeypatch: Any) -> None:
    class FakeFilesystem:
        def find(self, root: str, *, detail: bool) -> dict[str, dict[str, str | int]]:
            assert detail
            return {
                f"{root}/input_ids/data/c/0": {
                    "size": 11,
                    "crc32c": "crc-a",
                    "md5Hash": "md5-a",
                },
                f"{root}/input_ids/data/c/1": {
                    "size": 13,
                    "crc32c": "crc-b",
                    "md5Hash": "md5-b",
                },
                f"{root}/shard_ledger.json": {"size": 17},
                f"{root}/shard_ledger.json.bak": {"size": 19},
                f"{root}/___temp/00/input_ids/data/c/0": {"size": 23},
            }

    monkeypatch.setattr(launcher.gcsfs, "GCSFileSystem", lambda token: FakeFilesystem())
    expected_rows = [
        {"path": "input_ids/data/c/0", "size": 11, "crc32c": "crc-a", "md5": "md5-a"},
        {"path": "input_ids/data/c/1", "size": 13, "crc32c": "crc-b", "md5": "md5-b"},
    ]

    assert launcher._cache_object_contract("gs://bucket/cache") == {
        "object_count": 2,
        "total_bytes": 24,
        "metadata_sha256": launcher._canonical_sha256(expected_rows),
    }


def test_central2_deployment_preserves_rows_and_changes_runtime_placement(monkeypatch: Any) -> None:
    def load_cache(_cls: type[launcher.TokenizedCache], path: str) -> launcher.TokenizedCache:
        cache = launcher.TokenizedCache(path=path)
        cache.__dict__["record"] = SimpleNamespace(
            config={
                "format": {"text_key": "text"},
                "tags": [],
                "tokenizer": "meta-llama/Meta-Llama-3.1-8B",
            },
            source=None,
        )
        return cache

    monkeypatch.setattr(launcher.TokenizedCache, "raw_load", classmethod(load_cache))
    run_name = "lrod_ds_m100_0p80_c109_s0711"
    selected_runs = frozenset({run_name})
    central1_rows, central1_steps = launcher.build_training_steps(
        deployment=launcher.CENTRAL1_V5P_DEPLOYMENT,
        selected_stage="surface_discovery",
        selected_runs=selected_runs,
    )
    central2_rows, central2_steps = launcher.build_training_steps(
        deployment=launcher.CENTRAL2_V4_DEPLOYMENT,
        selected_stage="surface_discovery",
        selected_runs=selected_runs,
    )

    assert central2_rows == central1_rows
    central1 = materialized_config(central1_steps[0], launcher.CENTRAL1_V5P_DEPLOYMENT.marin_prefix)
    central2 = materialized_config(central2_steps[0], launcher.CENTRAL2_V4_DEPLOYMENT.marin_prefix)
    assert isinstance(central1, TrainLmOnPodConfig)
    assert isinstance(central2, TrainLmOnPodConfig)
    assert launcher._data_contract(central2.train_config) == launcher._data_contract(central1.train_config)
    assert central2.train_config.optimizer == central1.train_config.optimizer
    assert central2.resources.device.variant == "v4-8"
    assert central2.resources.regions == ("us-central2",)
    assert central2.resources.zone == "us-central2-b"
    assert central2_steps[0].name != central1_steps[0].name


def test_source_design_placement_remains_frozen() -> None:
    payload, _ = launcher.load_design()

    assert payload["placement"] == launcher.SOURCE_PLACEMENT
