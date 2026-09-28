# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pytest

from experiments.domain_phase_mix import build_proportional_controllability_eval_candidates as candidate_builder
from experiments.domain_phase_mix.launch_proportional_controllability_300m import (
    DEFAULT_EVAL_DATASETS_CACHE_PATH,
    DEFAULT_TPU_REGION,
    DEFAULT_TPU_TYPE,
    DEFAULT_TPU_ZONE,
    build_launch_artifacts,
    build_run_specs,
)


def _candidate_registry_rows(*, checkpoint_root: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(candidate_builder.EXPECTED_ROWS):
        rows.append(
            {
                "registry_id": f"proportional_controllability_300m_6b:run_{index:05d}",
                "family": candidate_builder.FAMILY,
                "scale": "300m_6b",
                "run_name": f"run_{index:05d}",
                "logical_status": "completed",
                "source_experiment": "source",
                "cohort": "cohort",
                "checkpoint_root": f"{checkpoint_root}/run_{index:05d}",
                "target_final_checkpoint_step": 22887,
                "intervention_id": f"intervention_{index:05d}",
                "intervention_type": "domain_deletion",
                "target_domain": "domain",
                "direction_id": None,
                "direction_type": None,
                "tilt_sign": None,
                "alpha": None,
                "base_mass": 0.1,
                "tv_distance": 0.1,
                "renormalizer": "renormalizer",
                "objective_metric_value": 1.0,
            }
        )
    return rows


def test_candidate_builder_refreshes_right_count_dead_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stale_registry = pd.DataFrame(_candidate_registry_rows(checkpoint_root="gs://marin-us-east5/stale"))
    registry_csv = tmp_path / "logical_runs.csv"
    stale_registry.to_csv(registry_csv, index=False)
    refreshed = pd.DataFrame(_candidate_registry_rows(checkpoint_root="gs://marin-us-east5/shortened"))

    monkeypatch.setattr(
        candidate_builder,
        "_has_exact_hf_checkpoint",
        lambda checkpoint_root, expected_step: str(checkpoint_root).startswith("gs://marin-us-east5/shortened"),
    )
    monkeypatch.setattr(candidate_builder, "_proportional_controllability_rows", lambda: (refreshed, []))

    frame = candidate_builder.build_candidate_frame(run_registry_csv=registry_csv, allow_incomplete=False)

    assert len(frame) == candidate_builder.EXPECTED_ROWS
    assert frame["checkpoint_root"].str.startswith("gs://marin-us-east5/shortened").all()


def test_candidate_builder_rejects_incomplete_exact_hf_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = pd.DataFrame(_candidate_registry_rows(checkpoint_root="gs://marin-us-east5/shortened"))
    registry_csv = tmp_path / "logical_runs.csv"
    registry.to_csv(registry_csv, index=False)

    def has_exact_hf_checkpoint(checkpoint_root: object, expected_step: object) -> bool:
        del expected_step
        return not str(checkpoint_root).endswith("run_00000")

    monkeypatch.setattr(candidate_builder, "_has_exact_hf_checkpoint", has_exact_hf_checkpoint)
    monkeypatch.setattr(candidate_builder, "_proportional_controllability_rows", lambda: (registry, []))

    with pytest.raises(ValueError, match="Controllability rows are not target-ready"):
        candidate_builder.build_candidate_frame(run_registry_csv=registry_csv, allow_incomplete=False)


def test_subset_retry_filters_training_steps_by_run_name():
    full_specs = build_run_specs()
    selected_run_names = (full_specs[0].run_name, full_specs[-1].run_name)

    artifacts = build_launch_artifacts(
        base_name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_proportional_controllability_300m",
        tpu_type=DEFAULT_TPU_TYPE,
        tpu_regions=(DEFAULT_TPU_REGION,),
        tpu_zone=DEFAULT_TPU_ZONE,
        eval_datasets_cache_path=DEFAULT_EVAL_DATASETS_CACHE_PATH,
        include_eval_harness=False,
        only_run_names=selected_run_names,
    )

    assert [run_spec.run_name for run_spec in artifacts.run_specs] == list(selected_run_names)
    assert [intervention.run_name for intervention in artifacts.interventions] == list(selected_run_names)
    assert len(artifacts.training_steps) == len(selected_run_names)
    assert all(run_spec.run_id in {full_specs[0].run_id, full_specs[-1].run_id} for run_spec in artifacts.run_specs)


def test_subset_retry_rejects_unknown_run_name():
    with pytest.raises(ValueError, match="Unknown proportional-controllability run names"):
        build_launch_artifacts(
            base_name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_proportional_controllability_300m",
            tpu_type=DEFAULT_TPU_TYPE,
            tpu_regions=(DEFAULT_TPU_REGION,),
            tpu_zone=DEFAULT_TPU_ZONE,
            eval_datasets_cache_path=DEFAULT_EVAL_DATASETS_CACHE_PATH,
            include_eval_harness=False,
            only_run_names=("pctrl_missing",),
        )
