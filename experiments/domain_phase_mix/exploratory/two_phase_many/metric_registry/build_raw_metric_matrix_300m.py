# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "pandas"]
# ///
"""Build a raw wide metric matrix for the 300M/6B mixture swarm.

Rows are trained models / mixtures. Columns include provenance, phase mixture
weights, exposure-average mixture weights, and every raw metric currently
available through the overlaid metric registry.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.build_eval_signal_to_noise_all_metrics_300m import (
    _default_extra_results_csvs,
    _load_noise_frame,
    _load_signal_frame,
    _metric_columns,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "raw_metric_matrix_300m"
EXPOSURE_PHASE0_FRACTION = 0.8
PHASE_WEIGHT_TOLERANCE = 1e-6
STRATIFIED_RUN_NAME = "baseline_stratified"
NOISE_SOURCE_RUN_NAME = "run_00097"
NOISE_TRAINER_SEED_RE = re.compile(r"trainer_seed_(?P<trainer_seed>\d+)")
NOISE_SIMULATED_EPOCH_SUBSET_SEED = 97
FIXED_SUBSET_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_300m_6b_fixed_subset"
VARIABLE_SUBSET_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_300m_6b_variable_subset"
PROPORTIONAL_VARIABLE_SUBSET_SOURCE_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/ngd3dm2_proportional_variable_subset_noise_300m_6b"
)
NOISE_SOURCE_RUN_ID = 97
PROPORTIONAL_NOISE_ANCHOR_RUN_NAME = "baseline_proportional"
REQUIRED_PROVENANCE_COLUMNS = (
    "registry_run_key",
    "run_name",
    "run_id",
    "scale",
    "cohort",
    "source_cohort",
    "source_experiment",
    "checkpoint_root",
    "status",
    "row_kind",
)
REQUIRED_NOISE_PROVENANCE_COLUMNS = (
    "noise_subset_mode",
    "noise_source_run_name",
    "noise_trainer_seed",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _phase_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith(prefix))


def _active_domain_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return [
        column
        for column in _phase_columns(frame, prefix)
        if pd.to_numeric(frame[column], errors="coerce").fillna(0.0).max() > 0.0
    ]


def _validate_phase_weights(frame: pd.DataFrame, phase0_columns: list[str], phase1_columns: list[str]) -> None:
    for label, columns in (("phase_0", phase0_columns), ("phase_1", phase1_columns)):
        sums = frame[columns].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        bad = frame.loc[(sums - 1.0).abs() > PHASE_WEIGHT_TOLERANCE, ["run_name"]].copy()
        if not bad.empty:
            bad["weight_sum"] = sums.loc[bad.index].to_numpy()
            raise ValueError(f"{label} weights do not sum to 1:\n{bad.to_string(index=False)}")


def _has_phase_weights(frame: pd.DataFrame) -> pd.Series:
    phase0_columns = _phase_columns(frame, "phase_0_")
    phase1_columns = _phase_columns(frame, "phase_1_")
    return frame[phase0_columns].sum(axis=1).gt(0.0) & frame[phase1_columns].sum(axis=1).gt(0.0)


def _hydrate_known_phase_weights(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fill phase weights for known baselines whose registry rows lack them."""
    out = frame.copy()
    original_has_weights = _has_phase_weights(out)
    source_rows = out[original_has_weights].copy()
    domains = [column.removeprefix("phase_0_") for column in _active_domain_columns(source_rows, "phase_0_")]
    if not domains:
        raise ValueError("Cannot hydrate known baselines without any observed domain columns")

    hydrated_rows: list[dict[str, Any]] = []
    stratified_mask = out["run_name"].eq(STRATIFIED_RUN_NAME) & ~original_has_weights
    if stratified_mask.any():
        if int(stratified_mask.sum()) != 1:
            raise ValueError(f"Expected at most one {STRATIFIED_RUN_NAME} row without weights")
        uniform_weight = 1.0 / len(domains)
        for phase_name in ("phase_0", "phase_1"):
            for column in _phase_columns(out, f"{phase_name}_"):
                out.loc[stratified_mask, column] = 0.0
            for domain in domains:
                out.loc[stratified_mask, f"{phase_name}_{domain}"] = uniform_weight
        hydrated_rows.extend(out.loc[stratified_mask, _id_columns(out)].to_dict(orient="records"))

    remaining_missing = out.loc[~_has_phase_weights(out), _id_columns(out)]
    if not remaining_missing.empty:
        raise ValueError(f"Rows still missing phase weights:\n{remaining_missing.to_string(index=False)}")
    return out, hydrated_rows


def _add_exposure_average_columns(
    frame: pd.DataFrame,
    phase0_columns: list[str],
    phase1_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    exposure_columns: list[str] = []
    phase1_by_domain = {column.removeprefix("phase_1_"): column for column in phase1_columns}
    for phase0_column in phase0_columns:
        domain = phase0_column.removeprefix("phase_0_")
        phase1_column = phase1_by_domain.get(domain)
        if phase1_column is None:
            raise ValueError(f"Missing phase-1 weight for domain {domain}")
        output_column = f"exposure_80_20_{domain}"
        out[output_column] = EXPOSURE_PHASE0_FRACTION * pd.to_numeric(out[phase0_column], errors="coerce") + (
            1.0 - EXPOSURE_PHASE0_FRACTION
        ) * pd.to_numeric(out[phase1_column], errors="coerce")
        exposure_columns.append(output_column)
    sums = out[exposure_columns].sum(axis=1)
    if not ((sums - 1.0).abs() <= PHASE_WEIGHT_TOLERANCE).all():
        raise ValueError("Exposure-average weights do not sum to 1")
    return out, exposure_columns


def _id_columns(frame: pd.DataFrame) -> list[str]:
    preferred = (
        "registry_run_key",
        "run_name",
        "run_id",
        "scale",
        "cohort",
        "source_cohort",
        "source_experiment",
        "checkpoint_root",
        "wandb_run_id",
        "status",
        "row_kind",
        "is_qsplit240_core",
        "noise_subset_mode",
        "noise_anchor_run_name",
        "noise_source_run_name",
        "noise_trainer_seed",
        "noise_data_seed",
        "noise_simulated_epoch_subset_seed",
        "trainer_seed",
        "data_seed",
        "simulated_epoch_subset_seed",
    )
    return [column for column in preferred if column in frame.columns]


def _matrix_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    phase0_columns = _active_domain_columns(frame, "phase_0_")
    phase1_columns = _active_domain_columns(frame, "phase_1_")
    if len(phase0_columns) != len(phase1_columns):
        raise ValueError(f"Phase column mismatch: {len(phase0_columns)} phase-0 vs {len(phase1_columns)} phase-1")
    _validate_phase_weights(frame, phase0_columns, phase1_columns)
    with_exposure, exposure_columns = _add_exposure_average_columns(frame, phase0_columns, phase1_columns)
    metric_columns = [
        column
        for column in _metric_columns(with_exposure)
        if pd.to_numeric(with_exposure[column], errors="coerce").notna().any()
    ]
    id_columns = _id_columns(with_exposure)
    ordered_columns = [*id_columns, *phase0_columns, *phase1_columns, *exposure_columns, *metric_columns]
    matrix = with_exposure[ordered_columns].copy()
    summary = {
        "rows": len(matrix),
        "id_columns": len(id_columns),
        "phase0_weight_columns": len(phase0_columns),
        "phase1_weight_columns": len(phase1_columns),
        "exposure_average_weight_columns": len(exposure_columns),
        "metric_columns": len(metric_columns),
        "columns": len(matrix.columns),
        "domains": [column.removeprefix("phase_0_") for column in phase0_columns],
    }
    return matrix, summary


def _write_matrix(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    matrix, summary = _matrix_frame(frame)
    matrix.to_csv(path, index=False)
    return {**summary, "path": str(path)}


def _validate_required_metadata(frame: pd.DataFrame, *, label: str, required_columns: tuple[str, ...]) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{label} is missing required provenance columns: {missing_columns}")
    empty_counts = {
        column: int(frame[column].isna().sum()) for column in required_columns if int(frame[column].isna().sum()) > 0
    }
    if empty_counts:
        examples = frame.loc[
            frame[list(empty_counts)].isna().any(axis=1),
            [column for column in ("row_kind", "run_name", "checkpoint_root") if column in frame.columns],
        ].head(10)
        raise ValueError(
            f"{label} has empty required provenance fields: {empty_counts}\n{examples.to_string(index=False)}"
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    return value


def _parsed_noise_trainer_seed(run_name: object) -> int | None:
    if not isinstance(run_name, str):
        return None
    match = NOISE_TRAINER_SEED_RE.search(run_name)
    if match is None:
        return None
    return int(match.group("trainer_seed"))


def _with_noise_seed_metadata(noise: pd.DataFrame, *, subset_mode: str) -> pd.DataFrame:
    out = noise.copy()
    if subset_mode not in {"fixed_run00097", "variable", "proportional_variable"}:
        raise ValueError(f"Unknown noise subset mode: {subset_mode}")
    source_experiment = {
        "fixed_run00097": FIXED_SUBSET_SOURCE_EXPERIMENT,
        "variable": VARIABLE_SUBSET_SOURCE_EXPERIMENT,
        "proportional_variable": PROPORTIONAL_VARIABLE_SUBSET_SOURCE_EXPERIMENT,
    }[subset_mode]
    out["scale"] = out.get("scale", "300m_6b")
    out["cohort"] = out.get("cohort", "seed_sweep")
    out["source_cohort"] = out.get("source_cohort", pd.Series(pd.NA, index=out.index)).fillna("seed_sweep")
    out["source_experiment"] = out.get("source_experiment", pd.Series(pd.NA, index=out.index)).fillna(source_experiment)
    out["status"] = out.get("status", pd.Series(pd.NA, index=out.index)).fillna("completed")
    out["run_id"] = pd.to_numeric(out.get("run_id", pd.Series(pd.NA, index=out.index)), errors="coerce").fillna(
        NOISE_SOURCE_RUN_ID
    )
    registry_keys = out.get("registry_run_key", pd.Series(pd.NA, index=out.index))
    synthesized_keys = (
        out["scale"].astype(str)
        + ":"
        + out["cohort"].astype(str)
        + ":"
        + out["source_experiment"].astype(str)
        + ":"
        + out["run_name"].astype(str)
    )
    out["registry_run_key"] = registry_keys.fillna(synthesized_keys)
    out["row_kind"] = {
        "fixed_run00097": "noise_fixed_subset",
        "variable": "noise_variable_subset",
        "proportional_variable": "noise_variable_subset_proportional",
    }[subset_mode]
    out["noise_subset_mode"] = subset_mode
    out["noise_anchor_run_name"] = (
        PROPORTIONAL_NOISE_ANCHOR_RUN_NAME if subset_mode == "proportional_variable" else NOISE_SOURCE_RUN_NAME
    )
    out["noise_source_run_name"] = out["noise_anchor_run_name"]
    parsed_trainer_seeds = pd.to_numeric(out["run_name"].map(_parsed_noise_trainer_seed), errors="coerce")
    out["noise_trainer_seed"] = parsed_trainer_seeds.astype("Int64")
    out["noise_data_seed"] = pd.NA
    subset_seed = NOISE_SIMULATED_EPOCH_SUBSET_SEED if subset_mode == "fixed_run00097" else pd.NA
    out["noise_simulated_epoch_subset_seed"] = subset_seed
    if "trainer_seed" not in out.columns:
        out["trainer_seed"] = parsed_trainer_seeds
    else:
        out["trainer_seed"] = pd.to_numeric(out["trainer_seed"], errors="coerce").fillna(parsed_trainer_seeds)
    if "data_seed" not in out.columns:
        out["data_seed"] = pd.NA
    if "simulated_epoch_subset_seed" not in out.columns:
        out["simulated_epoch_subset_seed"] = subset_seed
    elif subset_mode == "fixed_run00097":
        out["simulated_epoch_subset_seed"] = (
            pd.to_numeric(out["simulated_epoch_subset_seed"], errors="coerce")
            .fillna(NOISE_SIMULATED_EPOCH_SUBSET_SEED)
            .astype(int)
        )
    else:
        out["simulated_epoch_subset_seed"] = pd.NA
    return out


def _hydrate_noise_phase_weights(noise: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    matches = signal[signal["run_name"].eq(NOISE_SOURCE_RUN_NAME)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {NOISE_SOURCE_RUN_NAME} source row, found {len(matches)}")
    source = matches.iloc[0]
    out = noise.copy()
    for prefix in ("phase_0_", "phase_1_"):
        for column in _phase_columns(signal, prefix):
            out[column] = float(source[column])
    return out


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_300m = _load_signal_frame(_default_extra_results_csvs())
    # The shared SNR loader returns every 300M row so it can compare signal,
    # representative, seed-noise, and validation cohorts. This matrix is the
    # qsplit-core signal swarm only.
    signal = all_300m[all_300m["cohort"].eq("signal")].copy()
    if len(signal) != 242:
        raise ValueError(f"Expected 242 300M signal rows, found {len(signal)}")

    signal, hydrated_rows = _hydrate_known_phase_weights(signal)
    qsplit_core = signal[signal["is_qsplit240_core"].fillna(False)].copy()

    signal = signal.copy()
    signal["row_kind"] = "signal"
    signal["noise_subset_mode"] = pd.NA
    signal["noise_source_run_name"] = pd.NA
    signal["noise_trainer_seed"] = pd.NA
    signal["noise_data_seed"] = pd.NA
    signal["noise_simulated_epoch_subset_seed"] = pd.NA
    signal["is_qsplit240_core"] = signal["is_qsplit240_core"].fillna(False).astype(bool)
    fixed_noise = _load_noise_frame(_default_extra_results_csvs(), noise_subset_mode="fixed")
    if len(fixed_noise) != 10:
        raise ValueError(f"Expected 10 300M fixed-subset noise rows, found {len(fixed_noise)}")
    fixed_noise = _with_noise_seed_metadata(
        _hydrate_noise_phase_weights(fixed_noise, signal),
        subset_mode="fixed_run00097",
    )
    fixed_noise["is_qsplit240_core"] = False

    variable_noise = _load_noise_frame(_default_extra_results_csvs(), noise_subset_mode="variable")
    if len(variable_noise) not in {0, 10}:
        raise ValueError(f"Expected 0 or 10 300M variable-subset noise rows, found {len(variable_noise)}")
    if variable_noise.empty:
        variable_noise = fixed_noise.iloc[0:0].copy()
        variable_noise["row_kind"] = pd.Series(dtype="object")
        variable_noise["noise_subset_mode"] = pd.Series(dtype="object")
    else:
        variable_noise = _with_noise_seed_metadata(
            _hydrate_noise_phase_weights(variable_noise, signal),
            subset_mode="variable",
        )
        variable_noise["is_qsplit240_core"] = False

    proportional_noise = _load_noise_frame(_default_extra_results_csvs(), noise_subset_mode="proportional")
    if len(proportional_noise) not in {0, 10}:
        raise ValueError(
            f"Expected 0 or 10 300M proportional variable-subset noise rows, found {len(proportional_noise)}"
        )
    if proportional_noise.empty:
        proportional_noise = fixed_noise.iloc[0:0].copy()
        proportional_noise["row_kind"] = pd.Series(dtype="object")
        proportional_noise["noise_subset_mode"] = pd.Series(dtype="object")
        proportional_noise["noise_anchor_run_name"] = pd.Series(dtype="object")
    else:
        proportional_noise = _with_noise_seed_metadata(
            proportional_noise,
            subset_mode="proportional_variable",
        )
        proportional_noise["is_qsplit240_core"] = False

    noise_frames = [fixed_noise]
    if not variable_noise.empty:
        noise_frames.append(variable_noise)
    if not proportional_noise.empty:
        noise_frames.append(proportional_noise)
    noise_with_metadata = pd.concat(noise_frames, ignore_index=True, sort=False)
    with_noise = pd.concat([signal, noise_with_metadata], ignore_index=True, sort=False)
    _validate_required_metadata(signal, label="300M signal matrix", required_columns=REQUIRED_PROVENANCE_COLUMNS)
    _validate_required_metadata(
        fixed_noise,
        label="300M fixed-subset noise matrix",
        required_columns=(*REQUIRED_PROVENANCE_COLUMNS, *REQUIRED_NOISE_PROVENANCE_COLUMNS),
    )
    if not variable_noise.empty:
        _validate_required_metadata(
            variable_noise,
            label="300M variable-subset noise matrix",
            required_columns=(*REQUIRED_PROVENANCE_COLUMNS, *REQUIRED_NOISE_PROVENANCE_COLUMNS),
        )
    if not proportional_noise.empty:
        _validate_required_metadata(
            proportional_noise,
            label="300M proportional variable-subset noise matrix",
            required_columns=(*REQUIRED_PROVENANCE_COLUMNS, *REQUIRED_NOISE_PROVENANCE_COLUMNS),
        )
    _validate_required_metadata(with_noise, label="300M matrix with noise", required_columns=REQUIRED_PROVENANCE_COLUMNS)

    outputs = {
        "canonical": _write_matrix(
            signal,
            args.output_dir / "raw_metric_matrix_300m.csv",
        ),
        "with_noise": _write_matrix(
            with_noise,
            args.output_dir / "raw_metric_matrix_300m_with_noise.csv",
        ),
        "fixed_subset_noise_baseline": _write_matrix(
            fixed_noise,
            args.output_dir / "noise_baseline_run00097_fixed_subset_300m.csv",
        ),
        "variable_subset_noise_baseline": (
            {"rows": 0, "path": str(args.output_dir / "noise_baseline_run00097_variable_subset_300m.csv")}
            if variable_noise.empty
            else _write_matrix(variable_noise, args.output_dir / "noise_baseline_run00097_variable_subset_300m.csv")
        ),
        "proportional_variable_subset_noise_baseline": (
            {"rows": 0, "path": str(args.output_dir / "noise_baseline_proportional_variable_subset_300m.csv")}
            if proportional_noise.empty
            else _write_matrix(
                proportional_noise,
                args.output_dir / "noise_baseline_proportional_variable_subset_300m.csv",
            )
        ),
        "with_proportional_noise": (
            {"rows": 0, "path": str(args.output_dir / "raw_metric_matrix_300m_with_proportional_noise.csv")}
            if proportional_noise.empty
            else _write_matrix(
                pd.concat([signal, proportional_noise], ignore_index=True, sort=False),
                args.output_dir / "raw_metric_matrix_300m_with_proportional_noise.csv",
            )
        ),
        "noise_seed_rows": (
            noise_with_metadata[
                [
                    "run_name",
                    "row_kind",
                    "noise_subset_mode",
                    "noise_anchor_run_name",
                    "noise_source_run_name",
                    "noise_trainer_seed",
                    "noise_data_seed",
                    "noise_simulated_epoch_subset_seed",
                    "trainer_seed",
                    "data_seed",
                    "simulated_epoch_subset_seed",
                    "checkpoint_root",
                ]
            ].to_dict(orient="records")
        ),
        "noise_row_counts": {
            "fixed_subset": len(fixed_noise),
            "variable_subset": len(variable_noise),
            "proportional_variable_subset": len(proportional_noise),
            "total": len(noise_with_metadata),
        },
        "qsplit240_core_rows": len(qsplit_core),
        "hydrated_known_mixture_rows": hydrated_rows,
    }
    if variable_noise.empty:
        (args.output_dir / "noise_baseline_run00097_variable_subset_300m.csv").write_text(
            fixed_noise.iloc[0:0].to_csv(index=False),
            encoding="utf-8",
        )
    if proportional_noise.empty:
        (args.output_dir / "noise_baseline_proportional_variable_subset_300m.csv").write_text(
            fixed_noise.iloc[0:0].to_csv(index=False),
            encoding="utf-8",
        )
        (args.output_dir / "raw_metric_matrix_300m_with_proportional_noise.csv").write_text(
            signal.iloc[0:0].to_csv(index=False),
            encoding="utf-8",
        )
    safe_outputs = _json_safe(outputs)
    (args.output_dir / "summary.json").write_text(json.dumps(safe_outputs, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(safe_outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
