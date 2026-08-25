# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec", "gcsfs", "pandas"]
# ///
"""Materialize the frozen KL0.05 phase-1 branch-noise controls.

Run from the repository root with::

    PYTHONPATH=. uv run \
      experiments/domain_phase_mix/exploratory/two_phase_many/\
materialize_delphi_phase1_kl0p05_noise_controls_20260825.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import fsspec
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_phase1_common_branches_20260824 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_phase1_kl0p05_wave1_20260825 as wave1_materialize,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_phase1_kl0p05_noise_results_20260825"
DEFAULT_WAVE1_RESULTS = REFERENCE_OUTPUTS / "delphi_phase1_kl0p05_wave1_results_20260825" / "branch_results.csv"
DEFAULT_WAVE1_METRICS = (
    REFERENCE_OUTPUTS / "delphi_phase1_kl0p05_wave1_results_20260825" / "uncheatable_metrics_long.csv"
)
DEFAULT_EXPERIMENT_ROOT = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/" "delphi_3e18_phase1_common_branches_v6e8_20260825"
)
EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_3e18_phase1_common_branches_v6e8_20260825"
CANDIDATE_SHA256 = "fef07d4188ef05f4df4a43d1eda6a12f7d2daf69a1ae1eb777863fd20db732b6"
CONTINUATION_SHA256 = "9305b5c1598c9eb11e7f898f709bfb193f37802efaba40a43fbecd0d52c12355"
SELECTED_PREFIXES_SHA256 = "f72d89240e8fee7d52ee8e86650f455fee1604e8863fc0bb7e871639fac33729"
PREFIX_REPLAY_CODE_COMMIT = "2659c1bf8e7dbb0830b4476bb763a90a35d71837"
BRANCH_CODE_COMMIT = "8329827aa7902ea58f77a5457ac3468930f55f34"
NOISE_DESIGN_SHA256 = "554bf4372327a0ab539ab82da6c93c4013a1f587ff78fda353f65ba5a1226c35"
TARGET_PREFIX = "shared_bounded_ensemble_kl0p05"
RUN_ID_BASE = 952_000
RUN_ORDERS = tuple(range(228, 236))
DATA_SEEDS = tuple(range(962_000, 962_008))
EXPECTED_FULL_DESIGN_ROWS = 236
EXPECTED_FRESH_ROWS = 8
EXPECTED_GROUP_ROWS = 5
CONTINUATION_HARDWARE = base.TpuHardware(tpu_type="v6e-8", region="us-east5", zone="us-east5-b")
NOISE_GROUPS = {
    "control_proportional": "control_proportional",
    "fit_maximin_26": "fit_maximin_26",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--wave1-results", type=Path, default=DEFAULT_WAVE1_RESULTS)
    parser.add_argument("--wave1-metrics", type=Path, default=DEFAULT_WAVE1_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def expected_hardware_gate() -> dict[str, object]:
    payload = asdict(base.HardwareCanaryGate())
    payload["noise_run_orders"] = list(RUN_ORDERS)
    payload["provenance_comparison_mask"] = list(base.HardwareCanaryGate().provenance_comparison_mask)
    return payload


def matching_noise_manifest(
    fs: fsspec.AbstractFileSystem,
    root: str,
    *,
    allow_missing: bool = False,
) -> tuple[str, dict[str, object]] | None:
    matches: list[tuple[str, dict[str, object]]] = []
    for path in sorted(fs.glob(f"{root}/manifest-*/manifest.json")):
        payload = base.read_json(fs, path)
        rows = payload.get("branch_rows")
        if payload.get("experiment_name") != EXPERIMENT_NAME:
            continue
        if payload.get("candidate_weights_sha256") != CANDIDATE_SHA256:
            continue
        if payload.get("continuation_weights_sha256") != CONTINUATION_SHA256:
            continue
        if payload.get("selected_prefixes_sha256") != SELECTED_PREFIXES_SHA256:
            continue
        if payload.get("prefix_replay_code_commit") != PREFIX_REPLAY_CODE_COMMIT:
            continue
        if payload.get("code_commit") != BRANCH_CODE_COMMIT:
            continue
        if payload.get("branch_noise_design_sha256") != NOISE_DESIGN_SHA256:
            continue
        if payload.get("continuation_hardware") != asdict(CONTINUATION_HARDWARE):
            continue
        if payload.get("hardware_canary_gate") != expected_hardware_gate():
            continue
        if payload.get("branch_run_id_base") != RUN_ID_BASE:
            continue
        if payload.get("expected_full_design_rows") != EXPECTED_FULL_DESIGN_ROWS:
            continue
        if payload.get("selected_design_rows") != EXPECTED_FRESH_ROWS:
            continue
        if payload.get("fit_budget_rows") != 0 or payload.get("control_rows") != EXPECTED_FRESH_ROWS:
            continue
        if payload.get("same_prefix_branch_noise_rows") != EXPECTED_FRESH_ROWS:
            continue
        if payload.get("selected_run_orders") != list(RUN_ORDERS):
            continue
        if not isinstance(rows, list) or len(rows) != EXPECTED_FRESH_ROWS:
            continue
        matches.append((path, payload))
    if not matches and allow_missing:
        return None
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen noise manifest; found {[path for path, _ in matches]}")
    path, payload = matches[0]
    validate_manifest_rows(payload)
    return path, payload


def validate_manifest_rows(manifest: dict[str, object]) -> None:
    rows = manifest.get("branch_rows")
    if not isinstance(rows, list):
        raise ValueError("Noise manifest rows are malformed")
    observed_orders = tuple(int(row["run_order"]) for row in rows if isinstance(row, dict))
    observed_ids = tuple(int(row["run_id"]) for row in rows if isinstance(row, dict))
    observed_seeds = tuple(int(row["data_seed"]) for row in rows if isinstance(row, dict))
    if observed_orders != RUN_ORDERS:
        raise ValueError(f"Noise run orders changed: {observed_orders}")
    if observed_ids != tuple(RUN_ID_BASE + order for order in RUN_ORDERS):
        raise ValueError(f"Noise run-ID namespace changed: {observed_ids}")
    if observed_seeds != DATA_SEEDS:
        raise ValueError(f"Noise data seeds changed: {observed_seeds}")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Noise manifest row is malformed")
        prefix = row.get("prefix")
        if not isinstance(prefix, dict):
            raise ValueError("Noise manifest prefix is malformed")
        if prefix.get("candidate_id") != TARGET_PREFIX or prefix.get("repeat_seed") != 0:
            raise ValueError("Noise controls no longer use the exact KL0.05 seed-0 prefix")
        if row.get("trainer_seed") != 0 or row.get("fit_budget") is not False:
            raise ValueError("Noise controls changed trainer seed or entered the fit budget")
        if row.get("branch_role") != "same_prefix_branch_noise":
            raise ValueError("Noise-control branch role changed")


def output_is_available(fs: fsspec.AbstractFileSystem, root: str, run_name: str) -> bool:
    paths = sorted(fs.glob(f"{root}/{run_name}-*/checkpoints/eval_metrics.jsonl"))
    if len(paths) > 1:
        raise ValueError(f"Expected at most one output for {run_name}; found {paths}")
    return bool(paths)


def materialize_fresh_rows(
    fs: fsspec.AbstractFileSystem,
    root: str,
    manifest: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    rows = manifest["branch_rows"]
    if not isinstance(rows, list):
        raise ValueError("Noise manifest rows are malformed")
    for design_row in rows:
        if not isinstance(design_row, dict):
            raise ValueError("Noise manifest row is malformed")
        run_name = str(design_row["run_name"])
        if not output_is_available(fs, root, run_name):
            missing_rows.append(
                {
                    "run_order": int(design_row["run_order"]),
                    "run_id": int(design_row["run_id"]),
                    "run_name": run_name,
                    "data_seed": int(design_row["data_seed"]),
                }
            )
            continue
        row, metrics = base.materialize_design_row(
            fs,
            root,
            design_row,
            candidate_sha256=CANDIDATE_SHA256,
            continuation_sha256=CONTINUATION_SHA256,
            prefix_replay_code_commit=PREFIX_REPLAY_CODE_COMMIT,
            branch_code_commit=BRANCH_CODE_COMMIT,
            expected_experiment_name=EXPERIMENT_NAME,
            continuation_hardware=CONTINUATION_HARDWARE,
        )
        row["noise_group_id"] = str(design_row["noise_group_id"])
        row["noise_repeat_index"] = int(design_row["branch_noise_repeat_index"])
        result_rows.append(row)
        for metric in metrics:
            metric["noise_group_id"] = row["noise_group_id"]
            metric["noise_repeat_index"] = row["noise_repeat_index"]
        metric_rows.extend(metrics)
    results = pd.DataFrame(result_rows)
    metrics = pd.DataFrame(metric_rows)
    missing = pd.DataFrame(missing_rows)
    if not results.empty:
        results = results.sort_values("run_order").reset_index(drop=True)
        validate_fresh_results(results)
    if not metrics.empty:
        metrics = metrics.sort_values(["noise_group_id", "noise_repeat_index", "metric"]).reset_index(drop=True)
    if not missing.empty:
        missing = missing.sort_values("run_order").reset_index(drop=True)
    return results, metrics, missing


def validate_fresh_results(results: pd.DataFrame) -> None:
    if results.run_name.duplicated().any() or results.run_id.duplicated().any():
        raise ValueError("Noise-control run identities collide")
    if not results.prefix_candidate_id.eq(TARGET_PREFIX).all() or not results.prefix_repeat_seed.eq(0).all():
        raise ValueError("Noise results do not use the exact KL0.05 seed-0 prefix")
    if not results.trainer_seed.eq(0).all() or results.fit_budget.any():
        raise ValueError("Noise results changed trainer seed or entered the fit budget")
    phase_columns = [column for column in results if column.startswith(("phase_0_", "phase_1_"))]
    for group_id, group in results.groupby("noise_group_id", sort=False):
        if len(group) > 4:
            raise ValueError(f"Noise group {group_id} has too many fresh rows")
        if group.data_seed.nunique() != len(group):
            raise ValueError(f"Noise group {group_id} repeats a data seed")
        if any(group[column].nunique() != 1 for column in phase_columns):
            raise ValueError(f"Noise group {group_id} changes its policy")


def combine_with_seed_zero(
    fresh_results: pd.DataFrame,
    fresh_metrics: pd.DataFrame,
    wave1_results_path: Path,
    wave1_metrics_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not wave1_results_path.exists() or not wave1_metrics_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    wave1 = pd.read_csv(wave1_results_path)
    wave1_metrics = pd.read_csv(wave1_metrics_path)
    baseline_rows = []
    baseline_metrics = []
    for group_name, continuation_id in NOISE_GROUPS.items():
        matches = wave1[
            wave1.wave.eq("wave1a")
            & wave1.prefix_candidate_id.eq(TARGET_PREFIX)
            & wave1.prefix_repeat_seed.eq(0)
            & wave1.continuation_id.eq(continuation_id)
            & wave1.data_seed.eq(930_000)
            & wave1.trainer_seed.eq(0)
        ]
        if len(matches) > 1:
            raise ValueError(f"Wave-1 seed-zero anchor is ambiguous for {continuation_id}")
        if matches.empty:
            return pd.DataFrame(), pd.DataFrame()
        baseline = matches.iloc[0].to_dict()
        baseline["noise_group_id"] = f"{TARGET_PREFIX}/{group_name}"
        baseline["noise_repeat_index"] = 0
        baseline_rows.append(baseline)
        run_name = str(baseline["run_name"])
        group_metrics = wave1_metrics[wave1_metrics.run_name.eq(run_name)].copy()
        if group_metrics.empty:
            return pd.DataFrame(), pd.DataFrame()
        group_metrics["noise_group_id"] = baseline["noise_group_id"]
        group_metrics["noise_repeat_index"] = 0
        baseline_metrics.extend(group_metrics.to_dict(orient="records"))
    combined = pd.concat([pd.DataFrame(baseline_rows), fresh_results], ignore_index=True, sort=False)
    combined_metrics = pd.concat([pd.DataFrame(baseline_metrics), fresh_metrics], ignore_index=True, sort=False)
    group_sizes = combined.groupby("noise_group_id").size()
    if len(group_sizes) != len(NOISE_GROUPS) or not group_sizes.eq(EXPECTED_GROUP_ROWS).all():
        raise ValueError(f"Expected n={EXPECTED_GROUP_ROWS} per noise action; found {group_sizes.to_dict()}")
    if any(group.data_seed.nunique() != EXPECTED_GROUP_ROWS for _, group in combined.groupby("noise_group_id")):
        raise ValueError("Combined noise groups do not have five distinct data seeds")
    return combined, combined_metrics


def summarize_metrics(combined_metrics: pd.DataFrame) -> pd.DataFrame:
    if combined_metrics.empty:
        return pd.DataFrame()
    rows = []
    for (group_id, metric), group in combined_metrics.groupby(["noise_group_id", "metric"], sort=True):
        values = group.value.to_numpy(dtype=float)
        if len(values) != EXPECTED_GROUP_ROWS:
            raise ValueError(f"Noise summary for {group_id}/{metric} has n={len(values)}")
        rows.append(
            {
                "noise_group_id": group_id,
                "metric": metric,
                "n": len(values),
                "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)),
                "sem": float(values.std(ddof=1) / len(values) ** 0.5),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "range": float(values.max() - values.min()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    fs, root = fsspec.core.url_to_fs(args.experiment_root)
    match = matching_noise_manifest(fs, root, allow_missing=args.allow_incomplete)
    manifest_path = None
    manifest_sha256 = None
    if match is None:
        fresh_results = pd.DataFrame()
        fresh_metrics = pd.DataFrame()
        missing = pd.DataFrame(
            {
                "run_order": RUN_ORDERS,
                "run_id": tuple(RUN_ID_BASE + order for order in RUN_ORDERS),
                "run_name": (None,) * EXPECTED_FRESH_ROWS,
                "data_seed": DATA_SEEDS,
            }
        )
    else:
        manifest_path, manifest = match
        manifest_sha256 = hashlib.sha256(fs.cat(manifest_path)).hexdigest()
        fresh_results, fresh_metrics, missing = materialize_fresh_rows(fs, root, manifest)
    fresh_complete = len(fresh_results) == EXPECTED_FRESH_ROWS and missing.empty
    if fresh_complete:
        combined, combined_metrics = combine_with_seed_zero(
            fresh_results,
            fresh_metrics,
            args.wave1_results,
            args.wave1_metrics,
        )
    else:
        combined = pd.DataFrame()
        combined_metrics = pd.DataFrame()
    summary = summarize_metrics(combined_metrics)
    complete = fresh_complete and not summary.empty

    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing.to_csv(args.output_dir / "missing_rows.csv", index=False)
    coverage = {
        "complete": complete,
        "n5_summary_available": not summary.empty,
        "expected_fresh_rows": EXPECTED_FRESH_ROWS,
        "completed_fresh_rows": len(fresh_results),
        "missing_fresh_rows": len(missing),
        "noise_groups": sorted(f"{TARGET_PREFIX}/{group}" for group in NOISE_GROUPS),
        "expected_rows_per_group_including_seed_zero": EXPECTED_GROUP_ROWS,
        "experiment_root": args.experiment_root,
        "manifest_uri": base.gs_uri(manifest_path) if manifest_path is not None else None,
        "manifest_sha256": manifest_sha256,
        "branch_code_commit": BRANCH_CODE_COMMIT,
        "noise_design_sha256": NOISE_DESIGN_SHA256,
        "continuation_hardware": asdict(CONTINUATION_HARDWARE),
    }
    (args.output_dir / "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    if complete:
        final_frames = {
            "fresh_noise_results.csv": fresh_results,
            "fresh_noise_metrics_long.csv": fresh_metrics,
            "noise_results_n5.csv": combined,
            "noise_metrics_n5_long.csv": combined_metrics,
            "noise_summary_n5.csv": summary,
        }
        for name, frame in final_frames.items():
            frame.to_csv(args.output_dir / name, index=False)
        materialization_coverage = args.output_dir / wave1_materialize.MATERIALIZATION_COVERAGE
        materialization_coverage.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
        artifacts = {
            name: wave1_materialize.artifact_record(args.output_dir / name, len(frame))
            for name, frame in final_frames.items()
        }
        artifacts[wave1_materialize.MATERIALIZATION_COVERAGE] = wave1_materialize.artifact_record(
            materialization_coverage,
            1,
        )
        wave1_materialize.write_materialization_manifest(
            args.output_dir,
            artifacts,
            {
                "experiment_root": args.experiment_root,
                "experiment_name": EXPERIMENT_NAME,
                "branch_manifest_uri": coverage["manifest_uri"],
                "branch_manifest_sha256": manifest_sha256,
                "branch_code_commit": BRANCH_CODE_COMMIT,
                "noise_design_sha256": NOISE_DESIGN_SHA256,
                "wave1_results_sha256": wave1_materialize.local_file_sha256(args.wave1_results),
                "wave1_metrics_sha256": wave1_materialize.local_file_sha256(args.wave1_metrics),
            },
        )
    else:
        fresh_results.to_csv(args.output_dir / "partial_fresh_noise_results.csv", index=False)
        fresh_metrics.to_csv(args.output_dir / "partial_fresh_noise_metrics_long.csv", index=False)
        combined.to_csv(args.output_dir / "partial_noise_results_n5.csv", index=False)
        combined_metrics.to_csv(args.output_dir / "partial_noise_metrics_n5_long.csv", index=False)
        summary.to_csv(args.output_dir / "partial_noise_summary_n5.csv", index=False)
    print(json.dumps(coverage, indent=2, sort_keys=True))
    if (not complete or summary.empty) and not args.allow_incomplete:
        raise ValueError("Noise controls or their n=5 seed-zero anchors are incomplete")


if __name__ == "__main__":
    main()
