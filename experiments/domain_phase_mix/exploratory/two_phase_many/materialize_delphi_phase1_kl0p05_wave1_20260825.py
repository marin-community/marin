# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec", "gcsfs", "pandas"]
# ///
"""Materialize the two frozen KL0.05 Wave-1 continuation panels.

Run from the repository root with::

    PYTHONPATH=. uv run \
      experiments/domain_phase_mix/exploratory/two_phase_many/materialize_delphi_phase1_kl0p05_wave1_20260825.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import fsspec
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_phase1_common_branches_20260824 as base,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_phase1_kl0p05_wave1_results_20260825"
DEFAULT_EXPERIMENT_ROOT = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/" "delphi_3e18_phase1_common_branches_v6e8_20260825"
)
EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_3e18_phase1_common_branches_v6e8_20260825"
CANDIDATE_SHA256 = "fef07d4188ef05f4df4a43d1eda6a12f7d2daf69a1ae1eb777863fd20db732b6"
SELECTED_PREFIXES_SHA256 = "f72d89240e8fee7d52ee8e86650f455fee1604e8863fc0bb7e871639fac33729"
PREFIX_REPLAY_CODE_COMMIT = "2659c1bf8e7dbb0830b4476bb763a90a35d71837"
TARGET_PREFIX = "shared_bounded_ensemble_kl0p05"
CONTINUATION_HARDWARE = base.TpuHardware(tpu_type="v6e-8", region="us-east5", zone="us-east5-b")
EXPECTED_RESULT_ROWS = 108
EXPECTED_FIT_ROWS = 100
WAVE1A_ANCHOR_ID = "fit_maximin_00"
WAVE1B_ANCHOR_ID = "control_wave1a_anchor_fit_maximin_00"
MATERIALIZATION_MANIFEST = "materialization_manifest.json"
MATERIALIZATION_COVERAGE = "materialization_coverage.json"


@dataclass(frozen=True)
class WaveContract:
    name: str
    continuation_sha256: str
    branch_code_commit: str
    run_id_base: int
    selected_run_orders: tuple[int, ...]
    fit_rows: int
    control_rows: int


WAVE_CONTRACTS = (
    WaveContract(
        name="wave1a",
        continuation_sha256="9305b5c1598c9eb11e7f898f709bfb193f37802efaba40a43fbecd0d52c12355",
        branch_code_commit="d016caa0fbd0f1f50e29ffa0c9dea5d40f5438e2",
        run_id_base=950_000,
        selected_run_orders=tuple(range(57, 114)),
        fit_rows=50,
        control_rows=7,
    ),
    WaveContract(
        name="wave1b",
        continuation_sha256="2860d0e1f177f1728580ec1cdda05e049734e7977b868a8c0abd05d9d8bd0ec3",
        branch_code_commit="ff86f88be44ae99852467cbc6a8c46a0fa4c301e",
        run_id_base=951_000,
        selected_run_orders=(*range(57, 107), 108),
        fit_rows=50,
        control_rows=1,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def matching_wave_manifest(
    fs: fsspec.AbstractFileSystem,
    root: str,
    contract: WaveContract,
    *,
    allow_missing: bool = False,
) -> tuple[str, dict[str, object]] | None:
    matches: list[tuple[str, dict[str, object]]] = []
    expected_hardware = asdict(CONTINUATION_HARDWARE)
    expected_orders = list(contract.selected_run_orders)
    for path in sorted(fs.glob(f"{root}/manifest-*/manifest.json")):
        payload = base.read_json(fs, path)
        rows = payload.get("branch_rows")
        if payload.get("experiment_name") != EXPERIMENT_NAME:
            continue
        if payload.get("candidate_weights_sha256") != CANDIDATE_SHA256:
            continue
        if payload.get("continuation_weights_sha256") != contract.continuation_sha256:
            continue
        if payload.get("selected_prefixes_sha256") != SELECTED_PREFIXES_SHA256:
            continue
        if payload.get("prefix_replay_code_commit") != PREFIX_REPLAY_CODE_COMMIT:
            continue
        if payload.get("code_commit") != contract.branch_code_commit:
            continue
        if payload.get("continuation_hardware") != expected_hardware:
            continue
        if payload.get("selected_run_orders") != expected_orders:
            continue
        if not isinstance(rows, list) or len(rows) != len(expected_orders):
            continue
        matches.append((path, payload))
    if not matches and allow_missing:
        return None
    if len(matches) != 1:
        raise ValueError(f"Expected one {contract.name} manifest; found {[path for path, _ in matches]}")

    path, payload = matches[0]
    rows = payload["branch_rows"]
    if not isinstance(rows, list):
        raise ValueError(f"Malformed {contract.name} branch rows")
    fit_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Malformed {contract.name} branch row")
        run_order = int(row["run_order"])
        if int(row["run_id"]) != contract.run_id_base + run_order:
            raise ValueError(f"{contract.name} run-ID namespace changed for order {run_order}")
        fit_rows += bool(row["fit_budget"])
    if fit_rows != contract.fit_rows or len(rows) - fit_rows != contract.control_rows:
        raise ValueError(f"{contract.name} fit/control coverage changed")
    return path, payload


def output_is_available(fs: fsspec.AbstractFileSystem, root: str, run_name: str) -> bool:
    paths = sorted(fs.glob(f"{root}/{run_name}-*/checkpoints/eval_metrics.jsonl"))
    if len(paths) > 1:
        raise ValueError(f"Expected at most one output for {run_name}; found {paths}")
    return bool(paths)


def materialize_wave(
    fs: fsspec.AbstractFileSystem,
    root: str,
    contract: WaveContract,
    manifest: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    manifest_rows = manifest["branch_rows"]
    if not isinstance(manifest_rows, list):
        raise ValueError(f"Malformed {contract.name} manifest")
    results: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for design_row in manifest_rows:
        if not isinstance(design_row, dict):
            raise ValueError(f"Malformed {contract.name} manifest row")
        run_name = str(design_row["run_name"])
        identity = {
            "wave": contract.name,
            "continuation_weights_sha256": contract.continuation_sha256,
            "continuation_id": str(design_row["continuation_id"]),
            "run_order": int(design_row["run_order"]),
            "run_id": int(design_row["run_id"]),
            "run_name": run_name,
            "fit_budget": bool(design_row["fit_budget"]),
        }
        if not output_is_available(fs, root, run_name):
            missing.append(identity)
            continue
        row, row_metrics = base.materialize_design_row(
            fs,
            root,
            design_row,
            candidate_sha256=CANDIDATE_SHA256,
            continuation_sha256=contract.continuation_sha256,
            prefix_replay_code_commit=PREFIX_REPLAY_CODE_COMMIT,
            branch_code_commit=contract.branch_code_commit,
            expected_experiment_name=EXPERIMENT_NAME,
            continuation_hardware=CONTINUATION_HARDWARE,
        )
        row["wave"] = contract.name
        results.append(row)
        for metric in row_metrics:
            metric["wave"] = contract.name
        metrics.extend(row_metrics)
    return results, metrics, missing


def phase_hash(row: pd.Series, phase: str) -> str:
    values = {column: float(row[column]) for column in sorted(row.index) if column.startswith(f"{phase}_")}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def local_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_record(path: Path, rows: int) -> dict[str, object]:
    return {"sha256": local_file_sha256(path), "rows": rows}


def write_materialization_manifest(
    output_dir: Path,
    artifacts: dict[str, dict[str, object]],
    provenance: dict[str, object],
) -> Path:
    path = output_dir / MATERIALIZATION_MANIFEST
    payload = {
        "schema_version": "delphi_phase1_materialization_v1",
        "complete": True,
        "artifacts": artifacts,
        "provenance": provenance,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def validate_combined_results(results: pd.DataFrame, *, complete: bool) -> None:
    if results.run_name.duplicated().any():
        raise ValueError("Run names collide across Wave 1A and Wave 1B")
    if results.run_id.duplicated().any():
        raise ValueError("Run IDs collide across Wave 1A and Wave 1B")
    fit = results[results.fit_budget]
    if not fit.prefix_candidate_id.eq(TARGET_PREFIX).all() or not fit.prefix_repeat_seed.eq(0).all():
        raise ValueError("Fit rows are not all branches of the frozen KL0.05 seed-0 prefix")
    if not complete:
        return
    if len(results) != EXPECTED_RESULT_ROWS or len(fit) != EXPECTED_FIT_ROWS:
        raise ValueError(f"Wave-1 coverage changed: rows={len(results)}, fit_rows={len(fit)}")
    if fit.apply(lambda row: phase_hash(row, "phase_1"), axis=1).nunique() != EXPECTED_FIT_ROWS:
        raise ValueError("The 100 Wave-1 fit continuations are not runtime-distinct")


def anchor_contrast(results: pd.DataFrame) -> pd.DataFrame:
    wave1a = results[
        results.wave.eq("wave1a")
        & results.continuation_id.eq(WAVE1A_ANCHOR_ID)
        & results.prefix_candidate_id.eq(TARGET_PREFIX)
        & results.prefix_repeat_seed.eq(0)
    ]
    wave1b = results[results.wave.eq("wave1b") & results.continuation_id.eq(WAVE1B_ANCHOR_ID)]
    if wave1a.empty or wave1b.empty:
        return pd.DataFrame()
    if len(wave1a) != 1 or len(wave1b) != 1:
        raise ValueError("Cross-wave anchor identity is ambiguous")
    left = wave1a.iloc[0]
    right = wave1b.iloc[0]
    phase_columns: list[str] = [str(column) for column in results if str(column).startswith(("phase_0_", "phase_1_"))]
    if any(float(left[column]) != float(right[column]) for column in phase_columns):
        raise ValueError("Cross-wave anchor mixtures differ")
    if int(left.data_seed) != int(right.data_seed) or int(left.trainer_seed) != int(right.trainer_seed):
        raise ValueError("Cross-wave anchor seeds differ")
    return pd.DataFrame(
        [
            {
                "wave1a_run_name": left.run_name,
                "wave1b_run_name": right.run_name,
                "uncheatable_bpb_wave1a": float(left.uncheatable_bpb),
                "uncheatable_bpb_wave1b": float(right.uncheatable_bpb),
                "uncheatable_bpb_wave1b_minus_wave1a": float(right.uncheatable_bpb - left.uncheatable_bpb),
                "github_cpp_bpb_wave1a": float(left.github_cpp_bpb),
                "github_cpp_bpb_wave1b": float(right.github_cpp_bpb),
                "github_cpp_bpb_wave1b_minus_wave1a": float(right.github_cpp_bpb - left.github_cpp_bpb),
            }
        ]
    )


def main() -> None:
    args = parse_args()
    fs, root = fsspec.core.url_to_fs(args.experiment_root)
    result_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for contract in WAVE_CONTRACTS:
        match = matching_wave_manifest(fs, root, contract, allow_missing=args.allow_incomplete)
        if match is None:
            fit_orders = set(contract.selected_run_orders[: contract.fit_rows])
            missing_rows.extend(
                {
                    "wave": contract.name,
                    "continuation_weights_sha256": contract.continuation_sha256,
                    "continuation_id": None,
                    "run_order": run_order,
                    "run_id": contract.run_id_base + run_order,
                    "run_name": None,
                    "fit_budget": run_order in fit_orders,
                    "reason": "manifest_not_materialized",
                }
                for run_order in contract.selected_run_orders
            )
            continue
        manifest_path, manifest = match
        manifest_rows.append(
            {
                "wave": contract.name,
                "uri": base.gs_uri(manifest_path),
                "sha256": hashlib.sha256(fs.cat(manifest_path)).hexdigest(),
                "continuation_weights_sha256": contract.continuation_sha256,
                "branch_code_commit": contract.branch_code_commit,
                "run_id_base": contract.run_id_base,
                "expected_rows": contract.fit_rows + contract.control_rows,
            }
        )
        wave_results, wave_metrics, wave_missing = materialize_wave(fs, root, contract, manifest)
        result_rows.extend(wave_results)
        metric_rows.extend(wave_metrics)
        missing_rows.extend(wave_missing)

    results = pd.DataFrame(result_rows)
    metrics = pd.DataFrame(metric_rows)
    missing = pd.DataFrame(missing_rows)
    complete = not missing_rows
    if not results.empty:
        results = results.sort_values(["wave", "run_order", "run_id"]).reset_index(drop=True)
        validate_combined_results(results, complete=complete)
    if not metrics.empty:
        metrics = metrics.sort_values(["wave", "run_name", "metric"]).reset_index(drop=True)
    if not missing.empty:
        missing = missing.sort_values(["wave", "run_order", "run_id"]).reset_index(drop=True)
    anchor = anchor_contrast(results) if not results.empty else pd.DataFrame()
    if complete and anchor.empty:
        raise ValueError("Complete Wave 1 lacks its cross-wave anchor")

    fit_results = results[results.fit_budget] if not results.empty else pd.DataFrame()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing.to_csv(args.output_dir / "missing_rows.csv", index=False)
    coverage = {
        "complete": complete,
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "expected_fit_rows": EXPECTED_FIT_ROWS,
        "completed_result_rows": len(results),
        "completed_fit_rows": len(fit_results),
        "missing_result_rows": len(missing),
        "metric_rows": len(metrics),
        "experiment_root": args.experiment_root,
        "experiment_name": EXPERIMENT_NAME,
        "candidate_weights_sha256": CANDIDATE_SHA256,
        "selected_prefixes_sha256": SELECTED_PREFIXES_SHA256,
        "prefix_replay_code_commit": PREFIX_REPLAY_CODE_COMMIT,
        "continuation_hardware": asdict(CONTINUATION_HARDWARE),
        "manifests": manifest_rows,
        "cross_wave_anchor_available": not anchor.empty,
    }
    (args.output_dir / "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    if complete:
        final_frames = {
            "branch_results.csv": results,
            "branch_fit_matrix.csv": fit_results,
            "uncheatable_metrics_long.csv": metrics,
            "cross_wave_anchor.csv": anchor,
        }
        for name, frame in final_frames.items():
            frame.to_csv(args.output_dir / name, index=False)
        (args.output_dir / MATERIALIZATION_COVERAGE).write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
        artifacts = {name: artifact_record(args.output_dir / name, len(frame)) for name, frame in final_frames.items()}
        artifacts[MATERIALIZATION_COVERAGE] = artifact_record(args.output_dir / MATERIALIZATION_COVERAGE, 1)
        write_materialization_manifest(
            args.output_dir,
            artifacts,
            {
                "experiment_root": args.experiment_root,
                "experiment_name": EXPERIMENT_NAME,
                "candidate_weights_sha256": CANDIDATE_SHA256,
                "selected_prefixes_sha256": SELECTED_PREFIXES_SHA256,
                "prefix_replay_code_commit": PREFIX_REPLAY_CODE_COMMIT,
                "manifests": manifest_rows,
            },
        )
    else:
        results.to_csv(args.output_dir / "partial_branch_results.csv", index=False)
        fit_results.to_csv(args.output_dir / "partial_branch_fit_matrix.csv", index=False)
        metrics.to_csv(args.output_dir / "partial_uncheatable_metrics_long.csv", index=False)
        anchor.to_csv(args.output_dir / "partial_cross_wave_anchor.csv", index=False)
    print(json.dumps(coverage, indent=2, sort_keys=True))
    if not complete and not args.allow_incomplete:
        raise ValueError(f"Wave 1 is incomplete: {len(missing)} rows are missing")


if __name__ == "__main__":
    main()
