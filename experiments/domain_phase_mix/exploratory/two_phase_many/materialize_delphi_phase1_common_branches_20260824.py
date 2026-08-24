# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec", "gcsfs", "pandas"]
# ///
"""Materialize the frozen Delphi phase-1 branch panel for surrogate fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import fsspec
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_phase1_common_branch_results_20260824"
DEFAULT_EXPERIMENT_ROOT = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/" "delphi_3e18_phase1_common_branches_20260824"
)
EXPECTED_EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_3e18_phase1_common_branches_20260824"
EXPECTED_TERMINAL_STEP = 3_006
EXPECTED_FULL_ROWS = 232
EXPECTED_FIT_ROWS = 200
EXPECTED_PREFIX_COUNT = 4
EXPECTED_FIT_CONTINUATIONS = 50
EXPECTED_BRANCH_NOISE_ROWS = 4
PRIMARY_METRIC = "eval/uncheatable_eval/bpb"
DIAGNOSTIC_METRIC = "eval/uncheatable_eval/github_cpp/bpb"
BRANCH_PROVENANCE_FILENAME = "branch_provenance.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--expected-continuation-sha256", required=True)
    parser.add_argument("--expected-selected-prefixes-sha256", required=True)
    parser.add_argument("--prefix-replay-code-commit", required=True)
    parser.add_argument("--branch-code-commit", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_json(fs: fsspec.AbstractFileSystem, path: str) -> dict[str, object]:
    with fs.open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


def read_json_lines(fs: fsspec.AbstractFileSystem, path: str) -> list[dict[str, object]]:
    with fs.open(path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in {path}")
    return rows


def gs_uri(path: str) -> str:
    return "gs://" + path.removeprefix("gs://").lstrip("/")


def matching_full_manifest(
    fs: fsspec.AbstractFileSystem,
    root: str,
    *,
    candidate_sha256: str,
    continuation_sha256: str,
    selected_prefixes_sha256: str,
    prefix_replay_code_commit: str,
    branch_code_commit: str,
) -> tuple[str, dict[str, object]]:
    matches = []
    for path in sorted(fs.glob(f"{root}/manifest-*/manifest.json")):
        payload = read_json(fs, path)
        if payload.get("experiment_name") != EXPECTED_EXPERIMENT_NAME:
            continue
        if payload.get("candidate_weights_sha256") != candidate_sha256:
            continue
        if payload.get("continuation_weights_sha256") != continuation_sha256:
            continue
        if payload.get("selected_prefixes_sha256") != selected_prefixes_sha256:
            continue
        if payload.get("prefix_replay_code_commit") != prefix_replay_code_commit:
            continue
        if payload.get("code_commit") != branch_code_commit:
            continue
        rows = payload.get("branch_rows")
        if (
            payload.get("expected_full_design_rows") == EXPECTED_FULL_ROWS
            and payload.get("selected_design_rows") == EXPECTED_FULL_ROWS
            and isinstance(rows, list)
            and len(rows) == EXPECTED_FULL_ROWS
        ):
            matches.append((path, payload))
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen full-panel manifest; found {[path for path, _ in matches]}")
    return matches[0]


def metric_record(
    fs: fsspec.AbstractFileSystem,
    root: str,
    run_name: str,
) -> tuple[str, dict[str, object]]:
    paths = sorted(fs.glob(f"{root}/{run_name}-*/checkpoints/eval_metrics.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"Expected one output for {run_name}; found {paths}")
    records = [row for row in read_json_lines(fs, paths[0]) if int(row.get("step", -1)) == EXPECTED_TERMINAL_STEP]
    if len(records) != 1:
        raise ValueError(f"Expected one step-{EXPECTED_TERMINAL_STEP} metric row for {run_name}; found {len(records)}")
    return paths[0], records[0]


def materialize_rows(
    fs: fsspec.AbstractFileSystem,
    root: str,
    manifest: dict[str, object],
    *,
    candidate_sha256: str,
    continuation_sha256: str,
    prefix_replay_code_commit: str,
    branch_code_commit: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_rows = manifest["branch_rows"]
    if not isinstance(manifest_rows, list):
        raise ValueError("Branch manifest rows are malformed")
    result_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    observed_names: set[str] = set()
    for design_row in manifest_rows:
        if not isinstance(design_row, dict):
            raise ValueError("Branch manifest row is malformed")
        run_name = str(design_row["run_name"])
        if run_name in observed_names:
            raise ValueError(f"Duplicate manifest run name: {run_name}")
        observed_names.add(run_name)
        metric_path, record = metric_record(fs, root, run_name)
        if PRIMARY_METRIC not in record or DIAGNOSTIC_METRIC not in record:
            raise ValueError(f"Required terminal metrics are missing for {run_name}")
        output_root = str(PurePosixPath(metric_path).parents[1])
        checkpoint_path = f"{output_root}/checkpoints/step-{EXPECTED_TERMINAL_STEP}"
        metadata = read_json(fs, f"{checkpoint_path}/metadata.json")
        if metadata.get("step") != EXPECTED_TERMINAL_STEP or metadata.get("is_temporary") is not False:
            raise ValueError(f"Terminal checkpoint is not permanent for {run_name}: {metadata}")
        provenance_path = f"{output_root}/{BRANCH_PROVENANCE_FILENAME}"
        with fs.open(provenance_path, "rb") as handle:
            provenance_bytes = handle.read()
        provenance = json.loads(provenance_bytes)
        prefix = design_row["prefix"]
        if not isinstance(prefix, dict):
            raise ValueError(f"Prefix identity is malformed for {run_name}")
        phase_weights = design_row["phase_weights"]
        if not isinstance(phase_weights, dict):
            raise ValueError(f"Phase weights are malformed for {run_name}")
        expected_provenance = {
            "experiment_name": EXPECTED_EXPERIMENT_NAME,
            "run_name": run_name,
            "run_order": int(design_row["run_order"]),
            "run_id": int(design_row["run_id"]),
            "data_seed": int(design_row["data_seed"]),
            "trainer_seed": int(design_row["trainer_seed"]),
            "prefix_candidate_id": prefix["candidate_id"],
            "prefix_repeat_seed": int(prefix["repeat_seed"]),
            "prefix_checkpoint_uri": prefix["checkpoint_uri"],
            "prefix_provenance_sha256": prefix["provenance_sha256"],
            "prefix_replay_code_commit": prefix_replay_code_commit,
            "candidate_weights_sha256": candidate_sha256,
            "continuation_weights_sha256": continuation_sha256,
            "continuation_id": str(design_row["continuation_id"]),
            "phase_weights_sha256": hashlib.sha256(json.dumps(phase_weights, sort_keys=True).encode()).hexdigest(),
            "branch_code_commit": branch_code_commit,
            "terminal_checkpoint_uri": gs_uri(checkpoint_path),
            "terminal_checkpoint_step": EXPECTED_TERMINAL_STEP,
        }
        if provenance != expected_provenance:
            raise ValueError(f"Branch provenance mismatch for {run_name}: {provenance}")

        row: dict[str, object] = {
            "run_order": int(design_row["run_order"]),
            "run_id": int(design_row["run_id"]),
            "run_name": run_name,
            "data_seed": int(design_row["data_seed"]),
            "trainer_seed": int(design_row["trainer_seed"]),
            "fit_budget": bool(design_row["fit_budget"]),
            "branch_role": design_row["branch_role"],
            "continuation_id": design_row["continuation_id"],
            "continuation_role": design_row["continuation_role"],
            "prefix_candidate_id": prefix["candidate_id"],
            "prefix_repeat_seed": int(prefix["repeat_seed"]),
            "checkpoint_uri": gs_uri(checkpoint_path),
            "provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
            "uncheatable_bpb": float(record[PRIMARY_METRIC]),
            "github_cpp_bpb": float(record[DIAGNOSTIC_METRIC]),
        }
        phase_0 = phase_weights.get("phase_0")
        phase_1 = phase_weights.get("phase_1")
        if not isinstance(phase_0, dict) or not isinstance(phase_1, dict) or tuple(phase_0) != tuple(phase_1):
            raise ValueError(f"Phase weights disagree for {run_name}")
        row.update({f"phase_0_{bucket}": float(weight) for bucket, weight in phase_0.items()})
        row.update({f"phase_1_{bucket}": float(weight) for bucket, weight in phase_1.items()})
        result_rows.append(row)
        metric_rows.extend(
            {
                "run_name": run_name,
                "metric": key,
                "value": float(value),
            }
            for key, value in record.items()
            if key.startswith("eval/uncheatable_eval/") and isinstance(value, (float, int))
        )

    results = pd.DataFrame(result_rows).sort_values("run_order").reset_index(drop=True)
    if len(results) != EXPECTED_FULL_ROWS or int(results.fit_budget.sum()) != EXPECTED_FIT_ROWS:
        raise ValueError(f"Branch coverage changed: rows={len(results)}, fit_rows={int(results.fit_budget.sum())}")
    fit = results[results.fit_budget]
    per_prefix = fit.groupby("prefix_candidate_id").continuation_id.nunique()
    per_continuation = fit.groupby("continuation_id").prefix_candidate_id.nunique()
    if len(per_prefix) != EXPECTED_PREFIX_COUNT or not per_prefix.eq(EXPECTED_FIT_CONTINUATIONS).all():
        raise ValueError(f"Fit continuations are not fully crossed by prefix: {per_prefix.to_dict()}")
    if len(per_continuation) != EXPECTED_FIT_CONTINUATIONS or not per_continuation.eq(EXPECTED_PREFIX_COUNT).all():
        raise ValueError(f"Fit prefixes are not fully crossed by continuation: {per_continuation.to_dict()}")

    noise = results[results.branch_role.eq("same_prefix_branch_noise")]
    phase_columns = [column for column in results if column.startswith(("phase_0_", "phase_1_"))]
    if (
        len(noise) != EXPECTED_BRANCH_NOISE_ROWS
        or noise.prefix_candidate_id.nunique() != 1
        or noise.data_seed.nunique() != EXPECTED_BRANCH_NOISE_ROWS
        or noise.trainer_seed.nunique() != 1
        or any(noise[column].nunique() != 1 for column in phase_columns)
    ):
        raise ValueError("Same-checkpoint branch-noise controls changed")
    metrics = pd.DataFrame(metric_rows).sort_values(["run_name", "metric"]).reset_index(drop=True)
    return results, metrics


def main() -> None:
    args = parse_args()
    fs, root = fsspec.core.url_to_fs(args.experiment_root)
    manifest_path, manifest = matching_full_manifest(
        fs,
        root,
        candidate_sha256=args.expected_candidate_sha256,
        continuation_sha256=args.expected_continuation_sha256,
        selected_prefixes_sha256=args.expected_selected_prefixes_sha256,
        prefix_replay_code_commit=args.prefix_replay_code_commit,
        branch_code_commit=args.branch_code_commit,
    )
    results, metrics = materialize_rows(
        fs,
        root,
        manifest,
        candidate_sha256=args.expected_candidate_sha256,
        continuation_sha256=args.expected_continuation_sha256,
        prefix_replay_code_commit=args.prefix_replay_code_commit,
        branch_code_commit=args.branch_code_commit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "branch_results.csv", index=False)
    results[results.fit_budget].to_csv(args.output_dir / "branch_fit_matrix.csv", index=False)
    metrics.to_csv(args.output_dir / "uncheatable_metrics_long.csv", index=False)
    coverage = {
        "branch_manifest_uri": gs_uri(manifest_path),
        "branch_manifest_sha256": hashlib.sha256(fs.cat(manifest_path)).hexdigest(),
        "result_rows": len(results),
        "fit_rows": int(results.fit_budget.sum()),
        "control_rows": int((~results.fit_budget).sum()),
        "metric_rows": len(metrics),
        "candidate_weights_sha256": args.expected_candidate_sha256,
        "continuation_weights_sha256": args.expected_continuation_sha256,
        "selected_prefixes_sha256": args.expected_selected_prefixes_sha256,
        "prefix_replay_code_commit": args.prefix_replay_code_commit,
        "branch_code_commit": args.branch_code_commit,
    }
    (args.output_dir / "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    print(json.dumps(coverage, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
