# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "tabulate", "wandb"]
# ///

"""Materialize the completed fixed-TPP StarCoder matched-compute ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_starcoder_wsd80_scale_bo_stage1_20260801 as scale_analysis,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    starcoder_wsd80_epoch_accounting as epoch_accounting,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    starcoder_wsd80_training_identity as stream_identity,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DESIGN_PATH = SCRIPT_DIR.parents[1] / "starcoder_wsd80_fixed_total_tpp5_diagonal_design_20260904.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_fixed_total_tpp5_diagonal_20260905"

TRAIN_PROJECT = "marin-community/marin"
TRAIN_TAG = "starcoder_wsd80_fixed_total_tpp5_diagonal"
EXPECTED_RUNS = 60
EXPECTED_CELLS = 4
EXPECTED_COORDINATES = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DESIGN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb-timeout", type=int, default=240)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def use_application_default_credentials_for_gcloud() -> None:
    """Route non-interactive GCS reads through the current ADC token."""
    result = subprocess.run(
        ["gcloud", "auth", "application-default", "print-access-token"],
        check=True,
        capture_output=True,
        text=True,
    )
    token = result.stdout.strip()
    if not token:
        raise ValueError("gcloud returned an empty application-default access token")
    os.environ["CLOUDSDK_AUTH_ACCESS_TOKEN"] = token


def load_design(path: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    design = json.loads(path.read_text(encoding="utf-8"))
    runs = pd.DataFrame(design["runs"])
    cells = pd.DataFrame(design["cells"])
    if design.get("expected_run_count") != EXPECTED_RUNS or len(runs) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} design rows")
    if design.get("cell_count") != EXPECTED_CELLS or len(cells) != EXPECTED_CELLS:
        raise ValueError(f"Expected {EXPECTED_CELLS} design cells")
    if design.get("coordinate_count_per_cell") != EXPECTED_COORDINATES:
        raise ValueError(f"Expected {EXPECTED_COORDINATES} coordinates per cell")
    if runs["run_name"].duplicated().any():
        raise ValueError("Design contains duplicate run names")
    cell_counts = runs.groupby("cell_id").size()
    if len(cell_counts) != EXPECTED_CELLS or not cell_counts.eq(EXPECTED_COORDINATES).all():
        raise ValueError(f"Unexpected runs per cell: {cell_counts.to_dict()}")
    if not np.allclose(runs["phase_0_starcoder"], runs["phase_1_starcoder"], rtol=0.0, atol=0.0):
        raise ValueError("Fixed-TPP panel contains a non-tied mixture")
    return design, runs, cells


def collect_observations(
    design: dict[str, Any],
    manifest: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    timeout: int,
    workers: int,
) -> pd.DataFrame:
    """Join every frozen design row to one durable terminal observation."""
    api = wandb.Api(timeout=timeout)
    runs = list(api.runs(TRAIN_PROJECT, filters={"tags": TRAIN_TAG}, per_page=100))
    by_name: dict[str, list[Any]] = {}
    for run in runs:
        by_name.setdefault(str(run.name), []).append(run)

    ordered_runs = []
    for run_name in manifest["run_name"]:
        candidates = by_name.get(str(run_name), [])
        if len(candidates) != 1:
            raise ValueError(f"{run_name}: expected exactly one W&B run, found {len(candidates)}")
        run = candidates[0]
        if str(run.state) != "finished":
            raise ValueError(f"{run_name}: W&B state is {run.state!r}, expected 'finished'")
        ordered_runs.append(run)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        metrics = list(executor.map(scale_analysis.persisted_final_metric, ordered_runs))

    observations = manifest.copy()
    observations["starcoder_weight"] = observations["phase_0_starcoder"].astype(float)
    observations["starcoder_bpb"] = [metric.value for metric in metrics]
    observations["final_metric_step"] = [metric.step for metric in metrics]
    observations["expected_final_metric_step"] = observations["total_steps"].astype(int) - 1
    observations["metric_uri"] = [metric.uri for metric in metrics]
    observations["metric_source"] = "persisted eval_metrics.jsonl"
    observations["wandb_id"] = [str(run.id) for run in ordered_runs]
    observations["wandb_state"] = [str(run.state) for run in ordered_runs]
    observations["wandb_url"] = [str(run.url) for run in ordered_runs]

    incomplete = observations.loc[observations["final_metric_step"].ne(observations["expected_final_metric_step"])]
    if not incomplete.empty:
        details = incomplete[["run_name", "final_metric_step", "expected_final_metric_step"]].to_dict("records")
        raise ValueError(f"Panel contains partial metrics: {details}")
    if not np.isfinite(observations["starcoder_bpb"].to_numpy(dtype=float)).all():
        raise ValueError("Panel contains non-finite BPB values")

    boundary_fraction = float(design["phase_0_fraction"])
    stream_digests = []
    for (_, row), run in zip(observations.iterrows(), ordered_runs, strict=True):
        expected_policy = [
            {"boundary_step": 0, "starcoder_weight": float(row["phase_0_starcoder"])},
            {
                "boundary_step": int(int(row["total_steps"]) * boundary_fraction),
                "starcoder_weight": float(row["phase_1_starcoder"]),
            },
        ]
        observed_policy = stream_identity.policy_coordinates(run.config)
        differences = stream_identity.identity_differences(observed_policy, expected_policy)
        if differences:
            raise ValueError(f"{row['run_name']}: persisted policy disagrees with the design: {differences}")
        stream_digests.append(stream_identity.canonical_sha256(stream_identity.wandb_stream_identity(run.config)))
    observations["stream_identity_sha256"] = stream_digests

    observations = observations.merge(cells, on="cell_id", validate="many_to_one", suffixes=("", "_cell"))
    for column in ("hidden_size", "num_layers", "total_steps", "materialized_tokens"):
        cell_column = f"{column}_cell"
        if not observations[column].eq(observations[cell_column]).all():
            raise ValueError(f"Run and cell metadata disagree for {column}")
        observations = observations.drop(columns=cell_column)

    coordinate_roles = {row["coordinate_id"]: row["role"] for row in design["coordinates"]}
    observations["coordinate_role"] = observations["coordinate_id"].map(coordinate_roles)
    observations["starcoder_epochs"] = [
        epoch_accounting.simulated_materialized_epochs(weight, weight).starcoder.total
        for weight in observations["starcoder_weight"]
    ]
    return observations.sort_values(["rung", "starcoder_weight"]).reset_index(drop=True)


def summarize_curves(observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell_id, group in observations.groupby("cell_id", sort=False):
        minimum = group.loc[group["starcoder_bpb"].idxmin()]
        metadata = group.iloc[0]
        rows.append(
            {
                "cell_id": cell_id,
                "rung": int(metadata["rung"]),
                "total_parameters": int(metadata["total_parameters"]),
                "non_embedding_parameters": int(metadata["non_embedding_parameters"]),
                "materialized_tokens": int(metadata["materialized_tokens"]),
                "compute_flops": float(metadata["compute_flops"]),
                "total_parameter_tpp": float(metadata["total_parameter_tpp"]),
                "non_embedding_tpp": float(metadata["non_embedding_tpp"]),
                "observed_optimum_weight": float(minimum["starcoder_weight"]),
                "observed_optimum_epochs": float(minimum["starcoder_epochs"]),
                "observed_optimum_bpb": float(minimum["starcoder_bpb"]),
            }
        )
    return pd.DataFrame(rows).sort_values("rung").reset_index(drop=True)


def write_report(output_dir: Path, observations: pd.DataFrame, curves: pd.DataFrame) -> None:
    lines = [
        "# StarCoder fixed-total-TPP-5 matched-compute diagonal",
        "",
        f"- Durable terminal observations: {len(observations)}/{EXPECTED_RUNS}.",
        f"- Complete curves: {len(curves)}/{EXPECTED_CELLS}, each with {EXPECTED_COORDINATES} coordinates.",
        "- BPB is read from each checkpoint's persisted final `eval_metrics.jsonl`; "
        "W&B supplies identity and provenance.",
        "- Every curve uses simulated epoching against the same target training budget.",
        "",
        "## Measured optima",
        "",
        curves.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    use_application_default_credentials_for_gcloud()
    design, manifest, cells = load_design(args.design)
    observations = collect_observations(
        design,
        manifest,
        cells,
        timeout=args.wandb_timeout,
        workers=args.workers,
    )
    curves = summarize_curves(observations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(args.output_dir / "observations.csv", index=False)
    curves.to_csv(args.output_dir / "curve_summary.csv", index=False)
    (args.output_dir / "source_design.json").write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
    manifest_payload = {
        "design_path": str(args.design),
        "design_sha256": file_sha256(args.design),
        "run_count": len(observations),
        "cell_count": len(curves),
        "metric_source": "persisted eval_metrics.jsonl",
        "train_project": TRAIN_PROJECT,
        "train_tag": TRAIN_TAG,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir, observations, curves)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
