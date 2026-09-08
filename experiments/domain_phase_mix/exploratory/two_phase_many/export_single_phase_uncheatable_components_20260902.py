# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "wandb"]
# ///
"""Freeze exact Uncheatable component outcomes for the older one-phase swarms.

The 60M and 300M canonical panel CSVs contain the published aggregate but not
its seven component BPBs. This exporter reads the exact endpoint runs from W&B,
checks every aggregate against the canonical panel, and writes a deterministic
local input table. Benchmark fitting never talks to W&B.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import os
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "single_phase_componentwise_canonical_dsp_20260902" / "input"
CANONICAL = REFERENCE_OUTPUTS / "two_phase_surrogate_collaborator_packet_20260721" / "data" / "canonical"
SIXTY_M_PANEL = REFERENCE_OUTPUTS / "60m_39bucket_checkpoint_audit_20260724" / "fit_single_phase.csv"
Q240_MANIFEST = (
    REFERENCE_OUTPUTS
    / "single_phase_exposure_average_qsplit240_300m_6b"
    / "single_phase_exposure_average_qsplit240_300m_manifest.csv"
)
PARITY_MANIFEST = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_full_results_60m_300m_20260625"
    / "olmo_base_easy_full_results_60m_300m_manifest.csv"
)
WANDB_PROJECT = "marin-community/marin"
UNCHEATABLE_AGGREGATE = "eval/uncheatable_eval/bpb"
UNCHEATABLE_COMPONENTS = (
    "eval/uncheatable_eval/ao3_english/bpb",
    "eval/uncheatable_eval/arxiv_computer_science/bpb",
    "eval/uncheatable_eval/arxiv_physics/bpb",
    "eval/uncheatable_eval/bbc_news/bpb",
    "eval/uncheatable_eval/github_cpp/bpb",
    "eval/uncheatable_eval/github_python/bpb",
    "eval/uncheatable_eval/wikipedia_english/bpb",
)
EXPECTED_ROWS = {"60m": 242, "300m": 280}
AGGREGATE_TOLERANCE = 2e-6


@dataclasses.dataclass(frozen=True)
class RunSpec:
    row_id: str
    wandb_run_id: str
    expected_aggregate: float


def _checkpoint_run_id(checkpoint_root: str) -> str:
    run_id = checkpoint_root.rstrip("/").rsplit("/", 1)[-1]
    if "-" not in run_id:
        raise ValueError(f"Checkpoint root does not end in a W&B run id: {checkpoint_root}")
    return run_id


def _sixty_m_specs() -> list[RunSpec]:
    frame = pd.read_csv(SIXTY_M_PANEL)
    return [
        RunSpec(
            row_id=str(row.run_name),
            wandb_run_id=_checkpoint_run_id(str(row.checkpoint_root)),
            expected_aggregate=float(row.uncheatable_bpb),
        )
        for row in frame.itertuples(index=False)
    ]


def _q240_run_ids() -> dict[str, str]:
    expected_names = set(pd.read_csv(Q240_MANIFEST)["run_name"].astype(str))
    api = wandb.Api(timeout=90)
    runs = api.runs(
        WANDB_PROJECT,
        filters={"display_name": {"$regex": "spavg_q240_300m"}},
        per_page=100,
    )
    mapping = {str(run.name).rsplit("/", 1)[-1]: str(run.id) for run in runs}
    if set(mapping) != expected_names:
        missing = sorted(expected_names - set(mapping))
        unexpected = sorted(set(mapping) - expected_names)
        raise ValueError(f"300M q240 W&B identity mismatch; missing={missing}, unexpected={unexpected}")
    return mapping


def _three_hundred_m_specs() -> list[RunSpec]:
    canonical = pd.read_csv(CANONICAL / "300m_one_phase_fit.csv")
    expected = dict(zip(canonical["row_id"].astype(str), canonical["uncheatable_bpb"].astype(float), strict=True))
    mapping = _q240_run_ids()

    manifest = pd.read_csv(PARITY_MANIFEST)
    legacy = manifest[
        manifest["scale"].eq("300m_6b")
        & (
            manifest["run_name"].astype(str).str.startswith("pctrl_del_")
            | manifest["run_name"].eq("baseline_stratified")
        )
    ]
    for row in legacy.itertuples(index=False):
        mapping[f"singleavg_{row.run_name}"] = str(row.wandb_run_id)

    if set(mapping) != set(expected):
        missing = sorted(set(expected) - set(mapping))
        unexpected = sorted(set(mapping) - set(expected))
        raise ValueError(f"300M canonical identity mismatch; missing={missing}, unexpected={unexpected}")
    return [
        RunSpec(row_id=row_id, wandb_run_id=mapping[row_id], expected_aggregate=value)
        for row_id, value in expected.items()
    ]


def _fetch_run(api: wandb.Api, spec: RunSpec) -> dict[str, object]:
    run = api.run(f"{WANDB_PROJECT}/{spec.wandb_run_id}")
    summary = dict(run.summary)
    keys = (UNCHEATABLE_AGGREGATE, *UNCHEATABLE_COMPONENTS)
    missing = [key for key in keys if key not in summary]
    if missing:
        raise ValueError(f"{spec.wandb_run_id}: missing endpoint metrics {missing}")
    row: dict[str, object] = {
        "row_id": spec.row_id,
        "wandb_run_id": spec.wandb_run_id,
        UNCHEATABLE_AGGREGATE: float(summary[UNCHEATABLE_AGGREGATE]),
    }
    row.update({key: float(summary[key]) for key in UNCHEATABLE_COMPONENTS})
    error = abs(float(row[UNCHEATABLE_AGGREGATE]) - spec.expected_aggregate)
    if error > AGGREGATE_TOLERANCE:
        raise ValueError(
            f"{spec.row_id}: endpoint aggregate differs from canonical panel by {error:.3g} "
            f"({row[UNCHEATABLE_AGGREGATE]} versus {spec.expected_aggregate})"
        )
    return row


def export_panel(name: str, specs: list[RunSpec], output_dir: Path, workers: int) -> Path:
    if len(specs) != EXPECTED_ROWS[name] or len({spec.row_id for spec in specs}) != len(specs):
        raise ValueError(f"{name}: expected {EXPECTED_ROWS[name]} unique run specifications")
    api = wandb.Api(timeout=90)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(lambda spec: _fetch_run(api, spec), specs))
    by_row = {str(row["row_id"]): row for row in rows}
    frame = pd.DataFrame([by_row[spec.row_id] for spec in specs])
    numeric = frame[[UNCHEATABLE_AGGREGATE, *UNCHEATABLE_COMPONENTS]].to_numpy(float)
    if not np.isfinite(numeric).all() or np.any(numeric <= 0.0):
        raise ValueError(f"{name}: invalid component payload")
    path = output_dir / f"{name}_uncheatable_components.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    print(f"wrote {len(frame)} exact {name} rows to {path}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panels", default="60m,300m")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    builders = {"60m": _sixty_m_specs, "300m": _three_hundred_m_specs}
    selected = tuple(part.strip() for part in args.panels.split(",") if part.strip())
    unknown = set(selected) - set(builders)
    if unknown:
        raise ValueError(f"Unknown panels: {sorted(unknown)}")
    for panel in selected:
        export_panel(panel, builders[panel](), args.output_dir, args.workers)


if __name__ == "__main__":
    main()
