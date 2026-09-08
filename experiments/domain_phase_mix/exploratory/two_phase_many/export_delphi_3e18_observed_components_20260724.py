# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec", "gcsfs", "numpy", "pandas", "wandb>=0.19"]
# ///
"""Export component outcomes for the Delphi aggregate/order identification panels.

The exporter joins three already completed panels:

* the canonical 280-row two-phase fit swarm;
* the matched 280-row phase-tied swarm, including 42 exact aliases;
* the 200-row controlled frontier phase-fiber panel.

W&B summaries supply the seven Uncheatable components and the native 51
Table-9 components. Each fetched summary is cached independently so a rerun
only requests missing runs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    build_delphi_3e18_heldout_registry as heldouts,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_delphi_3e18_augmented_swarm as fit_export,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_3e18_observed_components_20260724"
FIT_DATA = REFERENCE_OUTPUTS / "delphi_augmented_swarm_3e18_20260714/delphi_augmented_swarm_3e18_wide.csv"
HELDOUT_DATA = REFERENCE_OUTPUTS / "delphi_3e18_append_only_heldouts_20260714/heldout_current.csv"
ONE_PHASE_DIR = REFERENCE_OUTPUTS / "delphi_one_phase_augmented_swarm_3e18_20260715"
ONE_PHASE_MANIFEST = ONE_PHASE_DIR / "training_manifest.csv"
ONE_PHASE_WEIGHTS = ONE_PHASE_DIR / "phase_weights.csv"
FIBER_DIR = REFERENCE_OUTPUTS / "delphi_3e18_frontier_phase_fiber_20260719"
FIBER_WEIGHTS = FIBER_DIR / "phase_weights.csv"
FIBER_RESULTS = REFERENCE_OUTPUTS / "delphi_3e18_frontier_phase_fiber_results_20260719/observed_results.csv"

TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
ONE_PHASE_SERIES = "delphi_one_phase_augmented_swarm_3e18_20260715"
EXPECTED_COUNTS = {
    "two_phase_fit": 280,
    "one_phase_fit": 280,
    "frontier_phase_fiber": 200,
}
UNCHEATABLE_TASKS = (
    "ao3_english",
    "arxiv_computer_science",
    "arxiv_physics",
    "bbc_news",
    "github_cpp",
    "github_python",
    "wikipedia_english",
)
UNCHEATABLE_COMPONENTS = tuple(f"eval/uncheatable_eval/{task}/bpb" for task in UNCHEATABLE_TASKS)
UNCHEATABLE_MACRO_KEYS = ("eval/uncheatable_eval/bpb", "eval/uncheatable_eval/macro_bpb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--wandb-timeout", type=int, default=180)
    return parser.parse_args()


def finite_value(summary: Mapping[str, object], key: str) -> float:
    value = summary.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Missing finite W&B summary value {key!r}")
    return float(value)


def run_id_from_url(url: str) -> str:
    value = str(url).rstrip("/").rsplit("/", maxsplit=1)[-1]
    if not value:
        raise ValueError(f"Cannot parse W&B run id from {url!r}")
    return value


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return {str(key): dict(value) for key, value in payload.items()}


def write_cache(path: Path, cache: Mapping[str, Mapping[str, Any]]) -> None:
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def fetch_summaries(
    api: wandb.Api,
    project: str,
    run_ids: Sequence[str],
    keys: Sequence[str],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    batch_size: int,
) -> None:
    unique_ids = sorted(set(run_ids))
    missing = [
        run_id
        for run_id in unique_ids
        if not all(key in cache.get(f"{project}/{run_id}", {}).get("values", {}) for key in keys)
    ]
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        for run in heldouts.batch_runs(api, project, batch, batch_size):
            summary = dict(run.summary)
            cache[f"{project}/{run.id}"] = {
                "run_id": run.id,
                "run_name": run.name,
                "run_url": run.url,
                "state": run.state,
                "values": {key: finite_value(summary, key) for key in keys},
            }
        write_cache(cache_path, cache)


def long_weights(
    frame: pd.DataFrame,
    row_key: str,
    phase_column: str,
    domains: Sequence[str],
    *,
    phase_values: tuple[object, object],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if frame.duplicated([row_key, phase_column, "domain"]).any():
        raise ValueError(f"Duplicate long-form weight key in {row_key}/{phase_column}")
    lookup = frame.set_index([row_key, phase_column, "domain"])["weight"]
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in frame[row_key].drop_duplicates():
        phase0 = np.asarray([lookup.loc[(name, phase_values[0], domain)] for domain in domains], dtype=float)
        phase1 = np.asarray([lookup.loc[(name, phase_values[1], domain)] for domain in domains], dtype=float)
        if not np.allclose([phase0.sum(), phase1.sum()], 1.0, atol=1e-10):
            raise ValueError(f"Policy weights do not sum to one for {name}")
        result[str(name)] = (phase0, phase1)
    return result


def weight_record(domains: Sequence[str], phase0: np.ndarray, phase1: np.ndarray) -> dict[str, float]:
    return {
        **{f"phase_0_{domain}": float(phase0[index]) for index, domain in enumerate(domains)},
        **{f"phase_1_{domain}": float(phase1[index]) for index, domain in enumerate(domains)},
    }


def source_records() -> tuple[list[str], list[str], list[dict[str, Any]]]:
    fit = pd.read_csv(FIT_DATA)
    domains = [
        column.removeprefix("phase_0_")
        for column in fit.columns
        if column.startswith("phase_0_") and f"phase_1_{column.removeprefix('phase_0_')}" in fit.columns
    ]
    table9_components = fit_export.table9_component_fields(fit.columns)
    if len(fit) != EXPECTED_COUNTS["two_phase_fit"] or len(domains) != 39:
        raise ValueError("Canonical Delphi fit data has unexpected dimensions")

    records: list[dict[str, Any]] = []
    fit_by_source = {str(row["run_name"]): row for _, row in fit.iterrows()}
    for _, row in fit.iterrows():
        phase0 = row[[f"phase_0_{domain}" for domain in domains]].to_numpy(dtype=float)
        phase1 = row[[f"phase_1_{domain}" for domain in domains]].to_numpy(dtype=float)
        records.append(
            {
                "panel": "two_phase_fit",
                "row_name": str(row["run_name"]),
                "source_row_name": str(row["run_name"]),
                "training_run_id": str(row["training_wandb_run_id"]),
                "table9_eval_run_id": str(row["table9_eval_run_id"]),
                "uncheatable_bpb": float(row["uncheatable_bpb"]),
                "table9_macro_bpb": float(row["table9_macro_bpb"]),
                **{component: float(row[component]) for component in table9_components},
                **weight_record(domains, phase0, phase1),
            }
        )

    heldout = pd.read_csv(HELDOUT_DATA)
    heldout = heldout.loc[heldout["training_series"].eq(ONE_PHASE_SERIES)].copy()
    heldout_by_base = {str(row["wandb_run_base"]): row for _, row in heldout.iterrows()}
    manifest = pd.read_csv(ONE_PHASE_MANIFEST).sort_values("run_order")
    one_phase_weights = long_weights(
        pd.read_csv(ONE_PHASE_WEIGHTS),
        "run_name",
        "phase",
        domains,
        phase_values=("phase_0", "phase_1"),
    )
    for _, source in manifest.iterrows():
        row_name = str(source["run_name"])
        source_name = str(source["source_run_name"])
        phase0, phase1 = one_phase_weights[row_name]
        if not np.allclose(phase0, phase1, atol=1e-12):
            raise ValueError(f"One-phase policy {row_name} is not tied")
        if source["disposition"] == "reused_exact_phase_tied_alias":
            row = fit_by_source[source_name]
            record = {
                "panel": "one_phase_fit",
                "row_name": row_name,
                "source_row_name": source_name,
                "training_run_id": str(row["training_wandb_run_id"]),
                "table9_eval_run_id": str(row["table9_eval_run_id"]),
                "uncheatable_bpb": float(row["uncheatable_bpb"]),
                "table9_macro_bpb": float(row["table9_macro_bpb"]),
                **{component: float(row[component]) for component in table9_components},
            }
        elif source["disposition"] == "scheduled_new_training":
            row = heldout_by_base[row_name]
            record = {
                "panel": "one_phase_fit",
                "row_name": row_name,
                "source_row_name": source_name,
                "training_run_id": str(row["wandb_run_id"]),
                "table9_eval_run_id": str(row["table9_eval_run_id"]),
                "uncheatable_bpb": float(row["uncheatable_bpb"]),
                "table9_macro_bpb": float(row["table9_macro_bpb"]),
            }
        else:
            raise ValueError(f"Unknown one-phase disposition {source['disposition']!r}")
        records.append({**record, **weight_record(domains, phase0, phase1)})

    fiber = pd.read_csv(FIBER_RESULTS)
    fiber_weights = long_weights(
        pd.read_csv(FIBER_WEIGHTS),
        "candidate_id",
        "phase",
        domains,
        phase_values=(0, 1),
    )
    for _, row in fiber.iterrows():
        row_name = str(row["candidate_id"])
        phase0, phase1 = fiber_weights[row_name]
        records.append(
            {
                "panel": "frontier_phase_fiber",
                "row_name": row_name,
                "source_row_name": str(row["anchor_source_run_name"]),
                "anchor_id": str(row["anchor_id"]),
                "contrast_family": str(row["contrast_family"]),
                "direction_id": str(row["direction_id"]),
                "sign": str(row["sign"]),
                "seed_block": int(row["seed_block"]),
                "training_run_id": run_id_from_url(str(row["training_wandb_url"])),
                "table9_eval_run_id": run_id_from_url(str(row["eval_wandb_url"])),
                "uncheatable_bpb": float(row["uncheatable_bpb"]),
                "table9_macro_bpb": float(row["table9_macro_bpb"]),
                **weight_record(domains, phase0, phase1),
            }
        )

    counts = pd.Series([record["panel"] for record in records]).value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected source-panel composition: {counts}")
    return domains, table9_components, records


def add_component_values(
    records: list[dict[str, Any]],
    table9_components: Sequence[str],
    cache: Mapping[str, Mapping[str, Any]],
) -> None:
    for record in records:
        training = cache[f"{TRAIN_PROJECT}/{record['training_run_id']}"]["values"]
        evaluation = cache[f"{EVAL_PROJECT}/{record['table9_eval_run_id']}"]["values"]
        for component in UNCHEATABLE_COMPONENTS:
            record[component] = float(training[component])
        for component in table9_components:
            record[component] = float(evaluation[fit_export.native_component_key(component)])

        table9_macro = float(np.mean([record[component] for component in table9_components]))
        if not math.isclose(table9_macro, float(record["table9_macro_bpb"]), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"Table-9 component mean mismatch for {record['row_name']}")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "component_summary_cache.json"
    cache = load_cache(cache_path)
    domains, table9_components, records = source_records()

    api = wandb.Api(timeout=args.wandb_timeout)
    fetch_summaries(
        api,
        TRAIN_PROJECT,
        [str(record["training_run_id"]) for record in records],
        (*UNCHEATABLE_COMPONENTS, *UNCHEATABLE_MACRO_KEYS),
        cache,
        cache_path,
        args.batch_size,
    )
    fetch_summaries(
        api,
        EVAL_PROJECT,
        [str(record["table9_eval_run_id"]) for record in records],
        tuple(fit_export.native_component_key(component) for component in table9_components),
        cache,
        cache_path,
        args.batch_size,
    )
    add_component_values(records, table9_components, cache)

    frame = pd.DataFrame.from_records(records)
    output_path = args.output_dir / "observed_component_panel.csv"
    frame.to_csv(output_path, index=False)
    summary = {
        "row_count": len(frame),
        "panel_counts": frame["panel"].value_counts().sort_index().to_dict(),
        "domain_count": len(domains),
        "uncheatable_component_count": len(UNCHEATABLE_COMPONENTS),
        "table9_component_count": len(table9_components),
        "unique_training_runs": int(frame["training_run_id"].nunique()),
        "unique_table9_eval_runs": int(frame["table9_eval_run_id"].nunique()),
        "output_csv": str(output_path),
        "cache_json": str(cache_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
