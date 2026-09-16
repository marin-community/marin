# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Expanded retrospective bank: the frozen registry plus every completed same-scale validation launch.

Every launch of ``collect_delphi_3e18_validation_results_20260906`` trains the Qwen3 360M/1.6B configuration of
the swarm with a single-phase mixture, so its measured runs are bank coordinates in the sense of
``prepare_single_phase_heldout_benchmark_20260902``: the builder's coordinate ids, fit-panel overlap rule and
aggregation are reused unchanged, and each launch becomes one source ``validation::<launch>``. The frozen registry
directory is left untouched; the expanded one is written beside it with the same four files, so the learning-curve
scorer can point ``benchmark.HELDOUT_DIR`` at either.

usage: uv run --offline --no-sync python expand_heldout_bank_20260913.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    collect_delphi_3e18_validation_results_20260906 as collector,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    prepare_single_phase_heldout_benchmark_20260902 as builder,
)

PANEL = "delphi_3e18_39bucket"
FROZEN_DIR = builder.DEFAULT_OUTPUT_DIR
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "single_phase_heldout_benchmark_expanded_20260913"
CUTOFF = "2026-09-13"
SOURCE_PREFIX = "validation::"
REGISTRY_FILES = ("heldout_runs.csv", "heldout_coordinates.csv", "heldout_coordinate_components.csv")


def run_key(launch: collector.Launch, row: pd.Series) -> str:
    """Key of a run in the launch's Table-9 component table: ``<candidate>@t<seed>`` for grouped launches."""
    if launch.grouped:
        return f"{row['candidate_id']}@t{int(row['trainer_seed'])}"
    return str(row["candidate_id"])


def launch_runs(
    launch: collector.Launch, buckets: tuple[str, ...], fit_weights: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audited run rows and per-run component records of one launch's measured runs."""
    results = pd.read_csv(launch.output_dir / "measured_results.csv")
    results = results[results["status"].eq("measured")].reset_index(drop=True)
    weight_rows = pd.concat([pd.read_csv(path) for path in launch.candidate_tables], ignore_index=True)
    components = pd.read_csv(launch.output_dir / "measured_table9_components.csv")
    table9_order = builder.table9_components()
    records: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    component_records: list[dict[str, object]] = []
    for _, result in results.iterrows():
        selected = weight_rows[weight_rows["candidate_id"].eq(result["candidate_id"])]
        if "target" in selected and "target" in result and pd.notna(result["target"]):
            with_target = selected[selected["target"].eq(result["target"])]
            selected = with_target if len(with_target) else selected
        if len(selected) != 39 or selected["domain"].nunique() != 39:
            raise ValueError(f"{launch.name}/{result['candidate_id']}: incomplete candidate weights")
        by_domain = dict(zip(selected["domain"], selected["weight"], strict=True))
        vectors.append(np.asarray([by_domain[bucket] for bucket in buckets], dtype=float))
        trainer_seed = result.get("trainer_seed", np.nan)
        row_id = f"delphi_validation::{launch.name}::{run_key(launch, result)}"
        records.append(
            {
                "panel": PANEL,
                "scale": builder.PANEL_SCALE[PANEL],
                "row_id": row_id,
                "source": f"{SOURCE_PREFIX}{launch.name}",
                "source_row_id": str(result["candidate_id"]),
                "source_experiment": launch.experiment_root,
                "proposal_model": launch.name,
                "proposal_target": str(result["target"]),
                "epoch_cap": result["epoch_cap"],
                "training_wandb_run_id": "",
                "training_wandb_url": "",
                "table9_eval_run_id": builder._wandb_id(result.get("table9_wandb_url", ""), ""),
                "table9_eval_url": result.get("table9_wandb_url", ""),
                "data_seed": result.get("data_seed", np.nan),
                "trainer_seed": trainer_seed,
                "phase_tv": 0.0,
                "uncheatable_bpb": result["uncheatable_bpb"],
                "table9_macro_bpb": result["table9_macro_bpb"],
            }
        )
        for position, component in enumerate(builder.UNCHEATABLE_COMPONENTS):
            column = "uncheatable_" + component.removeprefix("eval/uncheatable_eval/").removesuffix("/bpb") + "_bpb"
            component_records.append(
                builder._component_record(
                    row_id, PANEL, "uncheatable", position, component, result[column], "local_collector_endpoint"
                )
            )
        own = components[components["candidate_id"].eq(run_key(launch, result))]
        if len(own) != len(table9_order):
            raise ValueError(f"{launch.name}/{row_id}: {len(own)} Table-9 components, expected {len(table9_order)}")
        pairs = zip(own["component"], own["bpb"], strict=True)
        values = {builder._canonical_table9_component(name): value for name, value in pairs}
        for position, component in enumerate(table9_order):
            component_records.append(
                builder._component_record(
                    row_id, PANEL, "table9", position, component, values[component], "local_collector_table9"
                )
            )
    audited = builder._finalize_audit(pd.DataFrame(records), np.stack(vectors), fit_weights)
    return audited, pd.DataFrame(component_records)


def merge_components(frozen: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Coordinate components: frozen rows kept, new coordinates added, shared coordinates pooled by run count."""
    keys = ["panel", "coordinate_id", "target", "component_position", "component"]
    shared = frozen.merge(fresh, on=keys, suffixes=("_old", "_new"))
    if len(shared):
        total = shared["run_count_old"] + shared["run_count_new"]
        shared["bpb_mean"] = (
            shared["bpb_mean_old"] * shared["run_count_old"] + shared["bpb_mean_new"] * shared["run_count_new"]
        ) / total
        shared["bpb_sd"] = np.nan
        shared["run_count"] = total
        shared = shared[[*keys, "bpb_mean", "bpb_sd", "run_count"]]
    shared_keys = set(map(tuple, shared[keys].to_numpy())) if len(shared) else set()
    keep_frozen = frozen[[tuple(row) not in shared_keys for row in frozen[keys].to_numpy()]]
    keep_fresh = fresh[[tuple(row) not in shared_keys for row in fresh[keys].to_numpy()]]
    return (
        pd.concat([keep_frozen, keep_fresh, shared], ignore_index=True)
        .sort_values(["panel", "coordinate_id", "target", "component_position"])
        .reset_index(drop=True)
    )


def main() -> None:
    buckets = builder.domains()
    fit_weights = builder._fit_weights(PANEL, buckets)
    frozen_runs = pd.read_csv(FROZEN_DIR / "heldout_runs.csv")
    frozen_coordinates = pd.read_csv(FROZEN_DIR / "heldout_coordinates.csv")
    frozen_components = pd.read_csv(FROZEN_DIR / "heldout_coordinate_components.csv")
    audits, components = [], []
    for launch in collector.LAUNCHES.values():
        audited, component_records = launch_runs(launch, buckets, fit_weights)
        audits.append(audited)
        components.append(component_records)
    fresh = pd.concat(audits, ignore_index=True)
    fresh_components = pd.concat(components, ignore_index=True)
    if fresh["row_id"].duplicated().any():
        raise ValueError("duplicate run ids among the validation launches")
    if set(fresh["row_id"]) & set(frozen_runs["row_id"]):
        raise ValueError("validation run ids collide with the frozen registry")
    runs = pd.concat([frozen_runs, fresh], ignore_index=True)
    eligible = runs[runs["eligible"].astype(bool)]
    coordinates = builder.coordinate_table(eligible)
    fresh_eligible = fresh[fresh["eligible"].astype(bool)]
    fresh_coordinate_components = builder.coordinate_component_table(
        fresh_eligible, fresh_components[fresh_components["row_id"].isin(fresh_eligible["row_id"])]
    )
    coordinate_components = merge_components(frozen_components, fresh_coordinate_components)
    # The frozen coordinates must come out unchanged where no new run landed on them.
    frozen_ids = set(frozen_coordinates["coordinate_id"])
    new_ids = set(fresh_eligible["coordinate_id"])
    untouched = coordinates[coordinates["coordinate_id"].isin(frozen_ids - new_ids)].set_index("coordinate_id")
    reference = frozen_coordinates.set_index("coordinate_id").loc[untouched.index]
    for column in ("uncheatable_n", "uncheatable_mean_bpb", "table9_macro_n", "table9_macro_mean_bpb"):
        if not np.allclose(untouched[column].fillna(-1), reference[column].fillna(-1), atol=1e-12):
            raise ValueError(f"frozen coordinates changed in {column}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUTPUT_DIR / "heldout_runs.csv", index=False)
    coordinates.to_csv(OUTPUT_DIR / "heldout_coordinates.csv", index=False)
    coordinate_components.to_csv(OUTPUT_DIR / "heldout_coordinate_components.csv", index=False)
    delphi = coordinates[coordinates["panel"].eq(PANEL)]
    summary = {
        "schema_version": "expanded-1",
        "frozen_registry": str(FROZEN_DIR.relative_to(REPO_ROOT)),
        "frozen_hashes": {name: builder.file_sha256(FROZEN_DIR / name) for name in (*REGISTRY_FILES, "manifest.json")},
        "cutoff": CUTOFF,
        "built": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "launches": {launch.name: launch.experiment_root for launch in collector.LAUNCHES.values()},
        "validation_runs": len(fresh),
        "validation_runs_eligible": int(fresh["eligible"].sum()),
        "validation_exclusions": fresh["exclusion_reason"].value_counts().to_dict(),
        "validation_new_coordinates": len(new_ids - frozen_ids),
        "validation_shared_coordinates": len(new_ids & frozen_ids),
        "delphi_runs_eligible": int(eligible["panel"].eq(PANEL).sum()),
        "delphi_coordinates": len(delphi),
        "delphi_uncheatable_coordinates": int(delphi["uncheatable_n"].fillna(0).gt(0).sum()),
        "delphi_table9_coordinates": int(delphi["table9_macro_n"].fillna(0).gt(0).sum()),
        "sources": eligible[eligible["panel"].eq(PANEL)]["source"].value_counts().to_dict(),
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("sources", "launches", "frozen_hashes")}, indent=1))
    print(pd.Series(summary["sources"]).to_string())


if __name__ == "__main__":
    main()
