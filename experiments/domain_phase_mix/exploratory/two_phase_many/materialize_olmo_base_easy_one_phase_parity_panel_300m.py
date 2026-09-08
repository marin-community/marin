# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "wandb", "cvxpy", "plotly", "scipy"]
# ///
"""Materialize the one-phase OLMoBaseEval Easy/Table-9 parity fit panel.

The panel is meant for apples-to-apples one-phase OLMix/DSP fitting:

* 240 one-phase qsplit rows from the exposure-average 300M run.
* 1 shared stratified baseline whose weights were already phase-constant.
* 39 phase-constant 300M domain-deletion controls reused from proportional
  controllability.
* 11 proportional observations used only to replace the single proportional
  row's Table-9 targets with the proportional reference mean.

No live jobs are launched here. The script only reads W&B/local artifacts and
writes a reproducible local CSV bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_paper_faithful_olmix_300m as paper_olmix,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "olmo_base_easy_one_phase_parity_panel_300m_20260628"
SINGLE_PHASE_MANIFEST = (
    SCRIPT_DIR
    / "reference_outputs"
    / "single_phase_exposure_average_qsplit240_300m_6b"
    / "single_phase_exposure_average_qsplit240_300m_manifest.csv"
)
SINGLE_PHASE_TARGET_MANIFEST = (
    REPO_ROOT.parent
    / "marin-olmo-base-eval-table9"
    / "experiments"
    / "evals"
    / "olmo_base_eval_table9_single_phase_qsplit240_300m_targets.tsv"
)

WANDB_PROJECT = "marin-community/marin-eval"
WANDB_GROUP = "olmo_base_eval_table9_single_phase_qsplit240_300m"
N_EXPECTED_SINGLE_PHASE_ROWS = 240
N_EXPECTED_SHARED_STRATIFIED_ROWS = 1
N_EXPECTED_DELETION_ROWS = 39
N_EXPECTED_PROPORTIONAL_REFERENCES = 11
N_EXPECTED_COMPONENTS = 51
PHASE_TIE_TOL = 1e-12
MACRO_KEYS = (
    "olmo_base_easy/table9_51_component_macro_bpb",
    "olmo_base_easy/table9_macro_bpb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-group", default=WANDB_GROUP)
    return parser.parse_args()


def native_component_key(component: str) -> str:
    prefix = "olmo_base_eval/easy_bpb/"
    suffix = "/bpb"
    if component.startswith(prefix) and component.endswith(suffix):
        task = component.removeprefix(prefix).removesuffix(suffix)
        return f"olmo_base_easy/table9/{task}/bpb"
    return f"olmo_base_easy/table9/{component}/bpb"


def summary_value(summary: Any, key: str) -> Any:
    try:
        return summary.get(key)
    except AttributeError:
        return None


def first_available_summary_value(summary: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = summary_value(summary, key)
        if value is not None:
            return value
    return None


def validate_phase_tied(frame: pd.DataFrame, columns: list[str], *, label: str) -> float:
    domains = base.domain_names_from_phase_columns(columns)
    phase0 = frame[[f"phase_0_{domain}" for domain in domains]].astype(float).to_numpy()
    phase1 = frame[[f"phase_1_{domain}" for domain in domains]].astype(float).to_numpy()
    max_delta = float(np.max(np.abs(phase0 - phase1))) if len(frame) else 0.0
    if max_delta > PHASE_TIE_TOL:
        raise ValueError(f"{label} is not phase-tied: max |phase0-phase1| = {max_delta}")
    return max_delta


def collect_single_phase_wandb(
    *,
    project: str,
    group: str,
    components: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    api = wandb.Api()
    rows: list[dict[str, Any]] = []
    skipped_without_macro = 0
    native_key_by_component = {component: native_component_key(component) for component in components}
    runs = list(api.runs(project, filters={"group": group}, per_page=400))
    for run in runs:
        macro = first_available_summary_value(run.summary, MACRO_KEYS)
        if macro is None:
            skipped_without_macro += 1
            continue
        provenance = run.config.get("provenance") or {}
        run_name = provenance.get("run_name")
        if not run_name:
            raise ValueError(f"W&B run {run.id} has Table-9 macro but no provenance.run_name")
        row: dict[str, Any] = {
            "run_name": str(run_name),
            "eval_source_run_name": provenance.get("source_run_name"),
            "eval_source_run_id": provenance.get("source_run_id"),
            "eval_target_name": run.name,
            "wandb_run_id": run.id,
            "wandb_url": run.url,
            "wandb_state": run.state,
            "wandb_created_at": getattr(run, "created_at", "") or "",
            "wandb_updated_at": getattr(run, "updated_at", "") or "",
            "native_table9_macro_bpb": float(macro),
        }
        missing_components: list[str] = []
        for component, native_key in native_key_by_component.items():
            value = summary_value(run.summary, native_key)
            if value is None:
                missing_components.append(native_key)
                continue
            row[component] = float(value)
        if missing_components:
            raise ValueError(
                f"W&B run {run.id} ({run.name}) is missing Table-9 component keys: " f"{missing_components[:10]}"
            )
        computed_macro = float(np.mean([row[component] for component in components]))
        row["table9_macro_bpb"] = computed_macro
        if abs(computed_macro - float(macro)) > 1e-8:
            raise ValueError(f"W&B run {run.id} macro mismatch: component mean {computed_macro} vs summary {macro}")
        rows.append(row)

    if not rows:
        raise ValueError(f"No W&B runs with Table-9 macro found in group {group}")
    all_rows = pd.DataFrame(rows)
    all_rows = all_rows.sort_values(
        ["run_name", "wandb_created_at", "wandb_run_id"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    duplicate_mask = all_rows.duplicated("run_name", keep=False)
    duplicate_log = all_rows.loc[duplicate_mask].copy()
    duplicate_log["dedupe_action"] = "drop"
    latest_indices = all_rows.groupby("run_name", sort=False).tail(1).index
    duplicate_log.loc[duplicate_log.index.isin(latest_indices), "dedupe_action"] = "keep"
    deduped = all_rows.loc[latest_indices].sort_values("run_name", kind="mergesort").reset_index(drop=True)

    if len(deduped) != N_EXPECTED_SINGLE_PHASE_ROWS:
        raise ValueError(f"Expected {N_EXPECTED_SINGLE_PHASE_ROWS} deduped single-phase rows, found {len(deduped)}")
    metadata = {
        "wandb_project": project,
        "wandb_group": group,
        "wandb_runs_seen": len(runs),
        "wandb_runs_with_table9_macro": len(all_rows),
        "wandb_runs_skipped_without_table9_macro": int(skipped_without_macro),
        "deduped_single_phase_rows": len(deduped),
        "duplicate_run_name_groups": int(all_rows.loc[duplicate_mask, "run_name"].nunique()),
        "duplicate_rows": len(duplicate_log),
        "dedupe_rule": "sort by run_name, wandb_created_at, wandb_run_id; keep latest per provenance run_name",
    }
    return deduped, duplicate_log, metadata


def load_single_phase_panel(components: list[str], columns: list[str], wandb_rows: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_csv(SINGLE_PHASE_MANIFEST, low_memory=False)
    if len(manifest) != N_EXPECTED_SINGLE_PHASE_ROWS:
        raise ValueError(f"Expected {N_EXPECTED_SINGLE_PHASE_ROWS} manifest rows, found {len(manifest)}")
    target_manifest = pd.read_csv(SINGLE_PHASE_TARGET_MANIFEST, sep="\t")
    if len(target_manifest) != N_EXPECTED_SINGLE_PHASE_ROWS:
        raise ValueError(f"Expected {N_EXPECTED_SINGLE_PHASE_ROWS} target manifest rows, found {len(target_manifest)}")
    if set(target_manifest["run_name"]) != set(manifest["run_name"]):
        missing_targets = sorted(set(manifest["run_name"]).difference(target_manifest["run_name"]))
        extra_targets = sorted(set(target_manifest["run_name"]).difference(manifest["run_name"]))
        raise ValueError(
            "One-phase target manifest does not match training manifest: "
            f"missing_targets={missing_targets[:10]} extra_targets={extra_targets[:10]}"
        )
    validate_phase_tied(manifest, columns, label="single-phase manifest")
    if manifest["run_name"].str.contains("olmix", case=False, na=False).any():
        leaked = manifest.loc[manifest["run_name"].str.contains("olmix", case=False, na=False), "run_name"].tolist()
        raise ValueError(f"One-phase manifest contains adaptive OLMix leakage rows: {leaked}")

    keep_manifest = [
        "run_id",
        "run_name",
        "cohort",
        "model_family",
        "data_seed",
        "source_run_id",
        "source_run_name",
        "source_two_phase_experiment",
        "single_phase_strategy",
        "source_panel",
        "phase_tv",
        "scale",
        "scale_display_label",
        "target_budget",
        "target_final_checkpoint_step",
        *columns,
    ]
    signal = manifest[keep_manifest].copy()
    signal["source_experiment"] = signal["source_two_phase_experiment"]
    signal["panel_source"] = "single_phase_qsplit_signal"
    merged = signal.merge(wandb_rows, on="run_name", how="left", validate="one_to_one")
    missing = merged.loc[merged["table9_macro_bpb"].isna(), "run_name"].tolist()
    if missing:
        raise ValueError(f"Missing deduped W&B Table-9 rows for one-phase runs: {missing[:20]}")
    return merged


def load_deletion_panel(components: list[str], columns: list[str]) -> pd.DataFrame:
    deletion = base.load_deletion_weights(columns)
    if len(deletion) != N_EXPECTED_DELETION_ROWS:
        raise ValueError(f"Expected {N_EXPECTED_DELETION_ROWS} deletion rows, found {len(deletion)}")
    validate_phase_tied(deletion, columns, label="domain-deletion controls")
    deletion = deletion.copy()
    deletion["panel_source"] = "domain_deletion"

    olmo = paper_olmix.load_olmo_wide_with_table9_components()
    metrics = olmo[["run_name", *components]].copy()
    out = deletion.merge(metrics, on="run_name", how="left", validate="one_to_one")
    missing = out.loc[out[components].isna().any(axis=1), "run_name"].tolist()
    if missing:
        raise ValueError(f"Missing Table-9 values for deletion rows: {missing[:20]}")
    out["table9_macro_bpb"] = out[components].mean(axis=1)
    return out


def load_shared_stratified_panel(components: list[str], columns: list[str]) -> pd.DataFrame:
    """Represent the phase-tied stratified checkpoint in the single-phase panel."""
    two_phase_panel, _metadata = paper_olmix.build_fit_panel(columns)
    shared = two_phase_panel.loc[two_phase_panel["run_name"].eq("baseline_stratified")].copy()
    if len(shared) != N_EXPECTED_SHARED_STRATIFIED_ROWS:
        raise ValueError(f"Expected {N_EXPECTED_SHARED_STRATIFIED_ROWS} shared stratified row, found {len(shared)}")
    validate_phase_tied(shared, columns, label="shared stratified baseline")
    missing_components = sorted(set(components).difference(shared.columns))
    if missing_components:
        raise ValueError(f"Shared stratified baseline is missing Table-9 components: {missing_components}")

    shared["source_run_name"] = shared["run_name"]
    shared["source_panel"] = "shared_policy_intersection"
    shared["panel_source"] = "shared_stratified_baseline"
    shared["is_shared_checkpoint_alias"] = True
    shared["shared_checkpoint_run_name"] = "baseline_stratified"
    shared["run_name"] = "singleavg_baseline_stratified"
    return shared


def proportional_reference(components: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    olmo = paper_olmix.load_olmo_wide_with_table9_components()
    reference = olmo.loc[
        olmo["run_name"].eq("baseline_proportional") | olmo["panel"].eq("proportional_noise"),
        ["run_name", "panel", *components],
    ].copy()
    if len(reference) != N_EXPECTED_PROPORTIONAL_REFERENCES:
        raise ValueError(
            f"Expected {N_EXPECTED_PROPORTIONAL_REFERENCES} proportional references, found {len(reference)}"
        )
    means = reference[components].mean(axis=0)
    stds = reference[components].std(axis=0, ddof=1)
    reference["table9_macro_bpb"] = reference[components].mean(axis=1)
    return reference, means, stds


def build_fit_panel(
    *,
    signal: pd.DataFrame,
    shared_stratified: pd.DataFrame,
    deletion: pd.DataFrame,
    components: list[str],
    columns: list[str],
    reference_means: pd.Series,
) -> pd.DataFrame:
    keep = [
        "run_name",
        "source_experiment",
        "panel_source",
        "source_run_name",
        "source_panel",
        "is_shared_checkpoint_alias",
        "shared_checkpoint_run_name",
        *columns,
        *components,
        "table9_macro_bpb",
    ]
    signal_for_panel = signal.copy()
    signal_for_panel["is_shared_checkpoint_alias"] = False
    signal_for_panel["shared_checkpoint_run_name"] = pd.NA
    deletion_for_panel = deletion.copy()
    deletion_for_panel["is_shared_checkpoint_alias"] = False
    deletion_for_panel["shared_checkpoint_run_name"] = pd.NA
    deletion_for_panel["source_run_name"] = deletion_for_panel.get("target_domain", "")
    deletion_for_panel["source_panel"] = "proportional_domain_deletion"
    panel = pd.concat(
        [
            signal_for_panel[keep],
            shared_stratified[keep],
            deletion_for_panel[keep],
        ],
        ignore_index=True,
    )
    prop_mask = panel["run_name"].eq("singleavg_baseline_proportional")
    if int(prop_mask.sum()) != 1:
        raise ValueError("Expected exactly one singleavg_baseline_proportional row")
    panel.loc[prop_mask, components] = reference_means.to_numpy()
    panel.loc[prop_mask, "table9_macro_bpb"] = float(reference_means.mean())

    expected_rows = N_EXPECTED_SINGLE_PHASE_ROWS + N_EXPECTED_SHARED_STRATIFIED_ROWS + N_EXPECTED_DELETION_ROWS
    if len(panel) != expected_rows:
        raise ValueError(f"Expected {expected_rows} fit rows, found {len(panel)}")
    if int(panel["panel_source"].eq("single_phase_qsplit_signal").sum()) != N_EXPECTED_SINGLE_PHASE_ROWS:
        raise ValueError("Unexpected one-phase qsplit row count")
    if int(panel["panel_source"].eq("shared_stratified_baseline").sum()) != N_EXPECTED_SHARED_STRATIFIED_ROWS:
        raise ValueError("Unexpected shared stratified row count")
    if int(panel["panel_source"].eq("domain_deletion").sum()) != N_EXPECTED_DELETION_ROWS:
        raise ValueError("Unexpected domain-deletion row count")
    if int(panel[components].isna().sum().sum()) != 0:
        raise ValueError("Fit panel contains missing component values")
    validate_phase_tied(panel, columns, label="one-phase parity fit panel")
    return panel


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    components = paper_olmix.table9_component_order()
    if len(components) != N_EXPECTED_COMPONENTS:
        raise ValueError(f"Expected {N_EXPECTED_COMPONENTS} Table-9 components, found {len(components)}")
    manifest_columns = base.phase_columns(pd.read_csv(SINGLE_PHASE_MANIFEST, nrows=1))
    domains = base.domain_names_from_phase_columns(manifest_columns)

    single_eval, duplicate_log, wandb_metadata = collect_single_phase_wandb(
        project=args.wandb_project,
        group=args.wandb_group,
        components=components,
    )
    signal = load_single_phase_panel(components, manifest_columns, single_eval)
    shared_stratified = load_shared_stratified_panel(components, manifest_columns)
    deletion = load_deletion_panel(components, manifest_columns)
    reference, reference_means, reference_stds = proportional_reference(components)
    panel = build_fit_panel(
        signal=signal,
        shared_stratified=shared_stratified,
        deletion=deletion,
        components=components,
        columns=manifest_columns,
        reference_means=reference_means,
    )

    single_eval.to_csv(args.output_dir / "single_phase_table9_wide.csv", index=False)
    duplicate_log.to_csv(args.output_dir / "single_phase_wandb_duplicate_log.csv", index=False)
    reference.to_csv(args.output_dir / "proportional_reference_table9.csv", index=False)
    panel.to_csv(args.output_dir / "one_phase_augmented_fit_panel.csv", index=False)
    component_metadata = {
        "components": components,
        "native_component_keys": {component: native_component_key(component) for component in components},
        "domains": domains,
        "phase_columns": manifest_columns,
        "proportional_reference_component_means": {
            component: float(reference_means[component]) for component in components
        },
        "proportional_reference_component_stds": {
            component: float(reference_stds[component]) for component in components
        },
    }
    write_json(args.output_dir / "component_metadata.json", component_metadata)

    max_single_phase_delta = validate_phase_tied(signal, manifest_columns, label="single-phase output")
    max_deletion_delta = validate_phase_tied(deletion, manifest_columns, label="deletion output")
    summary = {
        **wandb_metadata,
        "output_dir": str(args.output_dir),
        "single_phase_manifest": str(SINGLE_PHASE_MANIFEST),
        "single_phase_target_manifest": str(SINGLE_PHASE_TARGET_MANIFEST),
        "source_olmo_full_wide": str(paper_olmix.OLMO_FULL_WIDE),
        "proportional_controllability_manifest": str(base.PCTRL_MANIFEST),
        "component_count": len(components),
        "domain_count": len(domains),
        "fit_row_count": len(panel),
        "single_phase_qsplit_rows": int(panel["panel_source"].eq("single_phase_qsplit_signal").sum()),
        "shared_stratified_rows": int(panel["panel_source"].eq("shared_stratified_baseline").sum()),
        "domain_deletion_rows": int(panel["panel_source"].eq("domain_deletion").sum()),
        "proportional_reference_observation_count": len(reference),
        "proportional_reference_macro_mean": float(reference_means.mean()),
        "proportional_reference_macro_std": float(reference["table9_macro_bpb"].std(ddof=1)),
        "single_phase_proportional_raw_macro": float(
            signal.loc[signal["run_name"].eq("singleavg_baseline_proportional"), "table9_macro_bpb"].iloc[0]
        ),
        "single_phase_proportional_fit_macro": float(
            panel.loc[panel["run_name"].eq("singleavg_baseline_proportional"), "table9_macro_bpb"].iloc[0]
        ),
        "best_observed_panel_run_name": str(panel.loc[panel["table9_macro_bpb"].idxmin(), "run_name"]),
        "best_observed_panel_macro_bpb": float(panel["table9_macro_bpb"].min()),
        "max_single_phase_phase_delta": max_single_phase_delta,
        "max_deletion_phase_delta": max_deletion_delta,
        "fit_rows_note": (
            "The 11 proportional references are not extra model rows; they replace the one proportional "
            "target row. The phase-tied stratified checkpoint is shared by the one- and two-phase policy "
            "classes, so the fit panel has 240 + 1 + 39 = 280 rows without redundant retraining."
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
