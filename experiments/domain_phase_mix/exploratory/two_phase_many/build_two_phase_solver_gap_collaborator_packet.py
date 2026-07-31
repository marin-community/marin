# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "tabulate",
# ]
# ///

"""Build a self-contained collaborator packet for the two-phase solver gap.

The packet is intentionally data-first:

* train panel: the 290-row 300M augmented two-phase reference panel, including
  proportional-repeatability rows used to form the proportional reference mean
* heldout panel: proportional-control tilts, the full single-phase augmented
  panel, and proportional perturbation diagnostics
* metrics: OLMoBaseEval Table-9 BPB components and uncheatable eval metrics
* code: standalone effective-exposure DSP implementation
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from marin.evaluation.olmo_base_eval.components import MMLU_CATEGORY_WEIGHTS

ROOT = Path(__file__).resolve().parents[4]
TWO_PHASE_DIR = ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many"
REFERENCE_OUTPUTS = TWO_PHASE_DIR / "reference_outputs"
OUT_DIR = REFERENCE_OUTPUTS / "two_phase_solver_gap_collaborator_packet_20260701"

TRAIN_TABLE9_PATH = REFERENCE_OUTPUTS / "olmo_base_easy_paper_faithful_olmix_300m_20260625/fit_panel_table9_macro.csv"
FULL_TABLE9_PATH = (
    REFERENCE_OUTPUTS / "olmo_base_easy_full_results_60m_300m_20260625/olmo_base_easy_full_results_60m_300m_wide.csv"
)
EXTRA_HELDOUT_PATH = (
    REFERENCE_OUTPUTS / "olmo_base_easy_extra_300m_heldout_eval_20260630/extra_300m_table9_heldout_panel.csv"
)
PCTRL_MANIFEST_PATH = REFERENCE_OUTPUTS / "proportional_controllability_300m_20260520/training_manifest.csv"
PCTRL_UNCHEATABLE_PATH = (
    REFERENCE_OUTPUTS / "pctrl_training_eval_wandb_collect_20260623/pctrl_final_metric_matrix_with_training_eval.csv"
)
QSP_UNCHEATABLE_PATH = (
    REFERENCE_OUTPUTS
    / "raw_metric_matrix_300m_training_eval_wandb_collect_20260623/pctrl_final_metric_matrix_with_training_eval.csv"
)
PPERT_UNCHEATABLE_PATH = (
    REFERENCE_OUTPUTS / "proportional_bump_confidence_reliability_20260616/ppert_300m_baseline_domain_bump_matrix.csv"
)
ONE_PHASE_SCORES_PATH = (
    REFERENCE_OUTPUTS
    / "one_phase_swarm_scores_export_300m_20260630/one_phase_augmented_fit_panel_uncheatable_table9_scores_300m.csv"
)
ONE_VS_TWO_SUMMARY_PATH = (
    REFERENCE_OUTPUTS / "one_vs_two_phase_best_mixture_comparison_20260701/one_vs_two_phase_best_mixture_summary.csv"
)
ONE_VS_TWO_DELTAS_PATH = (
    REFERENCE_OUTPUTS / "one_vs_two_phase_best_mixture_comparison_20260701/one_vs_two_phase_best_mixture_deltas.csv"
)
ONE_VS_TWO_HTML_PATH = (
    REFERENCE_OUTPUTS / "one_vs_two_phase_best_mixture_comparison_20260701/one_vs_two_phase_best_mixtures.html"
)
DSP_SOURCE_PATH = TWO_PHASE_DIR / "standalone_code/dsp_exact.py"
EPOCH_METADATA_PATH = TWO_PHASE_DIR / "two_phase_many_epoch_metadata.csv"
COMPONENT_METADATA_PATH = REFERENCE_OUTPUTS / "olmo_base_easy_paper_faithful_olmix_300m_20260625/component_metadata.json"
DELPHI_SCALING_COMPLETED_PATH = REFERENCE_OUTPUTS / "delphi_scaling_progress_20260625/delphi_scaling_completed_wandb.csv"

TABLE9_MACRO_COL = "table9_macro_bpb"
UNCH_BPB_COL = "eval/uncheatable_eval/bpb"
UNCH_MACRO_COL = "eval/uncheatable_eval/macro_bpb"


@dataclass(frozen=True)
class PacketPaths:
    root: Path
    data: Path
    code: Path
    figures: Path
    reviews: Path
    scripts: Path


def packet_paths() -> PacketPaths:
    return PacketPaths(
        root=OUT_DIR,
        data=OUT_DIR / "data",
        code=OUT_DIR / "code",
        figures=OUT_DIR / "figures",
        reviews=OUT_DIR / "reviews",
        scripts=OUT_DIR / "scripts",
    )


def reset_output(paths: PacketPaths) -> None:
    if paths.root.exists():
        shutil.rmtree(paths.root)
    for path in (paths.data, paths.code, paths.figures, paths.reviews, paths.scripts):
        path.mkdir(parents=True, exist_ok=True)


def table9_component_cols(frame: pd.DataFrame) -> list[str]:
    metadata = json.loads(COMPONENT_METADATA_PATH.read_text())
    cols = list(metadata["components"])
    if len(cols) != 51:
        raise ValueError(f"Expected 51 Table-9 BPB columns, found {len(cols)}")
    missing = sorted(set(cols).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing Table-9 component columns in fit panel: {missing}")
    return cols


def mmlu_metric_key(task: str) -> str:
    return f"olmo_base_eval/easy_bpb/{task}_rc/bpb"


def ensure_table9_columns(frame: pd.DataFrame, table9_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for category, weights in MMLU_CATEGORY_WEIGHTS.items():
        if category in out.columns:
            continue
        source_cols = [mmlu_metric_key(task) for task in weights]
        missing = sorted(set(source_cols).difference(out.columns))
        if missing:
            raise ValueError(f"Cannot reconstruct {category}; missing source columns: {missing}")
        weight_vector = np.asarray([weights[task] for task in weights], dtype=float)
        out[category] = out[source_cols].astype(float).to_numpy() @ weight_vector
    missing = sorted(set(table9_cols).difference(out.columns))
    if missing:
        raise ValueError(f"Missing Table-9 component columns after reconstruction: {missing}")
    return out


def phase_cols(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame.columns if c.startswith("phase_0_") or c.startswith("phase_1_")]
    if len(cols) != 78:
        raise ValueError(f"Expected 78 phase-weight columns, found {len(cols)}")
    return cols


def uncheatable_cols(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("eval/uncheatable_eval/")]


def slugify_column(name: str) -> str:
    return name.replace("eval/", "eval_").replace("/", "_")


def normalize_uncheatable(frame: pd.DataFrame, source_label: str) -> pd.DataFrame:
    cols = ["run_name", *uncheatable_cols(frame)]
    out = frame[cols].copy()
    rename = {c: slugify_column(c) for c in cols if c != "run_name"}
    out = out.rename(columns=rename)
    out["uncheatable_metric_source"] = source_label
    return out


def first_nonnull_by_run(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in frame.groupby("run_name", sort=False):
        rows.append(
            pd.Series(
                {
                    col: values.iloc[0] if not values.empty else np.nan
                    for col in group.columns
                    for values in [group[col].dropna()]
                }
            )
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def grouped_counts_dict(frame: pd.DataFrame, cols: list[str]) -> dict[str, int]:
    counts = frame.groupby(cols, dropna=False).size()
    result: dict[str, int] = {}
    for key, value in counts.items():
        if len(cols) == 1:
            label = str(key)
        else:
            label = " | ".join(str(part) for part in key)
        result[label] = int(value)
    return result


def attach_uncheatable(frame: pd.DataFrame) -> pd.DataFrame:
    sources = []
    qsp = normalize_uncheatable(pd.read_csv(QSP_UNCHEATABLE_PATH), "qsplit_or_proportional_noise_wandb")
    pctrl = normalize_uncheatable(pd.read_csv(PCTRL_UNCHEATABLE_PATH), "proportional_controllability_wandb")
    ppert = normalize_uncheatable(pd.read_csv(PPERT_UNCHEATABLE_PATH), "proportional_domain_bump_wandb")
    one_phase = pd.read_csv(ONE_PHASE_SCORES_PATH)
    one_cols = [
        "run_name",
        "panel_source",
        "eval_uncheatable_eval_bpb",
        "eval_uncheatable_eval_macro_bpb",
        "eval_bpb",
        "eval_macro_bpb",
        "uncheatable_source",
    ]
    one = one_phase[[c for c in one_cols if c in one_phase.columns]].copy()
    if "panel_source" in one.columns:
        single_domain = one["panel_source"].eq("domain_deletion")
        needs_packet_prefix = single_domain & ~one["run_name"].astype(str).str.startswith("singleavg_")
        one.loc[needs_packet_prefix, "run_name"] = "singleavg_" + one.loc[needs_packet_prefix, "run_name"].astype(str)
        one = one.drop(columns=["panel_source"])
    one["uncheatable_metric_source"] = "single_phase_scores_export"
    sources.extend([qsp, pctrl, ppert, one])

    all_uncheatable = pd.concat(sources, ignore_index=True, sort=False)
    all_uncheatable = first_nonnull_by_run(all_uncheatable)

    left = frame.copy()
    left["_uncheatable_join_run_name"] = left["run_name"]
    out = left.merge(
        all_uncheatable,
        left_on="_uncheatable_join_run_name",
        right_on="run_name",
        how="left",
        suffixes=("", "_uncheatable_source"),
    )
    out = out.drop(columns=["_uncheatable_join_run_name", "run_name_uncheatable_source"], errors="ignore")
    out["has_uncheatable_bpb"] = out["eval_uncheatable_eval_bpb"].notna()
    return out


def add_standard_metadata(
    frame: pd.DataFrame,
    *,
    split: str,
    policy_family: str,
    panel: str,
    method: str,
    source: str,
) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "split", split)
    out.insert(1, "policy_family", policy_family)
    out.insert(2, "packet_panel", panel)
    out.insert(3, "packet_method", method)
    out.insert(4, "packet_source", source)
    return out


def add_phase_family_labels(frame: pd.DataFrame, phases: list[str]) -> pd.DataFrame:
    """Add explicit training-family and phase-weight structure labels."""
    out = frame.copy()
    if "policy_family" in out.columns:
        out["training_phase_family"] = out["policy_family"].map(
            {"single_phase": "single_phase", "two_phase": "two_phase"}
        )
    else:
        out["training_phase_family"] = np.nan
    if "diagnostic_group" in out.columns:
        single_mask = out["diagnostic_group"].astype(str).str.startswith("single_phase_")
        out.loc[single_mask, "training_phase_family"] = "single_phase"
        out.loc[out["training_phase_family"].isna(), "training_phase_family"] = "two_phase"

    phase0_cols = sorted(c for c in phases if c.startswith("phase_0_"))
    phase1_cols = sorted(c for c in phases if c.startswith("phase_1_"))
    phase0_domains = [c.removeprefix("phase_0_") for c in phase0_cols]
    phase1_domains = [c.removeprefix("phase_1_") for c in phase1_cols]
    if phase0_domains != phase1_domains:
        raise ValueError("Phase 0 and phase 1 domain columns do not align")
    deltas = out[phase0_cols].to_numpy(dtype=float) - out[phase1_cols].to_numpy(dtype=float)
    out["phase_max_abs_delta"] = np.abs(deltas).max(axis=1)
    out["phase_weight_structure"] = np.where(out["phase_max_abs_delta"].le(1e-12), "tied_weights", "untied_weights")
    out["is_single_phase_checkpoint"] = out["training_phase_family"].eq("single_phase")
    return out


def add_correspondence_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Add row-level keys for matching single-phase rows to their two-phase source rows."""
    out = frame.copy()
    if "original_run_name" not in out.columns:
        out["original_run_name"] = out["run_name"]
    out["original_run_name"] = out["original_run_name"].fillna(out["run_name"])
    out["packet_row_id"] = out["training_phase_family"].astype(str) + "::" + out["run_name"].astype(str)
    out["phase_correspondence_key"] = pd.Series(pd.NA, index=out.index, dtype="object")

    single_mask = out["training_phase_family"].eq("single_phase")
    single_domain = single_mask & out["diagnostic_group"].eq("single_phase_domain_deletion")
    single_source = single_mask & ~single_domain & out.get("source_run_name", pd.Series(index=out.index)).notna()
    out.loc[single_domain, "phase_correspondence_key"] = out.loc[single_domain, "original_run_name"].astype(str)
    out.loc[single_source, "phase_correspondence_key"] = out.loc[single_source, "source_run_name"].astype(str)

    two_pairable = out["training_phase_family"].eq("two_phase") & out["diagnostic_group"].isin(
        ["two_phase_qsplit_signal", "two_phase_domain_deletion"]
    )
    out.loc[two_pairable, "phase_correspondence_key"] = out.loc[two_pairable, "run_name"].astype(str)

    repeat_mask = out["diagnostic_group"].eq("two_phase_proportional_noise")
    out.loc[repeat_mask, "phase_correspondence_key"] = out.loc[repeat_mask, "source_run_name"].astype(str)
    out["phase_pair_group"] = out["phase_correspondence_key"]
    out.loc[repeat_mask, "phase_pair_group"] = "baseline_proportional::repeatability_reference"
    return out


def add_pair_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Annotate rows with corresponding single/two-phase run names when present."""
    out = frame.copy()
    single_pairs = (
        out.loc[out["training_phase_family"].eq("single_phase") & out["phase_correspondence_key"].notna()]
        .groupby("phase_correspondence_key")["run_name"]
        .agg(lambda values: "|".join(sorted(values.astype(str).unique())))
    )
    two_pairs = (
        out.loc[
            out["training_phase_family"].eq("two_phase")
            & out["diagnostic_group"].isin(["two_phase_qsplit_signal", "two_phase_domain_deletion"])
            & out["phase_correspondence_key"].notna()
        ]
        .groupby("phase_correspondence_key")["run_name"]
        .agg(lambda values: "|".join(sorted(values.astype(str).unique())))
    )
    out["paired_single_phase_run_name"] = out["phase_correspondence_key"].map(single_pairs)
    out["paired_two_phase_run_name"] = out["phase_correspondence_key"].map(two_pairs)
    out["phase_pair_status"] = "not_pairable"
    pairable = out["phase_correspondence_key"].notna()
    paired = pairable & out["paired_single_phase_run_name"].notna() & out["paired_two_phase_run_name"].notna()
    out.loc[paired, "phase_pair_status"] = "paired_single_two"
    out.loc[
        pairable & out["training_phase_family"].eq("two_phase") & out["paired_single_phase_run_name"].isna(),
        "phase_pair_status",
    ] = "no_single_phase_counterpart"
    out.loc[
        pairable & out["training_phase_family"].eq("single_phase") & out["paired_two_phase_run_name"].isna(),
        "phase_pair_status",
    ] = "no_two_phase_counterpart"
    out.loc[out["diagnostic_group"].eq("two_phase_proportional_noise"), "phase_pair_status"] = (
        "repeat_reference_to_baseline"
    )
    return out


def phase_correspondence_table(all_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize row-by-row single/two-phase correspondences."""
    pairable = all_rows[all_rows["phase_correspondence_key"].notna()].copy()
    if pairable.empty:
        raise ValueError("No phase correspondence keys found")

    records: list[dict[str, Any]] = []
    for key, group in pairable.groupby("phase_correspondence_key", sort=True):
        single_rows = group[group["training_phase_family"].eq("single_phase")]
        two_rows = group[
            group["training_phase_family"].eq("two_phase")
            & group["diagnostic_group"].isin(["two_phase_qsplit_signal", "two_phase_domain_deletion"])
        ]
        repeat_rows = group[group["diagnostic_group"].eq("two_phase_proportional_noise")]
        records.append(
            {
                "phase_correspondence_key": key,
                "single_phase_run_names": "|".join(sorted(single_rows["run_name"].astype(str).unique())),
                "two_phase_run_names": "|".join(sorted(two_rows["run_name"].astype(str).unique())),
                "proportional_repeat_run_names": "|".join(sorted(repeat_rows["run_name"].astype(str).unique())),
                "single_phase_row_count": len(single_rows),
                "two_phase_row_count": len(two_rows),
                "proportional_repeat_row_count": len(repeat_rows),
                "has_single_phase": bool(len(single_rows)),
                "has_two_phase": bool(len(two_rows)),
                "has_proportional_repeat_reference": bool(len(repeat_rows)),
                "source_panels": "|".join(sorted(group["source_panel"].dropna().astype(str).unique())),
                "diagnostic_groups": "|".join(sorted(group["diagnostic_group"].dropna().astype(str).unique())),
            }
        )
    out = pd.DataFrame.from_records(records)
    out["pair_status"] = np.select(
        [
            out["has_single_phase"] & out["has_two_phase"],
            out["has_single_phase"] & ~out["has_two_phase"],
            ~out["has_single_phase"] & out["has_two_phase"],
        ],
        ["paired_single_two", "single_without_two_phase", "two_phase_without_single"],
        default="reference_only",
    )
    return out


def build_train_panel(table9_cols: list[str], phases: list[str]) -> pd.DataFrame:
    train = pd.read_csv(TRAIN_TABLE9_PATH)
    keep = [
        "run_name",
        "source_experiment",
        "panel_source",
        *phases,
        *table9_cols,
        TABLE9_MACRO_COL,
    ]
    train = train[keep].copy()
    train["source_run_name"] = train["run_name"]
    train["source_panel"] = train["panel_source"]
    train["diagnostic_group"] = (
        train["panel_source"]
        .map(
            {
                "qsplit_signal": "two_phase_qsplit_signal",
                "domain_deletion": "two_phase_domain_deletion",
            }
        )
        .fillna(train["panel_source"])
    )
    train = add_standard_metadata(
        train,
        split="train",
        policy_family="two_phase",
        panel="augmented_fit_panel",
        method="qsplit_plus_domain_deletions",
        source=str(TRAIN_TABLE9_PATH),
    )
    return train


def build_pctrl_tilt_heldout(table9_cols: list[str], phases: list[str], train_names: set[str]) -> pd.DataFrame:
    full = pd.read_csv(FULL_TABLE9_PATH)
    full = ensure_table9_columns(full, table9_cols)
    pctrl_table9 = full[(full["scale"] == "300m_6b") & (full["panel"] == "proportional_controllability")].copy()
    manifest = pd.read_csv(PCTRL_MANIFEST_PATH)
    merged = manifest.merge(
        pctrl_table9[["run_name", "output_name", "wandb_run_id", *table9_cols]],
        on="run_name",
        how="inner",
    )
    merged[TABLE9_MACRO_COL] = merged[table9_cols].mean(axis=1)
    heldout = merged[~merged["run_name"].isin(train_names)].copy()
    if len(heldout) != 78:
        raise ValueError(f"Expected 78 pctrl log-tilt heldout rows, found {len(heldout)}")
    keep = [
        "run_name",
        "source_experiment",
        "intervention_type",
        "target_domain",
        "direction_id",
        "direction_type",
        "tilt_sign",
        "alpha",
        "tv_distance",
        *phases,
        *table9_cols,
        TABLE9_MACRO_COL,
    ]
    heldout = heldout[[c for c in keep if c in heldout.columns]].copy()
    heldout["source_run_name"] = heldout["run_name"]
    heldout["source_panel"] = "proportional_controllability"
    heldout["diagnostic_group"] = "two_phase_log_tilt"
    heldout = add_standard_metadata(
        heldout,
        split="heldout",
        policy_family="two_phase",
        panel="proportional_controllability_tilts",
        method="paired_log_tilt",
        source=str(FULL_TABLE9_PATH),
    )
    return heldout


def build_proportional_noise_train(
    table9_cols: list[str], phases: list[str], proportional_row: pd.Series
) -> pd.DataFrame:
    full = pd.read_csv(FULL_TABLE9_PATH)
    full = ensure_table9_columns(full, table9_cols)
    noise = full[(full["scale"] == "300m_6b") & (full["panel"] == "proportional_noise")].copy()
    for col in phases:
        noise[col] = float(proportional_row[col])
    noise[TABLE9_MACRO_COL] = noise[table9_cols].mean(axis=1)
    noise["source_experiment"] = "pinlin_calvin_xu/data_mixture/ngd3dm2_proportional_noise_300m_6b"
    noise["source_run_name"] = "baseline_proportional"
    noise["source_panel"] = "proportional_noise"
    noise["diagnostic_group"] = "two_phase_proportional_noise"
    keep = [
        "run_name",
        "source_experiment",
        "source_run_name",
        "source_panel",
        "diagnostic_group",
        *phases,
        *table9_cols,
        TABLE9_MACRO_COL,
    ]
    noise = add_standard_metadata(
        noise[keep],
        split="train",
        policy_family="two_phase",
        panel="proportional_noise_reference",
        method="proportional_reference_repeats",
        source=str(FULL_TABLE9_PATH),
    )
    return noise


def build_one_phase_heldout(table9_cols: list[str], phases: list[str]) -> pd.DataFrame:
    one_phase = pd.read_csv(ONE_PHASE_SCORES_PATH)
    one_phase = ensure_table9_columns(one_phase, table9_cols)
    for phase_column in phases:
        domain = phase_column.removeprefix("phase_0_").removeprefix("phase_1_")
        weight_column = f"weight_{domain}"
        if weight_column not in one_phase.columns:
            raise ValueError(f"Missing one-phase mixture column {weight_column}")
        one_phase[phase_column] = one_phase[weight_column]
    keep = [
        "run_name",
        "source_experiment",
        "source_run_name",
        "source_panel",
        "panel_source",
        "training_wandb_name",
        "training_wandb_id",
        "training_wandb_state",
        "training_wandb_created_at",
        "training_wandb_url",
        "is_shared_checkpoint_alias",
        "shared_checkpoint_run_name",
        "target_uses_proportional_reference_mean",
        "proportional_reference_n",
        *phases,
        *table9_cols,
        TABLE9_MACRO_COL,
    ]
    one_phase = one_phase[[c for c in keep if c in one_phase.columns]].copy()
    one_phase["original_run_name"] = one_phase["run_name"]
    one_phase["diagnostic_group"] = one_phase["panel_source"].map(
        {
            "single_phase_qsplit_signal": "single_phase_300m_qsplit",
            "shared_stratified_baseline": "single_phase_shared_stratified",
            "domain_deletion": "single_phase_domain_deletion",
        }
    )
    if one_phase["diagnostic_group"].isna().any():
        bad = sorted(one_phase.loc[one_phase["diagnostic_group"].isna(), "panel_source"].dropna().unique())
        raise ValueError(f"Unknown one-phase panel_source values: {bad}")
    needs_packet_prefix = one_phase["diagnostic_group"].eq("single_phase_domain_deletion") & ~one_phase[
        "run_name"
    ].astype(str).str.startswith("singleavg_")
    one_phase.loc[needs_packet_prefix, "run_name"] = "singleavg_" + one_phase.loc[
        needs_packet_prefix, "run_name"
    ].astype(str)
    one_phase = add_standard_metadata(
        one_phase,
        split="heldout",
        policy_family="single_phase",
        panel="single_phase_augmented_panel",
        method="single_phase_exposure_average",
        source=str(ONE_PHASE_SCORES_PATH),
    )
    return one_phase


def collapsed_fit_matrix(train: pd.DataFrame, table9_cols: list[str]) -> pd.DataFrame:
    """Collapse proportional repeats into the baseline row for the default fit matrix."""
    baseline_mask = train["run_name"].eq("baseline_proportional")
    repeat_mask = train["diagnostic_group"].eq("two_phase_proportional_noise")
    if int(baseline_mask.sum()) != 1:
        raise ValueError(f"Expected one baseline_proportional row, found {int(baseline_mask.sum())}")
    if int(repeat_mask.sum()) != 10:
        raise ValueError(f"Expected 10 proportional-repeat rows, found {int(repeat_mask.sum())}")

    out = train.loc[~repeat_mask].copy().reset_index(drop=True)
    reference_rows = train.loc[baseline_mask | repeat_mask].copy()
    metric_columns = [
        col
        for col in [*table9_cols, TABLE9_MACRO_COL, "eval_uncheatable_eval_bpb", "eval_uncheatable_eval_macro_bpb"]
        if col in out.columns and col in reference_rows.columns
    ]
    baseline_idx = out.index[out["run_name"].eq("baseline_proportional")]
    if len(baseline_idx) != 1:
        raise ValueError("Collapsed fit matrix lost the baseline_proportional row")
    idx = int(baseline_idx[0])
    for col in metric_columns:
        values = pd.to_numeric(reference_rows[col], errors="coerce")
        if values.notna().any():
            out.loc[idx, col] = float(values.mean())
    out.loc[idx, "source_panel"] = "qsplit_signal_with_proportional_reference_mean"
    out.loc[idx, "diagnostic_group"] = "two_phase_qsplit_signal"
    out.loc[idx, "packet_method"] = "qsplit_plus_domain_deletions_proportional_reference_mean"
    out.loc[idx, "proportional_reference_n"] = len(reference_rows)
    out.loc[idx, "proportional_reference_repeat_n"] = int(repeat_mask.sum())
    out["fit_matrix_role"] = np.where(
        out["run_name"].eq("baseline_proportional"),
        "collapsed_proportional_reference_mean",
        out["diagnostic_group"],
    )
    return out


def build_extra_heldout(table9_cols: list[str], phases: list[str]) -> pd.DataFrame:
    extra = pd.read_csv(EXTRA_HELDOUT_PATH)
    extra = extra[~extra["panel"].eq("single_phase_qsplit")].copy()
    keep = [
        "run_name",
        "panel",
        "method",
        "source_experiment",
        "source_run_name",
        "source_panel",
        "diagnostic_group",
        "diagnostic_family",
        *phases,
        *table9_cols,
        TABLE9_MACRO_COL,
    ]
    extra = extra[[c for c in keep if c in extra.columns]].copy()
    extra = add_standard_metadata(
        extra,
        split="heldout",
        policy_family="two_phase",
        panel="extra_300m_diagnostics",
        method="proportional_perturbations",
        source=str(EXTRA_HELDOUT_PATH),
    )
    return extra


def table9_macro_values(frame: pd.DataFrame, table9_cols: list[str]) -> pd.Series:
    return frame[table9_cols].astype(float).mean(axis=1)


def assert_packet_invariants(
    *,
    train: pd.DataFrame,
    heldout: pd.DataFrame,
    all_rows: pd.DataFrame,
    table9_cols: list[str],
    phases: list[str],
) -> None:
    train_names = set(train["run_name"])
    heldout_names = set(heldout["run_name"])
    overlap = sorted(train_names.intersection(heldout_names))
    if overlap:
        raise ValueError(f"Train/heldout run_name overlap: {overlap[:10]}")
    if all_rows["run_name"].duplicated().any():
        duplicates = all_rows.loc[all_rows["run_name"].duplicated(), "run_name"].tolist()
        raise ValueError(f"Duplicate packet run_name values: {duplicates[:10]}")
    if "packet_row_id" in all_rows.columns and all_rows["packet_row_id"].duplicated().any():
        duplicates = all_rows.loc[all_rows["packet_row_id"].duplicated(), "packet_row_id"].tolist()
        raise ValueError(f"Duplicate packet_row_id values: {duplicates[:10]}")
    single_domain_rows = all_rows[all_rows["diagnostic_group"].eq("single_phase_domain_deletion")]
    if len(single_domain_rows) != 39:
        raise ValueError(f"Expected 39 single-phase domain-deletion rows, found {len(single_domain_rows)}")
    bad_single_domain_source = single_domain_rows[
        single_domain_rows["uncheatable_metric_source"].ne("single_phase_scores_export")
    ]
    if not bad_single_domain_source.empty:
        bad = bad_single_domain_source[["run_name", "original_run_name", "uncheatable_metric_source"]].to_dict(
            orient="records"
        )
        raise ValueError(f"Single-phase domain-deletion rows have wrong uncheatable provenance: {bad[:5]}")

    macro_delta = (table9_macro_values(all_rows, table9_cols) - all_rows[TABLE9_MACRO_COL].astype(float)).abs()
    if macro_delta.max() > 1e-10:
        idx = int(macro_delta.idxmax())
        raise ValueError(f"Table-9 macro mismatch for {all_rows.loc[idx, 'run_name']}: {float(macro_delta.loc[idx])}")

    single_phase = all_rows["diagnostic_group"].eq("single_phase_300m_qsplit")
    if single_phase.any():
        phase0_cols = sorted(c for c in phases if c.startswith("phase_0_"))
        phase1_cols = sorted(c for c in phases if c.startswith("phase_1_"))
        phase0_domains = [c.removeprefix("phase_0_") for c in phase0_cols]
        phase1_domains = [c.removeprefix("phase_1_") for c in phase1_cols]
        if phase0_domains != phase1_domains:
            raise ValueError("Phase 0 and phase 1 domain columns do not align")
        max_gap = all_rows.loc[single_phase, phase0_cols].to_numpy(dtype=float) - all_rows.loc[
            single_phase, phase1_cols
        ].to_numpy(dtype=float)
        max_gap_abs = float(np.abs(max_gap).max())
        if max_gap_abs > 1e-12:
            raise ValueError(f"Single-phase heldout rows have phase mismatch: max_abs_gap={max_gap_abs}")


def load_validation_outcomes() -> pd.DataFrame:
    scaling = pd.read_csv(DELPHI_SCALING_COMPLETED_PATH)
    rows = [
        {
            "comparison": "Uncheatable BPB / OLMix",
            "single_run_base": "olmix_onephase_uncheatable_d001_kl005_cap4_3e18",
            "two_phase_run_base": "olmix_d001_kl005_cap4_3e18",
        },
        {
            "comparison": "Uncheatable BPB / DSP",
            "single_run_base": "dsp_onephase_effexp_uncheatable_kl0p1_3e18",
            "two_phase_run_base": "dsp_effexp_kl01_3e18",
        },
        {
            "comparison": "Table-9 Macro BPB / OLMix",
            "single_run_base": "olmix_onephase_table9_d001_kl005_cap4_3e18",
            "two_phase_run_base": "olmix_table9_d001_kl005_cap4_3e18",
        },
        {
            "comparison": "Table-9 Macro BPB / DSP",
            "single_run_base": "dsp_onephase_effexp_table9_kl0p1_3e18",
            "two_phase_run_base": "dsp_split_table9_l2_0p01_kl0p3_3e18",
        },
    ]
    values: list[dict[str, Any]] = []
    for row in rows:
        single = scaling.loc[scaling["run_base"].eq(row["single_run_base"])]
        two_phase = scaling.loc[scaling["run_base"].eq(row["two_phase_run_base"])]
        if single.empty or two_phase.empty:
            continue
        single_row = single.iloc[0]
        two_phase_row = two_phase.iloc[0]
        values.append(
            {
                **row,
                "single_uncheatable_bpb_3e18": single_row.get("eval_uncheatable_eval_bpb"),
                "two_phase_uncheatable_bpb_3e18": two_phase_row.get("eval_uncheatable_eval_bpb"),
                "single_table9_bpb_3e18": single_row.get("olmo_base_easy_table9_51_component_macro_bpb"),
                "two_phase_table9_bpb_3e18": two_phase_row.get("olmo_base_easy_table9_51_component_macro_bpb"),
            }
        )
    return pd.DataFrame(values)


def write_enriched_one_vs_two_summary(paths: PacketPaths) -> pd.DataFrame:
    summary = pd.read_csv(ONE_VS_TWO_SUMMARY_PATH)
    outcomes = load_validation_outcomes()
    merged = summary.merge(outcomes, on="comparison", how="left")
    for old_col, new_col in [
        ("single_table9_bpb_3e18_if_known", "single_table9_bpb_3e18"),
        ("two_phase_table9_bpb_3e18_if_known", "two_phase_table9_bpb_3e18"),
    ]:
        if old_col in merged.columns and new_col in merged.columns:
            merged[new_col] = merged[new_col].fillna(merged[old_col])
            merged = merged.drop(columns=[old_col])
    for prefix in ("single", "two_phase"):
        table9_col = f"{prefix}_table9_bpb_3e18"
        uncheatable_col = f"{prefix}_uncheatable_bpb_3e18"
        merged[f"{prefix}_primary_bpb_3e18"] = np.where(
            merged["task"].eq("Table-9 Macro BPB"),
            merged[table9_col],
            merged[uncheatable_col],
        )
    merged["primary_gap_two_minus_single_bpb"] = merged["two_phase_primary_bpb_3e18"] - merged["single_primary_bpb_3e18"]
    merged.to_csv(paths.data / "one_vs_two_phase_best_mixture_summary.csv", index=False)
    outcomes.to_csv(paths.data / "one_vs_two_phase_validation_outcomes_3e18.csv", index=False)
    return merged


def write_readme(paths: PacketPaths, manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    validation_rows = pd.read_csv(paths.data / "one_vs_two_phase_best_mixture_summary.csv")
    validation_summary = validation_rows[
        [
            "comparison",
            "single_primary_bpb_3e18",
            "two_phase_primary_bpb_3e18",
            "primary_gap_two_minus_single_bpb",
        ]
    ].to_markdown(index=False, floatfmt=".6f")
    readme = f"""# 300M Two-Phase Solver Gap Collaborator Packet

This packet is for collaborators who want to inspect our 300M data-mixing modeling, fitting, and optimization problem without depending on the Marin repository. The central issue is a two-phase solver gap: we have a strong prior that phasing/curriculum should help, and the two-phase policy class contains the single-phase class, but the best validated single-phase mixtures we have found for both uncheatable BPB and OLMoBaseEval Table-9 BPB are as good as or better than the corresponding best two-phase mixtures found so far. This should be interpreted as evidence that our current two-phase surrogate/search procedure is not harvesting useful phase asymmetry, not as evidence that two-phase policies are intrinsically worse.

## Data

The primary tables are in `data/`.

- `train_augmented_fit_panel_300m.csv`: the raw 300M training/reference panel used for current two-phase fits. It has {summary["train_rows"]} rows: 241 qsplit/proportional rows, 39 domain deletions, and 10 proportional-repeatability rows.
- `fit_matrix_collapsed_proportional_300m.csv`: the {summary["fit_matrix_rows"]}-row matrix used by the default DSP command. It removes the 10 proportional-repeatability rows as independent observations and replaces `baseline_proportional` targets with the mean over the original proportional checkpoint plus those 10 repeats.
- `heldout_300m_checkpoint_metrics.csv`: heldout checkpoints not used by the default two-phase fit. It has {summary["heldout_rows"]} rows, including proportional-controllability log tilts, the full single-phase augmented panel, proportional perturbations, and one proportional-gradient point.
- `all_300m_checkpoint_metrics.csv`: union of train and heldout rows, {summary["all_rows"]} rows.
- `single_two_phase_correspondence_300m.csv`: row-by-row correspondence table keyed by `phase_correspondence_key`, usually the original two-phase `run_name`.
- `metric_registry/metrics_wide.csv` and `grp_no_l2/two_phase_many_epoch_metadata.csv`: minimal layout expected by the standalone DSP script.
- `table9_component_metadata.json`: canonical list of the 51 Table-9 BPB components used for the macro.

Every row has the 51 OLMoBaseEval Easy Table-9 BPB component columns and `table9_macro_bpb`, defined as the unweighted mean over those 51 BPB components. Uncheatable eval columns are included where locally collected. Coverage: `table9_macro_bpb` is present for {summary["table9_complete_rows"]}/{summary["all_rows"]} rows; `eval_uncheatable_eval_bpb` is present for {summary["uncheatable_complete_rows"]}/{summary["all_rows"]} rows. The 17 known uncheatable gaps are heldout proportional-perturbation diagnostics: 13 quality swaps, 3 family bumps, and 1 proportional-gradient point. No train rows or best-mixture validation rows are missing uncheatable BPB.

The `split` column marks `train` versus `heldout`. The raw train split includes proportional repeatability because those rows estimate the proportional reference mean and noise floor. The default two-phase model-development protocol fits `fit_matrix_collapsed_proportional_300m.csv`, not the 10 proportional repeats as duplicate independent mixture observations, and uses heldout rows for diagnostic retrodiction and solver-gap analysis.

Every row has explicit phase labels:

- `training_phase_family`: `single_phase` or `two_phase`, meaning which training policy family produced the checkpoint.
- `phase_weight_structure`: `tied_weights` or `untied_weights`, computed from whether phase-0 and phase-1 weights are numerically identical. Some two-phase diagnostics intentionally have tied weights; they are still labeled `training_phase_family=two_phase`.
- `phase_correspondence_key`: original row key used to match single-phase rows to their two-phase source rows.
- `paired_single_phase_run_name`, `paired_two_phase_run_name`, and `phase_pair_status`: row-level correspondence annotations.

The full 280-row single-phase augmented panel is included in heldout: 240 single-phase qsplit rows, the shared phase-tied stratified baseline, and 39 single-phase domain-deletion rows. The qsplit and domain-deletion rows are heldout because they are different checkpoints trained from single-phase exposure-average mixtures, even when their `source_run_name` points to a corresponding two-phase row. The stratified baseline is instead the exact shared checkpoint at the intersection of the one- and two-phase policy classes: its weights are already tied, so retraining it would only duplicate the same policy and seed. It is marked `is_shared_checkpoint_alias=true`; exclude it from diagnostics that require independent train/heldout observations. The row correspondence is materialized in `single_two_phase_correspondence_300m.csv`.

Single-phase domain-deletion source rows reuse the original two-phase-style names `pctrl_del_*`. To keep packet `run_name` values unique, these rows are named `singleavg_pctrl_del_*` in this packet, keep their original value in `original_run_name`, and match their two-phase counterpart through `phase_correspondence_key`.

The correspondence table has {summary["phase_correspondence_key_count"]} keys. Of these, {summary["paired_correspondence_key_count"]} have both a single-phase and a two-phase row; all 280 fit-panel policies now have an explicit counterpart. The 10 proportional repeatability rows point back to `baseline_proportional` but are marked `repeat_reference_to_baseline` rather than treated as independent paired mixtures.

## Headline Observation

See `data/one_vs_two_phase_best_mixture_summary.csv`, `data/one_vs_two_phase_validation_outcomes_3e18.csv`, `data/one_vs_two_phase_best_mixture_deltas.csv`, and `figures/one_vs_two_phase_best_mixtures.html`. The 3e18 validation-scale outcomes are:

{validation_summary}

Lower BPB is better, so positive gaps mean the best found two-phase candidate is worse than the best found single-phase candidate. The Table-9 DSP row compares the best found one-phase effective-exposure DSP candidate against the best found split-saturation two-phase DSP candidate; it is a frontier comparison across DSP variants, not a controlled same-functional-form phase ablation. Headline validation values are 3e18 scaling-validation results, not quantities reconstructed solely from the 300M fit panel.

For a repeat panel, one-phase DSP KL=0.1 averaged 1.074165 and split DSP L2=0.01 KL=0.3 averaged 1.089857 on Table-9 macro BPB, with split-minus-one difference 0.015692 BPB and Welch two-sided p = 0.00716. Because best-found comparisons are selected from unequal candidate pools, especially the 240 single-phase qsplit heldouts, use these numbers as evidence of a search/surrogate gap rather than an intrinsic phase-count conclusion.

## Effective-Exposure DSP

The self-contained implementation is `code/effective_exposure_dsp.py`. It is copied from the standalone DSP implementation used in our local analysis and does not import Marin modules. The effective-exposure variant uses per-domain exposure

```text
z_i = c0_i * w_i^(0) + gamma * c1_i * w_i^(1)
```

inside both the saturating benefit term and the overexposure penalty. The linear coefficients are solved by nonnegative least squares for fixed nonlinear parameters; nonlinear parameters are tuned by deterministic starts and bounded optimization. Run:

```bash
cd {paths.root.name}
uv run --no-project --script code/effective_exposure_dsp.py fit --data-dir data --target table9_macro_bpb --scale 300m_6b --run-set all_signal --variant effective_exposure --output-dir outputs/table9_effexp_fit
```

For packet-local fitting, `data/metric_registry/metrics_wide.csv` is the same collapsed {summary["fit_matrix_rows"]}-row fit matrix as `data/fit_matrix_collapsed_proportional_300m.csv`, with `scale=300m_6b` and `cohort=signal`. The script defaults are set to `--target table9_macro_bpb` and `--run-set all_signal`, so `uv run --no-project --script code/effective_exposure_dsp.py fit --data-dir data --variant effective_exposure --output-dir outputs/table9_effexp_fit` is also valid.

## What We Want Help With

We want collaborators to focus on modeling/fitting/optimization. The core question is: given this data, can we produce a two-phase mixture search procedure, preferably still simple and close to effective-exposure DSP, that reliably finds a two-phase candidate expected to beat the best single-phase candidates on heldout validation? Useful evidence can be either improved cross-validated decision diagnostics on the train panel and heldouts, or a convincing proposed 3e18 validation candidate with clear reasoning about why it should beat the current single-phase frontier.

## Provenance

`MANIFEST.json` records source paths, row counts, metric coverage, and generated file names. All paths are local provenance paths from the Marin checkout used to build the packet.

The packet itself was generated by `scripts/build_two_phase_solver_gap_collaborator_packet.py`; that copy is included so the exact assembly logic can be inspected. Rebuilding requires the Marin checkout and local source artifacts referenced in `MANIFEST.json`; using the shipped CSVs and DSP implementation does not.
"""
    (paths.root / "README.md").write_text(readme)


def write_review_summary(paths: PacketPaths) -> None:
    summary = """# CC Review Summary

This packet received a Claude Code review before handoff. The full review found two blocking issues:

1. The README described one-vs-two phase conclusions without concrete uncheatable validation values.
2. The standalone DSP script defaults/documentation did not reliably run against packet-local data.

Both blockers were patched. The packet now includes explicit one-vs-two validation tables, packet-local default DSP settings, and a smoke-tested `uv run --no-project --script code/effective_exposure_dsp.py ...` workflow.

A focused follow-up review was started after the blocker patches. It confirmed the main referenced files existed and surfaced that `one_vs_two_phase_best_mixture_deltas.csv` was present but not listed in the README/manifest. That omission was patched. The follow-up review was interrupted before producing a final complete verdict, so it should be treated as partial.

After the review, the packet split was corrected so proportional repeatability rows are part of the raw train/reference split rather than heldout. This matches the actual fitting convention: the raw train/reference table has 290 rows, while the default DSP fit matrix has 280 rows because it collapses the original proportional checkpoint plus 10 proportional repeats into an 11-row proportional reference mean.

The packet was then corrected again to include the full 280-row single-phase augmented panel, not just the 240 single-phase qsplit rows. Single-phase domain-deletion rows reuse the original `pctrl_del_*` names, so the packet preserves those values in `original_run_name` and assigns disambiguated packet `run_name` values such as `singleavg_pctrl_del_*`. The phase-tied `baseline_stratified` checkpoint is represented as `singleavg_baseline_stratified` with shared-checkpoint provenance rather than redundantly retrained. Every row has `training_phase_family`, `phase_weight_structure`, `phase_correspondence_key`, and pairing columns; `data/single_two_phase_correspondence_300m.csv` summarizes the row correspondence. This post-review correction was locally validated by rebuilding the packet, auditing row counts and pair counts, and rerunning the standalone DSP smoke fit.
"""
    (paths.reviews / "cc_review_summary.md").write_text(summary)


def write_manifest(paths: PacketPaths, manifest: dict[str, Any]) -> None:
    (paths.root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def write_packet_dsp_script(paths: PacketPaths) -> None:
    text = DSP_SOURCE_PATH.read_text()
    pep723 = """# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scipy",
#   "scikit-learn",
# ]
# ///
"""
    if not text.startswith("# /// script"):
        text = pep723 + "\n" + text
    text = text.replace(
        "The default command fits canonical DSP on the packet-local\n300M/6B panel:\n\n    python standalone_code/dsp_exact.py fit --output-dir outputs/dsp_canonical_300m",
        "The default command fits DSP on the packet-local 300M/6B panel:\n\n"
        "    uv run --no-project --script code/effective_exposure_dsp.py fit "
        "--data-dir data --variant effective_exposure --output-dir outputs/table9_effexp_fit",
    )
    text = text.replace('DEFAULT_TARGET = "eval/uncheatable_eval/bpb"', 'DEFAULT_TARGET = "table9_macro_bpb"')
    text = text.replace('DEFAULT_RUN_SET = "swarm_like_300m"', 'DEFAULT_RUN_SET = "all_signal"')
    (paths.code / "effective_exposure_dsp.py").write_text(text)


def zip_packet(paths: PacketPaths) -> Path:
    zip_path = paths.root.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(paths.root.rglob("*")):
            if any(part.startswith(".") for part in path.relative_to(paths.root).parts):
                continue
            if path.is_file():
                zf.write(path, path.relative_to(paths.root.parent))
    return zip_path


def main() -> None:
    paths = packet_paths()
    reset_output(paths)

    train_source = pd.read_csv(TRAIN_TABLE9_PATH)
    table9_cols = table9_component_cols(train_source)
    phases = phase_cols(train_source)

    base_train = build_train_panel(table9_cols, phases)
    proportional_row = base_train.loc[base_train["run_name"].eq("baseline_proportional")].iloc[0]
    proportional_noise = build_proportional_noise_train(table9_cols, phases, proportional_row)
    train = pd.concat([base_train, proportional_noise], ignore_index=True, sort=False)
    train_names = set(train["run_name"])
    pctrl_tilts = build_pctrl_tilt_heldout(table9_cols, phases, train_names)
    one_phase = build_one_phase_heldout(table9_cols, phases)
    extra = build_extra_heldout(table9_cols, phases)

    heldout = pd.concat([pctrl_tilts, one_phase, extra], ignore_index=True, sort=False)
    all_rows = pd.concat([train, heldout], ignore_index=True, sort=False)
    all_rows = attach_uncheatable(all_rows)
    all_rows = add_phase_family_labels(all_rows, phases)
    all_rows = add_correspondence_keys(all_rows)
    all_rows = add_pair_columns(all_rows)
    train = all_rows[all_rows["split"].eq("train")].copy()
    heldout = all_rows[all_rows["split"].eq("heldout")].copy()
    fit_matrix = collapsed_fit_matrix(train, table9_cols)
    assert_packet_invariants(train=train, heldout=heldout, all_rows=all_rows, table9_cols=table9_cols, phases=phases)
    train_out = all_rows[all_rows["split"].eq("train")].copy()
    heldout_out = all_rows[all_rows["split"].eq("heldout")].copy()
    correspondence = phase_correspondence_table(all_rows)

    train_out.to_csv(paths.data / "train_augmented_fit_panel_300m.csv", index=False)
    fit_matrix.to_csv(paths.data / "fit_matrix_collapsed_proportional_300m.csv", index=False)
    heldout_out.to_csv(paths.data / "heldout_300m_checkpoint_metrics.csv", index=False)
    all_rows.to_csv(paths.data / "all_300m_checkpoint_metrics.csv", index=False)
    correspondence.to_csv(paths.data / "single_two_phase_correspondence_300m.csv", index=False)

    metric_registry = fit_matrix.copy()
    metric_registry["scale"] = "300m_6b"
    metric_registry["cohort"] = "signal"
    (paths.data / "metric_registry").mkdir(exist_ok=True)
    metric_registry.to_csv(paths.data / "metric_registry/metrics_wide.csv", index=False)

    (paths.data / "grp_no_l2").mkdir(exist_ok=True)
    shutil.copy2(EPOCH_METADATA_PATH, paths.data / "grp_no_l2/two_phase_many_epoch_metadata.csv")

    write_packet_dsp_script(paths)
    shutil.copy2(Path(__file__), paths.scripts / "build_two_phase_solver_gap_collaborator_packet.py")
    shutil.copy2(COMPONENT_METADATA_PATH, paths.data / "table9_component_metadata.json")
    write_enriched_one_vs_two_summary(paths)
    shutil.copy2(ONE_VS_TWO_DELTAS_PATH, paths.data / "one_vs_two_phase_best_mixture_deltas.csv")
    shutil.copy2(ONE_VS_TWO_HTML_PATH, paths.figures / "one_vs_two_phase_best_mixtures.html")
    write_review_summary(paths)

    coverage = {
        "by_split": grouped_counts_dict(all_rows, ["split"]),
        "by_packet_panel": grouped_counts_dict(all_rows, ["split", "packet_panel"]),
        "by_diagnostic_group": value_counts_dict(all_rows["diagnostic_group"].fillna("<missing>")),
        "by_training_phase_family": grouped_counts_dict(all_rows, ["split", "training_phase_family"]),
        "by_phase_pair_status": value_counts_dict(all_rows["phase_pair_status"].fillna("<missing>")),
        "table9_missing_rows": all_rows.loc[all_rows[TABLE9_MACRO_COL].isna(), "run_name"].tolist(),
        "uncheatable_missing_rows": all_rows.loc[all_rows["eval_uncheatable_eval_bpb"].isna(), "run_name"].tolist(),
    }
    manifest: dict[str, Any] = {
        "packet_name": paths.root.name,
        "created_by": Path(__file__).name,
        "summary": {
            "train_rows": len(train_out),
            "fit_matrix_rows": len(fit_matrix),
            "heldout_rows": len(heldout_out),
            "all_rows": len(all_rows),
            "proportional_reference_repeat_rows": int(proportional_noise["run_name"].nunique()),
            "proportional_reference_total_rows": 11,
            "single_phase_rows": int(all_rows["training_phase_family"].eq("single_phase").sum()),
            "two_phase_rows": int(all_rows["training_phase_family"].eq("two_phase").sum()),
            "single_phase_qsplit_rows": int(all_rows["diagnostic_group"].eq("single_phase_300m_qsplit").sum()),
            "single_phase_shared_stratified_rows": int(
                all_rows["diagnostic_group"].eq("single_phase_shared_stratified").sum()
            ),
            "shared_checkpoint_alias_rows": int(all_rows["is_shared_checkpoint_alias"].fillna(False).sum()),
            "single_phase_domain_deletion_rows": int(
                all_rows["diagnostic_group"].eq("single_phase_domain_deletion").sum()
            ),
            "phase_correspondence_key_count": len(correspondence),
            "paired_correspondence_key_count": int(correspondence["pair_status"].eq("paired_single_two").sum()),
            "table9_component_count": len(table9_cols),
            "phase_weight_column_count": len(phases),
            "table9_complete_rows": int(all_rows[TABLE9_MACRO_COL].notna().sum()),
            "uncheatable_complete_rows": int(all_rows["eval_uncheatable_eval_bpb"].notna().sum()),
        },
        "coverage": coverage,
        "source_paths": {
            "train_table9": str(TRAIN_TABLE9_PATH),
            "full_table9": str(FULL_TABLE9_PATH),
            "extra_heldout": str(EXTRA_HELDOUT_PATH),
            "pctrl_manifest": str(PCTRL_MANIFEST_PATH),
            "pctrl_uncheatable": str(PCTRL_UNCHEATABLE_PATH),
            "qsplit_uncheatable": str(QSP_UNCHEATABLE_PATH),
            "ppert_uncheatable": str(PPERT_UNCHEATABLE_PATH),
            "one_phase_scores": str(ONE_PHASE_SCORES_PATH),
            "one_vs_two_summary": str(ONE_VS_TWO_SUMMARY_PATH),
            "delphi_scaling_completed": str(DELPHI_SCALING_COMPLETED_PATH),
            "dsp_source": str(DSP_SOURCE_PATH),
            "epoch_metadata": str(EPOCH_METADATA_PATH),
            "component_metadata": str(COMPONENT_METADATA_PATH),
        },
        "outputs": {
            "train": "data/train_augmented_fit_panel_300m.csv",
            "fit_matrix": "data/fit_matrix_collapsed_proportional_300m.csv",
            "heldout": "data/heldout_300m_checkpoint_metrics.csv",
            "all": "data/all_300m_checkpoint_metrics.csv",
            "phase_correspondence": "data/single_two_phase_correspondence_300m.csv",
            "table9_component_metadata": "data/table9_component_metadata.json",
            "one_vs_two_summary": "data/one_vs_two_phase_best_mixture_summary.csv",
            "one_vs_two_validation_outcomes": "data/one_vs_two_phase_validation_outcomes_3e18.csv",
            "one_vs_two_domain_deltas": "data/one_vs_two_phase_best_mixture_deltas.csv",
            "one_vs_two_figure": "figures/one_vs_two_phase_best_mixtures.html",
            "metric_registry": "data/metric_registry/metrics_wide.csv",
            "epoch_metadata": "data/grp_no_l2/two_phase_many_epoch_metadata.csv",
            "dsp": "code/effective_exposure_dsp.py",
            "builder": "scripts/build_two_phase_solver_gap_collaborator_packet.py",
            "cc_review_summary": "reviews/cc_review_summary.md",
            "readme": "README.md",
        },
    }
    write_readme(paths, manifest)
    write_manifest(paths, manifest)
    zip_path = zip_packet(paths)
    print(json.dumps({"packet_dir": str(paths.root), "zip": str(zip_path), **manifest["summary"]}, indent=2))


if __name__ == "__main__":
    main()
