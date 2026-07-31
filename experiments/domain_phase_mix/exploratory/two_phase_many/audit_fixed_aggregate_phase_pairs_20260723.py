# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
# ]
# ///
"""Audit fixed-aggregate Delphi 3e18 phase-order pairs bucket by bucket."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
    all_top_level_domain_names,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_3e18_fixed_aggregate_qualitative_audit_20260723"

FIBER_PANEL_DIR = REFERENCE_OUTPUTS / "delphi_3e18_frontier_phase_fiber_20260719"
FIBER_RESULTS_DIR = REFERENCE_OUTPUTS / "delphi_3e18_frontier_phase_fiber_results_20260719"
AGGRESSIVE_PANEL_DIR = REFERENCE_OUTPUTS / "delphi_3e18_aggressive_phase_asymmetry_20260722"
AGGRESSIVE_RESULTS_DIR = REFERENCE_OUTPUTS / "delphi_3e18_aggressive_phase_asymmetry_results_20260723"

# The 3006-step run realizes the nominal 80/20 split at a step boundary.
PHASE_FRACTIONS = np.asarray([0.7981376787495843, 0.2018623212504157], dtype=float)
SIMULATED_EPOCH_TARGET_BUDGET = 6_325_183_647_689
TARGET_COLUMNS = {
    "uncheatable": "uncheatable_bpb",
    "table9": "table9_macro_bpb",
}
TARGET_LABELS = {
    "uncheatable": "Uncheatable",
    "table9": "Table-9",
}
PRIMARY_TARGET = {
    "uncheatable_frontier": "uncheatable",
    "table9_frontier": "table9",
}
TARGET_ANCHOR = {target: anchor for anchor, target in PRIMARY_TARGET.items()}
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
PAIR_GAP_BANDS = (
    (0.010, "large"),
    (0.005, "moderate"),
    (0.002, "small"),
    (0.000, "trace"),
)


@dataclass(frozen=True)
class PanelInputs:
    name: str
    manifest: pd.DataFrame
    results: pd.DataFrame
    weights: pd.DataFrame
    pair_columns: tuple[str, ...]


@dataclass(frozen=True)
class PhaseOrderRule:
    rule_id: str
    label: str
    premise: str
    expected_direction: str
    domains: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def bucket_semantics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain in all_top_level_domain_names():
        if domain.startswith("dolma3_cc/"):
            topic_quality = domain.removeprefix("dolma3_cc/")
            topic, quality = topic_quality.rsplit("_", 1)
            rows.append(
                {
                    "domain": domain,
                    "pool": "Dolma 3",
                    "semantic_group": f"broad_web_{quality}",
                    "content_family": "web",
                    "quality_tier": quality,
                    "description": (
                        f"Common Crawl {topic.replace('_', ' ')}, Dolma quality buckets "
                        f"{'10-19' if quality == 'high' else '0-9'}"
                    ),
                    "phase_prior": "broad replay / general capability",
                }
            )
            continue

        metadata = {
            "dolma3_stack_edu": (
                "Dolma 3",
                "broad_code",
                "code",
                "base",
                "Educationally filtered source code",
                "broad replay with code specialization",
            ),
            "dolma3_arxiv": (
                "Dolma 3",
                "broad_science",
                "science",
                "base",
                "arXiv scientific papers",
                "broad replay with technical specialization",
            ),
            "dolma3_finemath_3plus": (
                "Dolma 3",
                "broad_math",
                "math",
                "base",
                "FineMath score 3+ mathematical web text",
                "scarce target-like math data",
            ),
            "dolma3_wikipedia": (
                "Dolma 3",
                "broad_reference",
                "reference",
                "base",
                "Wikipedia and Wikibooks reference text",
                "scarce factual reference data",
            ),
            "dolmino_common_crawl_hq": (
                "Dolmino",
                "dolmino_web",
                "web",
                "high",
                "High-quality Common Crawl across topics",
                "quality-upsampled web / late-data candidate",
            ),
            "dolmino_olmocr_pdfs_hq": (
                "Dolmino",
                "dolmino_science",
                "science",
                "high",
                "High-quality olmOCR PDFs",
                "quality-upsampled technical data / late-data candidate",
            ),
            "dolmino_stack_edu_fim": (
                "Dolmino",
                "dolmino_code",
                "code",
                "high",
                "StackEdu source code with fill-in-the-middle transformation",
                "late code adaptation candidate",
            ),
            "dolmino_stem_heavy_crawl": (
                "Dolmino",
                "dolmino_science",
                "science",
                "high",
                "STEM-heavy filtered crawl",
                "scarce late STEM adaptation candidate",
            ),
            "dolmino_synth_code": (
                "Dolmino",
                "dolmino_code",
                "code",
                "synthetic",
                "Synthetic code from CraneCode",
                "scarce synthetic late-data candidate",
            ),
            "dolmino_synth_instruction": (
                "Dolmino",
                "dolmino_instruction",
                "instruction",
                "synthetic",
                "Synthetic FLAN and Tulu instruction data",
                "instruction adaptation candidate",
            ),
            "dolmino_synth_math": (
                "Dolmino",
                "dolmino_math",
                "math",
                "synthetic",
                "Synthetic and verifiable mathematics mixture",
                "scarce repeated math adaptation candidate",
            ),
            "dolmino_synth_qa": (
                "Dolmino",
                "dolmino_qa",
                "qa",
                "synthetic",
                "Synthetic QA, flashcards, and reading comprehension",
                "large target-like QA pool",
            ),
            "dolmino_synth_thinking": (
                "Dolmino",
                "dolmino_reasoning",
                "reasoning",
                "synthetic",
                "Synthetic reasoning and verifiable traces",
                "scarce reasoning adaptation candidate",
            ),
        }[domain]
        pool, group, family, quality, description, phase_prior = metadata
        rows.append(
            {
                "domain": domain,
                "pool": pool,
                "semantic_group": group,
                "content_family": family,
                "quality_tier": quality,
                "description": description,
                "phase_prior": phase_prior,
            }
        )

    frame = pd.DataFrame(rows)
    frame["available_tokens"] = frame["domain"].map(TOP_LEVEL_DOMAIN_TOKEN_COUNTS).astype(int)
    frame["available_tokens_b"] = frame["available_tokens"] / 1e9
    return frame


def load_panel_inputs() -> list[PanelInputs]:
    fiber_manifest = pd.read_csv(FIBER_PANEL_DIR / "candidate_manifest.csv")
    fiber_results = pd.read_csv(FIBER_RESULTS_DIR / "observed_results.csv")
    fiber_results = fiber_manifest.merge(
        fiber_results[
            [
                "candidate_id",
                *TARGET_COLUMNS.values(),
                "uncheatable_same_seed_center_bpb",
                "uncheatable_delta_vs_same_seed_center",
                "table9_same_seed_center_bpb",
                "table9_delta_vs_same_seed_center",
            ]
        ],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    fiber_results = fiber_results[~fiber_results["contrast_family"].eq("center_control")].copy()

    aggressive_manifest = pd.read_csv(AGGRESSIVE_PANEL_DIR / "candidate_manifest.csv")
    aggressive_results = pd.read_csv(AGGRESSIVE_RESULTS_DIR / "observed_results_with_control_deltas.csv")
    aggressive_results = aggressive_manifest.merge(
        aggressive_results[
            [
                "candidate_id",
                *TARGET_COLUMNS.values(),
                "uncheatable_same_seed_control_bpb",
                "uncheatable_delta_vs_control",
                "table9_same_seed_control_bpb",
                "table9_delta_vs_control",
            ]
        ],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    aggressive_results = aggressive_results[aggressive_results["contrast_family"].eq("balanced_partition")].copy()

    return [
        PanelInputs(
            name="frontier_phase_fiber",
            manifest=fiber_manifest,
            results=fiber_results,
            weights=pd.read_csv(FIBER_PANEL_DIR / "phase_weights.csv"),
            pair_columns=("anchor_id", "contrast_family", "direction_id", "seed_block"),
        ),
        PanelInputs(
            name="aggressive_balanced_partition",
            manifest=aggressive_manifest,
            results=aggressive_results,
            weights=pd.read_csv(AGGRESSIVE_PANEL_DIR / "phase_weights.csv"),
            pair_columns=("anchor_id", "contrast_family", "direction_id", "target_phase_tv", "seed_block"),
        ),
    ]


def phase_matrix(weights: pd.DataFrame, candidate_id: str, domains: list[str]) -> np.ndarray:
    candidate = weights[weights["candidate_id"].eq(candidate_id)]
    matrix = (
        candidate.pivot(index="phase", columns="domain", values="weight")
        .reindex(index=[0, 1], columns=domains)
        .to_numpy(float)
    )
    if matrix.shape != (2, len(domains)) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid phase weights for {candidate_id}: shape={matrix.shape}")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError(f"Phase weights do not sum to one for {candidate_id}: {matrix.sum(axis=1)}")
    return matrix


def compact_domain_list(values: np.ndarray, domains: list[str], *, largest: bool, count: int = 5) -> str:
    order = np.argsort(values)
    if largest:
        order = order[::-1]
    selected = [index for index in order if abs(values[index]) > 1e-12][:count]
    return "; ".join(f"{domains[index]} {values[index]:+.4f}" for index in selected)


def compact_epoch_list(
    better: np.ndarray,
    worse: np.ndarray,
    domains: list[str],
    available_tokens: np.ndarray,
    *,
    count: int = 5,
) -> str:
    late_epoch_delta = PHASE_FRACTIONS[1] * (better[1] - worse[1]) * SIMULATED_EPOCH_TARGET_BUDGET / available_tokens
    order = np.argsort(np.abs(late_epoch_delta))[::-1][:count]
    return "; ".join(f"{domains[index]} {late_epoch_delta[index]:+.2f}" for index in order)


def phase_anatomy(
    plus: np.ndarray,
    minus: np.ndarray,
    semantics: pd.DataFrame,
) -> dict[str, object]:
    domains = semantics["domain"].tolist()
    available_tokens = semantics["available_tokens"].to_numpy(float)
    aggregate_plus = PHASE_FRACTIONS @ plus
    aggregate_minus = PHASE_FRACTIONS @ minus
    plus_contrast = plus[1] - plus[0]
    minus_contrast = minus[1] - minus[0]
    if np.max(np.abs(aggregate_plus - aggregate_minus)) > 1e-9:
        raise ValueError("Antithetic pair does not preserve aggregate weights")
    if np.max(np.abs(plus_contrast + minus_contrast)) > 1e-8:
        raise ValueError("Antithetic pair is not symmetric around its tied aggregate")

    result: dict[str, object] = {
        "aggregate_max_abs_difference": float(np.max(np.abs(aggregate_plus - aggregate_minus))),
        "plus_phase_tv": float(0.5 * np.abs(plus_contrast).sum()),
        "plus_later_top": compact_domain_list(plus_contrast, domains, largest=True),
        "plus_earlier_top": compact_domain_list(plus_contrast, domains, largest=False),
        "plus_phase0_top_weights": compact_domain_list(plus[0], domains, largest=True),
        "plus_phase1_top_weights": compact_domain_list(plus[1], domains, largest=True),
        "minus_phase0_top_weights": compact_domain_list(minus[0], domains, largest=True),
        "minus_phase1_top_weights": compact_domain_list(minus[1], domains, largest=True),
    }
    for group, group_rows in semantics.groupby("semantic_group", sort=False):
        indices = group_rows.index.to_numpy(int)
        result[f"plus_late_mass__{group}"] = float(plus_contrast[indices].sum())
    for pool, pool_rows in semantics.groupby("pool", sort=False):
        indices = pool_rows.index.to_numpy(int)
        result[f"plus_late_mass__pool_{pool.lower().replace(' ', '_')}"] = float(plus_contrast[indices].sum())
    result["_plus_matrix"] = plus
    result["_minus_matrix"] = minus
    result["_available_tokens"] = available_tokens
    return result


def gap_band(gap: float) -> str:
    for threshold, label in PAIR_GAP_BANDS:
        if gap >= threshold:
            return label
    raise AssertionError("Unreachable")


def control_columns(panel_name: str, target: str) -> tuple[str, str]:
    if panel_name == "frontier_phase_fiber":
        return f"{target}_same_seed_center_bpb", f"{target}_delta_vs_same_seed_center"
    return f"{target}_same_seed_control_bpb", f"{target}_delta_vs_control"


def build_pair_ledgers(
    panels: list[PanelInputs],
    semantics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    domains = semantics["domain"].tolist()
    physical_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []

    for panel in panels:
        grouped = panel.results.groupby(list(panel.pair_columns), sort=True, dropna=False)
        for pair_index, (key, group) in enumerate(grouped):
            if set(group["sign"]) != {"plus", "minus"} or len(group) != 2:
                raise ValueError(f"Expected one plus/minus row for {panel.name} pair {key}: {group['sign'].tolist()}")
            plus_row = group[group["sign"].eq("plus")].iloc[0]
            minus_row = group[group["sign"].eq("minus")].iloc[0]
            plus_id = str(plus_row["candidate_id"])
            minus_id = str(minus_row["candidate_id"])
            plus = phase_matrix(panel.weights, plus_id, domains)
            minus = phase_matrix(panel.weights, minus_id, domains)
            anatomy = phase_anatomy(plus, minus, semantics)
            pair_id = f"{panel.name}:{plus_row['anchor_id']}:{plus_row['direction_id']}:{pair_index:03d}"
            physical = {
                "pair_id": pair_id,
                "panel": panel.name,
                "anchor_id": plus_row["anchor_id"],
                "contrast_family": plus_row["contrast_family"],
                "direction_id": plus_row["direction_id"],
                "direction_label_plus": plus_row["direction_label"],
                "phase_tv": anatomy["plus_phase_tv"],
                "target_phase_tv": float(plus_row.get("target_phase_tv", anatomy["plus_phase_tv"])),
                "seed_block": int(plus_row["seed_block"]),
                "data_seed": int(plus_row["data_seed"]),
                "plus_candidate_id": plus_id,
                "minus_candidate_id": minus_id,
                **{key: value for key, value in anatomy.items() if not key.startswith("_")},
            }

            target_winners: dict[str, str] = {}
            for target, metric_column in TARGET_COLUMNS.items():
                plus_bpb = float(plus_row[metric_column])
                minus_bpb = float(minus_row[metric_column])
                better_sign = "plus" if plus_bpb < minus_bpb else "minus"
                better_row = plus_row if better_sign == "plus" else minus_row
                worse_row = minus_row if better_sign == "plus" else plus_row
                better_matrix = plus if better_sign == "plus" else minus
                worse_matrix = minus if better_sign == "plus" else plus
                center_column, delta_column = control_columns(panel.name, target)
                better_delta = float(better_row[delta_column])
                worse_delta = float(worse_row[delta_column])
                gap = abs(plus_bpb - minus_bpb)
                odd_order_effect = 0.5 * (plus_bpb - minus_bpb)
                even_asymmetry_cost = 0.5 * (plus_bpb + minus_bpb) - float(plus_row[center_column])
                better_contrast = better_matrix[1] - better_matrix[0]
                target_winners[target] = better_sign
                row: dict[str, object] = {
                    **physical,
                    "target": target,
                    "is_anchor_matched_target": PRIMARY_TARGET[str(plus_row["anchor_id"])] == target,
                    "plus_bpb": plus_bpb,
                    "minus_bpb": minus_bpb,
                    "center_bpb": float(plus_row[center_column]),
                    "plus_delta_vs_control": float(plus_row[delta_column]),
                    "minus_delta_vs_control": float(minus_row[delta_column]),
                    "better_sign": better_sign,
                    "better_candidate_id": str(better_row["candidate_id"]),
                    "worse_candidate_id": str(worse_row["candidate_id"]),
                    "better_bpb": min(plus_bpb, minus_bpb),
                    "worse_bpb": max(plus_bpb, minus_bpb),
                    "better_minus_control_bpb": better_delta,
                    "worse_minus_control_bpb": worse_delta,
                    "pair_gap_bpb": gap,
                    "odd_order_effect_bpb": odd_order_effect,
                    "abs_odd_order_effect_bpb": abs(odd_order_effect),
                    "even_asymmetry_cost_bpb": even_asymmetry_cost,
                    "even_cost_exceeds_order_signal": even_asymmetry_cost > abs(odd_order_effect),
                    "gap_band": gap_band(gap),
                    "better_beats_control": better_delta < 0.0,
                    "both_beat_control": better_delta < 0.0 and worse_delta < 0.0,
                    "both_worse_than_control": better_delta > 0.0 and worse_delta > 0.0,
                    "better_later_top": compact_domain_list(better_contrast, domains, largest=True),
                    "better_earlier_top": compact_domain_list(better_contrast, domains, largest=False),
                    "better_vs_worse_late_epoch_delta_top": compact_epoch_list(
                        better_matrix,
                        worse_matrix,
                        domains,
                        anatomy["_available_tokens"],
                    ),
                }
                for group, group_rows in semantics.groupby("semantic_group", sort=False):
                    indices = group_rows.index.to_numpy(int)
                    row[f"better_late_mass__{group}"] = float(better_contrast[indices].sum())
                target_rows.append(row)

            physical["same_better_sign_both_targets"] = target_winners["uncheatable"] == target_winners["table9"]
            physical["uncheatable_better_sign"] = target_winners["uncheatable"]
            physical["table9_better_sign"] = target_winners["table9"]
            physical_rows.append(physical)

    physical_frame = pd.DataFrame(physical_rows).sort_values(
        ["panel", "anchor_id", "contrast_family", "direction_id", "phase_tv"]
    )
    target_frame = pd.DataFrame(target_rows).sort_values(
        ["panel", "anchor_id", "contrast_family", "direction_id", "phase_tv", "target"]
    )
    if len(physical_frame) != 192 or len(target_frame) != 384:
        raise ValueError(
            f"Expected 192 physical and 384 target-specific pairs, got {len(physical_frame)}, {len(target_frame)}"
        )
    return physical_frame.reset_index(drop=True), target_frame.reset_index(drop=True)


def target_summary(target_pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scope, panel, target), group in pd.concat(
        [
            target_pairs.assign(scope="all"),
            target_pairs[target_pairs["is_anchor_matched_target"]].assign(scope="anchor_matched"),
        ],
        ignore_index=True,
    ).groupby(["scope", "panel", "target"], sort=True):
        rows.append(
            {
                "scope": scope,
                "panel": panel,
                "target": target,
                "n_pairs": len(group),
                "median_pair_gap_bpb": float(group["pair_gap_bpb"].median()),
                "mean_pair_gap_bpb": float(group["pair_gap_bpb"].mean()),
                "pairs_gap_ge_0p005": int(group["pair_gap_bpb"].ge(0.005).sum()),
                "pairs_gap_ge_0p010": int(group["pair_gap_bpb"].ge(0.010).sum()),
                "better_sign_beats_control": int(group["better_beats_control"].sum()),
                "both_signs_beat_control": int(group["both_beat_control"].sum()),
                "both_signs_worse_than_control": int(group["both_worse_than_control"].sum()),
                "median_best_delta_vs_control": float(group["better_minus_control_bpb"].median()),
                "best_best_delta_vs_control": float(group["better_minus_control_bpb"].min()),
            }
        )
    return pd.DataFrame(rows)


def odd_even_summary(target_pairs: pd.DataFrame) -> pd.DataFrame:
    matched = target_pairs[target_pairs["is_anchor_matched_target"]].copy()
    matched["design_radius"] = np.where(
        matched["panel"].eq("frontier_phase_fiber"),
        "local fiber",
        matched["target_phase_tv"].map(lambda value: f"TV {value:.2f}"),
    )
    rows: list[dict[str, object]] = []
    for (panel, target, design_radius), group in matched.groupby(
        ["panel", "target", "design_radius"],
        sort=True,
    ):
        rows.append(
            {
                "panel": panel,
                "target": target,
                "design_radius": design_radius,
                "n_pairs": len(group),
                "median_abs_order_effect_bpb": float(group["abs_odd_order_effect_bpb"].median()),
                "median_even_asymmetry_cost_bpb": float(group["even_asymmetry_cost_bpb"].median()),
                "mean_even_asymmetry_cost_bpb": float(group["even_asymmetry_cost_bpb"].mean()),
                "pairs_even_cost_exceeds_order_signal": int(group["even_cost_exceeds_order_signal"].sum()),
                "pairs_oracle_sign_beats_tied": int(group["better_beats_control"].sum()),
                "pairs_both_signs_worse_than_tied": int(group["both_worse_than_control"].sum()),
            }
        )
    return pd.DataFrame(rows)


def control_noise_summary(target_pairs: pd.DataFrame) -> pd.DataFrame:
    controls = target_pairs[["panel", "anchor_id", "target", "seed_block", "center_bpb"]].drop_duplicates()
    return (
        controls.groupby(["panel", "anchor_id", "target"], sort=True)["center_bpb"]
        .agg(control_count="count", control_mean_bpb="mean", control_sd_bpb="std")
        .reset_index()
    )


def domain_vs_rest_summary(target_pairs: pd.DataFrame, semantics: pd.DataFrame) -> pd.DataFrame:
    rows = target_pairs[target_pairs["contrast_family"].eq("domain_vs_rest")].copy()
    rows["domain_index"] = rows["direction_id"].str.removeprefix("domain_").astype(int)
    rows["domain"] = rows["domain_index"].map(dict(enumerate(semantics["domain"])))
    rows["domain_later_better"] = rows["better_sign"].eq("plus")
    rows["domain_later_minus_earlier_bpb"] = rows["plus_bpb"] - rows["minus_bpb"]
    rows["domain_later_minus_earlier_bpb_per_phase_tv"] = rows["domain_later_minus_earlier_bpb"] / rows["phase_tv"]
    return rows[
        [
            "anchor_id",
            "target",
            "is_anchor_matched_target",
            "domain",
            "phase_tv",
            "domain_later_better",
            "domain_later_minus_earlier_bpb",
            "domain_later_minus_earlier_bpb_per_phase_tv",
            "pair_gap_bpb",
            "better_minus_control_bpb",
            "worse_minus_control_bpb",
        ]
    ].merge(
        semantics[["domain", "pool", "semantic_group", "content_family", "quality_tier", "available_tokens_b"]],
        on="domain",
        how="left",
        validate="many_to_one",
    )


def domain_order_hypotheses(domain_summary: pd.DataFrame) -> pd.DataFrame:
    index_columns = ["target", "domain", "pool", "semantic_group", "content_family", "quality_tier"]
    pivot = domain_summary.pivot_table(
        index=index_columns,
        columns="anchor_id",
        values="domain_later_minus_earlier_bpb",
        aggfunc="first",
    ).reset_index()
    slope_pivot = (
        domain_summary.pivot_table(
            index=index_columns,
            columns="anchor_id",
            values="domain_later_minus_earlier_bpb_per_phase_tv",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={anchor: f"{anchor}_bpb_per_phase_tv" for anchor in PRIMARY_TARGET})
    )
    pivot = pivot.merge(slope_pivot, on=index_columns, how="inner", validate="one_to_one")
    anchor_columns = sorted(set(PRIMARY_TARGET) & set(pivot.columns))
    if len(anchor_columns) != 2:
        raise ValueError(f"Expected two frontier-anchor columns, found {anchor_columns}")
    first, second = anchor_columns
    pivot["mean_domain_later_minus_earlier_bpb"] = pivot[anchor_columns].mean(axis=1)
    pivot["mean_domain_later_minus_earlier_bpb_per_phase_tv"] = pivot[
        [f"{anchor}_bpb_per_phase_tv" for anchor in anchor_columns]
    ].mean(axis=1)
    pivot["same_sign_across_anchors"] = np.sign(pivot[first]) == np.sign(pivot[second])
    pivot["domain_later_preferred_across_anchors"] = pivot["same_sign_across_anchors"] & pivot[
        "mean_domain_later_minus_earlier_bpb"
    ].lt(0.0)
    return pivot.sort_values(
        ["target", "same_sign_across_anchors", "mean_domain_later_minus_earlier_bpb"],
        ascending=[True, False, True],
    )


def cross_anchor_direction(first_effect: float, second_effect: float) -> str:
    if first_effect < 0.0 and second_effect < 0.0:
        return "later_consistent"
    if first_effect > 0.0 and second_effect > 0.0:
        return "earlier_consistent"
    return "anchor_dependent"


def bucket_phase_order_dossier(domain_summary: pd.DataFrame, semantics: pd.DataFrame) -> pd.DataFrame:
    semantic_lookup = semantics.set_index("domain")
    rows: list[dict[str, object]] = []
    anchor_columns = sorted(PRIMARY_TARGET)
    for domain in semantics["domain"]:
        metadata = semantic_lookup.loc[domain]
        record: dict[str, object] = {
            "domain": domain,
            "pool": metadata["pool"],
            "semantic_group": metadata["semantic_group"],
            "content_family": metadata["content_family"],
            "quality_tier": metadata["quality_tier"],
            "description": metadata["description"],
            "phase_prior": metadata["phase_prior"],
            "available_tokens_b": float(metadata["available_tokens_b"]),
        }
        target_directions: dict[str, str] = {}
        for target in ("table9", "uncheatable"):
            target_rows = domain_summary[
                domain_summary["domain"].eq(domain) & domain_summary["target"].eq(target)
            ].set_index("anchor_id")
            effects = {
                anchor: float(target_rows.loc[anchor, "domain_later_minus_earlier_bpb"]) for anchor in anchor_columns
            }
            phase_tvs = {anchor: float(target_rows.loc[anchor, "phase_tv"]) for anchor in anchor_columns}
            slopes = {
                anchor: float(target_rows.loc[anchor, "domain_later_minus_earlier_bpb_per_phase_tv"])
                for anchor in anchor_columns
            }
            direction = cross_anchor_direction(effects[anchor_columns[0]], effects[anchor_columns[1]])
            target_directions[target] = direction
            matched = target_rows.loc[TARGET_ANCHOR[target]]
            net_gain = -float(matched["better_minus_control_bpb"])
            opposite_minus_tied = float(matched["worse_minus_control_bpb"])
            mean_effect = float(np.mean(list(effects.values())))
            mean_slope = float(np.mean(list(slopes.values())))
            if direction == "anchor_dependent":
                conclusion = "anchor-dependent"
            else:
                preferred = direction.removesuffix("_consistent")
                if net_gain <= 0.0:
                    conclusion = f"{preferred}-less-harmful; tied-better"
                elif opposite_minus_tied < 0.0:
                    conclusion = f"{preferred}-preferred; both-orders-helpful"
                else:
                    conclusion = f"{preferred}-directionally-helpful"
            record.update(
                {
                    f"{target}_table9_anchor_later_minus_earlier_bpb": effects["table9_frontier"],
                    f"{target}_uncheatable_anchor_later_minus_earlier_bpb": effects["uncheatable_frontier"],
                    f"{target}_table9_anchor_phase_tv": phase_tvs["table9_frontier"],
                    f"{target}_uncheatable_anchor_phase_tv": phase_tvs["uncheatable_frontier"],
                    f"{target}_table9_anchor_later_minus_earlier_bpb_per_phase_tv": slopes["table9_frontier"],
                    f"{target}_uncheatable_anchor_later_minus_earlier_bpb_per_phase_tv": slopes["uncheatable_frontier"],
                    f"{target}_cross_anchor_mean_later_minus_earlier_bpb": mean_effect,
                    f"{target}_cross_anchor_mean_later_minus_earlier_bpb_per_phase_tv": mean_slope,
                    f"{target}_cross_anchor_direction": direction,
                    f"{target}_effect_scale": gap_band(abs(mean_effect)),
                    f"{target}_matched_anchor_pair_gap_bpb": float(matched["pair_gap_bpb"]),
                    f"{target}_matched_anchor_preferred_minus_tied_bpb": float(matched["better_minus_control_bpb"]),
                    f"{target}_matched_anchor_opposite_minus_tied_bpb": opposite_minus_tied,
                    f"{target}_matched_anchor_net_gain_bpb": net_gain,
                    f"{target}_matched_anchor_both_orderings_worse_than_tied": bool(
                        float(matched["better_minus_control_bpb"]) > 0.0
                    ),
                    f"{target}_qualitative_conclusion": conclusion,
                }
            )

        table9_direction = target_directions["table9"]
        uncheatable_direction = target_directions["uncheatable"]
        if "anchor_dependent" in (table9_direction, uncheatable_direction):
            cross_target_pattern = "at_least_one_target_anchor_dependent"
        elif table9_direction == uncheatable_direction:
            cross_target_pattern = "same_direction_both_targets"
        else:
            cross_target_pattern = "objective_specific_reversal"
        record["cross_target_pattern"] = cross_target_pattern
        rows.append(record)
    return pd.DataFrame(rows)


def phase_order_rules(semantics: pd.DataFrame) -> tuple[PhaseOrderRule, ...]:
    scarce_threshold = float(semantics["available_tokens_b"].quantile(0.25))

    def domains_where(mask: pd.Series) -> tuple[str, ...]:
        return tuple(semantics.loc[mask, "domain"])

    return (
        PhaseOrderRule(
            rule_id="dolmino_late",
            label="All Dolmino belongs late",
            premise="Curated, filtered, or synthetic Dolmino data should be concentrated in the final phase.",
            expected_direction="later",
            domains=domains_where(semantics["pool"].eq("Dolmino")),
        ),
        PhaseOrderRule(
            rule_id="synthetic_late",
            label="Synthetic data belongs late",
            premise="Synthetic code, instruction, math, QA, and reasoning data should behave like finetuning data.",
            expected_direction="later",
            domains=domains_where(semantics["quality_tier"].eq("synthetic")),
        ),
        PhaseOrderRule(
            rule_id="high_quality_late",
            label="Higher-quality buckets belong late",
            premise="Dolma high-quality topic buckets and high-quality Dolmino buckets should benefit from recency.",
            expected_direction="later",
            domains=domains_where(semantics["quality_tier"].eq("high")),
        ),
        PhaseOrderRule(
            rule_id="broad_pretraining_early",
            label="Broad pretraining data belongs early",
            premise="Dolma 3 broad/pretraining buckets should build the base and only be replayed minimally late.",
            expected_direction="earlier",
            domains=domains_where(semantics["pool"].eq("Dolma 3")),
        ),
        PhaseOrderRule(
            rule_id="scarce_data_late",
            label="Scarce data belongs late",
            premise=(
                "Bottom-quartile corpus-size buckets "
                f"(at most {scarce_threshold:.2f}B available tokens) should be saved for the final phase."
            ),
            expected_direction="later",
            domains=domains_where(semantics["available_tokens_b"].le(scarce_threshold)),
        ),
        PhaseOrderRule(
            rule_id="code_late",
            label="Code data belongs late",
            premise="Code specialization should benefit from recency regardless of source or transformation.",
            expected_direction="later",
            domains=domains_where(semantics["content_family"].eq("code")),
        ),
        PhaseOrderRule(
            rule_id="math_qa_reasoning_late",
            label="Math, QA, and reasoning data belongs late",
            premise="Target-like math, QA, and reasoning data should behave like a final specialization mixture.",
            expected_direction="later",
            domains=domains_where(semantics["content_family"].isin(["math", "qa", "reasoning"])),
        ),
    )


def rule_test_results(
    dossier: pd.DataFrame,
    rules: tuple[PhaseOrderRule, ...],
) -> pd.DataFrame:
    dossier_lookup = dossier.set_index("domain")
    rows: list[dict[str, object]] = []
    for rule in rules:
        opposite = "earlier" if rule.expected_direction == "later" else "later"
        for target in ("table9", "uncheatable"):
            for domain in rule.domains:
                bucket = dossier_lookup.loc[domain]
                direction = str(bucket[f"{target}_cross_anchor_direction"])
                if direction == f"{rule.expected_direction}_consistent":
                    outcome = "supports"
                elif direction == f"{opposite}_consistent":
                    outcome = "counterexample"
                else:
                    outcome = "anchor_dependent"
                net_gain = float(bucket[f"{target}_matched_anchor_net_gain_bpb"])
                opposite_minus_tied = float(bucket[f"{target}_matched_anchor_opposite_minus_tied_bpb"])
                matched_effect = float(
                    bucket[f"{target}_{TARGET_ANCHOR[target].removesuffix('_frontier')}_anchor_later_minus_earlier_bpb"]
                )
                if outcome != "anchor_dependent":
                    if net_gain <= 0.0:
                        evidence_kind = "direction_only"
                        control_comment = (
                            f"The preferred ordering is less harmful, but tied remains better by "
                            f"{-net_gain:.6f} BPB."
                        )
                    elif opposite_minus_tied < 0.0:
                        evidence_kind = "both_orders_beat_tied"
                        control_comment = (
                            "Both asymmetric orderings beat tied; the preferred sign wins the pair gap, so "
                            "the net gain cannot be attributed to order alone."
                        )
                    else:
                        evidence_kind = "direction_specific"
                        control_comment = f"Only the preferred ordering beats tied, improving it by {net_gain:.6f} BPB."
                else:
                    evidence_kind = ""
                    control_comment = ""
                counterexample_kind = evidence_kind if outcome == "counterexample" else ""
                if outcome == "counterexample":
                    comment = (
                        f"{domain} is {str(bucket['description']).lower()}; {opposite} is preferred at both "
                        f"aggregate anchors (later-minus-earlier effects "
                        f"{float(bucket[f'{target}_table9_anchor_later_minus_earlier_bpb']):+.6f} at phase TV "
                        f"{float(bucket[f'{target}_table9_anchor_phase_tv']):.4f} and "
                        f"{float(bucket[f'{target}_uncheatable_anchor_later_minus_earlier_bpb']):+.6f} BPB at "
                        f"phase TV {float(bucket[f'{target}_uncheatable_anchor_phase_tv']):.4f}). "
                        f"{control_comment}"
                    )
                else:
                    comment = ""
                rows.append(
                    {
                        "rule_id": rule.rule_id,
                        "rule_label": rule.label,
                        "premise": rule.premise,
                        "expected_direction": rule.expected_direction,
                        "target": target,
                        "domain": domain,
                        "pool": bucket["pool"],
                        "content_family": bucket["content_family"],
                        "quality_tier": bucket["quality_tier"],
                        "available_tokens_b": float(bucket["available_tokens_b"]),
                        "observed_direction": direction,
                        "outcome": outcome,
                        "evidence_kind": evidence_kind,
                        "counterexample_kind": counterexample_kind,
                        "table9_anchor_later_minus_earlier_bpb": float(
                            bucket[f"{target}_table9_anchor_later_minus_earlier_bpb"]
                        ),
                        "uncheatable_anchor_later_minus_earlier_bpb": float(
                            bucket[f"{target}_uncheatable_anchor_later_minus_earlier_bpb"]
                        ),
                        "cross_anchor_mean_later_minus_earlier_bpb": float(
                            bucket[f"{target}_cross_anchor_mean_later_minus_earlier_bpb"]
                        ),
                        "cross_anchor_mean_later_minus_earlier_bpb_per_phase_tv": float(
                            bucket[f"{target}_cross_anchor_mean_later_minus_earlier_bpb_per_phase_tv"]
                        ),
                        "matched_anchor_later_minus_earlier_bpb": matched_effect,
                        "matched_anchor_net_gain_bpb": net_gain,
                        "matched_anchor_opposite_minus_tied_bpb": opposite_minus_tied,
                        "comment": comment,
                    }
                )
    return pd.DataFrame(rows)


def rule_summary_table(rule_results: pd.DataFrame, *, include_label: bool) -> pd.DataFrame:
    keys = ["rule_id"]
    if include_label:
        keys.append("rule_label")
    keys.append("target")
    rows = rule_results.assign(
        supports=rule_results["outcome"].eq("supports").astype(int),
        direction_specific_supports=(
            rule_results["outcome"].eq("supports") & rule_results["evidence_kind"].eq("direction_specific")
        ).astype(int),
        counterexamples=rule_results["outcome"].eq("counterexample").astype(int),
        direction_specific_counterexamples=(
            rule_results["outcome"].eq("counterexample") & rule_results["evidence_kind"].eq("direction_specific")
        ).astype(int),
        anchor_dependent=rule_results["outcome"].eq("anchor_dependent").astype(int),
    )
    return (
        rows.groupby(keys, sort=True)[
            [
                "supports",
                "direction_specific_supports",
                "counterexamples",
                "direction_specific_counterexamples",
                "anchor_dependent",
            ]
        ]
        .sum()
        .reset_index()
    )


def quality_tier_contrasts(dossier: pd.DataFrame) -> pd.DataFrame:
    cc_rows = dossier[dossier["domain"].str.startswith("dolma3_cc/")].copy()
    cc_rows["topic"] = cc_rows["domain"].str.removeprefix("dolma3_cc/").str.rsplit("_", n=1).str[0]
    cc_rows["tier"] = cc_rows["domain"].str.rsplit("_", n=1).str[1]
    rows: list[dict[str, object]] = []
    for target in ("table9", "uncheatable"):
        normalized_column = f"{target}_cross_anchor_mean_later_minus_earlier_bpb_per_phase_tv"
        raw_column = f"{target}_cross_anchor_mean_later_minus_earlier_bpb"
        normalized = cc_rows.pivot(index="topic", columns="tier", values=normalized_column)
        raw = cc_rows.pivot(index="topic", columns="tier", values=raw_column)
        for topic, normalized_values in normalized.iterrows():
            raw_values = raw.loc[topic]
            high_minus_low = float(normalized_values["high"] - normalized_values["low"])
            raw_high_minus_low = float(raw_values["high"] - raw_values["low"])
            rows.append(
                {
                    "target": target,
                    "topic": topic,
                    "high_later_minus_earlier_bpb_per_phase_tv": float(normalized_values["high"]),
                    "low_later_minus_earlier_bpb_per_phase_tv": float(normalized_values["low"]),
                    "high_minus_low_order_slope_bpb_per_phase_tv": high_minus_low,
                    "supports_high_quality_later_relative_to_low": high_minus_low < 0.0,
                    "high_later_minus_earlier_bpb_raw": float(raw_values["high"]),
                    "low_later_minus_earlier_bpb_raw": float(raw_values["low"]),
                    "high_minus_low_order_effect_bpb_raw": raw_high_minus_low,
                    "supports_high_quality_later_relative_to_low_raw": raw_high_minus_low < 0.0,
                }
            )
    return pd.DataFrame(rows)


def effect_transfer_summary(target_pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def append_summary(
        *,
        comparison: str,
        panel: str,
        slice_label: str,
        frame: pd.DataFrame,
        left_column: str,
        right_column: str,
    ) -> None:
        left = frame[left_column].to_numpy(float)
        right = frame[right_column].to_numpy(float)
        rows.append(
            {
                "comparison": comparison,
                "panel": panel,
                "slice": slice_label,
                "n": len(frame),
                "same_sign_fraction": float(np.mean(np.sign(left) == np.sign(right))),
                "pearson": float(frame[left_column].corr(frame[right_column], method="pearson")),
                "spearman": float(frame[left_column].corr(frame[right_column], method="spearman")),
            }
        )

    for panel, panel_rows in target_pairs.groupby("panel", sort=True):
        index_columns = ["anchor_id", "direction_id"]
        if panel == "aggressive_balanced_partition":
            index_columns.append("target_phase_tv")
        target_pivot = panel_rows.pivot_table(
            index=index_columns,
            columns="target",
            values="odd_order_effect_bpb",
            aggfunc="first",
        ).dropna()
        append_summary(
            comparison="cross_target_same_anchor",
            panel=panel,
            slice_label="all anchors",
            frame=target_pivot,
            left_column="uncheatable",
            right_column="table9",
        )

        for target, target_rows in panel_rows.groupby("target", sort=True):
            anchor_index = ["direction_id"]
            if panel == "aggressive_balanced_partition":
                anchor_index.append("target_phase_tv")
            anchor_pivot = target_rows.pivot_table(
                index=anchor_index,
                columns="anchor_id",
                values="odd_order_effect_bpb",
                aggfunc="first",
            ).dropna()
            append_summary(
                comparison="cross_anchor_same_target",
                panel=panel,
                slice_label=target,
                frame=anchor_pivot,
                left_column="table9_frontier",
                right_column="uncheatable_frontier",
            )

    return pd.DataFrame(rows)


def structured_schedule_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(AGGRESSIVE_RESULTS_DIR / "observed_results_with_control_deltas.csv")
    handcrafted_rows: list[dict[str, object]] = []
    handcrafted = results[results["contrast_family"].eq("handcrafted_late_quality")]
    for (anchor_id, direction_id), group in handcrafted.groupby(["anchor_id", "direction_id"], sort=True):
        target = PRIMARY_TARGET[anchor_id]
        delta_column = f"{target}_delta_vs_control"
        ordered = group.sort_values("target_phase_tv")
        record: dict[str, object] = {
            "anchor_id": anchor_id,
            "target": target,
            "direction_id": direction_id,
            "mean_delta_bpb": float(ordered[delta_column].mean()),
            "all_three_radii_beat_tied": bool(ordered[delta_column].lt(0.0).all()),
        }
        for row in ordered.itertuples(index=False):
            radius_label = str(row.target_phase_tv).replace(".", "p")
            record[f"delta_bpb_tv_{radius_label}"] = float(getattr(row, delta_column))
        handcrafted_rows.append(record)

    continuum_rows: list[dict[str, object]] = []
    continuum = results[results["contrast_family"].eq("dolmino_late_continuum")]
    for (anchor_id, direction_id), group in continuum.groupby(["anchor_id", "direction_id"], sort=True):
        target = PRIMARY_TARGET[anchor_id]
        delta_column = f"{target}_delta_vs_control"
        continuum_rows.append(
            {
                "anchor_id": anchor_id,
                "target": target,
                "direction_id": direction_id,
                "late_dolmino_share": float(group["phase_1_dolmino_share"].iloc[0]),
                "phase_tv": float(group["phase_tv"].iloc[0]),
                "replicate_count": len(group),
                "mean_delta_bpb": float(group[delta_column].mean()),
                "sd_delta_bpb": float(group[delta_column].std(ddof=1)),
                "all_replicates_beat_tied": bool(group[delta_column].lt(0.0).all()),
            }
        )

    return pd.DataFrame(handcrafted_rows), pd.DataFrame(continuum_rows)


def balanced_direction_consistency(target_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = target_pairs[target_pairs["contrast_family"].eq("balanced_partition")].copy()
    pivot = rows.pivot_table(
        index=["anchor_id", "target", "direction_id"],
        columns="target_phase_tv",
        values=["better_sign", "plus_bpb", "minus_bpb"],
        aggfunc="first",
    )
    records: list[dict[str, object]] = []
    for index, record in pivot.iterrows():
        anchor_id, target, direction_id = index
        signed_effects = {
            float(tv): float(record[("plus_bpb", tv)] - record[("minus_bpb", tv)]) for tv in (0.1, 0.25, 0.5)
        }
        signs = [np.sign(value) for value in signed_effects.values() if abs(value) > 1e-12]
        records.append(
            {
                "anchor_id": anchor_id,
                "target": target,
                "is_anchor_matched_target": PRIMARY_TARGET[anchor_id] == target,
                "direction_id": direction_id,
                **{f"plus_minus_bpb_tv_{str(tv).replace('.', 'p')}": value for tv, value in signed_effects.items()},
                "same_preferred_sign_all_radii": len(set(signs)) == 1,
                "preferred_sign_flips": int(sum(left != right for left, right in pairwise(signs))),
            }
        )
    return pd.DataFrame(records)


def helpful_bucket_rows(dossier: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for target in ("table9", "uncheatable"):
        conclusion_column = f"{target}_qualitative_conclusion"
        for row in dossier[dossier[conclusion_column].str.endswith("-directionally-helpful")].itertuples(index=False):
            conclusion = str(getattr(row, conclusion_column))
            records.append(
                {
                    "target": target,
                    "preferred_phase": conclusion.removesuffix("-directionally-helpful"),
                    "domain": row.domain,
                    "pool": row.pool,
                    "description": row.description,
                    "cross_anchor_mean_later_minus_earlier_bpb": getattr(
                        row, f"{target}_cross_anchor_mean_later_minus_earlier_bpb"
                    ),
                    "cross_anchor_mean_later_minus_earlier_bpb_per_phase_tv": getattr(
                        row, f"{target}_cross_anchor_mean_later_minus_earlier_bpb_per_phase_tv"
                    ),
                    "matched_anchor_net_gain_bpb": getattr(row, f"{target}_matched_anchor_net_gain_bpb"),
                }
            )
    return pd.DataFrame(records)


def bucket_phase_order_map(dossier: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "domain",
        "pool",
        "content_family",
        "quality_tier",
        "available_tokens_b",
        "phase_prior",
        "table9_qualitative_conclusion",
        "table9_matched_anchor_net_gain_bpb",
        "table9_cross_anchor_mean_later_minus_earlier_bpb_per_phase_tv",
        "uncheatable_qualitative_conclusion",
        "uncheatable_matched_anchor_net_gain_bpb",
        "uncheatable_cross_anchor_mean_later_minus_earlier_bpb_per_phase_tv",
        "cross_target_pattern",
    ]
    return dossier[columns].copy()


def write_bucket_dossier(
    output_path: Path,
    dossier: pd.DataFrame,
    rule_results: pd.DataFrame,
) -> None:
    helpful = helpful_bucket_rows(dossier)
    phase_order_map = bucket_phase_order_map(dossier)
    lines = [
        "# Bucket-by-bucket phase-order dossier",
        "",
        "This dossier evaluates each of the 39 buckets using the domain-versus-rest antithetic contrast at both "
        "aggregate anchors. A negative later-minus-earlier effect favors moving the named bucket later. "
        "`Directionally helpful` requires the same preferred direction at both anchors, the preferred ordering "
        "beating the same-seed tied control, and its reversal losing to tied. `Both-orders helpful` means both "
        "asymmetric schedules beat tied, so the net gain cannot be assigned to order. `Less harmful` means the "
        "direction is stable but tied still wins. These are descriptive labels after inspecting 39 buckets, two "
        "targets, and two anchors; no multiple-testing-adjusted significance is claimed.",
        "",
        "## Buckets with direction-specific matched-anchor gains",
        "",
        helpful.sort_values(
            ["target", "preferred_phase", "matched_anchor_net_gain_bpb"], ascending=[True, True, False]
        ).to_markdown(
            index=False,
            floatfmt=".6f",
        ),
        "",
        "## Compact 39-bucket map",
        "",
        "Positive matched-anchor net gain means the preferred ordering beats tied. The normalized slope is "
        "later-minus-earlier BPB per unit phase TV, so negative favors later and positive favors earlier. "
        "Normalization makes differently sized fibers comparable in units, but amplifies noise for tiny fibers; "
        "always read it with the raw effects in the individual notes.",
        "",
        phase_order_map.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Individual bucket notes",
        "",
    ]
    for index, bucket in enumerate(dossier.itertuples(index=False), start=1):
        lines.extend(
            [
                f"### {index:02d}. `{bucket.domain}`",
                "",
                f"- Semantics: {bucket.description}. Pool `{bucket.pool}`; family `{bucket.content_family}`; "
                f"quality tier `{bucket.quality_tier}`; {bucket.available_tokens_b:.3f}B available tokens.",
                f"- Prior before this audit: {bucket.phase_prior}.",
            ]
        )
        for target in ("table9", "uncheatable"):
            target_label = TARGET_LABELS[target]
            table9_effect = getattr(bucket, f"{target}_table9_anchor_later_minus_earlier_bpb")
            uncheatable_effect = getattr(bucket, f"{target}_uncheatable_anchor_later_minus_earlier_bpb")
            mean_effect = getattr(bucket, f"{target}_cross_anchor_mean_later_minus_earlier_bpb")
            table9_tv = getattr(bucket, f"{target}_table9_anchor_phase_tv")
            uncheatable_tv = getattr(bucket, f"{target}_uncheatable_anchor_phase_tv")
            mean_slope = getattr(bucket, f"{target}_cross_anchor_mean_later_minus_earlier_bpb_per_phase_tv")
            conclusion = getattr(bucket, f"{target}_qualitative_conclusion")
            net_gain = getattr(bucket, f"{target}_matched_anchor_net_gain_bpb")
            opposite_minus_tied = getattr(bucket, f"{target}_matched_anchor_opposite_minus_tied_bpb")
            lines.append(
                f"- {target_label}: **{conclusion}**. Later-minus-earlier is {table9_effect:+.6f} BPB at the "
                f"Table-9 anchor (phase TV {table9_tv:.4f}) and {uncheatable_effect:+.6f} at the Uncheatable "
                f"anchor (phase TV {uncheatable_tv:.4f}); raw mean {mean_effect:+.6f}, mean normalized slope "
                f"{mean_slope:+.4f} BPB per phase TV. Matched-anchor preferred-sign gain versus tied is "
                f"{net_gain:+.6f} BPB and the reversed sign is {opposite_minus_tied:+.6f} BPB versus tied."
            )
        if bucket.cross_target_pattern == "objective_specific_reversal":
            lines.append(
                "- Cross-target reading: objective-specific reversal; one shared phase-order sign is contradicted."
            )
        elif bucket.cross_target_pattern == "same_direction_both_targets":
            lines.append(
                "- Cross-target reading: the same descriptive direction holds across both targets and both anchors."
            )
        else:
            lines.append("- Cross-target reading: at least one target changes sign with the aggregate anchor.")

        counterexamples = rule_results[
            rule_results["domain"].eq(bucket.domain) & rule_results["outcome"].eq("counterexample")
        ]
        if counterexamples.empty:
            lines.append("- Universal-rule counterexamples: none under the seven preregistered qualitative rules.")
        else:
            tags = "; ".join(
                f"{TARGET_LABELS[row.target]} `{row.rule_id}` ({row.counterexample_kind})"
                for row in counterexamples.itertuples(index=False)
            )
            lines.append(f"- Universal-rule counterexamples: {tags}.")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_rule_counterexample_catalog(
    output_path: Path,
    rules: tuple[PhaseOrderRule, ...],
    rule_results: pd.DataFrame,
    quality_contrasts: pd.DataFrame,
) -> None:
    summary = rule_summary_table(rule_results, include_label=True)
    lines = [
        "# Explicit counterexamples to universal phase-order rules",
        "",
        "A counterexample requires the direction opposite the stated rule at both aggregate anchors. "
        "`Direction-specific` means only the preferred opposite ordering beats the same-seed tied control at the "
        "target-matched anchor. `Both-orders-beat-tied` means the opposite direction wins the pair but both signs "
        "beat tied, so order alone does not explain the gain. `Direction-only` means the opposite ordering is "
        "consistently less harmful but tied remains best. Anchor-dependent buckets also invalidate universality "
        "but are listed separately.",
        "",
        "## Rule-level summary",
        "",
        summary[
            [
                "rule_id",
                "rule_label",
                "target",
                "supports",
                "direction_specific_supports",
                "counterexamples",
                "direction_specific_counterexamples",
                "anchor_dependent",
            ]
        ].to_markdown(index=False),
        "",
    ]
    for rule in rules:
        lines.extend(
            [
                f"## `{rule.rule_id}`: {rule.label}",
                "",
                f"Premise: {rule.premise}",
                "",
                f"Expected direction: **{rule.expected_direction}**.",
                "",
            ]
        )
        counterexamples = rule_results[
            rule_results["rule_id"].eq(rule.rule_id) & rule_results["outcome"].eq("counterexample")
        ].copy()
        if counterexamples.empty:
            lines.append("No stable opposite-sign bucket was found; this rule is challenged only by anchor dependence.")
            lines.append("")
            continue
        counterexamples["absolute_cross_anchor_effect_bpb"] = counterexamples[
            "cross_anchor_mean_later_minus_earlier_bpb"
        ].abs()
        counterexamples = counterexamples.sort_values(
            ["target", "counterexample_kind", "absolute_cross_anchor_effect_bpb"],
            ascending=[True, True, False],
        )
        lines.extend(
            [
                counterexamples[
                    [
                        "target",
                        "domain",
                        "counterexample_kind",
                        "table9_anchor_later_minus_earlier_bpb",
                        "uncheatable_anchor_later_minus_earlier_bpb",
                        "matched_anchor_net_gain_bpb",
                        "comment",
                    ]
                ].to_markdown(index=False, floatfmt=".6f"),
                "",
            ]
        )

    quality_summary = (
        quality_contrasts.groupby("target")["supports_high_quality_later_relative_to_low"]
        .agg(["sum", "count"])
        .reset_index()
    )
    quality_raw_summary = (
        quality_contrasts.groupby("target")["supports_high_quality_later_relative_to_low_raw"]
        .agg(["sum", "count"])
        .reset_index()
    )
    quality_cross_target = quality_contrasts.pivot(
        index="topic",
        columns="target",
        values="supports_high_quality_later_relative_to_low",
    )
    quality_cross_target_summary = pd.DataFrame(
        [
            {
                "both_targets_support_high_later": int(quality_cross_target.all(axis=1).sum()),
                "both_targets_counter_high_later": int((~quality_cross_target).all(axis=1).sum()),
                "objective_specific": int((quality_cross_target["table9"] != quality_cross_target["uncheatable"]).sum()),
                "topic_count": len(quality_cross_target),
            }
        ]
    )
    quality_counterexamples = quality_contrasts[
        ~quality_contrasts["supports_high_quality_later_relative_to_low"]
    ].sort_values(["target", "high_minus_low_order_slope_bpb_per_phase_tv"], ascending=[True, False])
    lines.extend(
        [
            "## Paired Common Crawl quality-tier test",
            "",
            "The 13 Dolma Common Crawl topics provide a derived high-versus-low quality comparison from separate "
            "domain-versus-rest fibers. Because each bucket has a different feasible contrast radius, this table "
            "compares BPB effect per unit phase TV rather than raw pair gaps. A negative "
            "`high_minus_low_order_slope_bpb_per_phase_tv` means the high-quality tier has more relative reason "
            "to appear late. This is not a direct high-versus-low randomized contrast.",
            "",
            quality_summary.to_markdown(index=False),
            "",
            quality_cross_target_summary.to_markdown(index=False),
            "",
            "The exact topic count is sensitive to the estimand. Using unnormalized raw BPB effects instead gives:",
            "",
            quality_raw_summary.to_markdown(index=False),
            "",
            "This sensitivity changes individual topic labels, but not the conclusion that higher-quality-late is "
            "not universal.",
            "",
            "**Topics contradicting higher-quality-later relative to their low-quality counterpart**",
            "",
            quality_counterexamples.to_markdown(index=False, floatfmt=".6f"),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_pair_notes(
    output_path: Path,
    physical_pairs: pd.DataFrame,
    target_pairs: pd.DataFrame,
) -> None:
    targets_by_pair = {pair_id: group.set_index("target") for pair_id, group in target_pairs.groupby("pair_id")}
    lines = [
        "# Running qualitative audit: fixed-aggregate phase-order pairs",
        "",
        "This ledger covers all 192 physical antithetic pairs. In every entry, the two policies have the same "
        f"realized {100 * PHASE_FRACTIONS[0]:.4f}/{100 * PHASE_FRACTIONS[1]:.4f} aggregate mixture "
        "(nominal 80/20) and the same seed block; only phase order is reversed. `Later` and `earlier` lists "
        "report phase-1 minus phase-0 weight. Effect-size labels are descriptive, not significance tests: "
        "trace <0.002, small 0.002-0.005, moderate 0.005-0.010, and large >=0.010 BPB.",
        "",
    ]
    for audit_index, pair in enumerate(physical_pairs.itertuples(index=False), start=1):
        targets = targets_by_pair[pair.pair_id]
        uncheatable = targets.loc["uncheatable"]
        table9 = targets.loc["table9"]
        lines.extend(
            [
                f"## {audit_index:03d}. `{pair.pair_id}`",
                "",
                f"- Design: `{pair.panel}` / `{pair.anchor_id}` / `{pair.contrast_family}` / "
                f"`{pair.direction_id}`; phase TV {pair.phase_tv:.4f}; data seed {pair.data_seed}.",
                f"- Plus policy `{pair.plus_candidate_id}`: later `{pair.plus_later_top}`; earlier "
                f"`{pair.plus_earlier_top}`.",
                f"- Minus policy `{pair.minus_candidate_id}` reverses that ordering exactly; aggregate mismatch "
                f"{pair.aggregate_max_abs_difference:.2e}.",
                f"- Uncheatable: plus {uncheatable.plus_bpb:.6f}, minus {uncheatable.minus_bpb:.6f}; "
                f"{uncheatable.better_sign} is better by {uncheatable.pair_gap_bpb:.6f} "
                f"({uncheatable.gap_band}); best sign vs tied {uncheatable.better_minus_control_bpb:+.6f}; "
                f"odd order effect {uncheatable.odd_order_effect_bpb:+.6f}, even asymmetry cost "
                f"{uncheatable.even_asymmetry_cost_bpb:+.6f}.",
                f"- Table-9: plus {table9.plus_bpb:.6f}, minus {table9.minus_bpb:.6f}; "
                f"{table9.better_sign} is better by {table9.pair_gap_bpb:.6f} "
                f"({table9.gap_band}); best sign vs tied {table9.better_minus_control_bpb:+.6f}; "
                f"odd order effect {table9.odd_order_effect_bpb:+.6f}, even asymmetry cost "
                f"{table9.even_asymmetry_cost_bpb:+.6f}.",
                f"- Better ordering agrees across objectives: "
                f"{'yes' if pair.same_better_sign_both_targets else 'no'}.",
                f"- Uncheatable winner moves later: `{uncheatable.better_later_top}`; earlier: "
                f"`{uncheatable.better_earlier_top}`; largest phase-1 repetition changes: "
                f"`{uncheatable.better_vs_worse_late_epoch_delta_top}`.",
                f"- Table-9 winner moves later: `{table9.better_later_top}`; earlier: "
                f"`{table9.better_earlier_top}`; largest phase-1 repetition changes: "
                f"`{table9.better_vs_worse_late_epoch_delta_top}`.",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(output_dir: Path, domain_summary: pd.DataFrame, target_pairs: pd.DataFrame) -> None:
    domain_plot = domain_summary.copy()
    domain_plot["facet"] = domain_plot["anchor_id"].str.replace("_frontier", "", regex=False).str.title()
    domain_plot["target_label"] = domain_plot["target"].map(TARGET_LABELS)
    index = domain_plot["domain"].drop_duplicates()
    normalized_pivot = domain_plot.pivot_table(
        index="domain",
        columns=["facet", "target_label"],
        values="domain_later_minus_earlier_bpb_per_phase_tv",
        aggfunc="first",
    ).reindex(index=index)
    raw_pivot = domain_plot.pivot_table(
        index="domain",
        columns=["facet", "target_label"],
        values="domain_later_minus_earlier_bpb",
        aggfunc="first",
    ).reindex(index=index, columns=normalized_pivot.columns)
    phase_tv_pivot = domain_plot.pivot_table(
        index="domain",
        columns=["facet", "target_label"],
        values="phase_tv",
        aggfunc="first",
    ).reindex(index=index, columns=normalized_pivot.columns)
    customdata = np.dstack(
        [
            raw_pivot.to_numpy(float),
            phase_tv_pivot.to_numpy(float),
        ]
    )
    heatmap = go.Figure(
        go.Heatmap(
            z=normalized_pivot.to_numpy(float),
            x=[" / ".join(column) for column in normalized_pivot.columns],
            y=normalized_pivot.index,
            colorscale="RdYlGn_r",
            zmid=0.0,
            colorbar={"title": "domain-late minus<br>domain-early<br>BPB / phase TV"},
            customdata=customdata,
            hovertemplate=(
                "%{y}<br>%{x}<br>normalized slope=%{z:+.6f} BPB / phase TV"
                "<br>raw delta=%{customdata[0]:+.6f} BPB"
                "<br>phase TV=%{customdata[1]:.4f}<extra></extra>"
            ),
        )
    )
    heatmap.update_layout(
        title={
            "text": (
                "Fixed-aggregate one-domain phase reversals"
                "<br><sup>Negative (green) means placing the named bucket later won; "
                "effects are normalized by each fiber's feasible phase TV.</sup>"
            )
        },
        width=1200,
        height=1120,
        margin={"l": 300, "r": 80, "t": 100, "b": 150},
    )
    heatmap.write_html(output_dir / "domain_phase_order_heatmap.html", include_plotlyjs=True, config=PLOT_CONFIG)

    raw_heatmap = go.Figure(
        go.Heatmap(
            z=raw_pivot.to_numpy(float),
            x=[" / ".join(column) for column in raw_pivot.columns],
            y=raw_pivot.index,
            colorscale="RdYlGn_r",
            zmid=0.0,
            colorbar={"title": "domain-late minus<br>domain-early<br>raw BPB"},
            customdata=phase_tv_pivot.to_numpy(float),
            hovertemplate=("%{y}<br>%{x}<br>raw delta=%{z:+.6f} BPB" "<br>phase TV=%{customdata:.4f}<extra></extra>"),
        )
    )
    raw_heatmap.update_layout(
        title={
            "text": (
                "Fixed-aggregate one-domain phase reversals: raw tested interventions"
                "<br><sup>Do not compare magnitudes across buckets without accounting for phase TV.</sup>"
            )
        },
        width=1200,
        height=1120,
        margin={"l": 300, "r": 80, "t": 100, "b": 150},
    )
    raw_heatmap.write_html(
        output_dir / "domain_phase_order_raw_heatmap.html",
        include_plotlyjs=True,
        config=PLOT_CONFIG,
    )

    balanced = target_pairs[target_pairs["contrast_family"].eq("balanced_partition")].copy()
    balanced["plus_minus_bpb"] = balanced["plus_bpb"] - balanced["minus_bpb"]
    balanced["target_label"] = balanced["target"].map(TARGET_LABELS)
    balanced["anchor_label"] = balanced["anchor_id"].str.replace("_frontier", "", regex=False).str.title()
    scatter = px.scatter(
        balanced,
        x="target_phase_tv",
        y="plus_minus_bpb",
        color="target_label",
        facet_col="anchor_label",
        hover_data=["direction_id", "pair_gap_bpb", "better_minus_control_bpb"],
        color_discrete_map={"Uncheatable": "#4575B4", "Table-9": "#D73027"},
        title="Balanced antithetic phase-order effects across asymmetry radii",
    )
    scatter.add_hline(y=0.0, line_dash="dash", line_color="#334155")
    scatter.update_yaxes(title_text="plus-order BPB - reverse-order BPB")
    scatter.update_xaxes(title_text="phase total variation")
    scatter.update_layout(width=1300, height=650)
    scatter.write_html(output_dir / "balanced_antithetic_effects.html", include_plotlyjs=True, config=PLOT_CONFIG)


def write_report(
    output_dir: Path,
    semantics: pd.DataFrame,
    physical_pairs: pd.DataFrame,
    target_pairs: pd.DataFrame,
    summary: pd.DataFrame,
    odd_even: pd.DataFrame,
    control_noise: pd.DataFrame,
    domain_summary: pd.DataFrame,
    domain_hypotheses: pd.DataFrame,
    dossier: pd.DataFrame,
    rule_results: pd.DataFrame,
    quality_contrasts: pd.DataFrame,
    transfer_summary: pd.DataFrame,
    handcrafted: pd.DataFrame,
    continuum: pd.DataFrame,
    consistency: pd.DataFrame,
) -> None:
    matched = target_pairs[target_pairs["is_anchor_matched_target"]]
    matched_domain = domain_summary[domain_summary["is_anchor_matched_target"]]
    matched_consistency = consistency[consistency["is_anchor_matched_target"]]
    agreement = float(physical_pairs["same_better_sign_both_targets"].mean())
    stable_directions = float(matched_consistency["same_preferred_sign_all_radii"].mean())
    domain_later = (
        matched_domain.groupby(["target", "pool"])["domain_later_better"].agg(["sum", "count", "mean"]).reset_index()
    )
    strongest = matched.nlargest(12, "pair_gap_bpb")[
        [
            "pair_id",
            "target",
            "pair_gap_bpb",
            "better_sign",
            "better_minus_control_bpb",
            "better_later_top",
            "better_earlier_top",
        ]
    ]
    stable_domains = domain_hypotheses[domain_hypotheses["same_sign_across_anchors"]].copy()
    stable_late = (
        stable_domains[stable_domains["mean_domain_later_minus_earlier_bpb"].lt(0.0)]
        .sort_values(
            ["target", "mean_domain_later_minus_earlier_bpb_per_phase_tv"],
            ascending=[True, True],
        )
        .groupby("target", sort=True)
        .head(5)
    )
    stable_early = (
        stable_domains[stable_domains["mean_domain_later_minus_earlier_bpb"].gt(0.0)]
        .sort_values(
            ["target", "mean_domain_later_minus_earlier_bpb_per_phase_tv"],
            ascending=[True, False],
        )
        .groupby("target", sort=True)
        .head(5)
    )
    helpful = helpful_bucket_rows(dossier).sort_values(
        ["target", "preferred_phase", "matched_anchor_net_gain_bpb"],
        ascending=[True, True, False],
    )
    control_sd_by_target = {
        target: float(
            control_noise[
                control_noise["panel"].eq("frontier_phase_fiber")
                & control_noise["anchor_id"].eq(TARGET_ANCHOR[target])
                & control_noise["target"].eq(target)
            ]["control_sd_bpb"].item()
        )
        for target in ("table9", "uncheatable")
    }
    helpful["matched_gain_over_cross_seed_control_sd"] = helpful.apply(
        lambda row: float(row["matched_anchor_net_gain_bpb"]) / control_sd_by_target[str(row["target"])],
        axis=1,
    )
    rule_summary = rule_summary_table(rule_results, include_label=False)
    quality_summary = (
        quality_contrasts.groupby("target")["supports_high_quality_later_relative_to_low"]
        .agg(["sum", "count"])
        .reset_index()
    )
    quality_raw_summary = (
        quality_contrasts.groupby("target")["supports_high_quality_later_relative_to_low_raw"]
        .agg(["sum", "count"])
        .reset_index()
    )
    quality_cross_target = quality_contrasts.pivot(
        index="topic",
        columns="target",
        values="supports_high_quality_later_relative_to_low",
    )
    quality_cross_target_summary = pd.DataFrame(
        [
            {
                "both_targets_support_high_later": int(quality_cross_target.all(axis=1).sum()),
                "both_targets_counter_high_later": int((~quality_cross_target).all(axis=1).sum()),
                "objective_specific": int((quality_cross_target["table9"] != quality_cross_target["uncheatable"]).sum()),
                "topic_count": len(quality_cross_target),
            }
        ]
    )
    synth_qa = dossier.set_index("domain").loc["dolmino_synth_qa"]
    synth_qa_table9_gain = float(synth_qa["table9_matched_anchor_net_gain_bpb"])
    synth_qa_table9_reversal_cost = float(synth_qa["table9_matched_anchor_opposite_minus_tied_bpb"])
    useful_gap_counts = {
        target: {
            threshold: int(
                (
                    matched["target"].eq(target)
                    & matched["pair_gap_bpb"].ge(threshold)
                    & matched["better_minus_control_bpb"].lt(0)
                ).sum()
            )
            for threshold in (0.005, 0.010)
        }
        for target in ("table9", "uncheatable")
    }
    lines = [
        "# Fixed-aggregate phase-order qualitative audit",
        "",
        "## Scope",
        "",
        f"- Audited {len(physical_pairs)} physical antithetic policy pairs ({len(target_pairs)} target-specific "
        "comparisons), exceeding the requested 100-pair minimum.",
        f"- Every pair preserves the exact realized {100 * PHASE_FRACTIONS[0]:.4f}/"
        f"{100 * PHASE_FRACTIONS[1]:.4f} aggregate mixture (nominal 80/20), uses a common seed block, and "
        "reverses only the phase contrast.",
        "- The audit combines 96 local frontier-fiber pairs and 96 balanced-partition pairs spanning phase TV "
        "0.10, 0.25, and 0.50.",
        "- Headline outcome counts use the 192 target-matched comparisons: Table-9 at the Table-9 anchor and "
        "Uncheatable at the Uncheatable anchor. The other 192 cross-target observations are retained for transfer "
        "analysis rather than pooled into the primary result.",
        "- The pair ledger is descriptive evidence. It does not fit or select a phase-order model.",
        "",
        "## Bucket frame",
        "",
        f"The policy has {len(semantics)} buckets: 30 Dolma 3 broad/pretraining buckets and 9 Dolmino "
        "quality-filtered or synthetic annealing buckets. Available corpus sizes range from "
        f"{semantics['available_tokens_b'].min():.2f}B to {semantics['available_tokens_b'].max():.2f}B tokens. "
        "Consequently, equal weight movement can imply radically different repetition changes; the pair ledger "
        "reports simulated-epoch changes as well as weights.",
        "",
        "## Direct descriptive findings",
        "",
        f"- The same phase ordering wins on both objectives in {agreement:.1%} of physical pairs. Values near "
        "50% indicate that a universal objective-independent notion of `good data late` is insufficient.",
        f"- Among balanced target-matched directions, only {stable_directions:.1%} retain the same preferred sign "
        "at phase TV 0.10, 0.25, and 0.50. Direction effects therefore are often nonlinear, noise-scale, or both.",
        f"- In target-matched comparisons, {int(matched['pair_gap_bpb'].ge(0.005).sum())}/{len(matched)} pairs "
        "separate their two orderings by at least 0.005 BPB and "
        f"{int(matched['pair_gap_bpb'].ge(0.010).sum())}/{len(matched)} by at least 0.010 BPB.",
        f"- The better sign beats its tied control in {int(matched['better_beats_control'].sum())}/{len(matched)} "
        "target-matched pairs. This is an oracle-after-observation count, not an achievable selection rate.",
        f"- Both signs are worse than tied in {int(matched['both_worse_than_control'].sum())}/{len(matched)} "
        "target-matched pairs, directly showing an even asymmetry cost that an odd ordering-only model cannot absorb.",
        f"- Table-9 has {useful_gap_counts['table9'][0.005]} pairs with both a >=0.005 BPB ordering gap and "
        f"a better sign that beats tied; {useful_gap_counts['table9'][0.010]} remain at the >=0.010 threshold. "
        f"Uncheatable has {useful_gap_counts['uncheatable'][0.005]} and "
        f"{useful_gap_counts['uncheatable'][0.010]}, respectively. Thus, the strongest useful order signal is "
        "specific to Table-9 rather than a general property of the two-phase policy class.",
        "",
        "## Odd order effect versus even asymmetry cost",
        "",
        "For each antithetic pair, the directional order effect is "
        "$O(d)=[Y(+d)-Y(-d)]/2$ and the order-independent asymmetry cost is "
        "$C(d)=[Y(+d)+Y(-d)]/2-Y(0)$. A positive $C(d)$ means asymmetry is harmful on average; "
        "$C(d)>|O(d)|$ means neither ordering beats the tied control.",
        "",
        odd_even.to_markdown(index=False, floatfmt=".6f"),
        "",
        "The control spread below is the standard deviation across fresh tied-control runs, not a standard error. "
        "Many Uncheatable local-fiber order effects are smaller than this scale; Table-9 has larger directional "
        "effects but also noisier controls. Each antithetic plus/minus pair shares a data seed, whereas these "
        "controls vary the seed. The cross-seed control spread therefore does not identify the within-pair noise "
        "floor, and the audit cannot classify individual local-fiber effects as statistically resolved.",
        "",
        control_noise.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Domain-later counts at the target-matched anchor",
        "",
        domain_later.to_markdown(index=False, floatfmt=".3f"),
        "",
        "These counts use only each objective's matched frontier anchor. They are heterogeneous within each pool "
        "and objective and do not support treating all Dolmino buckets, all high-quality buckets, or all scarce "
        "buckets as sharing one late-training coefficient.",
        "",
        "### Cross-anchor domain hypotheses",
        "",
        "Negative values favor moving the named domain later; positive values favor moving it earlier. Unlike the "
        "preceding matched-anchor count, this table averages the effect over both aggregate anchors and retains a "
        "descriptive hypothesis only when both anchors agree on the sign. Domain fibers have different feasible "
        "phase-TV radii, so the tables are ranked by the normalized BPB-per-phase-TV column rather than raw gaps.",
        "",
        "**Largest consistent domain-later effects**",
        "",
        stable_late[
            [
                "target",
                "domain",
                "pool",
                "mean_domain_later_minus_earlier_bpb",
                "mean_domain_later_minus_earlier_bpb_per_phase_tv",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "**Largest consistent domain-earlier effects**",
        "",
        stable_early[
            [
                "target",
                "domain",
                "pool",
                "mean_domain_later_minus_earlier_bpb",
                "mean_domain_later_minus_earlier_bpb_per_phase_tv",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Bucket-specific descriptive recommendations",
        "",
        "The table below lists only direction-specific cases: the preferred early/late direction agrees across "
        "both aggregate anchors, beats the same-seed tied control at the target-matched anchor, and its reversal "
        "does not. Cases where both asymmetric schedules beat tied remain in the full dossier but are not credited "
        "to phase order. The last column divides the gain by the cross-seed tied-control SD only for scale context; "
        "it is not a z-score because each plus/minus pair shares a seed. These are qualitative hypotheses, not "
        "significance claims, and no multiple-testing correction has been applied.",
        "",
        helpful[
            [
                "target",
                "preferred_phase",
                "domain",
                "pool",
                "cross_anchor_mean_later_minus_earlier_bpb",
                "cross_anchor_mean_later_minus_earlier_bpb_per_phase_tv",
                "matched_anchor_net_gain_bpb",
                "matched_gain_over_cross_seed_control_sd",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        f"The lone Table-9 direction-specific case, `dolmino_synth_qa`, improves tied by only "
        f"{synth_qa_table9_gain:.6f} BPB when placed later, while placing it earlier is "
        f"{synth_qa_table9_reversal_cost:.6f} BPB worse than tied. Its large antithetic gap is therefore mainly an "
        "avoided early-placement penalty, not evidence of a comparably large late-placement gain.",
        "",
        "## Universal-rule falsification summary",
        "",
        "A counterexample has the direction opposite the rule at both aggregate anchors. Anchor-dependent buckets "
        "also invalidate universality but are counted separately. See `phase_order_rule_counterexamples.md` for "
        "the exact commented examples and whether the opposite direction actually beats tied.",
        "",
        rule_summary[
            [
                "rule_id",
                "target",
                "supports",
                "direction_specific_supports",
                "counterexamples",
                "direction_specific_counterexamples",
                "anchor_dependent",
            ]
        ].to_markdown(index=False),
        "",
        "For the paired Dolma Common Crawl tiers, higher-quality data has a stronger relative reason to appear "
        "late in the following number of 13 topics. This derived comparison normalizes each separate "
        "domain-versus-rest effect by its phase-TV radius:",
        "",
        quality_summary.to_markdown(index=False),
        "",
        quality_cross_target_summary.to_markdown(index=False),
        "",
        "Using unnormalized raw BPB effects instead gives:",
        "",
        quality_raw_summary.to_markdown(index=False),
        "",
        "The exact topic classifications are normalization-sensitive, but neither estimand supports a universal "
        "higher-quality-late rule.",
        "",
        "## Transfer of ordering effects",
        "",
        transfer_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Cross-anchor consistency is materially stronger than cross-objective consistency for several slices, but "
        "neither is strong enough to justify a universal phase-order coefficient. The target should remain explicit.",
        "",
        "## Conventional schedule probes",
        "",
        "These fixed-aggregate schedules directly test the literature-inspired heuristic that curated or target-like "
        "data should be concentrated late while some broad data is replayed. Negative deltas improve on the "
        "same-seed tied control.",
        "",
        handcrafted.sort_values(["target", "mean_delta_bpb"]).to_markdown(index=False, floatfmt=".6f"),
        "",
        "Only the Table-9 `premium_nonweb` and `math_reasoning` recipes improve at all three tested radii. No "
        "handcrafted recipe does so for Uncheatable. The all-Dolmino heuristic is not reliably beneficial.",
        "",
        continuum.sort_values(["target", "late_dolmino_share"]).to_markdown(index=False, floatfmt=".6f"),
        "",
        "The replicated continuum rejects aggressive Dolmino concentration: 90% and 100% Dolmino in the final "
        "phase are worse for both target-matched objectives, while the 75% Table-9 result is inconsistent across "
        "three seeds.",
        "",
        "## Largest target-matched pair separations",
        "",
        strongest.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation against the literature",
        "",
        "- *Replaying pre-training data improves fine-tuning* finds that retaining generic replay during late "
        "specialization can improve even the target task, with more benefit when less target data was seen early. "
        "The mixed signs here are therefore compatible with replay being content- and objective-dependent rather "
        "than a single broad-replay scalar.",
        "- *The Finetuner's Fallacy* finds that scarce target data can be more useful when introduced during "
        "pretraining, reducing overfit across repetitions and later forgetting. Our early-versus-late reversals for "
        "scarce buckets likewise make exposure history and repetition plausible state variables.",
        "- Neither paper implies that every curated or target-like bucket should be concentrated in the final "
        "phase. The failed all-Dolmino schedules and opposite signs among Dolmino buckets are not contradictions.",
        "- Large even costs at high phase TV agree with the idea that abrupt specialization can damage broad "
        "capability. The useful phase-order signal, where present, is local and directional.",
        "",
        "## Evidence tiers",
        "",
        "**Robust in the audited panels**",
        "",
        "- Phase TV 0.50 is dominated by even asymmetry harm: all 32 target-matched balanced pairs have both "
        "orderings worse than tied.",
        "- Concentrating 90% or 100% of phase 1 on Dolmino is worse than tied across all target-matched " "replicates.",
        "- Ordering effects are target-specific; cross-target sign agreement is only slightly above chance.",
        "",
        "**Suggestive and worth modeling**",
        "",
        "- Table-9 has useful moderate-asymmetry directions, especially the `premium_nonweb` and "
        "`math_reasoning` recipes. Their roughly 0.003 BPB gains need fresh confirmation because they are "
        "comparable to tied-control variation. Recipe-level gains do not establish that every constituent bucket "
        "shares a late-training coefficient.",
        "- Some domain-order signs recur across aggregate anchors, but their magnitude and stability vary sharply "
        "by target.",
        "",
        "**Not supported by these data**",
        "",
        "- A universal Dolmino-late, high-quality-late, or scarce-data-late coefficient.",
        "- A shared phase-order rule across Uncheatable and Table-9.",
        "- A general 0.01 BPB order advantage on Uncheatable at these anchors and this compute scale.",
        "",
        "## Independent review reconciliation",
        "",
        "An independent Claude review reproduced every frontier-fiber odd/even, control, and transfer statistic "
        "from `observed_results.csv`. It retracted an initial claim that the frontier outcomes were absent after "
        "the dedicated result artifact was supplied. The review identified three interpretation constraints now "
        "made explicit above: primary counts are target-matched rather than pooled, matched-anchor domain counts "
        "and cross-anchor hypotheses use different estimands, and cross-seed controls do not measure common-seed "
        "antithetic noise. The review independently agreed that aggressive asymmetry harm is the strongest "
        "transferable result and that content-specific order effects are weak, target-conditioned, and "
        "anchor-conditioned.",
        "",
        "## Modeling implications",
        "",
        "1. Decompose phase effects into an odd ordering term and an even asymmetry cost. Antithetic pairs identify "
        "these separately.",
        "2. Price phase movements in both token share and per-bucket repetition units. Small corpora can experience "
        "large epoch shocks under modest weight changes.",
        "3. Do not impose a universal `quality late` sign. Pool-level or family-level sharing must allow exceptions, "
        "especially across objectives.",
        "4. Train any phase-order component on fixed-aggregate contrasts and keep aggregate-response fitting separate.",
        "5. Treat high-TV balanced partitions primarily as curvature and harm evidence; the local frontier fiber is "
        "more relevant to estimating a first-order ordering gradient.",
        "",
        "## Artifacts",
        "",
        "- `running_pair_notes.md`: one structured qualitative entry for each of 192 physical pairs.",
        "- `physical_pair_ledger.csv`: pair identities, anatomy, and cross-target sign agreement.",
        "- `target_pair_ledger.csv`: target-specific outcomes and better/worse schedule anatomy.",
        "- `odd_even_summary.csv`: directional order signal and order-independent asymmetry cost by design radius.",
        "- `control_noise_summary.csv`: tied-control variation for contextualizing small pair differences.",
        "- `bucket_semantics.csv`: exact 39-bucket taxonomy and corpus sizes.",
        "- `domain_vs_rest_summary.csv`: named-domain late-versus-early reversals.",
        "- `domain_order_hypotheses.csv`: cross-anchor domain-order consistency.",
        "- `bucket_phase_order_dossier.csv` and `bucket_phase_order_dossier.md`: all 39 bucket-specific effects, "
        "net gains, and qualitative comments.",
        "- `bucket_phase_order_map.csv`: compact 39-row early/late lookup across both objectives.",
        "- `phase_order_rule_tests.csv` and `phase_order_rule_counterexamples.md`: explicit tests and commented "
        "counterexamples for seven common phase-order priors.",
        "- `quality_tier_contrasts.csv`: paired high-versus-low Common Crawl topic tests.",
        "- `effect_transfer_summary.csv`: cross-anchor and cross-target effect transfer.",
        "- `handcrafted_schedule_summary.csv`: conventional late-quality recipes by target and radius.",
        "- `dolmino_late_continuum_summary.csv`: replicated aggressive Dolmino-late schedules.",
        "- `balanced_direction_consistency.csv`: sign stability across aggressive radii.",
        "- `domain_phase_order_heatmap.html`: phase-TV-normalized bucket timing effects; "
        "`domain_phase_order_raw_heatmap.html`: raw intervention effects.",
        "- `balanced_antithetic_effects.html`: interactive direction and asymmetry-radius diagnostic.",
        "- Source outcomes: `delphi_3e18_frontier_phase_fiber_results_20260719/observed_results.csv` and "
        "`delphi_3e18_aggressive_phase_asymmetry_results_20260723/observed_results_with_control_deltas.csv`.",
        "",
        "## Primary references",
        "",
        "- Dolma 3 pool: https://huggingface.co/datasets/allenai/dolma3_pool",
        "- Dolma 3 Dolmino pool: https://huggingface.co/datasets/allenai/dolma3_dolmino_pool",
        "- Replaying pre-training data improves fine-tuning: https://arxiv.org/abs/2603.04964v1",
        "- The Finetuner's Fallacy: https://arxiv.org/abs/2603.16177",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    semantics = bucket_semantics()
    panels = load_panel_inputs()
    physical_pairs, target_pairs = build_pair_ledgers(panels, semantics)
    summary = target_summary(target_pairs)
    odd_even = odd_even_summary(target_pairs)
    control_noise = control_noise_summary(target_pairs)
    domain_summary = domain_vs_rest_summary(target_pairs, semantics)
    domain_hypotheses = domain_order_hypotheses(domain_summary)
    dossier = bucket_phase_order_dossier(domain_summary, semantics)
    phase_order_map = bucket_phase_order_map(dossier)
    rules = phase_order_rules(semantics)
    rule_results = rule_test_results(dossier, rules)
    quality_contrasts = quality_tier_contrasts(dossier)
    transfer_summary = effect_transfer_summary(target_pairs)
    handcrafted, continuum = structured_schedule_summaries()
    consistency = balanced_direction_consistency(target_pairs)

    semantics.to_csv(args.output_dir / "bucket_semantics.csv", index=False)
    physical_pairs.to_csv(args.output_dir / "physical_pair_ledger.csv", index=False)
    target_pairs.to_csv(args.output_dir / "target_pair_ledger.csv", index=False)
    summary.to_csv(args.output_dir / "target_summary.csv", index=False)
    odd_even.to_csv(args.output_dir / "odd_even_summary.csv", index=False)
    control_noise.to_csv(args.output_dir / "control_noise_summary.csv", index=False)
    domain_summary.to_csv(args.output_dir / "domain_vs_rest_summary.csv", index=False)
    domain_hypotheses.to_csv(args.output_dir / "domain_order_hypotheses.csv", index=False)
    dossier.to_csv(args.output_dir / "bucket_phase_order_dossier.csv", index=False)
    phase_order_map.to_csv(args.output_dir / "bucket_phase_order_map.csv", index=False)
    rule_results.to_csv(args.output_dir / "phase_order_rule_tests.csv", index=False)
    quality_contrasts.to_csv(args.output_dir / "quality_tier_contrasts.csv", index=False)
    transfer_summary.to_csv(args.output_dir / "effect_transfer_summary.csv", index=False)
    handcrafted.to_csv(args.output_dir / "handcrafted_schedule_summary.csv", index=False)
    continuum.to_csv(args.output_dir / "dolmino_late_continuum_summary.csv", index=False)
    consistency.to_csv(args.output_dir / "balanced_direction_consistency.csv", index=False)
    write_pair_notes(args.output_dir / "running_pair_notes.md", physical_pairs, target_pairs)
    write_bucket_dossier(args.output_dir / "bucket_phase_order_dossier.md", dossier, rule_results)
    write_rule_counterexample_catalog(
        args.output_dir / "phase_order_rule_counterexamples.md",
        rules,
        rule_results,
        quality_contrasts,
    )
    write_plots(args.output_dir, domain_summary, target_pairs)
    write_report(
        args.output_dir,
        semantics,
        physical_pairs,
        target_pairs,
        summary,
        odd_even,
        control_noise,
        domain_summary,
        domain_hypotheses,
        dossier,
        rule_results,
        quality_contrasts,
        transfer_summary,
        handcrafted,
        continuum,
        consistency,
    )
    quality_cross_target = quality_contrasts.pivot(
        index="topic",
        columns="target",
        values="supports_high_quality_later_relative_to_low",
    )
    summary_payload: dict[str, Any] = {
        "physical_pair_count": len(physical_pairs),
        "target_specific_pair_count": len(target_pairs),
        "panel_counts": physical_pairs["panel"].value_counts().to_dict(),
        "cross_target_better_sign_agreement": float(physical_pairs["same_better_sign_both_targets"].mean()),
        "anchor_matched_pairs": int(target_pairs["is_anchor_matched_target"].sum()),
        "anchor_matched_gap_ge_0p005": int(
            target_pairs.loc[target_pairs["is_anchor_matched_target"], "pair_gap_bpb"].ge(0.005).sum()
        ),
        "anchor_matched_gap_ge_0p010": int(
            target_pairs.loc[target_pairs["is_anchor_matched_target"], "pair_gap_bpb"].ge(0.010).sum()
        ),
        "directionally_helpful_bucket_counts": {
            target: {
                phase: int(dossier[f"{target}_qualitative_conclusion"].eq(f"{phase}-directionally-helpful").sum())
                for phase in ("earlier", "later")
            }
            for target in ("table9", "uncheatable")
        },
        "quality_tier_topics_supporting_high_later": {
            row.target: {
                "supporting_topics": int(row.sum),
                "topic_count": int(row.count),
            }
            for row in (
                quality_contrasts.groupby("target")["supports_high_quality_later_relative_to_low"]
                .agg(["sum", "count"])
                .reset_index()
                .itertuples(index=False)
            )
        },
        "quality_tier_topics_supporting_high_later_raw_bpb": {
            row.target: {
                "supporting_topics": int(row.sum),
                "topic_count": int(row.count),
            }
            for row in (
                quality_contrasts.groupby("target")["supports_high_quality_later_relative_to_low_raw"]
                .agg(["sum", "count"])
                .reset_index()
                .itertuples(index=False)
            )
        },
        "quality_tier_cross_target_overlap": {
            "both_targets_support_high_later": int(quality_cross_target.all(axis=1).sum()),
            "both_targets_counter_high_later": int((~quality_cross_target).all(axis=1).sum()),
            "objective_specific": int((quality_cross_target["table9"] != quality_cross_target["uncheatable"]).sum()),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
