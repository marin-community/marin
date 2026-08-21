# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  # Standalone HTML/CSS/JavaScript template readability.

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scipy",
# ]
# ///
"""Audit best and worst Delphi 3e18 two-phase policies across the full archive."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from experiments.domain_phase_mix.exploratory.two_phase_many.audit_fixed_aggregate_phase_pairs_20260723 import (
    FIBER_PANEL_DIR,
    PHASE_FRACTIONS,
    PhaseOrderRule,
    bucket_semantics,
    phase_order_rules,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_REGISTRY = REFERENCE_OUTPUTS / "delphi_3e18_append_only_heldouts_20260714" / "heldout_current.csv"
CONTROLLED_AUDIT_DIR = REFERENCE_OUTPUTS / "delphi_3e18_fixed_aggregate_qualitative_audit_20260723"
PROPORTIONAL_NOISE_SUMMARY = (
    REFERENCE_OUTPUTS / "delphi_3e18_proportional_noise_floor_20260703" / "noise_floor_summary.json"
)
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_3e18_broad_mixture_tail_audit_20260724"
D3_SOURCE = SCRIPT_DIR / "mixture_fit_debugger" / "node_modules" / "d3" / "dist" / "d3.min.js"

TARGET_COLUMNS = {
    "uncheatable": "uncheatable_bpb",
    "table9": "table9_macro_bpb",
}
TARGET_LABELS = {
    "uncheatable": "Uncheatable eval BPB",
    "table9": "OLMoBaseEval Table-9 macro BPB",
}
TRAINING_TOKENS = 1_576_534_016
SIMULATED_EPOCH_TARGET_BUDGET = 6_325_183_647_689
GLOBAL_TAIL_COUNT = 30
SERIES_MIN_POLICIES = 8
SERIES_TAIL_FRACTION = 0.25
GLOBAL_EVIDENCE_FRACTION = 0.10
GENUINE_TWO_PHASE_TV = 0.01
CONTROLLED_TARGET_ANCHOR = {
    "uncheatable": "uncheatable_frontier",
    "table9": "table9_frontier",
}


@dataclass(frozen=True)
class AuditData:
    policies: pd.DataFrame
    policy_buckets: pd.DataFrame
    semantics: pd.DataFrame
    proportional: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_mapping(value: object) -> dict[str, float]:
    assert isinstance(value, str)
    parsed = json.loads(value)
    assert isinstance(parsed, dict)
    return {str(key): float(weight) for key, weight in parsed.items()}


def unique_join(values: pd.Series, *, limit: int | None = None) -> str:
    unique = sorted({str(value) for value in values.dropna() if str(value)})
    if limit is not None and len(unique) > limit:
        return " | ".join(unique[:limit]) + f" | +{len(unique) - limit} more"
    return " | ".join(unique)


def representative_label(group: pd.DataFrame) -> str:
    candidates = [
        str(value)
        for column in ("candidate_id", "wandb_run_base", "wandb_run_name")
        for value in group[column].dropna()
        if str(value) and str(value) != "nan"
    ]
    if not candidates:
        return str(group.iloc[0]["heldout_id"])
    return min(set(candidates), key=lambda value: (len(value), value))


def load_audit_data(registry_path: Path) -> AuditData:
    observations = pd.read_csv(registry_path)
    observations = observations[
        observations["policy_class"].eq("two_phase")
        & observations["uncheatable_bpb"].notna()
        & observations["table9_macro_bpb"].notna()
    ].copy()
    assert len(observations) > 0

    semantics = bucket_semantics()
    domains = list(semantics["domain"])
    token_counts = semantics["available_tokens"].to_numpy(float)
    proportional = token_counts / token_counts.sum()

    policy_rows: list[dict[str, object]] = []
    bucket_rows: list[dict[str, object]] = []
    coordinate_signatures: dict[str, str] = {}
    for mixture_sha256, group in observations.groupby("mixture_sha256", sort=True):
        first = group.iloc[0]
        phase_0 = parse_mapping(first["phase_0_weights_json"])
        phase_1 = parse_mapping(first["phase_1_weights_json"])
        assert set(phase_0) == set(domains)
        assert set(phase_1) == set(domains)
        w0 = np.asarray([phase_0[domain] for domain in domains], dtype=float)
        w1 = np.asarray([phase_1[domain] for domain in domains], dtype=float)
        assert np.isclose(w0.sum(), 1.0, atol=1e-7)
        assert np.isclose(w1.sum(), 1.0, atol=1e-7)

        for row in group.itertuples(index=False):
            assert np.isclose(float(row.phase_0_fraction), float(first["phase_0_fraction"]), atol=1e-12)
            assert parse_mapping(row.phase_0_weights_json) == phase_0
            assert parse_mapping(row.phase_1_weights_json) == phase_1

        phase_0_fraction = float(first["phase_0_fraction"])
        phase_1_fraction = 1.0 - phase_0_fraction
        coordinate_signature = json.dumps(
            {
                "phase_0_fraction": phase_0_fraction,
                "phase_0": phase_0,
                "phase_1": phase_1,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        existing_sha256 = coordinate_signatures.setdefault(coordinate_signature, str(mixture_sha256))
        assert existing_sha256 == str(
            mixture_sha256
        ), f"Coordinate alias under distinct hashes: {existing_sha256} and {mixture_sha256}"
        aggregate = phase_0_fraction * w0 + phase_1_fraction * w1
        contrast = w1 - w0
        phase_tv = float(0.5 * np.abs(contrast).sum())
        if phase_tv < 1e-6:
            phase_regime = "tied_alias"
        elif phase_tv < GENUINE_TWO_PHASE_TV:
            phase_regime = "near_tied"
        else:
            phase_regime = "genuine_two_phase"
        actual_epochs = TRAINING_TOKENS * aggregate / token_counts
        simulated_epochs = SIMULATED_EPOCH_TARGET_BUDGET * aggregate / token_counts
        training_series_values = sorted(set(group["training_series"].dropna().astype(str)))
        policy_id = str(mixture_sha256)[:12]
        proposal_targets = unique_join(group["proposal_target"])
        candidate_kinds = unique_join(group["candidate_kind"])
        panel_tags = unique_join(group["panel_tag"])
        policy_rows.append(
            {
                "policy_id": policy_id,
                "mixture_sha256": mixture_sha256,
                "label": representative_label(group),
                "training_series": " | ".join(training_series_values),
                "series_count": len(training_series_values),
                "single_series": training_series_values[0] if len(training_series_values) == 1 else "",
                "panel_tags": panel_tags,
                "proposal_targets": proposal_targets,
                "candidate_kinds": candidate_kinds,
                "heldout_ids": unique_join(group["heldout_id"], limit=3),
                "wandb_run_names": unique_join(group["wandb_run_name"], limit=3),
                "observation_count": len(group),
                "uncheatable_bpb": float(group["uncheatable_bpb"].mean()),
                "uncheatable_sd": float(group["uncheatable_bpb"].std(ddof=1)) if len(group) > 1 else np.nan,
                "table9_macro_bpb": float(group["table9_macro_bpb"].mean()),
                "table9_sd": float(group["table9_macro_bpb"].std(ddof=1)) if len(group) > 1 else np.nan,
                "phase_0_fraction": phase_0_fraction,
                "phase_1_fraction": phase_1_fraction,
                "phase_tv": phase_tv,
                "phase_regime": phase_regime,
                "aggregate_tv_to_proportional": float(0.5 * np.abs(aggregate - proportional).sum()),
                "max_phase_weight": float(max(w0.max(), w1.max())),
                "max_aggregate_weight": float(aggregate.max()),
                "max_actual_epoch": float(actual_epochs.max()),
                "max_simulated_epoch": float(simulated_epochs.max()),
                "dolmino_phase_0_share": float(w0[semantics["pool"].eq("Dolmino")].sum()),
                "dolmino_phase_1_share": float(w1[semantics["pool"].eq("Dolmino")].sum()),
                "dolmino_aggregate_share": float(aggregate[semantics["pool"].eq("Dolmino")].sum()),
            }
        )
        for index, domain in enumerate(domains):
            bucket_rows.append(
                {
                    "policy_id": policy_id,
                    "mixture_sha256": mixture_sha256,
                    "domain": domain,
                    "phase_0_weight": w0[index],
                    "phase_1_weight": w1[index],
                    "aggregate_weight": aggregate[index],
                    "phase_contrast": contrast[index],
                    "proportional_weight": proportional[index],
                    "aggregate_minus_proportional": aggregate[index] - proportional[index],
                    "actual_epochs": actual_epochs[index],
                    "simulated_epochs": simulated_epochs[index],
                }
            )

    policies = pd.DataFrame(policy_rows)
    policies["uncheatable_percentile"] = policies["uncheatable_bpb"].rank(pct=True, method="average")
    policies["table9_percentile"] = policies["table9_macro_bpb"].rank(pct=True, method="average")
    policy_buckets = pd.DataFrame(bucket_rows).merge(
        semantics[
            [
                "domain",
                "pool",
                "semantic_group",
                "content_family",
                "quality_tier",
                "description",
                "phase_prior",
                "available_tokens_b",
            ]
        ],
        on="domain",
        how="left",
        validate="many_to_one",
    )
    assert policies["mixture_sha256"].is_unique
    assert len(policy_buckets) == len(policies) * len(domains)
    return AuditData(
        policies=policies,
        policy_buckets=policy_buckets,
        semantics=semantics,
        proportional=proportional,
    )


def repeat_noise_by_target(policies: pd.DataFrame) -> dict[str, float]:
    return {
        "uncheatable": float(policies["uncheatable_sd"].dropna().median()),
        "table9": float(policies["table9_sd"].dropna().median()),
    }


def holm_adjusted_pvalues(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    test_count = len(ordered)
    for rank, (key, pvalue) in enumerate(ordered):
        running_max = max(running_max, (test_count - rank) * pvalue)
        adjusted[key] = min(1.0, running_max)
    return adjusted


def statistical_resolution_summary(
    policies: pd.DataFrame,
    controlled_rules: pd.DataFrame,
    controlled_pairs: list[dict[str, object]],
) -> dict[str, object]:
    proportional_noise = json.loads(PROPORTIONAL_NOISE_SUMMARY.read_text())
    control_noise = pd.read_csv(CONTROLLED_AUDIT_DIR / "control_noise_summary.csv")
    pair_by_domain = {str(pair["domain"]): pair for pair in controlled_pairs}
    repeated = policies[policies["observation_count"].gt(1)]
    archive_noise = repeat_noise_by_target(policies)

    result: dict[str, object] = {
        "contract": (
            "The thresholds below use 1.96 * sqrt(2) * SD and assume independent, equal-variance runs. "
            "They are descriptive scale screens, not paired confidence intervals or p-values."
        ),
        "shared_seed_caveat": (
            "Each +d/-d/tied triple shares a data seed, but no exact triple is repeated across enough seeds to "
            "estimate the within-triple covariance. Cross-seed repeats therefore cannot identify the paired "
            "standard error."
        ),
        "multiplicity_caveat": (
            "The rough Holm screen treats the 39 bucket fibers as one family per target and uses the archive "
            "median repeat SD. It remains approximate because the variance is heterogeneous and the shared-seed "
            "correlation is unknown."
        ),
        "targets": {},
    }

    for target in TARGET_COLUMNS:
        matched_anchor = CONTROLLED_TARGET_ANCHOR[target]
        local_control = control_noise[
            control_noise["panel"].eq("frontier_phase_fiber")
            & control_noise["anchor_id"].eq(matched_anchor)
            & control_noise["target"].eq(target)
        ]
        assert len(local_control) == 1
        local_control_row = local_control.iloc[0]
        proportional_sd = float(proportional_noise[f"{target if target == 'uncheatable' else 'table9_macro'}_bpb_sd"])
        noise_sources = [
            {
                "id": "archive_median",
                "label": "Archive repeated coordinates",
                "detail": f"median SD across {len(repeated)} heterogeneous coordinates",
                "n": len(repeated),
                "sd": archive_noise[target],
                "sd_min": float(repeated[f"{target}_sd"].min()),
                "sd_max": float(repeated[f"{target}_sd"].max()),
            },
            {
                "id": "proportional",
                "label": "Proportional repeats",
                "detail": "one phase-tied coordinate",
                "n": int(proportional_noise["n_repeats"]),
                "sd": proportional_sd,
                "sd_min": None,
                "sd_max": None,
            },
            {
                "id": "matched_frontier",
                "label": "Matched frontier tied controls",
                "detail": "same aggregate anchor as the displayed bucket fibers",
                "n": int(local_control_row["control_count"]),
                "sd": float(local_control_row["control_sd_bpb"]),
                "sd_min": None,
                "sd_max": None,
            },
        ]
        for source in noise_sources:
            source["rough_95_difference_threshold"] = float(1.96 * np.sqrt(2.0) * source["sd"])

        pair_pvalues: dict[str, float] = {}
        pair_metrics: dict[str, dict[str, float]] = {}
        for domain, pair_group in pair_by_domain.items():
            anchors = pair_group["anchors"]
            assert isinstance(anchors, list)
            pair = next(anchor for anchor in anchors if anchor["anchor_id"] == matched_anchor)
            metric = pair["metrics"][target]
            pair_gap = float(metric["pair_gap_bpb"])
            z_score = pair_gap / (np.sqrt(2.0) * archive_noise[target])
            pvalue = float(2.0 * norm.sf(z_score))
            pair_pvalues[domain] = pvalue
            pair_metrics[domain] = {
                "pair_gap_bpb": pair_gap,
                "rough_independent_pvalue": pvalue,
            }
        adjusted_pvalues = holm_adjusted_pvalues(pair_pvalues)
        for domain, adjusted in adjusted_pvalues.items():
            pair_metrics[domain]["rough_holm_adjusted_pvalue"] = adjusted

        counterexamples: dict[str, dict[str, object]] = {}
        target_rules = controlled_rules[controlled_rules["target"].eq(target)]
        for rule in target_rules.itertuples(index=False):
            for counterexample in json.loads(rule.controlled_counterexamples_json):
                counterexamples[str(counterexample["domain"])] = counterexample

        counterexample_rows = []
        for domain, counterexample in counterexamples.items():
            metrics = pair_metrics[domain]
            counterexample_rows.append(
                {
                    "domain": domain,
                    "evidence_kind": str(counterexample["evidence_kind"]),
                    "pair_gap_bpb": metrics["pair_gap_bpb"],
                    "net_gain_vs_tied_bpb": float(counterexample["matched_anchor_net_gain_bpb"]),
                    "rough_independent_pvalue": metrics["rough_independent_pvalue"],
                    "rough_holm_adjusted_pvalue": metrics["rough_holm_adjusted_pvalue"],
                }
            )
        direction_specific = [row for row in counterexample_rows if row["evidence_kind"] == "direction_specific"]
        minimum_threshold = min(source["rough_95_difference_threshold"] for source in noise_sources)
        maximum_threshold = max(source["rough_95_difference_threshold"] for source in noise_sources)
        result["targets"][target] = {
            "noise_sources": noise_sources,
            "rough_95_threshold_min": minimum_threshold,
            "rough_95_threshold_max": maximum_threshold,
            "counterexample_domain_count": len(counterexample_rows),
            "direction_specific_counterexample_count": len(direction_specific),
            "nominal_pair_gap_count_archive_noise": sum(
                row["rough_independent_pvalue"] < 0.05 for row in counterexample_rows
            ),
            "holm_pair_gap_count_archive_noise": sum(
                row["rough_holm_adjusted_pvalue"] < 0.05 for row in counterexample_rows
            ),
            "direction_specific_gain_count_clearing_any_threshold": sum(
                abs(row["net_gain_vs_tied_bpb"]) >= minimum_threshold for row in direction_specific
            ),
            "direction_specific_gain_count_clearing_all_thresholds": sum(
                abs(row["net_gain_vs_tied_bpb"]) >= maximum_threshold for row in direction_specific
            ),
            "counterexamples": sorted(
                counterexample_rows,
                key=lambda row: (-float(row["pair_gap_bpb"]), str(row["domain"])),
            ),
            "pair_metrics": pair_metrics,
        }
    return result


def select_tail_policies(policies: pd.DataFrame) -> pd.DataFrame:
    policies = policies[policies["phase_regime"].eq("genuine_two_phase")].copy()
    rows: list[dict[str, object]] = []
    for target, metric in TARGET_COLUMNS.items():
        for tail, selected in (
            ("best", policies.nsmallest(GLOBAL_TAIL_COUNT, metric)),
            ("worst", policies.nlargest(GLOBAL_TAIL_COUNT, metric)),
        ):
            for rank, row in enumerate(selected.itertuples(index=False), start=1):
                rows.append(
                    {
                        "policy_id": row.policy_id,
                        "target": target,
                        "selection_scope": "global",
                        "selection_series": "",
                        "tail": tail,
                        "rank": rank,
                    }
                )

        single_series = policies[policies["series_count"].eq(1)]
        for series, group in single_series.groupby("single_series", sort=True):
            if len(group) < SERIES_MIN_POLICIES:
                continue
            for tail, selected in (("best", group.nsmallest(1, metric)), ("worst", group.nlargest(1, metric))):
                row = selected.iloc[0]
                rows.append(
                    {
                        "policy_id": row["policy_id"],
                        "target": target,
                        "selection_scope": "series_balanced",
                        "selection_series": series,
                        "tail": tail,
                        "rank": 1,
                    }
                )
    selections = pd.DataFrame(rows).drop_duplicates()
    return selections.merge(policies, on="policy_id", how="left", validate="many_to_one")


def format_bucket_list(frame: pd.DataFrame, column: str, *, largest: bool, count: int = 5) -> str:
    selected = frame.nlargest(count, column) if largest else frame.nsmallest(count, column)
    return "; ".join(f"{row.domain} {getattr(row, column):+.4f}" for row in selected.itertuples(index=False))


def controlled_evidence(dossier: pd.DataFrame, domain: str, target: str) -> tuple[str, str, float, float]:
    row = dossier.loc[domain]
    direction = str(row[f"{target}_cross_anchor_direction"])
    gain = float(row[f"{target}_matched_anchor_net_gain_bpb"])
    opposite_minus_tied = float(row[f"{target}_matched_anchor_opposite_minus_tied_bpb"])
    if direction == "anchor_dependent":
        return "anchor-dependent", "none", gain, opposite_minus_tied

    preferred = direction.removesuffix("_consistent")
    if gain > 0.0 and opposite_minus_tied >= 0.0:
        evidence = "direction-specific"
    elif gain > 0.0 and opposite_minus_tied < 0.0:
        evidence = "both orders beat tied"
    else:
        evidence = "tied remains better"
    return preferred, evidence, gain, opposite_minus_tied


def comment_selected_policies(
    selections: pd.DataFrame,
    policy_buckets: pd.DataFrame,
    controlled_dossier: pd.DataFrame,
) -> pd.DataFrame:
    controlled_dossier = controlled_dossier.set_index("domain")
    comments: list[dict[str, object]] = []
    for (policy_id, target), selection_group in selections.groupby(["policy_id", "target"], sort=True):
        row = selection_group.iloc[0]
        anatomy = policy_buckets[policy_buckets["policy_id"].eq(policy_id)].copy()
        metric = TARGET_COLUMNS[target]
        other_target = "table9" if target == "uncheatable" else "uncheatable"
        other_metric = TARGET_COLUMNS[other_target]
        later = format_bucket_list(anatomy, "phase_contrast", largest=True)
        earlier = format_bucket_list(anatomy, "phase_contrast", largest=False)
        aggregate_up = format_bucket_list(anatomy, "aggregate_minus_proportional", largest=True)
        aggregate_down = format_bucket_list(anatomy, "aggregate_minus_proportional", largest=False)
        repeated = anatomy.nlargest(5, "simulated_epochs")
        repeated_text = "; ".join(
            f"{item.domain} {item.simulated_epochs:.2f} epochs" for item in repeated.itertuples(index=False)
        )

        controlled_alignment = {"aligned": 0, "opposed": 0, "too_small": 0}
        controlled_examples: list[tuple[float, str]] = []
        for bucket in anatomy.itertuples(index=False):
            preferred, evidence, gain, _ = controlled_evidence(controlled_dossier, bucket.domain, target)
            if evidence != "direction-specific":
                continue
            contrast = float(bucket.phase_contrast)
            if abs(contrast) < 0.002:
                controlled_alignment["too_small"] += 1
                continue
            aligned = (preferred == "later" and contrast > 0.0) or (preferred == "earlier" and contrast < 0.0)
            key = "aligned" if aligned else "opposed"
            controlled_alignment[key] += 1
            controlled_examples.append(
                (
                    gain,
                    f"{bucket.domain} {'aligns' if aligned else 'opposes'} {preferred} ({contrast:+.4f}; "
                    f"controlled gain {gain:.4f})",
                )
            )
        strongest_controlled_examples = [
            text for _, text in sorted(controlled_examples, key=lambda item: item[0], reverse=True)[:4]
        ]

        scopes = ", ".join(
            sorted(
                {
                    f"{item.selection_scope}:{item.tail}"
                    + (f"[{item.selection_series}]" if item.selection_series else "")
                    for item in selection_group.itertuples(index=False)
                }
            )
        )
        narrative = (
            f"{row['label']} scores {float(row[metric]):.6f} on {TARGET_LABELS[target]} "
            f"and {float(row[other_metric]):.6f} on {TARGET_LABELS[other_target]}. "
            f"It was selected as {scopes}. Its phase TV is {float(row['phase_tv']):.3f}, "
            f"aggregate TV from proportional is {float(row['aggregate_tv_to_proportional']):.3f}, "
            f"and the largest simulated exposure is {float(row['max_simulated_epoch']):.2f} epochs. "
            f"Later concentration: {later}. Earlier concentration: {earlier}. "
            f"Aggregate enrichment over proportional: {aggregate_up}. Aggregate deficits: {aggregate_down}. "
            f"Highest repetition: {repeated_text}."
        )
        if strongest_controlled_examples:
            narrative += (
                f" Against direction-specific fixed-aggregate evidence it has "
                f"{controlled_alignment['aligned']} aligned and {controlled_alignment['opposed']} opposed "
                f"material bucket movements; strongest controlled comparisons: "
                f"{'; '.join(strongest_controlled_examples)}."
            )
        else:
            narrative += " It makes no material movement in a bucket with direction-specific controlled evidence."
        comments.append(
            {
                "policy_id": policy_id,
                "target": target,
                "selection_tags": scopes,
                "comment": narrative,
                "later_top": later,
                "earlier_top": earlier,
                "aggregate_up_top": aggregate_up,
                "aggregate_down_top": aggregate_down,
                "repetition_top": repeated_text,
                "controlled_aligned_count": controlled_alignment["aligned"],
                "controlled_opposed_count": controlled_alignment["opposed"],
                "controlled_small_count": controlled_alignment["too_small"],
            }
        )
    return pd.DataFrame(comments).merge(
        selections.drop_duplicates(["policy_id", "target"])[
            [
                "policy_id",
                "target",
                "label",
                "training_series",
                "panel_tags",
                "proposal_targets",
                "candidate_kinds",
                "observation_count",
                "uncheatable_bpb",
                "uncheatable_sd",
                "table9_macro_bpb",
                "table9_sd",
                "phase_tv",
                "aggregate_tv_to_proportional",
                "max_simulated_epoch",
                "dolmino_phase_0_share",
                "dolmino_phase_1_share",
                "dolmino_aggregate_share",
            ]
        ],
        on=["policy_id", "target"],
        how="left",
        validate="one_to_one",
    )


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(spearmanr(x, y).statistic)


def series_bucket_evidence(data: AuditData) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_columns = [
        "policy_id",
        "single_series",
        "series_count",
        "phase_regime",
        "uncheatable_bpb",
        "table9_macro_bpb",
    ]
    joined = data.policy_buckets.merge(
        data.policies[policy_columns],
        on="policy_id",
        how="left",
        validate="many_to_one",
    )
    joined = joined[joined["series_count"].eq(1) & joined["phase_regime"].eq("genuine_two_phase")].copy()

    series_rows: list[dict[str, object]] = []
    for (series, domain), group in joined.groupby(["single_series", "domain"], sort=True):
        policy_count = group["policy_id"].nunique()
        if policy_count < SERIES_MIN_POLICIES:
            continue
        for target, metric in TARGET_COLUMNS.items():
            tail_count = max(2, int(np.ceil(policy_count * SERIES_TAIL_FRACTION)))
            best = group.nsmallest(tail_count, metric)
            worst = group.nlargest(tail_count, metric)
            series_rows.append(
                {
                    "training_series": series,
                    "domain": domain,
                    "target": target,
                    "policy_count": policy_count,
                    "best_minus_worst_aggregate_weight": float(
                        best["aggregate_weight"].mean() - worst["aggregate_weight"].mean()
                    ),
                    "best_minus_worst_phase_contrast": float(
                        best["phase_contrast"].mean() - worst["phase_contrast"].mean()
                    ),
                    "aggregate_spearman": safe_spearman(group["aggregate_weight"], group[metric]),
                    "phase_contrast_spearman": safe_spearman(group["phase_contrast"], group[metric]),
                }
            )
    by_series = pd.DataFrame(series_rows)

    global_rows: list[dict[str, object]] = []
    for target, metric in TARGET_COLUMNS.items():
        genuine = data.policies[data.policies["phase_regime"].eq("genuine_two_phase")]
        tail_count = max(2, int(np.ceil(len(genuine) * GLOBAL_EVIDENCE_FRACTION)))
        best_ids = set(genuine.nsmallest(tail_count, metric)["policy_id"])
        worst_ids = set(genuine.nlargest(tail_count, metric)["policy_id"])
        for domain, group in data.policy_buckets.groupby("domain", sort=True):
            best = group[group["policy_id"].isin(best_ids)]
            worst = group[group["policy_id"].isin(worst_ids)]
            global_rows.append(
                {
                    "domain": domain,
                    "target": target,
                    "global_best_minus_worst_aggregate_weight": float(
                        best["aggregate_weight"].mean() - worst["aggregate_weight"].mean()
                    ),
                    "global_best_minus_worst_phase_contrast": float(
                        best["phase_contrast"].mean() - worst["phase_contrast"].mean()
                    ),
                }
            )
    global_evidence = pd.DataFrame(global_rows)

    summaries: list[dict[str, object]] = []
    for (domain, target), group in by_series.groupby(["domain", "target"], sort=True):
        aggregate_delta = group["best_minus_worst_aggregate_weight"].dropna()
        phase_delta = group["best_minus_worst_phase_contrast"].dropna()
        aggregate_corr = group["aggregate_spearman"].dropna()
        phase_corr = group["phase_contrast_spearman"].dropna()
        summaries.append(
            {
                "domain": domain,
                "target": target,
                "series_count": group["training_series"].nunique(),
                "series_policy_count_total": int(group["policy_count"].sum()),
                "median_series_best_minus_worst_aggregate_weight": float(aggregate_delta.median()),
                "q25_series_best_minus_worst_aggregate_weight": float(aggregate_delta.quantile(0.25)),
                "q75_series_best_minus_worst_aggregate_weight": float(aggregate_delta.quantile(0.75)),
                "fraction_series_best_has_more_aggregate": float((aggregate_delta > 0.0).mean()),
                "median_series_best_minus_worst_phase_contrast": float(phase_delta.median()),
                "q25_series_best_minus_worst_phase_contrast": float(phase_delta.quantile(0.25)),
                "q75_series_best_minus_worst_phase_contrast": float(phase_delta.quantile(0.75)),
                "fraction_series_best_places_later": float((phase_delta > 0.0).mean()),
                "median_aggregate_spearman": float(aggregate_corr.median()),
                "median_phase_contrast_spearman": float(phase_corr.median()),
                "fraction_series_more_aggregate_correlates_better": float((aggregate_corr < 0.0).mean()),
                "fraction_series_later_correlates_better": float((phase_corr < 0.0).mean()),
            }
        )
    summary = pd.DataFrame(summaries).merge(
        global_evidence,
        on=["domain", "target"],
        how="left",
        validate="one_to_one",
    )
    return by_series, summary


def classify_tail_direction(value: float, fraction: float) -> str:
    if value > 0.0005 and fraction >= 0.67:
        return "later-enriched"
    if value < -0.0005 and fraction <= 0.33:
        return "earlier-enriched"
    return "mixed"


def classify_aggregate_direction(value: float, fraction: float) -> str:
    if value > 0.0005 and fraction >= 0.67:
        return "more aggregate"
    if value < -0.0005 and fraction <= 0.33:
        return "less aggregate"
    return "mixed"


def bucket_intuition_table(
    data: AuditData,
    evidence: pd.DataFrame,
    controlled_dossier: pd.DataFrame,
    repeat_noise: dict[str, float],
) -> pd.DataFrame:
    dossier = controlled_dossier.set_index("domain")
    rows: list[dict[str, object]] = []
    for item in evidence.itertuples(index=False):
        controlled_preference, controlled_kind, controlled_gain, controlled_opposite = controlled_evidence(
            dossier,
            item.domain,
            item.target,
        )
        controlled_gain_over_noise = controlled_gain / repeat_noise[item.target]
        tail_phase = classify_tail_direction(
            item.median_series_best_minus_worst_phase_contrast,
            item.fraction_series_best_places_later,
        )
        tail_aggregate = classify_aggregate_direction(
            item.median_series_best_minus_worst_aggregate_weight,
            item.fraction_series_best_has_more_aggregate,
        )
        if controlled_kind == "direction-specific":
            if tail_phase.startswith(controlled_preference):
                synthesis = (
                    f"Tentative {controlled_preference}; the measured controlled sign and broad tails corroborate."
                )
            elif tail_phase == "mixed":
                synthesis = f"Tentative measured {controlled_preference} sign; broad tails are mixed."
            else:
                synthesis = (
                    f"Conflict: the controlled comparison favors {controlled_preference}, but broad tails favor "
                    f"{tail_phase}."
                )
            synthesis += (
                f" Its gain is {controlled_gain_over_noise:.2f}x the median repeat-coordinate SD, so "
                f"{'it clears' if controlled_gain_over_noise >= 1.0 else 'it does not clear'} that noise scale."
            )
        elif controlled_kind == "both orders beat tied":
            synthesis = (
                f"No isolated order claim: {controlled_preference} wins the pair, but both orders beat tied; "
                f"broad tails are {tail_phase}."
            )
        elif controlled_kind == "tied remains better":
            synthesis = (
                f"Avoid treating {controlled_preference} as beneficial: it only reduces harm versus the reverse; "
                f"broad tails are {tail_phase}."
            )
        else:
            synthesis = f"Anchor-dependent controlled effect; broad tails are {tail_phase}."
        synthesis += f" Aggregate association: {tail_aggregate}."
        rows.append(
            {
                **item._asdict(),
                "controlled_preference": controlled_preference,
                "controlled_evidence_kind": controlled_kind,
                "controlled_net_gain_bpb": controlled_gain,
                "controlled_opposite_minus_tied_bpb": controlled_opposite,
                "controlled_repeat_coordinate_sd_bpb": repeat_noise[item.target],
                "controlled_gain_over_repeat_coordinate_sd": controlled_gain_over_noise,
                "tail_phase_direction": tail_phase,
                "tail_aggregate_direction": tail_aggregate,
                "synthesis": synthesis,
            }
        )
    result = pd.DataFrame(rows).merge(
        data.semantics[
            [
                "domain",
                "pool",
                "semantic_group",
                "content_family",
                "quality_tier",
                "description",
                "phase_prior",
                "available_tokens_b",
            ]
        ],
        on="domain",
        how="left",
        validate="many_to_one",
    )
    return result


def rule_alignment(policy_buckets: pd.DataFrame, rule: PhaseOrderRule) -> pd.Series:
    selected = policy_buckets[policy_buckets["domain"].isin(rule.domains)]
    alignment = selected.groupby("policy_id")["phase_contrast"].sum()
    if rule.expected_direction == "earlier":
        alignment = -alignment
    return alignment


def policy_ref(policies: pd.DataFrame, policy_id: str | None, target: str) -> dict[str, object]:
    if policy_id is None:
        return {}
    row = policies.loc[policies["policy_id"].eq(policy_id)].iloc[0]
    metric = TARGET_COLUMNS[target]
    reference: dict[str, object] = {
        "policy_id": policy_id,
        "label": row["label"],
        "training_series": row["training_series"],
        "bpb": float(row[metric]),
        "phase_tv": float(row["phase_tv"]),
        "aggregate_tv_to_proportional": float(row["aggregate_tv_to_proportional"]),
    }
    if "rule_alignment" in row.index:
        reference["rule_alignment"] = float(row["rule_alignment"])
    return reference


def rule_evidence_table(
    data: AuditData,
    rules: tuple[PhaseOrderRule, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    genuine = data.policies[data.policies["phase_regime"].eq("genuine_two_phase")]
    single_series = genuine[genuine["series_count"].eq(1)]
    series_counts = single_series.groupby("single_series")["policy_id"].nunique()
    valid_series = set(series_counts[series_counts.ge(SERIES_MIN_POLICIES)].index)
    for rule in rules:
        alignment = rule_alignment(data.policy_buckets, rule)
        policies = genuine.merge(
            alignment.rename("rule_alignment"),
            left_on="policy_id",
            right_index=True,
            how="left",
            validate="one_to_one",
        )
        for target, metric in TARGET_COLUMNS.items():
            tail_count = max(2, int(np.ceil(len(policies) * GLOBAL_EVIDENCE_FRACTION)))
            best = policies.nsmallest(tail_count, metric)
            worst = policies.nlargest(tail_count, metric)
            series_deltas: list[float] = []
            for series, group in policies[policies["single_series"].isin(valid_series)].groupby(
                "single_series", sort=True
            ):
                assert series in valid_series
                count = max(2, int(np.ceil(len(group) * SERIES_TAIL_FRACTION)))
                series_deltas.append(
                    float(
                        group.nsmallest(count, metric)["rule_alignment"].mean()
                        - group.nlargest(count, metric)["rule_alignment"].mean()
                    )
                )
            best_counterexample = best[best["rule_alignment"].lt(0.0)].nsmallest(1, "rule_alignment")
            worst_conformer = worst[worst["rule_alignment"].gt(0.0)].nlargest(1, "rule_alignment")
            counterexample_id = None if best_counterexample.empty else str(best_counterexample.iloc[0]["policy_id"])
            conformer_id = None if worst_conformer.empty else str(worst_conformer.iloc[0]["policy_id"])
            rows.append(
                {
                    "rule_id": rule.rule_id,
                    "rule_label": rule.label,
                    "premise": rule.premise,
                    "expected_direction": rule.expected_direction,
                    "domain_count": len(rule.domains),
                    "target": target,
                    "global_best_minus_worst_alignment": float(
                        best["rule_alignment"].mean() - worst["rule_alignment"].mean()
                    ),
                    "median_series_best_minus_worst_alignment": float(np.median(series_deltas)),
                    "fraction_series_support_rule": float(np.mean(np.asarray(series_deltas) > 0.0)),
                    "series_count": len(series_deltas),
                    "best_counterexample_json": json.dumps(policy_ref(policies, counterexample_id, target)),
                    "worst_conformer_json": json.dumps(policy_ref(policies, conformer_id, target)),
                }
            )
    return pd.DataFrame(rows)


def controlled_rule_evidence_table() -> pd.DataFrame:
    tests = pd.read_csv(CONTROLLED_AUDIT_DIR / "phase_order_rule_tests.csv")
    required = {
        "rule_id",
        "rule_label",
        "premise",
        "expected_direction",
        "target",
        "domain",
        "outcome",
        "evidence_kind",
        "table9_anchor_later_minus_earlier_bpb",
        "uncheatable_anchor_later_minus_earlier_bpb",
        "matched_anchor_net_gain_bpb",
        "comment",
    }
    missing = required - set(tests.columns)
    if missing:
        raise ValueError(f"Controlled rule audit is missing columns: {sorted(missing)}")

    evidence_priority = {
        "direction_specific": 0,
        "both_orders_beat_tied": 1,
        "direction_only": 2,
    }
    rows: list[dict[str, object]] = []
    group_columns = ["rule_id", "rule_label", "premise", "expected_direction", "target"]
    for keys, group in tests.groupby(group_columns, sort=True, dropna=False):
        counterexamples = group[group["outcome"].eq("counterexample")].copy()
        representative: dict[str, object] = {}
        all_counterexamples: list[dict[str, object]] = []
        if not counterexamples.empty:
            counterexamples["evidence_priority"] = (
                counterexamples["evidence_kind"].map(evidence_priority).fillna(len(evidence_priority))
            )
            counterexamples["absolute_cross_anchor_effect"] = (
                counterexamples[
                    [
                        "table9_anchor_later_minus_earlier_bpb",
                        "uncheatable_anchor_later_minus_earlier_bpb",
                    ]
                ]
                .abs()
                .mean(axis=1)
            )
            row = counterexamples.sort_values(
                ["evidence_priority", "absolute_cross_anchor_effect", "matched_anchor_net_gain_bpb"],
                ascending=[True, False, False],
            ).iloc[0]
            counterexamples = counterexamples.sort_values(
                ["evidence_priority", "absolute_cross_anchor_effect", "domain"],
                ascending=[True, False, True],
            )
            all_counterexamples = [
                {
                    "domain": counterexample["domain"],
                    "evidence_kind": counterexample["evidence_kind"],
                    "matched_anchor_net_gain_bpb": float(counterexample["matched_anchor_net_gain_bpb"]),
                    "table9_anchor_later_minus_earlier_bpb": float(
                        counterexample["table9_anchor_later_minus_earlier_bpb"]
                    ),
                    "uncheatable_anchor_later_minus_earlier_bpb": float(
                        counterexample["uncheatable_anchor_later_minus_earlier_bpb"]
                    ),
                    "comment": counterexample["comment"],
                }
                for _, counterexample in counterexamples.iterrows()
            ]
            representative = {
                "domain": row["domain"],
                "evidence_kind": row["evidence_kind"],
                "matched_anchor_net_gain_bpb": float(row["matched_anchor_net_gain_bpb"]),
                "table9_anchor_later_minus_earlier_bpb": float(row["table9_anchor_later_minus_earlier_bpb"]),
                "uncheatable_anchor_later_minus_earlier_bpb": float(row["uncheatable_anchor_later_minus_earlier_bpb"]),
                "comment": row["comment"],
            }
        rule_id, rule_label, premise, expected_direction, target = keys
        rows.append(
            {
                "rule_id": rule_id,
                "rule_label": rule_label,
                "premise": premise,
                "expected_direction": expected_direction,
                "target": target,
                "controlled_supports": int(group["outcome"].eq("supports").sum()),
                "controlled_direction_specific_supports": int(
                    (group["outcome"].eq("supports") & group["evidence_kind"].eq("direction_specific")).sum()
                ),
                "controlled_counterexamples": int(group["outcome"].eq("counterexample").sum()),
                "controlled_direction_specific_counterexamples": int(
                    (group["outcome"].eq("counterexample") & group["evidence_kind"].eq("direction_specific")).sum()
                ),
                "controlled_anchor_dependent": int(group["outcome"].eq("anchor_dependent").sum()),
                "controlled_counterexample_json": json.dumps(representative),
                "controlled_counterexamples_json": json.dumps(all_counterexamples),
            }
        )
    return pd.DataFrame(rows)


def controlled_pair_payload(data: AuditData) -> list[dict[str, object]]:
    physical_pairs = pd.read_csv(CONTROLLED_AUDIT_DIR / "physical_pair_ledger.csv")
    target_pairs = pd.read_csv(CONTROLLED_AUDIT_DIR / "target_pair_ledger.csv")
    manifest = pd.read_csv(FIBER_PANEL_DIR / "candidate_manifest.csv")
    weights = pd.read_csv(FIBER_PANEL_DIR / "phase_weights.csv")

    domains = list(data.semantics["domain"])
    token_counts = data.semantics["available_tokens"].to_numpy(float)
    domain_pairs = physical_pairs[
        physical_pairs["panel"].eq("frontier_phase_fiber") & physical_pairs["contrast_family"].eq("domain_vs_rest")
    ].copy()
    if len(domain_pairs) != 2 * len(domains):
        raise ValueError(f"Expected two controlled pairs per domain, found {len(domain_pairs)} rows")

    centers = manifest[manifest["contrast_family"].eq("center_control")].copy()
    if centers.duplicated(["anchor_id", "seed_block"]).any():
        raise ValueError("Controlled center candidates are not unique by anchor and seed block")
    center_lookup = centers.set_index(["anchor_id", "seed_block"])

    def phase_matrix(candidate_id: str) -> np.ndarray:
        candidate = weights[weights["candidate_id"].eq(candidate_id)]
        matrix = (
            candidate.pivot(index="phase", columns="domain", values="weight")
            .reindex(index=[0, 1], columns=domains)
            .to_numpy(float)
        )
        if matrix.shape != (2, len(domains)) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid controlled phase weights for {candidate_id}: {matrix.shape}")
        if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9):
            raise ValueError(f"Controlled phase weights do not sum to one for {candidate_id}")
        return matrix

    by_domain: list[dict[str, object]] = []
    for domain_index, domain in enumerate(domains):
        direction_id = f"domain_{domain_index:02d}"
        pair_rows = domain_pairs[domain_pairs["direction_id"].eq(direction_id)].sort_values("anchor_id")
        if set(pair_rows["anchor_id"]) != set(CONTROLLED_TARGET_ANCHOR.values()):
            raise ValueError(f"Missing controlled anchor for {domain} / {direction_id}")

        anchors: list[dict[str, object]] = []
        for pair in pair_rows.itertuples(index=False):
            center_row = center_lookup.loc[(pair.anchor_id, pair.seed_block)]
            center_candidate_id = str(center_row["candidate_id"])
            plus = phase_matrix(str(pair.plus_candidate_id))
            minus = phase_matrix(str(pair.minus_candidate_id))
            center = phase_matrix(center_candidate_id)
            plus_aggregate = PHASE_FRACTIONS @ plus
            minus_aggregate = PHASE_FRACTIONS @ minus
            if not np.allclose(plus_aggregate, minus_aggregate, atol=1e-9):
                raise ValueError(f"Antithetic aggregate mismatch for {pair.pair_id}")
            if not np.allclose(center[0], center[1], atol=1e-9):
                raise ValueError(f"Center policy is not phase tied for {center_candidate_id}")
            if not np.allclose(plus_aggregate, center[0], atol=1e-9):
                raise ValueError(f"Pair and center aggregate mismatch for {pair.pair_id}")

            pair_targets = target_pairs[target_pairs["pair_id"].eq(pair.pair_id)].set_index("target")
            if set(pair_targets.index) != set(TARGET_COLUMNS):
                raise ValueError(f"Missing target rows for controlled pair {pair.pair_id}")
            metric_payload = {
                target: {
                    "plus_bpb": float(pair_targets.loc[target, "plus_bpb"]),
                    "minus_bpb": float(pair_targets.loc[target, "minus_bpb"]),
                    "center_bpb": float(pair_targets.loc[target, "center_bpb"]),
                    "plus_delta_vs_control": float(pair_targets.loc[target, "plus_delta_vs_control"]),
                    "minus_delta_vs_control": float(pair_targets.loc[target, "minus_delta_vs_control"]),
                    "better_sign": str(pair_targets.loc[target, "better_sign"]),
                    "pair_gap_bpb": float(pair_targets.loc[target, "pair_gap_bpb"]),
                    "odd_order_effect_bpb": float(pair_targets.loc[target, "odd_order_effect_bpb"]),
                    "even_asymmetry_cost_bpb": float(pair_targets.loc[target, "even_asymmetry_cost_bpb"]),
                    "gap_band": str(pair_targets.loc[target, "gap_band"]),
                }
                for target in TARGET_COLUMNS
            }
            simulated_epochs = SIMULATED_EPOCH_TARGET_BUDGET * plus_aggregate / token_counts
            anchors.append(
                {
                    "anchor_id": str(pair.anchor_id),
                    "anchor_label": TARGET_LABELS[
                        next(target for target, anchor in CONTROLLED_TARGET_ANCHOR.items() if anchor == pair.anchor_id)
                    ],
                    "pair_id": str(pair.pair_id),
                    "direction_id": direction_id,
                    "direction_label_plus": str(pair.direction_label_plus),
                    "seed_block": int(pair.seed_block),
                    "data_seed": int(pair.data_seed),
                    "phase_tv": float(pair.plus_phase_tv),
                    "aggregate_max_abs_difference": float(pair.aggregate_max_abs_difference),
                    "plus_candidate_id": str(pair.plus_candidate_id),
                    "minus_candidate_id": str(pair.minus_candidate_id),
                    "center_candidate_id": center_candidate_id,
                    "metrics": metric_payload,
                    "buckets": [
                        {
                            "domain": bucket_domain,
                            "pool": str(data.semantics.iloc[index]["pool"]),
                            "plus_w0": float(plus[0, index]),
                            "plus_w1": float(plus[1, index]),
                            "minus_w0": float(minus[0, index]),
                            "minus_w1": float(minus[1, index]),
                            "center_weight": float(center[0, index]),
                            "aggregate_weight": float(plus_aggregate[index]),
                            "simulated_epochs": float(simulated_epochs[index]),
                        }
                        for index, bucket_domain in enumerate(domains)
                    ],
                }
            )
        by_domain.append(
            {
                "domain": domain,
                "direction_id": direction_id,
                "description": str(data.semantics.iloc[domain_index]["description"]),
                "anchors": anchors,
            }
        )
    return by_domain


def hybrid_fixed_aggregate_analysis(
    registry_path: Path,
    semantics: pd.DataFrame,
    repeat_noise: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observations = pd.read_csv(registry_path)
    panel = observations[
        observations["training_series"].eq("delphi_3e18_hybrid_phase_ordering_validation_20260720")
    ].copy()
    rows: list[dict[str, object]] = []
    package_rows: list[dict[str, object]] = []
    domains = list(semantics["domain"])
    for target, metric in TARGET_COLUMNS.items():
        target_panel = panel[panel["proposal_target"].eq(target)]
        tied = target_panel[target_panel["candidate_kind"].eq("tied_separate_heads_anchor")].set_index(
            "aggregate_kl_coefficient"
        )
        fixed = target_panel[target_panel["candidate_kind"].str.startswith("fixed_aggregate_", na=False)]
        winners = fixed.loc[fixed.groupby("aggregate_kl_coefficient")[metric].idxmin()].copy()
        for winner in winners.itertuples(index=False):
            tied_row = tied.loc[winner.aggregate_kl_coefficient]
            anchor_candidates = fixed[fixed["aggregate_kl_coefficient"].eq(winner.aggregate_kl_coefficient)]
            candidate_count = len(anchor_candidates)
            expected_null_best_delta = float(repeat_noise[target] * norm.ppf((1.0 - 0.375) / (candidate_count + 0.25)))
            rows.append(
                {
                    "target": target,
                    "aggregate_kl_coefficient": float(winner.aggregate_kl_coefficient),
                    "winner_candidate_id": winner.candidate_id,
                    "winner_candidate_kind": winner.candidate_kind,
                    "phase_information_budget": float(winner.phase_information_budget),
                    "winner_bpb": float(getattr(winner, metric)),
                    "tied_bpb": float(tied_row[metric]),
                    "winner_minus_tied_bpb": float(getattr(winner, metric) - tied_row[metric]),
                    "candidate_count": candidate_count,
                    "candidate_fraction_beating_tied": float(
                        (anchor_candidates[metric] < float(tied_row[metric])).mean()
                    ),
                    "repeat_coordinate_sd_bpb": repeat_noise[target],
                    "expected_null_best_delta_bpb": expected_null_best_delta,
                }
            )

        contrasts = []
        for winner in winners.itertuples(index=False):
            phase_0 = parse_mapping(winner.phase_0_weights_json)
            phase_1 = parse_mapping(winner.phase_1_weights_json)
            contrasts.append([phase_1[domain] - phase_0[domain] for domain in domains])
        contrast_array = np.asarray(contrasts)
        all_candidate_contrasts = []
        for candidate in fixed.itertuples(index=False):
            phase_0 = parse_mapping(candidate.phase_0_weights_json)
            phase_1 = parse_mapping(candidate.phase_1_weights_json)
            all_candidate_contrasts.append([phase_1[domain] - phase_0[domain] for domain in domains])
        all_candidate_contrast_array = np.asarray(all_candidate_contrasts)
        for domain_index, domain in enumerate(domains):
            values = contrast_array[:, domain_index]
            candidate_values = all_candidate_contrast_array[:, domain_index]
            winner_later_rate = float((values > 0.002).mean())
            winner_earlier_rate = float((values < -0.002).mean())
            candidate_later_rate = float((candidate_values > 0.002).mean())
            candidate_earlier_rate = float((candidate_values < -0.002).mean())
            package_rows.append(
                {
                    "target": target,
                    "domain": domain,
                    "winner_count": len(values),
                    "later_count_above_0p002": int((values > 0.002).sum()),
                    "earlier_count_below_minus_0p002": int((values < -0.002).sum()),
                    "median_phase_contrast": float(np.median(values)),
                    "min_phase_contrast": float(values.min()),
                    "max_phase_contrast": float(values.max()),
                    "winner_later_rate_above_0p002": winner_later_rate,
                    "winner_earlier_rate_below_minus_0p002": winner_earlier_rate,
                    "candidate_pool_later_rate_above_0p002": candidate_later_rate,
                    "candidate_pool_earlier_rate_below_minus_0p002": candidate_earlier_rate,
                    "winner_minus_candidate_later_rate": winner_later_rate - candidate_later_rate,
                    "winner_minus_candidate_earlier_rate": winner_earlier_rate - candidate_earlier_rate,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(package_rows).merge(
        semantics[["domain", "pool", "description"]],
        on="domain",
        how="left",
        validate="many_to_one",
    )


def write_policy_comments(path: Path, comments: pd.DataFrame, selections: pd.DataFrame) -> None:
    lines = [
        "# Broad two-phase tail policy comments",
        "",
        "Each exact policy coordinate is collapsed across repeated seeds before selection. Global tails show the",
        "actual archive frontier and pathologies; series-balanced tails contribute one best and one worst policy",
        f"from each proposal series with at least {SERIES_MIN_POLICIES} unique coordinates. These comments are",
        "descriptive and do not identify phase-order effects independently of aggregate mixture composition.",
        "",
    ]
    selection_lookup = selections.groupby(["policy_id", "target"])
    for target in ("uncheatable", "table9"):
        target_comments = comments[comments["target"].eq(target)].sort_values(TARGET_COLUMNS[target])
        lines.extend([f"## {TARGET_LABELS[target]}", ""])
        for row in target_comments.itertuples(index=False):
            tags = selection_lookup.get_group((row.policy_id, target))
            tag_text = ", ".join(
                f"{item.selection_scope} {item.tail}"
                + (f" in `{item.selection_series}`" if item.selection_series else "")
                for item in tags.itertuples(index=False)
            )
            lines.extend(
                [
                    f"### `{row.policy_id}` - {row.label}",
                    "",
                    f"- Selection: {tag_text}.",
                    f"- Provenance: `{row.training_series}`; panel `{row.panel_tags}`.",
                    f"- Metrics: Uncheatable {row.uncheatable_bpb:.6f}; Table-9 {row.table9_macro_bpb:.6f}.",
                    f"- Geometry: phase TV {row.phase_tv:.4f}; aggregate TV to proportional "
                    f"{row.aggregate_tv_to_proportional:.4f}; max simulated epochs {row.max_simulated_epoch:.2f}.",
                    f"- Comment: {row.comment}",
                    "",
                ]
            )
    path.write_text("\n".join(lines))


def write_markdown_report(
    path: Path,
    data: AuditData,
    selections: pd.DataFrame,
    comments: pd.DataFrame,
    bucket_intuition: pd.DataFrame,
    broad_rule_stress: pd.DataFrame,
    controlled_rules: pd.DataFrame,
    hybrid_winners: pd.DataFrame,
    hybrid_package: pd.DataFrame,
) -> None:
    unique_selected = comments["policy_id"].nunique()
    global_series_rows: list[dict[str, object]] = []
    for (target, tail), group in selections[selections["selection_scope"].eq("global")].groupby(
        ["target", "tail"], sort=True
    ):
        series = {item for value in group["training_series"] for item in str(value).split(" | ") if item}
        global_series_rows.append({"target": target, "tail": tail, "proposal_series": len(series)})
    global_series = pd.DataFrame(global_series_rows)
    geometry_rows: list[dict[str, object]] = []
    for target, metric in TARGET_COLUMNS.items():
        target_global = selections[selections["target"].eq(target) & selections["selection_scope"].eq("global")]
        for tail in ("best", "worst"):
            ids = set(target_global.loc[target_global["tail"].eq(tail), "policy_id"])
            policies = data.policies[data.policies["policy_id"].isin(ids)]
            geometry_rows.append(
                {
                    "target": target,
                    "tail": tail,
                    "bpb_median": float(policies[metric].median()),
                    "phase_tv_median": float(policies["phase_tv"].median()),
                    "phase_tv_min": float(policies["phase_tv"].min()),
                    "phase_tv_max": float(policies["phase_tv"].max()),
                    "aggregate_tv_median": float(policies["aggregate_tv_to_proportional"].median()),
                    "max_simulated_epoch_median": float(policies["max_simulated_epoch"].median()),
                }
            )
    geometry = pd.DataFrame(geometry_rows)
    direction_specific = bucket_intuition[bucket_intuition["controlled_evidence_kind"].eq("direction-specific")]
    sub_noise_count = int(direction_specific["controlled_gain_over_repeat_coordinate_sd"].lt(1.0).sum())
    extreme_epochs = data.policies[data.policies["max_simulated_epoch"].gt(30.0)][
        [
            "policy_id",
            "label",
            "max_simulated_epoch",
            "uncheatable_bpb",
            "uncheatable_percentile",
            "table9_macro_bpb",
            "table9_percentile",
        ]
    ].sort_values("max_simulated_epoch")
    lines = [
        "# Broad Delphi 3e18 best/worst mixture audit",
        "",
        "## Scope and estimand",
        "",
        f"- {len(data.policies):,} coordinates labeled two-phase in the append-only Delphi 3e18 archive.",
        f"- {int(data.policies['phase_regime'].eq('genuine_two_phase').sum()):,} have phase TV at least "
        f"{GENUINE_TWO_PHASE_TV:.2f} and define the tail-selection population.",
        f"- {int(data.policies['observation_count'].sum()):,} observations before exact-coordinate collapse.",
        f"- {unique_selected} unique policies received row-level qualitative comments.",
        f"- Global tails use the best and worst {GLOBAL_TAIL_COUNT} coordinates for each objective.",
        f"- Proposal-balanced tails add one best and one worst coordinate per series with at least "
        f"{SERIES_MIN_POLICIES} policies.",
        "- Broad-tail associations combine aggregate composition and phase order. They are used to test whether",
        "  controlled bucket-level findings recur, not to replace fixed-aggregate causal contrasts.",
        "- Every policy was trained with the same 358M-parameter Delphi 3e18 configuration for about 1.58B",
        "  tokens and a 79.8138/20.1862 WSD phase split. Reported simulated epochs normalize aggregate weights",
        "  to a 6.325T-token production budget; they are not physical repetitions in these 1.58B-token runs.",
        "",
        "## Selection diversity",
        "",
        global_series.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The extreme tails are proposal-dependent. Global worst policies are disproportionately raw surrogate",
        "optima from low-sample fits, while frontier policies concentrate in low-epsilon and local-frontier",
        "panels. The series-balanced audit is therefore required before treating any pooled bucket enrichment",
        "as a reusable rule.",
        "",
        "## Tail geometry",
        "",
        geometry.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Both objectives have low-to-moderate phase TV in their best tail and much larger phase and aggregate",
        "extrapolation in their worst tail. This does not prove the optimum is near tied, but it rules out the",
        "claim that aggressive asymmetry is automatically useful.",
        "",
        "## Fixed-aggregate package evidence",
        "",
        "The hybrid panel compares four phase-order mechanisms at four independently selected aggregate",
        "anchors per target. The table reports the best of 20 asymmetric policies at each anchor against its",
        "exact tied control. Selecting the winner induces optimism, especially for noisy Table-9, so these rows",
        "are package hypotheses rather than unbiased treatment estimates. `candidate_fraction_beating_tied`",
        "reports the full panel rather than only its winner. `expected_null_best_delta_bpb` is a contextual",
        "winner-of-20 benchmark under independent normal candidate noise and one shared tied control at the",
        "median repeat-coordinate SD; it is not a hypothesis test.",
        "",
        hybrid_winners.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    package_display_rows: list[pd.DataFrame] = []
    for target in ("uncheatable", "table9"):
        target_package = hybrid_package[hybrid_package["target"].eq(target)]
        package_display_rows.append(target_package.nlargest(6, "median_phase_contrast"))
        package_display_rows.append(target_package.nsmallest(6, "median_phase_contrast"))
    package_display = pd.concat(package_display_rows).drop_duplicates(["target", "domain"])
    lines.extend(
        [
            "Buckets recurring in the late or early direction among the four selected package winners:",
            "",
            package_display[
                [
                    "target",
                    "domain",
                    "later_count_above_0p002",
                    "earlier_count_below_minus_0p002",
                    "median_phase_contrast",
                    "candidate_pool_later_rate_above_0p002",
                    "candidate_pool_earlier_rate_below_minus_0p002",
                    "winner_minus_candidate_later_rate",
                    "winner_minus_candidate_earlier_rate",
                ]
            ]
            .sort_values(["target", "median_phase_contrast"], ascending=[True, False])
            .to_markdown(index=False, floatfmt=".6f"),
            "",
            "Uncheatable has the clearer package-level result, but recurrence must be compared with the candidate",
            "pool. StackEdu/FIM were already later in more than 92% of candidates, so their 4/4 and 3/4 winner",
            "counts do not identify them. Science/technology-high and olmOCR are more enriched among winners than",
            "their candidate base rates; synthetic QA and common-crawl HQ recur early. For Table-9, StackEdu/FIM",
            "were later in every candidate and common-crawl HQ was early in every candidate, so winner recurrence",
            "cannot validate those choices. Synthetic QA changes sign across mechanisms and remains interaction-",
            "and anchor-dependent.",
            "",
            "## Per-bucket working map",
            "",
            "The table below combines the strict fixed-aggregate audit with proposal-series-balanced tail evidence.",
            "A controlled `direction-specific` label requires that the preferred ordering beat tied and its reversal",
            "not beat tied. This is an observed sign criterion, not a significance threshold: "
            f"{sub_noise_count}/{len(direction_specific)} such gains are smaller than the target's median",
            "repeat-coordinate SD. All broad-tail columns remain associative.",
            "",
        ]
    )
    display_columns = [
        "target",
        "domain",
        "controlled_preference",
        "controlled_evidence_kind",
        "controlled_net_gain_bpb",
        "controlled_repeat_coordinate_sd_bpb",
        "controlled_gain_over_repeat_coordinate_sd",
        "tail_phase_direction",
        "fraction_series_best_places_later",
        "median_series_best_minus_worst_phase_contrast",
        "tail_aggregate_direction",
        "fraction_series_best_has_more_aggregate",
        "median_series_best_minus_worst_aggregate_weight",
        "synthesis",
    ]
    lines.extend(
        [
            bucket_intuition[display_columns].to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Controlled antithetic counterexamples",
            "",
            "These are the causal order tests. Each `+d/-d` pair has the same aggregate mixture and data seed;",
            "the phase contrast is reversed exactly around the tied aggregate. A counterexample requires the",
            "direction opposite the claimed rule at both aggregate anchors. `direction_specific` is strongest:",
            "only the opposite ordering beats tied. `both_orders_beat_tied` identifies a pair winner but does not",
            "attribute the gain to order alone. `direction_only` means the opposite order is less harmful while",
            "tied remains best.",
            "",
        ]
    )
    controlled_display = controlled_rules.copy()
    controlled_display["representative_domain"] = controlled_display["controlled_counterexample_json"].map(
        lambda value: json.loads(value).get("domain", "")
    )
    controlled_display["representative_kind"] = controlled_display["controlled_counterexample_json"].map(
        lambda value: json.loads(value).get("evidence_kind", "")
    )
    controlled_display["representative_net_gain_bpb"] = controlled_display["controlled_counterexample_json"].map(
        lambda value: json.loads(value).get("matched_anchor_net_gain_bpb", np.nan)
    )
    controlled_display["table9_anchor_effect_bpb"] = controlled_display["controlled_counterexample_json"].map(
        lambda value: json.loads(value).get("table9_anchor_later_minus_earlier_bpb", np.nan)
    )
    controlled_display["uncheatable_anchor_effect_bpb"] = controlled_display["controlled_counterexample_json"].map(
        lambda value: json.loads(value).get("uncheatable_anchor_later_minus_earlier_bpb", np.nan)
    )
    lines.extend(
        [
            controlled_display[
                [
                    "target",
                    "rule_label",
                    "controlled_supports",
                    "controlled_direction_specific_supports",
                    "controlled_counterexamples",
                    "controlled_direction_specific_counterexamples",
                    "controlled_anchor_dependent",
                    "representative_domain",
                    "representative_kind",
                    "representative_net_gain_bpb",
                    "table9_anchor_effect_bpb",
                    "uncheatable_anchor_effect_bpb",
                ]
            ].to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Broad-tail heuristic stress tests (associative)",
            "",
            "These are not causal phase-order counterexamples. A best-tail heuristic violation is a top-decile",
            "policy that moves the rule's bucket set opposite the claimed direction; a worst-tail conformer",
            "follows the rule but still performs in the bottom decile. Aggregate composition and all other bucket",
            "moves vary simultaneously. These rows only show that a rule is not necessary or sufficient as a",
            "standalone archive-level prescription.",
            "",
        ]
    )
    rule_display = broad_rule_stress.copy()
    rule_display["best_heuristic_violation"] = rule_display["best_counterexample_json"].map(
        lambda value: json.loads(value).get("label", "")
    )
    rule_display["worst_conformer"] = rule_display["worst_conformer_json"].map(
        lambda value: json.loads(value).get("label", "")
    )
    rule_display["best_heuristic_violation_alignment"] = rule_display["best_counterexample_json"].map(
        lambda value: json.loads(value).get("rule_alignment", np.nan)
    )
    rule_display["worst_conformer_alignment"] = rule_display["worst_conformer_json"].map(
        lambda value: json.loads(value).get("rule_alignment", np.nan)
    )
    lines.extend(
        [
            rule_display[
                [
                    "target",
                    "rule_label",
                    "global_best_minus_worst_alignment",
                    "median_series_best_minus_worst_alignment",
                    "fraction_series_support_rule",
                    "best_heuristic_violation",
                    "best_heuristic_violation_alignment",
                    "worst_conformer",
                    "worst_conformer_alignment",
                ]
            ].to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Reading the evidence",
            "",
            "- Use controlled antithetic/domain-fiber results for phase-order claims and counterexamples.",
            "- Use broad tails to identify recurring aggregate deficits, overload, and candidate-specific",
            "  heuristic violations, not causal order reversals.",
            "- Do not infer that a bucket belongs early or late merely because it is enriched in an observed",
            "  frontier policy; aggregate mixture quality and proposal provenance remain confounded.",
            "- The interactive HTML report exposes every audited policy and its full 39-bucket anatomy.",
            "- `all Dolmino late` and `broad pretraining early` are algebraic complements at fixed aggregate;",
            "  their duplicate stress-test statistics are not independent evidence.",
            "",
            "## Manual-mixture hypotheses",
            "",
            "- **Uncheatable:** keep the validated unch05-like aggregate fixed. Test literature-high and",
            "  science/technology-high later; both controlled signs clear the median repeat-coordinate SD and broad",
            "  tails corroborate. olmOCR and arXiv later are weaker arms whose controlled signs are below that noise",
            "  scale. Test StackEdu/FIM as a joint package arm, not as independently",
            "  identified bucket effects. Synthetic math is the cleanest early move; synthetic QA, common-crawl",
            "  HQ, education/jobs-high, and health-high require paired sign ablations because their package,",
            "  single-bucket, and broad-tail evidence do not all agree.",
            "- **Table-9:** keep a validated t9b075/t9s05-like aggregate fixed. Treat the recurring late",
            "  StackEdu/FIM and synthetic-code/math/thinking bundle with common-crawl HQ early as a package",
            "  hypothesis, not independent bucket effects. These constituents conflict with, or are not isolated by,",
            "  at least one controlled comparison, so compare tied, proposed, and sign-reversed packages. FineMath",
            "  has no consistent support",
            "  and should receive its own sign ablation or be omitted. Separately test synthetic QA late, early, and",
            "  tied: it is the only measured direction-specific one-bucket late signal, but selected packages often",
            "  reverse it.",
            "- Balanced/unstructured TV=0.50 interventions were uniformly harmful, while structured schedules at",
            "  TV 0.41-0.50 occasionally reached the top decile or beat tied. Use 0.10-0.25 as the primary range and",
            "  explicitly test 0.33/0.50 only for structured packages; do not impose a universal hard TV cap.",
            f"- All {len(extreme_epochs)} observed policies above 30 production-budget-normalized per-bucket",
            "  simulated epochs fell at or above the 97.5th percentile (worse) on both objectives. This sparse",
            "  descriptive guardrail does not establish a safe response threshold below 30 epochs.",
            "",
            "Policies above 30 production-budget-normalized per-bucket simulated epochs:",
            "",
            extreme_epochs.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Files",
            "",
            "- `coordinate_policies.csv`: one row per deduplicated two-phase coordinate.",
            "- `policy_bucket_features.csv`: weights, contrasts, and exposure for every policy-bucket pair.",
            "- `tail_selections.csv`: global and series-balanced selection ledger.",
            "- `tail_policy_comments.csv` and `.md`: row-level qualitative audit.",
            "- `bucket_tail_evidence_by_series.csv`: proposal-series-specific bucket evidence.",
            "- `bucket_intuition.csv`: combined controlled and broad-tail working map.",
            "- `controlled_rule_evidence.csv`: fixed-aggregate antithetic rule tests and counterexamples.",
            "- `controlled_antithetic_pairs.json`: exact pair IDs, candidates, scores, and phase weights used by the interactive counterexample viewer.",
            "- `broad_tail_rule_stress.csv`: associative heuristic violations and conformers.",
            "- `hybrid_fixed_aggregate_winners.csv`: selected package winner versus tied at each anchor.",
            "- `hybrid_winner_phase_package.csv`: recurring bucket directions across selected package winners.",
            "- `extreme_simulated_epoch_failures.csv`: the six observed policies above 30 normalized epochs.",
            "- `phase_order_bucket_intuition.html`: standalone D3 report.",
        ]
    )
    path.write_text("\n".join(lines))


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.replace({np.nan: None, np.inf: None, -np.inf: None})
    return json.loads(clean.to_json(orient="records"))


def html_payload(
    data: AuditData,
    selections: pd.DataFrame,
    comments: pd.DataFrame,
    intuition: pd.DataFrame,
    broad_rule_stress: pd.DataFrame,
    controlled_rules: pd.DataFrame,
    controlled_pairs: list[dict[str, object]],
    hybrid_winners: pd.DataFrame,
    hybrid_package: pd.DataFrame,
    statistical_resolution: dict[str, object],
) -> dict[str, object]:
    selection_tags = (
        selections.groupby(["policy_id", "target"], sort=True)
        .apply(
            lambda group: [
                {
                    "scope": row.selection_scope,
                    "tail": row.tail,
                    "series": row.selection_series,
                    "rank": int(row.rank),
                }
                for row in group.itertuples(index=False)
            ],
            include_groups=False,
        )
        .to_dict()
    )
    comments_lookup = {(row.policy_id, row.target): row.comment for row in comments.itertuples(index=False)}
    policy_records: list[dict[str, object]] = []
    bucket_groups = {policy_id: group for policy_id, group in data.policy_buckets.groupby("policy_id", sort=False)}
    for row in data.policies.itertuples(index=False):
        buckets = bucket_groups[row.policy_id]
        assert list(buckets["domain"]) == list(data.semantics["domain"])
        policy_records.append(
            {
                "policy_id": row.policy_id,
                "label": row.label,
                "series": row.training_series,
                "panel_tags": row.panel_tags,
                "proposal_targets": row.proposal_targets,
                "candidate_kinds": row.candidate_kinds,
                "observation_count": int(row.observation_count),
                "uncheatable": row.uncheatable_bpb,
                "table9": row.table9_macro_bpb,
                "uncheatable_sd": None if pd.isna(row.uncheatable_sd) else row.uncheatable_sd,
                "table9_sd": None if pd.isna(row.table9_sd) else row.table9_sd,
                "phase_tv": row.phase_tv,
                "phase_regime": row.phase_regime,
                "aggregate_tv": row.aggregate_tv_to_proportional,
                "max_simulated_epoch": row.max_simulated_epoch,
                "dolmino_phase_0_share": row.dolmino_phase_0_share,
                "dolmino_phase_1_share": row.dolmino_phase_1_share,
                "uncheatable_percentile": row.uncheatable_percentile,
                "table9_percentile": row.table9_percentile,
                "selections": {
                    target: selection_tags.get((row.policy_id, target), []) for target in ("uncheatable", "table9")
                },
                "comments": {
                    target: comments_lookup.get((row.policy_id, target), "") for target in ("uncheatable", "table9")
                },
                "buckets": [
                    [
                        bucket.phase_0_weight,
                        bucket.phase_1_weight,
                        bucket.aggregate_weight,
                        bucket.simulated_epochs,
                    ]
                    for bucket in buckets.itertuples(index=False)
                ],
            }
        )
    headline_findings: dict[str, object] = {}
    for target, metric in TARGET_COLUMNS.items():
        global_target = selections[selections["target"].eq(target) & selections["selection_scope"].eq("global")]
        best_ids = set(global_target.loc[global_target["tail"].eq("best"), "policy_id"])
        worst_ids = set(global_target.loc[global_target["tail"].eq("worst"), "policy_id"])
        best = data.policies[data.policies["policy_id"].isin(best_ids)]
        worst = data.policies[data.policies["policy_id"].isin(worst_ids)]
        target_intuition = intuition[intuition["target"].eq(target)]
        corroborated = target_intuition[
            target_intuition["controlled_evidence_kind"].eq("direction-specific")
            & (
                target_intuition["tail_phase_direction"]
                == target_intuition["controlled_preference"].astype(str) + "-enriched"
            )
        ].sort_values("controlled_net_gain_bpb", ascending=False)
        conflicted = target_intuition[
            target_intuition["controlled_evidence_kind"].eq("direction-specific")
            & target_intuition["tail_phase_direction"].ne("mixed")
            & (
                target_intuition["tail_phase_direction"]
                != target_intuition["controlled_preference"].astype(str) + "-enriched"
            )
        ].sort_values("controlled_net_gain_bpb", ascending=False)
        headline_findings[target] = {
            "best_phase_tv": float(best["phase_tv"].median()),
            "worst_phase_tv": float(worst["phase_tv"].median()),
            "best_aggregate_tv": float(best["aggregate_tv_to_proportional"].median()),
            "worst_aggregate_tv": float(worst["aggregate_tv_to_proportional"].median()),
            "corroborated": list(corroborated["domain"]),
            "conflicted": list(conflicted["domain"]),
            "more_aggregate": list(
                target_intuition.nlargest(6, "median_series_best_minus_worst_aggregate_weight")["domain"]
            ),
            "less_aggregate": list(
                target_intuition.nsmallest(6, "median_series_best_minus_worst_aggregate_weight")["domain"]
            ),
            "best_bpb": float(best[metric].min()),
        }
    return {
        "policies": policy_records,
        "intuition": json_records(intuition),
        "broad_rule_stress": json_records(broad_rule_stress),
        "controlled_rules": json_records(controlled_rules),
        "controlled_pairs": controlled_pairs,
        "hybrid_winners": json_records(hybrid_winners),
        "hybrid_package": json_records(hybrid_package),
        "statistical_resolution": statistical_resolution,
        "domains": [
            {
                "domain": row.domain,
                "proportional": data.proportional[index],
            }
            for index, row in enumerate(data.semantics.itertuples(index=False))
        ],
        "summary": {
            "unique_policies": len(data.policies),
            "genuine_two_phase_policies": int(data.policies["phase_regime"].eq("genuine_two_phase").sum()),
            "observations": int(data.policies["observation_count"].sum()),
            "selected_policies": comments["policy_id"].nunique(),
            "proposal_series": data.policies["training_series"].nunique(),
            "phase_0_fraction": float(data.policies["phase_0_fraction"].median()),
            "phase_1_fraction": float(data.policies["phase_1_fraction"].median()),
            "repeat_noise_sd": repeat_noise_by_target(data.policies),
            "extreme_simulated_epoch_policy_count": int(data.policies["max_simulated_epoch"].gt(30.0).sum()),
            "headline_findings": headline_findings,
        },
    }


def build_html(payload: dict[str, object], d3_source: str) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Delphi 3e18 Phase-Order Field Guide</title>
<style>
:root {{
  --paper: #f5f0e5;
  --paper-light: #fffdf7;
  --ink: #173044;
  --muted: #63727a;
  --rule: #cfc5b2;
  --green: #2f855a;
  --yellow: #e1af38;
  --red: #c54b3c;
  --orange: #eb6a32;
  --blue: #2b6f91;
  --navy: #102b3b;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  color: var(--ink);
  background:
    linear-gradient(90deg, rgba(23,48,68,.035) 1px, transparent 1px) 0 0 / 56px 56px,
    linear-gradient(rgba(23,48,68,.035) 1px, transparent 1px) 0 0 / 56px 56px,
    var(--paper);
  font-family: "Avenir Next", "Gill Sans", sans-serif;
}}
header {{
  padding: 54px clamp(24px, 6vw, 92px) 40px;
  color: #f8f1df;
  background:
    radial-gradient(circle at 78% 20%, rgba(225,175,56,.28), transparent 28%),
    linear-gradient(125deg, #102b3b, #173f50 62%, #285a5b);
  border-bottom: 8px solid var(--orange);
}}
.eyebrow {{ color: #f5c552; font-size: 12px; letter-spacing: .18em; font-weight: 800; text-transform: uppercase; }}
h1, h2, h3 {{ font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif; margin: 0; }}
h1 {{ max-width: 1000px; font-size: clamp(42px, 7vw, 82px); line-height: .98; font-weight: 650; }}
.dek {{ max-width: 920px; margin: 24px 0 0; color: #d8e1df; font-size: 19px; line-height: 1.55; }}
.fact-strip {{
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px;
  background: rgba(255,255,255,.16); margin-top: 36px; max-width: 1050px;
}}
.fact {{ background: rgba(16,43,59,.62); padding: 18px; }}
.fact b {{ display: block; color: white; font: 600 28px/1 "Iowan Old Style", Georgia, serif; }}
.fact span {{ color: #bfcfcb; font-size: 12px; letter-spacing: .06em; text-transform: uppercase; }}
main {{ width: min(1560px, calc(100% - 36px)); margin: 30px auto 80px; }}
.section {{
  background: rgba(255,253,247,.94); border: 1px solid var(--rule);
  box-shadow: 0 18px 45px rgba(42,49,48,.08); margin: 24px 0; overflow: hidden;
}}
.section-heading {{ display: flex; gap: 28px; justify-content: space-between; align-items: end; padding: 28px 32px 18px; border-bottom: 1px solid var(--rule); }}
.section-heading h2 {{ font-size: clamp(27px, 3vw, 42px); }}
.section-heading p {{ max-width: 720px; margin: 0; color: var(--muted); line-height: 1.5; }}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
button, select, input {{
  font: inherit; color: var(--ink); background: #fffdf7; border: 1px solid #aeb4ad; padding: 9px 12px;
}}
button {{ cursor: pointer; font-weight: 700; }}
button.active {{ color: #fff; background: var(--navy); border-color: var(--navy); box-shadow: inset 4px 0 var(--orange); }}
.body-pad {{ padding: 24px 32px 32px; }}
.method-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); border-top: 1px solid var(--rule); }}
.method-card {{ padding: 22px 26px; border-right: 1px solid var(--rule); }}
.method-card:last-child {{ border-right: 0; }}
.method-card b {{ display: block; color: var(--orange); text-transform: uppercase; letter-spacing: .08em; font-size: 12px; margin-bottom: 7px; }}
.method-card p {{ margin: 0; line-height: 1.45; color: var(--muted); }}
.finding-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
.finding-card {{ border: 1px solid var(--rule); background: #fffdf7; padding: 24px; }}
.finding-card h3 {{ font-size: 29px; margin-bottom: 6px; }}
.finding-card .verdict {{ font: 600 20px/1.35 "Iowan Old Style", Georgia, serif; color: var(--navy); }}
.finding-card dl {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px 18px; margin: 20px 0; }}
.finding-card dt {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }}
.finding-card dd {{ margin: 2px 0 0; font-weight: 750; }}
.finding-card p {{ color: var(--muted); line-height: 1.5; }}
.bucket-list {{ display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }}
.bucket-list span {{ padding: 4px 7px; background: #eee6d5; font-size: 11px; border-radius: 2px; }}
.resolution-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
.resolution-card {{ border: 1px solid var(--rule); background: #fffdf7; }}
.resolution-card-head {{ padding: 20px 22px 16px; border-bottom: 1px solid var(--rule); }}
.resolution-card-head h3 {{ font-size: 28px; }}
.resolution-status {{
  display: inline-block; margin-top: 9px; padding: 5px 9px; color: #7c2d22; background: #f8dfd6;
  border-left: 3px solid var(--red); font-size: 11px; font-weight: 800; letter-spacing: .05em;
  text-transform: uppercase;
}}
.noise-table {{ width: 100%; font-size: 12px; }}
.noise-table th {{ position: static; padding: 8px 10px; font-size: 10px; }}
.noise-table td {{ padding: 9px 10px; }}
.noise-source-detail {{ color: var(--muted); font-size: 10px; }}
.resolution-summary {{ padding: 16px 22px 20px; color: var(--muted); line-height: 1.5; }}
.resolution-summary b {{ color: var(--ink); }}
.resolution-contract {{
  margin-top: 18px; padding: 16px 18px; color: #d7e1de; background: var(--navy); line-height: 1.5;
  border-top: 4px solid var(--yellow);
}}
.resolution-contract b {{ color: #f5c552; }}
.plot {{ width: 100%; min-height: 570px; }}
.axis text {{ fill: var(--muted); font-size: 12px; }}
.axis path, .axis line {{ stroke: #9aa5a4; }}
.grid line {{ stroke: #d9d2c5; stroke-dasharray: 3 5; }}
.grid path {{ display: none; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 18px; color: var(--muted); font-size: 13px; margin: 8px 0 0; }}
.legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}
.workspace {{ display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(420px, .85fr); gap: 0; }}
.workspace > div {{ min-width: 0; }}
.policy-detail {{ border-left: 1px solid var(--rule); background: #f8f3e8; }}
.detail-head {{ padding: 24px 26px 16px; border-bottom: 1px solid var(--rule); }}
.detail-head h3 {{ font-size: 27px; overflow-wrap: anywhere; }}
.detail-meta {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: var(--rule); margin-top: 18px; }}
.detail-meta div {{ background: #fffdf7; padding: 12px; }}
.detail-meta b {{ display: block; font: 600 22px/1.1 "Iowan Old Style", Georgia, serif; }}
.detail-meta span {{ color: var(--muted); font-size: 11px; text-transform: uppercase; }}
.comment {{ padding: 18px 26px; line-height: 1.55; color: #354a54; border-bottom: 1px solid var(--rule); }}
.anatomy-wrap {{ overflow: auto; max-height: 760px; padding: 18px 10px 24px; }}
.tooltip {{
  position: fixed; pointer-events: none; z-index: 20; max-width: 420px;
  background: rgba(16,43,59,.96); color: #fff; padding: 12px 14px; border-left: 4px solid #f5c552;
  opacity: 0; font-size: 13px; line-height: 1.45; box-shadow: 0 10px 30px rgba(0,0,0,.2);
}}
.matrix-wrap {{ overflow-x: auto; padding: 12px 24px 30px; }}
.matrix-note {{ color: var(--muted); margin: 0 8px 14px; line-height: 1.45; }}
.rule-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
.rule-card {{ border: 1px solid var(--rule); padding: 20px; background: #fffdf7; }}
.rule-card h3 {{ font-size: 22px; }}
.rule-card p {{ color: var(--muted); line-height: 1.45; }}
.rule-score {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin: 14px 0; }}
.rule-score div {{ background: #f1ecdf; padding: 10px; }}
.rule-score b {{ display: block; font: 600 20px/1 "Iowan Old Style", Georgia, serif; }}
.rule-score span {{ font-size: 10px; text-transform: uppercase; color: var(--muted); }}
.rule-subhead {{ margin-top: 15px; color: var(--orange); font-size: 11px; font-weight: 800; letter-spacing: .09em; text-transform: uppercase; }}
.controlled-counterexample {{ border-left: 4px solid var(--navy); padding: 11px 12px; background: #e8f0ef; margin-top: 8px; font-size: 13px; line-height: 1.45; }}
.controlled-counterexample-list {{ display: grid; gap: 7px; margin-top: 9px; }}
.pair-open {{
  display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 4px 14px; width: 100%;
  text-align: left; padding: 9px 10px; background: #fffdf7; border-color: #9eaaa7;
}}
.pair-open:hover {{ background: #fff1cf; border-color: var(--navy); }}
.pair-open b {{ overflow-wrap: anywhere; }}
.pair-open span {{ color: var(--muted); font-size: 11px; font-weight: 500; }}
.pair-open code {{ grid-row: 1 / span 2; grid-column: 2; align-self: center; color: var(--blue); font-size: 11px; }}
.counterexample {{ border-left: 4px solid var(--red); padding: 9px 12px; background: #f8ebe6; margin-top: 8px; font-size: 13px; }}
.conformer {{ border-left-color: var(--yellow); background: #fbf3d8; }}
.pair-dialog {{
  width: min(1120px, calc(100vw - 40px)); max-height: calc(100vh - 40px); padding: 0;
  color: var(--ink); background: var(--paper-light); border: 1px solid var(--rule);
  box-shadow: 0 28px 80px rgba(16,43,59,.28);
}}
.pair-dialog::backdrop {{ background: rgba(16,43,59,.70); backdrop-filter: blur(2px); }}
.pair-dialog-head {{
  position: sticky; top: 0; z-index: 3; display: flex; justify-content: space-between; gap: 24px;
  align-items: start; padding: 22px 26px 18px; color: #f8f1df; background: var(--navy);
  border-bottom: 5px solid var(--orange);
}}
.pair-dialog-head h2 {{ font-size: clamp(25px, 3vw, 40px); }}
.pair-dialog-head p {{ margin: 5px 0 0; color: #c9d4d2; }}
.pair-close {{ color: white; background: transparent; border-color: rgba(255,255,255,.45); }}
.pair-dialog-body {{ padding: 22px 26px 30px; }}
.pair-anchor-tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }}
.pair-trace {{
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 20px;
  padding: 14px 16px; background: #f1ecdf; border: 1px solid var(--rule); font-size: 12px;
}}
.pair-trace div {{ min-width: 0; overflow-wrap: anywhere; }}
.pair-trace b {{ display: block; color: var(--muted); font-size: 10px; letter-spacing: .06em; text-transform: uppercase; }}
.pair-metrics {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 1px; margin: 14px 0; background: var(--rule); }}
.pair-metrics div {{ padding: 12px 14px; background: #fffdf7; }}
.pair-metrics b {{ display: block; font: 600 21px/1.1 "Iowan Old Style", Georgia, serif; }}
.pair-metrics span {{ color: var(--muted); font-size: 10px; letter-spacing: .04em; text-transform: uppercase; }}
.pair-resolution {{
  margin: 14px 0; padding: 13px 15px; color: #493c35; background: #f8ebe6; border-left: 4px solid var(--red);
  font-size: 12px; line-height: 1.5;
}}
.pair-resolution b {{ color: #7c2d22; }}
.pair-anatomy-note {{ color: var(--muted); line-height: 1.45; margin: 12px 0 2px; }}
.pair-anatomy {{ min-width: 720px; }}
.pair-anatomy-wrap {{ overflow: auto; border: 1px solid var(--rule); background: #fffdf7; }}
.recipe-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
.recipe-card {{ padding: 24px; color: #eef4ef; background: linear-gradient(145deg, #173f50, #285a5b); border-top: 5px solid #f5c552; }}
.recipe-card h3 {{ font-size: 28px; color: white; }}
.recipe-card p {{ line-height: 1.5; color: #d3e0dc; }}
.recipe-card b {{ color: #f5c552; }}
.recipe-card ul {{ margin: 12px 0 0; padding-left: 20px; line-height: 1.55; }}
.policy-table-wrap {{ max-height: 640px; overflow: auto; border: 1px solid var(--rule); }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ position: sticky; top: 0; background: var(--navy); color: white; text-align: left; padding: 10px; z-index: 1; }}
td {{ border-bottom: 1px solid #ddd5c7; padding: 9px 10px; vertical-align: top; }}
tbody tr {{ cursor: pointer; }}
tbody tr:hover, tbody tr.selected {{ background: #fff1cf; }}
.pill {{ display: inline-block; padding: 2px 7px; margin: 1px 3px 1px 0; border-radius: 10px; background: #e8e1d2; font-size: 10px; }}
.footnote {{ padding: 22px 32px; color: var(--muted); line-height: 1.55; border-top: 1px solid var(--rule); }}
@media (max-width: 980px) {{
  .fact-strip, .method-grid, .finding-grid, .resolution-grid, .workspace, .rule-grid, .recipe-grid {{ grid-template-columns: 1fr; }}
  .policy-detail {{ border-left: 0; border-top: 1px solid var(--rule); }}
  .method-card {{ border-right: 0; border-bottom: 1px solid var(--rule); }}
  .pair-trace, .pair-metrics {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<header>
  <div class="eyebrow">Delphi 3e18 / 39-bucket qualitative audit</div>
  <h1>Phase-order field guide</h1>
  <p class="dek">A controlled-and-associative audit of the best and worst two-phase mixtures. The report
  separates what fixed-aggregate interventions identify from what broad observed tails merely suggest.</p>
  <div class="fact-strip">
    <div class="fact"><b id="fact-policies"></b><span>two-phase-labeled policies</span></div>
    <div class="fact"><b id="fact-genuine"></b><span>phase TV at least 0.01</span></div>
    <div class="fact"><b id="fact-selected"></b><span>individually commented</span></div>
    <div class="fact"><b>39</b><span>data buckets</span></div>
  </div>
</header>
<main>
  <section class="section">
    <div class="section-heading">
      <div><div class="eyebrow">01 / evidence contract</div><h2>Three layers, three claims</h2></div>
      <p>The fixed-aggregate audit can identify order. Broad tails expose recurring anatomy and
      counterexamples, but aggregate mixture quality and proposal provenance remain confounded.</p>
    </div>
    <div class="method-grid">
      <div class="method-card"><b>Controlled</b><p>Antithetic and domain-fiber interventions hold aggregate
      exposure fixed. Only these support tentative early/late claims.</p></div>
      <div class="method-card"><b>Series balanced</b><p>Best-versus-worst quartiles are computed inside each
      proposal family, then summarized equally across families.</p></div>
      <div class="method-card"><b>Global tails</b><p>The absolute frontier and failure cases reveal plausible
      schedules, overload, underexposure, and non-causal heuristic violations.</p></div>
    </div>
    <div class="footnote"><b>Study context.</b> All policies use the same 358M-parameter Delphi 3e18
    configuration, about 1.58B training tokens, a 79.8138/20.1862 WSD phase split, and the same 39 buckets.
    “Simulated epochs” rescale aggregate weights to a 6.325T-token production budget; they are not physical
    repetitions in these runs.</div>
  </section>

  <section class="section">
    <div class="section-heading">
      <div><div class="eyebrow">02 / synthesis</div><h2>What survives broader scrutiny</h2></div>
      <p>The dominant split is objective-specific aggregate composition. Useful phase order is local,
      comparatively small, and bucket-specific; aggressive asymmetry is not a substitute for choosing a
      good aggregate mixture.</p>
    </div>
    <div class="body-pad"><div id="finding-grid" class="finding-grid"></div></div>
  </section>

  <section class="section">
    <div class="section-heading">
      <div><div class="eyebrow">03 / statistical resolution</div><h2>Counterexamples are hypotheses, not discoveries</h2></div>
      <p>The antithetic design removes aggregate-mixture confounding, but one shared-seed triple per bucket
      does not estimate its paired uncertainty. The screens below put the observed effects against every
      available repeat scale without upgrading them into formal tests.</p>
    </div>
    <div class="body-pad">
      <div id="resolution-grid" class="resolution-grid"></div>
      <div class="resolution-contract" id="resolution-contract"></div>
    </div>
  </section>

  <section class="section">
    <div class="section-heading">
      <div><div class="eyebrow">04 / archive landscape</div><h2>Where the tails live</h2></div>
      <div class="toolbar">
        <button class="objective active" data-target="uncheatable">Uncheatable</button>
        <button class="objective" data-target="table9">Table-9</button>
        <select id="series-filter"><option value="">All proposal series</option></select>
      </div>
    </div>
    <div class="workspace">
      <div class="body-pad">
        <div id="scatter" class="plot"></div>
        <div class="legend">
          <span><i style="background:#2f855a"></i>best global tail</span>
          <span><i style="background:#c54b3c"></i>worst global tail</span>
          <span><i style="background:#b8b09f"></i>other policy</span>
          <span>Circle size grows with phase TV. Click any policy to inspect all 39 buckets.</span>
        </div>
      </div>
      <div class="policy-detail">
        <div class="detail-head">
          <div class="eyebrow">selected policy</div>
          <h3 id="detail-title">Select a point</h3>
          <div class="detail-meta" id="detail-meta"></div>
        </div>
        <div class="comment" id="detail-comment">The qualitative audit comment will appear here.</div>
        <div class="anatomy-wrap"><div id="anatomy"></div></div>
      </div>
    </div>
  </section>

  <section class="section">
    <div class="section-heading">
      <div><div class="eyebrow">05 / bucket map</div><h2>Per-bucket working intuition</h2></div>
      <div class="toolbar">
        <button class="objective active" data-target="uncheatable">Uncheatable</button>
        <button class="objective" data-target="table9">Table-9</button>
      </div>
    </div>
    <div class="matrix-wrap">
      <p class="matrix-note"><b>Controlled direction</b> identifies the local sign contrast within the audited
      fixed-aggregate fiber. It does not establish a universal bucket effect across anchors. The remaining
      columns are proposal-series-balanced associations: positive phase enrichment means better tails put the
      bucket later; positive aggregate enrichment means they use more of it overall. Hover every cell for the
      exact interpretation.</p>
      <div id="bucket-matrix"></div>
    </div>
  </section>

  <section class="section">
    <div class="section-heading">
      <div><div class="eyebrow">06 / rules under pressure</div><h2>Order reversals before archive heuristics</h2></div>
      <p>Only the fixed-aggregate antithetic pairs qualify as observed phase-order counterexamples; none is
      statistically resolved. Broad-tail violations are retained underneath as associative evidence that a
      rule is not sufficient across arbitrary mixtures.</p>
    </div>
    <div class="body-pad"><div id="rule-grid" class="rule-grid"></div></div>
  </section>

  <section class="section">
    <div class="section-heading">
      <div><div class="eyebrow">07 / candidate sketches</div><h2>What a manual intervention should test</h2></div>
      <p>These are preregistration sketches, not inferred optima. Keep the validated aggregate fixed and
      perturb only the phase contrast so the experiment identifies order rather than rediscovering mixture
      composition.</p>
    </div>
    <div class="body-pad">
      <div class="recipe-grid">
        <article class="recipe-card">
          <div class="eyebrow">Uncheatable / stronger evidence</div>
          <h3>Late technical package with donor sign checks</h3>
          <p><b>Anchor:</b> the validated unch05-like tied aggregate. <b>Contrast:</b> phase TV 0.10-0.25,
          with a 0.33 stress point only if budget allows.</p>
          <ul>
            <li>Core late arm: literature-high and science/technology-high. Their controlled signs exceed the
            archive median repeat-coordinate SD and broad tails corroborate, but neither is statistically
            resolved under the available repeat designs.</li>
            <li>Weaker late arms: olmOCR PDFs and arXiv. Their measured controlled signs are below that noise
            scale and require repeats.</li>
            <li>Add StackEdu/FIM as a joint package arm; FIM's preferred isolated order still loses to tied,
            so their isolated effects are not established.</li>
            <li>Test synthetic math early. Give synthetic QA, common-crawl HQ, education/jobs-high, and
            health-high paired early/late sign ablations rather than assuming they are early donors.</li>
            <li>Keep aggregate exposure unchanged and avoid production-budget-normalized per-bucket
            exposures outside the validated frontier range.</li>
          </ul>
        </article>
        <article class="recipe-card">
          <div class="eyebrow">Table-9 / noisier evidence</div>
          <h3>Package screen plus QA sign ablation</h3>
          <p><b>Anchor:</b> a validated t9b075/t9s05-like tied aggregate. <b>Contrast:</b> phase TV at most
          0.25, evaluated with repeats because single-run noise is comparable to the expected gain.</p>
          <ul>
            <li>Package arm: move StackEdu/FIM and synthetic code/math/thinking later, with common-crawl HQ
            earlier; compare tied and sign-reversed packages. Every constituent conflicts with, or is not
            isolated by, at least one controlled comparison.</li>
            <li>Do not interpret any package constituent as independently beneficial from this screen.</li>
            <li>Omit FineMath from the package or give it a separate sign ablation; no evidence tier supports
            placing it late consistently.</li>
            <li>Give synthetic QA a three-way tied/early/late ablation: its one-bucket fiber favors late, while
            selected package winners often put it early.</li>
          </ul>
        </article>
      </div>
    </div>
    <div class="footnote">Archive guardrails: no pool-wide Dolmino-late rule. Balanced/unstructured TV=0.50
    interventions were uniformly harmful, but structured TV=0.41-0.50 schedules occasionally reached the top
    decile or beat tied; 0.10-0.25 is the primary package range, not a universal hard cap. All six observed
    policies above 30 production-budget-normalized per-bucket simulated epochs were at or above the 97.5th
    percentile (worse) on both objectives. These sparse constraints are deployment hygiene, not proof that the
    surrogate surface is correct.</div>
  </section>

  <section class="section">
    <div class="section-heading">
      <div><div class="eyebrow">08 / audit ledger</div><h2>Every commented policy</h2></div>
      <div class="toolbar">
        <input id="policy-search" placeholder="Search policy or series">
        <select id="tail-filter">
          <option value="">Best and worst</option><option value="best">Best only</option><option value="worst">Worst only</option>
        </select>
      </div>
    </div>
    <div class="body-pad"><div class="policy-table-wrap"><table>
      <thead><tr><th>Policy</th><th>Selection</th><th>Uncheatable</th><th>Table-9</th><th>Phase TV</th><th>Series</th></tr></thead>
      <tbody id="policy-table"></tbody>
    </table></div></div>
    <div class="footnote">Lower BPB is better. Repeated seeds are collapsed by exact mixture hash before tail
    selection. The broad archive is an intervention-designed development set, not an IID sample. Observed
    frontiers include winner's-curse noise; the report never upgrades broad association into a causal claim.</div>
  </section>
</main>
<dialog class="pair-dialog" id="pair-dialog">
  <div class="pair-dialog-head">
    <div>
      <div class="eyebrow">controlled fixed-aggregate antithetic pair</div>
      <h2 id="pair-dialog-title"></h2>
      <p id="pair-dialog-subtitle"></p>
    </div>
    <button class="pair-close" id="pair-close" type="button">Close</button>
  </div>
  <div class="pair-dialog-body">
    <div class="pair-anchor-tabs" id="pair-anchor-tabs"></div>
    <div class="pair-trace" id="pair-trace"></div>
    <div class="pair-metrics" id="pair-metrics"></div>
    <div class="pair-resolution" id="pair-resolution"></div>
    <p class="pair-anatomy-note">
      <i>+d</i> places the focal bucket later and compensates across every other bucket; <i>-d</i> reverses that
      contrast. Within each phase, the solid orange bar is <i>+d</i> and the navy outline is <i>-d</i>.
      All three columns share one scale. Hover a bucket for exact weights and aggregate exposure.
    </p>
    <div class="pair-anatomy-wrap"><div class="pair-anatomy" id="pair-anatomy"></div></div>
  </div>
</dialog>
<div class="tooltip" id="tooltip"></div>
<script>{d3_source}</script>
<script>
const DATA = {payload_json};
DATA.policies.forEach(policy => {{
  policy.buckets = policy.buckets.map((values, index) => ({{
    domain: DATA.domains[index].domain,
    proportional: DATA.domains[index].proportional,
    w0: values[0],
    w1: values[1],
    aggregate: values[2],
    simulated_epochs: values[3]
  }}));
}});
const targetMeta = {{
  uncheatable: {{label: "Uncheatable", key: "uncheatable"}},
  table9: {{label: "Table-9", key: "table9"}}
}};
let target = "uncheatable";
let selectedPolicy = null;
const tooltip = d3.select("#tooltip");
const fmt = d3.format(".6f");
const signed = d3.format("+.6f");
const pct = d3.format("+.2%");
const weightFmt = d3.format(".2%");
const targetAnchor = {{
  uncheatable: "uncheatable_frontier",
  table9: "table9_frontier"
}};
const pairByDomain = new Map(DATA.controlled_pairs.map(pair => [pair.domain, pair]));
const pairDialog = document.getElementById("pair-dialog");
let activePairDomain = null;
let activePairAnchor = null;

d3.select("#fact-policies").text(d3.format(",")(DATA.summary.unique_policies));
d3.select("#fact-genuine").text(d3.format(",")(DATA.summary.genuine_two_phase_policies));
d3.select("#fact-selected").text(d3.format(",")(DATA.summary.selected_policies));

function bucketChips(values) {{
  return `<div class="bucket-list">${{values.map(value => `<span>${{value}}</span>`).join("")}}</div>`;
}}

function escapeHtml(value) {{
  const replacements = {{"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"}};
  return String(value).replace(/[&<>"']/g, character => replacements[character]);
}}
const findingCopy = {{
  uncheatable: "Broad capability favors a web/science/document aggregate. The strongest recurring late placements are literature, science/technology, PDFs, and arXiv; synthetic math is the clearest early placement.",
  table9: "Task loss favors a synthetic QA/math/reasoning and code aggregate. Only synthetic QA has a measured direction-specific controlled late sign, and it is just 0.10x the repeat-noise scale; most phase signals remain associative or tied-preferred."
}};
d3.select("#finding-grid").selectAll("article")
  .data(["uncheatable", "table9"])
  .join("article").attr("class", "finding-card")
  .html(name => {{
    const d = DATA.summary.headline_findings[name];
    return `<div class="eyebrow">${{targetMeta[name].label}}</div>` +
      `<h3>${{fmt(d.best_bpb)}} lowest observed (unrepeated)</h3>` +
      `<p>Median repeat-coordinate SD is ${{fmt(DATA.summary.repeat_noise_sd[name])}}; do not rank the
      near-frontier observations more finely than that noise scale.</p>` +
      `<p class="verdict">${{findingCopy[name]}}</p>` +
      `<dl><div><dt>Best-tail median phase TV</dt><dd>${{d.best_phase_tv.toFixed(3)}}</dd></div>` +
      `<div><dt>Worst-tail median phase TV</dt><dd>${{d.worst_phase_tv.toFixed(3)}}</dd></div>` +
      `<div><dt>Best-tail aggregate TV</dt><dd>${{d.best_aggregate_tv.toFixed(3)}}</dd></div>` +
      `<div><dt>Worst-tail aggregate TV</dt><dd>${{d.worst_aggregate_tv.toFixed(3)}}</dd></div></dl>` +
      `<p><b>More aggregate mass in better series tails</b>${{bucketChips(d.more_aggregate)}}</p>` +
      `<p><b>Less aggregate mass in better series tails</b>${{bucketChips(d.less_aggregate)}}</p>` +
      `<p><b>Controlled direction corroborated by broad tails</b>${{bucketChips(d.corroborated.length ? d.corroborated : ["none"])}}</p>` +
      `<p><b>Selected winners: late enrichment over candidate base rate</b>${{bucketChips(
        DATA.hybrid_package.filter(x => x.target === name)
          .sort((a,b) => d3.descending(a.winner_minus_candidate_later_rate,b.winner_minus_candidate_later_rate))
          .slice(0,6).map(x => x.domain)
      )}}</p>` +
      `<p><b>Selected winners: early enrichment over candidate base rate</b>${{bucketChips(
        DATA.hybrid_package.filter(x => x.target === name)
          .sort((a,b) => d3.descending(a.winner_minus_candidate_earlier_rate,b.winner_minus_candidate_earlier_rate))
          .slice(0,6).map(x => x.domain)
      )}}</p>` +
      (d.conflicted.length ? `<p><b>Controlled/tail conflict</b>${{bucketChips(d.conflicted)}}</p>` : "");
  }});

function noiseSourceRow(source) {{
  const range = source.sd_min == null ? "" :
    `<div class="noise-source-detail">observed SD range ${{fmt(source.sd_min)}}-${{fmt(source.sd_max)}}</div>`;
  return `<tr><td><b>${{source.label}}</b><div class="noise-source-detail">n=${{source.n}} · ${{source.detail}}</div>${{range}}</td>` +
    `<td>${{fmt(source.sd)}}</td><td>${{fmt(source.rough_95_difference_threshold)}}</td></tr>`;
}}

d3.select("#resolution-grid").selectAll("article")
  .data(["uncheatable", "table9"])
  .join("article").attr("class", "resolution-card")
  .html(name => {{
    const resolution = DATA.statistical_resolution.targets[name];
    const nominal = resolution.nominal_pair_gap_count_archive_noise;
    return `<div class="resolution-card-head"><div class="eyebrow">${{targetMeta[name].label}}</div>` +
      `<h3>No resolved counterexample</h3><span class="resolution-status">descriptive evidence only</span></div>` +
      `<table class="noise-table"><thead><tr><th>Noise source</th><th>SD</th><th>Rough 95% difference</th></tr></thead>` +
      `<tbody>${{resolution.noise_sources.map(noiseSourceRow).join("")}}</tbody></table>` +
      `<div class="resolution-summary">Across <b>${{resolution.counterexample_domain_count}} unique observed counterexample buckets</b>, ` +
      `<b>${{resolution.direction_specific_gain_count_clearing_any_threshold}}</b> direction-specific reverse-order gains over tied ` +
      `clear even the smallest displayed threshold. ` +
      `<b>${{nominal}}</b> +d/-d order gap${{nominal === 1 ? "" : "s"}} clear an uncorrected normal screen only under the archive-median SD; ` +
      `<b>${{resolution.holm_pair_gap_count_archive_noise}}</b> survive a 39-bucket rough Holm screen. ` +
      `The available independent-run thresholds span <b>${{fmt(resolution.rough_95_threshold_min)}}-${{fmt(resolution.rough_95_threshold_max)}} BPB</b>.</div>`;
  }});

d3.select("#resolution-contract").html(
  `<b>What these numbers do and do not mean.</b> ${{DATA.statistical_resolution.contract}} ` +
  `${{DATA.statistical_resolution.shared_seed_caveat}} ` +
  `${{DATA.statistical_resolution.multiplicity_caveat}} ` +
  `Formal resolution requires repeating the exact +d, -d, and tied triple across seeds and testing the paired ` +
  `order effect O(d) = [Y(+d) - Y(-d)] / 2 separately from the even asymmetry cost ` +
  `C(d) = [Y(+d) + Y(-d)] / 2 - Y(0).`
);

const allSeries = Array.from(new Set(DATA.policies.flatMap(d => d.series.split(" | ")))).filter(Boolean).sort();
d3.select("#series-filter").selectAll("option.series").data(allSeries).join("option")
  .attr("class", "series").attr("value", d => d).text(d => d);

function globalTail(policy, targetName, tail) {{
  return policy.selections[targetName].some(d => d.scope === "global" && d.tail === tail);
}}
function targetValue(policy) {{ return policy[targetMeta[target].key]; }}
function showTooltip(event, body) {{
  tooltip.html(body).style("opacity", 1)
    .style("left", Math.min(window.innerWidth - 440, event.clientX + 14) + "px")
    .style("top", Math.max(10, event.clientY - 18) + "px");
}}
function hideTooltip() {{ tooltip.style("opacity", 0); }}

function renderScatter() {{
  const host = d3.select("#scatter");
  host.selectAll("*").remove();
  const series = d3.select("#series-filter").property("value");
  const rows = DATA.policies.filter(d => !series || d.series.split(" | ").includes(series));
  const width = Math.max(620, host.node().clientWidth);
  const height = 570;
  const margin = {{top: 26, right: 24, bottom: 58, left: 74}};
  const svg = host.append("svg").attr("viewBox", `0 0 ${{width}} ${{height}}`);
  const x = d3.scaleLinear()
    .domain(d3.extent(rows, d => d.aggregate_tv)).nice()
    .range([margin.left, width - margin.right]);
  const yExtent = d3.extent(rows, targetValue);
  const yPad = Math.max(.002, (yExtent[1] - yExtent[0]) * .04);
  const y = d3.scaleLinear().domain([yExtent[0] - yPad, yExtent[1] + yPad]).nice()
    .range([margin.top, height - margin.bottom]);
  svg.append("g").attr("class", "grid").attr("transform", `translate(${{margin.left}},0)`)
    .call(d3.axisLeft(y).tickSize(-(width - margin.left - margin.right)).tickFormat(""));
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${{height - margin.bottom}})`)
    .call(d3.axisBottom(x).ticks(7));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${{margin.left}},0)`)
    .call(d3.axisLeft(y).ticks(8));
  svg.append("text").attr("x", (margin.left + width - margin.right) / 2).attr("y", height - 14)
    .attr("text-anchor", "middle").attr("font-weight", 700).text("Aggregate TV from proportional");
  svg.append("text").attr("transform", "rotate(-90)").attr("x", -(margin.top + height - margin.bottom) / 2)
    .attr("y", 18).attr("text-anchor", "middle").attr("font-weight", 700)
    .text(`${{targetMeta[target].label}} BPB (lower is better)`);
  svg.append("text").attr("x", margin.left + 8).attr("y", margin.top + 12)
    .attr("font-size", 10).attr("fill", "#63727a").text("better ↑");
  const color = d => globalTail(d, target, "best") ? "#2f855a" :
    globalTail(d, target, "worst") ? "#c54b3c" : "#b8b09f";
  const points = svg.append("g").selectAll("circle").data(rows, d => d.policy_id).join("circle")
    .attr("cx", d => x(d.aggregate_tv)).attr("cy", d => y(targetValue(d)))
    .attr("r", d => 3 + 8 * Math.sqrt(d.phase_tv))
    .attr("fill", color).attr("fill-opacity", d => globalTail(d, target, "best") || globalTail(d, target, "worst") ? .88 : .27)
    .attr("stroke", d => selectedPolicy && selectedPolicy.policy_id === d.policy_id ? "#102b3b" : "white")
    .attr("stroke-width", d => selectedPolicy && selectedPolicy.policy_id === d.policy_id ? 3 : .7)
    .on("mousemove", (event, d) => showTooltip(event,
      `<b>${{d.label}}</b><br>${{targetMeta[target].label}} ${{fmt(targetValue(d))}}<br>` +
      `Uncheatable ${{fmt(d.uncheatable)}} / Table-9 ${{fmt(d.table9)}}<br>` +
      `phase TV ${{d.phase_tv.toFixed(3)}}; aggregate TV ${{d.aggregate_tv.toFixed(3)}}<br>` +
      `<span style="color:#c9d4d2">${{d.series}}</span>`))
    .on("mouseleave", hideTooltip)
    .on("click", (_, d) => selectPolicy(d));
  points.filter(d => globalTail(d, target, "best") || globalTail(d, target, "worst")).raise();
}}

function selectPolicy(policy) {{
  selectedPolicy = policy;
  d3.select("#detail-title").text(policy.label);
  d3.select("#detail-meta").html(`
    <div><b>${{fmt(policy.uncheatable)}}</b><span>Uncheatable</span></div>
    <div><b>${{fmt(policy.table9)}}</b><span>Table-9</span></div>
    <div><b>${{policy.phase_tv.toFixed(3)}}</b><span>Phase TV</span></div>
    <div><b>${{policy.aggregate_tv.toFixed(3)}}</b><span>Aggregate TV</span></div>
    <div><b>${{policy.max_simulated_epoch.toFixed(2)}}</b><span>Max sim. epochs</span></div>
    <div><b>${{policy.observation_count}}</b><span>Seed observations</span></div>`);
  const comment = policy.comments[target] || "This policy is outside the individually commented tail selection. Its full anatomy is still shown.";
  d3.select("#detail-comment").html(`<b>${{policy.series}}</b><br>${{comment}}`);
  renderAnatomy(policy);
  renderScatter();
  renderPolicyTable();
}}

function renderAnatomy(policy) {{
  const host = d3.select("#anatomy");
  host.selectAll("*").remove();
  const rows = policy.buckets.slice().sort((a, b) => d3.descending(
    Math.max(a.w0, a.w1, a.aggregate, a.proportional),
    Math.max(b.w0, b.w1, b.aggregate, b.proportional)));
  const width = Math.max(520, host.node().clientWidth || 520);
  const rowHeight = 22, top = 45, left = 210, right = 20;
  const height = top + rows.length * rowHeight + 34;
  const columns = [
    {{key:"w0", label:"Phase 0", color:"#eb6a32"}},
    {{key:"w1", label:"Phase 1", color:"#2b8a73"}},
    {{key:"aggregate", label:"Aggregate", color:"#e1af38"}},
    {{key:"proportional", label:"Proportional", color:"#8798a1"}}
  ];
  const colWidth = (width - left - right) / columns.length;
  const maxValue = d3.max(rows, d => Math.max(d.w0, d.w1, d.aggregate, d.proportional));
  const scale = d3.scaleLinear().domain([0, maxValue]).range([0, colWidth - 28]);
  const svg = host.append("svg").attr("viewBox", `0 0 ${{width}} ${{height}}`);
  columns.forEach((column, ci) => {{
    svg.append("text").attr("x", left + ci * colWidth + 4).attr("y", 19)
      .attr("font-weight", 800).attr("fill", column.color).text(column.label);
  }});
  const row = svg.selectAll("g.bucket-row").data(rows).join("g").attr("class", "bucket-row")
    .attr("transform", (_, i) => `translate(0,${{top + i * rowHeight}})`)
    .on("mousemove", (event, d) => showTooltip(event,
      `<b>${{d.domain}}</b><br>phase 0 ${{weightFmt(d.w0)}} / phase 1 ${{weightFmt(d.w1)}}<br>` +
      `aggregate ${{weightFmt(d.aggregate)}} / proportional ${{weightFmt(d.proportional)}}<br>` +
      `simulated exposure ${{d.simulated_epochs.toFixed(2)}} epochs`))
    .on("mouseleave", hideTooltip);
  row.append("text").attr("x", left - 8).attr("y", 12).attr("text-anchor", "end")
    .attr("font-size", 10.5).text(d => d.domain.replace("dolma3_cc/", "cc/"));
  columns.forEach((column, ci) => {{
    row.append("rect").attr("x", left + ci * colWidth).attr("y", 1).attr("height", 13)
      .attr("width", d => scale(d[column.key])).attr("fill", column.color).attr("fill-opacity", .88);
  }});
  svg.append("text").attr("x", left).attr("y", height - 9).attr("fill", "#63727a").attr("font-size", 11)
    .text("Sorted by maximum displayed weight. Hover for weights and production-budget-normalized exposure.");
}}

function renderPairAnatomy(pair, focalDomain) {{
  const host = d3.select("#pair-anatomy");
  host.selectAll("*").remove();
  const rows = pair.buckets.slice().sort((a, b) => {{
    if (a.domain === focalDomain) return -1;
    if (b.domain === focalDomain) return 1;
    const aContrast = Math.max(Math.abs(a.plus_w1 - a.plus_w0), Math.abs(a.minus_w1 - a.minus_w0));
    const bContrast = Math.max(Math.abs(b.plus_w1 - b.plus_w0), Math.abs(b.minus_w1 - b.minus_w0));
    return d3.descending(aContrast, bContrast) ||
      d3.descending(Math.max(a.plus_w0, a.plus_w1, a.minus_w0, a.minus_w1), Math.max(b.plus_w0, b.plus_w1, b.minus_w0, b.minus_w1));
  }});
  const width = Math.max(760, host.node().clientWidth || 760);
  const rowHeight = 18, top = 48, left = 235, right = 20;
  const height = top + rows.length * rowHeight + 38;
  const columns = [
    {{plusKey: "plus_w0", minusKey: "minus_w0", label: "Phase 0"}},
    {{plusKey: "plus_w1", minusKey: "minus_w1", label: "Phase 1"}},
    {{plusKey: "center_weight", minusKey: null, label: "Tied aggregate"}}
  ];
  const colWidth = (width - left - right) / columns.length;
  const maxValue = d3.max(rows, row =>
    Math.max(row.plus_w0, row.plus_w1, row.minus_w0, row.minus_w1, row.center_weight)) || 1;
  const scale = d3.scaleLinear().domain([0, maxValue]).range([0, colWidth - 28]);
  const svg = host.append("svg").attr("viewBox", `0 0 ${{width}} ${{height}}`);

  columns.forEach((_, columnIndex) => {{
    svg.append("rect").attr("x", left + columnIndex * colWidth).attr("y", 0).attr("width", colWidth)
      .attr("height", height - 28).attr("fill", columnIndex % 2 ? "#f4f0e7" : "#faf7ef");
  }});
  columns.forEach((column, columnIndex) => {{
    svg.append("text").attr("x", left + columnIndex * colWidth + 8).attr("y", 22)
      .attr("font-weight", 800).attr("fill", "#173044").text(column.label);
  }});
  svg.append("text").attr("x", left + 8).attr("y", 39)
    .attr("font-size", 10).attr("fill", "#63727a").text("solid +d later · outline -d earlier");
  svg.append("text").attr("x", left + colWidth + 8).attr("y", 39)
    .attr("font-size", 10).attr("fill", "#63727a").text("solid +d later · outline -d earlier");
  svg.append("text").attr("x", left + 2 * colWidth + 8).attr("y", 39)
    .attr("font-size", 10).attr("fill", "#63727a").text("same aggregate");

  const row = svg.selectAll("g.pair-bucket-row").data(rows).join("g").attr("class", "pair-bucket-row")
    .attr("transform", (_, index) => `translate(0,${{top + index * rowHeight}})`)
    .on("mousemove", (event, bucket) => showTooltip(event,
      `<b>${{escapeHtml(bucket.domain)}}</b><br>` +
      `+d phase 0 ${{weightFmt(bucket.plus_w0)}} / phase 1 ${{weightFmt(bucket.plus_w1)}}<br>` +
      `-d phase 0 ${{weightFmt(bucket.minus_w0)}} / phase 1 ${{weightFmt(bucket.minus_w1)}}<br>` +
      `tied aggregate ${{weightFmt(bucket.center_weight)}}; simulated exposure ${{bucket.simulated_epochs.toFixed(2)}} epochs`))
    .on("mouseleave", hideTooltip);
  row.append("rect").attr("x", 0).attr("y", 0).attr("width", width).attr("height", rowHeight - 1)
    .attr("fill", bucket => bucket.domain === focalDomain ? "#fff1cf" : "transparent");
  row.append("text").attr("x", left - 10).attr("y", 13).attr("text-anchor", "end")
    .attr("font-size", 10).attr("font-weight", bucket => bucket.domain === focalDomain ? 800 : 500)
    .text(bucket => bucket.domain.replace("dolma3_cc/", "cc/"));
  columns.slice(0, 2).forEach((column, columnIndex) => {{
    const x = left + columnIndex * colWidth + 6;
    row.append("rect").attr("x", x).attr("y", 4).attr("height", 10)
      .attr("width", bucket => scale(bucket[column.plusKey])).attr("fill", "#eb6a32").attr("fill-opacity", .62);
    row.append("rect").attr("x", x).attr("y", 2).attr("height", 14)
      .attr("width", bucket => scale(bucket[column.minusKey])).attr("fill", "none")
      .attr("stroke", "#173044").attr("stroke-width", 1.25).attr("stroke-dasharray", "3 2");
  }});
  row.append("rect").attr("x", left + 2 * colWidth + 6).attr("y", 4).attr("height", 10)
    .attr("width", bucket => scale(bucket.center_weight)).attr("fill", "#8798a1").attr("fill-opacity", .88);
  [1, 2].forEach(columnIndex => {{
    svg.append("line").attr("x1", left + columnIndex * colWidth).attr("x2", left + columnIndex * colWidth)
      .attr("y1", 0).attr("y2", height - 28).attr("stroke", "#b7b0a4");
  }});
  svg.append("text").attr("x", left).attr("y", height - 10).attr("fill", "#63727a").attr("font-size", 11)
    .text("Focal bucket first; remaining buckets sorted by absolute phase contrast.");
}}

function renderPairModal() {{
  const pairGroup = pairByDomain.get(activePairDomain);
  if (!pairGroup) throw new Error(`Missing controlled pair payload for ${{activePairDomain}}`);
  const pair = pairGroup.anchors.find(anchor => anchor.anchor_id === activePairAnchor);
  if (!pair) throw new Error(`Missing ${{activePairAnchor}} pair for ${{activePairDomain}}`);
  const metric = pair.metrics[target];

  d3.select("#pair-dialog-title").text(pairGroup.domain);
  d3.select("#pair-dialog-subtitle").text(
    `${{pairGroup.description}} · ${{targetMeta[target].label}} scores · ${{pair.anchor_label}} aggregate anchor`
  );
  d3.select("#pair-anchor-tabs").selectAll("button").data(pairGroup.anchors, anchor => anchor.anchor_id).join("button")
    .attr("type", "button").classed("active", anchor => anchor.anchor_id === activePairAnchor)
    .text(anchor => `${{anchor.anchor_label}} aggregate`)
    .on("click", (_, anchor) => {{
      activePairAnchor = anchor.anchor_id;
      renderPairModal();
    }});
  d3.select("#pair-trace").html(
    `<div><b>Pair ID</b>${{escapeHtml(pair.pair_id)}}</div>` +
    `<div><b>Seed</b>block ${{pair.seed_block}} · data seed ${{pair.data_seed}}</div>` +
    `<div><b>+d candidate · focal later</b>${{escapeHtml(pair.plus_candidate_id)}}</div>` +
    `<div><b>-d candidate · focal earlier</b>${{escapeHtml(pair.minus_candidate_id)}}</div>` +
    `<div><b>Tied control</b>${{escapeHtml(pair.center_candidate_id)}}</div>` +
    `<div><b>Design checks</b>phase TV ${{pair.phase_tv.toFixed(4)}} · aggregate mismatch ${{pair.aggregate_max_abs_difference.toExponential(2)}}</div>`
  );
  const betterOrdering = metric.plus_bpb < metric.minus_bpb ? "+d later" : "-d earlier";
  d3.select("#pair-metrics").html(
    `<div><b>${{fmt(metric.plus_bpb)}}</b><span>+d later · ${{signed(metric.plus_delta_vs_control)}} vs tied</span></div>` +
    `<div><b>${{fmt(metric.minus_bpb)}}</b><span>-d earlier · ${{signed(metric.minus_delta_vs_control)}} vs tied</span></div>` +
    `<div><b>${{fmt(metric.center_bpb)}}</b><span>same-seed tied control</span></div>` +
    `<div><b>${{fmt(metric.pair_gap_bpb)}}</b><span>order gap · ${{betterOrdering}} wins</span></div>` +
    `<div><b>${{signed(metric.even_asymmetry_cost_bpb)}}</b><span>mean asymmetry cost vs tied</span></div>`
  );
  const resolution = DATA.statistical_resolution.targets[target];
  const clearedThresholds = resolution.noise_sources.filter(
    source => metric.pair_gap_bpb >= source.rough_95_difference_threshold
  ).length;
  const matchedAnchor = pair.anchor_id === targetAnchor[target];
  const matchedStats = matchedAnchor ? resolution.pair_metrics[pairGroup.domain] : null;
  const screenText = matchedStats == null ? "Cross-anchor view; no multiplicity screen is attached." :
    `Rough independent normal screen p=${{d3.format(".3g")(matchedStats.rough_independent_pvalue)}}; ` +
    `39-bucket Holm-adjusted p=${{d3.format(".3g")(matchedStats.rough_holm_adjusted_pvalue)}}.`;
  d3.select("#pair-resolution").html(
    `<b>Statistical status: unresolved.</b> The ${{fmt(metric.pair_gap_bpb)}} BPB order gap clears ` +
    `${{clearedThresholds}} of ${{resolution.noise_sources.length}} descriptive independent-run thresholds ` +
    `(${{fmt(resolution.rough_95_threshold_min)}}-${{fmt(resolution.rough_95_threshold_max)}} BPB). ` +
    `${{screenText}} The +d/-d/tied triple shares one data seed and has no replicated paired standard error.`
  );
  renderPairAnatomy(pair, pairGroup.domain);
}}

function openPairModal(domain) {{
  if (!pairByDomain.has(domain)) throw new Error(`Unknown controlled pair domain ${{domain}}`);
  activePairDomain = domain;
  activePairAnchor = targetAnchor[target];
  renderPairModal();
  if (!pairDialog.open) pairDialog.showModal();
}}

d3.select("#pair-close").on("click", () => pairDialog.close());
d3.select(pairDialog).on("click", event => {{
  if (event.target === pairDialog) pairDialog.close();
}});

function renderBucketMatrix() {{
  const host = d3.select("#bucket-matrix");
  host.selectAll("*").remove();
  const rows = DATA.intuition.filter(d => d.target === target).sort((a,b) =>
    d3.ascending(a.pool, b.pool) || d3.ascending(a.domain, b.domain));
  const width = Math.max(1120, host.node().clientWidth);
  const left = 330, top = 58, rowH = 27, cellW = (width - left - 24) / 4;
  const height = top + rows.length * rowH + 34;
  const svg = host.append("svg").attr("viewBox", `0 0 ${{width}} ${{height}}`);
  const columns = [
    {{key:"controlled", label:"Controlled order", value:d =>
      d.controlled_evidence_kind !== "direction-specific" ? 0 :
      d.controlled_preference === "later" ? d.controlled_net_gain_bpb :
      d.controlled_preference === "earlier" ? -d.controlled_net_gain_bpb : 0}},
    {{key:"phase", label:"Best-tail phase enrichment", value:d => d.median_series_best_minus_worst_phase_contrast}},
    {{key:"phaseFraction", label:"Series later-support", value:d => d.fraction_series_best_places_later - .5}},
    {{key:"aggregate", label:"Best-tail aggregate enrichment", value:d => d.median_series_best_minus_worst_aggregate_weight}}
  ];
  const extents = Object.fromEntries(columns.map(c => {{
    const max = d3.max(rows, d => Math.abs(c.value(d))) || 1;
    return [c.key, max];
  }}));
  const color = d3.scaleDiverging([-1, 0, 1], d3.interpolateRdYlGn);
  columns.forEach((column, ci) => {{
    svg.append("text").attr("x", left + ci * cellW + cellW / 2).attr("y", 21)
      .attr("text-anchor", "middle").attr("font-weight", 800).text(column.label);
    svg.append("text").attr("x", left + ci * cellW + cellW / 2).attr("y", 39)
      .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", "#63727a")
      .text(column.key === "controlled" ? "green = later; red = earlier; neutral = tied/ambiguous" :
        column.key === "aggregate" ? "green = more in better tails" : "green = later in better tails");
  }});
  const groups = svg.selectAll("g.matrix-row").data(rows).join("g").attr("class", "matrix-row")
    .attr("transform", (_, i) => `translate(0,${{top + i * rowH}})`);
  groups.append("rect").attr("x", 0).attr("width", width).attr("height", rowH)
    .attr("fill", (_, i) => i % 2 ? "#faf6ed" : "#fffdf7");
  groups.append("text").attr("x", 8).attr("y", 18).attr("font-size", 11.5)
    .attr("font-weight", d => d.controlled_evidence_kind === "direction-specific" ? 800 : 500)
    .text(d => d.domain);
  groups.append("text").attr("x", left - 10).attr("y", 18).attr("text-anchor", "end")
    .attr("font-size", 10).attr("fill", "#63727a").text(d => d.pool);
  columns.forEach((column, ci) => {{
    groups.append("rect").attr("x", left + ci * cellW + 1).attr("y", 1)
      .attr("width", cellW - 2).attr("height", rowH - 2)
      .attr("fill", d => color(column.value(d) / extents[column.key]))
      .attr("fill-opacity", d => column.key === "controlled"
        ? (d.controlled_evidence_kind === "direction-specific"
          ? Math.min(.82, .2 + .62 * Math.min(1, Math.abs(d.controlled_gain_over_repeat_coordinate_sd)))
          : .25)
        : .82)
      .on("mousemove", (event, d) => showTooltip(event,
        `<b>${{d.domain}}</b><br>${{d.description}}<br><br>` +
        `<b>${{d.synthesis}}</b><br>` +
        `controlled: ${{d.controlled_preference}}, ${{d.controlled_evidence_kind}}, gain ${{d.controlled_net_gain_bpb.toFixed(6)}} ` +
        `(${{d.controlled_gain_over_repeat_coordinate_sd.toFixed(2)}}x repeat-coordinate SD)<br>` +
        `series phase delta ${{pct(d.median_series_best_minus_worst_phase_contrast)}}; ` +
        `${{d3.format(".0%")(d.fraction_series_best_places_later)}} of series place it later<br>` +
        `series aggregate delta ${{pct(d.median_series_best_minus_worst_aggregate_weight)}}`))
      .on("mouseleave", hideTooltip);
  }});
}}

function policyLink(ref) {{
  if (!ref || !ref.policy_id) return "No example in this archive tail.";
  const alignment = ref.rule_alignment == null ? "" : `; rule alignment ${{pct(ref.rule_alignment)}}`;
  return `<b>${{ref.label}}</b><br>${{targetMeta[target].label}} ${{fmt(ref.bpb)}}; phase TV ${{ref.phase_tv.toFixed(3)}}${{alignment}}`;
}}
function controlledCounterexampleList(fixed) {{
  const refs = JSON.parse(fixed.controlled_counterexamples_json || "[]");
  if (!refs.length) return `<div class="controlled-counterexample">No cross-anchor antithetic counterexample.</div>`;
  const buttons = refs.map(ref => {{
    const pair = pairByDomain.get(ref.domain);
    if (!pair) throw new Error(`Missing pair payload for controlled counterexample ${{ref.domain}}`);
    const kind = ref.evidence_kind.replaceAll("_", " ");
    const resolution = DATA.statistical_resolution.targets[target].pair_metrics[ref.domain];
    return `<button type="button" class="pair-open" data-domain="${{escapeHtml(ref.domain)}}">` +
      `<b>${{escapeHtml(ref.domain)}}</b>` +
      `<span>${{escapeHtml(kind)}} · later-minus-earlier ${{signed(ref.uncheatable_anchor_later_minus_earlier_bpb)}} / ` +
      `${{signed(ref.table9_anchor_later_minus_earlier_bpb)}} BPB at Uncheatable / Table-9 anchors<br>` +
      `unresolved · rough 39-bucket Holm p=${{d3.format(".3g")(resolution.rough_holm_adjusted_pvalue)}}</span>` +
      `<code>${{pair.direction_id}}</code></button>`;
  }}).join("");
  return `<div class="controlled-counterexample"><b>Cross-anchor fixed-aggregate reversals</b>` +
    `<div class="controlled-counterexample-list">${{buttons}}</div></div>`;
}}
function renderRules() {{
  const rows = DATA.broad_rule_stress.filter(d => d.target === target);
  const controlled = new Map(DATA.controlled_rules
    .filter(d => d.target === target)
    .map(d => [d.rule_id, d]));
  const cards = d3.select("#rule-grid").selectAll("article").data(rows, d => d.rule_id).join("article")
    .attr("class", "rule-card").html(d => {{
      const fixed = controlled.get(d.rule_id);
      if (!fixed) throw new Error(`Missing controlled rule evidence for ${{d.rule_id}} / ${{target}}`);
      const counter = JSON.parse(d.best_counterexample_json || "{{}}");
      const conformer = JSON.parse(d.worst_conformer_json || "{{}}");
      return `<div class="eyebrow">${{d.expected_direction}} / ${{d.domain_count}} buckets</div>` +
        `<h3>${{d.rule_label}}</h3><p>${{d.premise}}</p>` +
        `<div class="rule-subhead">Controlled antithetic test</div>` +
        `<div class="rule-score"><div><b>${{fixed.controlled_supports}}</b><span>supporting buckets</span></div>` +
        `<div><b>${{fixed.controlled_counterexamples}}</b><span>counterexample buckets</span></div>` +
        `<div><b>${{fixed.controlled_anchor_dependent}}</b><span>anchor-dependent</span></div></div>` +
        `${{controlledCounterexampleList(fixed)}}` +
        `<p>${{fixed.controlled_direction_specific_counterexamples}} direction-specific counterexample(s); ` +
        `${{fixed.controlled_direction_specific_supports}} direction-specific support(s). Select a row to inspect both exact antithetic pairs.</p>` +
        `<div class="rule-subhead">Broad archive stress · associative only</div>` +
        `<div class="rule-score"><div><b>${{pct(d.global_best_minus_worst_alignment)}}</b><span>global tail alignment</span></div>` +
        `<div><b>${{pct(d.median_series_best_minus_worst_alignment)}}</b><span>median series alignment</span></div>` +
        `<div><b>${{d3.format(".0%")(d.fraction_series_support_rule)}}</b><span>series supporting</span></div></div>` +
        `<div class="counterexample"><b>Best-tail heuristic violation</b><br>${{policyLink(counter)}}</div>` +
        `<div class="counterexample conformer"><b>Worst-tail conformer</b><br>${{policyLink(conformer)}}</div>`;
    }});
  cards.selectAll("button.pair-open").on("click", function(event) {{
    event.stopPropagation();
    openPairModal(this.dataset.domain);
  }});
}}

function auditedPolicies() {{
  return DATA.policies.filter(d => d.selections[target].length);
}}
function renderPolicyTable() {{
  const query = d3.select("#policy-search").property("value").toLowerCase();
  const tail = d3.select("#tail-filter").property("value");
  const rows = auditedPolicies().filter(d => {{
    const tags = d.selections[target];
    const matchTail = !tail || tags.some(tag => tag.tail === tail);
    const haystack = `${{d.label}} ${{d.series}} ${{d.panel_tags}}`.toLowerCase();
    return matchTail && (!query || haystack.includes(query));
  }}).sort((a,b) => d3.ascending(targetValue(a), targetValue(b)));
  d3.select("#policy-table").selectAll("tr").data(rows, d => d.policy_id).join("tr")
    .classed("selected", d => selectedPolicy && d.policy_id === selectedPolicy.policy_id)
    .on("click", (_, d) => selectPolicy(d))
    .html(d => {{
      const tags = d.selections[target].map(tag =>
        `<span class="pill">${{tag.scope.replace("_"," ")}} ${{tag.tail}}</span>`).join("");
      return `<td><b>${{d.label}}</b><br><span style="color:#63727a">${{d.policy_id}}</span></td>` +
        `<td>${{tags}}</td><td>${{fmt(d.uncheatable)}}</td><td>${{fmt(d.table9)}}</td>` +
        `<td>${{d.phase_tv.toFixed(3)}}</td><td>${{d.series}}</td>`;
    }});
}}

function setTarget(next) {{
  target = next;
  d3.selectAll("button.objective").classed("active", function() {{ return this.dataset.target === target; }});
  renderScatter(); renderBucketMatrix(); renderRules(); renderPolicyTable();
  if (selectedPolicy) selectPolicy(selectedPolicy);
  if (pairDialog.open && activePairDomain) {{
    activePairAnchor = targetAnchor[target];
    renderPairModal();
  }}
}}
d3.selectAll("button.objective").on("click", function() {{ setTarget(this.dataset.target); }});
d3.select("#series-filter").on("change", renderScatter);
d3.select("#policy-search").on("input", renderPolicyTable);
d3.select("#tail-filter").on("change", renderPolicyTable);

renderScatter(); renderBucketMatrix(); renderRules(); renderPolicyTable();
const initial = auditedPolicies().sort((a,b) => d3.ascending(targetValue(a), targetValue(b)))[0];
if (initial) selectPolicy(initial);
</script>
</body>
</html>"""


def write_summary_json(
    path: Path,
    data: AuditData,
    selections: pd.DataFrame,
    comments: pd.DataFrame,
) -> None:
    payload = {
        "unique_two_phase_coordinates": len(data.policies),
        "genuine_two_phase_coordinates": int(data.policies["phase_regime"].eq("genuine_two_phase").sum()),
        "observation_rows": int(data.policies["observation_count"].sum()),
        "global_tail_count_per_target_side": GLOBAL_TAIL_COUNT,
        "series_min_policies": SERIES_MIN_POLICIES,
        "series_tail_fraction": SERIES_TAIL_FRACTION,
        "selected_policy_target_rows": len(comments),
        "selected_unique_policies": comments["policy_id"].nunique(),
        "selection_rows": len(selections),
        "repeated_coordinate_count": int(data.policies["observation_count"].gt(1).sum()),
        "targets": {
            target: {
                "minimum": float(data.policies[metric].min()),
                "median": float(data.policies[metric].median()),
                "maximum": float(data.policies[metric].max()),
            }
            for target, metric in TARGET_COLUMNS.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_audit_data(args.registry)
    controlled_dossier = pd.read_csv(CONTROLLED_AUDIT_DIR / "bucket_phase_order_dossier.csv")
    repeat_noise = repeat_noise_by_target(data.policies)
    rules = phase_order_rules(data.semantics)

    selections = select_tail_policies(data.policies)
    comments = comment_selected_policies(selections, data.policy_buckets, controlled_dossier)
    by_series, evidence = series_bucket_evidence(data)
    intuition = bucket_intuition_table(data, evidence, controlled_dossier, repeat_noise)
    broad_rule_stress = rule_evidence_table(data, rules)
    controlled_rules = controlled_rule_evidence_table()
    controlled_pairs = controlled_pair_payload(data)
    statistical_resolution = statistical_resolution_summary(data.policies, controlled_rules, controlled_pairs)
    hybrid_winners, hybrid_package = hybrid_fixed_aggregate_analysis(
        args.registry,
        data.semantics,
        repeat_noise,
    )
    extreme_epochs = data.policies[data.policies["max_simulated_epoch"].gt(30.0)].sort_values("max_simulated_epoch")
    (args.output_dir / "rule_tail_evidence.csv").unlink(missing_ok=True)

    data.policies.to_csv(args.output_dir / "coordinate_policies.csv", index=False)
    data.policy_buckets.to_csv(args.output_dir / "policy_bucket_features.csv", index=False)
    selections.to_csv(args.output_dir / "tail_selections.csv", index=False)
    comments.to_csv(args.output_dir / "tail_policy_comments.csv", index=False)
    by_series.to_csv(args.output_dir / "bucket_tail_evidence_by_series.csv", index=False)
    intuition.to_csv(args.output_dir / "bucket_intuition.csv", index=False)
    broad_rule_stress.to_csv(args.output_dir / "broad_tail_rule_stress.csv", index=False)
    controlled_rules.to_csv(args.output_dir / "controlled_rule_evidence.csv", index=False)
    (args.output_dir / "controlled_antithetic_pairs.json").write_text(
        json.dumps(controlled_pairs, indent=2, allow_nan=False) + "\n"
    )
    (args.output_dir / "statistical_resolution.json").write_text(
        json.dumps(statistical_resolution, indent=2, allow_nan=False) + "\n"
    )
    hybrid_winners.to_csv(args.output_dir / "hybrid_fixed_aggregate_winners.csv", index=False)
    hybrid_package.to_csv(args.output_dir / "hybrid_winner_phase_package.csv", index=False)
    extreme_epochs.to_csv(args.output_dir / "extreme_simulated_epoch_failures.csv", index=False)
    write_policy_comments(args.output_dir / "tail_policy_comments.md", comments, selections)
    write_markdown_report(
        args.output_dir / "report.md",
        data,
        selections,
        comments,
        intuition,
        broad_rule_stress,
        controlled_rules,
        hybrid_winners,
        hybrid_package,
    )
    write_summary_json(args.output_dir / "summary.json", data, selections, comments)

    assert D3_SOURCE.exists(), f"Missing local D3 bundle at {D3_SOURCE}"
    report_payload = html_payload(
        data,
        selections,
        comments,
        intuition,
        broad_rule_stress,
        controlled_rules,
        controlled_pairs,
        hybrid_winners,
        hybrid_package,
        statistical_resolution,
    )
    report_html = build_html(report_payload, D3_SOURCE.read_text())
    (args.output_dir / "phase_order_bucket_intuition.html").write_text(report_html)
    print(args.output_dir)


if __name__ == "__main__":
    main()
