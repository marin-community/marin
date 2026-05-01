# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "plotly", "scipy", "scikit-learn", "tabulate", "torch"]
# ///
"""Fit GRP/proxy models to suite-balanced 300M benchmark objectives.

The earlier 119-column aggregate was effectively an MMLU aggregate because 116
columns were MMLU standard or MMLU SL-verb subject metrics. This script first
collapses metrics into task families, then applies a light SNR/difficulty
shrinkage to prevent noisy or duplicated suites from dominating the objective.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry import (
    fit_grp_300m_perplexity_proxy_benchmark as proxy,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.fit_grp_300m_mean_choice_prob_norm import (
    _choice_prob_columns,
    _suite_name,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.fit_grp_300m_mmlu_choice_prob_norm import (
    DEFAULT_COARSE_TOP_K,
    DEFAULT_METHOD,
    DEFAULT_PROB_EPS,
    DEFAULT_RANDOM_STARTS,
    SCALE,
    _model_options,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.fit_grp_no_l2_benchmark_aggregates import (
    DEFAULT_FAMILY_SCHEME,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.build_metric_registry import (
    METRICS_WIDE_CSV,
    WEIGHT_PREFIXES,
)

COHORT = "signal"
TARGET_CHOICE = "suite_balanced_choice_prob_norm_snr_difficulty"
TARGET_ACCURACY = "suite_balanced_accuracy_snr_difficulty"
TARGET_CHOICE_EQUAL = "suite_balanced_choice_prob_norm_equal"
TARGET_ACCURACY_EQUAL = "suite_balanced_accuracy_equal"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "reference_outputs" / "grp_300m_suite_balanced_benchmark_20260429"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
PROXY_FITS_CSV = OUTPUT_DIR / "proxy_fits.csv"
SELECTED_FEATURES_CSV = OUTPUT_DIR / "selected_features.csv"
SINGLE_FEATURE_CSV = OUTPUT_DIR / "single_feature_baselines.csv"
OPTIMUM_DIAGNOSTICS_CSV = OUTPUT_DIR / "optimum_diagnostics.csv"
REPORT_MD = OUTPUT_DIR / "report.md"
LOGBOOK_MD = Path(".agents/logbooks/benchmark-proxy-optimization.md")

MMLU_STANDARD_GROUP = "mmlu_standard"
MMLU_SL_VERB_GROUP = "mmlu_sl_verb"
MMLU_STANDARD_TOPLEVEL = "mmlu_5shot"
MMLU_SL_VERB_TOPLEVEL = "mmlu_sl_verb_5shot"
EXCLUDED_SUITES = frozenset({"mmlu_pro_5shot"})

# These are from choice_logprob_norm_signal_to_noise.csv.  Non-MMLU groups do
# not yet have a fixed-mixture seed panel, so they receive a neutral reliability
# prior rather than an invented measurement.
GROUP_SNR = {
    MMLU_STANDARD_GROUP: 0.725619,
    MMLU_SL_VERB_GROUP: 1.349615,
}
DEFAULT_RELIABILITY = 0.75
WEIGHT_SHRINK_TO_EQUAL = 0.50
MIN_DIFFICULTY_WEIGHT = 0.50


@dataclass(frozen=True)
class SuiteGroup:
    """One suite-level target component."""

    name: str
    choice_columns: tuple[str, ...]
    accuracy_columns: tuple[str, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-starts", type=int, default=DEFAULT_RANDOM_STARTS)
    parser.add_argument("--coarse-top-k", type=int, default=DEFAULT_COARSE_TOP_K)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--prob-eps", type=float, default=DEFAULT_PROB_EPS)
    parser.add_argument("--family-scheme", default=DEFAULT_FAMILY_SCHEME)
    parser.add_argument("--block-variant", default="full")
    parser.add_argument("--max-selected-features", type=int, default=12)
    return parser.parse_args()


def _task_group(suite: str) -> str:
    if suite in EXCLUDED_SUITES:
        raise ValueError(f"Excluded suite should have been filtered: {suite}")
    if suite == MMLU_STANDARD_TOPLEVEL:
        return MMLU_STANDARD_GROUP
    if suite == MMLU_SL_VERB_TOPLEVEL:
        return MMLU_SL_VERB_GROUP
    if suite.startswith("mmlu_") and suite.endswith("_sl_verb_5shot"):
        return MMLU_SL_VERB_GROUP
    if suite.startswith("mmlu_") and suite.endswith("_5shot"):
        return MMLU_STANDARD_GROUP
    return suite


def _preferred_group_columns(frame: pd.DataFrame) -> tuple[SuiteGroup, ...]:
    choice_by_group: dict[str, list[str]] = {}
    for column in _choice_prob_columns(frame):
        suite = _suite_name(column)
        if suite in EXCLUDED_SUITES:
            continue
        group = _task_group(suite)
        accuracy_column = column.rsplit("/", 1)[0] + "/acc"
        if accuracy_column not in frame.columns:
            continue
        choice_by_group.setdefault(group, []).append(column)

    groups: list[SuiteGroup] = []
    for group, columns in sorted(choice_by_group.items()):
        preferred_suite = {
            MMLU_STANDARD_GROUP: MMLU_STANDARD_TOPLEVEL,
            MMLU_SL_VERB_GROUP: MMLU_SL_VERB_TOPLEVEL,
        }.get(group)
        if preferred_suite is not None:
            preferred = f"lm_eval/{preferred_suite}/choice_prob_norm"
            if preferred in columns:
                columns = [preferred]
        accuracy_columns = [column.rsplit("/", 1)[0] + "/acc" for column in columns]
        groups.append(SuiteGroup(group, tuple(sorted(columns)), tuple(sorted(accuracy_columns))))
    if len(groups) < 3:
        raise ValueError(f"Expected multiple benchmark groups, got {[group.name for group in groups]}")
    return tuple(groups)


def _reliability_from_snr(group: str) -> float:
    snr = GROUP_SNR.get(group)
    if snr is None:
        return DEFAULT_RELIABILITY
    return float(snr**2 / (1.0 + snr**2))


def _difficulty_from_accuracy(values: pd.Series) -> float:
    mean_accuracy = float(values.mean())
    bernoulli_std = np.sqrt(max(mean_accuracy * (1.0 - mean_accuracy), 0.0))
    return float(np.clip(bernoulli_std / 0.5, MIN_DIFFICULTY_WEIGHT, 1.0))


def _group_weights(data: pd.DataFrame, groups: tuple[SuiteGroup, ...]) -> pd.DataFrame:
    equal = np.full(len(groups), 1.0 / len(groups), dtype=float)
    reliability = np.asarray([_reliability_from_snr(group.name) for group in groups], dtype=float)
    difficulty = np.asarray(
        [_difficulty_from_accuracy(data[f"group__{group.name}__accuracy"]) for group in groups],
        dtype=float,
    )
    raw = reliability * difficulty
    raw = raw / raw.sum()
    shrunk = WEIGHT_SHRINK_TO_EQUAL * equal + (1.0 - WEIGHT_SHRINK_TO_EQUAL) * raw
    rows = []
    for idx, group in enumerate(groups):
        rows.append(
            {
                "group": group.name,
                "n_choice_columns": len(group.choice_columns),
                "choice_columns": json.dumps(group.choice_columns),
                "n_accuracy_columns": len(group.accuracy_columns),
                "accuracy_columns": json.dumps(group.accuracy_columns),
                "equal_weight": float(equal[idx]),
                "snr": GROUP_SNR.get(group.name, np.nan),
                "reliability": float(reliability[idx]),
                "difficulty": float(difficulty[idx]),
                "raw_reliability_difficulty_weight": float(raw[idx]),
                "snr_difficulty_weight": float(shrunk[idx]),
            }
        )
    return pd.DataFrame.from_records(rows)


def _is_perplexity_feature(column: str) -> bool:
    lower = column.lower()
    return (
        column.startswith(("eval/", "lm_eval/"))
        and any(token in lower for token in ("/bpb", "/loss", "perplexity"))
        and "logprob" not in lower
    )


def _build_dataset() -> tuple[pd.DataFrame, dict[str, tuple[str, ...]], pd.DataFrame]:
    frame = pd.read_csv(METRICS_WIDE_CSV, low_memory=False)
    groups = _preferred_group_columns(frame)
    required_columns = [column for group in groups for column in (*group.choice_columns, *group.accuracy_columns)]
    mask = frame["scale"].eq(SCALE) & frame["cohort"].eq(COHORT) & frame["is_qsplit240_core"].fillna(False)
    data = frame.loc[mask].copy()
    data = data.dropna(subset=required_columns).copy()

    for group in groups:
        data[f"group__{group.name}__choice_prob_norm"] = data[list(group.choice_columns)].mean(axis=1)
        data[f"group__{group.name}__accuracy"] = data[list(group.accuracy_columns)].mean(axis=1)

    weights = _group_weights(data, groups)
    equal_weights = dict(zip(weights["group"], weights["equal_weight"], strict=True))
    snr_weights = dict(zip(weights["group"], weights["snr_difficulty_weight"], strict=True))

    data[TARGET_CHOICE_EQUAL] = 0.0
    data[TARGET_ACCURACY_EQUAL] = 0.0
    data[TARGET_CHOICE] = 0.0
    data[TARGET_ACCURACY] = 0.0
    for group in groups:
        choice_column = f"group__{group.name}__choice_prob_norm"
        accuracy_column = f"group__{group.name}__accuracy"
        data[TARGET_CHOICE_EQUAL] += equal_weights[group.name] * data[choice_column]
        data[TARGET_ACCURACY_EQUAL] += equal_weights[group.name] * data[accuracy_column]
        data[TARGET_CHOICE] += snr_weights[group.name] * data[choice_column]
        data[TARGET_ACCURACY] += snr_weights[group.name] * data[accuracy_column]

    weight_columns = tuple(sorted(column for column in data.columns if column.startswith(WEIGHT_PREFIXES)))
    feature_columns = tuple(column for column in data.columns if _is_perplexity_feature(column))
    complete_features = tuple(column for column in feature_columns if data[column].notna().all())
    deployable_features = tuple(column for column in complete_features if column.startswith("eval/"))
    diagnostic_features = tuple(complete_features)
    group_target_columns = tuple(
        column
        for group in groups
        for column in (f"group__{group.name}__choice_prob_norm", f"group__{group.name}__accuracy")
    )
    id_columns = tuple(
        column
        for column in (
            "registry_run_key",
            "run_id",
            "run_name",
            "scale",
            "cohort",
            "source_run_name",
            "source_experiment",
            "wandb_run_id",
            "checkpoint_root",
            "status",
            "is_qsplit240_core",
        )
        if column in data.columns
    )
    dataset = data[
        list(id_columns)
        + list(weight_columns)
        + [
            TARGET_CHOICE,
            TARGET_ACCURACY,
            TARGET_CHOICE_EQUAL,
            TARGET_ACCURACY_EQUAL,
            *group_target_columns,
            *complete_features,
        ]
    ].copy()
    dataset = dataset.dropna(axis=1, how="all").reset_index(drop=True)
    feature_sets = {"eval_only": deployable_features, "eval_plus_lm_eval": diagnostic_features}

    if len(dataset) != 240:
        raise ValueError(f"Expected 240 qsplit-core rows, got {len(dataset)}")
    for prefix in ("phase_0_", "phase_1_"):
        columns = [column for column in dataset.columns if column.startswith(prefix)]
        sums = dataset[columns].sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-6):
            raise ValueError(f"{prefix} weights do not sum to 1")
    if not deployable_features:
        raise ValueError("No complete eval/* perplexity features found")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_DIR / "suite_balanced_dataset.csv", index=False)
    weights.to_csv(OUTPUT_DIR / "suite_group_weights.csv", index=False)
    (OUTPUT_DIR / "dataset_summary.json").write_text(
        json.dumps(
            {
                "n_rows": len(dataset),
                "scale": SCALE,
                "cohort": COHORT,
                "targets": [TARGET_CHOICE, TARGET_ACCURACY],
                "equal_weight_reference_targets": [TARGET_CHOICE_EQUAL, TARGET_ACCURACY_EQUAL],
                "n_groups": len(groups),
                "groups": weights.to_dict(orient="records"),
                "n_eval_only_features": len(deployable_features),
                "n_eval_plus_lm_eval_features": len(diagnostic_features),
                "weight_shrink_to_equal": WEIGHT_SHRINK_TO_EQUAL,
                "default_reliability": DEFAULT_RELIABILITY,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset, feature_sets, weights


def _write_report(
    *,
    group_weights: pd.DataFrame,
    proxy_frame: pd.DataFrame,
    single_feature: pd.DataFrame,
    summary: pd.DataFrame,
    optimum: pd.DataFrame,
) -> None:
    proxy_columns = [
        "candidate_id",
        "target",
        "feature_set",
        "model_type",
        "transform",
        "n_features",
        "n_selected_features",
        "full_proxy_spearman",
        "selected_proxy_spearman",
        "selected_proxy_regret_at_1",
        "selected_proxy_rmse",
    ]
    summary_columns = [
        "candidate_id",
        "candidate_kind",
        "target",
        "feature_set",
        "choice_spearman",
        "accuracy_spearman",
        "choice_regret_at_1",
        "accuracy_regret_at_1",
        "raw_predicted_proxy_metric",
        "raw_nearest_observed_tv",
        "raw_nearest_observed_choice",
        "raw_nearest_observed_accuracy",
        "top8actual_hull_nearest_observed_tv",
    ]
    body = [
        "# 300M Suite-Balanced Benchmark Optimization",
        "",
        "## Setup",
        "",
        f"- Rows: 240 qsplit-core `{SCALE}` rows.",
        "- Target groups collapse MMLU standard and MMLU SL-verb into suite-level components before averaging.",
        "- Primary targets use equal-suite weights shrunk toward SNR/difficulty weights; "
        "equal-suite targets are written for audit.",
        "- MMLU SNR evidence comes from `choice_logprob_norm_signal_to_noise.csv`; "
        "non-MMLU groups use a neutral reliability prior.",
        "",
        "## Group Weights",
        "",
        group_weights[
            [
                "group",
                "n_choice_columns",
                "equal_weight",
                "snr",
                "reliability",
                "difficulty",
                "snr_difficulty_weight",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Best Proxy Fits",
        "",
        proxy_frame.sort_values(["target", "feature_set", "selected_proxy_spearman"], ascending=[True, True, False])[
            proxy_columns
        ]
        .groupby(["target", "feature_set"], as_index=False)
        .head(2)
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Best Single Perplexity Baselines",
        "",
        single_feature.groupby(["target", "feature_set"], as_index=False)
        .head(1)[["target", "feature_set", "feature", "single_feature_spearman", "single_feature_regret_at_1"]]
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "## GRP Candidates",
        "",
        summary[summary_columns]
        .sort_values(["target", "candidate_kind", "choice_spearman"], ascending=[True, True, False])
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Optimum Diagnostics",
        "",
        optimum[
            [
                "candidate_id",
                "candidate_kind",
                "target",
                "opt_kind",
                "predicted_proxy_metric",
                "nearest_observed_tv",
                "nearest_observed_choice",
                "nearest_observed_accuracy",
                "nearest_observed_primary_regret",
                "raw_phase0_support_gt_1e4",
                "raw_phase1_support_gt_1e4",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        "- This target fixes the flat-average MMLU domination problem; MMLU contributes as one or two "
        "suite components, not 116 columns.",
        "- Suite balancing changes the validation target, so compare against flat-target runs by interpretation, "
        "not as a strict replacement.",
        "- Raw optima with large nearest-observed TV are still non-deployable even if the suite-balanced "
        "proxy fit improves.",
        "",
    ]
    REPORT_MD.write_text("\n".join(body), encoding="utf-8")


def _append_logbook(summary: pd.DataFrame) -> None:
    LOGBOOK_MD.parent.mkdir(parents=True, exist_ok=True)
    best_rows = summary.sort_values(["choice_spearman", "accuracy_spearman"], ascending=[False, False]).head(4)
    section = [
        "",
        "### 2026-04-29 - Suite-balanced 300M benchmark objective",
        "- Hypothesis: the flat 119-column objective overweights MMLU subject variants, so a "
        "suite-balanced target should be a better benchmark optimization target.",
        f"- Command: `uv run --with matplotlib --with torch python {Path(__file__)}`",
        "- Result summary:",
        best_rows[
            [
                "candidate_id",
                "candidate_kind",
                "target",
                "feature_set",
                "choice_spearman",
                "accuracy_spearman",
                "raw_nearest_observed_tv",
                "raw_nearest_observed_choice",
                "raw_nearest_observed_accuracy",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "- Artifacts: "
        "`experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
        "grp_300m_suite_balanced_benchmark_20260429/`.",
        "",
    ]
    with LOGBOOK_MD.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(section))


def main() -> None:
    args = _parse_args()
    proxy.OUTPUT_DIR = OUTPUT_DIR
    data, feature_sets, group_weights = _build_dataset()
    targets = (TARGET_CHOICE, TARGET_ACCURACY)
    single_feature = proxy._best_single_feature_baselines(data, feature_sets, targets=targets)
    single_feature.to_csv(SINGLE_FEATURE_CSV, index=False)
    proxy_frame, feature_frame, candidates = proxy._fit_proxy_candidates(
        data,
        feature_sets,
        max_selected_features=args.max_selected_features,
        targets=targets,
    )
    proxy_frame.to_csv(PROXY_FITS_CSV, index=False)
    feature_frame.to_csv(SELECTED_FEATURES_CSV, index=False)
    selected_candidates = proxy._selected_proxy_candidates(proxy_frame, candidates, targets=targets)
    start_bank = proxy._expanded_start_bank(args.random_starts)
    model_options = _model_options(args.block_variant)

    summary_rows: list[dict[str, Any]] = []
    optimum_rows: list[pd.DataFrame] = []
    for target in targets:
        print(f"Running direct GRP target baseline {target}", flush=True)
        summary, diag = proxy._run_direct_target_candidate(
            data,
            target,
            family_scheme=args.family_scheme,
            model_options=model_options,
            start_bank=start_bank,
            coarse_top_k=args.coarse_top_k,
            method=args.method,
            prob_eps=args.prob_eps,
            choice_target=TARGET_CHOICE,
            accuracy_target=TARGET_ACCURACY,
        )
        summary_rows.append(summary)
        optimum_rows.append(diag)
    for candidate in selected_candidates:
        print(f"Running scalar GRP candidate {candidate.candidate_id}", flush=True)
        summary, diag = proxy._run_scalar_candidate(
            data,
            candidate,
            family_scheme=args.family_scheme,
            model_options=model_options,
            start_bank=start_bank,
            coarse_top_k=args.coarse_top_k,
            method=args.method,
            prob_eps=args.prob_eps,
            choice_target=TARGET_CHOICE,
            accuracy_target=TARGET_ACCURACY,
        )
        summary_rows.append(summary)
        optimum_rows.append(diag)
        print(f"Running component GRP candidate {candidate.candidate_id}", flush=True)
        summary, diag = proxy._run_component_candidate(
            data,
            candidate,
            family_scheme=args.family_scheme,
            model_options=model_options,
            start_bank=start_bank,
            prob_eps=args.prob_eps,
            choice_target=TARGET_CHOICE,
            accuracy_target=TARGET_ACCURACY,
        )
        summary_rows.append(summary)
        optimum_rows.append(diag)

    summary_frame = pd.DataFrame.from_records(summary_rows)
    optimum_frame = pd.concat(optimum_rows, ignore_index=True)
    summary_frame.to_csv(SUMMARY_CSV, index=False)
    optimum_frame.to_csv(OPTIMUM_DIAGNOSTICS_CSV, index=False)
    _write_report(
        group_weights=group_weights,
        proxy_frame=proxy_frame,
        single_feature=single_feature,
        summary=summary_frame,
        optimum=optimum_frame,
    )
    _append_logbook(summary_frame)
    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
