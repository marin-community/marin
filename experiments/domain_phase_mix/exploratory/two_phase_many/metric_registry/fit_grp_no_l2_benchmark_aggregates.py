# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly", "scipy", "scikit-learn", "tabulate"]
# ///
"""Fit GRP no-L2 surrogates to 60M benchmark aggregate objectives."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit, logit, ndtri
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_grp_power_family_penalty_no_l2_retune import (
    REG_FIXED,
    VARIANT_NAME,
    _no_l2_param_keys,
    _parameter_counts,
    _pack_no_l2_params,
    _start_bank,
    _unpack_no_l2_params,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import build_two_phase_many_loop_config
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.build_metric_registry import (
    METRICS_WIDE_CSV,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GENERIC_FAMILY_NAMES,
    GenericFamilyPacket,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    CALIBRATION_CV_WEIGHT,
    CALIBRATION_FOLDMEAN_WEIGHT,
    CALIBRATION_TAIL_WEIGHT,
    LOWER_TAIL_FRAC,
    build_penalty_calibration_surrogate,
    compute_penalty_calibration_metrics,
    optimize_penalty_calibration_model,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    PacketData,
)
from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance, build_dataset_spec_from_frame

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "benchmark_aggregate_60m" / "grp_no_l2_fits"
FIT_DATASET_CSV = SCRIPT_DIR / "fit_datasets" / "eval_uncheatable_eval_bpb__60m_1p2b__signal__fit_swarm_60m_default.csv"
OLMO_BASE_EASY_OVERLAP_RESULTS_GLOB = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_qsplit240_olmo_base_easy_overlap_rerun/**/collect_results*/results.csv"
)
OLMO_BASE_EASY_OVERLAP_SELECTED_BASELINES_GLOB = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_selected_baselines_olmo_base_easy_overlap_rerun/**/collect_results*/results.csv"
)
OLMO_BASE_EASY_OVERLAP_OLMIX_BPB_GLOB = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_olmix_bpb_olmo_base_easy_overlap_rerun_*/collect_results*/results.csv"
)
OLMO_BASE_EASY_OVERLAP_CACHE_CSV = OUTPUT_DIR / "olmo_base_easy_overlap_results.csv"
BASELINE_DOWNSTREAM_METRICS_CSV = (
    TWO_PHASE_MANY_DIR.parent / "paper_plots" / "img" / "baseline_scaling_downstream_eval_metrics_merged.csv"
)
TARGETS_CSV = OUTPUT_DIR / "benchmark_aggregate_targets.csv"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
PARAMS_CSV = OUTPUT_DIR / "params.csv"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"
REPORT_MD = OUTPUT_DIR / "report.md"
MODEL_TARGET_COLUMN = "model_target"
CV_SEED = 0
DEFAULT_RANDOM_STARTS = 24
DEFAULT_COARSE_TOP_K = 8
DEFAULT_METHOD = "Powell"
MIN_OBJECTIVE_ROWS = 200
ACCURACY_AGGREGATE_EPS = 1e-4
DEFAULT_FAMILY_SCHEME = "default"
FAMILY_SCHEMES = (
    DEFAULT_FAMILY_SCHEME,
    "qa_reasoning",
    "qa_tech",
    "academic_reasoning",
    "academic_reasoning_stack_tech",
    "synthetic_reasoning",
    "synthetic_tech",
    "cc_high_vs_rest",
)

MMLU_ACC = "lm_eval/mmlu_5shot/acc"
MMLU_BPB = "lm_eval/mmlu_5shot/bpb"
MMLU_CHOICE_LOGPROB = "lm_eval/mmlu_5shot/choice_logprob"
UNCHEATABLE_BPB = "eval/uncheatable_eval/bpb"
PALOMA_MACRO_BPB = "eval/paloma/macro_bpb"
GSM8K_ACC = "lm_eval/gsm8k/exact_match,flexible-extract"
HUMANEVAL_PASS = "lm_eval/humaneval/pass@1,create_test"
OLMO_BASE_EASY_OVERLAP_MACRO_BPB = "lm_eval/olmo_base_easy_overlap/macro_bpb"
OLMO_BASE_EASY_OVERLAP_ACCURACY_COLUMNS = (
    MMLU_ACC,
    "lm_eval/arc_easy_5shot/acc",
    "lm_eval/arc_challenge_5shot/acc",
    "lm_eval/csqa_5shot/acc",
    "lm_eval/hellaswag_5shot/acc",
    "lm_eval/winogrande_5shot/acc",
    "lm_eval/socialiqa_5shot/acc",
    "lm_eval/piqa_5shot/acc",
    "lm_eval/sciq_5shot/acc",
    "lm_eval/lambada_0shot/acc",
    "lm_eval/medmcqa_5shot/acc",
)


@dataclass(frozen=True)
class AggregateObjective:
    """One aggregate benchmark objective."""

    slug: str
    source_column: str
    display_name: str
    family: str
    higher_is_better: bool
    transform: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downstream-results-csv", action="append", default=[])
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--coarse-top-k", type=int, default=DEFAULT_COARSE_TOP_K)
    parser.add_argument("--random-starts", type=int, default=DEFAULT_RANDOM_STARTS)
    parser.add_argument("--prob-eps", type=float, default=1e-4)
    parser.add_argument("--only-slug-prefix", action="append", default=[])
    parser.add_argument("--family-scheme", action="append", choices=FAMILY_SCHEMES, default=[])
    return parser.parse_args()


def _is_cc_high(domain_name: str) -> bool:
    return domain_name.startswith("dolma3_cc/") and domain_name.endswith("_high")


def _is_cc_low(domain_name: str) -> bool:
    return domain_name.startswith("dolma3_cc/") and domain_name.endswith("_low")


def _default_family(domain_name: str) -> str:
    is_broad = (
        domain_name.startswith("dolma3_cc/")
        or domain_name
        in {
            "dolma3_wikipedia",
            "dolmino_common_crawl_hq",
            "dolmino_olmocr_pdfs_hq",
            "dolmino_stem_heavy_crawl",
        }
        or domain_name.endswith("synth_qa")
    )
    is_tech = any(token in domain_name for token in ("stack_edu", "synth_code", "synth_math")) or domain_name in {
        "dolma3_arxiv",
        "dolma3_finemath_3plus",
    }
    is_reasoning = domain_name in {"dolmino_synth_instruction", "dolmino_synth_thinking"}
    assigned = [
        family
        for family, is_member in (("broad_text", is_broad), ("tech_code", is_tech), ("reasoning", is_reasoning))
        if is_member
    ]
    if len(assigned) != 1:
        raise ValueError(f"Expected exactly one default family for {domain_name!r}, got {assigned}")
    return assigned[0]


def _family_for_domain(domain_name: str, scheme: str) -> str:
    if scheme == "default":
        return _default_family(domain_name)
    if scheme == "qa_reasoning" and domain_name.endswith("synth_qa"):
        return "reasoning"
    if scheme == "qa_tech" and domain_name.endswith("synth_qa"):
        return "tech_code"
    if scheme in {"academic_reasoning", "academic_reasoning_stack_tech"}:
        if domain_name in {
            "dolma3_arxiv",
            "dolma3_finemath_3plus",
            "dolmino_synth_instruction",
            "dolmino_synth_thinking",
        } or any(token in domain_name for token in ("synth_math", "synth_qa")):
            return "reasoning"
        if scheme == "academic_reasoning" and "stack_edu" in domain_name:
            return "reasoning"
        if "stack_edu" in domain_name or "synth_code" in domain_name:
            return "tech_code"
        return "broad_text"
    if scheme == "synthetic_reasoning":
        if domain_name.startswith("dolmino_synth_"):
            return "reasoning"
        if "stack_edu" in domain_name or "synth_code" in domain_name:
            return "tech_code"
        return "broad_text"
    if scheme == "synthetic_tech":
        if domain_name.startswith("dolmino_synth_") or "stack_edu" in domain_name:
            return "tech_code"
        if domain_name in {"dolma3_arxiv", "dolma3_finemath_3plus"}:
            return "reasoning"
        return "broad_text"
    if scheme == "cc_high_vs_rest":
        if _is_cc_high(domain_name):
            return "broad_text"
        if _is_cc_low(domain_name) or domain_name in {
            "dolma3_wikipedia",
            "dolmino_common_crawl_hq",
            "dolmino_olmocr_pdfs_hq",
            "dolmino_stem_heavy_crawl",
        }:
            return "reasoning"
        return "tech_code"
    if scheme in {"qa_reasoning", "qa_tech"}:
        return _default_family(domain_name)
    raise ValueError(f"Unsupported family scheme: {scheme}")


def _generic_family_packet_from_base(base: PacketData, family_scheme: str) -> GenericFamilyPacket:
    pairs: list[tuple[int, int]] = []
    pair_topics: list[str] = []
    paired: set[int] = set()

    for idx, domain_name in enumerate(base.domain_names):
        if idx in paired:
            continue
        if domain_name.startswith("dolma3_cc/") and domain_name.endswith("_high"):
            low_name = domain_name[:-5] + "_low"
            if low_name in base.domain_names:
                low_idx = base.domain_names.index(low_name)
                pairs.append((idx, low_idx))
                pair_topics.append(domain_name[len("dolma3_cc/") : -5])
                paired.add(idx)
                paired.add(low_idx)

    family_map = {family_name: [] for family_name in GENERIC_FAMILY_NAMES}
    for idx, domain_name in enumerate(base.domain_names):
        family_map[_family_for_domain(domain_name, family_scheme)].append(idx)
    missing = [family_name for family_name, members in family_map.items() if not members]
    if missing:
        raise ValueError(f"Family scheme {family_scheme!r} produced empty families: {missing}")

    return GenericFamilyPacket(
        base=base,
        pairs=pairs,
        pair_topics=pair_topics,
        singletons=[idx for idx in range(base.m) if idx not in paired],
        family_map=family_map,
    )


def _read_optional_csv(path: Path | str) -> pd.DataFrame:
    value = str(path)
    if not value:
        return pd.DataFrame()
    try:
        return pd.read_csv(value)
    except FileNotFoundError:
        return pd.DataFrame()


def _list_gcs_paths(pattern: str) -> list[str]:
    output = subprocess.check_output(["gsutil", "ls", pattern], text=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _list_optional_gcs_paths(pattern: str) -> list[str]:
    try:
        return _list_gcs_paths(pattern)
    except subprocess.CalledProcessError:
        return []


def _read_gcs_csv(uri: str) -> pd.DataFrame:
    data = subprocess.check_output(["gsutil", "cat", uri], text=True)
    return pd.read_csv(io.StringIO(data))


def _merge_metric_columns(frame: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column in metrics.columns if column.startswith(("eval/", "lm_eval/"))]
    if not metric_columns:
        return frame
    slim = metrics[["checkpoint_root", *metric_columns]].drop_duplicates(subset=["checkpoint_root"], keep="last")
    out = frame.merge(slim, on="checkpoint_root", how="left", suffixes=("", "_extra"))
    for metric_column in metric_columns:
        extra_column = f"{metric_column}_extra"
        if extra_column not in out.columns:
            continue
        if metric_column in frame.columns:
            out[metric_column] = out[metric_column].combine_first(out[extra_column])
        else:
            out[metric_column] = out[extra_column]
        out = out.drop(columns=[extra_column])
    return out


def _load_olmo_base_easy_overlap_metrics() -> pd.DataFrame:
    qsplit_uris = sorted(_list_gcs_paths(OLMO_BASE_EASY_OVERLAP_RESULTS_GLOB))
    if len(qsplit_uris) != 8:
        raise ValueError(f"Expected 8 OLMoBase easy-overlap qsplit shards, found {len(qsplit_uris)}: {qsplit_uris}")
    baseline_uris = sorted(_list_gcs_paths(OLMO_BASE_EASY_OVERLAP_SELECTED_BASELINES_GLOB))
    if len(baseline_uris) != 3:
        raise ValueError(
            f"Expected 3 OLMoBase easy-overlap selected-baseline shards, found {len(baseline_uris)}: {baseline_uris}"
        )
    oneoff_uris = sorted(_list_optional_gcs_paths(OLMO_BASE_EASY_OVERLAP_OLMIX_BPB_GLOB))
    frame = pd.concat([_read_gcs_csv(uri) for uri in (*qsplit_uris, *baseline_uris, *oneoff_uris)], ignore_index=True)
    if frame["run_name"].duplicated().any():
        duplicated = sorted(frame.loc[frame["run_name"].duplicated(), "run_name"].unique())
        raise ValueError(f"Duplicate OLMoBase easy-overlap run names: {duplicated[:8]}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OLMO_BASE_EASY_OVERLAP_CACHE_CSV, index=False)
    return frame


def _load_fit_swarm_metrics(downstream_results_csvs: list[str]) -> pd.DataFrame:
    fit = pd.read_csv(FIT_DATASET_CSV)
    if len(fit) != 242:
        raise ValueError(f"Expected 242 fit-swarm rows in {FIT_DATASET_CSV}, found {len(fit)}")
    wide = pd.read_csv(METRICS_WIDE_CSV, low_memory=False)
    metric_columns = [column for column in wide.columns if column.startswith(("eval/", "lm_eval/"))]
    wide_metrics = wide[["checkpoint_root", *metric_columns]].drop_duplicates(subset=["checkpoint_root"], keep="first")
    frame = fit.merge(wide_metrics, on="checkpoint_root", how="left")
    if len(frame) != len(fit):
        raise ValueError("Fit-swarm merge against metrics_wide changed row count")

    downstream_frames = [_read_optional_csv(BASELINE_DOWNSTREAM_METRICS_CSV)]
    downstream_frames.extend(_read_optional_csv(path) for path in downstream_results_csvs)
    for downstream in downstream_frames:
        if downstream.empty or "checkpoint_root" not in downstream.columns:
            continue
        frame = _merge_metric_columns(frame, downstream)

    overlap = _load_olmo_base_easy_overlap_metrics()
    missing_overlap = sorted(set(frame["run_name"]) - set(overlap["run_name"]))
    overlap_metric_columns = [column for column in overlap.columns if column.startswith(("eval/", "lm_eval/"))]
    frame = frame.merge(
        overlap[["run_name", *overlap_metric_columns]].drop_duplicates(subset=["run_name"], keep="last"),
        on="run_name",
        how="left",
        suffixes=("", "_overlap"),
    )
    if len(frame) != len(fit):
        raise ValueError("Fit-swarm merge against OLMoBase easy-overlap results changed row count")
    for metric_column in overlap_metric_columns:
        overlap_column = f"{metric_column}_overlap"
        if overlap_column not in frame.columns:
            continue
        if metric_column in fit.columns:
            frame[metric_column] = frame[metric_column].combine_first(frame[overlap_column])
        elif metric_column in frame.columns and metric_column != overlap_column:
            frame[metric_column] = frame[metric_column].combine_first(frame[overlap_column])
        else:
            frame[metric_column] = frame[overlap_column]
        frame = frame.drop(columns=[overlap_column])
    frame["olmo_base_easy_overlap_missing"] = frame["run_name"].isin(missing_overlap)

    missing_keys = frame["registry_run_key"].isna()
    if missing_keys.any():
        raise ValueError("Some fit-swarm rows did not match metrics_wide registry keys")
    return frame


def _add_aggregate_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["benchmark_accuracy_mean"] = out[[MMLU_ACC, GSM8K_ACC, HUMANEVAL_PASS]].mean(axis=1, skipna=False)
    out["benchmark_accuracy_available_mean"] = out[[MMLU_ACC, GSM8K_ACC, HUMANEVAL_PASS]].mean(axis=1, skipna=True)
    out["benchmark_accuracy_available_count"] = out[[MMLU_ACC, GSM8K_ACC, HUMANEVAL_PASS]].notna().sum(axis=1)
    if all(column in out.columns for column in OLMO_BASE_EASY_OVERLAP_ACCURACY_COLUMNS):
        easy_accuracy_columns = list(OLMO_BASE_EASY_OVERLAP_ACCURACY_COLUMNS)
        out["olmo_base_easy_overlap_accuracy_mean"] = out[easy_accuracy_columns].mean(axis=1, skipna=False)
        out["olmo_base_easy_overlap_accuracy_available_mean"] = out[easy_accuracy_columns].mean(axis=1, skipna=True)
        out["olmo_base_easy_overlap_accuracy_available_count"] = out[easy_accuracy_columns].notna().sum(axis=1)
        clipped = out[easy_accuracy_columns].clip(ACCURACY_AGGREGATE_EPS, 1.0 - ACCURACY_AGGREGATE_EPS)
        out["olmo_base_easy_overlap_accuracy_task_logit_mean"] = clipped.map(logit).mean(axis=1, skipna=False)
        out["olmo_base_easy_overlap_accuracy_task_probit_mean"] = clipped.map(ndtri).mean(axis=1, skipna=False)
        standardized = out[easy_accuracy_columns].copy()
        standardized = (standardized - standardized.mean(axis=0, skipna=True)) / standardized.std(
            axis=0, ddof=1, skipna=True
        )
        out["olmo_base_easy_overlap_accuracy_zmean"] = standardized.mean(axis=1, skipna=False)
        rank_normalized = pd.DataFrame(index=out.index)
        for column in easy_accuracy_columns:
            values = out[column]
            non_null = values.notna()
            transformed = pd.Series(np.nan, index=out.index, dtype=float)
            ranks = stats.rankdata(values.loc[non_null].to_numpy(dtype=float), method="average")
            quantiles = (ranks - 0.5) / float(len(ranks))
            transformed.loc[non_null] = ndtri(np.clip(quantiles, ACCURACY_AGGREGATE_EPS, 1.0 - ACCURACY_AGGREGATE_EPS))
            rank_normalized[column] = transformed
        out["olmo_base_easy_overlap_accuracy_task_rank_zmean"] = rank_normalized.mean(axis=1, skipna=False)
    else:
        out["olmo_base_easy_overlap_accuracy_mean"] = np.nan
        out["olmo_base_easy_overlap_accuracy_available_mean"] = np.nan
        out["olmo_base_easy_overlap_accuracy_available_count"] = 0
        out["olmo_base_easy_overlap_accuracy_task_logit_mean"] = np.nan
        out["olmo_base_easy_overlap_accuracy_task_probit_mean"] = np.nan
        out["olmo_base_easy_overlap_accuracy_zmean"] = np.nan
        out["olmo_base_easy_overlap_accuracy_task_rank_zmean"] = np.nan
    return out


def _objective_specs(frame: pd.DataFrame) -> tuple[AggregateObjective, ...]:
    specs = [
        AggregateObjective(
            slug="mmlu_acc_raw",
            source_column=MMLU_ACC,
            display_name="MMLU 5-shot accuracy raw-probability",
            family="accuracy",
            higher_is_better=True,
            transform="raw_probability",
        ),
        AggregateObjective(
            slug="mmlu_acc_logit",
            source_column=MMLU_ACC,
            display_name="MMLU 5-shot accuracy logit-probability",
            family="accuracy",
            higher_is_better=True,
            transform="logit_probability",
        ),
        AggregateObjective(
            slug="mmlu_bpb",
            source_column=MMLU_BPB,
            display_name="MMLU 5-shot BPB",
            family="bpb",
            higher_is_better=False,
            transform="identity",
        ),
        AggregateObjective(
            slug="mmlu_choice_logprob",
            source_column=MMLU_CHOICE_LOGPROB,
            display_name="MMLU 5-shot choice logprob",
            family="logprob",
            higher_is_better=True,
            transform="identity",
        ),
    ]
    if frame["benchmark_accuracy_mean"].notna().sum() >= MIN_OBJECTIVE_ROWS:
        specs.extend(
            [
                AggregateObjective(
                    slug="benchmark_accuracy_mean_raw",
                    source_column="benchmark_accuracy_mean",
                    display_name="Mean(MMLU, GSM8K, HumanEval) raw-probability",
                    family="accuracy",
                    higher_is_better=True,
                    transform="raw_probability",
                ),
                AggregateObjective(
                    slug="benchmark_accuracy_mean_logit",
                    source_column="benchmark_accuracy_mean",
                    display_name="Mean(MMLU, GSM8K, HumanEval) logit-probability",
                    family="accuracy",
                    higher_is_better=True,
                    transform="logit_probability",
                ),
            ]
        )
    if (
        OLMO_BASE_EASY_OVERLAP_MACRO_BPB in frame.columns
        and frame[OLMO_BASE_EASY_OVERLAP_MACRO_BPB].notna().sum() >= MIN_OBJECTIVE_ROWS
    ):
        specs.append(
            AggregateObjective(
                slug="olmo_base_easy_overlap_macro_bpb",
                source_column=OLMO_BASE_EASY_OVERLAP_MACRO_BPB,
                display_name="OLMoBaseEval easy-overlap macro BPB",
                family="bpb",
                higher_is_better=False,
                transform="identity",
            )
        )
    if frame["olmo_base_easy_overlap_accuracy_mean"].notna().sum() >= MIN_OBJECTIVE_ROWS:
        specs.extend(
            [
                AggregateObjective(
                    slug="olmo_base_easy_overlap_accuracy_raw",
                    source_column="olmo_base_easy_overlap_accuracy_mean",
                    display_name="OLMoBaseEval easy-overlap mean accuracy raw-probability",
                    family="accuracy",
                    higher_is_better=True,
                    transform="raw_probability",
                ),
                AggregateObjective(
                    slug="olmo_base_easy_overlap_accuracy_logit",
                    source_column="olmo_base_easy_overlap_accuracy_mean",
                    display_name="OLMoBaseEval easy-overlap mean accuracy logit-probability",
                    family="accuracy",
                    higher_is_better=True,
                    transform="logit_probability",
                ),
                AggregateObjective(
                    slug="olmo_base_easy_overlap_accuracy_arcsin_sqrt",
                    source_column="olmo_base_easy_overlap_accuracy_mean",
                    display_name="OLMoBaseEval easy-overlap mean accuracy arcsin-sqrt probability",
                    family="accuracy",
                    higher_is_better=True,
                    transform="arcsin_sqrt_probability",
                ),
                AggregateObjective(
                    slug="olmo_base_easy_overlap_accuracy_probit",
                    source_column="olmo_base_easy_overlap_accuracy_mean",
                    display_name="OLMoBaseEval easy-overlap mean accuracy probit-probability",
                    family="accuracy",
                    higher_is_better=True,
                    transform="probit_probability",
                ),
                AggregateObjective(
                    slug="olmo_base_easy_overlap_accuracy_rank_normal",
                    source_column="olmo_base_easy_overlap_accuracy_mean",
                    display_name="OLMoBaseEval easy-overlap mean accuracy rank-normal diagnostic",
                    family="accuracy_rank",
                    higher_is_better=True,
                    transform="rank_normal",
                ),
                AggregateObjective(
                    slug="olmo_base_easy_overlap_accuracy_zmean",
                    source_column="olmo_base_easy_overlap_accuracy_zmean",
                    display_name="OLMoBaseEval easy-overlap per-task z-scored mean accuracy",
                    family="accuracy_zscore",
                    higher_is_better=True,
                    transform="identity",
                ),
                AggregateObjective(
                    slug="olmo_base_easy_overlap_accuracy_task_logit_mean",
                    source_column="olmo_base_easy_overlap_accuracy_task_logit_mean",
                    display_name="OLMoBaseEval easy-overlap per-task mean logit accuracy",
                    family="accuracy_logit_mean",
                    higher_is_better=True,
                    transform="identity",
                ),
                AggregateObjective(
                    slug="olmo_base_easy_overlap_accuracy_task_probit_mean",
                    source_column="olmo_base_easy_overlap_accuracy_task_probit_mean",
                    display_name="OLMoBaseEval easy-overlap per-task mean probit accuracy",
                    family="accuracy_probit_mean",
                    higher_is_better=True,
                    transform="identity",
                ),
                AggregateObjective(
                    slug="olmo_base_easy_overlap_accuracy_task_rank_zmean",
                    source_column="olmo_base_easy_overlap_accuracy_task_rank_zmean",
                    display_name="OLMoBaseEval easy-overlap per-task rank-normal mean accuracy",
                    family="accuracy_rank_zscore",
                    higher_is_better=True,
                    transform="identity",
                ),
            ]
        )
    return tuple(
        spec
        for spec in specs
        if spec.source_column in frame.columns and frame[spec.source_column].notna().sum() >= MIN_OBJECTIVE_ROWS
    )


def _to_model_target(values: np.ndarray, spec: AggregateObjective, prob_eps: float) -> np.ndarray:
    if spec.transform == "raw_probability":
        transformed = np.clip(values, prob_eps, 1.0 - prob_eps)
    elif spec.transform == "logit_probability":
        transformed = logit(np.clip(values, prob_eps, 1.0 - prob_eps))
    elif spec.transform == "arcsin_sqrt_probability":
        transformed = np.arcsin(np.sqrt(np.clip(values, prob_eps, 1.0 - prob_eps)))
    elif spec.transform == "probit_probability":
        transformed = ndtri(np.clip(values, prob_eps, 1.0 - prob_eps))
    elif spec.transform == "rank_normal":
        ranks = stats.rankdata(values, method="average")
        quantiles = (ranks - 0.5) / float(len(values))
        transformed = ndtri(np.clip(quantiles, prob_eps, 1.0 - prob_eps))
    elif spec.transform == "identity":
        transformed = values.astype(float)
    else:
        raise ValueError(f"Unsupported transform: {spec.transform}")
    return -transformed if spec.higher_is_better else transformed


def _model_target_to_metric(values: np.ndarray | float, spec: AggregateObjective) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    signed = -array if spec.higher_is_better else array
    if spec.transform == "logit_probability":
        return expit(signed)
    if spec.transform == "arcsin_sqrt_probability":
        return np.sin(signed) ** 2
    if spec.transform == "probit_probability":
        return stats.norm.cdf(signed)
    if spec.transform == "rank_normal":
        return signed
    return signed


def _packet_from_frame(frame: pd.DataFrame, spec: AggregateObjective, prob_eps: float, family_scheme: str):
    fit_frame = frame.copy()
    fit_frame[MODEL_TARGET_COLUMN] = _to_model_target(
        fit_frame[spec.source_column].to_numpy(dtype=float), spec, prob_eps
    )
    loop = build_two_phase_many_loop_config(
        objective_metric=MODEL_TARGET_COLUMN,
        name=f"grp_no_l2_{spec.slug}",
    )
    dataset_spec = build_dataset_spec_from_frame(
        fit_frame,
        objective_metric=MODEL_TARGET_COLUMN,
        name=f"grp_no_l2_{spec.slug}",
        loop=loop,
    )
    base = PacketData(
        frame=fit_frame.reset_index(drop=True),
        name_col="run_name",
        y=dataset_spec.y,
        w=dataset_spec.weights,
        m=dataset_spec.M,
        c0=np.asarray(dataset_spec.epoch_multipliers[0], dtype=float),
        c1=np.asarray(dataset_spec.epoch_multipliers[1], dtype=float),
        domain_names=list(dataset_spec.domain_names),
    )
    return _generic_family_packet_from_base(base, family_scheme)


def _expanded_start_bank(random_starts: int) -> tuple[dict[str, float], ...]:
    starts = list(_start_bank())
    if random_starts <= 0:
        return tuple(starts)
    rng = np.random.default_rng(0)
    seed_pack = _pack_no_l2_params(starts[0])
    for _ in range(random_starts):
        starts.append(_unpack_no_l2_params(seed_pack + rng.normal(0.0, 0.85, size=seed_pack.shape)))
    seen: set[tuple[tuple[str, float], ...]] = set()
    deduped: list[dict[str, float]] = []
    for row in starts:
        key = tuple(sorted((name, round(float(value), 8)) for name, value in row.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append({name: float(value) for name, value in row.items()})
    return tuple(deduped)


def _oof_target_metrics(packet, params: dict[str, float]) -> dict[str, float]:
    y = packet.base.y
    kf = KFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    oof = np.zeros_like(y)
    fold_regrets: list[float] = []
    for train_idx, test_idx in kf.split(packet.base.w):
        model = build_penalty_calibration_surrogate(packet, params=params, variant_name=VARIANT_NAME).fit(
            packet.base.w[train_idx],
            y[train_idx],
        )
        pred = model.predict(packet.base.w[test_idx])
        oof[test_idx] = pred
        fold_regrets.append(float(y[test_idx][int(np.argmin(pred))] - np.min(y[test_idx])))
    residuals = oof - y
    tail_count = max(5, int(np.ceil(float(LOWER_TAIL_FRAC) * float(len(y)))))
    tail_idx = np.argsort(oof)[:tail_count]
    lower_tail_optimism = float(np.mean(np.maximum(y[tail_idx] - oof[tail_idx], 0.0)))
    mean_regret = float(np.mean(fold_regrets))
    objective = (
        CALIBRATION_CV_WEIGHT * float(np.sqrt(np.mean(residuals**2)))
        + CALIBRATION_FOLDMEAN_WEIGHT * mean_regret
        + CALIBRATION_TAIL_WEIGHT * lower_tail_optimism
    )
    return {
        "target_cv_rmse": float(np.sqrt(np.mean(residuals**2))),
        "target_cv_mae": float(np.mean(np.abs(residuals))),
        "target_cv_spearman": float(stats.spearmanr(y, oof).statistic),
        "target_cv_foldmean_regret_at_1": mean_regret,
        "target_lower_tail_optimism": lower_tail_optimism,
        "objective": float(objective),
    }


def _coarse_rows(packet, start_bank: tuple[dict[str, float], ...]) -> pd.DataFrame:
    rows = [
        {
            "stage": "coarse",
            "start_id": int(start_id),
            **params,
            **_oof_target_metrics(packet, params),
        }
        for start_id, params in enumerate(start_bank)
    ]
    return pd.DataFrame.from_records(rows).sort_values(
        ["objective", "target_cv_rmse", "target_cv_foldmean_regret_at_1"],
        ascending=[True, True, True],
    )


def _refine_rows(packet, start_bank: tuple[dict[str, float], ...], coarse_top_k: int, method: str):
    coarse = _coarse_rows(packet, start_bank)
    chosen_ids = coarse["start_id"].head(int(coarse_top_k)).tolist()
    best_row: dict[str, Any] | None = None
    refine_rows: list[dict[str, Any]] = []
    for chosen_rank, start_id in enumerate(chosen_ids):
        start = _pack_no_l2_params(start_bank[start_id])
        cache: dict[tuple[float, ...], float] = {}

        def objective(z: np.ndarray, _cache: dict[tuple[float, ...], float] = cache) -> float:
            key = tuple(np.round(np.asarray(z, dtype=float), 8))
            if key not in _cache:
                params = _unpack_no_l2_params(z)
                _cache[key] = _oof_target_metrics(packet, params)["objective"]
            return _cache[key]

        options = {
            "L-BFGS-B": {"maxiter": 160, "ftol": 1e-8},
            "Nelder-Mead": {"maxiter": 900, "xatol": 1e-4, "fatol": 1e-8},
            "Powell": {"maxiter": 120, "xtol": 1e-4, "ftol": 1e-8},
        }.get(method, {"maxiter": 240})
        result = minimize(objective, start, method=method, options=options)
        params = _unpack_no_l2_params(np.asarray(result.x, dtype=float))
        row = {
            "stage": "refine",
            "chosen_rank": int(chosen_rank),
            "start_id": int(start_id),
            "success": bool(result.success),
            "message": str(result.message),
            **params,
            **_oof_target_metrics(packet, params),
        }
        refine_rows.append(row)
        if best_row is None or float(row["objective"]) < float(best_row["objective"]):
            best_row = row
    if best_row is None:
        raise RuntimeError("No refined fit produced a best row")
    return coarse, best_row, pd.DataFrame.from_records(refine_rows).sort_values("objective")


def _family_shares(packet, weights: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for phase_idx, phase_name in enumerate(("phase0", "phase1")):
        for family_name, indices in packet.family_map.items():
            out[f"raw_{phase_name}_{family_name}_share"] = float(np.sum(weights[phase_idx, list(indices)]))
        out[f"raw_{phase_name}_support_gt_1e4"] = int(np.sum(weights[phase_idx] > 1e-4))
        positive_weights = weights[phase_idx][weights[phase_idx] > 0.0]
        entropy = -float(np.sum(positive_weights * np.log(positive_weights)))
        out[f"raw_{phase_name}_entropy"] = entropy
    return out


def _prediction_rows(packet, params: dict[str, float], spec: AggregateObjective) -> pd.DataFrame:
    y_target = packet.base.y
    y_metric = _model_target_to_metric(y_target, spec)
    kf = KFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    oof_target = np.zeros_like(y_target)
    for train_idx, test_idx in kf.split(packet.base.w):
        model = build_penalty_calibration_surrogate(packet, params=params, variant_name=VARIANT_NAME).fit(
            packet.base.w[train_idx],
            y_target[train_idx],
        )
        oof_target[test_idx] = model.predict(packet.base.w[test_idx])
    return pd.DataFrame(
        {
            "run_name": packet.base.frame[packet.base.name_col].astype(str).to_numpy(),
            "actual_metric": y_metric,
            "predicted_metric": _model_target_to_metric(oof_target, spec),
            "actual_model_target": y_target,
            "predicted_model_target": oof_target,
        }
    )


def _plot_predictions(rows: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    low = float(min(rows["actual_metric"].min(), rows["predicted_metric"].min()))
    high = float(max(rows["actual_metric"].max(), rows["predicted_metric"].max()))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rows["actual_metric"],
            y=rows["predicted_metric"],
            mode="markers",
            marker={"size": 8, "color": rows["actual_metric"], "colorscale": "RdYlGn_r", "showscale": True},
            text=rows["run_name"],
            hovertemplate="run=%{text}<br>actual=%{x:.6f}<br>pred=%{y:.6f}<extra></extra>",
            name="OOF predictions",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            line={"color": "black", "dash": "dash"},
            hoverinfo="skip",
            name="y=x",
        )
    )
    fig.update_layout(title=title, xaxis_title="Actual metric", yaxis_title="OOF predicted metric")
    fig.write_html(path)


def _plot_residuals(rows: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    residual = rows["predicted_metric"] - rows["actual_metric"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rows["actual_metric"],
            y=residual,
            mode="markers",
            marker={"size": 8, "color": rows["actual_metric"], "colorscale": "RdYlGn_r", "showscale": True},
            text=rows["run_name"],
            hovertemplate="run=%{text}<br>actual=%{x:.6f}<br>residual=%{y:.6f}<extra></extra>",
            name="Residuals",
        )
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="black")
    fig.update_layout(title=title, xaxis_title="Actual metric", yaxis_title="OOF predicted - actual")
    fig.write_html(path)


def _fit_objective(
    frame: pd.DataFrame,
    spec: AggregateObjective,
    *,
    family_scheme: str,
    method: str,
    coarse_top_k: int,
    random_starts: int,
    prob_eps: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit_frame = frame.loc[frame[spec.source_column].notna()].reset_index(drop=True)
    if len(fit_frame) < MIN_OBJECTIVE_ROWS:
        raise ValueError(f"{spec.slug} has only {len(fit_frame)} non-null rows")
    packet = _packet_from_frame(fit_frame, spec, prob_eps, family_scheme)
    start_bank = _expanded_start_bank(random_starts)
    coarse, best, refine = _refine_rows(packet, start_bank, coarse_top_k, method)
    params = {key: float(best[key]) for key in _no_l2_param_keys()}
    params["reg"] = REG_FIXED
    model = build_penalty_calibration_surrogate(packet, params=params, variant_name=VARIANT_NAME).fit(
        packet.base.w,
        packet.base.y,
    )
    full_metrics = compute_penalty_calibration_metrics(packet, model, seed=CV_SEED)
    raw_result, phase0, phase1 = optimize_penalty_calibration_model(packet, model, seed=CV_SEED)
    raw_weights = np.stack([phase0, phase1], axis=0)
    raw_distances = average_phase_tv_distance(packet.base.w, raw_weights[None, :, :])
    nearest_idx = int(np.argmin(raw_distances))
    train_pred_target = model.predict(packet.base.w)
    pred_rows = _prediction_rows(packet, params, spec)
    predicted_observed_idx = int(np.argmin(train_pred_target))
    best_observed_idx = int(
        np.argmax(pred_rows["actual_metric"]) if spec.higher_is_better else np.argmin(pred_rows["actual_metric"])
    )
    metric_residual = pred_rows["predicted_metric"] - pred_rows["actual_metric"]
    best_metric = float(pred_rows.loc[best_observed_idx, "actual_metric"])
    predicted_observed_metric = float(pred_rows.loc[predicted_observed_idx, "actual_metric"])
    raw_metric = float(_model_target_to_metric(float(raw_result.fun), spec))
    raw_nearest_metric = float(pred_rows.loc[nearest_idx, "actual_metric"])
    regret_sign = 1.0 if spec.higher_is_better else -1.0

    objective_dir = (
        OUTPUT_DIR / spec.slug
        if family_scheme == DEFAULT_FAMILY_SCHEME
        else OUTPUT_DIR / f"{spec.slug}__{family_scheme}"
    )
    objective_dir.mkdir(parents=True, exist_ok=True)
    coarse.to_csv(objective_dir / "coarse.csv", index=False)
    refine.to_csv(objective_dir / "refine.csv", index=False)
    pred_rows.to_csv(objective_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(
        {
            "domain_name": packet.base.domain_names,
            "phase0_weight": phase0,
            "phase0_epochs": phase0 * packet.base.c0,
            "phase1_weight": phase1,
            "phase1_epochs": phase1 * packet.base.c1,
        }
    ).to_csv(objective_dir / "raw_optimum_weights.csv", index=False)
    _plot_predictions(pred_rows, objective_dir / "predicted_vs_actual.html", spec.display_name)
    _plot_residuals(pred_rows, objective_dir / "residuals.html", spec.display_name)

    summary = {
        "slug": spec.slug,
        "family_scheme": family_scheme,
        "source_column": spec.source_column,
        "display_name": spec.display_name,
        "family": spec.family,
        "higher_is_better": spec.higher_is_better,
        "transform": spec.transform,
        "n": len(packet.base.y),
        "method": method,
        "coarse_top_k": int(coarse_top_k),
        "start_bank_size": len(start_bank),
        "metric_oof_rmse": float(np.sqrt(np.mean(metric_residual**2))),
        "metric_oof_mae": float(np.mean(np.abs(metric_residual))),
        "metric_oof_spearman": float(
            stats.spearmanr(pred_rows["actual_metric"], pred_rows["predicted_metric"]).statistic
        ),
        "best_observed_run_name": str(pred_rows.loc[best_observed_idx, "run_name"]),
        "best_observed_metric": best_metric,
        "predicted_observed_run_name": str(pred_rows.loc[predicted_observed_idx, "run_name"]),
        "predicted_observed_metric": predicted_observed_metric,
        "predicted_observed_regret": regret_sign * (best_metric - predicted_observed_metric),
        "raw_predicted_optimum_metric": raw_metric,
        "raw_nearest_observed_run_name": str(pred_rows.loc[nearest_idx, "run_name"]),
        "raw_nearest_observed_metric": raw_nearest_metric,
        "raw_nearest_observed_regret": regret_sign * (best_metric - raw_nearest_metric),
        "raw_nearest_observed_tv": float(raw_distances[nearest_idx]),
        **{
            key: float(value)
            for key, value in full_metrics.items()
            if isinstance(value, int | float | np.integer | np.floating)
        },
        **{
            key: float(value) for key, value in best.items() if isinstance(value, int | float | np.integer | np.floating)
        },
        **_family_shares(packet, raw_weights),
    }
    params_row = {"slug": spec.slug, "family_scheme": family_scheme, **params, **_parameter_counts(packet)}
    return summary, params_row


def _write_report(summary: pd.DataFrame, targets: pd.DataFrame) -> None:
    coverage = {
        MMLU_ACC: int(targets[MMLU_ACC].notna().sum()) if MMLU_ACC in targets else 0,
        GSM8K_ACC: int(targets[GSM8K_ACC].notna().sum()) if GSM8K_ACC in targets else 0,
        HUMANEVAL_PASS: int(targets[HUMANEVAL_PASS].notna().sum()) if HUMANEVAL_PASS in targets else 0,
        "benchmark_accuracy_mean": int(targets["benchmark_accuracy_mean"].notna().sum()),
        OLMO_BASE_EASY_OVERLAP_MACRO_BPB: (
            int(targets[OLMO_BASE_EASY_OVERLAP_MACRO_BPB].notna().sum())
            if OLMO_BASE_EASY_OVERLAP_MACRO_BPB in targets
            else 0
        ),
        "olmo_base_easy_overlap_accuracy_mean": int(targets["olmo_base_easy_overlap_accuracy_mean"].notna().sum()),
        "olmo_base_easy_overlap_accuracy_zmean": int(targets["olmo_base_easy_overlap_accuracy_zmean"].notna().sum()),
        "olmo_base_easy_overlap_accuracy_task_logit_mean": int(
            targets["olmo_base_easy_overlap_accuracy_task_logit_mean"].notna().sum()
        ),
        "olmo_base_easy_overlap_accuracy_task_probit_mean": int(
            targets["olmo_base_easy_overlap_accuracy_task_probit_mean"].notna().sum()
        ),
        "olmo_base_easy_overlap_accuracy_task_rank_zmean": int(
            targets["olmo_base_easy_overlap_accuracy_task_rank_zmean"].notna().sum()
        ),
    }
    columns = [
        "family_scheme",
        "display_name",
        "n",
        "metric_oof_rmse",
        "metric_oof_spearman",
        "best_observed_metric",
        "predicted_observed_metric",
        "predicted_observed_regret",
        "raw_predicted_optimum_metric",
        "raw_nearest_observed_metric",
        "raw_nearest_observed_regret",
        "raw_nearest_observed_tv",
        "raw_phase1_broad_text_share",
        "raw_phase1_tech_code_share",
        "raw_phase1_reasoning_share",
    ]
    body = [
        "# 60M Benchmark-Aggregate GRP Fits",
        "",
        "## Coverage",
        "",
        pd.Series(coverage, name="non_null_rows").to_frame().to_markdown(),
        "",
        "## Fit Summary",
        "",
        (
            summary[columns].to_markdown(index=False, floatfmt=".6f")
            if not summary.empty
            else "No complete objectives to fit."
        ),
        "",
        "## Notes",
        "",
        "- MMLU is already complete for the 242-row fit swarm.",
        "- Full benchmark mean is only fit once MMLU, GSM8K, and HumanEval are non-null for every row.",
        "- Accuracy objectives include both raw probability and logit-probability fits when complete.",
        "- OLMoBaseEval easy-overlap metrics are read from the qsplit240 and selected-baseline reruns when available.",
        "- OLMoBaseEval objectives use the rows covered by qsplit240, selected-baseline, and optional Olmix BPB reruns.",
        "",
    ]
    REPORT_MD.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    family_schemes = tuple(args.family_scheme) if args.family_scheme else (DEFAULT_FAMILY_SCHEME,)
    frame = _add_aggregate_targets(_load_fit_swarm_metrics(args.downstream_results_csv))
    frame.to_csv(TARGETS_CSV, index=False)
    specs = _objective_specs(frame)
    if args.only_slug_prefix:
        prefixes = tuple(str(prefix) for prefix in args.only_slug_prefix)
        specs = tuple(spec for spec in specs if spec.slug.startswith(prefixes))
        if not specs:
            raise ValueError(f"No objective specs matched prefixes {prefixes!r}")
    rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    for family_scheme in family_schemes:
        for spec in specs:
            summary, params = _fit_objective(
                frame,
                spec,
                family_scheme=family_scheme,
                method=args.method,
                coarse_top_k=args.coarse_top_k,
                random_starts=args.random_starts,
                prob_eps=args.prob_eps,
            )
            rows.append(summary)
            param_rows.append(params)
            print(
                f"fit {spec.slug} family_scheme={family_scheme}: "
                f"metric_oof_rmse={summary['metric_oof_rmse']:.6f} "
                f"metric_oof_spearman={summary['metric_oof_spearman']:.6f}"
            )
    summary_frame = pd.DataFrame.from_records(rows)
    params_frame = pd.DataFrame.from_records(param_rows)
    summary_frame.to_csv(SUMMARY_CSV, index=False)
    params_frame.to_csv(PARAMS_CSV, index=False)
    SUMMARY_JSON.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(summary_frame, frame)
    print(f"Wrote {TARGETS_CSV}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {PARAMS_CSV}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
