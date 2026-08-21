# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
# ]
# ///
"""Build a coordinate-safe registry of the 60M, 39-bucket checkpoints.

The two-phase qsplit panel and its independently trained, exposure-matched
single-phase panel are distinct fit policy classes. Every remaining checkpoint
with complete weights and Uncheatable BPB is classified against the union of
those fit coordinates. Coordinate aliases and repeated observations remain
visible, but never enter the coordinate-disjoint heldout set.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix import launch_proportional_controllability_60m as pctrl60  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_paper_faithful_olmix_300m as table9,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/60m_39bucket_checkpoint_audit_20260724"
FIT_TWO_PATH = (
    SCRIPT_DIR / "metric_registry/fit_datasets/eval_uncheatable_eval_bpb__60m_1p2b__signal__fit_swarm_60m_default.csv"
)
FIT_ONE_MANIFEST_PATH = (
    SCRIPT_DIR / "reference_outputs/single_phase_exposure_average_60m_1p2b/single_phase_exposure_average_manifest.csv"
)
FIT_ONE_RESULTS_PATH = (
    SCRIPT_DIR / "reference_outputs/single_phase_exposure_average_60m_1p2b/analysis/single_phase_fit_dataset.csv"
)
TWO_PHASE_ALL_PATH = SCRIPT_DIR / "two_phase_many_all_60m_1p2b.csv"
LOGICAL_RUNS_PATH = SCRIPT_DIR / "run_registry/logical_runs.csv"
METRICS_WIDE_PATH = SCRIPT_DIR / "metric_registry/metrics_wide.csv"
NOISE_RESULTS_PATH = SCRIPT_DIR / "run00097_seed_study_backfill/results.csv"
PROPORTIONAL_NOISE_PATH = (
    SCRIPT_DIR / "metric_registry/raw_metric_matrix_300m/noise_baseline_proportional_variable_subset_60m_1p2b.csv"
)
OLMO_MANIFEST_PATH = (
    SCRIPT_DIR / "reference_outputs/olmo_base_easy_full_results_60m_300m_20260625/"
    "olmo_base_easy_full_results_60m_300m_manifest.csv"
)
OLMO_WIDE_PATH = OLMO_MANIFEST_PATH.with_name("olmo_base_easy_full_results_60m_300m_wide.csv")
TABLE9_GAP_RESULTS_PATH = SCRIPT_DIR / "reference_outputs/60m_table9_gap_completion_20260725/table9_eval_results.csv"
SCALE_SUMMARY_PATH = SCRIPT_DIR / "analysis_dataset/summary.json"

UNCEATABLE_COLUMN = "eval/uncheatable_eval/bpb"
PHASE_FRACTIONS = (0.8, 0.2)
FLOAT_TOLERANCE = 1e-9
POLICY_HASH_DECIMALS = 12
SINGLE_PHASE_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_single_phase_exposure_average_60m_1p2b"


def write_text_if_changed(path: Path, text: str) -> None:
    if path.exists() and path.read_text() == text:
        return
    path.write_text(text)


def write_frame_if_changed(path: Path, frame: pd.DataFrame) -> None:
    write_text_if_changed(path, frame.to_csv(index=False))


def canonical_domains(frame: pd.DataFrame) -> list[str]:
    domains = [
        column.removeprefix("phase_0_")
        for column in frame.columns
        if column.startswith("phase_0_") and f"phase_1_{column.removeprefix('phase_0_')}" in frame.columns
    ]
    if len(domains) != 39:
        raise ValueError(f"Expected 39 canonical domains, found {len(domains)}")
    return domains


def normalized_phase(frame: pd.DataFrame, domains: list[str], phase: int) -> np.ndarray:
    columns = [f"phase_{phase}_{domain}" for domain in domains]
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing phase-{phase} columns: {missing[:5]}")
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Phase-{phase} weights contain non-finite values")
    row_sums = values.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError(f"Phase-{phase} contains a non-positive row sum")
    return values / row_sums[:, None]


def policy_hash(phase0: np.ndarray, phase1: np.ndarray) -> str:
    payload = np.round(np.concatenate([phase0, phase1]), POLICY_HASH_DECIMALS).astype("<f8").tobytes()
    return hashlib.sha256(payload).hexdigest()


def add_standard_weights(
    frame: pd.DataFrame,
    domains: list[str],
    *,
    phase0: np.ndarray | None = None,
    phase1: np.ndarray | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    p0 = normalized_phase(result, domains, 0) if phase0 is None else np.asarray(phase0, dtype=float)
    p1 = normalized_phase(result, domains, 1) if phase1 is None else np.asarray(phase1, dtype=float)
    if p0.shape != (len(result), len(domains)) or p1.shape != p0.shape:
        raise ValueError("Weight arrays do not match the source frame")
    for index, domain in enumerate(domains):
        result[f"phase_0_{domain}"] = p0[:, index]
        result[f"phase_1_{domain}"] = p1[:, index]
    result["policy_hash"] = [policy_hash(row0, row1) for row0, row1 in zip(p0, p1, strict=True)]
    result["phase_tv"] = 0.5 * np.abs(p0 - p1).sum(axis=1)
    result["policy_class"] = np.where(result["phase_tv"] < FLOAT_TOLERANCE, "single_phase", "two_phase")
    return result


def standard_columns(domains: list[str]) -> list[str]:
    return [
        "observation_id",
        "run_name",
        "source_experiment",
        "source_family",
        "source_kind",
        "wandb_run_id",
        "checkpoint_root",
        "uncheatable_bpb",
        "table9_macro_bpb",
        "policy_hash",
        "policy_class",
        "phase_tv",
        "paired_run_name",
        "intervention_type",
        "target_domain",
        "direction_id",
        "direction_type",
        *[f"phase_0_{domain}" for domain in domains],
        *[f"phase_1_{domain}" for domain in domains],
    ]


def finalize_source(frame: pd.DataFrame, domains: list[str]) -> pd.DataFrame:
    result = add_standard_weights(frame, domains)
    for column in standard_columns(domains):
        if column not in result:
            result[column] = np.nan
    return result[standard_columns(domains)].copy()


def table9_lookup() -> dict[str, float]:
    wide = pd.read_csv(OLMO_WIDE_PATH, low_memory=False)
    wide = wide[wide["scale"].eq("60m_1p2b")].copy()
    for category, weights in table9.MMLU_CATEGORY_WEIGHTS.items():
        columns = [table9.mmlu_metric_key(task) for task in weights]
        coefficient = np.asarray([weights[task] for task in weights], dtype=float)
        wide[category] = wide[columns].to_numpy(dtype=float) @ coefficient
    components = table9.table9_component_order()
    missing = sorted(set(components).difference(wide.columns))
    if missing:
        raise ValueError(f"Native OLMoBaseEval export is missing Table-9 components: {missing}")
    values = wide[components].mean(axis=1)
    lookup = dict(zip(wide["run_name"].astype(str), values.astype(float), strict=True))

    gap_results = pd.read_csv(TABLE9_GAP_RESULTS_PATH, low_memory=False)
    required_columns = {"run_name", "table9_macro_bpb"}
    missing_gap_columns = sorted(required_columns.difference(gap_results.columns))
    if missing_gap_columns:
        raise ValueError(f"Table-9 gap results are missing columns: {missing_gap_columns}")
    if gap_results["run_name"].duplicated().any():
        duplicates = gap_results.loc[gap_results["run_name"].duplicated(keep=False), "run_name"].unique()
        raise ValueError(f"Table-9 gap results contain duplicate run names: {duplicates[:10].tolist()}")
    if gap_results["table9_macro_bpb"].isna().any():
        raise ValueError("Table-9 gap results contain missing macro BPB values")

    for row in gap_results.itertuples(index=False):
        run_name = str(row.run_name)
        macro_bpb = float(row.table9_macro_bpb)
        previous = lookup.get(run_name)
        if previous is not None and abs(previous - macro_bpb) > 1e-9:
            raise ValueError(f"Conflicting Table-9 values for {run_name}: {previous} != {macro_bpb}")
        lookup[run_name] = macro_bpb
    return lookup


def checkpoint_root_lookup() -> dict[str, str]:
    source = pd.read_csv(
        METRICS_WIDE_PATH,
        usecols=["scale", "wandb_run_id", "checkpoint_root"],
        low_memory=False,
    )
    source = source[
        source["scale"].eq("60m_1p2b") & source["wandb_run_id"].notna() & source["checkpoint_root"].notna()
    ].copy()
    source["wandb_run_id"] = source["wandb_run_id"].astype(str)
    source["checkpoint_root"] = source["checkpoint_root"].astype(str).str.rstrip("/")
    source = source[source["checkpoint_root"].ne("")]

    root_counts = source.groupby("wandb_run_id")["checkpoint_root"].nunique()
    conflicts = root_counts[root_counts.ne(1)]
    if not conflicts.empty:
        raise ValueError(f"Metric registry contains conflicting 60M checkpoint roots: {conflicts.index[:10].tolist()}")
    source = source.drop_duplicates("wandb_run_id")
    unexpected = source[~source["checkpoint_root"].str.startswith("gs://marin-us-east5/checkpoints/")]
    if not unexpected.empty:
        raise ValueError(
            f"Metric registry contains non-east5 60M checkpoint roots: {unexpected.head().to_dict('records')}"
        )
    return dict(zip(source["wandb_run_id"], source["checkpoint_root"], strict=True))


def load_fit_two(domains: list[str], table9_by_run: dict[str, float]) -> pd.DataFrame:
    source = pd.read_csv(FIT_TWO_PATH)
    result = source.rename(
        columns={
            "objective_metric": "uncheatable_bpb",
            "cohort": "source_family",
        }
    ).copy()
    result["observation_id"] = "fit_two:" + result["run_name"].astype(str)
    result["source_kind"] = "fit_two_phase"
    result["table9_macro_bpb"] = result["run_name"].astype(str).map(table9_by_run)
    result["paired_run_name"] = "singleavg_" + result["run_name"].astype(str)
    return finalize_source(result, domains)


def load_fit_one(domains: list[str], table9_by_run: dict[str, float]) -> pd.DataFrame:
    manifest = pd.read_csv(FIT_ONE_MANIFEST_PATH)
    results = pd.read_csv(FIT_ONE_RESULTS_PATH)
    source = manifest.merge(results, on="run_name", how="inner", validate="one_to_one")
    if len(source) != 242:
        raise ValueError(f"Expected 242 single-phase rows, found {len(source)}")
    result = source.rename(
        columns={
            UNCEATABLE_COLUMN: "uncheatable_bpb",
            "cohort": "source_family",
        }
    ).copy()
    result["observation_id"] = "fit_one:" + result["run_name"].astype(str)
    result["source_experiment"] = SINGLE_PHASE_SOURCE_EXPERIMENT
    result["source_kind"] = "fit_single_phase"
    result["table9_macro_bpb"] = result["run_name"].astype(str).map(table9_by_run)
    result["paired_run_name"] = result["source_run_name"]
    return finalize_source(result, domains)


def expand_coalesced_common_crawl(
    frame: pd.DataFrame,
    domains: list[str],
    reference: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    reference_row = reference.loc[reference["run_name"].eq("baseline_proportional")]
    if len(reference_row) != 1:
        raise ValueError("The two-phase fit panel must contain one proportional reference")
    reference_row = reference_row.iloc[0]
    for phase in (0, 1):
        for suffix in ("high", "low"):
            for domain in [name for name in domains if name.startswith("dolma3_cc/") and name.endswith(f"_{suffix}")]:
                column = f"phase_{phase}_{domain}"
                if column not in result:
                    result[column] = np.nan
                missing = result[column].isna()
                if not missing.any():
                    continue
                topic = domain.removesuffix(f"_{suffix}")
                aggregate_column = f"phase_{phase}_{topic}"
                peer = f"{topic}_{'low' if suffix == 'high' else 'high'}"
                denominator = float(reference_row[f"phase_{phase}_{domain}"] + reference_row[f"phase_{phase}_{peer}"])
                ratio = float(reference_row[f"phase_{phase}_{domain}"]) / denominator
                result.loc[missing, column] = result.loc[missing, aggregate_column] * ratio
    return result


def load_two_phase_archive(
    domains: list[str],
    table9_by_run: dict[str, float],
    checkpoint_by_wandb: dict[str, str],
    fit_two_reference: pd.DataFrame,
) -> pd.DataFrame:
    source = pd.read_csv(TWO_PHASE_ALL_PATH, low_memory=False)
    source = expand_coalesced_common_crawl(source, domains, fit_two_reference)
    result = source.rename(columns={UNCEATABLE_COLUMN: "uncheatable_bpb"}).copy()
    result["observation_id"] = (
        "two_archive:" + result["source_experiment"].astype(str) + ":" + result["run_name"].astype(str)
    )
    result["source_family"] = "two_phase_many_60m_archive"
    result["source_kind"] = "historical_two_phase"
    result["checkpoint_root"] = result["wandb_run_id"].astype(str).map(checkpoint_by_wandb)
    missing_roots = result[result["checkpoint_root"].isna()]
    if not missing_roots.empty:
        raise ValueError(
            "Historical 60M archive rows are missing checkpoint provenance: "
            f"{missing_roots[['run_name', 'wandb_run_id']].head().to_dict('records')}"
        )
    result["table9_macro_bpb"] = result["run_name"].astype(str).map(table9_by_run)
    return finalize_source(result, domains)


def load_logical_extras(domains: list[str], table9_by_run: dict[str, float]) -> pd.DataFrame:
    source = pd.read_csv(LOGICAL_RUNS_PATH, low_memory=False)
    source = source[source["scale"].eq("60m_1p2b")].copy()
    source = source[source["objective_metric_value"].notna()].copy()
    phase_columns = [f"phase_{phase}_{domain}" for phase in (0, 1) for domain in domains]
    source = source[source[phase_columns].notna().all(axis=1)].copy()
    result = source.rename(
        columns={
            "objective_metric_value": "uncheatable_bpb",
            "family": "source_family",
        }
    ).copy()
    result["observation_id"] = (
        "logical:" + result["source_experiment"].astype(str) + ":" + result["run_name"].astype(str)
    )
    result["source_kind"] = "logical_registry"
    result["table9_macro_bpb"] = result["run_name"].astype(str).map(table9_by_run)
    return finalize_source(result, domains)


def load_noise_repeats(domains: list[str], table9_by_run: dict[str, float]) -> pd.DataFrame:
    source = pd.read_csv(NOISE_RESULTS_PATH, low_memory=False)
    source = source[source["cohort"].eq("seed_sweep")].copy()
    if len(source) != 10:
        raise ValueError(f"Expected ten 60M seed repeats, found {len(source)}")
    result = source.rename(columns={UNCEATABLE_COLUMN: "uncheatable_bpb", "cohort": "source_family"}).copy()
    result["observation_id"] = "noise:" + result["run_name"].astype(str)
    result["source_experiment"] = "run00097_seed_study_backfill"
    result["source_kind"] = "repeat_noise"
    result["table9_macro_bpb"] = result["run_name"].astype(str).map(table9_by_run)
    return finalize_source(result, domains)


def load_proportional_noise_repeats(domains: list[str], table9_by_run: dict[str, float]) -> pd.DataFrame:
    source = pd.read_csv(PROPORTIONAL_NOISE_PATH, low_memory=False)
    if len(source) != 10:
        raise ValueError(f"Expected ten proportional 60M repeats, found {len(source)}")
    result = source.rename(columns={UNCEATABLE_COLUMN: "uncheatable_bpb"}).copy()
    result["observation_id"] = "proportional_noise:" + result["run_name"].astype(str)
    result["source_family"] = "proportional_variable_subset_noise_60m"
    result["source_kind"] = "proportional_noise"
    result["table9_macro_bpb"] = result["run_name"].astype(str).map(table9_by_run)
    return finalize_source(result, domains)


def read_final_uncheatable(checkpoint_root: str) -> float:
    metrics_uri = f"{checkpoint_root.rstrip('/')}/checkpoints/eval_metrics.jsonl"
    final_value: float | None = None
    final_step = -1
    with fsspec.open(metrics_uri, "rt") as handle:
        for line in handle:
            record = json.loads(line)
            if UNCEATABLE_COLUMN not in record:
                continue
            step = int(record.get("step", -1))
            if step >= final_step:
                final_step = step
                final_value = float(record[UNCEATABLE_COLUMN])
    if final_value is None:
        raise ValueError(f"No {UNCEATABLE_COLUMN} in {metrics_uri}")
    return final_value


def load_pctrl_metrics(manifest: pd.DataFrame) -> pd.DataFrame:
    cache_path = OUTPUT_DIR / "pctrl_uncheatable_metrics.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if set(cached["run_name"]) == set(manifest["run_name"]):
            return cached
    roots = dict(zip(manifest["run_name"].astype(str), manifest["checkpoint_root"].astype(str), strict=True))
    with ThreadPoolExecutor(max_workers=16) as executor:
        values = list(executor.map(read_final_uncheatable, roots.values()))
    cached = pd.DataFrame({"run_name": list(roots), "uncheatable_bpb": values})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_frame_if_changed(cache_path, cached)
    return cached


def load_proportional_controllability(domains: list[str], table9_by_run: dict[str, float]) -> pd.DataFrame:
    specs = pctrl60.build_run_specs()
    manifest = pd.read_csv(OLMO_MANIFEST_PATH)
    manifest = manifest[(manifest["scale"] == "60m_1p2b") & (manifest["panel"] == "proportional_controllability")]
    if len(specs) != 117 or len(manifest) != 117:
        raise ValueError(f"Expected 117 proportional interventions, found {len(specs)} specs and {len(manifest)} runs")
    metrics = load_pctrl_metrics(manifest)
    metadata = manifest.merge(metrics, on="run_name", how="inner", validate="one_to_one").set_index("run_name")
    rows: list[dict[str, Any]] = []
    for spec in specs:
        record = metadata.loc[spec.run_name]
        row: dict[str, Any] = {
            "observation_id": f"pctrl:{spec.run_name}",
            "run_name": spec.run_name,
            "source_experiment": record["source_experiment"],
            "source_family": "proportional_controllability_60m",
            "source_kind": "proportional_controllability",
            "wandb_run_id": record["wandb_run_id"],
            "checkpoint_root": record["checkpoint_root"],
            "uncheatable_bpb": record["uncheatable_bpb"],
            "table9_macro_bpb": table9_by_run.get(spec.run_name, np.nan),
            "intervention_type": spec.intervention_type,
            "target_domain": spec.target_domain,
            "direction_id": spec.direction_id,
            "direction_type": spec.direction_type,
        }
        for phase in (0, 1):
            for domain in domains:
                row[f"phase_{phase}_{domain}"] = spec.phase_weights[f"phase_{phase}"][domain]
        rows.append(row)
    return finalize_source(pd.DataFrame(rows), domains)


def deduplicate_observations(
    fit_two: pd.DataFrame,
    fit_one: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_hashes = set(fit_two["policy_hash"]) | set(fit_one["policy_hash"])
    fit_source_identities = set(fit_two["source_experiment"].astype(str) + ":" + fit_two["run_name"].astype(str)) | set(
        fit_one["source_experiment"].astype(str) + ":" + fit_one["run_name"].astype(str)
    )
    candidates = candidates.drop_duplicates("observation_id", keep="first").copy()
    source_identity = candidates["source_experiment"].astype(str) + ":" + candidates["run_name"].astype(str)
    candidates["is_fit_source_alias"] = source_identity.isin(fit_source_identities)
    candidates["split"] = np.where(candidates["policy_hash"].isin(fit_hashes), "repeat", "heldout")
    candidates["is_coordinate_alias"] = False
    candidates["canonical_observation_id"] = candidates["observation_id"]

    for policy, indices in candidates.groupby("policy_hash", sort=False).groups.items():
        ordered = list(indices)
        canonical = ordered[0]
        if policy not in fit_hashes:
            candidates.loc[ordered[1:], "split"] = "repeat"
        candidates.loc[ordered[1:], "is_coordinate_alias"] = True
        candidates.loc[ordered, "canonical_observation_id"] = candidates.loc[canonical, "observation_id"]

    heldout = candidates[(candidates["split"] == "heldout") & ~candidates["is_coordinate_alias"]].copy()
    repeats = candidates[candidates["split"] == "repeat"].copy()
    if set(heldout["policy_hash"]) & fit_hashes:
        raise ValueError("Heldout coordinates overlap a fit policy")
    if heldout["policy_hash"].duplicated().any():
        raise ValueError("Heldout registry contains duplicate coordinates")
    return candidates, heldout, repeats


def compute_comparison() -> pd.DataFrame:
    summary = json.loads(SCALE_SUMMARY_PATH.read_text())
    scale_metadata = {row["scale"]: row for row in summary["scale_metadata"]}
    rows = []
    for scale, tokens in (("60m_1p2b", 1_199_833_088), ("300m_6b", 5_999_951_872)):
        metadata = scale_metadata[scale]
        for convention, parameter_key in (
            ("non_embedding", "non_embedding_params"),
            ("tied_total", "tied_total_params"),
        ):
            parameters = int(metadata[parameter_key])
            rows.append(
                {
                    "setting": scale,
                    "parameter_convention": convention,
                    "parameters": parameters,
                    "training_tokens": tokens,
                    "tokens_per_parameter": tokens / parameters,
                    "six_n_d_flops": 6 * parameters * tokens,
                }
            )
    delphi_parameters = 358_306_688
    delphi_tokens = 1_576_534_016
    rows.append(
        {
            "setting": "delphi_3e18",
            "parameter_convention": "reported_total",
            "parameters": delphi_parameters,
            "training_tokens": delphi_tokens,
            "tokens_per_parameter": delphi_tokens / delphi_parameters,
            "six_n_d_flops": 6 * delphi_parameters * delphi_tokens,
        }
    )
    return pd.DataFrame(rows)


def write_report(
    fit_two: pd.DataFrame,
    fit_one: pd.DataFrame,
    heldout: pd.DataFrame,
    repeats: pd.DataFrame,
    all_candidates: pd.DataFrame,
    compute: pd.DataFrame,
) -> None:
    heldout_counts = Counter(heldout["source_kind"])
    independent_repeats = repeats[~repeats["is_fit_source_alias"]]
    metric_rows = []
    for label, frame in (
        ("fit_two_phase", fit_two),
        ("fit_single_phase", fit_one),
        ("heldout", heldout),
        ("repeat_or_alias", repeats),
    ):
        metric_rows.append(
            {
                "split": label,
                "observations": len(frame),
                "unique_coordinates": frame["policy_hash"].nunique(),
                "uncheatable_complete": int(frame["uncheatable_bpb"].notna().sum()),
                "table9_complete": int(frame["table9_macro_bpb"].notna().sum()),
            }
        )
    coverage = pd.DataFrame(metric_rows)
    write_frame_if_changed(OUTPUT_DIR / "metric_coverage.csv", coverage)
    source_family_counts = pd.DataFrame(
        [{"source_kind": source, "heldout_coordinates": count} for source, count in sorted(heldout_counts.items())]
    )
    write_frame_if_changed(OUTPUT_DIR / "source_family_counts.csv", source_family_counts)

    paired = fit_two.merge(
        fit_one,
        left_on="run_name",
        right_on="paired_run_name",
        suffixes=("_two_phase", "_single_phase"),
        validate="one_to_one",
    )
    comparison_rows = []
    for metric in ("uncheatable_bpb", "table9_macro_bpb"):
        two_values = paired[f"{metric}_two_phase"]
        one_values = paired[f"{metric}_single_phase"]
        delta = two_values - one_values
        interval = stats.t.interval(
            0.95,
            len(delta) - 1,
            loc=float(delta.mean()),
            scale=float(stats.sem(delta)),
        )
        comparison_rows.append(
            {
                "metric": metric,
                "paired_rows": len(delta),
                "single_phase_mean": one_values.mean(),
                "two_phase_mean": two_values.mean(),
                "mean_two_minus_one": delta.mean(),
                "mean_delta_ci95_lower": interval[0],
                "mean_delta_ci95_upper": interval[1],
                "fraction_two_phase_better": (delta < 0).mean(),
                "single_phase_best": one_values.min(),
                "two_phase_best": two_values.min(),
                "best_two_minus_one": two_values.min() - one_values.min(),
                "paired_pearson": one_values.corr(two_values),
                "paired_spearman": one_values.corr(two_values, method="spearman"),
            }
        )
    paired_comparison = pd.DataFrame(comparison_rows)
    write_frame_if_changed(OUTPUT_DIR / "paired_policy_comparison.csv", paired_comparison)

    complete_table9_heldout = heldout[heldout["table9_macro_bpb"].notna()]
    best_table9_heldout = complete_table9_heldout.loc[complete_table9_heldout["table9_macro_bpb"].idxmin()]
    missing_table9_heldout = heldout[heldout["table9_macro_bpb"].isna()]
    missing_table9_with_root = int(missing_table9_heldout["checkpoint_root"].notna().sum())
    tpp = compute.pivot(index="setting", columns="parameter_convention", values="tokens_per_parameter")
    report = f"""# 60M 39-bucket checkpoint audit

## Canonical split

- Two-phase fit observations: **{len(fit_two)}** ({fit_two["policy_hash"].nunique()} unique coordinates).
- Single-phase fit observations: **{len(fit_one)}** ({fit_one["policy_hash"].nunique()} unique coordinates).
- Fit-coordinate union: **{len(set(fit_two["policy_hash"]) | set(fit_one["policy_hash"]))}** coordinates.
- Coordinate-disjoint heldouts: **{len(heldout)}**.
- Repeat or coordinate-alias observations: **{len(repeats)}**.
- Independently trained repeats after excluding fit-source mirrors: **{len(independent_repeats)}**.
- Audited non-fit candidate observations before coordinate classification: **{len(all_candidates)}**.

The two fit policy classes remain separate even at their three shared tied coordinates. No heldout
coordinate overlaps either fit panel. Repeats and source aliases remain visible but are not counted as
heldout evidence.

## Metric coverage

{coverage.to_markdown(index=False)}

Both 242-row fit panels now have complete native Table-9 coverage. Of the {len(heldout)}
coordinate-disjoint heldouts, {len(complete_table9_heldout)} have Table-9. The remaining
{len(missing_table9_heldout)} are historical two-phase archive entries; {missing_table9_with_root}
have checkpoint-root provenance recovered from the metric registry and await native Table-9 evaluation.
The strongest complete heldout is `{best_table9_heldout["run_name"]}` at
{best_table9_heldout["table9_macro_bpb"]:.6f} Table-9 macro BPB and
{best_table9_heldout["uncheatable_bpb"]:.6f} Uncheatable BPB.

## Matched one-phase versus two-phase policies

{paired_comparison.to_markdown(index=False, floatfmt=".6f")}

These are paired descriptive comparisons over the swarm design, not estimates of the policy-class optima.
Lower BPB is better, so a positive `mean_two_minus_one` favors the tied single-phase counterpart.

## Heldout provenance

{source_family_counts.to_markdown(index=False)}

## Compute and tokens per parameter

{compute.to_markdown(index=False, floatfmt=".4g")}

The historical `60m_1p2b` and `300m_6b` labels overstate non-embedding parameter counts. Using
non-embedding parameters, the settings are approximately 59M/1.20B (TPP {tpp.loc["60m_1p2b", "non_embedding"]:.2f})
and 103M/6.00B (TPP {tpp.loc["300m_6b", "non_embedding"]:.2f}). Delphi is 358M/1.58B
(TPP {tpp.loc["delphi_3e18", "reported_total"]:.2f}). The `6ND` FLOP values are comparable only within a
consistent parameter-count convention.
"""
    write_text_if_changed(OUTPUT_DIR / "report.md", report)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = pd.read_csv(FIT_TWO_PATH, nrows=1)
    domains = canonical_domains(seed)
    table9_by_run = table9_lookup()
    checkpoint_by_wandb = checkpoint_root_lookup()

    fit_two = load_fit_two(domains, table9_by_run)
    fit_one = load_fit_one(domains, table9_by_run)
    if len(fit_two) != 242 or len(fit_one) != 242:
        raise ValueError(f"Expected 242 fit observations per policy class, found {len(fit_two)} and {len(fit_one)}")

    sources = [
        load_two_phase_archive(domains, table9_by_run, checkpoint_by_wandb, fit_two),
        load_proportional_noise_repeats(domains, table9_by_run),
        load_logical_extras(domains, table9_by_run),
        load_proportional_controllability(domains, table9_by_run),
        load_noise_repeats(domains, table9_by_run),
    ]
    candidates = pd.concat(sources, ignore_index=True)

    # Prefer the broad historical archive over its logical-registry mirror.
    candidates["_source_priority"] = candidates["source_kind"].map(
        {
            "historical_two_phase": 0,
            "proportional_controllability": 1,
            "proportional_noise": 2,
            "logical_registry": 3,
            "repeat_noise": 4,
        }
    )
    candidates = candidates.sort_values(["_source_priority", "observation_id"]).drop(columns="_source_priority")
    source_identity = candidates["source_experiment"].astype(str) + ":" + candidates["run_name"].astype(str)
    candidates = candidates.loc[~source_identity.duplicated()].copy()

    all_candidates, heldout, repeats = deduplicate_observations(fit_two, fit_one, candidates)
    fit_two["split"] = "fit"
    fit_one["split"] = "fit"
    fit_two["fit_policy"] = "two_phase"
    fit_one["fit_policy"] = "single_phase"

    write_frame_if_changed(OUTPUT_DIR / "fit_two_phase.csv", fit_two)
    write_frame_if_changed(OUTPUT_DIR / "fit_single_phase.csv", fit_one)
    write_frame_if_changed(OUTPUT_DIR / "heldout_observations.csv", heldout)
    write_frame_if_changed(OUTPUT_DIR / "repeat_observations.csv", repeats)
    write_frame_if_changed(OUTPUT_DIR / "all_nonfit_observations.csv", all_candidates)
    compute = compute_comparison()
    write_frame_if_changed(OUTPUT_DIR / "compute_comparison.csv", compute)
    write_report(fit_two, fit_one, heldout, repeats, all_candidates, compute)

    summary = {
        "fit_two_observations": len(fit_two),
        "fit_two_coordinates": fit_two["policy_hash"].nunique(),
        "fit_one_observations": len(fit_one),
        "fit_one_coordinates": fit_one["policy_hash"].nunique(),
        "fit_coordinate_union": len(set(fit_two["policy_hash"]) | set(fit_one["policy_hash"])),
        "fit_coordinate_overlap": len(set(fit_two["policy_hash"]) & set(fit_one["policy_hash"])),
        "heldout_coordinates": len(heldout),
        "repeat_or_alias_observations": len(repeats),
        "independent_repeat_observations": int((~repeats["is_fit_source_alias"]).sum()),
        "heldout_checkpoint_roots": int(heldout["checkpoint_root"].notna().sum()),
        "repeat_checkpoint_roots": int(repeats["checkpoint_root"].notna().sum()),
        "heldout_source_counts": dict(Counter(heldout["source_kind"])),
    }
    write_text_if_changed(OUTPUT_DIR / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
