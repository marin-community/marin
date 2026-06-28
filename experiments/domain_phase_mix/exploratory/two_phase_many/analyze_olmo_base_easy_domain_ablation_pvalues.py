# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""OLMoBaseEval Easy BPB p-values for 300M proportional domain deletions.

This analysis compares each 300M leave-one-domain-out checkpoint against the
pooled proportional reference: the original 300M proportional checkpoint plus
the 10 proportional-repeat checkpoints. It uses the same predictive t-statistic
as the generic domain-ablation p-value diagnostics, but reads OLMoBaseEval SC
outputs directly so the CSQA BPB overlay fixes are included.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_716_MANIFEST = (
    SCRIPT_DIR / "reference_outputs" / "olmo_base_eval_sc_716_20260620" / "olmo_base_eval_sc_716_manifest.csv"
)
DEFAULT_PROP_NOISE_MANIFEST = (
    SCRIPT_DIR
    / "reference_outputs"
    / "olmo_base_eval_proportional_noise_20260623"
    / "olmo_base_eval_proportional_noise_manifest.csv"
)
DEFAULT_METRICS_ROOT = SCRIPT_DIR / "reference_outputs" / "olmo_base_eval_sc_300m_metrics_20260623"
DEFAULT_DOMAIN_METADATA = (
    SCRIPT_DIR
    / "reference_outputs"
    / "ppert_bump_vs_log_tilt_comparison_20260614"
    / "domain_ablation_vs_local_gradient_domain_comparison.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "olmo_base_easy_domain_ablation_pvalues_20260625"

SCALE = "300m_6b"
BASELINE_RUN_NAME = "baseline_proportional"
PCTRL_PANEL = "proportional_controllability"
PROP_NOISE_PANEL = "proportional_noise"
KEY_PREFIX = "olmo_base_eval/easy_bpb"
TRUE_BPB_METRIC_PATHS = {"bits_per_byte:bits_per_byte", "primary_score:average"}
OVERLAY_SUFFIX = "_csqa_bpb_fix"


@dataclass(frozen=True)
class MetricValue:
    benchmark_key: str
    olmo_task: str
    metric_path: str
    value_bpb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_716_MANIFEST)
    parser.add_argument("--proportional-noise-manifest", type=Path, default=DEFAULT_PROP_NOISE_MANIFEST)
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--domain-metadata", type=Path, default=DEFAULT_DOMAIN_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def canonical_olmo_task(task_name: str) -> str:
    """Normalize task-name variant ordering while preserving substantive suffixes."""
    parts = [part for part in task_name.split(":") if part not in {"bpb", "olmo3base"}]
    return ":".join(parts)


def benchmark_key(task_name: str) -> str:
    safe = canonical_olmo_task(task_name).replace(":", "_").replace("/", "_")
    return f"{KEY_PREFIX}/{safe}/bpb"


def metric_family(benchmark: str) -> str:
    leaf = benchmark.removeprefix(f"{KEY_PREFIX}/").removesuffix("/bpb")
    if leaf.startswith("mmlu"):
        return "olmo_mmlu"
    if leaf.startswith("minerva_math"):
        return "olmo_math"
    if leaf.startswith("mbpp") or leaf.startswith("mt_mbpp") or leaf.startswith("humaneval"):
        return "olmo_code"
    return "olmo_qa"


def metric_values_from_json(path: Path) -> dict[str, MetricValue]:
    data = json.loads(path.read_text())
    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Missing summary object in {path}")

    values: dict[str, MetricValue] = {}
    for task_name, row in sorted(summary.items()):
        if not isinstance(row, dict):
            continue
        metric_path = str(row.get("metric", ""))
        if metric_path not in TRUE_BPB_METRIC_PATHS:
            continue
        score = finite_float(row.get("score"))
        if score is None:
            continue
        if "bpb" not in task_name.split(":"):
            continue
        key = benchmark_key(task_name)
        values[key] = MetricValue(
            benchmark_key=key,
            olmo_task=canonical_olmo_task(task_name),
            metric_path=metric_path,
            value_bpb=score,
        )
    return values


def metric_map_for_output(metrics_root: Path, output_name: str) -> dict[str, MetricValue]:
    output_dir = metrics_root / "outputs" / output_name
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    values = metric_values_from_json(metrics_path)

    overlay_path = metrics_root / "outputs" / f"{output_name}{OVERLAY_SUFFIX}" / "metrics.json"
    if overlay_path.is_file():
        values.update(metric_values_from_json(overlay_path))
    return values


def load_manifests(manifest_path: Path, proportional_noise_manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    prop_noise = pd.read_csv(proportional_noise_manifest_path)
    frame = pd.concat([manifest, prop_noise], ignore_index=True)
    required = {"scale", "panel", "run_name", "output_name", "wandb_run_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return frame


def domain_slug(domain: str) -> str:
    return domain.replace("/", "_")


def load_domain_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["target_domain", "base_mass"]).drop_duplicates()
    frame["domain_slug"] = frame["target_domain"].map(domain_slug)
    if frame["domain_slug"].duplicated().any():
        duplicated = frame.loc[frame["domain_slug"].duplicated(), "domain_slug"].tolist()
        raise ValueError(f"Duplicate domain slugs: {duplicated}")
    return frame


def load_metrics_long(manifest: pd.DataFrame, metrics_root: Path) -> pd.DataFrame:
    wanted = manifest[
        manifest["scale"].eq(SCALE)
        & (
            (manifest["panel"].eq("parity") & manifest["run_name"].eq(BASELINE_RUN_NAME))
            | (manifest["panel"].eq(PCTRL_PANEL) & manifest["run_name"].str.startswith("pctrl_del_", na=False))
            | manifest["panel"].eq(PROP_NOISE_PANEL)
        )
    ].copy()
    if wanted.empty:
        raise ValueError("No 300M baseline/deletion/proportional-noise rows found in manifests")

    rows: list[dict[str, Any]] = []
    missing_outputs: list[str] = []
    missing_csqa: list[str] = []
    for row in wanted.itertuples(index=False):
        try:
            metric_map = metric_map_for_output(metrics_root, str(row.output_name))
        except FileNotFoundError:
            missing_outputs.append(str(row.output_name))
            continue
        if f"{KEY_PREFIX}/csqa/bpb" not in metric_map:
            missing_csqa.append(str(row.output_name))
        for metric in metric_map.values():
            rows.append(
                {
                    "scale": str(row.scale),
                    "panel": str(row.panel),
                    "run_name": str(row.run_name),
                    "output_name": str(row.output_name),
                    "wandb_run_id": str(row.wandb_run_id),
                    "benchmark_key": metric.benchmark_key,
                    "metric": metric.benchmark_key,
                    "olmo_task": metric.olmo_task,
                    "metric_path": metric.metric_path,
                    "value_bpb": metric.value_bpb,
                }
            )

    if missing_outputs:
        sample = ", ".join(missing_outputs[:10])
        raise ValueError(f"Missing metrics.json for {len(missing_outputs)} outputs; examples: {sample}")
    if missing_csqa:
        sample = ", ".join(missing_csqa[:10])
        raise ValueError(f"Missing corrected CSQA BPB for {len(missing_csqa)} outputs; examples: {sample}")
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise ValueError("No OLMo BPB metrics loaded")
    return metrics


def compute_cell_pvalues(metrics: pd.DataFrame, domain_metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = metrics[
        (metrics["panel"].eq("parity") & metrics["run_name"].eq(BASELINE_RUN_NAME))
        | metrics["panel"].eq(PROP_NOISE_PANEL)
    ].copy()
    deletion = metrics[
        metrics["panel"].eq(PCTRL_PANEL) & metrics["run_name"].str.startswith("pctrl_del_", na=False)
    ].copy()
    deletion["domain_slug"] = deletion["run_name"].str.removeprefix("pctrl_del_")
    deletion = deletion.merge(domain_metadata, on="domain_slug", how="left", validate="many_to_one")
    if deletion["target_domain"].isna().any():
        missing = sorted(deletion.loc[deletion["target_domain"].isna(), "domain_slug"].unique())
        raise ValueError(f"Could not map deletion domains: {missing}")

    ref_counts = reference.groupby("benchmark_key")["run_name"].nunique()
    if not ref_counts.eq(11).all():
        bad = ref_counts[~ref_counts.eq(11)].sort_index()
        raise ValueError(f"Expected 11 proportional reference rows per benchmark; bad counts:\n{bad}")
    deletion_counts = deletion.groupby("benchmark_key")["run_name"].nunique()
    if not deletion_counts.eq(39).all():
        bad = deletion_counts[~deletion_counts.eq(39)].sort_index()
        raise ValueError(f"Expected 39 deletion rows per benchmark; bad counts:\n{bad}")

    rows: list[dict[str, Any]] = []
    for benchmark, reference_group in reference.groupby("benchmark_key", sort=True):
        # Utility is higher-is-better. For BPB, utility is negative BPB.
        ref_utility = -reference_group["value_bpb"].to_numpy(dtype=float)
        n_noise = int(len(ref_utility))
        noise_sd = float(np.std(ref_utility, ddof=1))
        if not np.isfinite(noise_sd) or noise_sd <= 0.0:
            continue
        base_utility = float(np.mean(ref_utility))
        df = n_noise - 1
        predictive_sd = noise_sd * math.sqrt(1.0 + 1.0 / n_noise)
        matching_deletions = deletion[deletion["benchmark_key"].eq(benchmark)]
        for deleted in matching_deletions.itertuples(index=False):
            deletion_utility = -float(deleted.value_bpb)
            delta = deletion_utility - base_utility
            t_stat = delta / predictive_sd
            p_harm = float(student_t.cdf(t_stat, df=df))
            p_improve = float(student_t.sf(t_stat, df=df))
            p_two_sided = float(2.0 * min(p_harm, p_improve))
            rows.append(
                {
                    "benchmark_key": benchmark,
                    "metric": benchmark,
                    "metric_family": metric_family(benchmark),
                    "metric_kind": "bpb",
                    "lower_is_better": True,
                    "olmo_task": str(deleted.olmo_task),
                    "target_domain": str(deleted.target_domain),
                    "base_mass": float(deleted.base_mass),
                    "proportional_reference_bpb": float(reference_group["value_bpb"].mean()),
                    "deletion_bpb": float(deleted.value_bpb),
                    "proportional_reference_utility": base_utility,
                    "domain_deletion_utility_delta": delta,
                    "noise_n": n_noise,
                    "noise_sd": noise_sd,
                    "predictive_sd": predictive_sd,
                    "t_statistic": t_stat,
                    "p_harm": min(max(p_harm, 0.0), 1.0),
                    "p_improve": min(max(p_improve, 0.0), 1.0),
                    "p_two_sided": min(max(p_two_sided, 0.0), 1.0),
                    "deletion_run_name": str(deleted.run_name),
                    "deletion_output_name": str(deleted.output_name),
                    "deletion_wandb_run_id": str(deleted.wandb_run_id),
                }
            )
    cell = pd.DataFrame(rows)
    if cell.empty:
        raise ValueError("No p-values computed; all reference variances may be zero")

    benchmark_summary = (
        cell.groupby("benchmark_key", as_index=False)
        .agg(
            metric=("metric", "first"),
            metric_family=("metric_family", "first"),
            metric_kind=("metric_kind", "first"),
            olmo_task=("olmo_task", "first"),
            n_domains=("target_domain", "nunique"),
            noise_n=("noise_n", "first"),
            noise_sd=("noise_sd", "first"),
            min_p_harm_raw=("p_harm", "min"),
            min_p_two_sided_raw=("p_two_sided", "min"),
            mean_deletion_delta=("domain_deletion_utility_delta", "mean"),
            fraction_domains_harm=("domain_deletion_utility_delta", lambda s: float((s < 0.0).mean())),
        )
        .sort_values("min_p_harm_raw")
    )
    benchmark_summary["min_p_harm_bonferroni"] = np.minimum(
        1.0, benchmark_summary["min_p_harm_raw"] * benchmark_summary["n_domains"]
    )
    benchmark_summary["min_p_two_sided_bonferroni"] = np.minimum(
        1.0, benchmark_summary["min_p_two_sided_raw"] * benchmark_summary["n_domains"]
    )
    return cell, benchmark_summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifests(args.manifest, args.proportional_noise_manifest)
    metrics = load_metrics_long(manifest, args.metrics_root)
    domain_metadata = load_domain_metadata(args.domain_metadata)
    cell, benchmark = compute_cell_pvalues(metrics, domain_metadata)

    metrics.to_csv(args.output_dir / "olmo_base_easy_bpb_metrics_long.csv", index=False)
    cell.to_csv(args.output_dir / "domain_ablation_cell_pvalues.csv", index=False)
    benchmark.to_csv(args.output_dir / "domain_ablation_benchmark_min_pvalues.csv", index=False)

    summary = {
        "scale": SCALE,
        "reference_rows_per_metric": 11,
        "reference_definition": "baseline_proportional plus 10 proportional_noise repeats",
        "deleted_domain_count": int(cell["target_domain"].nunique()),
        "benchmark_count": int(cell["benchmark_key"].nunique()),
        "cell_count": int(len(cell)),
        "csqa_key_present": bool((metrics["benchmark_key"] == f"{KEY_PREFIX}/csqa/bpb").any()),
        "metric_families": {
            str(key): int(value)
            for key, value in (
                cell[["benchmark_key", "metric_family"]]
                .drop_duplicates()["metric_family"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "benchmarks_bonferroni_harm_p_lt_0p05": int((benchmark["min_p_harm_bonferroni"] < 0.05).sum()),
        "raw_cells_harm_p_lt_0p05": int((cell["p_harm"] < 0.05).sum()),
        "bonferroni_cells_harm_p_lt_0p05": int(
            (cell.groupby("benchmark_key")["p_harm"].transform(lambda s: np.minimum(1.0, s * len(s))) < 0.05).sum()
        ),
        "outputs": {
            "metrics_long_csv": str(args.output_dir / "olmo_base_easy_bpb_metrics_long.csv"),
            "cell_pvalues_csv": str(args.output_dir / "domain_ablation_cell_pvalues.csv"),
            "benchmark_min_pvalues_csv": str(args.output_dir / "domain_ablation_benchmark_min_pvalues.csv"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
