# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scipy",
# ]
# ///

"""Analyze the frozen StarCoder WSD80 physical-full-pool intervention."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

SCRIPT_DIR = Path(__file__).resolve().parent
DESIGN_PATH = SCRIPT_DIR.parents[1] / "starcoder_wsd80_full_pool_design_20260804.json"
BASE_EFFECTS_PATH = (
    SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_batch_repetition_results_20260805" / "paired_seed_effects.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_full_pool_results_20260807"

CHECKPOINT_PREFIX = (
    "gs://marin-us-central1/checkpoints/pinlin_calvin_xu/data_mixture/" "starcoder_wsd80_full_pool_intervention_20260804"
)
CHECKPOINT_VERSION = "2026.07.11"
FROZEN_METRIC_NAME = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
PERSISTED_METRIC_KEY = FROZEN_METRIC_NAME
TIMING_KEYS = frozenset({"eval/loading_time", "eval/total_time"})
EXPECTED_POLICIES = frozenset({"A_phase", "B_agg018", "C_tied070"})
EXPECTED_SEEDS = frozenset(range(20_260_811, 20_260_817))
EXPECTED_RUN_COUNT = 18
ALPHA = 0.05
EQUIVALENCE_MARGIN = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DESIGN_PATH)
    parser.add_argument("--base-effects", type=Path, default=BASE_EFFECTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--eval-dir",
        type=Path,
        help="Optional directory containing <run_name>.jsonl; otherwise read central1 GCS.",
    )
    parser.add_argument("--skip-hf-check", action="store_true")
    return parser.parse_args()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def load_design(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    design = json.loads(path.read_text(encoding="utf-8"))
    claimed_hash = design.pop("design_sha256", None)
    observed_hash = canonical_sha256(design)
    if claimed_hash != observed_hash:
        raise ValueError(f"Frozen design hash mismatch: {observed_hash} != {claimed_hash}")
    design["design_sha256"] = claimed_hash
    if design.get("expected_run_count") != EXPECTED_RUN_COUNT:
        raise ValueError("Frozen design run count changed")
    analysis = design.get("analysis", {})
    if analysis.get("primary_metric") != FROZEN_METRIC_NAME:
        raise ValueError("Frozen semantic objective changed")
    if float(analysis.get("equivalence_margin_bpb")) != EQUIVALENCE_MARGIN:
        raise ValueError("Frozen equivalence margin changed")

    manifest = pd.DataFrame(design["runs"])
    if len(manifest) != EXPECTED_RUN_COUNT or manifest["run_name"].duplicated().any():
        raise ValueError("Frozen design does not contain 18 unique runs")
    if set(manifest["policy_id"]) != EXPECTED_POLICIES:
        raise ValueError("Frozen policy block is incomplete")
    if set(manifest["pair_seed"]) != EXPECTED_SEEDS:
        raise ValueError("Frozen seed block is incomplete")
    counts = manifest.groupby(["policy_id", "pair_seed"]).size()
    if len(counts) != EXPECTED_RUN_COUNT or not counts.eq(1).all():
        raise ValueError("Frozen policy-by-seed block is not one-to-one")
    return design, manifest


def metric_uri(run_name: str) -> str:
    return f"{CHECKPOINT_PREFIX}/{run_name}/{CHECKPOINT_VERSION}/checkpoints/eval_metrics.jsonl"


def hf_uri(run_name: str, final_step: int) -> str:
    return f"{CHECKPOINT_PREFIX}/{run_name}/{CHECKPOINT_VERSION}/hf/step-{final_step}/model.safetensors"


def gcs_cat(uri: str) -> str:
    result = subprocess.run(
        ["gcloud", "storage", "cat", uri],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def gcs_exists(uri: str) -> bool:
    result = subprocess.run(
        ["gcloud", "storage", "ls", uri],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and uri in result.stdout.splitlines()


def scientific_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in TIMING_KEYS}


def collect_observations(
    manifest: pd.DataFrame,
    eval_dir: Path | None,
    check_hf: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in manifest.to_dict("records"):
        run_name = str(row["run_name"])
        final_step = int(row["total_steps"]) - 1
        uri = metric_uri(run_name)
        payload = (eval_dir / f"{run_name}.jsonl").read_text() if eval_dir is not None else gcs_cat(uri)
        records = [json.loads(line) for line in payload.splitlines() if line.strip()]
        endpoint = [record for record in records if int(record.get("step", -1)) == final_step]
        if not endpoint:
            raise ValueError(f"{run_name}: missing final metric at step {final_step}")
        reference = scientific_record(endpoint[0])
        if any(scientific_record(duplicate) != reference for duplicate in endpoint[1:]):
            raise ValueError(f"{run_name}: duplicate endpoint scientific metrics differ")
        value = endpoint[0].get(PERSISTED_METRIC_KEY)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"{run_name}: missing finite {PERSISTED_METRIC_KEY}")
        model_uri = hf_uri(run_name, final_step)
        hf_present = gcs_exists(model_uri) if check_hf else None
        if check_hf and not hf_present:
            raise ValueError(f"{run_name}: missing final HF export {model_uri}")
        rows.append(
            {
                **row,
                "programming_languages_bpb": float(value),
                "final_metric_step": final_step,
                "metric_uri": uri,
                "metric_record_count": len(records),
                "endpoint_duplicate_count": len(endpoint) - 1,
                "hf_model_uri": model_uri,
                "hf_model_present": hf_present,
                "wandb_url": f"https://wandb.ai/marin-community/marin/runs/{run_name}",
            }
        )
    observations = pd.DataFrame(rows)
    if len(observations) != EXPECTED_RUN_COUNT or observations["programming_languages_bpb"].isna().any():
        raise ValueError("Endpoint collection is incomplete")
    return observations.sort_values(["policy_id", "pair_seed"]).reset_index(drop=True)


def paired_effects(observations: pd.DataFrame, base_effects_path: Path) -> pd.DataFrame:
    effects = observations.pivot(
        index="pair_seed", columns="policy_id", values="programming_languages_bpb"
    ).reset_index()
    effects["delta_order_bpb"] = effects["B_agg018"] - effects["A_phase"]
    effects["delta_aggregate_bpb"] = effects["B_agg018"] - effects["C_tied070"]
    effects["delta_global_bpb"] = effects["C_tied070"] - effects["A_phase"]
    effects["delta_tied_envelope_bpb"] = effects[["B_agg018", "C_tied070"]].min(axis=1) - effects["A_phase"]
    identity_error = (effects["delta_order_bpb"] - effects["delta_aggregate_bpb"] - effects["delta_global_bpb"]).abs()
    if float(identity_error.max()) > 1e-12:
        raise ValueError("Order-gap decomposition identity failed")

    base = pd.read_csv(base_effects_path)
    base = base.loc[base["condition_id"].eq("base"), ["pair_seed", "delta_order_bpb"]]
    if len(base) != len(EXPECTED_SEEDS) or set(base["pair_seed"]) != EXPECTED_SEEDS:
        raise ValueError("Frozen base-condition paired block is incomplete")
    effects = effects.merge(base.rename(columns={"delta_order_bpb": "base_delta_order_bpb"}), on="pair_seed")
    effects["gamma_fullpool_bpb"] = effects["delta_order_bpb"] - effects["base_delta_order_bpb"]
    return effects.sort_values("pair_seed").reset_index(drop=True)


def interval(values: pd.Series, confidence: float) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
    half_width = float(stats.t.ppf((1.0 + confidence) / 2.0, len(array) - 1) * standard_error)
    return mean - half_width, mean + half_width


def summarize(values: pd.Series) -> dict[str, float | int]:
    array = values.to_numpy(dtype=float)
    ci90_low, ci90_high = interval(values, 0.90)
    ci95_low, ci95_high = interval(values, 0.95)
    test = stats.ttest_1samp(array, 0.0)
    return {
        "n": len(array),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)),
        "ci90_low": ci90_low,
        "ci90_high": ci90_high,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "two_sided_p": float(test.pvalue),
        "positive_count": int(np.sum(array > 0.0)),
    }


def tost(values: pd.Series, margin: float) -> dict[str, float | bool | str]:
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
    degrees_freedom = len(array) - 1
    lower_t = (mean + margin) / standard_error
    upper_t = (mean - margin) / standard_error
    lower_p = float(stats.t.sf(lower_t, degrees_freedom))
    upper_p = float(stats.t.cdf(upper_t, degrees_freedom))
    persistence_p = float(stats.t.sf((mean - margin) / standard_error, degrees_freedom))
    ci90_low, ci90_high = interval(values, 0.90)
    equivalent = ci90_low > -margin and ci90_high < margin
    materially_persistent = ci90_low > margin
    decision = "equivalent" if equivalent else "materially_persistent" if materially_persistent else "inconclusive"
    return {
        "margin_bpb": margin,
        "lower_p": lower_p,
        "upper_p": upper_p,
        "tost_p": max(lower_p, upper_p),
        "persistence_p": persistence_p,
        "equivalent": equivalent,
        "materially_persistent": materially_persistent,
        "decision": decision,
    }


def exact_two_sided_mde(sample_sd: float, n: int, alpha: float, target_power: float = 0.80) -> float:
    degrees_freedom = n - 1
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, degrees_freedom))

    def power(effect: float) -> float:
        noncentrality = effect * math.sqrt(n) / sample_sd
        return float(
            stats.nct.cdf(-critical, degrees_freedom, noncentrality)
            + stats.nct.sf(critical, degrees_freedom, noncentrality)
        )

    upper = sample_sd
    while power(upper) < target_power:
        upper *= 2.0
        if upper > 20.0 * sample_sd:
            raise ValueError("Could not bracket the paired-test MDE")
    return float(optimize.brentq(lambda effect: power(effect) - target_power, 0.0, upper))


def markdown_contrast_row(label: str, result: dict[str, Any]) -> str:
    interval_text = f"[{result['ci95_low']:+.6f}, {result['ci95_high']:+.6f}]"
    seed_text = f"{result['positive_count']}/{result['n']}"
    return f"| {label} | {result['mean']:+.6f} | {interval_text} | {seed_text} | {result['two_sided_p']:.3g} |"


def write_report(
    design: dict[str, Any],
    observations: pd.DataFrame,
    results: dict[str, Any],
    output_dir: Path,
) -> None:
    policy_means = observations.groupby("policy_id")["programming_languages_bpb"].mean()
    order = results["delta_order"]
    aggregate = results["delta_aggregate"]
    global_effect = results["delta_global"]
    gamma = results["gamma_fullpool"]
    tost_result = results["tost"]
    lines = [
        "# StarCoder WSD80 physical-full-pool intervention results",
        "",
        "All 18 frozen runs have final Programming Languages BPB metrics and final HF exports in "
        "`gs://marin-us-central1`.",
        "Five Iris children exited during process teardown after final evaluation and export. "
        "No scientific row is missing, so no retry is warranted.",
        "",
        "## Frozen primary result",
        "",
        f"The paired full-pool order gap `B - A` is **{order['mean']:+.6f} BPB** "
        f"(90% CI [{order['ci90_low']:+.6f}, {order['ci90_high']:+.6f}], n={order['n']}). "
        f"The preregistered decision is **{tost_result['decision']}** relative to the "
        f"+/-{EQUIVALENCE_MARGIN:.3f}-BPB practical-null band.",
        "",
        f"The residual phase-order gain therefore does not collapse when exact cache-index repetition is removed. "
        f"All {order['positive_count']}/{order['n']} paired seeds favor A over its aggregate-matched tied control B.",
        "",
        "## Attribution contrasts",
        "",
        "| contrast | mean BPB | 95% CI | positive seeds | two-sided p |",
        "|---|---:|---:|---:|---:|",
        markdown_contrast_row("`B - A` order gap", order),
        markdown_contrast_row("`B - C` aggregate effect", aggregate),
        markdown_contrast_row("`C - A` global two-phase gain", global_effect),
        markdown_contrast_row("full-pool minus base order gap", gamma),
        "",
        "Policy means: " + ", ".join(f"`{policy}` {value:.6f}" for policy, value in policy_means.items()) + ".",
        "",
        "The full-pool intervention increases the A-vs-B order gap relative to the historical "
        f"repeated-subset base by {gamma['mean']:+.6f} BPB. It therefore falsifies the preregistered "
        "hypothesis that the fixed-policy phase advantage is mostly caused by exact cache-index repetition. "
        "It does not isolate semantic duplication or estimate reoptimized one-phase and two-phase optima.",
        "",
        "## Audit",
        "",
        f"- Frozen design SHA-256: `{design['design_sha256']}`",
        f"- Complete endpoints: `{len(observations)}/{EXPECTED_RUN_COUNT}`",
        f"- Complete HF exports: `{int(observations['hf_model_present'].fillna(False).sum())}/{EXPECTED_RUN_COUNT}`",
        f"- Base paired SD: `{results['base_delta_order_sd']:.6f}` BPB",
        "- Exact two-sided 80%-power MDE at alpha 0.05 from the frozen base SD: "
        f"`{results['base_two_sided_mde_80']:.6f}` BPB",
        f"- Equivalence TOST p-value: `{tost_result['tost_p']:.3g}`",
        f"- One-sided material-persistence p-value: `{tost_result['persistence_p']:.3g}`",
        "",
        "Artifacts: `endpoint_observations.csv`, `paired_seed_effects.csv`, and `analysis_summary.json`.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    design, manifest = load_design(args.design)
    observations = collect_observations(manifest, args.eval_dir, not args.skip_hf_check)
    effects = paired_effects(observations, args.base_effects)

    base_sd = float(effects["base_delta_order_bpb"].std(ddof=1))
    results: dict[str, Any] = {
        "design_sha256": design["design_sha256"],
        "endpoint_count": len(observations),
        "hf_export_count": int(observations["hf_model_present"].fillna(False).sum()),
        "delta_order": summarize(effects["delta_order_bpb"]),
        "delta_aggregate": summarize(effects["delta_aggregate_bpb"]),
        "delta_global": summarize(effects["delta_global_bpb"]),
        "delta_tied_envelope": summarize(effects["delta_tied_envelope_bpb"]),
        "gamma_fullpool": summarize(effects["gamma_fullpool_bpb"]),
        "tost": tost(effects["delta_order_bpb"], EQUIVALENCE_MARGIN),
        "base_delta_order_sd": base_sd,
        "base_two_sided_mde_80": exact_two_sided_mde(base_sd, len(effects), ALPHA),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(args.output_dir / "endpoint_observations.csv", index=False)
    effects.to_csv(args.output_dir / "paired_seed_effects.csv", index=False)
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(design, observations, results, args.output_dir)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
