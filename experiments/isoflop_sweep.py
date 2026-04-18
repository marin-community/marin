# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Registry of ISOFlop sweep runs.

This module defines the sweep configurations for different datasets and budgets.
Heuristic implementations live in experiments/scaling_law_sweeps/.
"""

import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

import fsspec

from experiments.common_pile.tokenize_common_pile import comma_main_mixture
from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.simple import downloads
from experiments.scaling_law_sweeps import c_adamc as c_adamc_heuristic
from experiments.scaling_law_sweeps import completed_adamh as completed_adamh_heuristic
from experiments.tootsie.exp1295_32b import nemotron_mix
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config
from marin.scaling_laws import FitScalingLawsResult, IsoFlopRecord, fit_scaling_laws, round_flops_to_bucket
from marin.scaling_laws.eval_metrics_reader import read_eval_records
from marin.utilities.wandb_utils import WANDB_ENTITY, WANDB_PROJECT

logger = logging.getLogger(__name__)

# ---------------- Levanter WandB Metric Keys ----------------
THROUGHPUT_TOKENS_KEY = "throughput/total_tokens"
THROUGHPUT_GFLOPS_KEY = "throughput/total_gflops"
PARAMETER_COUNT_KEY = "parameter_count"


def parse_isoflop_run_name(run_name: str) -> str | None:
    """Parse experiment name from isoflop run name."""
    run_name = re.sub(r"-[0-9a-fA-F]{6}$", "", run_name)
    for pattern in [
        r"isoflop-(?:[0-9.e+]+)-N(?:[0-9.e+]+)-B(?:\d+)-(.+)",
        r"isoflop-(?:[0-9.e+]+)-d(?:\d+)-L(?:\d+)-B(?:\d+)-(.+)",
    ]:
        match = re.match(pattern, run_name)
        if match:
            return match.group(1)
    return None


def transform_levanter_metrics(
    raw_records: list[dict],
    metric_key: str,
    label_map: dict[str, str] | None = None,
    min_flops: float = 1e18,
) -> list[IsoFlopRecord]:
    """Transform raw Levanter metrics into IsoFlopRecord list."""
    records = []
    for raw in raw_records:
        run_path = raw.get("run_path", "")
        run_name = os.path.basename(run_path.rstrip("/"))
        summary = raw.get("summary", {}) or {}

        tokens = summary.get(THROUGHPUT_TOKENS_KEY)
        total_gflops = summary.get(THROUGHPUT_GFLOPS_KEY)
        metric = summary.get(metric_key)
        params = summary.get(PARAMETER_COUNT_KEY)

        if any(v is None for v in [tokens, total_gflops, metric, params]):
            continue

        flops = round_flops_to_bucket(total_gflops * 1e9)
        if flops < min_flops:
            continue

        exp_name = parse_isoflop_run_name(run_name) or run_name
        label = (label_map or {}).get(exp_name, exp_name)

        records.append(
            IsoFlopRecord(
                tokens=float(tokens), metric=float(metric), flops=float(flops), params=float(params), label=label
            )
        )

    logger.info(f"Transformed {len(records)} records from {len(raw_records)} raw records")
    return records


def load_isoflop_records(config: "IsoFlopAnalysisConfig") -> list[IsoFlopRecord]:
    raw_records = read_eval_records(
        training_runs=config.training_runs,
        metrics_filename=config.metrics_filename,
        wandb_entity_project=config.wandb_entity_project,
    )
    if not raw_records:
        logger.warning("No eval metrics found in training runs")
        return []

    label_map = dict(config.label_map) if config.label_map else None
    records = transform_levanter_metrics(raw_records, config.metric_key, label_map)
    if not records:
        logger.warning("No valid isoflop data after transformation")
        return []

    logger.info(f"Loaded {len(records)} runs for scaling law analysis")
    return records


def save_isoflop_analysis_result(result: FitScalingLawsResult, output_path: str) -> None:
    fs, _, _ = fsspec.get_fs_token_paths(output_path)
    fs.makedirs(output_path, exist_ok=True)

    result_path = os.path.join(output_path, "isoflop_analysis_result.json")
    result_dict = {
        "minima_records": [
            {
                "label": r.label,
                "flops": r.flops,
                "optimal_tokens": r.optimal_tokens,
                "loss_at_optimal": r.loss_at_optimal,
                "optimal_params": r.optimal_params,
                "scaling_alpha": r.scaling_alpha,
                "scaling_A": r.scaling_A,
            }
            for r in result.minima_records
        ],
        "scaling_fits": {k: list(v) for k, v in result.scaling_fits.items()},
    }
    with fs.open(result_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    logger.info(f"Saved results to {result_path}")

    fit_curves_path = os.path.join(output_path, "fit_curves.json")
    fit_curves_json = {f"{label}|{flops}": list(coeffs) for (label, flops), coeffs in result.fit_curves.items()}
    with fs.open(fit_curves_path, "w") as f:
        json.dump(fit_curves_json, f, indent=2)
    logger.info(f"Saved fit curves to {fit_curves_path}")


# --- Budget configurations ---
LEGACY_BUDGETS: tuple[float, ...] = (3e18, 9e18, 1.8e19, 3e19, 9e19, 1.8e20, 3e20)

DEFAULT_METRIC_KEY = "eval/paloma/c4_en/bpb"


# ---------------- IsoFlop Analysis ----------------


@dataclass(frozen=True, kw_only=True)
class IsoFlopAnalysisConfig:
    """Configuration for IsoFLOP scaling law analysis."""

    training_runs: Sequence[str]
    output_path: str
    metric_key: str = DEFAULT_METRIC_KEY
    label_map: tuple[tuple[str, str], ...] | None = None
    metrics_filename: str = "tracker_metrics.jsonl"
    wandb_entity_project: str = f"{WANDB_ENTITY}/{WANDB_PROJECT}"


def run_isoflop_analysis_step(config: IsoFlopAnalysisConfig) -> FitScalingLawsResult:
    """Execute IsoFLOP scaling law analysis."""
    records = load_isoflop_records(config)
    if not records:
        return FitScalingLawsResult(minima_records=[], scaling_fits={}, fit_curves={})

    result = fit_scaling_laws(records)

    logger.info(f"Found {len(result.minima_records)} optimal configurations")
    for label, scaling_fit in result.scaling_fits.items():
        logger.info(f"  {label}: D* = {scaling_fit.A:.2e} * C^{scaling_fit.alpha:.3f}")

    save_isoflop_analysis_result(result, config.output_path)
    return result


# --- Tokenized Datasets ---
dclm_tokenized = default_tokenize(
    name="dclm_baseline",
    dataset=downloads["dclm_baseline"],
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/dclm_baseline-0206f1/")

dclm_mix = lm_mixture_data_config(
    components={"dclm": dclm_tokenized},
    weights={"dclm": 1.0},
    num_validation_sequences={"dclm": 1024},
)

dolma3_mix_tokenized = default_tokenize(
    name="dolma3_mix-150B-1025",
    dataset=downloads["dolma3_mix_150b_1025"],
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/dolma3_mix-150B-1025-15d04ee/")

dolma3_mix = lm_mixture_data_config(
    components={"dolma3_mix-150B-1025": dolma3_mix_tokenized},
    weights={"dolma3_mix-150B-1025": 1.0},
    num_validation_sequences={"dolma3_mix-150B-1025": 1024},
)

# --- Original C-AdamC sweeps (from main) ---
MARIN_SCALING_SUITES = {
    "nemotron": c_adamc_heuristic.create_isoflop_sweep_steps(
        tokenized=nemotron_mix,
        experiment_name="nemo-wider-depth-adapt",
        budgets=LEGACY_BUDGETS,
    ),
    "common_pile_feistel": c_adamc_heuristic.create_isoflop_sweep_steps(
        tokenized=comma_main_mixture(),
        experiment_name="comma-mix-feistel",
        budgets=LEGACY_BUDGETS,
    ),
    "dclm-default": c_adamc_heuristic.create_isoflop_sweep_steps(
        tokenized=dclm_mix,
        experiment_name="dclm-default",
        budgets=LEGACY_BUDGETS,
    ),
    "dolma3_mix_150b": c_adamc_heuristic.create_isoflop_sweep_steps(
        tokenized=dolma3_mix,
        experiment_name="dolma3-mix-150b-1025",
        budgets=LEGACY_BUDGETS,
    ),
    # --- Completed AdamH sweeps ---
    "nemotron-completed-adamh": completed_adamh_heuristic.create_isoflop_sweep_steps(
        tokenized=nemotron_mix,
        experiment_name="adamh_scaling_v6",
        budgets=LEGACY_BUDGETS,
    ),
}

if __name__ == "__main__":
    steps, _ = MARIN_SCALING_SUITES["nemotron-completed-adamh"]
    executor_main(steps=steps)
