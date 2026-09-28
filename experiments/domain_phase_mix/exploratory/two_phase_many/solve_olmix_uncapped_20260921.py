# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy>=1.5", "numpy>=2.0", "pandas>=2.2"]
# ///
"""Olmix's optimum without an epoch cap or KL penalty on the matched Qwen-fitted laws, for visualization.

Reuses the frozen per-task log-linear laws of ``delphi_matched_olmix_3e18_20260908`` (fitted on the 280-run Qwen3
360M/1.6B swarm) and the same exact proposer, with the per-bucket caps removed and the KL coefficient at zero, then
rounds to the 1/2048 runtime grid with the remainder given to the largest buckets. Nothing is launched; the output
table has the columns of the matched candidate table so figure builders can read it the same way.

usage: PYTHONPATH=. uv run --offline --no-sync python <this file>
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass

import numpy as np

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import TOP_LEVEL_DOMAIN_TOKEN_COUNTS
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_delphi_selection_20260906 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_matched_olmix_20260908 as matched,
)

OUTPUT = matched.SCRIPT_DIR / "reference_outputs" / "olmix_uncapped_20260921"
CASES = (("olmixq_u_kl0_nocap", "uncheatable"), ("olmixq_t9_kl0_nocap", "table9"))


@dataclass(frozen=True)
class SavedLaw:
    """The two fields of a fitted law that the proposer reads, restored from ``laws_<target>.json``."""

    log_c: float
    coefficients: list[float]


def saved_laws(target: str) -> list[SavedLaw]:
    records = json.loads((matched.OUTPUT / f"laws_{target}.json").read_text())
    return [SavedLaw(record["log_c"], list(record["coefficients"])) for record in records]


def quantize(weights: np.ndarray) -> np.ndarray:
    """Round to the runtime grid with no cap: floor, then the remainder to the largest buckets."""
    counts = np.floor(weights * matched.MIXTURE_BLOCK_SIZE + 1e-12).astype(np.int64)
    remaining = matched.MIXTURE_BLOCK_SIZE - int(counts.sum())
    for bucket in np.argsort(-weights)[:remaining]:
        counts[bucket] += 1
    if int(counts.sum()) != matched.MIXTURE_BLOCK_SIZE:
        raise RuntimeError("runtime quantization did not fill the block grid")
    return counts


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = benchmark.read_npz(matched.FROZEN_PANEL)
    buckets = [str(b) for b in panel["buckets"]]
    launcher_order = [buckets.index(domain) for domain in base.DOMAIN_NAMES]
    tokens = np.asarray([TOP_LEVEL_DOMAIN_TOKEN_COUNTS[b] for b in buckets], float)
    natural = tokens / tokens.sum()
    rows = []
    summary = {}
    for candidate_id, target in CASES:
        laws = saved_laws(target)
        objective_weights = np.asarray(panel[f"{target}_aggregation_weights"], float)
        continuous, status = matched.solve_exact(laws, objective_weights, natural, np.ones(len(buckets)), 0.0)
        counts = quantize(continuous)
        runtime = counts / matched.MIXTURE_BLOCK_SIZE
        epochs = matched.TARGET_BUDGET * runtime / tokens
        summary[candidate_id] = {
            "target": target,
            "solver": status,
            "predicted_continuous": matched.predict_objective(laws, objective_weights, continuous),
            "predicted_runtime": matched.predict_objective(laws, objective_weights, runtime),
            "active_buckets_runtime": int((counts > 0).sum()),
            "max_epochs_runtime": float(epochs.max()),
            "largest_weight": float(runtime.max()),
            "tv_to_proportional": float(np.abs(runtime - natural).sum() / 2),
        }
        for index in launcher_order:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "target": target,
                    "target_label": matched.TARGET_LABELS[target],
                    "epoch_cap": "none",
                    "surrogate": matched.SURROGATE,
                    "domain": buckets[index],
                    "runtime_count": int(counts[index]),
                    "weight": float(runtime[index]),
                    "materialized_epochs": float(epochs[index]),
                }
            )
        print(candidate_id, json.dumps(summary[candidate_id], indent=1), flush=True)
    with (OUTPUT / "candidate_weights.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {OUTPUT / 'candidate_weights.csv'}")


if __name__ == "__main__":
    main()
