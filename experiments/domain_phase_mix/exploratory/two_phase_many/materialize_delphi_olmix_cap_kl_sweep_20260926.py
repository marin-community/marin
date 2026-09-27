# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy>=1.5", "numpy>=2.0", "pandas>=2.2", "scipy>=1.14"]
# ///
"""Materialize the matched-Olmix epoch-cap x KL sweep at Qwen3 360M/1.6B (3e18 FLOPs).

The matched Olmix policy is the per-task log-linear law fitted on the frozen 280-run Qwen swarm
(`materialize_delphi_matched_olmix_20260908.py`, whose saved laws this script reads) optimized by the exact
KL-regularized proposer. This sweep crosses the epoch caps {1, 4, 8, 12, uncapped} with the KL axis of the
paper's MARINER KL table (lambda in {0, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5}) for both objectives:
90 cells. Cells whose continuous solutions lie within total variation 0.01 of an earlier cell (the cap stops binding
once lambda is large) are measured by that earlier cell's run, and cap 1 is trained only at lambda 0 because every
cap-1 mixture is close to proportional. That leaves 39 runs (18 Uncheatable, 21 OlmoBaseEval Easy).

Outputs (``--output-dir``): ``candidate_weights.csv`` in the launcher's format (one row per run and bucket, on the
1/2048 runtime grid under the cell's cap; uncapped cells carry the nominal cap ceil(max epochs)), ``grid_map.csv``
(all 90 cells, with the run that measures each one and the distance to it) and ``summary.json``.

usage: uv run --with cvxpy --with clarabel python materialize_delphi_olmix_cap_kl_sweep_20260926.py
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base  # noqa: E402
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import TOP_LEVEL_DOMAIN_TOKEN_COUNTS  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_delphi_matched_olmix_20260908 as matched,
)

LAWS = SCRIPT_DIR / "reference_outputs" / "delphi_matched_olmix_3e18_20260908"
OUTPUT = SCRIPT_DIR / "reference_outputs" / "delphi_olmix_cap_kl_sweep_3e18_20260926"
CAPS = (1, 4, 8, 12, None)
LAMBDAS = (0.0, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5)
TARGETS = {"uncheatable": ("u", "Uncheatable"), "table9": ("t9", "Table 9")}
DUPLICATE_TV = 0.01
BLOCK = matched.MIXTURE_BLOCK_SIZE
BUDGET = matched.TARGET_BUDGET


def slug(value: float) -> str:
    return "0" if value == 0 else f"{value:g}".replace(".", "p")


def quantize(weights: np.ndarray, tokens: np.ndarray, cap: int | None) -> np.ndarray:
    """Round to the runtime grid, keeping every bucket within the cap (or within its pool when uncapped)."""
    limit = np.full_like(tokens, BLOCK) if cap is None else np.floor(cap * tokens / BUDGET * BLOCK + 1e-9)
    maximum = np.minimum(np.asarray(limit, float), BLOCK).astype(np.int64)
    counts = np.minimum(np.floor(weights * BLOCK + 1e-12).astype(np.int64), maximum)
    remaining = BLOCK - int(counts.sum())
    for bucket in np.argsort(-weights):
        if remaining == 0:
            break
        added = min(int(maximum[bucket] - counts[bucket]), remaining)
        if added > 0:
            counts[bucket] += added
            remaining -= added
    if remaining != 0 or np.any(counts > maximum):
        raise RuntimeError("runtime quantization failed")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = matched.benchmark.read_npz(matched.FROZEN_PANEL)
    buckets = [str(b) for b in panel["buckets"]]
    if set(buckets) != set(base.DOMAIN_NAMES):
        raise ValueError("Frozen panel buckets differ from the launcher's domains")
    tokens = np.asarray([TOP_LEVEL_DOMAIN_TOKEN_COUNTS[b] for b in buckets], float)
    natural = tokens / tokens.sum()

    grid, runs = [], []
    for target, (short, label) in TARGETS.items():
        laws = json.loads((LAWS / f"laws_{target}.json").read_text())
        fits = [SimpleNamespace(log_c=law["log_c"], coefficients=law["coefficients"]) for law in laws]
        objective_weights = np.asarray(panel[f"{target}_aggregation_weights"], float)

        def predict(w: np.ndarray, fits=fits, objective_weights=objective_weights) -> float:
            return float(
                sum(
                    o * (math.exp(f.log_c) + math.exp(float(np.dot(f.coefficients, w))))
                    for o, f in zip(objective_weights, fits, strict=True)
                )
            )

        measured: list[dict] = []
        for kl in LAMBDAS:
            for cap in CAPS:
                caps = np.ones_like(tokens) if cap is None else np.minimum(1.0, tokens * cap / BUDGET)
                weights, status = matched.solve_exact(fits, objective_weights, natural, caps, kl)
                cell = {
                    "target": target,
                    "cap": "none" if cap is None else cap,
                    "kl": kl,
                    "solver": status,
                    "olmix_predicted": predict(weights),
                    "tv_to_proportional": 0.5 * float(np.abs(weights - natural).sum()),
                }
                twin = next(
                    (m for m in measured if 0.5 * float(np.abs(m["weights"] - weights).sum()) < DUPLICATE_TV), None
                )
                if twin is None and cap == 1 and kl > 0:
                    twin = next(m for m in measured if m["cap"] == 1 and m["kl"] == 0)
                if twin is not None:
                    cell |= {
                        "run": twin["candidate_id"],
                        "run_directly": False,
                        "tv_to_run": 0.5 * float(np.abs(twin["weights"] - weights).sum()),
                    }
                    grid.append(cell)
                    continue
                counts = quantize(weights, tokens, cap)
                runtime = counts / BLOCK
                epochs = BUDGET * runtime / tokens
                nominal = cap if cap is not None else math.ceil(float(epochs.max()) - 1e-9)
                name = f"olmixq_{short}_kl{slug(kl)}_{'' if cap is not None else 'nocap_'}cap{nominal:02d}"
                measured.append({"candidate_id": name, "cap": cap, "kl": kl, "weights": weights})
                cell |= {
                    "run": name,
                    "run_directly": True,
                    "tv_to_run": 0.0,
                    "max_epochs": float(epochs.max()),
                    "runtime_tv_to_continuous": 0.5 * float(np.abs(runtime - weights).sum()),
                }
                grid.append(cell)
                surrogate = f"olmix_loglinear_d001_qwen3e18_kl{slug(kl)}_{'nocap' if cap is None else f'cap{cap}'}"
                for index in [buckets.index(d) for d in base.DOMAIN_NAMES]:
                    runs.append(
                        {
                            "candidate_id": name,
                            "target": target,
                            "target_label": label,
                            "epoch_cap": nominal,
                            "surrogate": surrogate,
                            "domain": buckets[index],
                            "runtime_count": int(counts[index]),
                            "weight": counts[index] / BLOCK,
                            "materialized_epochs": float(epochs[index]),
                        }
                    )
    path = args.output_dir / "candidate_weights.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    with (args.output_dir / "grid_map.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target",
                "cap",
                "kl",
                "run",
                "run_directly",
                "tv_to_run",
                "olmix_predicted",
                "tv_to_proportional",
                "max_epochs",
                "runtime_tv_to_continuous",
                "solver",
            ],
        )
        writer.writeheader()
        writer.writerows(grid)
    ids = list(dict.fromkeys(r["candidate_id"] for r in runs))
    summary = {
        "runs": len(ids),
        "by_target": {t: sum(1 for i in ids if i.startswith(f"olmixq_{s}_")) for t, (s, _) in TARGETS.items()},
        "cells": len(grid),
        "candidate_ids": ids,
        "candidate_weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("runs", "by_target", "cells", "candidate_weights_sha256")}))


if __name__ == "__main__":
    main()
