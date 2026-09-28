# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Can a surrogate find the few policies where two phases help a lot, not just on average?

The headroom bound says the phase-decision axis is nearly exhausted: the incumbent already
captures 74 to 79 percent of the whole always-tied-to-clairvoyant span, so a perfect model
could add only about 1.5 run sigma of realized value on Uncheatable. That bound is on the
*average* over the 238-pair population, and averages are the wrong summary for a fat tail.
Individual pairs run to -0.049 BPB on Uncheatable and -0.114 on Table-9, fifty and thirty-five
times the average gain a better model buys.

So the question the bound does not answer: can a model pick out those few policies? A
proposal campaign trains a handful of candidates, not the whole population, so what matters
operationally is the realized ``Delta`` of the model's top few recommendations, not its
value-weighted accuracy over everything.

Two designs are compared, both at a matched run budget with an identical holdout, exactly as
in the pairs-versus-all-two-phase comparison. Three tail measures:

``top_k_realized``   mean observed ``Delta`` over the k pairs the model ranks most negative.
                     This is what a campaign of size k actually gets.
``tail_capture``     that value divided by the mean observed ``Delta`` of the true best k,
                     so 1.0 means the model found the best k available and 0 means its picks
                     were no better than always tying.
``hit_rate``         fraction of the model's top k that fall in the true best decile of
                     ``Delta``, which asks whether the picks are in the right region at all.

A random ranker is included as the floor, because with roughly half of all pairs showing a
negative ``Delta`` a model can look competent on accuracy while its ranking carries no
information about magnitude.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_pairs_versus_all_two_phase_20260726 import (  # noqa: E402
    CENSOR_FRACTION,
    PHASE_HOLDOUT,
    SEED,
    load_pairs,
    panel_of,
    select_shape,
    training_panel,
)
from dual_objective_harness_20260726 import RUN_SIGMA, build_benchmark  # noqa: E402
from swarm39_harness_20260725 import TABLE9, UNCHEATABLE, fit_head  # noqa: E402
from swarm39_models_20260725 import build_hierarchical_phase_replay  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "phase_tail_capture_20260726"

TARGETS = (UNCHEATABLE, TABLE9)
TOP_K = (1, 3, 5, 10)
DRAWS = 10
L2 = 0.0


def tail_measures(predicted: np.ndarray, observed: np.ndarray, k: int) -> dict[str, float]:
    """Realized value and tail capture of the k most-negative predicted deltas."""
    order = np.argsort(predicted)[:k]
    realized = float(np.mean(observed[order]))
    best_possible = float(np.mean(np.sort(observed)[:k]))
    decile = np.quantile(observed, 0.10)
    return {
        f"top{k}_realized": realized,
        f"top{k}_best_possible": best_possible,
        # Guard the degenerate case where the best k are not actually negative.
        f"top{k}_capture": float(realized / best_possible) if best_possible < 0.0 else float("nan"),
        f"top{k}_hit_rate": float(np.mean(observed[order] <= decile)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(build_benchmark().fit_300m)
    total = len(pairs)
    rows: list[dict[str, Any]] = []

    for target in TARGETS:
        n_censored = max(1, int(CENSOR_FRACTION * total))
        censored = np.argsort(pairs.tied[target])[:n_censored]
        remaining = np.setdiff1d(np.arange(total), censored)

        for draw in range(DRAWS):
            rng = np.random.default_rng(SEED + draw)
            shuffled = rng.permutation(remaining)
            holdout = shuffled[:PHASE_HOLDOUT]
            pool = shuffled[PHASE_HOLDOUT:]
            budget = (len(pool) // 2) * 2
            truth = pairs.delta[target][holdout]

            for design in ("all_two_phase", "pairs"):
                panel = training_panel(pairs, target, design, pool, budget)
                shape = select_shape(panel, target)
                matrix = build_hierarchical_phase_replay(panel, shape).matrix
                intercept, coefficients = fit_head(matrix, panel.targets[target], L2)

                def predict(
                    phase0: np.ndarray,
                    phase1: np.ndarray,
                    _target: str = target,
                    _shape: dict = shape,
                    _intercept: float = intercept,
                    _coefficients: np.ndarray = coefficients,
                ) -> np.ndarray:
                    probe = panel_of(pairs, phase0, phase1, _target, np.zeros(len(phase0)))
                    return _intercept + build_hierarchical_phase_replay(probe, _shape).matrix @ _coefficients

                predicted = predict(pairs.phase0[holdout], pairs.phase1[holdout]) - predict(
                    pairs.aggregate[holdout], pairs.aggregate[holdout]
                )
                record: dict[str, Any] = {"target": target, "design": design, "draw": draw}
                for k in TOP_K:
                    record.update(tail_measures(predicted, truth, k))
                rows.append(record)

            # Random ranker floor, same holdout and draws.
            record = {"target": target, "design": "random", "draw": draw}
            for k in TOP_K:
                record.update(tail_measures(rng.standard_normal(len(truth)), truth, k))
            rows.append(record)
            print(f"  {target} draw {draw + 1}/{DRAWS}")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "tail_capture.csv", index=False)

    print("\n=== can the model find the pairs where two phases help a lot? (mean over draws) ===")
    for target in TARGETS:
        block = frame[frame.target == target]
        print(f"\n### {target}  (run sigma {RUN_SIGMA[target]})")
        columns = [f"top{k}_{stat}" for k in TOP_K for stat in ("realized", "capture", "hit_rate")]
        print(block.groupby("design")[columns].mean().to_string(float_format=lambda v: f"{v:+.5f}"))
        best = block.groupby("design")[[f"top{k}_best_possible" for k in TOP_K]].mean().iloc[0]
        print("  best possible: " + "  ".join(f"top{k}={best[f'top{k}_best_possible']:+.5f}" for k in TOP_K))

    print("\n=== paired over draws: pairs minus all_two_phase, realized value of the picks ===")
    summary = []
    for target, block in frame.groupby("target"):
        a = block[block.design == "all_two_phase"].set_index("draw")
        b = block[block.design == "pairs"].set_index("draw")
        for k in TOP_K:
            metric = f"top{k}_realized"
            delta = (b[metric] - a[metric]).to_numpy()
            summary.append(
                {
                    "target": target,
                    "k": k,
                    "mean_delta_bpb": float(delta.mean()),
                    "in_run_sigma": float(delta.mean() / RUN_SIGMA[target]),
                    "ci95_low": float(np.quantile(delta, 0.025)),
                    "ci95_high": float(np.quantile(delta, 0.975)),
                    "fraction_better": float(np.mean(delta < 0)),
                }
            )
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(OUTPUT_DIR / "paired_tail_summary.csv", index=False)
    print(summary_frame.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))

    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps({"top_k": list(TOP_K), "draws": DRAWS, "phase_holdout": PHASE_HOLDOUT}, indent=2)
    )


if __name__ == "__main__":
    main()
