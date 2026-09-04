# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Which epoch-cap policies keep the measured-best bank coordinates feasible, and what a model picks inside them.

A policy assigns each bucket a maximum number of materialized epochs (uniform, or by bucket type following the
domain-repetition literature: synthetic and math highest, code next, curated text lower, Common Crawl lowest).
For each policy the script lists the feasible Delphi bank coordinates, the best measured value among them, and
the regret of a frozen model's pick restricted to the feasible set. Development evidence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_heldout_selection_20260903 as selection,
)

PANEL = "delphi_3e18_39bucket"
TYPE_ORDER = ("synthetic", "math", "code", "curated", "cc_high", "cc_low")


def bucket_type(name: str) -> str:
    if name.startswith("dolma3_cc/"):
        return "cc_high" if name.endswith("_high") else "cc_low"
    if "synth" in name:
        return "synthetic"
    if "math" in name:
        return "math"
    if "stack" in name or "code" in name:
        return "code"
    return "curated"


POLICIES: dict[str, dict[str, float]] = {
    "uniform_4": dict.fromkeys(TYPE_ORDER, 4.0),
    "uniform_6": dict.fromkeys(TYPE_ORDER, 6.0),
    "uniform_8": dict.fromkeys(TYPE_ORDER, 8.0),
    "uniform_16": dict.fromkeys(TYPE_ORDER, 16.0),
    "typed_tight": {"synthetic": 8.0, "math": 6.0, "code": 5.0, "curated": 4.0, "cc_high": 3.0, "cc_low": 2.0},
    "typed_loose": {"synthetic": 16.0, "math": 12.0, "code": 8.0, "curated": 6.0, "cc_high": 4.0, "cc_low": 3.0},
    "typed_frontier": {"synthetic": 16.0, "math": 16.0, "code": 8.0, "curated": 16.0, "cc_high": 4.0, "cc_low": 2.0},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="predictions.csv from the selection scorer")
    parser.add_argument("--model", default="weibull_softplus_unscaled")
    args = parser.parse_args()
    harness.HELDOUT_DIR = args.registry_dir.resolve()
    panel = harness.load_panel(PANEL)
    types = np.array([bucket_type(name) for name in panel.buckets])
    predictions = pd.read_csv(args.predictions)
    rows = []
    for target in ("uncheatable", "table9"):
        bank = selection.load_bank(panel, target)
        _frame, features = harness.heldout_features(panel, target)
        guess = predictions[predictions["target"].eq(target) & predictions["model"].eq(args.model)].set_index(
            "coordinate_id"
        )["prediction"]
        guess = guess.reindex(bank.coordinate_id).to_numpy(float)
        for policy, caps in POLICIES.items():
            limit = np.array([caps[kind] for kind in types])
            feasible = (features.exposures <= limit[None, :] + 1e-9).all(axis=1)
            if feasible.sum() == 0:
                rows.append({"target": target, "policy": policy, "feasible": 0})
                continue
            loss = bank.measured[feasible]
            pick = int(np.argmin(guess[feasible]))
            rows.append(
                {
                    "target": target,
                    "policy": policy,
                    "feasible": int(feasible.sum()),
                    "best_feasible_measured": float(loss.min()),
                    "gap_to_bank_frontier": float(loss.min() - bank.measured.min()),
                    "frontier_feasible": bool(feasible[int(np.argmin(bank.measured))]),
                    "model_pick_measured": float(loss[pick]),
                    "model_pick_regret_in_policy": float(loss[pick] - loss.min()),
                }
            )
    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / f"cap_policies_{args.model.replace('@', '_')}.csv", index=False)
    pd.set_option("display.width", 220)
    print("bucket types:", pd.Series(types).value_counts().to_dict())
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
