# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare all paired native12,288 and draft16,384 measured endpoints directly."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=Path(__file__).resolve().parent)
    record = parser.parse_args().record
    source = record / "measurements/crossover16k-efficiency.json"
    d = json.loads(source.read_text())
    rows = []
    for p in d["pairs"]:
        draft = next(r for r in p["curves"]["4"] if r["horizon"] == 16384)
        native = next(r for r in p["curves"]["8"] if r["horizon"] == 12288)
        dc = draft["single_horizon_execution_seconds"]
        nc = native["update_seconds"]
        rows.append(
            {
                "seed": p["seed"],
                "trial": p["trial"],
                "draft_horizon": 16384,
                "native_horizon": 12288,
                "draft_gain": draft["fresh_gain"],
                "native_gain": native["fresh_gain"],
                "draft_cost": dc,
                "native_cost": nc,
                "draft_over_native_cost": dc / nc,
                "native_loss_advantage": native["fresh_gain"] - draft["fresh_gain"],
                "native_dominates": native["fresh_gain"] > draft["fresh_gain"] and nc < dc,
            }
        )
    assert len(rows) == 12 and len({(r["seed"], r["trial"]) for r in rows}) == 12
    result = {
        "scope": (
            "Post-hoc direct comparison of all twelve measured endpoint pairs; no interp"
            "olation or selection of passing pairs. Native12288 versus draft16384. Draft"
            " includes minimal copy/check overhead; native update execution."
        ),
        "source": str(source.relative_to(record)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "pairs": rows,
        "native_dominates_pairs": sum(r["native_dominates"] for r in rows),
        "median_draft_over_native_cost": median(r["draft_over_native_cost"] for r in rows),
        "mean_native_loss_advantage": mean(r["native_loss_advantage"] for r in rows),
    }
    (record / "measurements/crossover16k-measured-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
