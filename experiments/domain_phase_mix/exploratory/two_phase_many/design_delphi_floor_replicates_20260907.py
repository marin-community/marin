# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replicate the best single-run Table-9 coordinates of the frozen bank at the validation seed.

The bank's Table-9 floor is a 0.023 BPB band of competing optima, almost all measured once (run SD 0.0038), and
no panel-fitted surrogate orders that band (`analyze_delphi_top_band_ordering_20260906.py`). The cheapest
measured answer to "which mixture is best" is to replicate the best under-replicated coordinates at the
validation data seed with two trainer seeds each, so that every candidate floor coordinate has three runs and
its difference from the 26-run centre (1.0639, SD 0.0041) has a standard error of about 0.0024 BPB.

Selection rule, fixed in advance: the five best-measured optima-stratum coordinates with fewer than three runs.
Weights come from the frozen bank features; four of the five are recorded as continuous targets, so rounding
them to the 2048-count runtime grid moves at most 0.003 TV (reported per row). The nominal 17-epoch cap exceeds
every candidate's largest materialized epoch and never binds. Two identical tables are written, one per trainer
seed, because the sweep loader aliases identical mixtures within a table. Nothing is launched.

usage: uv run python design_delphi_floor_replicates_20260907.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_delphi_selection_20260906 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    design_delphi_frontier_factorial_20260906 as factorial,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_delphi_one_phase_surrogate_challengers_20260831 as grid,
)

FROZEN = benchmark.DEFAULT_OUTPUT
DEFAULT_OUTPUT = SCRIPT_DIR / "reference_outputs" / "delphi_floor_replicates_design_20260907"
INTERVENTIONS = ("conditional_epoch_dose_response", "archive::delphi_baseline_mixtures_issue6607_20260623")
CANDIDATES = 5
MAX_EXISTING_RUNS = 2
TRAINER_SEEDS = (0, 1)
# No cap binds: 17 exceeds every candidate's largest materialized epoch, so the bank coordinate is reproduced
# exactly; the suffix is the loader's requirement, not a policy.
EPOCH_CAP = 17.0
KERNEL_BANDWIDTH = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = benchmark.read_npz(FROZEN / "inputs" / "panel.npz")
    buckets = [str(b) for b in data["buckets"]]
    inventory = pd.Series(data["inventory"], index=buckets)
    bank = benchmark.read_npz(FROZEN / "inputs" / "table9_bank_features.npz")
    labels = pd.read_csv(FROZEN / "inputs" / "table9_bank_labels.csv").set_index("coordinate_id")
    frame = pd.DataFrame(bank["weights"], columns=buckets, index=bank["coordinate_id"].astype(str))
    labels = labels.loc[frame.index]
    optima = labels[~labels.sources.str.contains("|".join(INTERVENTIONS))].sort_values("measured_mean_bpb")
    chosen = optima[optima.run_count.le(MAX_EXISTING_RUNS)].head(CANDIDATES)
    centre = frame.loc[factorial.CENTRE_ID]
    measured = labels.measured_mean_bpb.to_numpy(float)
    maximum = np.floor(np.minimum(1.0, EPOCH_CAP / inventory.to_numpy()) * factorial.BLOCK_SIZE + 1e-12).astype(np.int64)
    design_rows, weight_rows = [], []
    for rank, (coordinate, row) in enumerate(chosen.iterrows(), start=1):
        weights = frame.loc[coordinate].to_numpy(float)
        counts = grid.prefix_materializer.constrained_counts(weights, maximum)
        if int(counts.sum()) != factorial.BLOCK_SIZE or np.any(counts > maximum):
            raise ValueError(f"{coordinate}: runtime counts violate the grid or the cap")
        runtime = counts / factorial.BLOCK_SIZE
        epochs = runtime * inventory.to_numpy()
        distance = np.abs(frame.to_numpy() - runtime[None, :]).sum(axis=1) / 2
        kernel = np.exp(-0.5 * (distance / KERNEL_BANDWIDTH) ** 2)
        name = f"floor_{rank}_{coordinate.split(':')[1][:8]}_cap{int(EPOCH_CAP):02d}"
        design_rows.append(
            {
                "candidate_id": name,
                "coordinate_id": coordinate,
                "sources": row.sources,
                "existing_runs": int(row.run_count),
                "measured_mean_bpb": float(row.measured_mean_bpb),
                "tv_to_centre": float(np.abs(runtime - centre.to_numpy()).sum() / 2),
                "tv_moved_by_grid": float(np.abs(runtime - weights).sum() / 2),
                "max_materialized_epoch": float(epochs.max()),
                "max_epoch_bucket": buckets[int(np.argmax(epochs))],
                "kernel_forecast_tv0.05": float(kernel @ measured / kernel.sum()),
            }
        )
        weight_rows.extend(
            {
                "candidate_id": name,
                "target": "table9",
                "target_label": "Table-9 macro",
                "epoch_cap": int(EPOCH_CAP),
                "domain": bucket,
                "runtime_count": int(count),
                "weight": float(count / factorial.BLOCK_SIZE),
                "materialized_epochs": float(epoch),
            }
            for bucket, count, epoch in zip(buckets, counts, epochs, strict=True)
        )
    design = pd.DataFrame(design_rows)
    design.to_csv(args.output_dir / "design.csv", index=False)
    table = pd.DataFrame(weight_rows)
    digests = {}
    for seed in TRAINER_SEEDS:
        path = args.output_dir / f"candidate_weights_t{seed}.csv"
        table.to_csv(path, index=False)
        digests[seed] = hashlib.sha256(path.read_bytes()).hexdigest()
    centre_row = labels.loc[factorial.CENTRE_ID]
    readme = [
        "# Floor replicates around the Table-9 optimum (prepared 2026-09-07, not launched)",
        "",
        f"Rule: the {CANDIDATES} best-measured optima-stratum coordinates of the frozen bank with at most "
        f"{MAX_EXISTING_RUNS} existing runs, each trained twice more at data seed 662009 (trainer seeds "
        f"{', '.join(map(str, TRAINER_SEEDS))}). Reference: the centre `{factorial.CENTRE_ID.split(':')[1][:8]}` "
        f"with {int(centre_row.run_count)} runs at {centre_row.measured_mean_bpb:.4f}.",
        "",
        design.round(4).to_markdown(index=False),
        "",
        "Tables: " + ", ".join(f"`candidate_weights_t{seed}.csv` sha256 `{digest}`" for seed, digest in digests.items()),
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n")
    pd.set_option("display.width", 250)
    print(design.round(4).to_string(index=False))
    for seed, digest in digests.items():
        print(f"candidate_weights_t{seed}.csv sha256 {digest}")


if __name__ == "__main__":
    main()
