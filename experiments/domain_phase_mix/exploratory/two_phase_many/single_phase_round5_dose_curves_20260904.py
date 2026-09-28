# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Fitted single-bucket dose curves of a panel-fitted additive model, per Table-9 component.

For each requested bucket the model is queried at rows that expose only that bucket, on an epoch grid, so the
returned value minus the value at zero exposure is the bucket's fitted contribution curve. Exact for additive
models such as ``weibull_softplus_unscaled``; other models get the same query and should be read as a slice.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_proposal_predictions_20260903 as proposals,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round5_olmix_gap_20260904 as gap,
)

DEFAULT_BUCKETS = (
    "dolmino_synth_qa",
    "dolmino_olmocr_pdfs_hq",
    "dolma3_stack_edu",
    "dolmino_stack_edu_fim",
    "dolmino_synth_code",
    "dolmino_synth_math",
    "dolmino_synth_instruction",
    "dolma3_cc/industrial_low",
    "dolma3_cc/food_and_dining_low",
    "dolma3_cc/electronics_and_hardware_low",
    "dolmino_common_crawl_hq",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="weibull_softplus_unscaled")
    parser.add_argument("--buckets", nargs="+", default=list(DEFAULT_BUCKETS))
    parser.add_argument("--max-epochs", type=float, default=10.0)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=gap.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = harness.load_panel(gap.PANEL)
    buckets = tuple(panel.buckets)
    group = panel.group(gap.TARGET)
    inventory = panel.features.inventory
    grid = np.arange(0.0, args.max_epochs + 1e-9, args.step)
    labels = ["zero"]
    rows = [np.zeros(len(buckets))]
    for bucket in args.buckets:
        index = buckets.index(bucket)
        for epochs in grid[1:]:
            weights = np.zeros(len(buckets))
            weights[index] = epochs / inventory[index]
            labels.append(f"{bucket}:{epochs:.2f}")
            rows.append(weights)
    query_weights = np.vstack(rows)
    query = dataclasses.replace(
        panel.features,
        exposures=query_weights * inventory[None, :],
        weights=query_weights,
        label="dose_curves_round5",
    )
    with harness.parallel_config(backend="loky", inner_max_num_threads=1):
        parts = Parallel(n_jobs=args.workers, verbose=5)(
            delayed(proposals.fit_predict)(args.model, gap.TARGET, index, query)
            for index in range(len(group.components))
        )
    predicted = np.stack(parts, axis=1)
    contribution = predicted[1:] - predicted[:1]
    records = []
    for row, label in enumerate(labels[1:]):
        bucket, epochs = label.rsplit(":", 1)
        panel_max = float(panel.features.exposures[:, buckets.index(bucket)].max())
        for column, component in enumerate(group.components):
            records.append(
                {
                    "model": args.model,
                    "bucket": bucket,
                    "epochs": float(epochs),
                    "panel_max_epochs": panel_max,
                    "component": gap.short_name(component),
                    "family": gap.family(component),
                    "contribution": float(contribution[row, column]),
                }
            )
    table = pd.DataFrame(records)
    table.to_csv(args.output_dir / f"dose_curves_{args.model}.csv", index=False)
    macro = table.groupby(["bucket", "epochs"])["contribution"].mean().unstack("epochs")
    pd.set_option("display.width", 250)
    print(macro[[c for c in macro.columns if abs(c / 1.0 - round(c)) < 1e-9]].round(4).to_string())


if __name__ == "__main__":
    main()
