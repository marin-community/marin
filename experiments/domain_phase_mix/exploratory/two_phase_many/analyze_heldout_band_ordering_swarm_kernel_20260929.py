# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Best-k ordering of the held-out optima for a kernel smoother over the swarm alone.

A Nadaraya-Watson smoother on total-variation distance predicts each held-out optimum from the 280 swarm runs (never
from another held-out optimum), at bandwidths 0.05, 0.1 and 0.2; a mixture farther than every swarm run gets the
swarm mean. Scored like MARINER in `analyze_heldout_band_ordering_pbt_20260929.py` on each objective's mean and on
arXiv physics and SciQ alone, with the source-block bootstrap interval.

Writes `swarm_kernel_band_ordering.csv` to `reference_outputs/delphi_top_band_ordering_20260928/`.

usage: uv run --offline --no-sync python analyze_heldout_band_ordering_swarm_kernel_20260929.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[3]))

import analyze_delphi_top_band_ordering_20260906 as top_band  # noqa: E402
import analyze_heldout_band_ordering_pbt_20260929 as pbt  # noqa: E402
import analyze_heldout_component_ordering_20260929 as ordering  # noqa: E402
import analyze_heldout_component_predictability_20260929 as predictability  # noqa: E402
import materialize_delphi_comparator_proposals_20260909 as mat  # noqa: E402
import preview_band_ordering_snr_subsets_20260929 as preview  # noqa: E402

BANDWIDTHS = (0.05, 0.1, 0.2)
COMPONENTS = {"uncheatable": "arxiv_physics", "table9": "sciq"}


def swarm_kernel(
    mixtures: np.ndarray, swarm_weights: np.ndarray, swarm_values: np.ndarray, bandwidth: float
) -> np.ndarray:
    distance = 0.5 * np.abs(mixtures[:, None, :] - swarm_weights[None, :, :]).sum(axis=2)
    kernel = np.exp(-0.5 * (distance / bandwidth) ** 2)
    mass = kernel.sum(axis=1)
    return np.where(mass > 1e-12, (kernel @ swarm_values) / np.maximum(mass, 1e-300), swarm_values.mean())


def main() -> None:
    panel, swarm, from_panel = ordering.swarm_inputs()
    buckets = [str(b) for b in panel.buckets]
    swarm_weights = np.asarray(swarm.weights, float)[:, from_panel]  # swarm order -> panel bucket order
    rows = []
    for target, component in COMPONENTS.items():
        full = mat.ms.read_objectives(mat.ms.DATA / "objectives.csv")[target]
        objective_values = np.stack([swarm.outcomes[c].to_numpy(float) for c in full.components], axis=1).mean(axis=1)
        labels = ordering.held_out_labels(target)
        features = np.load(preview.BENCHMARK / "inputs" / f"{target}_bank_features.npz", allow_pickle=True)
        held = (
            pd.DataFrame(features["weights"], index=[str(c) for c in features["coordinate_id"]], columns=buckets)
            .loc[labels.coordinate_id]
            .to_numpy(float)
        )
        frame = ordering.component_frame(target, component)
        component_values = frame.swarm_measured
        component_mixtures = (
            (frame.exposures / pd.Series(swarm.inventory, index=list(swarm.buckets)))
            .reindex(columns=buckets)
            .to_numpy(float)
        )
        for bandwidth in BANDWIDTHS:
            label = f"swarm kernel, TV {bandwidth}"
            rows.extend(
                pbt.band_rows(target, label, labels, swarm_kernel(held, swarm_weights, objective_values, bandwidth))
            )
            predicted = swarm_kernel(component_mixtures, swarm_weights, component_values, bandwidth)
            for size in pbt.BAND_SIZES:
                k = len(frame.measured) if size == 0 else size
                row = {"target": target, "method": label, "component": component, "band": k}
                row.update(
                    predictability.band_statistics(
                        frame.measured, predicted, frame.blocks, frame.run_sd, frame.counts, k
                    )
                )
                order = np.argsort(frame.measured, kind="stable")[:k]
                row["sign_accuracy_ci_low"], row["sign_accuracy_ci_high"] = top_band.bootstrap_accuracy(
                    frame.measured[order], predicted[order], frame.blocks[order], frame.run_sd, None
                )
                rows.append(row)
        nearest = 0.5 * np.abs(held[:, None, :] - swarm_weights[None, :, :]).sum(axis=2).min(axis=1)
        print(
            f"{target}: nearest-swarm-run distance median {np.median(nearest):.2f}, "
            f"min {nearest.min():.2f}, max {nearest.max():.2f}",
            flush=True,
        )
    table = pd.DataFrame(rows)
    table.to_csv(pbt.OUTPUT / "swarm_kernel_band_ordering.csv", index=False)
    print(
        table[
            [
                "target",
                "method",
                "component",
                "band",
                "sign_accuracy",
                "sign_accuracy_ci_low",
                "sign_accuracy_ci_high",
                "spearman",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
