# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Marginal optimal epoch count of every bucket on every evaluation, from the frozen MARINER fits.

For each panel component, MARINER is fitted on all rows with the paper's frozen registry entry. The
exponent is bucket-separable and the link is monotone, so the epoch count that minimizes the fitted
loss when only one bucket's exposure varies (others at the panel's mean policy) is that bucket's
marginal optimum for that evaluation, independent of pool size. Run from the repository root:
PYTHONPATH=. uv run --offline --no-sync python \
    -m experiments.domain_phase_mix.exploratory.two_phase_many.marginal_epoch_optima_20260919
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_single_phase_observatory_20260902 as bench
from experiments.domain_phase_mix.exploratory.two_phase_many import single_phase_observatory_models_20260902 as models
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)

OUTPUT = Path(__file__).resolve().parent / "reference_outputs" / "marginal_epoch_optima_20260919"
PANELS = ("delphi_3e18_39bucket", "300m_39bucket")
EPOCHS = np.concatenate([[0.0], np.logspace(-2, np.log10(64.0), 500)])
PROBES = (1.0, 2.0, 4.0, 8.0, 16.0)


def sweep_features(panel: bench.BenchPanel, bucket: int, component: str) -> models.Features:
    """Rows where only ``bucket`` varies over EPOCHS; the other buckets sit at the panel's mean weights."""
    base = panel.features
    weights = np.repeat(base.weights.mean(axis=0, keepdims=True), len(EPOCHS), axis=0)
    weights[:, bucket] = EPOCHS / base.inventory[bucket]
    features = models.features_from_panel(
        weights, base.inventory, base.buckets_names, early_fraction=base.early_fraction, label=f"{panel.name}|sweep"
    )
    return dataclasses.replace(features, component=component)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    entry = registry.ENTRY_BY_ID[registry.FROZEN_ID]
    rows = []
    started = time.monotonic()
    for panel_name in PANELS:
        panel = bench.load_panel(panel_name)
        train = np.arange(panel.rows)
        inner = bench.heldout_inner_folds(panel)
        family_of = {}
        for name, members in zip(panel.features.families.names, panel.features.families.members, strict=True):
            for member in members:
                family_of[int(member)] = name
        for group in panel.groups:
            for index, component in enumerate(group.components):
                features = dataclasses.replace(registry.apply_transform(panel.features, entry), component=component)
                model = entry.build(features)
                response = group.outcomes[:, index].astype(float)
                fitted = model.fit(features, response, train, inner, 0)
                design = model.base.design(features, fitted.shape)
                coefficients = dict(zip(design.names, np.asarray(fitted.head.coefficients, dtype=float), strict=True))
                for bucket in range(len(panel.buckets)):
                    query = sweep_features(panel, bucket, component)
                    loss = np.asarray(model.predict(fitted, query, np.arange(len(EPOCHS))), dtype=float)
                    best = int(np.argmin(loss))
                    minimum = float(loss[best])
                    excess = {
                        f"excess_pct_at_{int(e)}": 100 * (float(np.interp(e, EPOCHS, loss)) - minimum) / minimum
                        for e in PROBES
                    }
                    benefit = coefficients.get(f"bucket_signal:{bucket}", float("nan"))
                    harm_keys = [
                        k for k in coefficients if k.endswith(f":{bucket}") and not k.startswith("bucket_signal")
                    ]
                    harm = float(sum(coefficients[k] for k in harm_keys)) if harm_keys else float("nan")
                    rows.append(
                        {
                            "panel": panel_name,
                            "target": group.name,
                            "component": component,
                            "bucket": panel.buckets[bucket],
                            "family": family_of.get(bucket, ""),
                            "inventory_epochs_at_weight_1": float(panel.features.inventory[bucket]),
                            "optimum_epochs": float(EPOCHS[best]),
                            "boundary": best in (0, len(EPOCHS) - 1),
                            "depth_pct": 100 * (float(loss[0]) - minimum) / minimum,
                            **excess,
                            "benefit_coefficient": benefit,
                            "harm_coefficient": harm,
                            "shape": json.dumps(fitted.shape, sort_keys=True),
                            "inner_cv_rmse": float(fitted.diagnostics.get("inner_cv_rmse", float("nan"))),
                        }
                    )
                print(f"{panel_name} {group.name} {component}: {time.monotonic() - started:.0f}s", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "marginal_optima.csv", index=False)
    print("rows", len(frame))


if __name__ == "__main__":
    main()
