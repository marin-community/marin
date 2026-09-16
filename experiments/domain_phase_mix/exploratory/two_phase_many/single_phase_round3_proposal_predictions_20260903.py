# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Predictions of several panel-fitted models at proposed mixtures, against the bank frontier.

Reads the mixtures JSON written by ``single_phase_round3_proposals_20260903.py``, fits each requested model per
component on the canonical panel (heldout inner folds), and reports the predicted value at each proposal, the
prediction at the measured frontier coordinate, and the implied gain. Agreement across models is the point.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
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
    single_phase_observatory_registry_20260902 as registry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_heldout_selection_20260903 as selection,
)

PANEL = "delphi_3e18_39bucket"


def fit_predict(model_id: str, target: str, component_index: int, query) -> np.ndarray:
    panel = harness.load_panel(PANEL)
    entry = registry.ENTRY_BY_ID[model_id]
    group = panel.group(target)
    component = group.components[component_index]
    features = dataclasses.replace(registry.apply_transform(panel.features, entry), component=str(component))
    model = entry.build(features)
    fitted = model.fit(
        features,
        group.outcomes[:, component_index],
        np.arange(panel.rows),
        harness.heldout_inner_folds(panel),
        harness._seed(harness.FitTask(model_id, PANEL, target, component_index, component, 0, 0)),
    )
    query_features = dataclasses.replace(registry.apply_transform(query, entry), component=str(component))
    return np.asarray(model.predict(fitted, query_features, np.arange(query.rows)), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-dir", type=Path, required=True, help="heldout registry directory (use the corrected view)"
    )
    parser.add_argument("--output-dir", type=Path, default=harness.DEFAULT_OUTPUT_DIR / "heldout_round3_corrected")
    parser.add_argument("--mixtures", nargs="+", required=True, help="mixtures JSON files from the proposal search")
    parser.add_argument(
        "--models",
        default="weibull_softplus_unscaled,weibull_softplus_unscaled@log_deficit_bounded_link,"
        "weibull_softplus_unscaled@link_by_cv,dsp_total_exposure_concentration,olmix_loglinear_taskwise",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tag", default="proposal_predictions")
    args = parser.parse_args()
    harness.HELDOUT_DIR = args.registry_dir.resolve()
    panel = harness.load_panel(PANEL)
    mixtures: dict[str, dict[str, float]] = {}
    for path in args.mixtures:
        payload = json.loads(Path(path).read_text())
        mixtures.update({f"{Path(path).stem}:{key}": value for key, value in payload.items()})
    keys = list(mixtures)
    weights = np.array([[mixtures[key][bucket] for bucket in panel.buckets] for key in keys])
    rows = []
    for target in ("uncheatable", "table9"):
        bank = selection.load_bank(panel, target)
        _frame, bank_features = harness.heldout_features(panel, target)
        frontier = int(np.argmin(bank.measured))
        query_weights = np.vstack([weights, bank_features.weights[[frontier]]])
        query = dataclasses.replace(
            panel.features,
            exposures=query_weights * panel.features.inventory[None, :],
            weights=query_weights,
            label="proposals",
        )
        group = panel.group(target)
        for model_id in [token.strip() for token in args.models.split(",") if token.strip()]:
            with harness.parallel_config(backend="loky", inner_max_num_threads=1):
                parts = Parallel(n_jobs=args.workers)(
                    delayed(fit_predict)(model_id, target, index, query) for index in range(len(group.components))
                )
            predicted = np.stack(parts, axis=1) @ group.aggregation_weights
            for index, key in enumerate(keys):
                rows.append(
                    {
                        "target": target,
                        "model": model_id,
                        "proposal": key,
                        "predicted": float(predicted[index]),
                        "predicted_frontier": float(predicted[-1]),
                        "gain_vs_frontier": float(predicted[-1] - predicted[index]),
                        "frontier_measured": float(bank.measured[frontier]),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / f"{args.tag}.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    for target in ("uncheatable", "table9"):
        subset = table[table["target"].eq(target)]
        wide = subset.pivot(index="proposal", columns="model", values="gain_vs_frontier")
        wide.columns = [
            column.replace("weibull_softplus_unscaled", "WSPU")
            .replace("dsp_total_exposure", "DSP")
            .replace("olmix_loglinear_taskwise", "OLMix")
            for column in wide.columns
        ]
        wide["models_agreeing_gain"] = (wide > 0).sum(axis=1)
        print(
            f"\n=== {target}: predicted gain over the frontier (positive = better than the measured-best bank "
            f"coordinate); frontier measured {subset['frontier_measured'].iloc[0]:.4f}"
        )
        print(wide.round(4).sort_values("models_agreeing_gain", ascending=False).to_string())


if __name__ == "__main__":
    main()
