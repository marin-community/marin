# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Best-k ordering of the held-out optima for RegMix, on the objectives and on the two matched-SNR components.

RegMix (the registry's `lightgbm_regmix`, one LightGBM head per component, aggregated with the objective's weights) is
fitted on the full 280-run panel by the harness's heldout-stage protocol (`fit_component` of
`materialize_delphi_comparator_proposals_20260909.py`) and scored on the held-out optima like MARINER in
`analyze_heldout_band_ordering_pbt_20260929.py`: pairwise sign accuracy among the k best-measured optima with the
source-block bootstrap interval, on each objective's mean and on the components of `COMPONENTS` alone.

Writes `regmix_band_ordering.csv` and `regmix_heldout_task_predictions.csv` (every component's prediction at every
held-out optimum) to `reference_outputs/delphi_top_band_ordering_20260928/`.

usage: uv run --offline --no-sync --with lightgbm python analyze_heldout_band_ordering_regmix_20260929.py [--workers 8]
"""

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[3]))

import analyze_delphi_top_band_ordering_20260906 as top_band  # noqa: E402
import analyze_heldout_band_ordering_pbt_20260929 as pbt  # noqa: E402
import analyze_heldout_component_ordering_20260929 as ordering  # noqa: E402
import analyze_heldout_component_predictability_20260929 as predictability  # noqa: E402
import materialize_delphi_comparator_proposals_20260909 as mat  # noqa: E402
import preview_band_ordering_snr_subsets_20260929 as preview  # noqa: E402

MODEL_ID = "lightgbm_regmix"
LABEL = "RegMix"
COMPONENTS = {
    "uncheatable": ("arxiv_physics", "arxiv_computer_science"),
    "table9": (
        "sciq",
        "hellaswag",
        "socialiqa",
        "piqa",
        "minerva_math_geometry",
        "mmlu_other",
        "mt_mbpp_typescript",
        "mt_mbpp_csharp",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    panel = mat.bench.load_panel(mat.PANEL)
    inner = mat.bench.heldout_inner_folds(panel)
    rows = []
    task_predictions = []
    for target, components in COMPONENTS.items():
        group = panel.group(target)
        fits = Parallel(n_jobs=args.workers)(
            delayed(mat.fit_component)(panel, MODEL_ID, target, index, inner) for index in range(len(group.components))
        )
        surrogate = mat.ObservatorySurrogate(panel, MODEL_ID, target, list(fits))
        labels = ordering.held_out_labels(target)
        features = np.load(preview.BENCHMARK / "inputs" / f"{target}_bank_features.npz", allow_pickle=True)
        weights = pd.DataFrame(
            features["weights"],
            index=[str(c) for c in features["coordinate_id"]],
            columns=[str(b) for b in panel.buckets],
        ).loc[labels.coordinate_id]
        rows.extend(pbt.band_rows(target, LABEL, labels, surrogate.predict(weights.to_numpy(float))))
        print(f"{target} objective: scored", flush=True)
        names = [preview.short_name(str(c)) for c in group.components]
        tasks = pd.DataFrame(surrogate.predict_tasks(weights.to_numpy(float)), columns=names, index=labels.coordinate_id)
        stacked = tasks.stack().rename("prediction").reset_index()
        stacked.columns = ["coordinate_id", "component", "prediction"]
        task_predictions.append(stacked.assign(target=target))
        for component in components:
            frame = ordering.component_frame(target, component)
            predicted = tasks[component].reindex(frame.coordinate_id).to_numpy()
            for size in pbt.BAND_SIZES:
                k = len(frame.measured) if size == 0 else size
                row = {"target": target, "method": LABEL, "component": component, "band": k}
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
            print(f"{target} {component}: scored", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(pbt.OUTPUT / "regmix_band_ordering.csv", index=False)
    pd.concat(task_predictions, ignore_index=True).to_csv(
        pbt.OUTPUT / "regmix_heldout_task_predictions.csv", index=False
    )
    print(
        table[
            ["target", "method", "component", "band", "sign_accuracy", "sign_accuracy_ci_low", "sign_accuracy_ci_high"]
        ]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
