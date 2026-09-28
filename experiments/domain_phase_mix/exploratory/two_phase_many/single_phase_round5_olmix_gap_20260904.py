# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-component audit of the WSPU-versus-OLMix Table-9 gap at the matched seed.

Fits the frozen successor (``weibull_softplus_unscaled``) per Table-9 component on the canonical 280-run panel
with the heldout inner-fold protocol, predicts every component at the OLMix mixture and at the frozen WSPU cap
6/7/8 mixtures, and compares the predicted OLMix-to-WSPU deltas with the observed matched-seed deltas. The
prediction is decomposed per bucket by hybrid mixtures (OLMix with one bucket swapped to the WSPU share), which is
exact for the additive successor; the additivity residual is reported. Extrapolation flags mark buckets whose
materialized epochs exceed the panel maximum.
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
    single_phase_round3_proposal_predictions_20260903 as proposals,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round4_cap_policies_20260903 as policies,
)

PANEL = "delphi_3e18_39bucket"
TARGET = "table9"
CAPS = (6, 7, 8)
SWEEP_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_one_phase_weibull_softplus_epoch_cap_sweep_20260902"
DEFAULT_OUTPUT_DIR = harness.DEFAULT_OUTPUT_DIR / "olmix_gap_round5"
FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("minerva_math", "math"),
    ("codex_humaneval", "code"),
    ("mbpp", "code"),
    ("mt_mbpp", "code"),
    ("arc_", "arc"),
    ("mmlu_", "mmlu"),
    ("basic_skills", "basic_skills"),
)
COMMONSENSE = frozenset({"csqa", "hellaswag", "winogrande", "socialiqa", "piqa"})
QA_READING = frozenset({"coqa", "drop", "jeopardy", "naturalqs", "squad", "sciq", "lambada", "medmcqa"})


def short_name(component: str) -> str:
    return component.split("/")[-2] if "/" in component else component


def family(component: str) -> str:
    name = short_name(component)
    for prefix, label in FAMILY_PREFIXES:
        if name.startswith(prefix):
            return label
    if name in COMMONSENSE:
        return "commonsense"
    if name in QA_READING:
        return "qa_reading"
    raise ValueError(f"unmapped Table-9 component {component}")


def olmix_weights(path: Path, buckets: tuple[str, ...]) -> np.ndarray:
    table = pd.read_csv(path, index_col=0)["weight"]
    outside = table.drop(index=[bucket for bucket in buckets if bucket in table.index])
    if float(outside.sum()) > 1e-9:
        raise ValueError(f"OLMix weights place mass outside the panel buckets: {outside[outside > 0].to_dict()}")
    weights = table.reindex(buckets).fillna(0.0).to_numpy(float)
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError("OLMix weights do not sum to one")
    return weights


def sweep_weights(buckets: tuple[str, ...]) -> tuple[dict[int, np.ndarray], dict[int, set[str]]]:
    table = pd.read_csv(SWEEP_DIR / "candidate_weights.csv")
    table = table[table["target"].eq(TARGET)]
    weights: dict[int, np.ndarray] = {}
    active: dict[int, set[str]] = {}
    for cap in CAPS:
        rows = table[table["candidate_id"].eq(f"wspu_table9_cap{cap:02d}")].set_index("domain")
        weights[cap] = rows["weight"].reindex(buckets).to_numpy(float)
        if not np.isclose(weights[cap].sum(), 1.0, atol=1e-6):
            raise ValueError(f"cap {cap} weights do not sum to one")
        active[cap] = set(rows.index[rows["cap_active"].astype(bool)])
    return weights, active


def evaluation_table(path: Path, components: tuple[str, ...]) -> pd.DataFrame:
    table = pd.read_csv(path, index_col=0)
    columns = [short_name(component) for component in components]
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"evaluation table lacks components {missing}")
    values = table[columns].astype(float)
    values.columns = list(components)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--olmix-weights", type=Path, required=True, help="CSV with bucket index and weight column")
    parser.add_argument("--evals", type=Path, required=True, help="CSV of W&B Table-9 component BPBs per run")
    parser.add_argument("--model", default="weibull_softplus_unscaled")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = harness.load_panel(PANEL)
    buckets = tuple(panel.buckets)
    group = panel.group(TARGET)
    components = tuple(group.components)
    inventory = panel.features.inventory
    olmix = olmix_weights(args.olmix_weights, buckets)
    caps, active = sweep_weights(buckets)
    evals = evaluation_table(args.evals, components)

    labels = ["olmix"] + [f"cap{cap}" for cap in CAPS]
    rows = [olmix] + [caps[cap] for cap in CAPS]
    for cap in CAPS:
        for index in range(len(buckets)):
            hybrid = olmix.copy()
            hybrid[index] = caps[cap][index]
            labels.append(f"hybrid:cap{cap}:{index}")
            rows.append(hybrid)
    query_weights = np.vstack(rows)
    query = dataclasses.replace(
        panel.features,
        exposures=query_weights * inventory[None, :],
        weights=query_weights,
        label="olmix_gap_round5",
    )
    with harness.parallel_config(backend="loky", inner_max_num_threads=1):
        parts = Parallel(n_jobs=args.workers, verbose=5)(
            delayed(proposals.fit_predict)(args.model, TARGET, index, query) for index in range(len(components))
        )
    predicted = pd.DataFrame(np.stack(parts, axis=1), index=labels, columns=list(components))

    mixtures = predicted.loc[["olmix"] + [f"cap{cap}" for cap in CAPS]]
    long = mixtures.reset_index(names="mixture").melt(id_vars="mixture", var_name="component", value_name="predicted")
    long["family"] = long["component"].map(family)
    long.to_csv(args.output_dir / "component_predictions.csv", index=False)

    delta_rows = []
    additivity = {}
    for cap in CAPS:
        matched = evals.loc[f"wspu_cap{cap}_matched"] - evals.loc["olmix_matched"]
        original = evals.loc[f"wspu_cap{cap}_orig"] - evals.loc["olmix_matched"]
        seed_gap = evals.loc[f"wspu_cap{cap}_orig"] - evals.loc[f"wspu_cap{cap}_matched"]
        predicted_delta = predicted.loc[f"cap{cap}"] - predicted.loc["olmix"]
        hybrid_sum = sum(
            predicted.loc[f"hybrid:cap{cap}:{index}"] - predicted.loc["olmix"] for index in range(len(buckets))
        )
        additivity[f"cap{cap}"] = float(np.abs(hybrid_sum - predicted_delta).max())
        for component in components:
            delta_rows.append(
                {
                    "cap": cap,
                    "component": short_name(component),
                    "family": family(component),
                    "predicted_olmix": float(predicted.loc["olmix", component]),
                    "observed_olmix": float(evals.loc["olmix_matched", component]),
                    "predicted_wspu": float(predicted.loc[f"cap{cap}", component]),
                    "observed_wspu_matched": float(evals.loc[f"wspu_cap{cap}_matched", component]),
                    "predicted_delta": float(predicted_delta[component]),
                    "observed_delta_matched": float(matched[component]),
                    "observed_delta_original_seed": float(original[component]),
                    "residual_matched": float(matched[component] - predicted_delta[component]),
                    "residual_original_seed": float(original[component] - predicted_delta[component]),
                    "wspu_seed_gap": float(seed_gap[component]),
                    "panel_repeat_sd": float(panel.component_repeat_sd.get(component, np.nan)),
                    "level_residual_olmix": float(
                        evals.loc["olmix_matched", component] - predicted.loc["olmix", component]
                    ),
                    "level_residual_wspu": float(
                        evals.loc[f"wspu_cap{cap}_matched", component] - predicted.loc[f"cap{cap}", component]
                    ),
                }
            )
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(args.output_dir / "component_deltas.csv", index=False)

    decomposition = []
    panel_max = panel.features.exposures.max(axis=0)
    for cap in CAPS:
        for index, bucket in enumerate(buckets):
            contribution = predicted.loc[f"hybrid:cap{cap}:{index}"] - predicted.loc["olmix"]
            for component in components:
                decomposition.append(
                    {
                        "cap": cap,
                        "bucket": bucket,
                        "bucket_type": policies.bucket_type(bucket),
                        "component": short_name(component),
                        "family": family(component),
                        "contribution_delta": float(contribution[component]),
                    }
                )
    pd.DataFrame(decomposition).to_csv(args.output_dir / "bucket_decomposition.csv", index=False)

    extrapolation = []
    panel_weights = panel.features.weights
    for label, weights in (("olmix", olmix), *[(f"cap{cap}", caps[cap]) for cap in CAPS]):
        tv = 0.5 * np.abs(panel_weights - weights[None, :]).sum(axis=1)
        for index, bucket in enumerate(buckets):
            extrapolation.append(
                {
                    "mixture": label,
                    "bucket": bucket,
                    "bucket_type": policies.bucket_type(bucket),
                    "weight": float(weights[index]),
                    "weight_olmix": float(olmix[index]),
                    "epochs": float(weights[index] * inventory[index]),
                    "epochs_olmix": float(olmix[index] * inventory[index]),
                    "panel_max_epochs": float(panel_max[index]),
                    "beyond_panel": bool(weights[index] * inventory[index] > panel_max[index] * (1 + 1e-9)),
                    "cap_active": bool(label != "olmix" and bucket in active[int(label[3:])]),
                    "nearest_panel_tv": float(tv.min()),
                }
            )
    pd.DataFrame(extrapolation).to_csv(args.output_dir / "extrapolation.csv", index=False)

    summary = {
        "model": args.model,
        "macro_predicted": {label: float(mixtures.loc[label].mean()) for label in mixtures.index},
        "macro_observed_matched": {
            "olmix": float(evals.loc["olmix_matched"].mean()),
            **{f"cap{cap}": float(evals.loc[f"wspu_cap{cap}_matched"].mean()) for cap in CAPS},
        },
        "sweep_runtime_predictions": (
            pd.read_csv(SWEEP_DIR / "candidate_summary.csv")
            .query("target == 'table9'")
            .set_index("epoch_cap")["runtime_predicted_bpb"]
            .to_dict()
        ),
        "additivity_max_abs_gap": additivity,
        "nearest_panel_tv": {
            label: float((0.5 * np.abs(panel_weights - weights[None, :]).sum(axis=1)).min())
            for label, weights in (("olmix", olmix), *[(f"cap{cap}", caps[cap]) for cap in CAPS])
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
