# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Materialize the trained-proposal comparators of the complexity ladder at 3e18.

The offline ladder (`reference_outputs/complexity_ladder_20260909/`) scores every surrogate on prediction and on
selection among measured runs. This script produces the mixture each comparator proposes for training, so that
optimization quality can be measured in new runs: the additive quadratic in log-epochs under MARINER's floor and
log link, the natural cubic spline under the same link, RegMix's gradient-boosted trees, Hellinger kernel ridge, and
MARINER with the benefit power fixed at one.

Every model is fitted on all 280 runs of the frozen Qwen3 3e18 swarm with the harness's heldout-stage protocol
(the certify inner folds, the proportional run pinned to the training side, the same floor anchors and noise
margins). Smooth models are minimized by MARINER's optimizer from the reference implementation: SLSQP from the
proportional mixture and four seeded swarm rows, polished, rounded to the 1/2048 runtime grid with the exchange
search, no epoch cap and no penalty. The trees follow RegMix's recipe: one million candidates from the swarm's
sampling law (uniform on the simplex), the top 128 predictions averaged, then rounded to the grid. The MARINER
variant uses the reference implementation with its shape grid restricted to power one.

OpenMP is pinned to one thread before any import: scikit-learn (the inner folds' k-means) and LightGBM each load
their own libomp, and two active runtimes segfault in the same process. Single-threaded, a LightGBM fit takes
three seconds instead of thirty, and the whole script runs in about ten minutes.

usage: uv run --offline --no-sync --with lightgbm python materialize_delphi_comparator_proposals_20260909.py
           [--output-dir DIR] [--candidate-samples N] [--comparators KEY,KEY]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# See the module docstring: one OpenMP thread everywhere, set before numpy, scikit-learn or LightGBM load.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_THREAD_LIMIT"] = "1"

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
REFERENCE_PACKAGE = REPO_ROOT.parent / "mixture-selection"
for root in (REPO_ROOT, REFERENCE_PACKAGE):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import mixture_selection as ms  # noqa: E402

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_registry_20260902 as registry,
)

PANEL = "delphi_3e18_39bucket"
OUTPUT = SCRIPT_DIR / "reference_outputs" / "delphi_comparator_proposals_3e18_20260909"
TARGET_LABELS = {"uncheatable": "Uncheatable", "table9": "Table 9"}
TARGET_KEYS = {"uncheatable": "u", "table9": "t9"}
# The frozen procedure's unconstrained proposals (cap inactive), from the reference package.
MARINER_POLICY = {"uncheatable": "lwspu_u_snc_cap06", "table9": "lwspu_t9_snc_cap08"}
MARINER_ID = "weibull_softplus_unscaled@kappa_floor_link_flat15_nocap"
CANDIDATE_SAMPLES = 1_000_000
TOP_K = 128
SAMPLE_SEED = 20_260_909
SAMPLE_CHUNK = 100_000
NOMINAL_CAPS = (6, 8, 12, 16, 24, 32, 64)
PARITY_TV = 1e-9


@dataclasses.dataclass(frozen=True)
class Comparator:
    key: str
    label: str
    model_id: str
    optimizer: str  # slsqp | regmix_topk | reference_power1


COMPARATORS = (
    Comparator("quad", "Quadratic in log-epochs, floor link", registry.QUADRATIC_LINKED_ID, "slsqp"),
    Comparator("spline", "Natural cubic spline in log-epochs, floor link", registry.SPLINE_LINKED_ID, "slsqp"),
    Comparator("lgbm", "LightGBM (RegMix)", "lightgbm_regmix", "regmix_topk"),
    Comparator("krr", "Hellinger kernel ridge", registry.KRR_ID, "slsqp"),
    Comparator("mk1", "MARINER, exponential benefit", f"{MARINER_ID}_kappa1", "reference_power1"),
    # Added 2026-09-14 for the second validation batch: the Table 1 ablations fitted by the harness's heldout-stage
    # protocol, so their proposals are comparable with the quadratic and spline rows.
    Comparator("cvx", "MARINER, fully convex objective (convex harm)", f"{MARINER_ID}_raw_epoch_hinge", "slsqp"),
    Comparator("add", "MARINER, additive response", "weibull_softplus_unscaled", "slsqp"),
)


@dataclasses.dataclass(frozen=True)
class SurrogateObjective:
    """What the reference optimizer reads from an objective fit: the exposure scale and a batch predictor."""

    inventory: np.ndarray
    predict: Callable[[np.ndarray], np.ndarray]


def fit_component(
    panel: bench.BenchPanel, model_id: str, target: str, index: int, inner: tuple[tuple[np.ndarray, np.ndarray], ...]
) -> models.Fitted:
    """The harness's heldout-stage fit of one component on every run of the panel."""
    entry = registry.ENTRY_BY_ID[model_id]
    group = panel.group(target)
    features = dataclasses.replace(
        registry.apply_transform(panel.features, entry), component=str(group.components[index])
    )
    model = entry.build(features)
    return model.fit(features, group.outcomes[:, index], np.arange(panel.rows), inner, 0)


class ObservatorySurrogate:
    """Per-task observatory fits of one model, aggregated with the objective's evaluation weights."""

    def __init__(self, panel: bench.BenchPanel, model_id: str, target: str, fits: list[models.Fitted]) -> None:
        self.panel = panel
        self.entry = registry.ENTRY_BY_ID[model_id]
        self.group = panel.group(target)
        self.fits = fits
        self.models = []
        for component in self.group.components:
            features = dataclasses.replace(
                registry.apply_transform(panel.features, self.entry), component=str(component)
            )
            self.models.append(self.entry.build(features))
        if len(self.models) != len(fits):
            raise ValueError("one fit per component expected")

    def predict_tasks(self, mixtures: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(mixtures, dtype=float))
        features = models.features_from_panel(
            rows,
            self.panel.features.inventory,
            self.panel.buckets,
            early_fraction=self.panel.features.early_fraction,
            label=self.panel.features.label,
        )
        query = registry.apply_transform(features, self.entry)
        out = np.empty((len(rows), len(self.models)))
        index = np.arange(len(rows))
        for column, (model, fitted, component) in enumerate(
            zip(self.models, self.fits, self.group.components, strict=True)
        ):
            out[:, column] = model.predict(fitted, dataclasses.replace(query, component=str(component)), index)
        return out

    def predict(self, mixtures: np.ndarray) -> np.ndarray:
        return self.predict_tasks(mixtures) @ self.group.aggregation_weights


def reference_swarm_in_panel_order(panel: bench.BenchPanel) -> tuple[ms.Swarm, np.ndarray]:
    """The reference package's swarm and the permutation taking its bucket order to the panel's."""
    swarm = ms.read_swarm(ms.DATA / "swarm_weights.csv", ms.DATA / "swarm_outcomes.csv", ms.DATA / "buckets.csv")
    if set(swarm.buckets) != set(panel.buckets):
        raise ValueError("reference swarm buckets differ from the panel's")
    to_panel = np.asarray([swarm.buckets.index(bucket) for bucket in panel.buckets])
    if not np.allclose(swarm.weights[:, to_panel], panel.features.weights, atol=1e-12):
        raise ValueError("reference swarm weights differ from the panel's (row order or values)")
    if not np.allclose(swarm.inventory[to_panel], panel.features.inventory, rtol=1e-9):
        raise ValueError("reference exposure scale differs from the panel's inventory")
    return swarm, to_panel


def reference_fit(swarm: ms.Swarm, target: str, powers: tuple[float, ...] | None) -> ms.ObjectiveFit:
    """The reference objective fit, optionally with the shape grid restricted to the given benefit powers."""
    objectives = ms.read_objectives(ms.DATA / "objectives.csv")
    folds = ms.final_inner_folds(swarm, pd.read_csv(ms.DATA / "splits.csv"))
    full_grid = ms.SHAPES
    if powers is not None:
        ms.SHAPES = tuple(shape for shape in full_grid if shape["power"] in powers)
    try:
        return ms.fit_objective(swarm, objectives[target], ms.read_anchors(ms.DATA / "anchors.csv", target), folds)
    finally:
        ms.SHAPES = full_grid


def reference_policy(candidate_id: str, buckets: tuple[str, ...]) -> np.ndarray:
    table = pd.read_csv(ms.DATA / "reference_policies.csv")
    rows = table[table.candidate_id == candidate_id].set_index("domain")
    if len(rows) != len(buckets):
        raise ValueError(f"reference policy {candidate_id} has {len(rows)} rows")
    return rows.loc[list(buckets), "weight"].to_numpy(float)


def regmix_topk(predict: Callable[[np.ndarray], np.ndarray], buckets: int, samples: int) -> tuple[np.ndarray, dict]:
    """RegMix's proposal: average the top-k predicted candidates among uniform-simplex samples."""
    rng = np.random.default_rng(SAMPLE_SEED)
    best_values: list[np.ndarray] = []
    best_rows: list[np.ndarray] = []
    for start in range(0, samples, SAMPLE_CHUNK):
        chunk = rng.dirichlet(np.ones(buckets), size=min(SAMPLE_CHUNK, samples - start))
        values = predict(chunk)
        keep = np.argsort(values)[:TOP_K]
        best_values.append(values[keep])
        best_rows.append(chunk[keep])
    values = np.concatenate(best_values)
    rows = np.concatenate(best_rows)
    order = np.argsort(values)[:TOP_K]
    proposal = rows[order].mean(axis=0)
    return proposal, {
        "candidate_samples": samples,
        "top_k": TOP_K,
        "best_candidate_prediction": float(values[order[0]]),
        "top_k_mean_prediction": float(values[order].mean()),
        "top_k_mean_tv_spread": float(np.mean(np.abs(rows[order] - proposal).sum(axis=1) / 2)),
    }


def nominal_cap(max_epochs: float) -> int:
    for cap in NOMINAL_CAPS:
        if max_epochs <= cap:
            return cap
    raise ValueError(f"no nominal cap holds {max_epochs} epochs")


def summarize(weights: np.ndarray, inventory: np.ndarray, natural: np.ndarray) -> dict[str, float]:
    epochs = weights * inventory
    return {
        "max_epochs": float(epochs.max()),
        "q95_epochs": float(np.quantile(epochs, 0.95)),
        "active_buckets": int((weights > 0).sum()),
        "effective_buckets": float(
            np.exp(-np.sum(np.where(weights > 0, weights * np.log(np.maximum(weights, 1e-300)), 0)))
        ),
        "tv_to_proportional": float(np.abs(weights - natural).sum() / 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--candidate-samples", type=int, default=CANDIDATE_SAMPLES)
    parser.add_argument("--comparators", default=None, help="comma-separated comparator keys; default every comparator")
    args = parser.parse_args()
    selected = (
        tuple(COMPARATORS)
        if args.comparators is None
        else tuple(next(c for c in COMPARATORS if c.key == key.strip()) for key in args.comparators.split(","))
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = bench.load_panel(PANEL)
    buckets = tuple(str(bucket) for bucket in panel.buckets)
    if set(buckets) != set(base.DOMAIN_NAMES):
        raise ValueError("panel buckets differ from the launcher's domains")
    launcher_order = [buckets.index(domain) for domain in base.DOMAIN_NAMES]
    inventory = panel.features.inventory
    natural = 1.0 / inventory
    natural = natural / natural.sum()
    inner = bench.heldout_inner_folds(panel)
    swarm, to_panel = reference_swarm_in_panel_order(panel)
    from_panel = np.argsort(to_panel)

    mariner_fits: dict[str, ms.ObjectiveFit] = {}
    mariner_policies: dict[str, np.ndarray] = {}
    for target in TARGET_LABELS:
        started = time.monotonic()
        fit = reference_fit(swarm, target, None)
        weights, _ = ms.continuous_optimum(fit, swarm.weights)
        counts, _ = ms.runtime_policy(fit, weights)
        policy = reference_policy(MARINER_POLICY[target], buckets)
        parity = float(np.abs(counts[to_panel] / ms.MIXTURE_BLOCK_SIZE - policy).sum() / 2)
        if parity > PARITY_TV:
            raise ValueError(f"reference MARINER fit does not reproduce {MARINER_POLICY[target]}: TV {parity}")
        mariner_fits[target] = fit
        mariner_policies[target] = policy
        print(
            f"{target}: MARINER reference fit reproduced {MARINER_POLICY[target]} in {time.monotonic() - started:.0f}s",
            flush=True,
        )

    rows: list[dict[str, object]] = []
    solutions = []
    summary: dict[str, dict[str, object]] = {}
    fit_records: dict[str, list[dict[str, object]]] = {}
    for target, label in TARGET_LABELS.items():
        group = panel.group(target)
        mariner = mariner_fits[target]
        mariner_predict = lambda w, fit=mariner: fit.predict(np.atleast_2d(w)[:, from_panel])  # noqa: E731
        surrogates: dict[str, Callable[[np.ndarray], np.ndarray]] = {"mariner": mariner_predict}
        proposals: dict[str, np.ndarray] = {"mariner": mariner_policies[target]}
        for comparator in selected:
            started = time.monotonic()
            candidate_key = f"cmp_{TARGET_KEYS[target]}_{comparator.key}"
            if comparator.optimizer == "reference_power1":
                fit = reference_fit(swarm, target, (1.0,))
                predict = lambda w, fit=fit: fit.predict(np.atleast_2d(w)[:, from_panel])  # noqa: E731
                objective = SurrogateObjective(inventory, predict)
                continuous, starts = ms.continuous_optimum(objective, panel.features.weights)
                counts, runtime_summary = ms.runtime_policy(objective, continuous)
                optimizer_record: dict[str, object] = {"starts": starts, **runtime_summary}
                fit_records[candidate_key] = [
                    {
                        "component": t.component,
                        "shape": t.shape,
                        "ridge": t.ridge,
                        "kappa": t.kappa,
                        "flat": t.flat_profile,
                    }
                    for t in fit.tasks
                ]
            else:
                fits = []
                for index in range(len(group.components)):
                    fits.append(fit_component(panel, comparator.model_id, target, index, inner))
                    if comparator.key == "lgbm":
                        print(f"  {candidate_key}: fitted {index + 1}/{len(group.components)}", flush=True)
                surrogate = ObservatorySurrogate(panel, comparator.model_id, target, fits)
                predict = surrogate.predict
                objective = SurrogateObjective(inventory, predict)
                if comparator.optimizer == "slsqp":
                    continuous, starts = ms.continuous_optimum(objective, panel.features.weights)
                    counts, runtime_summary = ms.runtime_policy(objective, continuous)
                    optimizer_record = {"starts": starts, **runtime_summary}
                elif comparator.optimizer == "regmix_topk":
                    continuous, sampling = regmix_topk(predict, len(buckets), args.candidate_samples)
                    counts = ms.constrained_counts(continuous, np.full(len(buckets), ms.MIXTURE_BLOCK_SIZE))
                    optimizer_record = {
                        **sampling,
                        "continuous_prediction": float(predict(continuous[None])[0]),
                        "runtime_prediction": float(predict((counts / ms.MIXTURE_BLOCK_SIZE)[None])[0]),
                    }
                else:
                    raise ValueError(comparator.optimizer)
                fit_records[candidate_key] = [
                    {
                        "component": str(component),
                        "shape": fitted.shape,
                        "ridge": fitted.ridge,
                        "floor": float(getattr(fitted.head, "floor", float("nan"))),
                        "kappa": float(getattr(fitted.head, "kappa", float("nan"))),
                        "inner_cv_rmse": float(fitted.diagnostics.get("inner_cv_rmse", float("nan"))),
                    }
                    for component, fitted in zip(group.components, fits, strict=True)
                ]
            runtime = counts / ms.MIXTURE_BLOCK_SIZE
            cap = nominal_cap(float((runtime * inventory).max()))
            candidate_id = f"{candidate_key}_cap{cap:02d}"
            surrogates[comparator.key] = predict
            proposals[comparator.key] = runtime
            summary[candidate_id] = {
                "target": target,
                "comparator": comparator.label,
                "model_id": comparator.model_id,
                "optimizer": comparator.optimizer,
                "nominal_cap_inactive": True,
                **summarize(runtime, inventory, natural),
                "tv_runtime_to_continuous": float(np.abs(runtime - continuous).sum() / 2),
                "tv_to_mariner": float(np.abs(runtime - mariner_policies[target]).sum() / 2),
                "self_prediction_runtime": float(predict(runtime[None])[0]),
                "self_prediction_at_mariner": float(predict(mariner_policies[target][None])[0]),
                "mariner_prediction_runtime": float(mariner_predict(runtime[None])[0]),
                "mariner_prediction_at_mariner": float(mariner_predict(mariner_policies[target][None])[0]),
                "optimizer_record": optimizer_record,
                "seconds": time.monotonic() - started,
            }
            solutions.append(
                pd.DataFrame(
                    {"candidate_id": candidate_id, "bucket": buckets, "continuous": continuous, "runtime": runtime}
                )
            )
            for index in launcher_order:
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "target": target,
                        "target_label": label,
                        "epoch_cap": cap,
                        "surrogate": comparator.model_id,
                        "domain": buckets[index],
                        "runtime_count": int(counts[index]),
                        "weight": float(runtime[index]),
                        "materialized_epochs": float(runtime[index] * inventory[index]),
                    }
                )
            print(
                candidate_id,
                json.dumps(
                    {k: v for k, v in summary[candidate_id].items() if k != "optimizer_record"},
                    indent=None,
                    default=float,
                ),
                flush=True,
            )
        # Every surrogate's prediction at every proposal, for the cross-prediction table.
        names = list(proposals)
        matrix = pd.DataFrame(
            {predictor: [float(surrogates[predictor](proposals[p][None])[0]) for p in names] for predictor in names},
            index=pd.Index(names, name="proposal"),
        )
        matrix.to_csv(args.output_dir / f"cross_predictions_{target}.csv")
        tv = pd.DataFrame(
            {a: [float(np.abs(proposals[a] - proposals[b]).sum() / 2) for b in names] for a in names},
            index=pd.Index(names, name="proposal"),
        )
        tv.to_csv(args.output_dir / f"tv_between_proposals_{target}.csv")

    with (args.output_dir / "candidate_weights.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pd.concat(solutions).to_csv(args.output_dir / "solutions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (args.output_dir / "fits.json").write_text(json.dumps(fit_records, indent=1, default=float))
    manifest: dict[str, Any] = {
        "panel": PANEL,
        "reference_package": str(REFERENCE_PACKAGE),
        "candidate_samples": args.candidate_samples,
        "top_k": TOP_K,
        "sample_seed": SAMPLE_SEED,
        "start_seed": ms.START_SEED,
        "panel_starts": ms.PANEL_STARTS,
        "block_size": ms.MIXTURE_BLOCK_SIZE,
        "mariner_policies": MARINER_POLICY,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
