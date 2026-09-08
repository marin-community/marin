# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Delphi 3e18 mixture proposals from a panel-fitted separable surrogate, with epoch caps and box trust regions.

The surrogate is fitted per component on the canonical 280-run panel exactly as the heldout stage does. For an
identity-link successor the aggregate prediction is a sum of per-bucket cost curves, so the constrained optimum on
the exact 1/2048 mixture grid is found by min-plus dynamic programming with per-bucket count bounds: an epoch cap
(count <= cap / inventory) and, optionally, a box of half-width ``box`` around an anchor mixture (a bank
coordinate, or the measured-best bank coordinate for the target). Nothing is launched; the output is a table of
proposals with their predicted values, distances, and nearest measured bank neighbours.
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
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_registry_20260902 as registry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_heldout_selection_20260903 as selection,
)

PANEL = "delphi_3e18_39bucket"
BLOCKS = 2_048
NO_CAP = 10_000.0


@dataclasses.dataclass(frozen=True)
class ComponentCurve:
    weight: float
    intercept: float
    benefit: np.ndarray  # per-bucket amplitude
    harm: np.ndarray
    shape: dict[str, float]


def fit_curve(model_id: str, target: str, component_index: int, fixed_shape: dict | None) -> ComponentCurve:
    panel = harness.load_panel(PANEL)
    entry = registry.ENTRY_BY_ID[model_id]
    group = panel.group(target)
    component = group.components[component_index]
    features = dataclasses.replace(registry.apply_transform(panel.features, entry), component=str(component))
    model = entry.build(features)
    if not isinstance(model, models.GridModel):
        raise TypeError(f"{model_id} is not a grid model")
    response = group.outcomes[:, component_index]
    rows = np.arange(panel.rows)
    if fixed_shape is None:
        fitted = model.fit(
            features,
            response,
            rows,
            harness.heldout_inner_folds(panel),
            harness._seed(harness.FitTask(model_id, PANEL, target, component_index, component, 0, 0)),
        )
        shape, ridge, spec = (
            fitted.shape,
            fitted.ridge,
            model.head_for(fitted.shape, models.LinkKind(str(fitted.diagnostics["link"]))),
        )
        head = fitted.head
    else:
        shape = {key: float(value) for key, value in fixed_shape.items() if key != "ridge"}
        ridge = float(fixed_shape.get("ridge", model.ridge_grid[0]))
        spec = model.head_for(shape)
        head = models.fit_head(model.design(features, shape), response, ridge, spec)
    if spec.link is not models.LinkKind.IDENTITY:
        raise ValueError("the dynamic programme needs an identity link")
    design = model.design(features, shape)
    buckets = features.buckets
    expected = tuple(f"bucket_signal:{index}" for index in range(buckets)) + tuple(
        f"bucket_overexposure:{index}" for index in range(buckets)
    )
    if design.names != expected:
        raise ValueError("design is not the per-bucket benefit + harm layout")
    return ComponentCurve(
        float(group.aggregation_weights[component_index]),
        head.intercept,
        head.coefficients[:buckets].copy(),
        head.coefficients[buckets:].copy(),
        dict(shape),
    )


def bucket_cost(curves: tuple[ComponentCurve, ...], bucket: int, exposures: np.ndarray) -> np.ndarray:
    total = np.zeros_like(exposures, dtype=float)
    for curve in curves:
        benefit = models.weibull_response(exposures, curve.shape["rate"], curve.shape["power"])
        harm = models.softplus_harm(exposures, curve.shape["threshold"])
        total += curve.weight * (-curve.benefit[bucket] * benefit + curve.harm[bucket] * harm)
    return total


def predict(curves: tuple[ComponentCurve, ...], exposures: np.ndarray) -> np.ndarray:
    values = np.atleast_2d(exposures)
    result = np.full(values.shape[0], sum(curve.weight * curve.intercept for curve in curves))
    for bucket in range(values.shape[1]):
        result += bucket_cost(curves, bucket, values[:, bucket])
    return result


def solve(curves: tuple[ComponentCurve, ...], inventory: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Min-plus dynamic programme over integer block counts with per-bucket bounds."""
    if lower.sum() > BLOCKS or upper.sum() < BLOCKS:
        raise ValueError("infeasible bounds")
    best = np.full(BLOCKS + 1, np.inf)
    best[0] = 0.0
    choices = np.full((len(inventory), BLOCKS + 1), -1, dtype=np.int32)
    for bucket in range(len(inventory)):
        counts = np.arange(int(lower[bucket]), int(upper[bucket]) + 1)
        costs = bucket_cost(curves, bucket, inventory[bucket] * counts / BLOCKS)
        updated = np.full(BLOCKS + 1, np.inf)
        selected = np.full(BLOCKS + 1, -1, dtype=np.int32)
        for count, cost in zip(counts, costs, strict=True):
            candidate = best[: BLOCKS + 1 - count] + cost
            target = updated[count:]
            better = candidate < target
            target[better] = candidate[better]
            selected[count:][better] = count
        best, choices[bucket] = updated, selected
    if not np.isfinite(best[BLOCKS]):
        raise RuntimeError("no feasible allocation")
    result = np.zeros(len(inventory), dtype=int)
    remaining = BLOCKS
    for bucket in range(len(inventory) - 1, -1, -1):
        count = int(choices[bucket, remaining])
        result[bucket] = count
        remaining -= count
    if remaining != 0:
        raise RuntimeError("broken backpointers")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-dir", type=Path, required=True, help="heldout registry directory (use the corrected view)"
    )
    parser.add_argument("--output-dir", type=Path, default=harness.DEFAULT_OUTPUT_DIR / "heldout_round3_corrected")
    parser.add_argument("--model", default="weibull_softplus_unscaled")
    parser.add_argument("--targets", default="uncheatable,table9")
    parser.add_argument("--caps", default="4,6,8,10,12,16,none")
    parser.add_argument("--boxes", default="none,0.02,0.05", help="box half-widths around the anchor, in weight")
    parser.add_argument(
        "--anchor", default="frontier", help="'frontier' (measured-best bank coordinate) or a coordinate id suffix"
    )
    parser.add_argument(
        "--fixed-shape", default=None, help="JSON shape (rate, power, threshold, ridge) applied to every component"
    )
    parser.add_argument("--tag", default="proposals")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    harness.HELDOUT_DIR = args.registry_dir.resolve()
    panel = harness.load_panel(PANEL)
    inventory = panel.features.inventory
    fixed = json.loads(args.fixed_shape) if args.fixed_shape else None
    rows = []
    mixtures = {}
    for target in [token.strip() for token in args.targets.split(",") if token.strip()]:
        group = panel.group(target)
        with harness.parallel_config(backend="loky", inner_max_num_threads=1):
            curves = tuple(
                Parallel(n_jobs=args.workers)(
                    delayed(fit_curve)(args.model, target, index, fixed) for index in range(len(group.components))
                )
            )
        bank = selection.load_bank(panel, target)
        _frame, bank_features = harness.heldout_features(panel, target)
        bank_prediction = predict(curves, bank_features.exposures)
        if args.anchor == "frontier":
            anchor_index = int(np.argmin(bank.measured))
        else:
            anchor_index = int(np.flatnonzero([cid.endswith(args.anchor) for cid in bank.coordinate_id])[0])
        anchor = bank_features.weights[anchor_index]
        for cap_token in args.caps.split(","):
            cap = NO_CAP if cap_token.strip() == "none" else float(cap_token)
            for box_token in args.boxes.split(","):
                box = None if box_token.strip() == "none" else float(box_token)
                upper = np.floor(np.minimum(1.0, cap / inventory) * BLOCKS + 1e-9)
                lower = np.zeros(len(inventory))
                if box is not None:
                    upper = np.minimum(upper, np.floor((anchor + box) * BLOCKS + 1e-9))
                    lower = np.maximum(lower, np.ceil((anchor - box) * BLOCKS - 1e-9))
                try:
                    counts = solve(curves, inventory, lower, upper)
                except (ValueError, RuntimeError) as error:
                    rows.append({"target": target, "cap": cap_token, "box": box_token, "status": str(error)})
                    continue
                weights = counts / BLOCKS
                exposures = weights * inventory
                value = float(predict(curves, exposures[None, :])[0])
                nearest = int(np.argmin(np.abs(bank_features.weights - weights[None, :]).sum(axis=1)))
                key = f"{target}|cap={cap_token}|box={box_token}"
                mixtures[key] = {bucket: float(weight) for bucket, weight in zip(panel.buckets, weights, strict=True)}
                rows.append(
                    {
                        "target": target,
                        "cap": cap_token,
                        "box": box_token,
                        "status": "ok",
                        "predicted": value,
                        "predicted_at_anchor": float(bank_prediction[anchor_index]),
                        "anchor_measured": float(bank.measured[anchor_index]),
                        "predicted_rank_in_bank": int((bank_prediction < value).sum()) + 1,
                        "l1_to_anchor": float(np.abs(weights - anchor).sum()),
                        "l1_to_panel": float(np.abs(panel.features.weights - weights[None, :]).sum(axis=1).min()),
                        "max_epochs": float(exposures.max()),
                        "buckets_over_4_epochs": int((exposures > 4).sum()),
                        "effective_buckets": float(np.exp(-(weights[weights > 0] * np.log(weights[weights > 0])).sum())),
                        "nearest_bank_l1": float(np.abs(bank_features.weights[nearest] - weights).sum()),
                        "nearest_bank_measured": float(bank.measured[nearest]),
                        "nearest_bank_predicted": float(bank_prediction[nearest]),
                        "top_buckets": ", ".join(
                            f"{panel.buckets[index]}={weights[index]:.3f}" for index in np.argsort(-weights)[:6]
                        ),
                    }
                )
    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.tag}_{args.model.replace('@', '_')}"
    table.to_csv(args.output_dir / f"{stem}.csv", index=False)
    (args.output_dir / f"{stem}_mixtures.json").write_text(json.dumps(mixtures, indent=1, sort_keys=True) + "\n")
    pd.set_option("display.width", 250)
    columns = [
        "target",
        "cap",
        "box",
        "status",
        "predicted",
        "predicted_at_anchor",
        "anchor_measured",
        "predicted_rank_in_bank",
        "l1_to_anchor",
        "l1_to_panel",
        "max_epochs",
        "effective_buckets",
        "nearest_bank_l1",
        "nearest_bank_measured",
        "top_buckets",
    ]
    print(table.loc[:, [column for column in columns if column in table.columns]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
