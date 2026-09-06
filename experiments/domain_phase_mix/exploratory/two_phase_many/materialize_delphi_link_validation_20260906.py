# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the bounded log-deficit link's optima for 3e18 validation, matched to the coupling validation.

The surrogate is WSPU with the bounded log-deficit link (`weibull_softplus_unscaled@log_deficit_bounded_link`)
fitted on the canonical 280 rows in `delphi_link_selection_20260906` (fold -1 of the frozen selection
benchmark). Its per-component heads are reconstructed from the saved shapes and ridges, checked against the
saved predictions, and minimized under the epoch caps of the coupling validation (Uncheatable cap 6, Table 9
caps 6 and 8, KL 0) from the same five starts. Continuous optima are rounded to the runtime grid and refined
by the existing one-count exchange on the link predictor, and the launcher table is written in the schema of
`launch_delphi_wspu_coupling_validation_3e18`. Nothing here launches a job.

usage: uv run python materialize_delphi_link_validation_20260906.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
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
    materialize_delphi_one_phase_surrogate_challengers_20260831 as grid,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    optimize_delphi_matched_policies_20260906 as policy,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_registry_20260902 as registry,
)

REFERENCE = SCRIPT_DIR / "reference_outputs"
FROZEN = benchmark.DEFAULT_OUTPUT
SELECTION = REFERENCE / "delphi_link_selection_20260906"
FOLLOWUP = REFERENCE / "delphi_coupling_followup_20260906"
OLD_CANDIDATES = REFERENCE / "delphi_one_phase_weibull_softplus_epoch_cap_sweep_20260902" / "candidate_weights.csv"
OLD_MEASURED = REFERENCE / "delphi_one_phase_weibull_softplus_epoch_cap_sweep_20260902" / "measured_results.csv"
DEFAULT_OUTPUT = REFERENCE / "delphi_link_validation_3e18_20260906"
WSPU = "weibull_softplus_unscaled"
LINK = "weibull_softplus_unscaled@log_deficit_bounded_link"
CASES = (("uncheatable", 6), ("table9", 6), ("table9", 8))
TARGET_TAGS = {"uncheatable": "u", "table9": "t9"}
TARGET_LABELS = {"uncheatable": "Uncheatable", "table9": "Table-9 macro"}
BUCKETS = 39
BLOCK_SIZE = grid.MIXTURE_BLOCK_SIZE
PARITY_TOLERANCE = 1e-8


@dataclasses.dataclass(frozen=True)
class LinkSurrogate:
    """A grid model's per-component heads in closed form over its named design columns.

    Supported columns: ``bucket_signal:i`` (minus the Weibull benefit of bucket i), ``bucket_overexposure:i``
    (the softplus harm), and ``interaction:hub_plus:i`` / ``interaction:hub_minus:i`` (plus or minus the total
    benefit times bucket i's benefit). Components with a log-deficit link exponentiate the linear predictor above
    their floor, capped at their fitted cap; identity-link components use it directly.
    """

    model_id: str
    target: str
    inventory: np.ndarray
    aggregation: np.ndarray
    names: tuple[str, ...]
    intercept: np.ndarray
    coefficients: np.ndarray
    rate: np.ndarray
    power: np.ndarray
    threshold: np.ndarray
    floor: np.ndarray
    cap: np.ndarray
    identity: np.ndarray

    def columns(self, weights: np.ndarray) -> np.ndarray:
        """Design columns for every query and component, shape (queries, components, columns)."""
        exposure = np.atleast_2d(weights)[:, None, :] * self.inventory[None, None, :]
        benefit = -np.expm1(-((self.rate[None, :, None] * np.maximum(exposure, 0)) ** self.power[None, :, None]))
        harm = np.logaddexp(0.0, np.log1p(np.maximum(exposure, 0)) - self.threshold[None, :, None]) ** 2
        hub = benefit.sum(axis=2, keepdims=True)
        blocks = []
        for name in self.names:
            kind, _, index = name.rpartition(":")
            column = int(index)
            if kind == "bucket_signal":
                blocks.append(-benefit[:, :, column])
            elif kind == "bucket_overexposure":
                blocks.append(harm[:, :, column])
            elif kind == "interaction:hub_plus":
                blocks.append(hub[:, :, 0] * benefit[:, :, column])
            elif kind == "interaction:hub_minus":
                blocks.append(-hub[:, :, 0] * benefit[:, :, column])
            else:
                raise ValueError(f"Unsupported design column {name}")
        return np.stack(blocks, axis=2)

    def linear(self, weights: np.ndarray) -> np.ndarray:
        return self.intercept[None, :] + np.einsum("nck,ck->nc", self.columns(weights), self.coefficients)

    def atomic(self, weights: np.ndarray) -> np.ndarray:
        linear = self.linear(weights)
        clipped = np.clip(linear, -models.LINK_CLIP, np.minimum(models.LINK_CLIP, self.cap[None, :]))
        linked = self.floor[None, :] + np.exp(clipped)
        return np.where(self.identity[None, :], linear, linked)

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return self.atomic(weights) @ self.aggregation


def candidate_id(target: str, cap: int, tag: str) -> str:
    return f"lwspu_{TARGET_TAGS[target]}_{tag}_cap{cap:02d}"


def reconstruct(model_id: str, target: str, selection: Path = SELECTION) -> LinkSurrogate:
    """Rebuild the fold -1 heads of ``model_id`` from a selection benchmark's shards and check parity."""
    data = benchmark.read_npz(FROZEN / "inputs" / "panel.npz")
    bank = benchmark.read_npz(FROZEN / "inputs" / f"{target}_bank_features.npz")
    feature = benchmark.feature_set(data, benchmark.PANEL, data["weights"], data["exposures"])
    entry = registry.ENTRY_BY_ID[model_id]
    feature = registry.apply_transform(feature, entry)
    components = data[f"{target}_components"]
    outcomes = data[f"{target}_outcomes"]
    intercepts, coefficients, rates, powers, thresholds, floors, caps, identity = [], [], [], [], [], [], [], []
    train_predictions, bank_predictions = [], []
    names: tuple[str, ...] | None = None
    for index, name in enumerate(components):
        shard = benchmark.read_npz(selection / "baseline_shards" / model_id / target / f"r0_f-1_c{index}.npz")
        train = shard["train"]
        if len(train) != data["weights"].shape[0]:
            raise ValueError("The final fit must use every canonical row")
        model = entry.build(dataclasses.replace(feature, component=str(name)))
        if not isinstance(model, models.GridModel):
            raise ValueError("Expected a grid model")
        shape = json.loads(str(shard["shape_json"]))
        design = model.design(feature, shape)
        if names is None:
            names = tuple(design.names)
        elif tuple(design.names) != names:
            raise ValueError("Components of one model must share their design columns")
        spec = model.head_for(shape)
        head = models.fit_head(
            models.Design(design.values[train], design.ridge, design.names),
            outcomes[train, index],
            float(shard["ridge"]),
            spec,
        )
        intercepts.append(head.intercept)
        coefficients.append(head.coefficients)
        rates.append(float(shape["rate"]))
        powers.append(float(shape["power"]))
        thresholds.append(float(shape["threshold"]))
        is_identity = spec.link is models.LinkKind.IDENTITY
        identity.append(is_identity)
        floors.append(0.0 if is_identity else float(head.floor))
        caps.append(float("inf") if is_identity else float(head.cap))
        train_predictions.append(shard["train_prediction"])
        bank_predictions.append(shard["bank_prediction"])
    if names is None:
        raise ValueError("No components")
    surrogate = LinkSurrogate(
        model_id,
        target,
        data["inventory"],
        data[f"{target}_aggregation_weights"],
        names,
        np.array(intercepts),
        np.stack(coefficients),
        np.array(rates),
        np.array(powers),
        np.array(thresholds),
        np.array(floors),
        np.array(caps),
        np.array(identity),
    )
    train_error = float(np.max(np.abs(surrogate.atomic(data["weights"]) - np.column_stack(train_predictions))))
    bank_error = float(np.max(np.abs(surrogate.atomic(bank["weights"]) - np.column_stack(bank_predictions))))
    if max(train_error, bank_error) > PARITY_TOLERANCE:
        raise ValueError(f"{model_id}/{target} reconstruction parity failed: {train_error:.2e} / {bank_error:.2e}")
    print(f"reconstructed {model_id}/{target}: parity {train_error:.1e} (panel) {bank_error:.1e} (bank)", flush=True)
    return surrogate


def optimize(surrogate: LinkSurrogate, cap: int, data: dict) -> tuple[np.ndarray, list[dict]]:
    natural = 1 / data["inventory"]
    natural = natural / natural.sum()
    indices = np.random.default_rng(policy.START_SEED).choice(len(data["weights"]), policy.PANEL_STARTS, replace=False)
    upper = np.minimum(1.0, cap / data["inventory"])
    starts = [policy.project_start(row, upper, natural) for row in (natural, *data["weights"][indices])]
    records, solutions = [], []
    for index, start in enumerate(starts):
        weights, diagnostics = policy.optimize_start(surrogate.predict, start, natural, upper, 0.0)
        records.append({"start": index, **diagnostics})
        solutions.append(weights)
    # Every endpoint is projected to the feasible set by optimize_start, so the proposal is the lowest objective
    # over all starts after a polish pass with a coarser finite-difference step (SLSQP's 1e-8 step reports
    # "positive directional derivative" on the hub objective without certifying the stationary point).
    saved = policy.OPTIMIZER_OPTIONS
    policy.OPTIMIZER_OPTIONS = {"maxiter": 2000, "ftol": 1e-10, "eps": 1e-6}
    try:
        for index, start in enumerate(list(solutions)):
            weights, diagnostics = policy.optimize_start(surrogate.predict, start, natural, upper, 0.0)
            polished = diagnostics["objective"] <= records[index]["objective"]
            if polished:
                solutions[index] = weights
                records[index] = {"start": index, **diagnostics}
            records[index]["polished"] = polished
    finally:
        policy.OPTIMIZER_OPTIONS = saved
    feasible = [
        i for i, row in enumerate(records) if row["raw_feasibility_violation"] <= policy.FEASIBILITY_TOLERANCE * 10
    ]
    if not feasible:
        raise ValueError("No restart ended feasibly")
    converged = feasible
    spread = max(records[i]["objective"] for i in feasible) - min(records[i]["objective"] for i in feasible)
    for row in records:
        row["endpoint_objective_spread"] = spread
    best = min(converged, key=lambda i: records[i]["objective"])
    for row in records:
        row["selected"] = row["start"] == best
    return solutions[best], records


def runtime(surrogate: LinkSurrogate, weights: np.ndarray, inventory: np.ndarray, cap: int) -> tuple[np.ndarray, dict]:
    maximum = np.floor(np.minimum(1.0, cap / inventory) * BLOCK_SIZE + 1e-12).astype(np.int64)
    predict = surrogate.predict
    initial = grid.prefix_materializer.constrained_counts(weights, maximum)
    counts, steps = grid.refine_runtime_counts(predict, initial, maximum)
    runtime_weights = counts / BLOCK_SIZE
    if int(counts.sum()) != BLOCK_SIZE or counts.min() < 0 or np.any(counts > maximum):
        raise ValueError("Invalid refined runtime allocation")
    if not np.array_equal(grid.prefix_materializer.runtime_counts(runtime_weights), counts):
        raise ValueError("Runtime allocation does not survive the realizer")
    continuous_prediction = float(predict(weights[None])[0])
    initial_prediction = float(predict((initial / BLOCK_SIZE)[None])[0])
    runtime_prediction = float(predict(runtime_weights[None])[0])
    if runtime_prediction > initial_prediction + grid.REFINE_TOLERANCE:
        raise ValueError("Runtime refinement increased its objective")
    return counts, {
        "continuous_prediction": continuous_prediction,
        "initial_grid_prediction": initial_prediction,
        "runtime_prediction": runtime_prediction,
        "runtime_minus_continuous_prediction": runtime_prediction - continuous_prediction,
        "continuous_to_runtime_tv": float(np.abs(runtime_weights - weights).sum() / 2),
        "exchange_steps": int(steps),
        "max_materialized_epoch": float((runtime_weights * inventory).max()),
        "q95_materialized_epoch": float(np.sort(runtime_weights * inventory)[round(0.95 * (len(inventory) - 1))]),
    }


def comparators(buckets: tuple[str, ...]) -> dict[tuple[str, int], np.ndarray]:
    """The frozen kappa-0 WSPU policies of the coupling follow-up, by (target, cap)."""
    table = pd.read_csv(FOLLOWUP / "continuous_policies" / "selected_policies.csv")
    result = {}
    for target, cap in CASES:
        rows = table[(table.model == WSPU) & (table.target == target) & (table.cap == cap) & (table.kl_coefficient == 0)]
        if len(rows) != 1:
            raise ValueError(f"Missing kappa-0 comparator for {target}/{cap}")
        result[target, cap] = np.asarray([rows.iloc[0][f"weight::{bucket}"] for bucket in buckets], dtype=float)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=LINK, help="registry id whose fold -1 shards exist in --selection-dir")
    parser.add_argument("--selection-dir", type=Path, default=SELECTION)
    parser.add_argument("--cases", default="uncheatable:6,table9:6,table9:8", help="target:cap pairs")
    parser.add_argument("--tag", default="bl", help="candidate id tag, lwspu_<target>_<tag>_cap<cap>")
    args = parser.parse_args()
    cases = tuple((item.split(":")[0], int(item.split(":")[1])) for item in args.cases.split(","))
    if args.model not in registry.ENTRY_BY_ID:
        raise ValueError(f"Unknown registry entry: {args.model}")
    offline = args.output_dir / "offline_materialization"
    runtime_dir = args.output_dir / "runtime_materialization"
    offline.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    data = benchmark.read_npz(FROZEN / "inputs" / "panel.npz")
    buckets = tuple(map(str, data["buckets"]))
    inventory = data["inventory"]
    targets = tuple(dict.fromkeys(target for target, _ in cases))
    links = {target: reconstruct(args.model, target, args.selection_dir) for target in targets}
    wspus = {target: reconstruct(WSPU, target, args.selection_dir) for target in targets}
    controls = comparators(buckets)
    old = pd.read_csv(OLD_CANDIDATES)
    measured = pd.read_csv(OLD_MEASURED) if OLD_MEASURED.exists() else None
    policies, restarts, weight_rows, candidate_rows, mapping = [], [], [], [], []
    for target, cap in cases:
        link, wspu = links[target], wspus[target]
        weights, records = optimize(link, cap, data)
        for row in records:
            restarts.append({"target": target, "cap": cap, **row})
        counts, diagnostics = runtime(link, weights, inventory, cap)
        runtime_weights = counts / BLOCK_SIZE
        control = controls[target, cap]
        old_rows = old[(old.target == target) & (old.epoch_cap == cap)].set_index("domain")
        old_weights = old_rows.loc[list(buckets), "weight"].to_numpy(float) if len(old_rows) else None
        name = candidate_id(target, cap, args.tag)
        summary = {
            "candidate_id": name,
            "target": target,
            "epoch_cap": cap,
            "surrogate": args.model,
            "kl_coefficient": 0.0,
            "target_flops": 3e18,
            "measurement_status": "unknown_not_run",
            **diagnostics,
            "wspu_prediction_at_link_optimum": float(wspu.predict(runtime_weights[None])[0]),
            "link_prediction_at_wspu_kappa0_policy": float(link.predict(control[None])[0]),
            "wspu_prediction_at_wspu_kappa0_policy": float(wspu.predict(control[None])[0]),
            "tv_to_wspu_kappa0_policy": float(np.abs(runtime_weights - control).sum() / 2),
            "tv_to_measured_wspu_candidate": (
                None if old_weights is None else float(np.abs(runtime_weights - old_weights).sum() / 2)
            ),
            "measured_wspu_candidate_id": None if not len(old_rows) else str(old_rows.candidate_id.iloc[0]),
            **{f"link_{k}": v for k, v in policy.support_distances(data["weights"], runtime_weights).items()},
            **{f"wspu_kappa0_{k}": v for k, v in policy.support_distances(data["weights"], control).items()},
            "effective_buckets": float(1 / np.square(runtime_weights).sum()),
            "wspu_kappa0_effective_buckets": float(1 / np.square(control).sum()),
        }
        if measured is not None and len(old_rows):
            hit = measured[measured.candidate_id.eq(summary["measured_wspu_candidate_id"])]
            if len(hit):
                summary["measured_wspu_candidate_row"] = hit.iloc[0].to_dict()
        policies.append(
            {
                **summary,
                "weights": {b: float(w) for b, w in zip(buckets, weights, strict=True)},
                "runtime_counts": {b: int(c) for b, c in zip(buckets, counts, strict=True)},
            }
        )
        mapping.append(summary)
        for bucket, scale, count in zip(buckets, inventory, counts, strict=True):
            weight_rows.append(
                {"policy_id": name, "bucket": bucket, "continuous_weight": float(weights[list(buckets).index(bucket)])}
            )
            candidate_rows.append(
                {
                    "candidate_id": name,
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "epoch_cap": cap,
                    "surrogate": args.model,
                    "domain": bucket,
                    "runtime_count": int(count),
                    "weight": float(count / BLOCK_SIZE),
                    "materialized_epochs": float(scale * count / BLOCK_SIZE),
                }
            )
    coordinates = [tuple(row["runtime_counts"].values()) for row in policies]
    if len(set(coordinates)) != len(cases):
        raise ValueError("Policies alias on the runtime grid")
    pd.DataFrame(restarts).to_csv(offline / "restart_diagnostics.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(offline / "weights.csv", index=False)
    (offline / "policies.json").write_text(json.dumps(policies, indent=2, sort_keys=True, default=str))
    pd.DataFrame(candidate_rows).to_csv(runtime_dir / "candidate_weights.csv", index=False)
    pd.DataFrame([{k: v for k, v in row.items() if k != "measured_wspu_candidate_row"} for row in mapping]).to_csv(
        runtime_dir / "candidate_mapping.csv", index=False
    )
    digest = benchmark.sha256(runtime_dir / "candidate_weights.csv")
    (runtime_dir / "summary.json").write_text(
        json.dumps(
            {
                "candidate_ids": [row["candidate_id"] for row in mapping],
                "candidate_weights_sha256": digest,
                "surrogate": args.model,
                "selection_benchmark_dir": str(args.selection_dir.resolve().relative_to(REPO_ROOT)),
                "frozen_inputs": {
                    name: benchmark.sha256(FROZEN / "inputs" / name)
                    for name in ("panel.npz", "uncheatable_bank_features.npz", "table9_bank_features.npz")
                },
                "sources": {
                    str(p.relative_to(REPO_ROOT)): benchmark.sha256(p)
                    for p in (
                        Path(__file__).resolve(),
                        Path(policy.__file__),
                        Path(grid.__file__),
                        Path(models.__file__),
                        Path(registry.__file__),
                    )
                },
                "no_jobs_launched": True,
            },
            indent=2,
        )
    )
    pd.set_option("display.width", 250)
    show = pd.DataFrame(mapping)
    print(
        show[
            [
                "candidate_id",
                "continuous_prediction",
                "runtime_prediction",
                "wspu_prediction_at_link_optimum",
                "link_prediction_at_wspu_kappa0_policy",
                "wspu_prediction_at_wspu_kappa0_policy",
                "tv_to_wspu_kappa0_policy",
                "tv_to_measured_wspu_candidate",
                "link_hull_distance_tv",
                "link_nearest_panel_tv",
                "effective_buckets",
                "wspu_kappa0_effective_buckets",
                "max_materialized_epoch",
                "exchange_steps",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("candidate_weights.csv sha256:", digest)


if __name__ == "__main__":
    main()
