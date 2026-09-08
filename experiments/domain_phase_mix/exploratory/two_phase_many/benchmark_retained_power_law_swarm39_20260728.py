# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""Does the retained-power-law surrogate hold up away from StarCoder?

The model was designed against a two-bucket panel where a two-phase policy beats the whole one-phase
class. That is exactly the situation the 39-bucket panels do not present, so the risk is that a form
tuned to express phase order buys its WSD80 accuracy with parameters that are unidentifiable at 39
buckets and simply add variance there. This benchmark runs it through the existing swarm39 harness,
against the incumbents already registered on that track, under the same grouped out-of-fold protocol,
the same ridge grid and the same targets.

Nothing about the model changes between panels. The shape grid, the amplitude bounds and the three
terms are identical; only the panel geometry differs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# The swarm39 modules import each other by bare name, so they must be imported the same way here.
# Importing the harness by package path as well would create a second copy of Design and Model, and the
# zoo's models would then be instances of classes this file's fit_model does not recognise.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import retained_power_law_model_20260728 as retained  # noqa: E402
import swarm39_harness_20260725 as harness  # noqa: E402
import swarm39_models_20260725 as zoo  # noqa: E402

DEFAULT_OUTPUT_DIR = harness.REFERENCE_OUTPUTS / "retained_power_law_swarm39_20260728"
SCALES = ("60m", "delphi_3e18")
TARGETS = (harness.UNCHEATABLE, harness.TABLE9)
# The swarm39 harness selects shape and ridge jointly, so the model's own ridge grid is handed over
# and only the shape parameters are enumerated here.
MODEL_NAME = "retained_power_law"


def geometry_of(panel: harness.Panel) -> retained.Geometry:
    """Panel geometry, with the phase fraction derived from the epoch multipliers rather than taken
    from ``panel.alpha``.

    The two disagree on the 60M panel, where ``alpha`` is the nominal 0.80 but ``c0``/``c1`` imply
    0.7981377. The model's within-window intensities are ``c0*w0/alpha0`` and ``c1*w1/alpha1``, so they
    are only equal at a tied policy when the fraction matches the multipliers. Feeding the nominal
    value produced a concentration gap of 2.1 at tied policies instead of zero, silently contaminating
    both 60M cells. Deriving the fraction from the multipliers makes the invariant hold by
    construction on every panel.
    """
    implied = float(np.median(panel.c0 / (panel.c0 + panel.c1)))
    return retained.Geometry(c0=panel.c0, c1=panel.c1, phase_0_fraction=implied, family_index=panel.family_index)


def weights_of(panel: harness.Panel) -> np.ndarray:
    return np.stack([panel.phase0, panel.phase1], axis=1)


def _shape_of(shape: dict) -> retained.Shape:
    return retained.Shape(
        benefit_exponent=shape["benefit_exponent"],
        benefit_offset=shape["benefit_offset"],
        damage_exponent=shape["damage_exponent"],
        damage_threshold=shape["damage_threshold"],
        retention=shape["retention"],
        late_multiplier=shape["late_multiplier"],
        ordering_channel=shape["ordering_channel"],
    )


def build_design(panel: harness.Panel, shape: dict) -> harness.Design:
    """Adapter to the swarm39 harness's design-matrix contract."""
    parameters = _shape_of(shape)
    geometry = geometry_of(panel)
    matrix = retained.design_matrix(weights_of(panel), geometry, parameters)
    families = [panel.family_names[index] for index in np.unique(geometry.families)]
    excess = [panel.buckets[index] for index in geometry.excess_domains]
    block = [f"family:{name}" for name in families] + [f"bucket:{name}" for name in excess]
    ordering = (
        [f"ordering_benefit_late:{name}" for name in families]
        + [f"ordering_benefit_early:{name}" for name in families]
        + [f"ordering_damage_late:{name}" for name in families]
        + [f"ordering_damage_early:{name}" for name in families]
        + ["asymmetry_up", "asymmetry_down"]
        if parameters.ordering_channel
        else []
    )
    names = tuple(
        [f"benefit:{n}" for n in block]
        + [f"damage:{n}" for n in block]
        + ["concentration_up", "concentration_down"]
        + ordering
    )
    assert len(names) == matrix.shape[1], (
        f"design has {matrix.shape[1]} columns but {len(names)} names; " "the adapter and the model have drifted apart"
    )
    return harness.Design(matrix=matrix, names=names)


def penalty_scale(panel: harness.Panel, shape: dict) -> np.ndarray:
    """Free family amplitudes; penalised bucket departures and ordering columns.

    The multiplier vector depends on the shape, because the ordering columns are only present for
    shapes that enable them.
    """
    return retained.penalty_multipliers(geometry_of(panel), _shape_of(shape))


def shapes() -> list[dict]:
    return [
        {
            "benefit_exponent": shape.benefit_exponent,
            "benefit_offset": shape.benefit_offset,
            "damage_exponent": shape.damage_exponent,
            "damage_threshold": shape.damage_threshold,
            "retention": shape.retention,
            "late_multiplier": shape.late_multiplier,
            "ordering_channel": shape.ordering_channel,
        }
        for shape in retained.shape_grid()
    ]


def retained_model() -> harness.Model:
    return harness.Model(
        name=MODEL_NAME,
        build=build_design,
        shapes=shapes,
        l2_grid=retained.RIDGE_GRID,
        penalty_scale=penalty_scale,
        # Without this the harness fits its own nonnegative least-squares head and the robust
        # estimator that defines this model is never used, so the swarm scores would measure a
        # different model than the StarCoder scores do.
        head=retained.solve_head,
    )


def nested_out_of_fold(
    panel: harness.Panel,
    build_model: Callable[[harness.Panel], harness.Model],
    target: str,
    n_splits: int,
    seed: int,
    outer_workers: int,
) -> tuple[float, float]:
    """Out-of-fold RMSE with shape and ridge selected inside each outer fold.

    ``Fit.oof_rmse`` is the *minimum* score over the shape and ridge grid on the folds used to make
    that choice, so it is a selection score rather than an estimate of error on new data, and it is
    optimistic in proportion to how many combinations were searched. This model searches 1620;
    several incumbents search a handful. Ranking on that quantity would reward grid size.

    Here the selection is repeated from scratch on each outer training split and scored on the outer
    test split it never saw, which costs a factor of ``n_splits`` in runtime and is the only version of
    the number that is comparable across models with different grid sizes.

    The model is rebuilt from each outer training split rather than passed in ready-made. Several
    incumbents derive their shape grid from panel exposures, so a model constructed once from the whole
    fit panel carries outer-test covariates into every fold; for ``separate_heads`` that alone moved the
    60M uncheatable figure from 0.20 to 0.016.
    """
    observed = panel.targets[target]
    panel = panel.subset(np.isfinite(observed))
    observed = panel.targets[target]
    splits = harness.grouped_splits(panel, n_splits, seed)

    def fit_outer_fold(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        training = panel.subset(train)
        fold_model = build_model(training)
        inner_fit = harness.fit_model(
            training, fold_model, target, n_splits=n_splits, seed=seed, split_fn=harness.grouped_splits
        )
        return test, inner_fit.predict(panel.subset(test), fold_model)

    predictions = np.full(len(observed), np.nan)
    workers = min(outer_workers, len(splits))
    if workers == 1:
        completed = (fit_outer_fold(train, test) for train, test in splits)
        for index, (test, predicted) in enumerate(completed, start=1):
            predictions[test] = predicted
            print(f"    nested outer fold {index}/{len(splits)} complete", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(fit_outer_fold, train, test) for train, test in splits]
            for index, future in enumerate(as_completed(futures), start=1):
                test, predicted = future.result()
                predictions[test] = predicted
                print(f"    nested outer fold {index}/{len(splits)} complete", flush=True)
    assert np.all(np.isfinite(predictions)), "outer folds did not cover every row"
    residual = predictions - observed
    return float(np.sqrt(np.mean(residual**2))), float(np.median(np.abs(residual)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", type=int, default=5)
    # The four panel-and-target cells are independent, and a nested cell costs hours, so they are
    # filterable to let one process take one cell and the wall clock fall to a single cell's cost.
    parser.add_argument("--scales", nargs="+", default=list(SCALES), choices=list(SCALES))
    parser.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    parser.add_argument(
        "--outer-workers",
        type=int,
        default=1,
        help="Parallel outer nested-CV folds. Inner selection and the statistical protocol are unchanged.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Run only these model names. By default retained power law and the incumbent zoo are evaluated.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.outer_workers < 1:
        raise ValueError("--outer-workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scale in args.scales:
        fit_panel, heldout_panel = harness.load_scale(scale)
        # The incumbent zoo takes the fit panel because some of its shape grids are panel-derived.
        # Factories rather than instances, so every model is rebuilt from each outer training split.
        builders: list[Callable[[harness.Panel], harness.Model]] = [lambda _panel: retained_model()]
        for name in (model.name for model in zoo.observatory_baselines(fit_panel)):
            builders.append(
                lambda panel, chosen=name: next(m for m in zoo.observatory_baselines(panel) if m.name == chosen)
            )
        for name in (model.name for model in zoo.candidates()):
            builders.append(lambda _panel, chosen=name: next(m for m in zoo.candidates() if m.name == chosen))
        candidates = [build(fit_panel) for build in builders]
        if args.models:
            requested = set(args.models)
            available = {model.name for model in candidates}
            missing = requested - available
            if missing:
                raise ValueError(f"unknown models {sorted(missing)}; available models are {sorted(available)}")
            chosen = [
                (model, build) for model, build in zip(candidates, builders, strict=True) if model.name in requested
            ]
            candidates = [model for model, _build in chosen]
            builders = [build for _model, build in chosen]
        for target in args.targets:
            print(f"[{scale} / {target}] {len(candidates)} models, {len(fit_panel)} fit rows", flush=True)
            for model, build in zip(candidates, builders, strict=True):
                print(f"  {scale:<12} {target:<18} {model.name:<34} started", flush=True)
                # Some zoo entries assume panel features this comparison does not supply. Record the
                # failure and keep going rather than losing the whole table to one incumbent.
                try:
                    fit = harness.fit_model(
                        fit_panel,
                        model,
                        target,
                        n_splits=args.splits,
                        seed=args.seed,
                        split_fn=harness.grouped_splits,
                    )
                    nested, nested_median = nested_out_of_fold(
                        fit_panel,
                        build,
                        target,
                        args.splits,
                        args.seed,
                        args.outer_workers,
                    )
                except Exception as error:
                    print(
                        f"  {scale:<12} {target:<18} {model.name:<34} SKIPPED: {type(error).__name__}: {error}",
                        flush=True,
                    )
                    continue
                observed = heldout_panel.targets[target]
                usable = np.isfinite(observed)
                held = heldout_panel.subset(usable)
                predicted = fit.predict(held, model)
                residual = predicted - held.targets[target]
                rows.append(
                    {
                        "scale": scale,
                        "target": target,
                        "model": model.name,
                        "fit_rows": len(fit_panel),
                        "selection_rmse": fit.oof_rmse,
                        "nested_oof_rmse": nested,
                        # These panels are outlier-heavy, which is why the head is robust; reporting
                        # only squared error would hide the typical-policy comparison the head exists
                        # to improve, exactly as it did on the StarCoder panel.
                        "nested_oof_median": nested_median,
                        "heldout_rows": len(held),
                        "heldout_rmse": float(np.sqrt(np.mean(residual**2))),
                        "heldout_median": float(np.median(np.abs(residual))),
                    }
                )
                print(
                    f"  {scale:<12} {target:<18} {model.name:<34} "
                    f"nested {nested:.6f}  selection {fit.oof_rmse:.6f}  heldout {rows[-1]['heldout_rmse']:.6f}",
                    flush=True,
                )
                checkpoint = args.output_dir / f"partial_{scale}__{target.replace('_bpb', '')}.csv"
                pd.DataFrame([row for row in rows if row["scale"] == scale and row["target"] == target]).to_csv(
                    checkpoint, index=False
                )

    assert rows, "every model failed; the comparison produced nothing"
    required = set(args.models or (MODEL_NAME,))
    produced = {row["model"] for row in rows}
    missing = required - produced
    assert not missing, f"models {sorted(missing)} produced no rows; the comparison is incomplete"
    table = pd.DataFrame(rows)
    suffix = "_".join(args.scales) + "__" + "_".join(t.replace("_bpb", "") for t in args.targets)
    table.to_csv(args.output_dir / f"swarm39_comparison_{suffix}.csv", index=False)
    print("\n" + "=" * 104)
    print("NESTED OUT-OF-FOLD RMSE, RANKED WITHIN EACH PANEL AND TARGET")
    print("selection_rmse is the within-fold selection minimum, shown only to expose its optimism")
    print("=" * 104)
    for (scale, target), block in table.groupby(["scale", "target"]):
        ranked = block.sort_values("nested_oof_rmse")
        marker = ranked["model"].tolist().index(MODEL_NAME) + 1
        print(f"\n{scale} / {target}   retained_power_law ranks {marker} of {len(ranked)}")
        print(
            ranked[["model", "nested_oof_rmse", "nested_oof_median", "heldout_rmse", "heldout_median"]].to_string(
                index=False, float_format=lambda value: f"{value:.6f}"
            )
        )
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
