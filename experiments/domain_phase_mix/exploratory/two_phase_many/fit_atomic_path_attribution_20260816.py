# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Path-attributed exposure and damage on the with-repetition two-bucket panel (ATOM-012).

Scores the candidates registered in `atom012_path_attribution_preregistration_20260816.md` over the whole
replay ladder and all four horizons, one atomic objective at a time. Criteria and signatures were fixed
before any fit; this driver only reports them.

`path-tied` is the ablation that matters. It shares theta, both laws, and every other choice with
`path-exposure`, differing only in that the phase attribution is summed back together. If the split is
not what produces the result, the two score the same.

Usage: ``uv run python ... [--supports full,m025,...] [--family code] [--workers N]``
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    atomic_surface_panel_20260811 as panel_module,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_atomic_contrast_criterion_20260812 as contrast_module,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_atomic_stage1_20260811 as stage1,
)

CANDIDATES = (
    "two-bucket",
    "two-horizon",
    "two-bucket-damage",
    "two-horizon-damage",
    "two-bucket-split-damage",
    "two-horizon-split-damage",
    "path-tied",
    "path-exposure",
    "path-damage",
)
SUPPORTS = ("full", "m0125", "m025", "m050", "m100", "m200", "m400")
CODE_MARKERS = ("programing", "github", "arxiv_computer")
MARGIN = 0.002  # BPB; the deployment margin the non-inferiority statistic is tested against


@cache
def _panels(support: str):
    return tuple(panel_module.panels_by_horizon(panel_module.load_support(support)))


def evaluate(support: str, horizon_index: int, key: str, name: str) -> dict:
    """One candidate on one target at one condition: fit quality, predicted gain, and what it selects."""
    panel = _panels(support)[horizon_index]
    folds = panel_module.spatial_folds(panel)
    response = panel.target(key)

    theta = contrast_module.select(panel, response, folds, name, "level")
    columns = stage1.design(panel, name, theta)
    predictions = np.empty_like(response)
    for train, test in folds:
        predictions[test] = columns[test] @ stage1.solve(columns[train], response[train])
    coefficients = stage1.solve(columns, response)

    axis = np.linspace(0.0, 1.0, stage1.GRID)
    g0, g1 = np.meshgrid(axis, axis, indexing="ij")
    flat0, flat1 = g0.ravel(), g1.ravel()
    surface = stage1.design(stage1._grid_panel(panel, flat0, flat1), name, theta) @ coefficients
    tied_surface = stage1.design(stage1._grid_panel(panel, axis, axis), name, theta) @ coefficients

    # A single-index model is EXACTLY FLAT along a whole family of policies, so `argmin` does not choose
    # its recommendation -- the raveling order does, returning the lowest phase-0 share on the plateau.
    # At m400 that lands on the p0 = 0 edge, which is where the true code optimum happens to be, and the
    # model is credited with a recommendation it cannot make. Scoring the plateau's mean realised loss
    # says instead what a practitioner following an indifferent model would get on average.
    plateau = np.flatnonzero(surface <= surface.min() + 1e-10 * max(abs(float(surface.min())), 1.0))
    nearest = np.argmin(
        (flat0[plateau, None] - panel.phase_0[None, :]) ** 2 + (flat1[plateau, None] - panel.phase_1[None, :]) ** 2,
        axis=1,
    )
    landed = float(response[nearest].mean())
    centre = (float(flat0[plateau].mean()), float(flat1[plateau].mean()))

    best_tied = int(np.argmin(np.where(panel.tied, response, np.inf)))
    return {
        "support": support,
        "horizon": panel.horizon,
        "target": key,
        "candidate": name,
        "code": any(marker in key for marker in CODE_MARKERS),
        "rmse": float(np.sqrt(np.mean((predictions - response) ** 2))),
        "spread": float(response.std()),
        "gain": float(tied_surface.min() - surface.min()),
        "observed_gain": float(response[best_tied] - response.min()),
        "regret": landed - float(response.min()),
        "versus_tied": landed - float(response[best_tied]),
        "plateau": len(plateau) / len(surface),
        "where": centre,
        "amplitudes": tuple(float(value) for value in coefficients[1:]),
    }


def _work(item):
    return evaluate(*item)


def non_inferiority(values: np.ndarray) -> float:
    """Upper 95% confidence bound on the mean, the quantity the deployment margin is compared against."""
    if len(values) < 2:
        return float("nan")
    return float(values.mean() + stats.t.ppf(0.95, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def report(records: list[dict], supports: tuple[str, ...]) -> None:
    horizons = sorted({r["horizon"] for r in records})
    print("\nPRIMARY: code targets, |predicted gain - observed gain| and selected-policy regret")
    print("             incumbent best at m400/7.408B was gain error 0.00421, regret 0.00612\n")
    for support in supports:
        for horizon in horizons:
            rows = [r for r in records if r["support"] == support and r["horizon"] == horizon and r["code"]]
            if not rows:
                continue
            observed = np.median([r["observed_gain"] for r in rows])
            print(f"  --- {support:5s} horizon {horizon:6.3f}B   observed code gain {observed:+.5f} ---")
            for name in CANDIDATES:
                subset = [r for r in rows if r["candidate"] == name]
                if not subset:
                    continue
                error = np.array([abs(r["gain"] - r["observed_gain"]) for r in subset])
                regret = np.array([r["regret"] for r in subset])
                relative = np.array([r["rmse"] / max(r["spread"], 1e-12) for r in subset])
                print(
                    f"    {name:20s} gain err {np.median(error):.5f}  regret {np.median(regret):.5f}  "
                    f"U {non_inferiority(regret):.5f}  vs-tied {np.median([r['versus_tied'] for r in subset]):+.5f}  "
                    f"rmse/sd {np.median(relative):.3f}  flat {np.median([r['plateau'] for r in subset]):.3f}"
                )
        print()

    print("SIGNATURE 3: what the fitted model recommends, and the amplitude ratio behind it")
    for support in supports:
        rows = [r for r in records if r["support"] == support and r["code"] and r["horizon"] == horizons[-1]]
        if not rows:
            continue
        for r in [r for r in rows if r["candidate"] == "two-bucket-split-damage"]:
            # Amplitudes are non-negative throughout, so each ratio is the mechanism's own claim about
            # timing: above one means the decay phase carries more of that effect than the stable phase.
            harm_early, harm_late = r["amplitudes"][-2], r["amplitudes"][-1]
            print(
                f"  {support:5s} {r['target'].split('/')[-2][:26]:26s} picks "
                f"({r['where'][0]:.3f}, {r['where'][1]:.3f})  damage early {harm_early:.5f} late {harm_late:.5f}"
                f"   late/early {harm_late / max(harm_early, 1e-12):10.3f}"
            )

    control = [r for r in records if not r["code"]]
    if control:
        print("\nNEGATIVE CONTROL: broad-text targets, which cannot respond to repeating StarCoder")
        for support in sorted({r["support"] for r in control}):
            print(f"  --- {support} ---")
            for name in CANDIDATES:
                subset = [r for r in control if r["candidate"] == name and r["support"] == support]
                if not subset:
                    continue
                error = np.array([abs(r["gain"] - r["observed_gain"]) for r in subset])
                gain = np.array([r["gain"] for r in subset])
                print(
                    f"    {name:20s} gain err {np.median(error):.5f}  predicted gain {np.median(gain):+.5f}  "
                    f"regret {np.median([r['regret'] for r in subset]):.5f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supports", default=",".join(SUPPORTS))
    parser.add_argument("--family", default="code", choices=("code", "broad", "both"))
    parser.add_argument("--horizons", default="", help="comma-separated horizon indices; default all")
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    supports = tuple(args.supports.split(","))
    candidates = tuple(args.candidates.split(","))
    targets = panel_module.atomic_targets()
    code = [k for k in targets if any(marker in k for marker in CODE_MARKERS)]
    broad = [k for k in targets if k not in code]
    chosen = {"code": code, "broad": broad, "both": list(targets)}[args.family]
    indices = [int(value) for value in args.horizons.split(",")] if args.horizons else list(range(4))

    items = [(s, h, k, n) for s in supports for h in indices for k in chosen for n in candidates]
    print(f"ATOM-012 path attribution: {len(items)} fits over {len(supports)} supports x {len(indices)} horizons")
    print(f"{len(chosen)} {args.family} targets x {len(candidates)} candidates, spatial leave-region-out folds\n")

    records = []
    frames = panel_module.load_all_supports()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=panel_module.seed_cache, initargs=(frames,)) as pool:
        for done, record in enumerate(pool.map(_work, items, chunksize=4), start=1):
            records.append(record)
            if done % 50 == 0:
                print(f"  ...{done}/{len(items)}", flush=True)
    report(records, supports)


if __name__ == "__main__":
    main()
