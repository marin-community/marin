# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Rank surrogates on SINGLE-PHASE mixture optimisation, and on which panel they should be fit.

The prefix search and the two-stage optimiser both need one thing the two-phase programme never
isolated: a surrogate that picks a good single-phase mixture. The existing model benchmark reports a
`single_phase` stratum, but every model in it was fitted on the TWO-PHASE fit panel, whose median phase
separation is TV 0.497. A single-phase fit panel of 280 rows exists for both scales and was never wired
into `load_scale`, so the obvious question -- does fitting on single-phase data produce a better
single-phase optimiser -- has not been asked.

This asks it. Each model is fitted on both panels and scored only on held-out rows whose two phases are
identical, by the value of the policy it SELECTS rather than by fit error, since selection is what the
prefix search consumes. Calibration slope and bias are reported beside it because an optimiser is driven
by predicted differences, not predicted levels: a model that ranks well but reads optimistic will still
walk the search into unsupported mixtures.
"""

import argparse
import contextlib
import dataclasses
import itertools
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import fit_swarm39_split_damage_20260817 as split_damage  # noqa: E402
import general_mixture_surrogate_20260809 as gen  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import swarm39_harness_20260725 as swarm39  # noqa: E402
import swarm39_models_20260725 as zoo  # noqa: E402
from scipy import stats  # noqa: E402

ONE_PHASE_DATASET = {"300m": "300m_one_phase_fit", "delphi_3e18": "delphi_3e18_one_phase_fit"}
QUALITY_SUFFIXES = ("_high", "_low")
GROUPINGS = ("strata", "domain", "semantic")
QUALITY_WEIGHT_BOUND = (-6.0, 2.0)


def domain_of(bucket: str) -> str:
    """The corpus a bucket came from, with any quality suffix removed.

    This is the only bucket grouping the panel legitimately carries: some buckets are quality splits of
    one domain, exactly as in the GrugMoE production swarm. It says nothing about what a bucket contains.
    """
    for suffix in QUALITY_SUFFIXES:
        if bucket.endswith(suffix):
            return bucket[: -len(suffix)]
    return bucket


def regroup(panel, grouping: str):
    """Replace the panel's family partition, which every model in the zoo reads through `family_pool`.

    `semantic` is the retired hand-assigned partition and is kept for archival comparison only. `domain`
    groups quality splits of one corpus and is label-free, but it leaves thirteen singleton groups, and a
    singleton's pooled column is its own per-bucket column, so the design goes to rank deficit 55 of 196
    against 7 of 58. `strata` cuts equal-count strata on log epochs per unit weight, needing only the token
    counts the exposure columns are already built from.
    """
    if grouping == "semantic":
        return panel
    if grouping == "domain":
        keys = [domain_of(bucket) for bucket in panel.buckets]
        names = tuple(sorted(set(keys)))
        index = np.array([names.index(key) for key in keys])
    elif grouping == "strata":
        probe = gen.Panel(np.stack([panel.phase0, panel.phase1], axis=1), panel.c0, panel.c1, panel.family_index)
        index = probe.exposure_stratum()
        names = tuple(f"stratum_{k}" for k in range(int(index.max()) + 1))
    else:
        raise ValueError(f"unknown grouping {grouping!r}")
    return dataclasses.replace(panel, family_index=index, family_names=names)


def quality_split_pairs(panel) -> tuple[tuple[int, int], ...]:
    """Departure-column index pairs for buckets that are quality splits of one domain.

    The per-bucket departure block starts after the pooled columns, so bucket ``b`` owns column
    ``pooled_width + b``. Tying those pairs asks two quality splits of the same corpus to carry the same
    readout departure unless the data insists otherwise -- which is the one piece of grouping structure
    the panel legitimately has, used as a prior rather than as a partition.
    """
    offset = gen.pooled_width(
        gen.Panel(np.stack([panel.phase0, panel.phase1], axis=1), panel.c0, panel.c1, panel.family_index), "split"
    )
    groups: dict[str, list[int]] = {}
    for position, bucket in enumerate(panel.buckets):
        groups.setdefault(domain_of(bucket), []).append(position)
    pairs = []
    for members in groups.values():
        for first, second in itertools.pairwise(members):
            pairs.append((offset + first, offset + second))
    return tuple(pairs)


DSP_REFERENCE = "effective_exposure_dsp"
"""The long-standing incumbent. Every margin in this benchmark is quoted against it."""

RECOMMENDED = "general"
"""The configuration to deploy, quoted alongside the per-cell leader.

The per-cell leader is a different model in four of six cells, so it is a winner's-curse artefact rather
than a recommendation. This one is consistently near the top, so its margin over DSP is the honest
headline."""


def later_mechanisms() -> list[swarm39.Model]:
    """Builders added after the model benchmark froze, none of them ever scored on single-phase rows.

    Each carries a distinct mechanism the programme identified and tested elsewhere: a family-benefit
    term on top of bounded saturation, the compact-retained-state extensions, an explicit breadth and
    geometry channel, and two multiplicative responses. The last two are the interesting ones here --
    ``log_deficit`` fits ``log(BPB - floor)`` so a prediction cannot fall below an entropy floor, which is
    the structural cure for the out-of-support optimism that drives an optimiser into unsupported
    mixtures.
    """
    return [
        swarm39.Model("bounded_hierarchical", zoo.build_bounded_hierarchical, zoo.bounded_saturation_shapes),
        swarm39.Model("crs_plus", zoo.build_crs_plus, zoo.crs_plus_shapes),
        swarm39.Model("crs_plus_breadth", zoo.build_crs_plus_breadth, zoo.crs_plus_shapes),
        swarm39.Model("crs_plus_geometry", zoo.build_crs_plus_geometry, zoo.crs_plus_shapes),
        swarm39.Model("crs_plus_heads", zoo.build_crs_plus_heads, zoo.crs_plus_shapes),
        swarm39.Model("structured_benefit", zoo.build_structured_benefit, zoo.structured_shapes),
        swarm39.Model("log_ratio_deficit", zoo.build_log_ratio_deficit, zoo.log_ratio_shapes),
        swarm39.Model(
            "multiplicative_deficit", zoo.build_multiplicative_deficit, zoo.multiplicative_shapes, link="log_deficit"
        ),
    ]


SEEDS = 3


def one_phase_panel(scale: str) -> swarm39.Panel:
    """The single-phase fit panel, built through the same loader the two-phase panel uses."""
    dataset = ONE_PHASE_DATASET[scale]
    domains, c0, c1, family_index, family_names = swarm39._exposure(dataset)
    frame = pd.read_csv(swarm39.CANONICAL / f"{dataset}.csv")
    swarm39.assert_sealed_absent(frame, f"{scale} one-phase fit")
    panel = swarm39._panel_from_weight_columns(
        scale,
        "fit",
        frame,
        domains,
        c0,
        c1,
        family_index,
        family_names,
        "phase_0_weight::",
        "phase_1_weight::",
        {swarm39.UNCHEATABLE: swarm39.UNCHEATABLE, swarm39.TABLE9: swarm39.TABLE9},
    )
    separation = float(panel.phase_tv.max())
    assert separation < 1e-9, f"{dataset} is not single-phase (max separation {separation:.2e})"
    return panel


def single_phase_heldout(scale: str) -> swarm39.Panel:
    _fit, held = swarm39.load_scale(scale)
    return held.subset(held.phase_tv <= 1e-9)


def score(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    """Selection value first; calibration and bias beside it because an optimiser rides the gradient."""
    order = np.argsort(predicted)
    best = float(observed.min())
    slope = float(np.polyfit(predicted, observed, 1)[0]) if np.std(predicted) > 1e-12 else float("nan")
    return {
        "regret@1": float(observed[order[0]]) - best,
        "regret@3": float(observed[order[:3]].min()) - best,
        "spearman": float(stats.spearmanr(predicted, observed).statistic),
        "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "bias": float(np.mean(predicted - observed)),
        "calibration": slope,
    }


def zoo_scores(fit_panel, held, model, target: str) -> dict[str, float]:
    fitted = swarm39.fit_model(fit_panel, model, target)
    return score(fitted.predict(held, model), held.targets[target])


@contextlib.contextmanager
def quality_shrinkage(pairs: tuple[tuple[int, int], ...]):
    """Add quality-split pairs to the departure penalty for the duration of a fit.

    `fit_head` already knows how to shrink two coefficients toward each other; this only widens the set of
    pairs it is given, so the fitter itself is reused unchanged. The weight is shared with the damage
    departure it already carries, which is a compromise: one knob now serves two shrinkages.
    """
    original = split_damage.departure_pairs

    def patched(panel, variant):
        return original(panel, variant) + pairs

    split_damage.departure_pairs = patched
    try:
        yield
    finally:
        split_damage.departure_pairs = original


def general_scores(fit_panel, held, target: str, extra_pairs: tuple[tuple[int, int], ...] = ()) -> dict[str, float]:
    """GEN-001 through the split-damage head, pooled by whatever partition the panel now carries."""
    index = fit_panel.family_index
    ok = np.isfinite(fit_panel.targets[target])
    train = gen.Panel(np.stack([fit_panel.phase0[ok], fit_panel.phase1[ok]], axis=1), fit_panel.c0, fit_panel.c1, index)
    query = gen.Panel(np.stack([held.phase0, held.phase1], axis=1), fit_panel.c0, fit_panel.c1, index)
    response = fit_panel.targets[target][ok]
    with quality_shrinkage(extra_pairs):
        predictions = [
            split_damage.predict(query, split_damage.fit_variant(train, response, "split", seed), "split")
            for seed in range(SEEDS)
        ]
    return score(np.median(predictions, axis=0), held.targets[target])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", default="300m,delphi_3e18")
    parser.add_argument("--extended", action="store_true", help="add the mechanisms never scored on single-phase")
    parser.add_argument("--grouping", choices=GROUPINGS, default="strata", help="semantic is archival only")
    args = parser.parse_args()

    collected = []
    print(f"bucket grouping: {args.grouping}")
    for scale in args.scales.split(","):
        two_phase_fit, _held = swarm39.load_scale(scale)
        two_phase_fit = regroup(two_phase_fit, args.grouping)
        held = regroup(single_phase_heldout(scale), args.grouping)
        fit_panels = [("two-phase", two_phase_fit)]
        if scale in ONE_PHASE_DATASET:
            fit_panels.append(("one-phase", regroup(one_phase_panel(scale), args.grouping)))
        else:
            print(f"\n[{scale}] no single-phase fit panel in the catalog; fitting on the two-phase panel only")
        models = zoo.observatory_baselines(two_phase_fit) + zoo.candidates()
        if args.extended:
            models = models + later_mechanisms()
        for target in (swarm39.UNCHEATABLE, swarm39.TABLE9):
            usable = np.isfinite(held.targets[target])
            panel = held.subset(usable)
            rows = []
            for fit_name, fit_panel in fit_panels:
                for model in models:
                    rows.append(
                        {"model": model.name, "fitted_on": fit_name} | zoo_scores(fit_panel, panel, model, target)
                    )
                rows.append({"model": "general", "fitted_on": fit_name} | general_scores(fit_panel, panel, target))
                rows.append(
                    {"model": "general+quality", "fitted_on": fit_name}
                    | general_scores(fit_panel, panel, target, quality_split_pairs(fit_panel))
                )
            table = pd.DataFrame(rows).sort_values(["regret@1", "spearman"], ascending=[True, False])
            table["cell"] = f"{scale}/{target.split('_')[0]}"
            table["ties_best"] = table["regret@1"] <= table["regret@1"].min() + 1e-12
            collected.append(table)
            print(f"\n=== {scale} / {target} -- single-phase held-out rows, n={len(panel.phase0)} ===")
            print(table.drop(columns=["cell", "ties_best"]).to_string(index=False, float_format=lambda v: f"{v:+.5f}"))

    if not collected:
        return
    everything = pd.concat(collected)
    print(f"\n\n=== across {everything['cell'].nunique()} cells ===")
    summary = everything.groupby(["model", "fitted_on"]).agg(
        cells_tied_best=("ties_best", "sum"),
        mean_spearman=("spearman", "mean"),
        mean_calibration=("calibration", "mean"),
    )
    print(
        summary.sort_values(["cells_tied_best", "mean_spearman"], ascending=False).to_string(
            float_format=lambda v: f"{v:+.4f}"
        )
    )

    print(f"\n=== every configuration against the {DSP_REFERENCE} incumbent, per cell ===")
    margins = []
    for cell, group in everything.groupby("cell"):
        dsp = group[group["model"] == DSP_REFERENCE]
        if dsp.empty:
            continue
        reference = dsp.sort_values("spearman", ascending=False).iloc[0]
        leader = group.sort_values(["regret@1", "spearman"], ascending=[True, False]).iloc[0]
        picked = group[group["model"] == RECOMMENDED].sort_values("spearman", ascending=False)
        chosen = picked.iloc[0] if not picked.empty else leader
        margins.append(
            {
                "cell": cell,
                "leader": f"{leader['model']}/{leader['fitted_on']}",
                "leader_rho": leader["spearman"],
                "recommended_fit": chosen["fitted_on"],
                "recommended_rho": chosen["spearman"],
                "dsp_rho": reference["spearman"],
                "d_rho": chosen["spearman"] - reference["spearman"],
                "recommended_regret": chosen["regret@1"],
                "dsp_regret": reference["regret@1"],
            }
        )
    margin_table = pd.DataFrame(margins)
    print(margin_table.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))
    print(
        f"  {RECOMMENDED} beats {DSP_REFERENCE} on rank correlation by {margin_table['d_rho'].mean():+.4f} on "
        f"average (median {margin_table['d_rho'].median():+.4f}), winning "
        f"{int((margin_table['d_rho'] > 0).sum())} of {len(margin_table)} cells"
    )

    # Paired within model and cell, so this isolates the panel and not the model mix.
    wide = everything.pivot_table(index=["model", "cell"], columns="fitted_on", values="spearman")
    delta = (wide["one-phase"] - wide["two-phase"]).dropna()
    regressions = delta[delta <= 0]
    if len(regressions):
        print(f"\n  panel switch HURTS these model-cell pairs: {', '.join(f'{m}@{c}' for m, c in regressions.index)}")
    print(
        f"\nfitting on the single-phase panel instead of the two-phase panel changes rank correlation by "
        f"{delta.mean():+.4f} on average (median {delta.median():+.4f}); "
        f"it helps in {int((delta > 0).sum())} of {len(delta)} model-cell pairs"
    )


if __name__ == "__main__":
    main()
