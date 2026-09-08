# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""GEN-002 transfer and negative controls on WSD80, over several seeds (GEN-010).

The round charter freezes a gate on the negative controls: predicted phase gain on each broad-text
target must stay at or below 0.005 BPB. That gate had been evaluated on ONE seed, which cannot
distinguish a model that puts no gain on broad text from a model whose broad-text gain happens to land
low on the seed that was run. Selection is reseeded per seed, so the spread across seeds is the relevant
quantity and a single draw understates it.

Reports every target's predicted gain per seed plus the across-seed spread. Code targets are POSITIVE
controls: real phase gain there is the desired behaviour, so they are marked and exempted rather than
counted as violations.

Usage: ``uv run python ... [seeds]``, default 0-4.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_multitarget_interference_evidence_20260806 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_general_surrogate_wsd80_20260810 as driver,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    general_mixture_surrogate_20260809 as model,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    multitarget_ile_wsd80_20260806 as wsd,
)

CONTROL_GAIN_LIMIT = harness.WSD_NEGATIVE_GAIN_LIMIT
CODE_MARKERS = ("programing_languages", "github_")


def selected_shapes(seeds: list[int]) -> dict[int, tuple]:
    """One selection per seed, not per target.

    ``select`` fits the shared nonlinear parameters against ALL targets jointly -- that is the whole
    point of the multi-target direction -- so on the full row set it depends on the seed alone. Caching
    it turns 26 targets x 5 seeds of selection into 5.
    """
    rows = np.arange(len(driver.TARGETS))
    return {seed: model.unpack(driver.select(rows, seed), driver.N_FAMILIES, driver.N_STRATA) for seed in seeds}


def grid_designs(shape) -> tuple:
    axis = np.linspace(0.0, 1.0, driver.SURFACE_GRID)
    grid_0, grid_1 = np.meshgrid(axis, axis, indexing="ij")
    full = model.design(driver.grid_panel(wsd.grid_weights(grid_0.ravel(), grid_1.ravel())), shape)
    tied_axis = np.linspace(0.0, 1.0, driver.SURFACE_GRID * driver.SURFACE_GRID // 4)
    tied = model.design(driver.grid_panel(wsd.grid_weights(tied_axis, tied_axis)), shape)
    return full, tied


def predicted_gain(response: np.ndarray, shape, ridge: float, designs: tuple) -> float:
    """Best two-phase surface value minus the best tied value, both from the full-data fit."""
    free, constrained = model.design(driver.SWARM, shape)
    b, a = model.fit_head(free, constrained, response, ridge, model.pooled_width(driver.SWARM))
    (fg, cg), (ft, ct) = designs
    return float((ft @ b + ct @ a).min() - (fg @ b + cg @ a).min())


def main() -> None:
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3, 4]
    names = driver.reference.TARGETS.names
    values = driver.TARGETS
    stamp = f"[cfg NF={driver.N_FAMILIES} NS={driver.N_STRATA}]"
    print(f"GEN-010: GEN-002 controls over seeds {seeds} {stamp}")
    print(f"negative-control gate: predicted phase gain <= {CONTROL_GAIN_LIMIT} BPB on every broad-text target")
    print("'+' marks a POSITIVE control (code), where real gain is expected and the gate does not apply\n")

    shapes = selected_shapes(seeds)
    designs = {seed: grid_designs(shapes[seed][0]) for seed in seeds}

    violations = 0
    checked = 0
    for name in names:
        response = values[:, names.index(name)]
        gains = np.array([predicted_gain(response, shapes[s][0], shapes[s][1], designs[s]) for s in seeds])
        is_code = any(k in name for k in CODE_MARKERS)
        worst = float(gains.max())
        if not is_code:
            checked += 1
            failed = int((gains > CONTROL_GAIN_LIMIT).sum())
            violations += failed > 0
            flag = f"   <-- {failed}/{len(seeds)} seeds above limit" if failed else ""
        else:
            flag = ""
        marker = "+" if is_code else " "
        per_seed = " ".join(f"{g:+.6f}" for g in gains)
        print(f" {marker}{name:64s} worst {worst:+.6f}  spread {np.ptp(gains):.6f}  [{per_seed}]{flag}")

    print(f"\n{stamp} {checked - violations}/{checked} negative controls hold on ALL {len(seeds)} seeds")


if __name__ == "__main__":
    main()
