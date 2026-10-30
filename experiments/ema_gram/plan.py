# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan pilots, expand LR boundaries, and lock paired confirmation settings."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ALPHAS = (0.0, 0.125, 0.25, 1 / 3, 0.5)
RIDGES = (1e-4, 1e-2)
SEEDS = (0, 42, 62746)
PILOT_TOKENS = 250_000_000
SELECTION_METRIC = "eval/paloma/c4_en/loss"


@dataclass(frozen=True)
class Cell:
    alpha: float
    ridge: float
    learning_rate: float
    seed: int = 0
    phase: str = "pilot"
    slug: str = "3e18"
    horizon: int = 100

    def __post_init__(self):
        if (
            self.alpha not in ALPHAS
            or self.ridge not in RIDGES
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("Use the specified exponents, ridges and a positive finite learning rate")
        if self.phase not in ("pilot", "confirmation", "horizon", "scale") or self.seed not in SEEDS:
            raise ValueError("Unknown phase or seed")
        if self.slug != ("3e19" if self.phase == "scale" else "3e18"):
            raise ValueError("Use the 447M recipe before the 998M scale comparison")
        if self.horizon not in ((100, 1000, 10000) if self.phase == "horizon" else (100,)):
            raise ValueError("Unexpected EMA horizon")
        if self.phase == "pilot" and self.seed != 0:
            raise ValueError("Pilot tuning uses seed 0")

    @property
    def name(self) -> str:
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]
        return f"ema-gram-{self.phase}-{self.slug}-a{self.alpha:g}-seed{self.seed}-{digest}"


@dataclass(frozen=True)
class Result:
    name: str
    status: str
    validation_loss: float | None
    completed_steps: int
    expected_steps: int
    run_url: str


def pilots(centers: list[float]) -> list[Cell]:
    """Six trials per exponent; centers are explicit and independently chosen."""
    if len(centers) != len(ALPHAS) or any(not math.isfinite(c) or c <= 0 for c in centers):
        raise ValueError("Supply five finite positive LR centers, one per exponent")
    return [
        Cell(alpha, ridge, center * factor)
        for alpha, center in zip(ALPHAS, centers, strict=True)
        for ridge in ((RIDGES[0],) if alpha == 0 else RIDGES)
        for factor in ((2**e for e in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)) if alpha == 0 else (0.5, 1.0, 2.0))
    ]


def evaluated_groups(cells: list[Cell], results: list[Result]) -> list[tuple[list[Cell], Cell]]:
    """Require every planned pilot to finish or explicitly fail; rank only complete runs."""
    by_name = {r.name: r for r in results}
    if len(by_name) != len(results) or len({c.name for c in cells}) != len(cells):
        raise ValueError("Duplicate cells or results")
    if set(by_name) != {c.name for c in cells} or any(c.phase != "pilot" for c in cells):
        raise ValueError("Results must cover exactly the planned pilots")
    for row in results:
        if row.expected_steps != math.ceil(PILOT_TOKENS / (8 * 4096)):
            raise ValueError("Pilot result uses a different token budget")
        if row.status not in ("completed", "failed"):
            raise ValueError(f"Unfinished pilot: {row.name}")
        if row.status == "completed" and (
            row.completed_steps != row.expected_steps
            or row.expected_steps <= 0
            or row.validation_loss is None
            or not math.isfinite(row.validation_loss)
        ):
            raise ValueError(f"Incomplete or nonfinite pilot: {row.name}")
    groups = []
    for alpha, ridge in sorted({(c.alpha, c.ridge) for c in cells}):
        group = sorted((c for c in cells if (c.alpha, c.ridge) == (alpha, ridge)), key=lambda c: c.learning_rate)
        valid = [c for c in group if by_name[c.name].status == "completed"]
        if not valid:
            raise ValueError(f"No successful pilot at alpha={alpha}, ridge={ridge}")
        winner = min(valid, key=lambda c: by_name[c.name].validation_loss)
        groups.append((group, winner))
    return groups


def expand(cells: list[Cell], results: list[Result]) -> list[Cell]:
    expanded = list(cells)
    for group, winner in evaluated_groups(cells, results):
        if winner == group[0]:
            expanded.append(replace(winner, learning_rate=winner.learning_rate / 2))
        if winner == group[-1]:
            expanded.append(replace(winner, learning_rate=winner.learning_rate * 2))
    return expanded


def lock_settings(cells: list[Cell], results: list[Result], loss_target: float) -> dict:
    """Freeze tuning evidence and a common loss target before any confirmation run."""
    if not math.isfinite(loss_target) or loss_target <= 0:
        raise ValueError("Predeclare a positive finite held-out loss target")
    groups = evaluated_groups(cells, results)
    required = {(a, r) for a in ALPHAS for r in ((RIDGES[0],) if a == 0 else RIDGES)}
    if {(c.alpha, c.ridge) for c in cells} != required:
        raise ValueError("All exponent/ridge groups must be piloted")
    if any(len(group) < (6 if winner.alpha == 0 else 3) for group, winner in groups):
        raise ValueError("Pilot grids must contain six baseline rates and three rates per nonzero exponent/ridge")
    if len(expand(cells, results)) != len(cells):
        raise ValueError("An LR boundary won; expand and evaluate before locking")
    by_name = {r.name: r for r in results}
    selected = [
        min((w for _, w in groups if w.alpha == a), key=lambda c: by_name[c.name].validation_loss) for a in ALPHAS
    ]
    best = min(selected, key=lambda c: by_name[c.name].validation_loss)
    confirm = {0.0, 0.25, 0.5, best.alpha}
    payload = {
        "metric": SELECTION_METRIC,
        "loss_target": loss_target,
        "selected": [asdict(c) for c in selected if c.alpha in confirm],
        "pilots": [asdict(c) for c in cells],
        "results": [asdict(r) for r in results],
    }
    return {**payload, "sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}


def confirmations(lock: dict, *, phase: str = "confirmation") -> list[Cell]:
    payload = {k: v for k, v in lock.items() if k != "sha256"}
    if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != lock["sha256"]:
        raise ValueError("Locked settings changed")
    # Re-validate the evidence instead of trusting an editable lockfile's digest alone.
    expected = lock_settings(
        [Cell(**c) for c in lock["pilots"]], [Result(**r) for r in lock["results"]], lock["loss_target"]
    )
    if expected != lock:
        raise ValueError("Lock does not match pilot selection")
    selected = [Cell(**c) for c in lock["selected"]]
    if phase == "confirmation":
        return [replace(c, phase=phase, seed=s) for c in selected for s in SEEDS]
    if phase not in ("horizon", "scale"):
        raise ValueError(f"Unknown phase: {phase}")
    return [
        replace(c, phase=phase, seed=s, horizon=h, slug="3e19" if phase == "scale" else "3e18")
        for c in selected
        if c.alpha in (0.25, 0.5)
        for s in SEEDS
        for h in ((100, 1000, 10000) if phase == "horizon" else (100,))
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pilot", "expand", "lock", "confirmation", "horizon", "scale"))
    parser.add_argument("--centers", type=float, nargs=5)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--loss-target", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "pilot":
        if args.centers is None:
            parser.error("pilot requires --centers")
        value = [asdict(c) for c in pilots(args.centers)]
    elif args.action in ("expand", "lock"):
        if args.plan is None or args.results is None:
            parser.error("expand and lock require --plan and --results")
        cells = [Cell(**c) for c in json.loads(args.plan.read_text())]
        results = [Result(**r) for r in json.loads(args.results.read_text())]
        if args.action == "expand":
            value = [asdict(c) for c in expand(cells, results)]
        else:
            if args.loss_target is None:
                parser.error("lock requires --loss-target")
            value = lock_settings(cells, results, args.loss_target)
    else:
        if args.lock is None:
            parser.error("confirmation and follow-ups require --lock")
        value = [asdict(c) for c in confirmations(json.loads(args.lock.read_text()), phase=args.action)]
    args.output.write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
