# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paloma macro loss of an experiment's runs from W&B, and each variant's delta against its same-seed control.

    uv run python -m experiments.belay.results X0221 [--json]
    uv run python -m experiments.belay.results X0221 --record --registry-dir ~/marin_wt_registry_demo

``--record`` writes each finished variant's loss, delta and step count into its registry trial, names the control
and its loss as the baseline, sets the entry to ``done``, then validates, commits and pushes the registry branch.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import wandb
import yaml

from experiments.belay.submit import commit_registry, load_spec, run_name

WANDB_PATH = "marin-community/marin_moe"
METRIC = "eval/paloma/macro_loss"
# Same-seed copies of one config spread about 0.003; a seed change moves the loss about 0.007 (ladder seed noise).
NOISE = 0.002


@dataclass(frozen=True)
class RunResult:
    run: str
    name: str
    variant: str
    seed: int
    control: bool
    state: str
    step: int | None
    loss: float | None
    url: str | None
    delta: float | None = None
    verdict: str | None = None


def verdict(delta: float, seeds: int) -> str:
    if abs(delta) <= NOISE:
        return f"neutral: within the {NOISE} noise threshold"
    if delta > 0:
        return "worse than control"
    return "better than control" + ("; one seed, confirm with seed 1" if seeds == 1 else "")


def experiment_results(experiment_id: str) -> list[RunResult]:
    spec = load_spec(experiment_id)
    api = wandb.Api()
    found = {}
    for run in spec.runs.values():
        name = run_name(experiment_id, run.name)
        matches = list(api.runs(WANDB_PATH, filters={"display_name": name}, order="-created_at"))
        found[run.name] = matches[0] if matches else None
    results = []
    for run in spec.runs.values():
        w = found[run.name]
        loss = w.summary.get(METRIC) if w is not None else None
        results.append(
            RunResult(
                run=run.name,
                name=run_name(experiment_id, run.name),
                variant=run.variant,
                seed=run.seed,
                control=run.control,
                state=w.state if w is not None else "not started",
                step=w.summary.get("_step") if w is not None else None,
                loss=float(loss) if loss is not None else None,
                url=w.url if w is not None else None,
            )
        )
    controls = {r.seed: r for r in results if r.control and r.loss is not None}
    seeds_per_variant = {}
    for r in results:
        if not r.control:
            seeds_per_variant[r.variant] = seeds_per_variant.get(r.variant, 0) + 1
    out = []
    for r in results:
        control = controls.get(r.seed)
        if r.control or r.loss is None or control is None:
            out.append(r)
            continue
        delta = round(r.loss - control.loss, 4)
        out.append(RunResult(**{**asdict(r), "delta": delta, "verdict": verdict(delta, seeds_per_variant[r.variant])}))
    return out


def record(experiment_id: str, results: list[RunResult], registry_dir: Path) -> Path:
    """Write finished results into the experiment's registry trials (matched by W&B run name)."""
    unfinished = [r.name for r in results if r.state != "finished" or r.loss is None]
    if unfinished:
        raise SystemExit(f"refusing to record: not finished with a Paloma macro loss: {unfinished}")
    (path,) = (registry_dir / "experiments/belay/registry").glob(f"{experiment_id}-*.yaml")
    entry = yaml.safe_load(path.read_text())
    by_name = {r.name: r for r in results}
    controls = {r.seed: r for r in results if r.control}
    for trial in entry.get("trials", []):
        runs = [by_name[name] for name in trial.get("wandb", []) if name in by_name]
        if not runs:
            continue
        trial["loss"] = [round(r.loss, 4) for r in runs]
        trial["delta"] = [r.delta for r in runs]
        trial["steps"] = int(max(r.step for r in runs))
        trial["baseline"] = ", ".join(
            f"{controls[r.seed].name} {controls[r.seed].loss:.4f} (same commit, seed {r.seed})" for r in runs
        )
    entry["status"] = "done"
    path.write_text(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True, width=120))
    commit_registry(registry_dir, path, f"Belay registry: {experiment_id} results recorded")
    return path


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment_id")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--record", action="store_true", help="write the results into the registry")
    parser.add_argument("--registry-dir", type=Path, help="checkout of the registry branch (with --record)")
    args = parser.parse_args(argv)
    results = experiment_results(args.experiment_id)
    if args.record:
        if args.registry_dir is None:
            raise SystemExit("--record needs --registry-dir")
        print(f"recorded in {record(args.experiment_id, results, args.registry_dir)}")
    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
        return
    for r in results:
        loss = f"{r.loss:.4f}" if r.loss is not None else "-"
        delta = f"{r.delta:+.4f}" if r.delta is not None else ""
        print(f"{r.name:<32} {r.state:<10} step={r.step} {METRIC}={loss} {delta} {r.verdict or ''}")


if __name__ == "__main__":
    main(sys.argv[1:])
