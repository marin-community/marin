# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paloma macro loss of an experiment's runs from W&B, and each variant's verdict against its same-seed control.

    uv run python -m experiments.belay.results X0221 [--json]

The verdict is the fixed rule: Δ = variant - control at the same seed and commit, on ``CONFIG.metric``; Δ ≤
-``CONFIG.noise`` is a win, Δ ≥ +``CONFIG.noise`` a loss, anything between is noise. Sessions record results with
``experiments.belay.steps record`` (``experiments/belay/RUNBOOK.md``), which calls ``record`` here: it writes each
variant's loss, delta and step count into its registry trial, names the control as the baseline, sets the entry to
``done``, then validates, commits and pushes the registry branch.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import wandb
import yaml

from experiments.belay.config import CONFIG
from experiments.belay.submit import commit_registry, load_spec, run_name


class Verdict(StrEnum):
    WIN = "win"
    NOISE = "noise"
    LOSS = "loss"


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
    verdict: Verdict | None = None


def verdict(delta: float) -> Verdict:
    if delta <= -CONFIG.noise:
        return Verdict.WIN
    if delta >= CONFIG.noise:
        return Verdict.LOSS
    return Verdict.NOISE


def experiment_results(experiment_id: str) -> list[RunResult]:
    spec = load_spec(experiment_id)
    api = wandb.Api()
    found = {}
    for run in spec.runs.values():
        name = run_name(experiment_id, run.name)
        matches = list(api.runs(CONFIG.wandb_project, filters={"display_name": name}, order="-created_at"))
        found[run.name] = matches[0] if matches else None
    results = []
    for run in spec.runs.values():
        w = found[run.name]
        loss = w.summary.get(CONFIG.metric) if w is not None else None
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
    out = []
    for r in results:
        control = controls.get(r.seed)
        if r.control or r.loss is None or control is None:
            out.append(r)
            continue
        delta = round(r.loss - control.loss, 4)
        out.append(RunResult(**{**asdict(r), "delta": delta, "verdict": verdict(delta)}))
    return out


def record(experiment_id: str, results: list[RunResult], registry_dir: Path, *, dry_run: bool) -> Path:
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
    text = yaml.safe_dump(entry, sort_keys=False, allow_unicode=True, width=120)
    if dry_run:
        print(f"would write {path}:\n{text}")
        return path
    path.write_text(text)
    commit_registry(registry_dir, path, f"Belay registry: {experiment_id} results recorded")
    return path


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment_id")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    results = experiment_results(args.experiment_id)
    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
        return
    for r in results:
        loss = f"{r.loss:.4f}" if r.loss is not None else "-"
        delta = f"{r.delta:+.4f}" if r.delta is not None else ""
        print(f"{r.name:<32} {r.state:<10} step={r.step} {CONFIG.metric}={loss} {delta} {r.verdict or ''}")


if __name__ == "__main__":
    main(sys.argv[1:])
