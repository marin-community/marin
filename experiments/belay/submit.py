# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch Marin Belay runs from their committed spec, ``experiments/belay/runs/<experiment_id>.yaml``.

A run is fully specified by a pushed commit plus a run name, so every Belay run is reviewable in the diff. Sessions
launch through ``experiments.belay.steps launch`` (``experiments/belay/RUNBOOK.md``); this module holds the checks
and the launch itself, and its command line launches one run by hand:

    uv run python -m experiments.belay.submit X0221 seed0 --commit <sha> --comment-url <where the plan was posted>

Every launch is refused unless: the checkout is on this week's experiment branch (``CONFIG.week_branch``), that
branch contains ``CONFIG.base_branch``, ``--commit`` is its clean and pushed HEAD, the registry checkout is on
``CONFIG.registry_branch``, no run name was used before, and Belay's active runs plus these fit the soft node cap.
Runs are named ``belay-<id>-<run>``, write under ``CONFIG.output_root``, run on ``CONFIG.cluster`` at
``CONFIG.band`` priority, and are recorded as trials in the registry entry (committed and pushed on the registry
branch). Control runs are recorded as the trial's baseline rather than as trials of their own.
"""

import argparse
import datetime
import getpass
import itertools
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml

from experiments.belay.config import CONFIG

RUNS_DIR = Path(__file__).parent / "runs"
RUN_PREFIX = "belay-"
# Longest run key in a spec: W&B caps names at 128 characters and the job tree adds more.
MAX_RUN_KEY = 12
# 8xH100 nodes per training run, by rung.
NODES_BY_SIZE = {"d512": 1, "d768": 1, "d1024": 1, "d1280": 1, "d1280L16": 4}
# Fixed launcher modules a spec may name; anything else is a code change to review here.
LAUNCHERS = {
    "s7n": ("experiments.grug.fast_track.belay_s7", "--ladder", "s7n"),  # the current recipe, Belay Ckpt Oct 20 2026
    "ckpt1": ("experiments.grug.fast_track.belay_ckpt1",),
    "baseline": ("experiments.grug.fast_track.launch",),
}
ACTIVE_STATES = ("pending", "running")
_JOB_LINE = re.compile(r"^(/\S+)\s+(\w+)\s+(\S+)")


class Protocol(StrEnum):
    EQUAL_STEPS = "equal_steps"
    FIXED_TIME = "fixed_time"


@dataclass(frozen=True)
class RunSpec:
    """One run of an experiment, as committed in its spec file."""

    name: str
    size: str
    seed: int
    variant: str
    flags: tuple[str, ...] = ()
    control: bool = False

    @property
    def nodes(self) -> int:
        return NODES_BY_SIZE[self.size]


@dataclass(frozen=True)
class ExperimentSpec:
    experiment: str
    launcher: str
    protocol: Protocol
    runs: dict[str, RunSpec] = field(default_factory=dict)


def run_name(experiment_id: str, run: str) -> str:
    return f"{RUN_PREFIX}{experiment_id}-{run}"


def load_spec(experiment_id: str, runs_dir: Path = RUNS_DIR) -> ExperimentSpec:
    path = runs_dir / f"{experiment_id}.yaml"
    data = yaml.safe_load(path.read_text())
    if data.get("experiment") != experiment_id:
        raise ValueError(f"{path}: 'experiment' must be {experiment_id!r}")
    launcher = data.get("launcher")
    if launcher not in LAUNCHERS:
        raise ValueError(f"{path}: launcher must be one of {sorted(LAUNCHERS)}, got {launcher!r}")
    runs = {}
    for name, run in data["runs"].items():
        if len(name) > MAX_RUN_KEY:
            raise ValueError(f"{path}: run key {name!r} is longer than {MAX_RUN_KEY} characters")
        if run["size"] not in NODES_BY_SIZE:
            raise ValueError(f"{path}: run {name}: size must be one of {sorted(NODES_BY_SIZE)}")
        runs[name] = RunSpec(
            name=name,
            size=run["size"],
            seed=int(run["seed"]),
            variant=run["variant"],
            flags=tuple(str(f) for f in run.get("flags", ())),
            control=bool(run.get("control", False)),
        )
    return ExperimentSpec(experiment_id, launcher, Protocol(data.get("protocol", Protocol.EQUAL_STEPS)), runs)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def require_week_branch(repo: Path, today: datetime.date) -> str:
    """This week's experiment branch, after checking ``repo`` is on it and it contains ``CONFIG.base_branch``."""
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    expected = CONFIG.week_branch(today)
    if branch != expected:
        raise SystemExit(
            f"refusing: {repo} is on {branch}, not this week's branch {expected}.\n"
            "next: uv run python -m experiments.belay.steps week"
        )
    _git(repo, "fetch", "--quiet", "origin", CONFIG.base_branch)
    contains = subprocess.run(["git", "merge-base", "--is-ancestor", f"origin/{CONFIG.base_branch}", "HEAD"], cwd=repo)
    if contains.returncode != 0:
        raise SystemExit(
            f"refusing: {branch} does not contain origin/{CONFIG.base_branch}.\n"
            "next: uv run python -m experiments.belay.steps week  (merges the base in, or tells you to escalate)"
        )
    return branch


def require_pushed_head(repo: Path, commit: str, branch: str) -> str:
    """The full SHA of ``commit`` after checking it is the clean HEAD of ``repo`` and pushed to ``branch``."""
    head = _git(repo, "rev-parse", "HEAD")
    full = _git(repo, "rev-parse", commit)
    if full != head:
        raise SystemExit(f"refusing: --commit {commit} is not the checked-out HEAD ({head[:10]})")
    dirty = [line for line in _git(repo, "status", "--porcelain", "--untracked-files=no").splitlines() if line]
    if dirty:
        raise SystemExit(f"refusing: tracked files differ from HEAD: {dirty}")
    _git(repo, "fetch", "--quiet", "origin", branch)
    if _git(repo, "rev-parse", f"origin/{branch}") != full:
        raise SystemExit(f"refusing: {full[:10]} is not the tip of origin/{branch}.\nnext: git push origin {branch}")
    return full


def require_registry_checkout(registry_dir: Path) -> None:
    branch = _git(registry_dir, "rev-parse", "--abbrev-ref", "HEAD")
    if branch != CONFIG.registry_branch:
        raise SystemExit(f"refusing: {registry_dir} is on {branch}, not the registry branch {CONFIG.registry_branch}")


def iris_jobs(prefix: str) -> list[tuple[str, str, str]]:
    """``(job_id, state, submitted_at)`` for every Iris job whose id starts with ``prefix``."""
    out = subprocess.run(
        ["uv", "run", "iris", "--cluster", "marin", "job", "list", "--prefix", prefix, "--limit", "500"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [(m.group(1), m.group(2), m.group(3)) for line in out.splitlines() if (m := _JOB_LINE.match(line))]


def belay_job_prefix() -> str:
    return f"/{getpass.getuser()}/{RUN_PREFIX}"


def nodes_in_use(jobs: list[tuple[str, str, str]], runs_dir: Path = RUNS_DIR) -> int:
    """Nodes held or requested by active Belay runs, from each run's spec (1 node when the spec is not found)."""
    total = 0
    for job_id, state, _ in jobs:
        parts = job_id.strip("/").split("/")
        if len(parts) != 2 or state not in ACTIVE_STATES:
            continue  # count each run once, by its coordinator job
        name = parts[1].removeprefix(RUN_PREFIX).removesuffix("-coord")
        experiment_id, _, run = name.partition("-")
        try:
            total += load_spec(experiment_id, runs_dir).runs[run].nodes
        except (FileNotFoundError, KeyError):
            total += 1
    return total


def launch_command(spec: ExperimentSpec, run: RunSpec, name: str) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "-m",
        *LAUNCHERS[spec.launcher],
        "--size",
        run.size,
        *run.flags,
        "--seed",
        str(run.seed),
        "--run-id",
        name,
        "--output-path",
        f"{CONFIG.output_root}/{name}",
        "--submit",
        "--target-cluster",
        CONFIG.cluster,
        "--priority",
        CONFIG.band,
    ]


def settings_of(flags: tuple[str, ...]) -> dict[str, str]:
    """The ``--model-set`` / ``--opt-set`` overrides in ``flags`` as ``{name: value}``."""
    settings = {}
    for option, value in itertools.pairwise(flags):
        if option in ("--model-set", "--opt-set"):
            key, _, val = value.partition("=")
            settings[key] = val
    return settings


def record_trial(
    registry_dir: Path, spec: ExperimentSpec, run: RunSpec, name: str, commit: str, comment_url: str
) -> Path:
    """Add ``name`` to the registry entry: a new trial for its variant (or the existing one), or the baseline."""
    entries = list((registry_dir / "experiments/belay/registry").glob(f"{spec.experiment}-*.yaml"))
    if len(entries) != 1:
        raise SystemExit(f"registry: expected one entry for {spec.experiment}, found {entries}")
    path = entries[0]
    entry = yaml.safe_load(path.read_text())
    trials = entry.setdefault("trials", [])
    if run.control:
        for trial in trials:
            if trial.get("commit") == commit[:10] and not trial.get("baseline"):
                trial["baseline"] = f"{name} (same commit, seed {run.seed})"
    else:
        trial = next((t for t in trials if t["variant"] == run.variant and t.get("commit") == commit[:10]), None)
        if trial is None:
            controls = [r for r in spec.runs.values() if r.control and r.seed == run.seed]
            trial = {
                "id": f"t{len(trials) + 1:03d}",
                "variant": run.variant,
                "rung": run.size,
                "protocol": spec.protocol.value,
                "batch": spec.experiment,
                "date": datetime.date.today().isoformat(),
                "comment_url": comment_url,
                "baseline": f"{run_name(spec.experiment, controls[0].name)} (same commit)" if controls else None,
                "wandb": [],
                "settings": settings_of(run.flags),
                "commit": commit[:10],
            }
            trial = {k: v for k, v in trial.items() if v is not None}
            trials.append(trial)
        if name not in trial["wandb"]:
            trial["wandb"].append(name)
        entry["status"] = "running"
    path.write_text(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True, width=120))
    return path


def commit_registry(registry_dir: Path, path: Path, message: str) -> None:
    subprocess.run([sys.executable, "-m", "experiments.belay.registry", "validate"], cwd=registry_dir, check=True)
    _git(registry_dir, "add", str(path))
    if not _git(registry_dir, "status", "--porcelain", str(path)):
        return  # e.g. a control run, which the variant's trial already names as its baseline
    _git(registry_dir, "commit", "--quiet", "-m", message)
    branch = _git(registry_dir, "rev-parse", "--abbrev-ref", "HEAD")
    _git(registry_dir, "push", "--quiet", "origin", branch)


def submit_runs(
    repo: Path, spec: ExperimentSpec, runs: list[str], commit: str, comment_url: str, *, dry_run: bool
) -> list[str]:
    """Check every launch rule for ``runs`` of ``spec``, then submit each and record it in the registry.

    All checks run before the first submit, so a refusal never leaves half an experiment launched. With
    ``dry_run`` the checks still run (read-only: git, Iris) and the commands are printed, not run.
    """
    branch = require_week_branch(repo, datetime.date.today())
    full = require_pushed_head(repo, commit, branch)
    require_registry_checkout(CONFIG.registry_checkout)
    unknown = [r for r in runs if r not in spec.runs]
    if unknown:
        raise SystemExit(f"{spec.experiment} has no run(s) {unknown}; runs: {sorted(spec.runs)}")
    names = {r: run_name(spec.experiment, r) for r in runs}
    jobs = iris_jobs(belay_job_prefix())
    used = [n for n in names.values() if any(job_id.endswith(f"/{n}-coord") for job_id, _, _ in jobs)]
    if used:
        raise SystemExit(f"refusing: {used} already ran on Iris; a rerun needs a new run key in the spec")
    in_use = nodes_in_use(jobs)
    needed = sum(spec.runs[r].nodes for r in runs)
    if in_use + needed > CONFIG.node_cap:
        raise SystemExit(
            f"held: soft cap {CONFIG.node_cap} nodes ({in_use} in use, {spec.experiment} needs {needed}).\n"
            "next: stop here; the idea stays approved and a later session launches it"
        )
    print(f"checks passed: branch {branch} at {full[:10]}, {in_use}/{CONFIG.node_cap} nodes in use, needs {needed}")
    for run in runs:
        command = launch_command(spec, spec.runs[run], names[run])
        print(f"{'would launch' if dry_run else 'launching'} {names[run]}:\n  {' '.join(command)}")
        if dry_run:
            continue
        subprocess.run(command, cwd=repo, check=True)
        path = record_trial(CONFIG.registry_checkout, spec, spec.runs[run], names[run], full, comment_url)
        commit_registry(CONFIG.registry_checkout, path, f"Belay registry: {spec.experiment} {run} submitted")
        print(f"{names[run]}: submitted; recorded in {path.name}")
    return list(names.values())


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment_id")
    parser.add_argument("run")
    parser.add_argument("--commit", required=True, help="the pushed SHA to run; must be the checked-out HEAD")
    parser.add_argument("--comment-url", required=True, help="where the plan was posted")
    parser.add_argument("--dry-run", action="store_true", help="run every check and print the launch, submit nothing")
    args = parser.parse_args(argv)
    repo = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    spec = load_spec(args.experiment_id)
    submit_runs(repo, spec, [args.run], args.commit, args.comment_url, dry_run=args.dry_run)


if __name__ == "__main__":
    main(sys.argv[1:])
