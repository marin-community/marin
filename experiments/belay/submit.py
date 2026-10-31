# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch one Marin Belay run from its committed spec, ``experiments/belay/runs/<experiment_id>.yaml``.

A run is fully specified by a pushed commit plus a run name, so every Belay run is reviewable in the diff:

    uv run python -m experiments.belay.submit X0221 seed0 --commit <sha> --registry-dir ~/marin_wt_registry_demo

The helper refuses a commit that is not the clean, pushed HEAD of a Belay branch, applies the soft node cap
(Belay's running and pending runs plus this one must fit), names the run ``belay-<id>-<run>``, writes its outputs
under ``tmp/ttl=30d/belay/runs/``, submits at batch priority by default, and records the commit and run name as a
trial in the registry entry (committed and pushed on the registry branch). Control runs are recorded as the trial's
baseline rather than as trials of their own.
"""

import argparse
import datetime
import fnmatch
import getpass
import itertools
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml

RUNS_DIR = Path(__file__).parent / "runs"
CLUSTER = "cw-us-east-02a"
RUN_PREFIX = "belay-"
OUTPUT_ROOT = "tmp/ttl=30d/belay/runs"
DEFAULT_NODE_CAP = 8
DEFAULT_BAND = "batch"
BANDS = ("batch", "interactive", "production")
# A Belay run must execute code that lives on a Belay experiment or checkpoint branch.
ALLOWED_BRANCHES = ("marin_belay_v1_w*", "marin_belay_v1_ckpt*", "marin_belay_v1_loomdemo*")
REGISTRY_BRANCHES = ("marin_belay_v1_registry*",)
# 8xH100 nodes per training run, by rung.
NODES_BY_SIZE = {"d512": 1, "d768": 1, "d1024": 1, "d1280": 1, "d1280L16": 4}
# Fixed launcher modules a spec may name; anything else is a code change to review here.
LAUNCHERS = {
    "ckpt1": "experiments.grug.fast_track.belay_ckpt1",
    "s7": "experiments.grug.fast_track.belay_s7",
    "baseline": "experiments.grug.fast_track.launch",
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


def require_pushed_head(repo: Path, commit: str, allowed: tuple[str, ...]) -> str:
    """The full SHA of ``commit`` after checking it is the clean HEAD of ``repo`` and on an allowed remote branch."""
    head = _git(repo, "rev-parse", "HEAD")
    full = _git(repo, "rev-parse", commit)
    if full != head:
        raise SystemExit(f"refusing: --commit {commit} is not the checked-out HEAD ({head[:10]})")
    dirty = [line for line in _git(repo, "status", "--porcelain", "--untracked-files=no").splitlines() if line]
    if dirty:
        raise SystemExit(f"refusing: tracked files differ from HEAD: {dirty}")
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    _git(repo, "fetch", "--quiet", "origin", branch)
    remote = [b.strip().removeprefix("origin/") for b in _git(repo, "branch", "-r", "--contains", full).splitlines()]
    if not any(fnmatch.fnmatch(b, pattern) for b in remote for pattern in allowed):
        raise SystemExit(f"refusing: {full[:10]} is not pushed to a branch matching {allowed} (on: {remote})")
    return full


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


def launch_command(spec: ExperimentSpec, run: RunSpec, name: str, cluster: str, band: str) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "-m",
        LAUNCHERS[spec.launcher],
        "--size",
        run.size,
        *run.flags,
        "--seed",
        str(run.seed),
        "--run-id",
        name,
        "--output-path",
        f"{OUTPUT_ROOT}/{name}",
        "--submit",
        "--target-cluster",
        cluster,
        "--priority",
        band,
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
    _git(registry_dir, "commit", "--quiet", "-m", message)
    branch = _git(registry_dir, "rev-parse", "--abbrev-ref", "HEAD")
    _git(registry_dir, "push", "--quiet", "origin", branch)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment_id")
    parser.add_argument("run")
    parser.add_argument("--commit", required=True, help="the pushed SHA to run; must be the checked-out HEAD")
    parser.add_argument("--registry-dir", type=Path, required=True, help="checkout of the registry branch")
    parser.add_argument("--comment-url", required=True, help="where the batch plan was posted")
    parser.add_argument("--band", choices=BANDS, default=DEFAULT_BAND)
    parser.add_argument("--cluster", default=CLUSTER)
    parser.add_argument("--node-cap", type=int, default=DEFAULT_NODE_CAP)
    parser.add_argument("--dry-run", action="store_true", help="check and print the launch without submitting")
    args = parser.parse_args(argv)

    repo = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    commit = require_pushed_head(repo, args.commit, ALLOWED_BRANCHES)
    registry_branch = _git(args.registry_dir, "rev-parse", "--abbrev-ref", "HEAD")
    if not any(fnmatch.fnmatch(registry_branch, p) for p in REGISTRY_BRANCHES):
        raise SystemExit(f"refusing: {args.registry_dir} is on {registry_branch}, not a registry branch")
    spec = load_spec(args.experiment_id)
    if args.run not in spec.runs:
        raise SystemExit(f"{args.experiment_id} has no run {args.run!r}; runs: {sorted(spec.runs)}")
    run = spec.runs[args.run]
    name = run_name(args.experiment_id, args.run)

    jobs = iris_jobs(belay_job_prefix())
    if any(job_id.endswith(f"/{name}-coord") for job_id, _, _ in jobs):
        raise SystemExit(f"refusing: {name} already ran; give the rerun a new run name (e.g. {args.run}-r)")
    in_use = nodes_in_use(jobs)
    if in_use + run.nodes > args.node_cap:
        raise SystemExit(
            f"held: soft cap {args.node_cap} nodes ({in_use} in use, {name} needs {run.nodes}); "
            "wait for a run to finish"
        )

    command = launch_command(spec, run, name, args.cluster, args.band)
    print(f"{name}: {in_use}/{args.node_cap} nodes in use; launching at {commit[:10]}:\n  {' '.join(command)}")
    if args.dry_run:
        return
    subprocess.run(command, cwd=repo, check=True)
    path = record_trial(args.registry_dir, spec, run, name, commit, args.comment_url)
    commit_registry(args.registry_dir, path, f"Belay registry: {args.experiment_id} {args.run} submitted ({name})")
    print(f"{name}: submitted; recorded in {path.name}")


if __name__ == "__main__":
    main(sys.argv[1:])
