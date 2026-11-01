# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run specs and launches: ``experiments/belay/runs/X####.yaml`` names each run, and a pushed commit runs it.

A spec names its Track (``config.TRACKS``) and has one variant run and one same-commit control at ``CONFIG.seed``,
both launched with the Track's baseline command. Runs are named ``belay-<id>-<run>``, write under
``CONFIG.output_root``, and run on the Track's cluster at its priority.
"""

import dataclasses
import getpass
import re
import subprocess
import sys
import typing
from dataclasses import dataclass
from pathlib import Path

import yaml

from experiments.belay.config import CONFIG, TRACKS, Track
from experiments.grug.fast_track.launch import _parse_as
from experiments.grug.fast_track.model import GrugModelConfig
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

RUNS_DIR = Path(__file__).parent / "runs"
RUN_PREFIX = "belay-"
# Every setting a spec may change: launcher option -> the config class whose field it sets.
SETTING_OPTIONS = {"--model-set": GrugModelConfig, "--opt-set": GrugMoeMuonHConfig}
MAX_VARIANT = 60
ACTIVE_JOB_STATES = ("pending", "running")
_JOB_LINE = re.compile(r"^(/\S+)\s+(\w+)\s+(\S+)")


@dataclass(frozen=True)
class RunSpec:
    name: str
    seed: int
    variant: str
    flags: tuple[str, ...] = ()
    control: bool = False


@dataclass(frozen=True)
class ExperimentSpec:
    experiment: str
    track: Track
    runs: dict[str, RunSpec]


@dataclass(frozen=True)
class IrisJob:
    job_id: str
    state: str
    submitted_at: str


def run_name(experiment_id: str, run: str) -> str:
    return f"{RUN_PREFIX}{experiment_id}-{run}"


def spec_path(experiment_id: str, runs_dir: Path = RUNS_DIR) -> Path:
    return runs_dir / f"{experiment_id}.yaml"


def parse_spec(text: str) -> ExperimentSpec:
    data = yaml.safe_load(text)
    runs = {
        name: RunSpec(
            name=name,
            seed=int(run["seed"]),
            variant=run["variant"],
            flags=tuple(str(f) for f in run.get("flags", ())),
            control=bool(run.get("control", False)),
        )
        for name, run in data["runs"].items()
    }
    return ExperimentSpec(data["experiment"], parse_track(str(data.get("track"))), runs)


TRACK_LINE = re.compile(r"^\W*track\W*:\W*([A-Za-z0-9-]+)", re.IGNORECASE | re.MULTILINE)
DEFAULT_TRACK = Track.SPEEDRUN


def track_of_comment(text: str) -> tuple[Track, bool]:
    """The Track on a comment's ``Track:`` line, and whether it was defaulted (no such line: ``speedrun``)."""
    match = TRACK_LINE.search(text)
    if match is None:
        return DEFAULT_TRACK, True
    return parse_track(match.group(1)), False


def parse_track(text: str) -> Track:
    if text not in {t.value for t in Track}:
        raise SystemExit(f"refusing: track {text!r} is not one of {', '.join(t.value for t in Track)}")
    return Track(text)


def load_spec(experiment_id: str, runs_dir: Path = RUNS_DIR) -> ExperimentSpec:
    path = spec_path(experiment_id, runs_dir)
    if not path.exists():
        raise SystemExit(f"refusing: no run spec {path.name} on this branch")
    return parse_spec(path.read_text())


def check_setting(option: str, setting: str) -> None:
    """Refuse a ``--model-set``/``--opt-set`` value whose field does not exist or whose value does not parse."""
    cls = SETTING_OPTIONS[option]
    name, sep, value = setting.partition("=")
    if not sep:
        raise SystemExit(f"refusing: {option} {setting!r} is not name=value")
    if name not in {f.name for f in dataclasses.fields(cls)}:
        raise SystemExit(f"refusing: {cls.__name__} has no field {name!r}; the fields are listed by new-idea")
    _parse_as(typing.get_type_hints(cls)[name], value, name)


def setting_fields() -> list[str]:
    """One line per field a spec may set: ``--opt-set name: type = default``."""
    lines = []
    for option, cls in SETTING_OPTIONS.items():
        hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            if f.init and f.default is not dataclasses.MISSING:
                lines.append(f"{option} {f.name}: {getattr(hints[f.name], '__name__', hints[f.name])} = {f.default!r}")
    return lines


def check_variant(variant: str, settings: list[tuple[str, str]]) -> None:
    if not settings:
        raise SystemExit("refusing: give at least one --opt-set or --model-set setting")
    if len(variant) > MAX_VARIANT:
        raise SystemExit(f"refusing: --variant is longer than {MAX_VARIANT} characters")
    for option, setting in settings:
        check_setting(option, setting)


def run_suffix(experiment_id: str, jobs: list[IrisJob]) -> str:
    """A suffix for the spec's run keys so no run name repeats one that already ran on Iris ("" when none did)."""
    for suffix in ("", *(f"-{c}" for c in "bcdefgh")):
        names = [run_name(experiment_id, f"{key}{suffix}") for key in (f"seed{CONFIG.seed}", "control")]
        if not any(j.job_id.endswith(f"/{n}-coord") for j in jobs for n in names):
            return suffix
    raise SystemExit(f"refusing: {experiment_id} has run too many times; a maintainer decides")


def new_spec_text(
    experiment_id: str, track: Track, variant: str, settings: list[tuple[str, str]], suffix: str = ""
) -> str:
    """The run spec for one variant and its same-commit control on ``track``, after checking every setting.
    ``suffix`` (from ``run_suffix``) keeps run names unique when the id ran before."""
    check_variant(variant, settings)
    flags = [part for setting in settings for part in setting]
    variant_run = {"seed": CONFIG.seed, "variant": variant, "flags": flags}
    control_run = {"seed": CONFIG.seed, "variant": f"{track} control", "control": True}
    data = {
        "experiment": experiment_id,
        "track": track.value,
        "runs": {f"seed{CONFIG.seed}{suffix}": variant_run, f"control{suffix}": control_run},
    }
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120)


def iris_jobs() -> list[IrisJob]:
    """Every Iris job under this user's ``belay-`` prefix."""
    prefix = f"/{getpass.getuser()}/{RUN_PREFIX}"
    out = subprocess.run(
        ["uv", "run", "iris", "--cluster", "marin", "job", "list", "--prefix", prefix, "--limit", "500"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [IrisJob(m.group(1), m.group(2), m.group(3)) for line in out.splitlines() if (m := _JOB_LINE.match(line))]


def nodes_in_use(jobs: list[IrisJob], runs_dir: Path = RUNS_DIR) -> int:
    """Nodes held by active Belay runs: each coordinator job counts its Track's nodes (1 without a spec here)."""
    total = 0
    for job in jobs:
        if job.job_id.count("/") != 2 or job.state not in ACTIVE_JOB_STATES:
            continue
        experiment = job.job_id.rsplit("/", 1)[1].removeprefix(RUN_PREFIX).split("-")[0]
        path = spec_path(experiment, runs_dir)
        total += TRACKS[parse_spec(path.read_text()).track].nodes if path.exists() else 1
    return total


def refuse_reused_names(names: list[str], jobs: list[IrisJob]) -> None:
    used = [n for n in names if any(j.job_id.endswith(f"/{n}-coord") for j in jobs)]
    if used:
        raise SystemExit(f"refusing: {used} already ran on Iris; a rerun needs a new run key in the spec")


def refuse_over_cap(needed: int, jobs: list[IrisJob]) -> None:
    in_use = nodes_in_use(jobs)
    if in_use + needed > CONFIG.node_cap:
        raise SystemExit(
            f"held: soft cap {CONFIG.node_cap} nodes ({in_use} in use, {needed} needed).\n"
            "next: stop; the idea stays approved and a later session launches it"
        )


def track_command(track: Track, name: str, flags: tuple[str, ...] = (), python: str = sys.executable) -> list[str]:
    """The exact launch of one run on ``track``: its baseline command plus ``flags``, named ``name``."""
    baseline = TRACKS[track]
    if baseline.launch is None:
        raise SystemExit(f"refusing: track {track} has no launch recipe; a maintainer runs it")
    command = [python, "-m", *baseline.launch, *flags]
    command += ["--seed", str(CONFIG.seed), "--run-id", name]
    if baseline.output_path:
        command += ["--output-path", f"{CONFIG.output_root}/{name}"]
    command += ["--submit", "--target-cluster", baseline.cluster, "--priority", baseline.band]
    return command


def launch_command(spec: ExperimentSpec, run: RunSpec) -> list[str]:
    """The run's command; it runs with this interpreter from the checkout of the Track's code branch."""
    return track_command(spec.track, run_name(spec.experiment, run.name), run.flags)


def check_launch(spec: ExperimentSpec, jobs: list[IrisJob]) -> None:
    """Refuse a spec whose Track has no launch recipe, that reuses a run name, or that would pass the node cap."""
    baseline = TRACKS[spec.track]
    if baseline.launch is None:
        raise SystemExit(f"refusing: track {spec.track} has no launch recipe; a maintainer runs it")
    refuse_reused_names([run_name(spec.experiment, r) for r in spec.runs], jobs)
    refuse_over_cap(baseline.nodes * len(spec.runs), jobs)


def settings_of(flags: tuple[str, ...]) -> dict[str, str]:
    """The ``name=value`` settings in a run's flags, as ``{name: value}``."""
    settings = {}
    for flag in flags:
        if flag in SETTING_OPTIONS:
            continue
        name, _, value = flag.partition("=")
        settings[name] = value
    return settings
