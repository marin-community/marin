# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run specs and launches: ``experiments/belay/runs/X####.yaml`` names each run, and a pushed commit runs it.

A spec names its Track (``config.TRACKS``) and has one variant run and one same-commit control at ``CONFIG.seed``,
both launched with the Track's baseline command. Runs are named ``belay-<id>-<run>``, write under
``CONFIG.output_root``, and run on the Track's cluster at its priority.
"""

import dataclasses
import getpass
import itertools
import re
import shlex
import subprocess
import typing
from dataclasses import dataclass
from enum import StrEnum
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


class Stage(StrEnum):
    """A run's place in its Track's comparison, in order. Only ``EQUAL_DATA_THEN_TIME`` tracks go past ``DATA``."""

    DATA = "data"
    """The variant and its control at the baseline's steps (equal data)."""
    PROBE = "probe"
    """A short variant run that measures its s/step."""
    TIME = "time"
    """The variant at train_seconds / its s/step steps, against the stage-1 control (equal time)."""


STAGE_ORDER = (Stage.DATA, Stage.PROBE, Stage.TIME)
PROBE_STEPS = 150
PROBE_SKIP = 20  # compile and warm-up steps left out of the probe's s/step


@dataclass(frozen=True)
class RunSpec:
    name: str
    seed: int
    variant: str
    flags: tuple[str, ...] = ()
    control: bool = False
    stage: Stage = Stage.DATA
    shared: str | None = None
    """A control another spec launched at the same code and seed (its run name); this spec reuses that run."""


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


def run_id(spec: ExperimentSpec, run: RunSpec) -> str:
    """The run's Iris and W&B name: its own, or the shared control's."""
    return run.shared or run_name(spec.experiment, run.name)


def own_runs(runs: list[RunSpec]) -> list[RunSpec]:
    """The runs this spec launches itself (a shared control is launched by its owner)."""
    return [r for r in runs if r.shared is None]


def sharers(experiment_id: str, runs_dir: Path = RUNS_DIR) -> list[str]:
    """Experiments whose spec reuses one of ``experiment_id``'s controls."""
    prefix = run_name(experiment_id, "")
    out = []
    for path in sorted(runs_dir.glob("X*.yaml")):
        spec = parse_spec(path.read_text())
        if any((r.shared or "").startswith(prefix) for r in spec.runs.values()):
            out.append(spec.experiment)
    return out


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
            stage=Stage(run.get("stage", Stage.DATA)),
            shared=run.get("shared"),
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


def run_suffix(experiment_id: str, jobs: list[IrisJob], keys: tuple[str, ...] = ()) -> str:
    """A suffix for run keys (default: the stage-1 pair) so no run name repeats one that already ran on Iris."""
    keys = keys or (f"seed{CONFIG.seed}", "control")
    for suffix in ("", *(f"-{c}" for c in "bcdefgh")):
        names = [run_name(experiment_id, f"{key}{suffix}") for key in keys]
        if not any(j.job_id.endswith(f"/{n}-coord") for j in jobs for n in names):
            return suffix
    raise SystemExit(f"refusing: {experiment_id} has run too many times; a maintainer decides")


def new_spec_text(
    experiment_id: str,
    track: Track,
    variant: str,
    settings: list[tuple[str, str]],
    suffix: str = "",
    shared_control: str | None = None,
) -> str:
    """The run spec for one variant and its same-commit control on ``track``, after checking every setting.
    ``suffix`` (from ``run_suffix``) keeps run names unique when the id ran before; ``shared_control`` (from
    ``shareable_control``) reuses another spec's running control instead of launching one."""
    check_variant(variant, settings)
    flags = [part for setting in settings for part in setting]
    variant_run = {"seed": CONFIG.seed, "variant": variant, "flags": flags}
    control_run = {"seed": CONFIG.seed, "variant": f"{track} control", "control": True}
    if shared_control is not None:
        control_run["shared"] = shared_control
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


def refuse_over_cap(needed: int, jobs: list[IrisJob], cap: int, grant: int | None, experiment: str) -> None:
    """Hold a launch past the node cap: the global ``cap`` over every Belay run, or for an idea with a maintainer's
    ``grant``, the grant over that idea's own runs."""
    if grant is not None:
        own = nodes_in_use([j for j in jobs if f"/{run_name(experiment, '')}" in j.job_id])
        if own + needed <= grant:
            return
        limit, in_use = f"node grant {grant} for {experiment}", own
    else:
        limit, in_use = f"soft cap {cap} nodes", nodes_in_use(jobs)
        if in_use + needed <= cap:
            return
    raise SystemExit(
        f"held: {limit} ({in_use} in use, {needed} needed).\n"
        "next: stop; the idea stays approved and a later session launches it"
    )


def track_command(
    track: Track, name: str, flags: tuple[str, ...] = (), python: str = "uv run python", seed: int = CONFIG.seed
) -> list[str]:
    """The exact launch of one run on ``track``: its baseline command plus ``flags``, named ``name``, at ``seed``."""
    baseline = TRACKS[track]
    if baseline.launch is None:
        raise SystemExit(f"refusing: track {track} has no launch recipe; a maintainer runs it")
    command = [*python.split(), "-m", *baseline.launch, *flags]
    command += ["--seed", str(seed), "--run-id", name]
    if baseline.output_path:
        command += ["--output-path", f"{CONFIG.output_root}/{name}"]
    command += ["--submit", "--target-cluster", baseline.cluster, "--priority", baseline.band]
    return command


def launch_command(spec: ExperimentSpec, run: RunSpec) -> list[str]:
    """The run's command, run from the checkout of the Track's code branch (``uv run`` uses that checkout's
    environment)."""
    return track_command(spec.track, run_name(spec.experiment, run.name), run.flags, seed=run.seed)


def with_launched(text: str, launched: dict[str, list[str]]) -> str:
    """The spec ``text`` with each run's submitted command recorded under ``launched`` (a record for readers; the
    launcher never reads it back)."""
    data = yaml.safe_load(text)
    data.setdefault("launched", {}).update({key: shlex.join(command) for key, command in launched.items()})
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120)


def check_launch(spec: ExperimentSpec, jobs: list[IrisJob], cap: int, grant: int | None = None) -> None:
    """Refuse a spec whose Track has no launch recipe or that reuses a run name; hold one that would pass the node
    cap (``cap``, from the app) or the idea's node ``grant``."""
    baseline = TRACKS[spec.track]
    if baseline.launch is None:
        raise SystemExit(f"refusing: track {spec.track} has no launch recipe; a maintainer runs it")
    own = own_runs(list(spec.runs.values()))
    refuse_reused_names([run_name(spec.experiment, r.name) for r in own], jobs)
    refuse_over_cap(baseline.nodes * len(own), jobs, cap, grant, spec.experiment)


def settings_of(flags: tuple[str, ...]) -> dict[str, str]:
    """The ``--model-set`` / ``--opt-set`` settings in a run's flags, as ``{name: value}``."""
    settings = {}
    for option, value in itertools.pairwise(flags):
        if option in SETTING_OPTIONS:
            name, _, setting = value.partition("=")
            settings[name] = setting
    return settings


def variant_settings(spec: ExperimentSpec) -> tuple[str, ...]:
    """The stage-1 variant's ``--model-set`` / ``--opt-set`` flags, which every later stage's variant repeats."""
    variant = next(r for r in spec.runs.values() if r.stage is Stage.DATA and not r.control)
    out: list[str] = []
    for option, value in itertools.pairwise(variant.flags):
        if option in SETTING_OPTIONS:
            out += [option, value]
    return tuple(out)


def latest_stage(spec: ExperimentSpec) -> Stage:
    return max((r.stage for r in spec.runs.values()), key=STAGE_ORDER.index)


def stage_runs(spec: ExperimentSpec, stage: Stage) -> list[RunSpec]:
    return [r for r in spec.runs.values() if r.stage is stage]


def spec_with_runs(text: str, runs: dict[str, dict]) -> str:
    """The spec ``text`` with ``runs`` (key -> run fields) added, for the next stage."""
    data = yaml.safe_load(text)
    data["runs"].update(runs)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120)


def next_stage_runs(spec: ExperimentSpec, stage: Stage, suffix: str, steps: int | None = None) -> dict[str, dict]:
    """The runs that start ``stage`` (PROBE, or TIME at ``steps``), keyed like the spec."""
    variant = next(r for r in spec.runs.values() if r.stage is Stage.DATA and not r.control)
    flags = list(variant_settings(spec))
    if stage is Stage.PROBE:
        probe_flags = [*flags, "--num-steps", str(PROBE_STEPS)]
        probe = {"seed": variant.seed, "variant": f"{variant.variant} (timing probe)", "flags": probe_flags}
        return {f"probe{suffix}": {**probe, "stage": "probe"}}
    assert steps is not None
    timed_flags = [*flags, "--num-steps", str(steps)]
    timed = {"seed": variant.seed, "variant": variant.variant, "flags": timed_flags, "stage": "time"}
    return {f"seed{variant.seed}-t{suffix}": timed}
