# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run specs and launches: ``experiments/belay/runs/X####.yaml`` names each run, and a pushed commit runs it.

A spec names its Track (``config.TRACKS``) and has one variant run and one same-commit control at ``CONFIG.seed``,
both launched with the Track's baseline command. Runs are named ``belay-<id>-<run>``, write under
``CONFIG.output_root``, and run on the Track's cluster at its priority.
"""

import dataclasses
import itertools
import math
import re
import shlex
import subprocess
import sys
import typing
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml
from iris.cluster.client.job_info import resolve_job_user

from experiments.belay.config import CONFIG, PRIORITY_BANDS, TRACKS, Track
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
# Iris's refusal of a band above the submitting user's max_band (iris.cluster.controller.jobs).
BAND_REFUSAL = re.compile(r"Budget identity (\S+) cannot submit (\w+) jobs \(max band: (\w+)\)")


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
# An approver's instructions name a Track anywhere in the text ("... Track: ladder ...").
TRACK_MENTION = re.compile(r"\btrack\W*:\W*([A-Za-z0-9-]+)", re.IGNORECASE)
DEFAULT_TRACK = Track.SPEEDRUN
LADDER = "ladder"
"""A ``Track:`` value meaning the competitive ladder: one trial per rung, launched together."""
LADDER_RUNGS = {
    Track.COMPETITIVE_D512: "d512",
    Track.COMPETITIVE_D768: "d768",
    Track.COMPETITIVE_D1024: "d1024",
    Track.COMPETITIVE_D1280: "d1280",
}
"""Each ladder rung's Track and the tag its experiment id carries (``X0037d768``), so each rung has its own spec
and run names. The app records a rung's run under its idea's registry id and keeps the tagged id as its experiment."""
REGISTRY_ID = re.compile(r"X\d{4}")


@dataclass(frozen=True)
class HeroMfuResult:
    trial_id: str
    step_time_change: float
    """The variant's hero-scale s/step relative to its control: +0.05 is 5% slower."""


def hero_mfu_result(entry: dict) -> HeroMfuResult | None:
    """The entry's most recent hero-mfu trial with a result, else None; refuses a result that is not a number."""
    trials = [t for t in entry.get("trials") or [] if t.get("rung") == Track.HERO_MFU.value and t.get("delta")]
    if not trials:
        return None
    trial = trials[-1]
    change = float(trial["delta"][0])
    if not math.isfinite(change):
        raise SystemExit(f"refusing: hero-mfu trial {trial['id']} has no step time result ({change})")
    return HeroMfuResult(trial["id"], change)


def hero_mfu_steps(track: Track, hero: HeroMfuResult) -> int:
    """A ladder rung's steps at equal compute with its baseline at hero-scale MFU: a variant 5% slower per step
    at hero scale trains 1/1.05 of the baseline's steps."""
    assert track in LADDER_RUNGS, track
    steps = TRACKS[track].baseline_steps
    assert steps is not None, track
    return round(steps / (1 + hero.step_time_change))


def tracks_of_comment(text: str, instructions: str | None = None) -> tuple[tuple[Track, ...], bool]:
    """The Tracks an idea runs on, and whether they were defaulted (no Track named: ``speedrun``). An approver's
    instructions win over the comment's ``Track:`` line; ``ladder`` means every competitive rung."""
    match = TRACK_MENTION.search(instructions or "") or TRACK_LINE.search(text)
    if match is None:
        return (DEFAULT_TRACK,), True
    if match.group(1).lower() == LADDER:
        return tuple(LADDER_RUNGS), False
    return (parse_track(match.group(1)),), False


def previous_tracks(entry: dict) -> tuple[Track, ...] | None:
    """The Tracks of an entry's last attempt, for a rerun that names none: every competitive rung when that attempt
    (the trials dated like its last one) ran more than one, else the last trial's Track; with no trial on a current
    Track, the entry's own Track (a study's ``custom``). None when there is neither."""
    tracks = {t.value for t in Track}
    trials = [t for t in entry.get("trials") or [] if t.get("rung") in tracks]
    if trials:
        last = trials[-1]
        rungs = {t["rung"] for t in trials if t.get("date") == last.get("date")} & {r.value for r in LADDER_RUNGS}
        return tuple(LADDER_RUNGS) if len(rungs) > 1 else (Track(last["rung"]),)
    track = entry.get("track")
    return (Track(track),) if track in tracks else None


def experiment_id(registry_id: str, track: Track, ladder: bool) -> str:
    """The id of an idea's spec and runs on ``track``: the registry id, plus the rung tag on a ladder."""
    return f"{registry_id}{LADDER_RUNGS[track]}" if ladder else registry_id


def registry_of(experiment: str) -> str:
    """The registry id an experiment id belongs to (``X0037d768`` -> ``X0037``)."""
    match = REGISTRY_ID.match(experiment)
    if match is None:
        raise SystemExit(f"refusing: {experiment!r} is not a Belay experiment id")
    return match.group(0)


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
    num_steps: int | None = None,
) -> str:
    """The run spec for one variant and its same-commit control on ``track``, after checking every setting.
    ``suffix`` (from ``run_suffix``) keeps run names unique when the id ran before; ``shared_control`` (from
    ``shareable_control``) reuses another spec's running control instead of launching one; ``num_steps`` sets the
    variant's steps (a hero-MFU-matched ladder rung's, from ``hero_mfu_steps``)."""
    check_variant(variant, settings)
    flags = [part for setting in settings for part in setting]
    if num_steps is not None:
        flags += ["--num-steps", str(num_steps)]
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
    """Every Iris job under a ``belay-`` prefix of Belay's Iris user, and of this process's user (a maintainer's own
    runs) when that differs."""
    jobs = []
    for user in dict.fromkeys((CONFIG.iris_user, resolve_job_user())):
        out = subprocess.run(
            [
                "uv",
                "run",
                "iris",
                "--cluster",
                "marin",
                "job",
                "list",
                "--prefix",
                f"/{user}/{RUN_PREFIX}",
                "--limit",
                "500",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        jobs += [
            IrisJob(m.group(1), m.group(2), m.group(3)) for line in out.splitlines() if (m := _JOB_LINE.match(line))
        ]
    return jobs


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


def idea_jobs(jobs: list[IrisJob], registry_id: str) -> list[IrisJob]:
    """The jobs of every run under ``registry_id``, its ladder rungs' included."""
    out = []
    for job in jobs:
        match = REGISTRY_ID.match(job.job_id.rsplit("/", 1)[1].removeprefix(RUN_PREFIX))
        if match is not None and match.group() == registry_id:
            out.append(job)
    return out


def refuse_over_cap(needed: int, jobs: list[IrisJob], cap: int, grant: int | None, experiment: str) -> None:
    """Hold a launch past the node cap: the global ``cap`` over every Belay run, or for an idea with a maintainer's
    ``grant``, the grant over that idea's own runs (every rung of a ladder)."""
    if grant is not None:
        own = nodes_in_use(idea_jobs(jobs, registry_of(experiment)))
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


def check_ladder_launch(tracks: list[Track], jobs: list[IrisJob], cap: int, grant: int | None, registry_id: str) -> None:
    """Hold a ladder before any rung launches when its rungs not yet launched (``tracks``) would together pass the
    node cap or the idea's grant: a ladder's rungs launch together, so the idea stays approved until all of them fit."""
    refuse_over_cap(sum(TRACKS[t].nodes for t in tracks), jobs, cap, grant, registry_id)


def track_command(
    track: Track,
    name: str,
    flags: tuple[str, ...] = (),
    python: str = "uv run python",
    seed: int = CONFIG.seed,
    output_root: str = CONFIG.output_root,
    band: str | None = None,
) -> list[str]:
    """The exact launch of one run on ``track``: its baseline command plus ``flags``, named ``name``, at ``seed``,
    writing under ``output_root`` (a study passes its own prefix), at Iris priority ``band`` (default: the Track's;
    ``launch_band`` resolves a grant)."""
    baseline = TRACKS[track]
    if baseline.launch is None:
        raise SystemExit(f"refusing: track {track} has no launch recipe; a maintainer runs it")
    command = [*python.split(), "-m", *baseline.launch, *flags]
    command += ["--seed", str(seed), "--run-id", name]
    if baseline.output_path:
        command += ["--output-path", f"{output_root}/{name}"]
    command += ["--submit", "--target-cluster", baseline.cluster, "--priority", band or baseline.band]
    return command


def launch_command(spec: ExperimentSpec, run: RunSpec, band: str | None = None) -> list[str]:
    """The run's command at ``band`` (default: the Track's), run from the checkout of the Track's code branch
    (``uv run`` uses that checkout's environment)."""
    return track_command(spec.track, run_name(spec.experiment, run.name), run.flags, seed=run.seed, band=band)


def launch_band(track: Track, grant: str | None, experiment: str) -> str:
    """The Iris band ``experiment``'s runs on ``track`` launch at: an approver's ``grant`` when set, else the Track's
    band. A grant only raises the band; one below the Track's is refused with an escalation."""
    band = TRACKS[track].band
    if grant is None:
        return band
    if grant not in PRIORITY_BANDS or band not in PRIORITY_BANDS:
        raise SystemExit(f"refusing: priority {grant!r} for {experiment} is not one of {', '.join(PRIORITY_BANDS)}")
    if PRIORITY_BANDS.index(grant) < PRIORITY_BANDS.index(band):
        raise SystemExit(
            f"refusing: {experiment}'s priority grant {grant} is below track {track}'s band {band}; "
            "a grant only raises it.\n"
            f'next: uv run python -m experiments.belay escalate --registry-id {experiment} --title "{experiment}: '
            f'priority grant {grant} is below the {track} band {band}" --body "Re-queue it with a higher band, or none."'
        )
    return grant


def submit_run(command: list[str], repo: Path, experiment: str) -> None:
    """Submit one run; when Iris refuses its band for the submitting user, escalate with Iris's reason instead of
    retrying lower."""
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode == 0:
        return
    print(result.stderr, end="", file=sys.stderr)
    refusal = BAND_REFUSAL.search(result.stdout + result.stderr)
    if refusal is None:
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
    user, band, allowed = refusal.groups()
    reason = f"priority {band.lower()} not allowed for user {user} (max band {allowed.lower()})"
    raise SystemExit(
        f"refusing: {reason}.\n"
        f'next: uv run python -m experiments.belay escalate --registry-id {experiment} --title "{experiment}: {reason}" '
        '--body "Grant the band in Iris user budgets, or lower the request (re-queue without a priority grant)."'
    )


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
