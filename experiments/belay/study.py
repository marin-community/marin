# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Studies on the ``custom`` track: a question answered by an analysis of trained checkpoints, with no verdict.

A study spec (``experiments/belay/runs/X####.yaml``, ``kind: study``) lists runs, each launched through an existing
Track's command (``B tracks``) with its checkpoints under ``tmp/ttl=30d/belay/studies/X####/``, and analysis steps,
each a script ``experiments/belay/studies/X####.py`` run on one run's checkpoint by ``B analyze``. The maintainer's
approval sets the budget (nodes and hours); a study whose runs could exceed it is refused. New model code, production
priority and the hero rack are not studies: those go to a maintainer.

    experiment: X0300
    kind: study
    track: custom
    question: Do routed experts specialize by token type at d512?
    runs:
      base:
        track: speedrun          # launched with this Track's command
        seed: 0
        variant: speedrun recipe, checkpoints kept
        flags: []                # extra launcher flags; --model-set/--opt-set must name existing fields
        hours: 0.5               # the session's estimate of the run's wall time
    analysis:
      - script: experiments/belay/studies/X0300.py
        input: base              # the run whose final checkpoint it reads
"""

import re
import shlex
from dataclasses import dataclass, field

import yaml

from experiments.belay.config import CONFIG, GB200_CLUSTER, TRACKS, Track
from experiments.belay.posts import BOT_MARK, title
from experiments.belay.submit import SETTING_OPTIONS, check_setting, run_name, track_command

STUDY_KIND = "study"
STUDY_ROOT = "tmp/ttl=30d/belay/studies"
STUDIES_DIR = "experiments/belay/studies"
# Launcher flags Belay sets itself (name, seed, output, cluster, priority), and the ones that would put checkpoints
# outside the study's prefix: a study may not pass them.
RESERVED_FLAGS = frozenset(
    {"--run-id", "--seed", "--output-path", "--submit", "--target-cluster", "--priority", "--temporary-checkpoints-only"}
)
RESUME_FLAG = "--resume-from"
SAVE_CHECKPOINTS = "--save-checkpoints"


class StudyStatus:
    """A study's thread title status, in order. ``report`` is final: a study has no verdict."""

    PLAN = "study plan"
    RUNNING = "study running"
    ANALYSIS = "study analysis"
    REPORT = "report"


@dataclass(frozen=True)
class Budget:
    """What the maintainer approved: at most ``nodes`` nodes at once and each run at most ``hours`` long. A study
    launches its runs together, so it never uses more than ``nodes`` x ``hours`` node-hours."""

    nodes: int
    hours: float


@dataclass(frozen=True)
class StudyRun:
    name: str
    track: Track
    seed: int
    variant: str
    hours: float
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Analysis:
    script: str
    input: str
    """The key of the run whose checkpoint the script reads."""


@dataclass(frozen=True)
class Study:
    experiment: str
    question: str
    runs: dict[str, StudyRun]
    analysis: list[Analysis] = field(default_factory=list)


def is_study(text: str) -> bool:
    return yaml.safe_load(text).get("kind") == STUDY_KIND


def parse_study(text: str) -> Study:
    data = yaml.safe_load(text)
    if data.get("kind") != STUDY_KIND or data.get("track") != Track.CUSTOM.value:
        raise SystemExit("refusing: a study spec has kind: study and track: custom")
    runs = {
        key: StudyRun(
            name=key,
            track=Track(run["track"]),
            seed=int(run["seed"]),
            variant=str(run["variant"]),
            hours=float(run["hours"]),
            flags=tuple(str(f) for f in run.get("flags") or ()),
        )
        for key, run in data["runs"].items()
    }
    analysis = [Analysis(str(a["script"]), str(a["input"])) for a in data.get("analysis") or ()]
    return Study(data["experiment"], str(data["question"]), runs, analysis)


def study_prefix(experiment: str) -> str:
    """Where every checkpoint and report of the study lives, below the cluster's bucket."""
    return f"{STUDY_ROOT}/{experiment}"


def script_path(experiment: str) -> str:
    """The one analysis script a study may add (through ``code-change``)."""
    return f"{STUDIES_DIR}/{experiment}.py"


def escalation(experiment: str, reason: str) -> SystemExit:
    """A refusal whose next step is section 6: a maintainer decides."""
    return SystemExit(
        f"refusing: {reason}\n"
        f'next: uv run python -m experiments.belay escalate --registry-id {experiment} --title "{experiment}: '
        f'{reason}" --body "The study needs a maintainer."'
    )


def refuse_run(experiment: str, run: StudyRun) -> None:
    """Refuse a run a session may not launch for a study: no launch recipe, a maintainer-approved Track, production
    priority or the hero rack, a flag Belay sets itself, a checkpoint outside the study's prefix, or a setting no
    config field takes (which would need new model code)."""
    if run.track is Track.CUSTOM:
        raise escalation(experiment, f"run {run.name} names track custom; name the Track whose command it runs")
    baseline = TRACKS[run.track]
    if baseline.needs_go_ahead or baseline.band == "production" or baseline.cluster == GB200_CLUSTER:
        raise escalation(experiment, f"run {run.name}: track {run.track} is production priority or the hero rack")
    if baseline.launch is None or not baseline.output_path:
        raise escalation(experiment, f"run {run.name}: track {run.track} cannot place a study's checkpoints")
    if run.hours <= 0:
        raise SystemExit(f"refusing: run {run.name} needs a positive hours estimate")
    flags = list(run.flags)
    for i, flag in enumerate(flags):
        option = flag.split("=", 1)[0]
        if option in RESERVED_FLAGS:
            raise SystemExit(f"refusing: run {run.name} passes {option}; Belay sets it (checkpoints stay in the study)")
        if option == RESUME_FLAG:
            value = flag.split("=", 1)[1] if "=" in flag else (flags[i + 1] if i + 1 < len(flags) else "")
            refuse_input(experiment, value)
        if option in SETTING_OPTIONS:
            if i + 1 >= len(flags):
                raise SystemExit(f"refusing: run {run.name} ends with {option} and no setting")
            try:
                check_setting(option, flags[i + 1])
            except SystemExit as error:
                raise escalation(experiment, f"run {run.name} needs new model code ({error})") from None


def refuse_over_budget(study: Study, budget: Budget) -> None:
    """Refuse a study whose runs, launched together, could pass the approved budget."""
    nodes = sum(TRACKS[r.track].nodes for r in study.runs.values())
    longest = max(study.runs.values(), key=lambda r: r.hours)
    if nodes > budget.nodes:
        raise escalation(study.experiment, f"its runs need {nodes} nodes at once; the budget is {budget.nodes}")
    if longest.hours > budget.hours:
        raise escalation(study.experiment, f"run {longest.name} is {longest.hours} h; the budget is {budget.hours} h")


def check_study(study: Study, budget: Budget | None) -> None:
    """Every check before a study's spec is committed: a budget, each run, the budget, and its analysis steps."""
    if budget is None:
        raise escalation(study.experiment, "the approval set no study budget (nodes and hours)")
    if not study.runs:
        raise SystemExit("refusing: a study needs at least one run")
    for run in study.runs.values():
        refuse_run(study.experiment, run)
    refuse_over_budget(study, budget)
    for step in study.analysis:
        if step.script != script_path(study.experiment):
            raise SystemExit(f"refusing: the analysis script is {script_path(study.experiment)}, not {step.script}")
        if step.input not in study.runs:
            raise SystemExit(f"refusing: analysis input {step.input!r} is not a run; runs: {', '.join(study.runs)}")


def study_command(study: Study, run: StudyRun) -> list[str]:
    """The run's launch: its Track's command, with checkpoints saved under the study's prefix."""
    name = run_name(study.experiment, run.name)
    flags = (*run.flags, SAVE_CHECKPOINTS)
    return track_command(run.track, name, flags, seed=run.seed, output_root=study_prefix(study.experiment))


def checkpoint_root(study: Study, run: StudyRun) -> str:
    """Where the run's saved checkpoints land (``<step-N>`` directories below it)."""
    bucket = f"s3://marin-{TRACKS[run.track].cluster.removeprefix('cw-')}"
    return f"{bucket}/{study_prefix(study.experiment)}/{run_name(study.experiment, run.name)}/checkpoints"


def refuse_input(experiment: str, path: str) -> None:
    """Refuse a checkpoint path outside ``s3://<bucket>/tmp/ttl=30d/belay/studies/<experiment>/``."""
    pattern = rf"s3://marin-[a-z0-9-]+/{re.escape(study_prefix(experiment))}/\S+"
    if not re.fullmatch(pattern, path) or ".." in path.split("/"):
        raise SystemExit(f"refusing: {path!r} is not under s3://<bucket>/{study_prefix(experiment)}/")


def run_rows(study: Study) -> list[str]:
    lines = ["| Run | Track | Hardware | Nodes | Est. hours | Checkpoints |", "|---|---|---|---|---|---|"]
    for run in study.runs.values():
        baseline = TRACKS[run.track]
        name = run_name(study.experiment, run.name)
        link = f"[{name}](https://wandb.ai/{CONFIG.wandb_project}/runs/{name})"
        lines.append(
            f"| {link} | {run.track} | {baseline.hardware} | {baseline.nodes} | {run.hours:g} | "
            f"`{checkpoint_root(study, run)}` |"
        )
    return lines


def study_post(
    study: Study, name: str, status: str, budget: Budget, commit: str, report_url: str | None = None
) -> tuple[str, str]:
    """The study's thread title and body: the question, the runs with their checkpoints, the budget, the analysis
    steps and, once written, the report."""
    heading = title(study.experiment, name, Track.CUSTOM, status)
    node_hours = sum(TRACKS[r.track].nodes * r.hours for r in study.runs.values())
    spec = f"https://github.com/{CONFIG.repository}/blob/{commit}/experiments/belay/runs/{study.experiment}.yaml"
    parts = [
        f"{BOT_MARK} **{heading}**",
        f"**Question.** {study.question}",
        "\n".join(run_rows(study)),
        f"Budget: {budget.nodes} nodes, {budget.hours:g} h; planned {node_hours:g} of "
        f"{budget.nodes * budget.hours:g} node-hours. Spec: [{study.experiment}.yaml]({spec})",
    ]
    for step in study.analysis:
        command = shlex.join(
            ["uv", "run", "python", "-m", "experiments.belay", "analyze", study.experiment, "--script", step.script]
        )
        parts.append(f"Analysis: `{step.script}` on {run_name(study.experiment, step.input)} (`{command} ...`)")
    if report_url is not None:
        parts.append(f"Report: [{study.experiment} report]({report_url}). A study has no verdict.")
    return heading, "\n\n".join(parts)
