# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Marin Belay experiment registry: one YAML file per experiment in ``experiments/belay/registry/``.

Every idea that enters Marin Belay (marin#9451) gets a permanent id, ``X0001``, ``X0002``, ..., assigned when it is
approved on the intake issue (marin#9514) or when the Belay agent starts it. Its file, ``<id>-<slug>.yaml``, records
what the idea is, where it came from, every trial run of it (rung, protocol, W&B runs, the code commit and flags,
results) and the decisions taken on it (adopted, removed, closed). The registry is the source of truth for the
Results section of the Belay page, built with ``build``.

    uv run python -m experiments.belay.registry validate
    uv run python -m experiments.belay.registry build > results.json
    uv run python -m experiments.belay.registry new --slug sqrt-softplus-router --name "..." --category "MoE and routing"
"""

import argparse
import dataclasses
import json
import re
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml

REGISTRY_DIR = Path(__file__).parent / "registry"
ID_RE = re.compile(r"^X\d{4}$")
SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
# A trial id is local to its experiment; elsewhere a trial is cited as "<experiment id>.<trial id>", e.g. X0014.t005.
TRIAL_ID_RE = re.compile(r"^t\d{3}$")
# A git commit SHA, full or abbreviated to at least 10 characters.
COMMIT_RE = re.compile(r"^[0-9a-f]{10,40}$")
# One list shared with the intake bot (scripts/ci/belay_intake.py), which reads it from this branch.
CATEGORIES: tuple[str, ...] = tuple(json.loads((REGISTRY_DIR / "categories.json").read_text()))


class Status(StrEnum):
    """Where an experiment is in its lifecycle."""

    APPROVED = "approved"
    """Accepted for testing (intake approval or agent proposal); no trial yet."""
    RUNNING = "running"
    """At least one trial is in flight."""
    DONE = "done"
    """Trials finished with no decision recorded yet."""
    ADOPTED = "adopted"
    """In the current recipe."""
    REMOVED = "removed"
    """Was adopted, later taken out."""
    CLOSED = "closed"
    """Decided against (lost, neutral, or not worth its cost)."""


class Source(StrEnum):
    INTAKE = "intake"
    """Approved on the intake issue."""
    MAINTAINER = "maintainer"
    AGENT = "agent"
    COMMUNITY = "community"
    BASELINE = "baseline"


class EventType(StrEnum):
    ADOPTED = "adopted"
    REMOVED = "removed"
    CLOSED = "closed"


@dataclass(frozen=True)
class Paper:
    """A source paper: its arXiv id, its title, or both."""

    arxiv: str | None = None
    title: str | None = None


@dataclass(frozen=True)
class Intake:
    """The approved intake comment an experiment came from."""

    url: str
    author: str
    approved_by: str


@dataclass(frozen=True)
class Trial:
    """One measured comparison: a variant at a rung against a baseline, possibly over several seeds."""

    id: str
    variant: str
    rung: str
    protocol: str
    batch: str
    date: str
    comment_url: str
    baseline: str | None = None
    steps: int | None = None
    delta: tuple[float, ...] = ()
    loss: tuple[float, ...] = ()
    wandb: tuple[str, ...] = ()
    settings: dict[str, str] = field(default_factory=dict)
    commit: str | None = None
    """The commit the trial ran, on GitHub."""
    step_time_change: str | None = None
    tuned: str | None = None
    note: str | None = None
    derived: bool | None = None
    """The delta is derived from other trials rather than measured against its own control."""


@dataclass(frozen=True)
class Event:
    type: EventType
    batch: str
    date: str
    comment_url: str
    note: str | None = None
    candidate: str | None = None
    as_adopted: str | None = None


@dataclass(frozen=True)
class Experiment:
    id: str
    slug: str
    name: str
    category: str
    summary: str
    source: Source
    status: Status
    first_batch: int | None = None
    tags: tuple[str, ...] = ()
    papers: tuple[Paper, ...] = ()
    details: tuple[str, ...] = ()
    knobs: tuple[str, ...] = ()
    intake: Intake | None = None
    trials: tuple[Trial, ...] = ()
    events: tuple[Event, ...] = ()

    @property
    def filename(self) -> str:
        return f"{self.id}-{self.slug}.yaml"


def _build(cls, data: dict):
    """``cls`` from a YAML mapping, rejecting unknown and missing keys and converting nested records, enums and
    sequences."""
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    nested = {"papers": Paper, "trials": Trial, "events": Event}
    kwargs = {}
    for key, value in data.items():
        if value is None:
            continue
        if key in nested:
            kwargs[key] = tuple(_build(nested[key], item) for item in value)
        elif key == "intake":
            kwargs[key] = None if value is None else _build(Intake, value)
        elif key in ("status", "source", "type"):
            kwargs[key] = {"status": Status, "source": Source, "type": EventType}[key](value)
        elif isinstance(value, list):
            kwargs[key] = tuple(value)
        elif key == "settings":
            kwargs[key] = {str(k): str(v) for k, v in value.items()}
        else:
            kwargs[key] = value
    required = {
        name
        for name, f in fields.items()
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    }
    missing = required - set(kwargs)
    if missing:
        raise ValueError(f"{cls.__name__}: missing keys {sorted(missing)}")
    return cls(**kwargs)


def _plain(value):
    """A dataclass tree as plain YAML/JSON data, leaving out empty and unset fields."""
    if dataclasses.is_dataclass(value):
        out = {}
        for f in dataclasses.fields(value):
            v = getattr(value, f.name)
            if v is None or v == () or v == {}:
                continue
            out[f.name] = _plain(v)
        return out
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def load_experiment(path: Path) -> Experiment:
    with open(path) as f:
        data = yaml.safe_load(f)
    try:
        experiment = _build(Experiment, data)
    except ValueError as e:
        raise ValueError(f"{path.name}: {e}") from e
    if path.name != experiment.filename:
        raise ValueError(f"{path.name}: file must be named {experiment.filename}")
    return experiment


def load_registry(directory: Path = REGISTRY_DIR) -> list[Experiment]:
    """Every experiment in ``directory``, in id order, validated."""
    experiments = sorted((load_experiment(p) for p in directory.glob("X*.yaml")), key=lambda e: e.id)
    validate(experiments)
    return experiments


def validate(experiments: list[Experiment]) -> None:
    """Raise ValueError on the first malformed or inconsistent entry."""
    seen_ids, seen_slugs = set(), set()
    for e in experiments:
        if not ID_RE.match(e.id):
            raise ValueError(f"{e.id}: id must look like X0001")
        if not SLUG_RE.match(e.slug):
            raise ValueError(f"{e.id}: slug {e.slug!r} must be lowercase words joined by '-'")
        if e.id in seen_ids or e.slug in seen_slugs:
            raise ValueError(f"{e.id}: duplicate id or slug {e.slug!r}")
        seen_ids.add(e.id)
        seen_slugs.add(e.slug)
        if e.category not in CATEGORIES:
            raise ValueError(f"{e.id}: category {e.category!r} is not one of {CATEGORIES}")
        if any(p.arxiv is None and p.title is None for p in e.papers):
            raise ValueError(f"{e.id}: a paper needs an arxiv id or a title")
        if e.source is Source.INTAKE and e.intake is None:
            raise ValueError(f"{e.id}: intake experiments must record their intake comment")
        trial_ids = [t.id for t in e.trials]
        if bad := [i for i in trial_ids if not TRIAL_ID_RE.match(i)]:
            raise ValueError(f"{e.id}: trial ids {bad} must look like t001")
        numbers = [int(i[1:]) for i in trial_ids]
        if numbers != sorted(set(numbers)):
            raise ValueError(f"{e.id}: trial ids must increase down the file, with no repeats")
        for t in e.trials:
            if t.commit is not None and not COMMIT_RE.match(t.commit):
                raise ValueError(
                    f"{e.id}.{t.id}: commit {t.commit!r} must be a lowercase hex SHA of 10 to 40 characters"
                )


def write_experiment(experiment: Experiment, directory: Path = REGISTRY_DIR) -> Path:
    path = directory / experiment.filename
    with open(path, "w") as f:
        yaml.safe_dump(_plain(experiment), f, sort_keys=False, allow_unicode=True, width=120)
    return path


def next_id(experiments: list[Experiment]) -> str:
    return f"X{max((int(e.id[1:]) for e in experiments), default=0) + 1:04d}"


# A Paloma macro loss in the text of a trial or event (e.g. a baseline "df1-ctrl 2.9543"), or a signed delta.
_LOSS_IN_TEXT = re.compile(r"(?<![\d.\w])([-+\u2212]?)(\d+\.\d{3,})(?![\d])")
_PLACEHOLDER_TEXT_FIELDS = ("baseline", "note", "variant", "tuned", "step_time_change")


def _placeholder_text(text: str, ratio: float) -> str:
    def scale(m: re.Match) -> str:
        sign, value = m.group(1), float(m.group(2))
        is_loss = not sign and 1.8 <= value <= 4.5
        is_delta = bool(sign) and value < 0.1
        return f"{sign}{value * ratio:.4f}" if is_loss or is_delta else m.group(0)

    return _LOSS_IN_TEXT.sub(scale, text)


def _placeholder_bpb(idea: dict, ratio: float) -> None:
    """Rescale an idea's macro-loss numbers to placeholder BPB in place: trial ``delta`` / ``loss`` and the losses and
    signed deltas written in trial and event text."""
    for trial in idea["trials"]:
        for key in ("delta", "loss"):
            if key in trial:
                trial[key] = [round(v * ratio, 4) for v in trial[key]]
        for key in _PLACEHOLDER_TEXT_FIELDS:
            if key in trial:
                trial[key] = _placeholder_text(trial[key], ratio)
    for event in idea["events"]:
        if "note" in event:
            event["note"] = _placeholder_text(event["note"], ratio)


def page_data(experiments: list[Experiment], generated_from: str, placeholder_bpb_ratio: float | None = None) -> dict:
    """The Belay page's results JSON: its ``ideas`` keep the page's own field names, with ``xid`` added. The registry
    records Paloma macro loss; with ``placeholder_bpb_ratio`` the page shows it as placeholder BPB (loss times the
    ratio) until trials record measured BPB."""
    ideas = []
    for e in experiments:
        idea = _plain(e)
        idea["xid"] = idea.pop("id")
        idea["id"] = idea.pop("slug")
        idea.setdefault("trials", [])
        idea.setdefault("events", [])
        idea.setdefault("tags", [])
        if placeholder_bpb_ratio is not None:
            _placeholder_bpb(idea, placeholder_bpb_ratio)
        ideas.append(idea)
    return {"generated_from": generated_from, "categories": list(CATEGORIES), "ideas": ideas}


def week_summary(experiments: list[Experiment], since: str, until: str) -> str:
    """Markdown list of the experiments with a trial or a decision dated in [since, until) (ISO dates)."""
    lines = []
    for e in experiments:
        trials = [t for t in e.trials if since <= t.date < until]
        events = [ev for ev in e.events if since <= ev.date < until]
        if not trials and not events:
            continue
        parts = []
        if trials:
            rungs = ", ".join(sorted({t.rung for t in trials}))
            parts.append(f"{len(trials)} trial{'s' if len(trials) > 1 else ''} ({rungs})")
        parts += [ev.type.value + (f": {ev.note}" if ev.note else "") for ev in events]
        lines.append(f"- **{e.id}** {e.name} — {'; '.join(parts)}")
    if not lines:
        return "No experiments were tried or decided this week."
    return f"Experiments tried or decided ({len(lines)}):\n\n" + "\n".join(lines)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="load and check every entry")
    build = sub.add_parser("build", help="write the Belay page's results JSON to stdout")
    build.add_argument("--generated-from", default="experiments/belay/registry")
    build.add_argument("--placeholder-bpb-ratio", type=float, help="show macro loss as placeholder BPB (loss x ratio)")
    summary = sub.add_parser("summary", help="markdown summary of a week's experiments")
    summary.add_argument("--since", required=True, help="ISO date, inclusive")
    summary.add_argument("--until", required=True, help="ISO date, exclusive")
    new = sub.add_parser("new", help="allocate the next id and write an approved entry")
    new.add_argument("--slug", required=True)
    new.add_argument("--name", required=True)
    new.add_argument("--category", required=True, choices=CATEGORIES)
    new.add_argument("--summary", required=True)
    new.add_argument("--source", required=True, choices=[s.value for s in Source])
    new.add_argument("--first-batch", type=int)
    new.add_argument("--intake-url")
    new.add_argument("--intake-author")
    new.add_argument("--approved-by")
    args = parser.parse_args(argv)

    experiments = load_registry()
    if args.command == "validate":
        print(f"{len(experiments)} experiments OK")
    elif args.command == "summary":
        print(week_summary(experiments, args.since, args.until))
    elif args.command == "build":
        data = page_data(experiments, args.generated_from, args.placeholder_bpb_ratio)
        json.dump(data, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    else:
        intake = None
        if args.intake_url:
            intake = Intake(url=args.intake_url, author=args.intake_author, approved_by=args.approved_by)
        experiment = Experiment(
            id=next_id(experiments),
            slug=args.slug,
            name=args.name,
            category=args.category,
            summary=args.summary,
            source=Source(args.source),
            status=Status.APPROVED,
            first_batch=args.first_batch,
            intake=intake,
        )
        validate([*experiments, experiment])
        print(write_experiment(experiment))


if __name__ == "__main__":
    main(sys.argv[1:])
