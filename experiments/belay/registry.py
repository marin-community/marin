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
CATEGORIES = (
    "Attention variants",
    "Linear attention / KDA",
    "Depth connections",
    "Embeddings and inputs",
    "MoE and routing",
    "Optimizers",
    "Normalization",
    "Output head and loss",
    "Training dynamics and schedules",
    "Systems and speed",
    "Recipe stacks",
)


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
    """``cls`` from a YAML mapping, rejecting unknown keys and converting nested records, enums and sequences."""
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
        experiment = _build(Experiment, yaml.safe_load(f))
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


def write_experiment(experiment: Experiment, directory: Path = REGISTRY_DIR) -> Path:
    path = directory / experiment.filename
    with open(path, "w") as f:
        yaml.safe_dump(_plain(experiment), f, sort_keys=False, allow_unicode=True, width=120)
    return path


def next_id(experiments: list[Experiment]) -> str:
    return f"X{max((int(e.id[1:]) for e in experiments), default=0) + 1:04d}"


def page_data(experiments: list[Experiment], generated_from: str) -> dict:
    """The Belay page's results JSON: its ``ideas`` keep the page's own field names, with ``xid`` added."""
    ideas = []
    for e in experiments:
        idea = _plain(e)
        idea["xid"] = idea.pop("id")
        idea["id"] = idea.pop("slug")
        idea.setdefault("trials", [])
        idea.setdefault("events", [])
        idea.setdefault("tags", [])
        ideas.append(idea)
    return {"generated_from": generated_from, "categories": list(CATEGORIES), "ideas": ideas}


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="load and check every entry")
    build = sub.add_parser("build", help="write the Belay page's results JSON to stdout")
    build.add_argument("--generated-from", default="experiments/belay/registry")
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
    elif args.command == "build":
        json.dump(page_data(experiments, args.generated_from), sys.stdout, ensure_ascii=False, separators=(",", ":"))
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
