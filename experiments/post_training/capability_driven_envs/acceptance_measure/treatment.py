# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The three arms as data: JSON overlays of dotted keys, resolved into one ``Treatment`` per arm.

An arm file is ``{"name", "description", "extends", "set"}``. ``extends`` names the arm file it overlays
(relative to its own directory) or is null for the base, and ``set`` maps dotted keys to values. A key of kind
``prompt`` holds the path of one text file and ``prompts`` a list of them; ``files`` maps workspace paths to text
files; every path is relative to the arm file's directory, as in the policy overlays of PR 9 ("prompts and policy
as data"). The base arm must set every key in ``KEYS`` and an overlay may set only keys in ``KEYS``, so a typo is
refused rather than ignored.

``Treatment.digest`` is the sha256 of the resolved values with every file replaced by its content's sha256, so two
arms with the same text and settings share a digest whatever their file names. ``KEYS[key].surfaces`` names what
the key may change in a rendered allocation (the dry run's diff): a key changed between two arms must change each
of its surfaces, and no surface may change unless a changed key names it.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from taskforge.content_hash import digest

ARMS_DIR = Path(__file__).parent / "arms"
ARM_NAMES = ("A", "B", "C")


class Kind(StrEnum):
    SCALAR = "scalar"
    PROMPT = "prompt"
    PROMPTS = "prompts"
    FILES = "files"


@dataclass(frozen=True)
class Key:
    kind: Kind
    surfaces: tuple[str, ...] = ()
    """What the key may change in a rendered allocation, besides its own ``policy:<key>`` entry."""
    policy_field: str | None = None
    """The PR 9/12 ``Policy`` field (abstractions.md section 5.3) this key becomes, when one exists."""


KEYS: Mapping[str, Key] = {
    "models.author": Key(Kind.SCALAR, ("author.model",), "author.model"),
    "models.build": Key(Kind.SCALAR, ("build.model",)),
    "models.controls": Key(Kind.SCALAR, ("controls.model",)),
    "models.adversary": Key(Kind.SCALAR, ("adversary.model",), "validation.adversaries[].sampling"),
    "models.solver": Key(Kind.SCALAR, ("solver.model",)),
    "models.shadow_judge": Key(Kind.SCALAR, ("shadow.judge",)),
    "author.turns": Key(Kind.SCALAR, ("author.system",), "author.turns"),
    "author.turn_max_tokens": Key(Kind.SCALAR, ("author.sampling",), "author.turn_max_tokens"),
    "author.session_token_cap": Key(Kind.SCALAR, (), "author.session_token_cap"),
    "author.max_continuations": Key(Kind.SCALAR, ("author.sampling",)),
    "author.revision_same_session.max_prompt_tokens": Key(Kind.SCALAR, (), "author.session"),
    "author.revision_same_session.max_turns_used": Key(Kind.SCALAR, (), "author.session"),
    "author.guidance": Key(Kind.PROMPTS, ("author.system",), "author.guidance"),
    "author.reference_files": Key(Kind.FILES, ("author.workspace",)),
    "author.tools": Key(Kind.SCALAR, ("author.tools",)),
    "author.probe_battery.enforce_at_submit": Key(Kind.SCALAR, ("author.submit_rule",), "author.feedback=enforce"),
    "author.probe_battery.witness_replay": Key(Kind.SCALAR, ("author.battery",)),
    "author.probe_battery.max_refusals": Key(Kind.SCALAR, ("author.submit_rule",)),
    "author.unchanged_resubmit.text": Key(Kind.PROMPT, ("author.unchanged_resubmit",)),
    "author.unchanged_resubmit.max_refusals": Key(Kind.SCALAR, ("author.unchanged_resubmit",)),
    "author.repeated_error.text": Key(Kind.PROMPT, ("author.repeated_error",)),
    "author.repeated_error.hint_after": Key(Kind.SCALAR, ("author.repeated_error",)),
    "author.repeated_error.hold_after": Key(Kind.SCALAR, ("author.repeated_error",)),
    "build.wall_timeout": Key(Kind.SCALAR),
    "validation.solver_k": Key(Kind.SCALAR, ("validation.plan",), "validation.solver_k"),
    "validation.max_retries": Key(Kind.SCALAR, ("validation.plan",), "validation.max_retries"),
    "validation.retry_backoff": Key(Kind.SCALAR, ("validation.plan",)),
    "validation.command_timeout": Key(Kind.SCALAR, ("validation.plan",)),
    "validation.adversaries": Key(
        Kind.SCALAR, ("adversary.brief.false_accept", "adversary.brief.false_reject"), "validation.adversaries"
    ),
    "budget.max_triage_repairs": Key(Kind.SCALAR, (), "budget.max_triage_repairs"),
    "budget.max_build_revisions": Key(Kind.SCALAR, (), "budget.max_build_revisions"),
    "budget.max_validation_retries": Key(Kind.SCALAR, (), "budget.max_validation_retries"),
    "budget.max_modify_rounds": Key(Kind.SCALAR, (), "budget.max_modify_rounds"),
    "review.band.low": Key(Kind.SCALAR, ("review.band",), "review.band"),
    "review.band.high": Key(Kind.SCALAR, ("review.band",), "review.band"),
    "review.feedback": Key(Kind.SCALAR, ("review.modify_feedback",), "author.feedback"),
    "review.class_feedback": Key(Kind.PROMPT, ("review.modify_feedback",)),
    "review.saturated_accept": Key(Kind.SCALAR, ("review.saturated_decision",), "review.band.accepted"),
    "shadow.l3": Key(Kind.SCALAR, ("shadow.l3",)),
    "shadow.l4": Key(Kind.SCALAR, ("shadow.l4",)),
    "shadow.judge.enabled": Key(Kind.SCALAR, ("shadow.judge",)),
    "shadow.judge.prompt": Key(Kind.PROMPT, ("shadow.judge",)),
    "shadow.judge.max_claims_per_component": Key(Kind.SCALAR, ("shadow.judge",)),
}

TOOLS = frozenset({"embed_file", "run_probe_battery"})
FEEDBACK = frozenset({"witness", "class"})


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ArmFile:
    path: Path
    name: str
    description: str
    extends: Path | None
    values: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return sha256_file(self.path)


def read_arm_file(path: Path) -> ArmFile:
    doc = json.loads(path.read_text())
    unknown = sorted(set(doc) - {"name", "description", "extends", "set"})
    if unknown:
        raise ValueError(f"{path}: unknown top-level fields {unknown}")
    values = doc["set"]
    bad = sorted(set(values) - set(KEYS))
    if bad:
        raise ValueError(f"{path}: no such treatment key: {', '.join(bad)}")
    resolved: dict[str, Any] = {}
    for key, value in values.items():
        resolved[key] = _resolve_paths(path.parent, key, value)
    extends = None if doc["extends"] is None else (path.parent / doc["extends"]).resolve()
    return ArmFile(path.resolve(), doc["name"], doc["description"], extends, resolved)


def _resolve_paths(base: Path, key: str, value: Any) -> Any:
    """``value`` with the text-file paths of a file-valued key made absolute; each file must exist."""

    def one(raw: str) -> Path:
        resolved = (base / raw).resolve()
        if not resolved.is_file():
            raise ValueError(f"{key}: {raw} is not a file (relative to {base})")
        return resolved

    match KEYS[key].kind:
        case Kind.PROMPT:
            return None if value is None else one(value)
        case Kind.PROMPTS:
            return tuple(one(item) for item in value)
        case Kind.FILES:
            return {workspace: one(item) for workspace, item in sorted(value.items())}
        case Kind.SCALAR:
            return value


@dataclass(frozen=True)
class Treatment:
    """One arm, resolved: every key's value, with text files as absolute paths, and the chain of arm files."""

    name: str
    values: Mapping[str, Any]
    chain: tuple[ArmFile, ...]
    """The arm files from the base to this arm."""
    texts: dict[Path, str] = field(default_factory=dict, compare=False, repr=False)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def text(self, key: str) -> str | None:
        """The content of a ``prompt`` key's file, or None when it is unset."""
        if KEYS[key].kind is not Kind.PROMPT:
            raise ValueError(f"{key} is not a prompt key")
        path = self.values[key]
        return None if path is None else self._read(path)

    def texts_of(self, key: str) -> tuple[str, ...]:
        if KEYS[key].kind is not Kind.PROMPTS:
            raise ValueError(f"{key} is not a prompts key")
        return tuple(self._read(path) for path in self.values[key])

    def files_of(self, key: str) -> dict[str, str]:
        if KEYS[key].kind is not Kind.FILES:
            raise ValueError(f"{key} is not a files key")
        return {workspace: self._read(path) for workspace, path in self.values[key].items()}

    def _read(self, path: Path) -> str:
        if path not in self.texts:
            self.texts[path] = path.read_text()
        return self.texts[path]

    def content_values(self) -> dict[str, Any]:
        """The values with every file replaced by ``{"sha256": ...}``: what the digest and the diff compare."""
        return {key: _content(KEYS[key].kind, value) for key, value in sorted(self.values.items())}

    @property
    def digest(self) -> str:
        return digest(self.content_values())

    def manifest(self) -> dict[str, Any]:
        """The arm's identity for ``manifest.json``: its files' shas, the resolved digest and the file digests."""
        return {
            "name": self.name,
            "arm_files": [{"path": arm.path.name, "sha256": arm.sha256} for arm in self.chain],
            "policy_digest": self.digest,
            "values": self.content_values(),
            "files": {
                key: _file_listing(KEYS[key].kind, value)
                for key, value in sorted(self.values.items())
                if KEYS[key].kind is not Kind.SCALAR and value
            },
        }


def _content(kind: Kind, value: Any) -> Any:
    match kind:
        case Kind.PROMPT:
            return None if value is None else {"sha256": sha256_file(value)}
        case Kind.PROMPTS:
            return [{"sha256": sha256_file(path)} for path in value]
        case Kind.FILES:
            return {workspace: {"sha256": sha256_file(path)} for workspace, path in value.items()}
        case Kind.SCALAR:
            return value


def _file_listing(kind: Kind, value: Any) -> Any:
    root = Path(__file__).parent
    match kind:
        case Kind.PROMPT:
            return str(value.relative_to(root))
        case Kind.PROMPTS:
            return [str(path.relative_to(root)) for path in value]
        case Kind.FILES:
            return {workspace: str(path.relative_to(root)) for workspace, path in value.items()}
        case Kind.SCALAR:
            return value


def load_treatment(path: Path) -> Treatment:
    """Resolve the arm file at ``path`` through its ``extends`` chain.

    Raises:
        ValueError: a file names an unknown key, a path does not exist, the base leaves a key unset, the chain
            loops, or a value fails its check (``check``).
    """
    chain: list[ArmFile] = []
    seen: set[Path] = set()
    current: Path | None = path.resolve()
    while current is not None:
        if current in seen:
            raise ValueError(f"{path}: the extends chain loops at {current}")
        seen.add(current)
        arm = read_arm_file(current)
        chain.insert(0, arm)
        current = arm.extends
    missing = sorted(set(KEYS) - set(chain[0].values))
    if missing:
        raise ValueError(f"{chain[0].path}: the base arm leaves {', '.join(missing)} unset")
    values: dict[str, Any] = {}
    for arm in chain:
        values |= arm.values
    treatment = Treatment(chain[-1].name, values, tuple(chain))
    check(treatment)
    return treatment


def check(treatment: Treatment) -> None:
    """Refuse values the driver cannot carry out."""
    tools = set(treatment["author.tools"])
    if not tools <= TOOLS:
        raise ValueError(f"author.tools names unknown tools {sorted(tools - TOOLS)}")
    if treatment["author.probe_battery.enforce_at_submit"] and "run_probe_battery" not in tools:
        raise ValueError("author.probe_battery.enforce_at_submit needs the run_probe_battery tool")
    if treatment["review.feedback"] not in FEEDBACK:
        raise ValueError(f"review.feedback is one of {sorted(FEEDBACK)}")
    if (treatment["review.feedback"] == "class") != (treatment["review.class_feedback"] is not None):
        raise ValueError("review.feedback 'class' and review.class_feedback are set together")
    if not 0 <= treatment["review.band.low"] <= treatment["review.band.high"] <= 1:
        raise ValueError("review.band needs 0 <= low <= high <= 1")
    for workspace in treatment["author.reference_files"]:
        if not workspace.startswith("reference/"):
            raise ValueError(f"author.reference_files: {workspace} must lie under reference/ (read-only)")


def load_arms(arms_dir: Path = ARMS_DIR, names: tuple[str, ...] = ARM_NAMES) -> dict[str, Treatment]:
    return {name: load_treatment(arms_dir / f"{name}.json") for name in names}


def changed_keys(a: Treatment, b: Treatment) -> tuple[str, ...]:
    """The keys whose content differs between two arms."""
    left, right = a.content_values(), b.content_values()
    return tuple(key for key in sorted(KEYS) if left[key] != right[key])


def intended_surfaces(a: Treatment, b: Treatment) -> frozenset[str]:
    """The surfaces the keys changed between ``a`` and ``b`` may change, with their own ``policy:`` entries."""
    keys = changed_keys(a, b)
    return frozenset({f"policy:{key}" for key in keys} | {s for key in keys for s in KEYS[key].surfaces})
