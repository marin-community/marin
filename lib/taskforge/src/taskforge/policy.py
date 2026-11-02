# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The text a run sends its models, as data: ``Prompt`` templates grouped by stage into a ``Policy``.

``load_policy(defaults_dir, *overlays)`` reads ``policy.json`` in ``defaults_dir``, a JSON object from each
dotted field name (``author.system``) to a prompt file relative to that directory. An overlay is a JSON object of
the same shape, naming any subset of the fields, with paths relative to the overlay's own directory; each overlay
replaces the files of the fields it names, in order. A key that names no field is refused, and so is a default
that leaves a field without a file. ``taskforge.defaults.DEFAULT_POLICY`` is the policy of the package's own files.

A prompt file holds its template and then one newline, which the loader drops, so a template that ends with a
newline ends its file with a blank line. A template is rendered with ``str.format``: ``{name}`` is a field the
stage fills, ``{{`` and ``}}`` are literal braces.
"""

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import get_type_hints

from taskforge.content_hash import digest

POLICY_FILE = "policy.json"


@dataclass(frozen=True)
class Prompt:
    """A template loaded from a prompt file.

    Attributes:
        name: The file's path as its policy file names it.
        template: The file's text without its final newline.
    """

    name: str
    template: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.template.encode()).hexdigest()

    def render(self, **fields: object) -> str:
        return self.template.format(**fields)


@dataclass(frozen=True)
class ProposerPolicy:
    """The capability proposer's text: the planning call, then one proposal document per slot.

    Attributes:
        system: The system turn of every call. Field: ``solver_machine_rule``.
        plan: The planning request. Fields: ``n``, ``environments``, ``max_per_pairing``,
            ``research_criterion`` and ``capability_record``.
        write: The request for one slot's document. Fields: ``slot``, ``document_template``,
            ``solver_machine_rule``, ``research_criterion`` (indented two spaces), ``headings``,
            ``capability_record``, ``portfolio`` and ``planned_slot``.
        repair: The follow-up to a document the parser refused. Field: ``error``.
        research_criterion: When a task's build needs research, quoted by ``plan`` and ``write``.
        solver_machine_rule: The rule for the ``solver_machine`` header value. Fields: one per
            ``SolverMachine`` member, named by its value.
    """

    system: Prompt
    plan: Prompt
    write: Prompt
    repair: Prompt
    research_criterion: Prompt
    solver_machine_rule: Prompt


@dataclass(frozen=True)
class AuthorPolicy:
    """The build-program author's text, for the one-shot and the agentic author.

    Attributes:
        system: The rules. Fields: ``grader`` (the grader step's name) and ``imports``.
        user: The request. Fields: ``proposal``, ``reference`` and ``template``.
        revision: A revision request. Fields: ``kind`` and ``detail``.
        grading_reference: The SDK reference's section on what ``grade_candidate`` returns.
        session: The agentic section appended to ``system``. Fields: ``program``, ``proposal``, ``sdk``,
            ``template``, ``notes`` (workspace paths), ``write_chars`` and ``turns``.
        seed: The first user turn of a session seeded from an earlier one. Fields: ``program``, ``notes``
            and ``submitted``.
        handoff: The lead of a session after one that did not submit. Field: ``reason``.
        no_revision: The lead of a session after one that submitted, with nothing to revise.
        edit_and_submit: What follows ``revision`` in a session. Field: ``program``.
        length_nudge: The turn after a reply the output limit cut off. Field: ``limit``.
        nudge: The turn after a reply without a tool call. Field: ``program``.
    """

    system: Prompt
    user: Prompt
    revision: Prompt
    grading_reference: Prompt
    session: Prompt
    seed: Prompt
    handoff: Prompt
    no_revision: Prompt
    edit_and_submit: Prompt
    length_nudge: Prompt
    nudge: Prompt


@dataclass(frozen=True)
class AdversaryPrompts:
    """The validation adversary's text.

    Attributes:
        brief: The brief of every adversary and both modes. Fields: ``goal``, ``tools``, ``context``,
            ``submissions`` and ``probes``.
        whitebox: The header of the white-box section.
        submit: The ``submit`` tool's description.
        false_accept: The ``FALSE_ACCEPT`` adversary's goal.
        false_reject: The ``FALSE_REJECT`` adversary's goal.
        shortcut: The ``SHORTCUT`` adversary's goal.
    """

    brief: Prompt
    whitebox: Prompt
    submit: Prompt
    false_accept: Prompt
    false_reject: Prompt
    shortcut: Prompt


@dataclass(frozen=True)
class Policy:
    """Every prompt of a run, by stage."""

    proposer: ProposerPolicy
    author: AuthorPolicy
    adversary: AdversaryPrompts

    @property
    def digest(self) -> str:
        """The sha256 of each field's template sha256, by dotted field name; file names do not count."""
        return digest({key: prompt.sha256 for key, prompt in _prompts(self).items()})


def _fields() -> dict[str, type]:
    """Each group of ``Policy`` with its record type."""
    return get_type_hints(Policy)


def policy_keys() -> tuple[str, ...]:
    """The dotted name of every prompt field of ``Policy``, in declaration order."""
    return tuple(f"{group}.{field.name}" for group, record in _fields().items() for field in dataclasses.fields(record))


def _prompts(policy: Policy) -> dict[str, Prompt]:
    return {key: getattr(getattr(policy, key.split(".")[0]), key.split(".")[1]) for key in policy_keys()}


def _prompt_files(path: Path) -> dict[str, tuple[str, Path]]:
    """The fields ``path`` names, each with its file name and the file's path."""
    entries = json.loads(path.read_text())
    unknown = sorted(set(entries) - set(policy_keys()))
    if unknown:
        raise ValueError(f"{path} names no such policy field: {', '.join(unknown)}")
    return {key: (name, path.parent / name) for key, name in entries.items()}


def _load_prompt(name: str, path: Path) -> Prompt:
    text = path.read_text()
    if not text.endswith("\n"):
        raise ValueError(f"{path} does not end with a newline")
    return Prompt(name, text[:-1])


def load_policy(defaults_dir: Path, *overlays: Path) -> Policy:
    """The policy of ``defaults_dir/policy.json`` with each overlay's files in its place, in order.

    Raises:
        ValueError: a file names a field ``Policy`` lacks, the defaults leave a field without a file, or a
            prompt file does not end with a newline.
    """
    files = _prompt_files(defaults_dir / POLICY_FILE)
    missing = [key for key in policy_keys() if key not in files]
    if missing:
        raise ValueError(f"{defaults_dir / POLICY_FILE} has no file for {', '.join(missing)}")
    for overlay in overlays:
        files |= _prompt_files(overlay)
    groups = {
        group: record(
            **{field.name: _load_prompt(*files[f"{group}.{field.name}"]) for field in dataclasses.fields(record)}
        )
        for group, record in _fields().items()
    }
    return Policy(**groups)
