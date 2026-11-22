# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backfill declared task requirements from the task's own bytes, with per-field evidence.

The stage runs after conversion and deduplication, before quality review, filtering and
verification, so every downstream reader sees the fields it declares. Static rules match
explicit usage in the files a task carries — solution and verifier scripts, build contexts,
declarative configs — and the capabilities they prove are written into the spec's
``environment_requirements`` with the evidence carried in a task label. The stage is
conservative by construction: a field is written only when a rule matched a file's
executable text, an already-declared field is never rewritten, and an unresolved capability
stays "not stated", which keeps the task's current route. Rule matches that survive only in
comments or prose are recorded as the ambiguous residual for the LLM pass instead.

Every rule result is counted per field and rule in the ``requirements/manifest.json``
receipt beside the review and verify manifests, so the pilot can measure the residual rate
and the per-rule false positives and false negatives the comments below expect.
"""

import hashlib
import json
import pickle
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from functools import partial
from typing import Any

import pyarrow as pa
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import ShardInfo

from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ScriptGrader, TaskSpec, VerifyitGrader
from taskcompendium.pipeline.conversion import ConvertedRow
from taskcompendium.pipeline.deduplication import RETAINED_SCHEMA, DeduplicatedPopulation
from taskcompendium.pipeline.execution_telemetry import PhaseTelemetry, execute_phase
from taskcompendium.pipeline.models import NormalizedTask
from taskcompendium.pipeline.shard_outputs import ShardOutput, write_shard_outputs
from taskcompendium.pipeline.sources import raw_input_sha256
from taskcompendium.runtime.resources import resource_bytes

REQUIREMENTS_REVISION = "1"
"""The backfill stage's rule vocabulary and receipt schema revision."""

REQUIREMENTS_MANIFEST_PATH = "requirements/manifest.json"
"""The backfill receipt, relative to a source's output, beside the review and verify manifests."""

LABEL_KEY = "requirements.backfill.capabilities"
"""The task label holding the evidence for every capability this stage declared."""

NETWORK_CAPABILITY = "network"
GIT_REPOSITORY_CAPABILITY = "git_repository"
PROCESS_CAPABILITY = "process"
PYTHON_CAPABILITY = "python"
"""The capability vocabulary the rules may declare; matches the task-machine tier table's names."""

OUTCOMES = ("declared", "unchanged", "ambiguous", "not_stated")
"""The per-capability outcomes a task can record: written with evidence, already declared,
residual the static rules could not decide, or no signal at all."""


@dataclass(frozen=True)
class BackfillPolicy:
    """The backfill policy a source pipeline run applies to every normalized task.

    ``rules_revision`` names the static rule set and rides in every receipt and task label;
    changing it changes the receipt identity, so re-ingesting under a new rule set never
    silently overwrites an old one. ``residual_threshold`` bounds the ambiguous share the
    receipt reports before the LLM residual is worth running; ``None`` skips the comparison.
    """

    rules_revision: str
    residual_threshold: float | None = None


@dataclass(frozen=True)
class RequirementRule:
    """One static rule over a task file: a content pattern over its executable text, or a file name.

    An ``ambiguous`` rule never declares anything; it marks the capabilities it names as the
    residual the static pass defers, and only when no declared rule decided them.
    """

    name: str
    capabilities: tuple[str, ...]
    pattern: re.Pattern[str] | None = None
    filename_pattern: re.Pattern[str] | None = None
    ambiguous: bool = False

    def matches(self, path: str, text: str | None, *, executable: bool) -> bool:
        """Match a file by its name, or by its decoded text when the file can carry commands."""
        if self.filename_pattern is not None:
            return self.filename_pattern.search(_file_name(path)) is not None
        return executable and text is not None and self.pattern is not None and self.pattern.search(text) is not None


COMMAND_FILE_SUFFIXES = (".sh", ".bash", ".py", ".toml", ".yaml", ".yml", ".json", ".mk", ".cfg")
COMMAND_FILE_NAMES = ("Makefile",)
"""Files whose text may hold commands the rules read; prose carriers such as READMEs stay out."""


def _file_name(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _is_command_file(path: str) -> bool:
    name = _file_name(path)
    return name.startswith("Dockerfile") or name in COMMAND_FILE_NAMES or path.endswith(COMMAND_FILE_SUFFIXES)


def _executable_text(text: str) -> str:
    """Drop the ``#`` comment tail of every line, so prose in comments cannot declare a capability.

    A ``#`` inside a JSON string, URL fragment or shell parameter keeps its text: only a hash at
    line start or after whitespace starts a comment in the file types the rules read.
    """
    lines = []
    for line in text.splitlines():
        for index, character in enumerate(line):
            if character == "#" and (index == 0 or line[index - 1] in (" ", "\t")):
                line = line[:index]
                break
        lines.append(line)
    return "\n".join(lines)


DECLARED_RULES: tuple[RequirementRule, ...] = (
    RequirementRule(
        "network_fetch_command",
        (NETWORK_CAPABILITY,),
        re.compile(r"\b(?:curl|wget)\b"),
    ),
    RequirementRule(
        "package_install_command",
        (NETWORK_CAPABILITY,),
        re.compile(
            r"\b(?:pip3?(?:\.\d+)?\s+install|pipx\s+install|uv\s+(?:pip\s+install|add|sync)"
            r"|apt(?:-get)?\s+install|conda\s+install|gem\s+install|npm\s+(?:install|add)|yarn\s+add)\b"
        ),
    ),
    RequirementRule(
        "dependency_manifest_file",
        (NETWORK_CAPABILITY,),
        filename_pattern=re.compile(r"^(?:requirements[\w.-]*\.txt|pyproject\.toml|environment\.ya?ml)$"),
    ),
    RequirementRule(
        "git_remote_command",
        (GIT_REPOSITORY_CAPABILITY, NETWORK_CAPABILITY),
        re.compile(r"\bgit\s+(?:clone|fetch|pull|push|ls-remote|submodule\s+(?:update|add))\b"),
    ),
    RequirementRule(
        "background_process_command",
        (PROCESS_CAPABILITY,),
        re.compile(
            r"\b(?:nohup|setsid|pkill|killall|start-stop-daemon)\b|\bps\s+(?:aux|-)|(?:systemctl|service)\s+\S+\s+(?:start|restart)\b"
        ),
    ),
    RequirementRule(
        "python_runtime_command",
        (PYTHON_CAPABILITY,),
        re.compile(r"\b(?:python3?(?:\.\d+)?|pytest)\b"),
    ),
)
"""Rules that declare a capability from explicit usage.

Expected false positives and false negatives, checked by the pilot's hand-verified sample:
``network_fetch_command`` and ``package_install_command`` miss network access through
standard-library or framework calls and cannot prove absence; ``dependency_manifest_file``
over-declares network for vendored dependencies that no longer need resolving;
``git_remote_command`` misses git through frameworks; ``background_process_command`` misses
background work through service supervisors it does not name; ``python_runtime_command``
misses Python invoked through a shebang with no literal name in the script body.
"""

AMBIGUOUS_RULES: tuple[RequirementRule, ...] = (
    RequirementRule(
        "network_mention",
        (NETWORK_CAPABILITY,),
        re.compile(r"\b(?:internet|network|online|download|upload)\b", re.IGNORECASE),
        ambiguous=True,
    ),
    RequirementRule(
        "install_mention",
        (NETWORK_CAPABILITY,),
        re.compile(r"\binstall(?:s|ed|ing|ation)?\b", re.IGNORECASE),
        ambiguous=True,
    ),
    RequirementRule(
        "http_client_import",
        (NETWORK_CAPABILITY,),
        re.compile(r"^\s*(?:import|from)\s+(?:requests|urllib\.|urllib\s|httpx|aiohttp|socket)\b", re.MULTILINE),
        ambiguous=True,
    ),
    RequirementRule(
        "git_mention",
        (GIT_REPOSITORY_CAPABILITY,),
        re.compile(r"\bgit\b", re.IGNORECASE),
        ambiguous=True,
    ),
    RequirementRule(
        "process_mention",
        (PROCESS_CAPABILITY,),
        re.compile(r"\b(?:daemon|background process|long[- ]running|server)\b", re.IGNORECASE),
        ambiguous=True,
    ),
    RequirementRule(
        "python_mention",
        (PYTHON_CAPABILITY,),
        re.compile(r"\b(?:pytest|unittest|python)\b", re.IGNORECASE),
        ambiguous=True,
    ),
)
"""Signals that mark a capability ambiguous instead of declaring it.

They match comment and prose text as well as code, which is exactly why they never declare:
"download the model weights" in a comment means network access a rule may have missed, but
the words alone do not prove the task's machine needs it. The LLM residual decides these.
"""

RULES = DECLARED_RULES + AMBIGUOUS_RULES

RULE_CAPABILITIES: tuple[str, ...] = tuple(
    dict.fromkeys(capability for rule in RULES for capability in rule.capabilities)
)
"""Every capability the rule set may report an outcome for, in rule order."""


def task_file_inventory(task: TaskSpec) -> Iterator[tuple[str, str, bytes]]:
    """Every file the task carries as (role, path, decoded bytes).

    The static rules read the task's resources through this function only, so the shared
    resource inventory can replace it without touching the rules. Roles follow the resource
    groups the spec declares today, plus the actor and grader build contexts.
    """
    for role, resources in (
        ("all", task.resources.all),
        ("worker", task.resources.worker),
        ("oracle", task.resources.oracle),
        ("verifier", task.resources.verifier),
    ):
        for resource in resources:
            yield role, resource.path, resource_bytes(resource)
    if task.environment_requirements.docker_build is not None:
        for resource in task.environment_requirements.docker_build.files:
            yield "actor_build", resource.path, resource_bytes(resource)
    if isinstance(task.grader, ScriptGrader | VerifyitGrader) and task.grader.environment is not None:
        if task.grader.environment.docker_build is not None:
            for resource in task.grader.environment.docker_build.files:
                yield "grader_build", resource.path, resource_bytes(resource)


@dataclass(frozen=True)
class BackfillRecord:
    """A converted row with its backfill counters and whether its capabilities stay ambiguous."""

    row: ConvertedRow
    counts: Counter[str]
    ambiguous: bool


def _rule_hits(
    path: str, data: bytes, executable: bool, counts: Counter[str]
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    """Return the declared evidence per capability and the ambiguous capabilities one file contributes."""
    declared: dict[str, list[dict[str, Any]]] = {}
    ambiguous: set[str] = set()
    text: str | None = None
    if executable:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = None
    code = _executable_text(text) if text is not None else None
    for rule in RULES:
        haystack = text if rule.ambiguous else code
        if not rule.matches(path, haystack, executable=executable):
            continue
        counts[f"rule:{rule.name}"] += 1
        if rule.ambiguous:
            ambiguous.update(rule.capabilities)
            continue
        entry = {"rule": rule.name, "path": path, "sha256": hashlib.sha256(data).hexdigest()}
        for capability in rule.capabilities:
            entries = declared.setdefault(capability, [])
            if entry not in entries:
                entries.append(entry)
    return declared, ambiguous


@dataclass(frozen=True)
class TaskBackfill:
    """One task's backfill outcome: its (possibly rewritten) spec, per-capability outcomes and evidence."""

    task: TaskSpec
    outcome: str
    """The task-level outcome: declared, ambiguous, unchanged, or not_stated."""
    outcomes: dict[str, str]
    """The per-capability outcome for every capability the rule set may infer."""
    written: dict[str, list[dict[str, Any]]]
    """The declared evidence written into the task's label, per capability."""
    rule_hits: Counter[str]
    """How many files each rule matched, the drift metric the pilot's ledger reads."""
    ambiguous: bool
    machine_less: bool


def backfill_task(task: TaskSpec, policy: BackfillPolicy) -> TaskBackfill:
    """Apply the policy's static rules to one task.

    The task's declared machine semantics gate the rules: a task with no machine has no
    machine to describe, so the rules skip it and nothing is written. Returns the task
    with every evidence-backed capability declared and its evidence label (unchanged when
    nothing qualifies), the per-capability outcomes, the per-rule hit counts, and whether
    any capability stays ambiguous for the LLM residual.
    """
    rule_hits: Counter[str] = Counter()
    if task.environment_requirements.command_semantics is None:
        return TaskBackfill(task, "not_stated", {}, {}, rule_hits, False, True)
    declared: dict[str, list[dict[str, Any]]] = {}
    ambiguous: set[str] = set()
    seen: set[str] = set()
    for _, path, data in task_file_inventory(task):
        if path in seen:
            continue
        seen.add(path)
        file_declared, file_ambiguous = _rule_hits(path, data, _is_command_file(path), rule_hits)
        for capability, entries in file_declared.items():
            declared.setdefault(capability, [])
            for entry in entries:
                if entry not in declared[capability]:
                    declared[capability].append(entry)
        ambiguous.update(file_ambiguous)
    already = set(task.environment_requirements.capabilities)
    written: dict[str, list[dict[str, Any]]] = {}
    outcomes: dict[str, str] = {}
    for capability in RULE_CAPABILITIES:
        if capability in already:
            outcomes[capability] = "unchanged"
        elif capability in declared:
            written[capability] = declared[capability]
            outcomes[capability] = "declared"
        elif capability in ambiguous:
            outcomes[capability] = "ambiguous"
        else:
            outcomes[capability] = "not_stated"
    if written:
        environment = task.environment_requirements.model_copy(
            update={"capabilities": task.environment_requirements.capabilities + tuple(sorted(written))}
        )
        evidence = {
            "revision": REQUIREMENTS_REVISION,
            "policy_revision": policy.rules_revision,
            "capabilities": written,
        }
        label = json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        new_task = task.model_copy(
            update={"environment_requirements": environment, "labels": {**task.labels, LABEL_KEY: label}}
        )
        return TaskBackfill(new_task, "declared", outcomes, written, rule_hits, False, False)
    if ambiguous - already:
        return TaskBackfill(task, "ambiguous", outcomes, written, rule_hits, True, False)
    if already:
        return TaskBackfill(task, "unchanged", outcomes, written, rule_hits, False, False)
    return TaskBackfill(task, "not_stated", outcomes, written, rule_hits, False, False)


def _backfill_record(row: ConvertedRow, *, policy: BackfillPolicy) -> BackfillRecord:
    """Backfill one converted row's capabilities and count every field, rule and outcome."""
    counts: Counter[str] = Counter()
    counts["task:total"] += 1
    if not isinstance(row.result, NormalizedTask):
        counts["task:rejection"] += 1
        return BackfillRecord(row, counts, False)
    counts["task:normalized"] += 1
    result = backfill_task(row.result.task, policy)
    counts.update(result.rule_hits)
    if result.machine_less:
        counts["task:machine_less"] += 1
        return BackfillRecord(row, counts, False)
    for capability, outcome in result.outcomes.items():
        counts[f"cap:{capability}:{outcome}"] += 1
    counts[f"task:{result.outcome}"] += 1
    return BackfillRecord(replace(row, result=replace(row.result, task=result.task)), counts, result.ambiguous)


AMBIGUOUS_SCHEMA = pa.schema([("task_id", pa.string()), ("source_locator", pa.string())])


def _retained_row(record: BackfillRecord) -> dict[str, Any]:
    row = record.row
    return {
        "task_id": row.raw.id,
        "source_locator": row.raw.source.row,
        "raw_input_sha256": raw_input_sha256(row.original_data),
        "converted": pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL),
    }


def _ambiguous_row(record: BackfillRecord) -> dict[str, str] | None:
    if not record.ambiguous:
        return None
    return {"task_id": record.row.raw.id, "source_locator": record.row.raw.source.row}


def _write_backfilled(
    records: Iterator[BackfillRecord], shard: ShardInfo, *, path: str, ambiguous_path: str
) -> Iterator[Counter[str]]:
    """Write one shard's backfilled population and ambiguous tasks, counting every record."""
    counts: Counter[str] = Counter()

    def counted() -> Iterator[BackfillRecord]:
        for record in records:
            counts.update(record.counts)
            yield record

    write_shard_outputs(
        counted(),
        shard,
        [
            ShardOutput(str(StoragePath(path) / "retained/part-{shard:05d}.parquet"), RETAINED_SCHEMA, _retained_row),
            ShardOutput(str(StoragePath(ambiguous_path) / "part-{shard:05d}.parquet"), AMBIGUOUS_SCHEMA, _ambiguous_row),
        ],
    )
    yield counts


def _capability_counts(counts: Counter[str]) -> dict[str, dict[str, int]]:
    return {
        capability: {outcome: counts[f"cap:{capability}:{outcome}"] for outcome in OUTCOMES}
        for capability in RULE_CAPABILITIES
    }


def _receipt(counts: Counter[str], policy: BackfillPolicy, population_path: str, ambiguous_path: str) -> dict[str, Any]:
    """Assemble the requirements manifest: per field, rule and outcome counts, and its identity."""
    normalized = counts["task:normalized"]
    machine_less = counts["task:machine_less"]
    machine_ful = normalized - machine_less
    ambiguous_count = counts["task:ambiguous"]
    share = ambiguous_count / machine_ful if machine_ful else 0.0
    threshold = policy.residual_threshold
    receipt: dict[str, Any] = {
        "revision": REQUIREMENTS_REVISION,
        "policy": {"rules_revision": policy.rules_revision, "residual_threshold": threshold},
        "population": population_path,
        "counts": {
            "tasks": {
                "normalized": normalized,
                "machine_less": machine_less,
                "rejections": counts["task:rejection"],
            },
            "outcomes": {outcome: counts[f"task:{outcome}"] for outcome in (*OUTCOMES, "rejection")},
            "capabilities": _capability_counts(counts),
            "rules": {rule.name: counts[f"rule:{rule.name}"] for rule in RULES},
        },
        "ambiguous": {"count": ambiguous_count, "tasks": ambiguous_path},
        "residual": {
            "ambiguous_share": share,
            "over_threshold": threshold is not None and share > threshold,
        },
        # The LLM residual's ledger summary; the static-only stage records None.
        "llm": None,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def backfill_population(
    context: ZephyrContext,
    population: DeduplicatedPopulation,
    output_path: str,
    policy: BackfillPolicy,
    *,
    telemetry: PhaseTelemetry | None = None,
) -> tuple[DeduplicatedPopulation, dict[str, Any]]:
    """Declare evidence-backed capabilities on every normalized task and return the receipt.

    Writes the backfilled population and the ambiguous-task set under ``output_path``; the
    returned population keeps the original deduplication's drop manifest, since backfill
    changes declared fields and labels but never drops a row.
    """
    path = StoragePath(output_path)
    if (path / "population" / "manifest.json").exists():
        raise FileExistsError(f"Backfill output already exists and requires a new artifact version: {path}")
    result = execute_phase(
        context,
        population.rows()
        .map(partial(_backfill_record, policy=policy))
        .map_shard(partial(_write_backfilled, path=str(path / "population"), ambiguous_path=str(path / "ambiguous"))),
        telemetry=telemetry,
        operation="backfill",
    )
    counts: Counter[str] = Counter()
    for part in result.results:
        counts.update(part)
    if counts["task:total"] != population.unique_count:
        raise ValueError(f"Backfill output covers {counts['task:total']} of {population.unique_count} retained rows")
    backfilled = DeduplicatedPopulation(
        str(path / "population"),
        population.input_rows,
        population.raw_count,
        population.unique_count,
        population.within_source_duplicates,
        drops_path=population.path,
    )
    manifest = StoragePath(path / "population") / "manifest.json"
    manifest.write_text(json.dumps(asdict(backfilled), indent=2) + "\n")
    return backfilled, _receipt(counts, policy, str(path / "population"), str(path / "ambiguous"))
