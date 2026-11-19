# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Vintage-pinned task contracts for the frozen difficulty sample receipts.

A difficulty receipt binds each selected task to ``canonical_sha256`` of its
parsed ``TaskSpec`` serialization. That serialization has drifted twice: the
``verifier_execution_requirements`` and ``completion_protocol`` fields joined
the schema after the 2026-11-13 freeze, and the 2026-11-17 serializer
(d9e801196d and follow-ups) began omitting *absent* late-joining fields that
the earlier schema serialized as explicit nulls. Receipts frozen on both
sides of that change are durable and immutable, so the difficulty lane cannot
recompute a single digest space for all of them — it must name the vintage a
receipt was frozen in and reproduce exactly that serialization.

Every difficulty contract computation (sample freeze and continuation
preparation) goes through :func:`difficulty_task_contract`, and every frozen
receipt is resolved to its vintage by :func:`resolve_receipt_vintage` before
its contracts are compared. A receipt that matches no recorded vintage is a
real drift — a converter, runtime or environment change after the freeze — and
fails with the task ID, the frozen and prepared digests, and the field-level
difference, instead of a bare mismatch.
"""

from dataclasses import dataclass

from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec

# The 2026-11-13 freeze ran before the late-joining fields existed: its dumps
# carried an explicit null for the absent execution_requirements default and no
# other late-joining key. Receipts for the original 2026.11.13.8 queue are in
# this digest space.
CONTRACT_VINTAGE_NULL_DEFAULTS = "2026-11-13-null-defaults"
# The 2026-11-17 serializer omits absent late-joining fields and keeps explicit
# nulls and values. Receipts frozen at or after that change (the resampled
# 2026.11.17.5 queue) are in this digest space.
CONTRACT_VINTAGE_ABSENT_OMITTED = "2026-11-17-absent-omitted"

# Newest first: resolution prefers the current serialization.
CONTRACT_VINTAGES = (CONTRACT_VINTAGE_ABSENT_OMITTED, CONTRACT_VINTAGE_NULL_DEFAULTS)

LATE_JOINING_FIELDS = ("execution_requirements", "verifier_execution_requirements", "completion_protocol")

VINTAGE_BASIS = {
    CONTRACT_VINTAGE_ABSENT_OMITTED: (
        "TaskSpec dumps omit absent late-joining fields and keep explicit nulls and values "
        "(TaskSpec._serialize_with_explicit_nulls as landed 2026-11-17)"
    ),
    CONTRACT_VINTAGE_NULL_DEFAULTS: (
        "TaskSpec dumps serialize the absent execution_requirements default as an explicit null; "
        "verifier_execution_requirements and completion_protocol had not joined the schema"
    ),
}


class ContractVintageError(ValueError):
    """A frozen receipt matches no recorded contract serialization vintage."""


def difficulty_task_payload(task: TaskSpec, vintage: str) -> dict:
    """The task's JSON payload exactly as the named receipt vintage serialized it.

    Raises ``ContractVintageError`` when the parsed task cannot belong to the
    vintage (a late-joining field carries evidence the vintage's schema did not
    exist) or when the current serializer no longer reproduces the vintage's
    documented behavior, so a future serialization change fails loudly here
    instead of silently drifting the digests again.
    """
    payload = task.model_dump(mode="json")
    if vintage == CONTRACT_VINTAGE_ABSENT_OMITTED:
        for name in LATE_JOINING_FIELDS:
            if name in task.model_fields_set:
                if name not in payload:
                    raise ContractVintageError(
                        f"The serializer dropped explicitly set {name}; the absent-omitted "
                        "vintage keeps explicit nulls and values (serialization drift)"
                    )
            elif name in payload:
                raise ContractVintageError(
                    f"The serializer emitted absent {name}; the absent-omitted vintage " "omits it (serialization drift)"
                )
        return payload
    if vintage == CONTRACT_VINTAGE_NULL_DEFAULTS:
        for name in ("verifier_execution_requirements", "completion_protocol"):
            if name in task.model_fields_set:
                raise ContractVintageError(
                    f"{name} carries explicit evidence that joined the schema after the "
                    "null-defaults vintage; this receipt cannot be a null-defaults freeze"
                )
        if "execution_requirements" not in task.model_fields_set:
            payload["execution_requirements"] = None
        return payload
    raise ContractVintageError(f"Unknown contract vintage: {vintage}")


def difficulty_task_contract(task: TaskSpec, runtime: dict, vintage: str) -> dict[str, str]:
    """The frozen-receipt contract for one task, in the named vintage's digest space."""
    return {
        "task_id": task.id,
        "task_sha256": canonical_sha256(difficulty_task_payload(task, vintage)),
        "runtime_sha256": canonical_sha256(runtime),
    }


@dataclass(frozen=True)
class TaskMismatch:
    """One prepared task whose digest no recorded vintage reproduces."""

    task_id: str
    frozen_sha256: str
    prepared: dict[str, str]
    vintage_payload_differences: list[str]

    def describe(self) -> str:
        lines = [
            f"  task {self.task_id}: frozen {self.frozen_sha256[:16]}",
            *(f"    {vintage}: {sha[:16]}" for vintage, sha in self.prepared.items()),
        ]
        lines.extend(f"    differs by: {difference}" for difference in self.vintage_payload_differences)
        return "\n".join(lines)


def payload_differences(first: dict, second: dict) -> list[str]:
    """Field-level differences between two vintage serializations of one task."""
    differences = []
    for name in sorted(set(first) | set(second)):
        if name not in first:
            differences.append(f"{name} present only in the second serialization ({second[name]!r})")
        elif name not in second:
            differences.append(f"{name} present only in the first serialization ({first[name]!r})")
        elif first[name] != second[name]:
            differences.append(f"{name}: {first[name]!r} vs {second[name]!r}")
    return differences


def _vintage_digest(task: TaskSpec, vintage: str) -> str:
    """The vintage's digest, or the reason the vintage cannot represent the task."""
    try:
        return canonical_sha256(difficulty_task_payload(task, vintage))
    except ContractVintageError as error:
        return f"<not representable: {error}>"


def resolve_receipt_vintage(frozen: dict[str, str], tasks: list[TaskSpec], runtimes: list[dict]) -> str:
    """The receipt vintage whose serialization reproduces every frozen task digest.

    A receipt is frozen by one code state, so every task must resolve to the
    same vintage. When a task's bytes changed after the freeze (converter,
    runtime or environment drift), no vintage matches and the error names the
    task, both candidate digests and the field-level difference.
    """
    mismatch: TaskMismatch | None = None
    for vintage in CONTRACT_VINTAGES:
        matches_all = True
        mismatch = None
        for task, runtime in zip(tasks, runtimes, strict=True):
            frozen_sha = frozen.get(task.id)
            if frozen_sha is None:
                continue
            try:
                contract = difficulty_task_contract(task, runtime, vintage)
            except ContractVintageError:
                matches_all = False
                if mismatch is None:
                    mismatch = TaskMismatch(
                        task_id=task.id,
                        frozen_sha256=frozen_sha,
                        prepared={item: _vintage_digest(task, item) for item in CONTRACT_VINTAGES},
                        vintage_payload_differences=[],
                    )
                continue
            if contract["task_sha256"] != frozen_sha:
                matches_all = False
                if mismatch is None:
                    other = next(item for item in CONTRACT_VINTAGES if item != vintage)
                    try:
                        differences = payload_differences(
                            difficulty_task_payload(task, vintage), difficulty_task_payload(task, other)
                        )
                    except ContractVintageError:
                        differences = []
                    mismatch = TaskMismatch(
                        task_id=task.id,
                        frozen_sha256=frozen_sha,
                        prepared={item: _vintage_digest(task, item) for item in CONTRACT_VINTAGES},
                        vintage_payload_differences=differences,
                    )
        if matches_all:
            return vintage
    assert mismatch is not None
    raise ContractVintageError(
        "Prepared task differs from the frozen sample receipt under every recorded contract "
        f"vintage ({', '.join(CONTRACT_VINTAGES)}). The task bytes changed after the receipt "
        "was frozen — a converter, recipe, runtime or environment artifact change — or the "
        "receipt was frozen by an unrecorded schema vintage:\n" + mismatch.describe()
    )


def describe_vintage(vintage: str) -> str:
    """The human-readable basis for a resolved contract vintage."""
    return VINTAGE_BASIS[vintage]
