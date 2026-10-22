"""Audit rendered lm-eval targets for malformed mapping serializations.

Run a representative live audit without evaluating a model:

    uv run python -m eval.regression.lm_eval_task_contracts --sample-size 64

The audit streams one raw record for each sampled task, applies that task's real
``process_docs`` hook, and renders its target.  A target equal to the comma-separated
keys of a mapping is almost always a template bug: Jinja's ``join`` iterates mappings by
key.  This catches stale raw-field references that stay available after preprocessing.
"""

from __future__ import annotations

import argparse
import random
import signal
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any

import datasets
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.utils import apply_template

DEFAULT_SAMPLE_SIZE = 64
DEFAULT_DOCS_PER_TASK = 3
DEFAULT_RANDOM_SEED = 20260723
DEFAULT_TASK_TIMEOUT = 30
EVALCHEMY_TASK_OVERRIDES = Path(__file__).parents[1] / "lm_eval_tasks"


@dataclass(frozen=True)
class TargetContractViolation:
    """A rendered target that serializes mapping keys instead of a gold value."""

    field_path: str
    target: str


def mapping_key_target_violation(document: Mapping[str, Any], target: object) -> TargetContractViolation | None:
    """Return the mapping-key serialization violation in a rendered target, if any.

    Jinja renders ``{{ mapping | join(',') }}`` as the mapping's keys.  A preprocessing
    step often leaves the raw mapping available, which makes that error look valid until
    an eval is already running.  The same check works for every lm-eval task, not only
    DROP, and avoids trying to infer the benchmark's target schema.
    """
    if not isinstance(target, str):
        return None
    for field_path, value in _mappings(document):
        keys = ",".join(str(key) for key in value)
        if target == keys:
            return TargetContractViolation(field_path=field_path, target=target)
    return None


def _mappings(document: Mapping[str, Any], prefix: str = "") -> Sequence[tuple[str, Mapping[str, Any]]]:
    found: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in document.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            found.append((path, value))
            found.extend(_mappings(value, path))
    return found


def _target(config: dict[str, Any], document: Mapping[str, Any]) -> object:
    target_spec = config["doc_to_target"]
    if callable(target_spec):
        return target_spec(document)
    if target_spec in document:
        return document[target_spec]
    return apply_template(target_spec, document)


def _sample_split(config: dict[str, Any]) -> str | None:
    return config.get("test_split") or config.get("validation_split") or config.get("training_split")


def _processed_documents(config: dict[str, Any], docs_per_task: int) -> Sequence[Mapping[str, Any]]:
    if config.get("custom_dataset") or not config.get("dataset_path"):
        raise ValueError("task uses a custom or missing dataset")
    split = _sample_split(config)
    if split is None:
        raise ValueError("task has no dataset split")
    raw_docs = datasets.load_dataset(
        path=config["dataset_path"],
        name=config.get("dataset_name"),
        split=split,
        streaming=True,
        **(config.get("dataset_kwargs") or {}),
    )
    samples = list(raw_docs.take(docs_per_task))
    if not samples:
        raise ValueError(f"split {split!r} is empty")
    process_docs = config.get("process_docs")
    if process_docs is None:
        return samples
    return list(process_docs(datasets.Dataset.from_list(samples)))


def audit_task(config: dict[str, Any], docs_per_task: int) -> Sequence[TargetContractViolation]:
    """Check the rendered targets for a bounded stream sample from one task."""
    violations = []
    for document in _processed_documents(config, docs_per_task):
        if violation := mapping_key_target_violation(document, _target(config, document)):
            violations.append(violation)
    return violations


def _candidate_task_entries(task_manager: TaskManager) -> list[tuple[str, Any]]:
    entries = []
    for task_name, entry in task_manager.task_index.items():
        if entry.kind.name != "TASK" or entry.yaml_path is None:
            continue
        if isinstance(entry.cfg.get("doc_to_target"), str):
            entries.append((task_name, entry))
    return entries


@contextmanager
def _task_deadline(timeout: int):
    """Bound one optional remote task source without leaving the audit wedged."""
    previous_handler = signal.getsignal(signal.SIGALRM)

    def raise_timeout(_signal_number: int, _frame: FrameType | None) -> None:
        raise TimeoutError(f"task audit exceeded {timeout} seconds")

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--docs-per-task", type=int, default=DEFAULT_DOCS_PER_TASK)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--task-timeout", type=int, default=DEFAULT_TASK_TIMEOUT)
    parser.add_argument("--tasks", help="Comma-separated task names to audit instead of a random sample")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.sample_size < 1 or args.docs_per_task < 1 or args.task_timeout < 1:
        raise SystemExit("--sample-size, --docs-per-task, and --task-timeout must be positive")

    task_manager = TaskManager(include_path=[str(EVALCHEMY_TASK_OVERRIDES)])
    candidates = _candidate_task_entries(task_manager)
    if args.tasks:
        requested = {task.strip() for task in args.tasks.split(",") if task.strip()}
        selected = [(task_name, entry) for task_name, entry in candidates if task_name in requested]
        missing = requested - {task_name for task_name, _ in selected}
        if missing:
            raise SystemExit(f"unknown or unsupported task(s): {', '.join(sorted(missing))}")
    else:
        selected = random.Random(args.seed).sample(candidates, min(args.sample_size, len(candidates)))
    checked = 0
    skipped: list[tuple[str, str]] = []
    violations: list[tuple[str, TargetContractViolation]] = []
    for task_name, entry in selected:
        try:
            with _task_deadline(args.task_timeout):
                config = load_yaml(entry.yaml_path, resolve_func=True)
                task_violations = audit_task(config, args.docs_per_task)
        except Exception as exc:  # Network, gated datasets, and optional task dependencies are audit skips.
            skipped.append((task_name, f"{type(exc).__name__}: {exc}"))
            continue
        checked += 1
        violations.extend((task_name, violation) for violation in task_violations)

    print(
        f"lm-eval target audit: selected={len(selected)} checked={checked} "
        f"skipped={len(skipped)} violations={len(violations)}"
    )
    for task_name, violation in violations:
        print(f"VIOLATION {task_name}: target={violation.target!r} matches mapping {violation.field_path}")
    for task_name, reason in skipped:
        print(f"SKIPPED {task_name}: {reason}")
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
