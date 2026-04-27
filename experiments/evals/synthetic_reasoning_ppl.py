# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in synthetic reasoning PPL dev slices for issue #5052.

This first cut keeps scope intentionally small and deterministic:

- one pure-Python "stepmath" arithmetic profile
- one native algorithmic solver (`euclid_gcd`)
- one CLRS-style alias (`clrs_binary_search`)
- two renderers (`canonical_json`, `oai_chat_symbolic`)

The output is a tiny raw-text registry for perplexity-gap experiments. Each row keeps the
structured surface form under ``surface`` and uses a deterministic JSON serialization in
``text`` so the raw eval path sees the exact symbolic / JSON-like trace.
"""

from __future__ import annotations

import json
import math
import posixpath
import random
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from rigging.filesystem import open_url
from zephyr.writers import atomic_rename

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.execution.executor import THIS_OUTPUT_PATH, ExecutorStep, VersionedValue, versioned
from marin.utils import fsspec_mkdirs

EPIC_5005 = 5005
ISSUE_5052 = 5052
SYNTHETIC_REASONING_SOURCE_COMMIT = "c4a59c3e1"
DEV_SEED_BASE = 1 << 30
DEV_SEED_STRIDE = 8


class SyntheticReasoningFamily(StrEnum):
    STEPMATH = "stepmath"
    NATIVE = "native"
    CLRS_STYLE = "clrs_style"


class SyntheticReasoningRenderer(StrEnum):
    CANONICAL_JSON = "canonical_json"
    OAI_CHAT_SYMBOLIC = "oai_chat_symbolic"


@dataclass(frozen=True)
class StepRecord:
    operation: str
    before: Any
    after: Any
    details: str


@dataclass(frozen=True)
class ExampleRecord:
    prompt: str
    problem: dict[str, Any]
    steps: list[StepRecord]
    final_answer: Any
    difficulty: dict[str, Any]


@dataclass(frozen=True)
class SyntheticReasoningPplSlice:
    family: SyntheticReasoningFamily
    task_name: str
    renderer: SyntheticReasoningRenderer
    seed_start: int
    seed_stop: int
    difficulty_bucket: str
    notes: str = ""
    solver_task_name: str | None = None

    @property
    def raw_relative_path(self) -> str:
        return posixpath.join(self.family.value, self.task_name, f"{self.renderer.value}.jsonl.gz")

    @property
    def registry_key(self) -> str:
        return posixpath.join("synthetic_reasoning_ppl", self.family.value, self.task_name, self.renderer.value)

    @property
    def tags(self) -> tuple[str, ...]:
        return (
            "synthetic_reasoning_ppl",
            f"epic:{EPIC_5005}",
            f"issue:{ISSUE_5052}",
            f"family:{self.family.value}",
            f"task:{self.task_name}",
            f"renderer:{self.renderer.value}",
            f"seed_range:{self.seed_start}:{self.seed_stop}",
        )


@dataclass
class SyntheticReasoningPplConfig:
    output_path: str | VersionedValue[str] = THIS_OUTPUT_PATH
    examples_per_slice: int = 8
    source_commit: str = SYNTHETIC_REASONING_SOURCE_COMMIT
    slices: tuple[SyntheticReasoningPplSlice, ...] = field(default_factory=lambda: SYNTHETIC_REASONING_PPL_SLICES)
    cache_key: dict[str, Any] | VersionedValue[dict[str, Any]] = field(default_factory=dict, repr=False)


def _slice(
    *,
    family: SyntheticReasoningFamily,
    task_name: str,
    renderer: SyntheticReasoningRenderer,
    seed_start: int,
    difficulty_bucket: str,
    notes: str = "",
    solver_task_name: str | None = None,
) -> SyntheticReasoningPplSlice:
    return SyntheticReasoningPplSlice(
        family=family,
        task_name=task_name,
        renderer=renderer,
        seed_start=seed_start,
        seed_stop=seed_start + DEV_SEED_STRIDE,
        difficulty_bucket=difficulty_bucket,
        notes=notes,
        solver_task_name=solver_task_name,
    )


SYNTHETIC_REASONING_PPL_SLICES: tuple[SyntheticReasoningPplSlice, ...] = (
    _slice(
        family=SyntheticReasoningFamily.STEPMATH,
        task_name="arithmetic",
        renderer=SyntheticReasoningRenderer.CANONICAL_JSON,
        seed_start=DEV_SEED_BASE,
        difficulty_bucket="easy",
        notes="Deterministic arithmetic traces with raw JSON surfaces.",
    ),
    _slice(
        family=SyntheticReasoningFamily.STEPMATH,
        task_name="arithmetic",
        renderer=SyntheticReasoningRenderer.OAI_CHAT_SYMBOLIC,
        seed_start=DEV_SEED_BASE,
        difficulty_bucket="easy",
        notes="Same held-out arithmetic traces rendered as OAI chat rows.",
    ),
    _slice(
        family=SyntheticReasoningFamily.NATIVE,
        task_name="euclid_gcd",
        renderer=SyntheticReasoningRenderer.CANONICAL_JSON,
        seed_start=DEV_SEED_BASE + 16,
        difficulty_bucket="medium",
        notes="Canonical symbolic rows from a native algorithmic solver.",
    ),
    _slice(
        family=SyntheticReasoningFamily.CLRS_STYLE,
        task_name="clrs_binary_search",
        renderer=SyntheticReasoningRenderer.OAI_CHAT_SYMBOLIC,
        seed_start=DEV_SEED_BASE + 24,
        difficulty_bucket="medium",
        notes="CLRS-style alias rendered as symbolic OAI chat messages.",
        solver_task_name="binary_search",
    ),
)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def _step(operation: str, before: Any, after: Any, details: str) -> StepRecord:
    return StepRecord(operation=operation, before=before, after=after, details=details)


def _stepmath_arithmetic(seed: int) -> ExampleRecord:
    rng = random.Random(seed)
    operands = [rng.randint(2, 12) for _ in range(4)]
    left_sum = operands[0] + operands[1]
    product = left_sum * operands[2]
    final_answer = product - operands[3]
    expression = f"(({operands[0]} + {operands[1]}) * {operands[2]}) - {operands[3]}"

    return ExampleRecord(
        prompt=f"Simplify the arithmetic expression {expression}.",
        problem={"expression": expression, "operands": operands},
        steps=[
            _step(
                "EVAL_ADD",
                {"expression": f"{operands[0]} + {operands[1]}"},
                {"value": left_sum},
                f"{operands[0]} + {operands[1]} = {left_sum}",
            ),
            _step(
                "EVAL_MUL",
                {"expression": f"{left_sum} * {operands[2]}"},
                {"value": product},
                f"{left_sum} * {operands[2]} = {product}",
            ),
            _step(
                "EVAL_SUB",
                {"expression": f"{product} - {operands[3]}"},
                {"value": final_answer},
                f"{product} - {operands[3]} = {final_answer}",
            ),
        ],
        final_answer=final_answer,
        difficulty={"num_operations": 3, "max_abs_operand": max(abs(value) for value in operands)},
    )


def _euclid_gcd(seed: int) -> ExampleRecord:
    rng = random.Random(seed)
    a = rng.randint(100, 3_000)
    b = rng.randint(100, 3_000)
    x, y = max(a, b), min(a, b)
    steps: list[StepRecord] = []
    while y != 0:
        quotient, remainder = divmod(x, y)
        steps.append(
            _step(
                "EUCLID_MOD",
                {"x": x, "y": y},
                {"x": y, "y": remainder},
                f"{x} = {quotient} * {y} + {remainder}",
            )
        )
        x, y = y, remainder

    return ExampleRecord(
        prompt=f"Use Euclid's algorithm to compute gcd({a}, {b}).",
        problem={"a": a, "b": b},
        steps=steps,
        final_answer=x,
        difficulty={"max_value": max(a, b), "num_steps": len(steps)},
    )


def _binary_search(seed: int) -> ExampleRecord:
    rng = random.Random(seed)
    n = rng.randint(8, 14)
    values = sorted(rng.sample(range(-80, 81), n))
    choose_present = rng.random() < 0.7
    if choose_present:
        target = values[rng.randrange(len(values))]
    else:
        missing_values = sorted(set(range(-100, 101)) - set(values))
        target = missing_values[rng.randrange(len(missing_values))]

    lo = 0
    hi = len(values) - 1
    answer = -1
    steps: list[StepRecord] = []
    while lo <= hi:
        mid = (lo + hi) // 2
        current = values[mid]
        before = {"lo": lo, "hi": hi, "mid": mid, "arr[mid]": current}
        if current == target:
            answer = mid
            steps.append(_step("COMPARE", before, {"result": "found", "index": mid}, "target found"))
            break
        if current < target:
            lo = mid + 1
            steps.append(
                _step("COMPARE", before, {"result": "search_right", "new_lo": lo, "hi": hi}, "arr[mid] < target")
            )
            continue
        hi = mid - 1
        steps.append(_step("COMPARE", before, {"result": "search_left", "lo": lo, "new_hi": hi}, "arr[mid] > target"))

    if answer == -1:
        steps.append(_step("TERMINATE", {"lo": lo, "hi": hi}, {"index": -1}, "interval exhausted"))

    return ExampleRecord(
        prompt=f"Use binary search to find {target} in sorted array {values}. Return -1 if absent.",
        problem={"array": values, "target": target},
        steps=steps,
        final_answer=answer,
        difficulty={"n": n, "present": choose_present},
    )


def _example_for_slice(slice_: SyntheticReasoningPplSlice, seed: int) -> ExampleRecord:
    if slice_.family == SyntheticReasoningFamily.STEPMATH and slice_.task_name == "arithmetic":
        return _stepmath_arithmetic(seed)
    if slice_.family == SyntheticReasoningFamily.NATIVE and slice_.task_name == "euclid_gcd":
        return _euclid_gcd(seed)
    if slice_.family == SyntheticReasoningFamily.CLRS_STYLE and slice_.task_name == "clrs_binary_search":
        return _binary_search(seed)
    raise ValueError(f"unsupported synthetic reasoning slice: {slice_.registry_key}")


def _render_symbolic_step(step: StepRecord) -> str:
    return f"[{step.operation}] {_json(step.before)} -> {_json(step.after)}"


def _canonical_surface(example: ExampleRecord) -> dict[str, Any]:
    return {
        "prompt": example.prompt,
        "problem": example.problem,
        "steps": [
            {
                "index": index,
                "operation": step.operation,
                "before": step.before,
                "after": step.after,
                "details": step.details,
            }
            for index, step in enumerate(example.steps)
        ],
        "final_answer": example.final_answer,
    }


def _oai_chat_surface(example: ExampleRecord) -> dict[str, Any]:
    assistant_lines = ["Step-by-step solution:"]
    for index, step in enumerate(example.steps, start=1):
        assistant_lines.append(f"{index}. {_render_symbolic_step(step)}")
    assistant_lines.append(f"Final answer: {_json(example.final_answer)}")
    return {
        "messages": [
            {"role": "user", "content": example.prompt},
            {"role": "assistant", "content": "\n".join(assistant_lines)},
        ]
    }


def _surface_for_renderer(renderer: SyntheticReasoningRenderer, example: ExampleRecord) -> dict[str, Any]:
    if renderer == SyntheticReasoningRenderer.CANONICAL_JSON:
        return _canonical_surface(example)
    if renderer == SyntheticReasoningRenderer.OAI_CHAT_SYMBOLIC:
        return _oai_chat_surface(example)
    raise ValueError(f"unsupported renderer: {renderer}")


def _metadata_for_row(
    *,
    slice_: SyntheticReasoningPplSlice,
    seed: int,
    source_commit: str,
    difficulty: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generator_family": slice_.family.value,
        "task_name": slice_.task_name,
        "solver_task_name": slice_.solver_task_name or slice_.task_name,
        "renderer": slice_.renderer.value,
        "seed": seed,
        "seed_range": {"start": slice_.seed_start, "stop": slice_.seed_stop},
        "difficulty": difficulty,
        "difficulty_bucket": slice_.difficulty_bucket,
        "source_commit": source_commit,
    }


def _record_for_seed(
    *,
    slice_: SyntheticReasoningPplSlice,
    seed: int,
    source_commit: str,
) -> dict[str, Any]:
    example = _example_for_slice(slice_, seed)
    surface = _surface_for_renderer(slice_.renderer, example)
    metadata = _metadata_for_row(slice_=slice_, seed=seed, source_commit=source_commit, difficulty=example.difficulty)
    return {
        "id": f"{slice_.family.value}:{slice_.task_name}:{seed}",
        "text": _json(surface),
        "surface": surface,
        "metadata": metadata,
        "source": slice_.registry_key,
    }


def _write_slice(path: str, records: list[dict[str, Any]]) -> None:
    fsspec_mkdirs(posixpath.dirname(path), exist_ok=True)
    with atomic_rename(path) as temp_path:
        with open_url(temp_path, "wt", encoding="utf-8", compression="gzip") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
                handle.write("\n")


def generate_synthetic_reasoning_ppl(config: SyntheticReasoningPplConfig) -> dict[str, Any]:
    if config.examples_per_slice <= 0:
        raise ValueError("examples_per_slice must be positive")

    output_path = str(config.output_path)
    fsspec_mkdirs(output_path, exist_ok=True)
    manifest_slices: list[dict[str, Any]] = []

    for slice_ in config.slices:
        max_examples = slice_.seed_stop - slice_.seed_start
        if config.examples_per_slice > max_examples:
            raise ValueError(
                f"examples_per_slice={config.examples_per_slice} exceeds held-out range for {slice_.registry_key}"
            )

        records = [
            _record_for_seed(slice_=slice_, seed=seed, source_commit=config.source_commit)
            for seed in range(slice_.seed_start, slice_.seed_start + config.examples_per_slice)
        ]
        output_file = posixpath.join(output_path, slice_.raw_relative_path)
        _write_slice(output_file, records)
        manifest_slices.append(
            {
                "registry_key": slice_.registry_key,
                "output_file": output_file,
                "family": slice_.family.value,
                "task_name": slice_.task_name,
                "renderer": slice_.renderer.value,
                "seed_range": {"start": slice_.seed_start, "stop": slice_.seed_stop},
                "difficulty_bucket": slice_.difficulty_bucket,
                "examples": len(records),
            }
        )

    manifest = {
        "source_commit": config.source_commit,
        "held_out_seed_base": DEV_SEED_BASE,
        "examples_per_slice": config.examples_per_slice,
        "slices": manifest_slices,
    }
    manifest_path = posixpath.join(output_path, "manifest.json")
    with atomic_rename(manifest_path) as temp_path:
        with open_url(temp_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def synthetic_reasoning_ppl_step(
    *,
    name: str = "raw/synthetic_reasoning_ppl/issue5052",
    examples_per_slice: int = 8,
    source_commit: str = SYNTHETIC_REASONING_SOURCE_COMMIT,
) -> ExecutorStep[SyntheticReasoningPplConfig]:
    return ExecutorStep(
        name=name,
        fn=generate_synthetic_reasoning_ppl,
        config=SyntheticReasoningPplConfig(
            examples_per_slice=examples_per_slice,
            source_commit=source_commit,
            cache_key=versioned(
                {
                    "examples_per_slice": examples_per_slice,
                    "source_commit": source_commit,
                    "slices": [
                        {
                            "family": slice_.family.value,
                            "task_name": slice_.task_name,
                            "renderer": slice_.renderer.value,
                            "seed_start": slice_.seed_start,
                            "seed_stop": slice_.seed_stop,
                            "difficulty_bucket": slice_.difficulty_bucket,
                            "solver_task_name": slice_.solver_task_name,
                        }
                        for slice_ in SYNTHETIC_REASONING_PPL_SLICES
                    ],
                }
            ),
        ),
    )


synthetic_reasoning_ppl_raw = synthetic_reasoning_ppl_step()


def synthetic_reasoning_raw_validation_sets(
    *,
    raw_root: str | None = None,
    synthetic_reasoning_raw: ExecutorStep | None = None,
) -> dict[str, RawTextEvaluationDataset]:
    datasets: dict[str, RawTextEvaluationDataset] = {}
    if raw_root is None and synthetic_reasoning_raw is None:
        synthetic_reasoning_raw = synthetic_reasoning_ppl_raw

    for slice_ in SYNTHETIC_REASONING_PPL_SLICES:
        if raw_root is not None:
            source: str | ExecutorStep = posixpath.join(raw_root, slice_.raw_relative_path)
        else:
            assert synthetic_reasoning_raw is not None
            source = synthetic_reasoning_raw.cd(slice_.raw_relative_path)
        datasets[slice_.registry_key] = raw_text_dataset(source, tags=slice_.tags)
    return datasets


def verify_example_record(
    example: ExampleRecord, *, family: SyntheticReasoningFamily, task_name: str
) -> tuple[bool, str]:
    if family == SyntheticReasoningFamily.STEPMATH and task_name == "arithmetic":
        expression = str(example.problem["expression"])
        lhs = eval(expression, {"__builtins__": {}}, {})
        return (True, "ok") if lhs == example.final_answer else (False, "stepmath_final_answer_mismatch")
    if family == SyntheticReasoningFamily.NATIVE and task_name == "euclid_gcd":
        expected = math.gcd(int(example.problem["a"]), int(example.problem["b"]))
        return (True, "ok") if expected == int(example.final_answer) else (False, "gcd_mismatch")
    if family == SyntheticReasoningFamily.CLRS_STYLE and task_name == "clrs_binary_search":
        values = list(example.problem["array"])
        target = int(example.problem["target"])
        expected = values.index(target) if target in values else -1
        return (True, "ok") if expected == int(example.final_answer) else (False, "binary_search_mismatch")
    return False, "unsupported_verifier"
