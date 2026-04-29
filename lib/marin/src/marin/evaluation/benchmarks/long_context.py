# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic long-context sanity checks for Marin eval runs.

These bespoke tasks are intentionally small and deterministic. `passkey`
places one generated identifier inside filler text and asks the model to copy
it back. `kv` builds a generated key-value registry and asks for one value.
They are useful for catching obvious context-extension failures, but they are
not a replacement for external long-context suites such as lm-eval tasks,
RULER, or NIAH.
"""

import json
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from marin.evaluation.evaluation_config import EvalTaskConfig

_NON_WORD_RE = re.compile(r"[^a-z0-9]+")
_FILLER_VOCAB = (
    "amber",
    "cedar",
    "delta",
    "elm",
    "flint",
    "grove",
    "harbor",
    "island",
    "juniper",
    "keystone",
    "lumen",
    "meadow",
)


class LongContextFamily(StrEnum):
    PASSKEY = "passkey"
    KV = "kv"


@dataclass(frozen=True)
class LongContextExample:
    example_id: str
    prompt: str
    gold_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LongContextTask:
    task_name: str
    family: LongContextFamily
    context_len: int
    max_gen_toks: int
    metric_names: tuple[str, ...]
    examples: tuple[LongContextExample, ...]


def build_long_context_task(
    eval_task: EvalTaskConfig,
    *,
    max_eval_instances: int | None = None,
) -> LongContextTask:
    task_kwargs = dict(eval_task.task_kwargs or {})
    family = LongContextFamily(task_kwargs["family"])
    context_len = int(task_kwargs["context_len"])
    max_gen_toks = int(task_kwargs.get("max_gen_toks", 32))

    if family is LongContextFamily.PASSKEY:
        examples = _build_passkey_examples(context_len=context_len, task_kwargs=task_kwargs)
        metric_names = ("exact_match",)
    elif family is LongContextFamily.KV:
        examples = _build_kv_examples(context_len=context_len, task_kwargs=task_kwargs)
        metric_names = ("exact_match",)
    else:
        raise ValueError(f"Unsupported long-context family: {family}")

    if max_eval_instances is not None:
        examples = examples[:max_eval_instances]

    return LongContextTask(
        task_name=eval_task.name,
        family=family,
        context_len=context_len,
        max_gen_toks=max_gen_toks,
        metric_names=metric_names,
        examples=tuple(examples),
    )


def evaluate_long_context_tasks(
    evals: Sequence[EvalTaskConfig],
    *,
    completion_fn: Callable[[list[str], int], list[str]],
    max_eval_instances: int | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for eval_task in evals:
        task = build_long_context_task(eval_task, max_eval_instances=max_eval_instances)
        prompts = [example.prompt for example in task.examples]
        predictions = completion_fn(prompts, task.max_gen_toks)
        if len(predictions) != len(prompts):
            raise ValueError(
                f"Completion function returned {len(predictions)} predictions for {len(prompts)} prompts "
                f"on task {task.task_name!r}"
            )
        results.append(score_long_context_task(task, predictions))

    return results


def write_long_context_results(output_dir: str | Path, results: Sequence[dict[str, Any]]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_tasks = []
    for result in results:
        task_name = result["task_name"]
        task_summary = {key: value for key, value in result.items() if key != "examples"}
        summary_tasks.append(task_summary)
        (output_path / f"{task_name}.json").write_text(json.dumps(result, indent=2, sort_keys=True))

    summary_payload = {"tasks": summary_tasks}
    (output_path / "results.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True))


def score_long_context_task(task: LongContextTask, predictions: Sequence[str]) -> dict[str, Any]:
    exact_total = 0.0
    example_results: list[dict[str, Any]] = []

    for example, prediction in zip(task.examples, predictions, strict=True):
        exact_match = float(_normalize_text(prediction) == _normalize_text(example.gold_answer))
        exact_total += exact_match
        example_result: dict[str, Any] = {
            "example_id": example.example_id,
            "prediction": prediction,
            "gold_answer": example.gold_answer,
            "correct": bool(exact_match),
            "metadata": example.metadata,
        }
        example_results.append(example_result)

    num_examples = len(task.examples)
    if num_examples == 0:
        raise ValueError(f"Long-context task {task.task_name!r} produced zero examples")

    metrics: dict[str, float] = {"exact_match": exact_total / num_examples}

    return {
        "task_name": task.task_name,
        "family": task.family.value,
        "context_len": task.context_len,
        "max_gen_toks": task.max_gen_toks,
        "num_examples": num_examples,
        "metrics": metrics,
        "examples": example_results,
    }


def _build_passkey_examples(*, context_len: int, task_kwargs: dict[str, Any]) -> list[LongContextExample]:
    num_examples = int(task_kwargs.get("num_examples", 128))
    seed = int(task_kwargs.get("seed", 0))
    rng = random.Random(seed)

    examples: list[LongContextExample] = []
    for index in range(num_examples):
        passkey = f"key-{rng.randrange(10**12):012d}"
        filler_prefix = _build_filler(target_units=max(context_len // 2, 32), rng=rng)
        filler_suffix = _build_filler(target_units=max(context_len // 2, 32), rng=rng)
        prompt = (
            "Read the following long context carefully.\n\n"
            f"{filler_prefix}\n\n"
            f"Embedded passkey: {passkey}\n\n"
            f"{filler_suffix}\n\n"
            "Question: What is the embedded passkey?\n"
            "Answer with only the passkey."
        )
        examples.append(
            LongContextExample(
                example_id=f"passkey-{index}",
                prompt=prompt,
                gold_answer=passkey,
                metadata={"family": LongContextFamily.PASSKEY.value, "context_len": context_len},
            )
        )

    return examples


def _build_kv_examples(*, context_len: int, task_kwargs: dict[str, Any]) -> list[LongContextExample]:
    num_examples = int(task_kwargs.get("num_examples", 128))
    seed = int(task_kwargs.get("seed", 0))
    num_pairs = int(task_kwargs.get("num_pairs", max(8, context_len // 16)))
    rng = random.Random(seed)

    examples: list[LongContextExample] = []
    for index in range(num_examples):
        pairs = [(f"item_{pair_index:04d}", f"value_{rng.randrange(10**8):08d}") for pair_index in range(num_pairs)]
        target_key, target_value = rng.choice(pairs)
        serialized_pairs = "\n".join(f"{key}: {value}" for key, value in pairs)
        filler = _build_filler(target_units=max(context_len // 4, 16), rng=rng)
        prompt = (
            "You are given a long registry of key-value pairs.\n\n"
            f"{serialized_pairs}\n\n"
            f"{filler}\n\n"
            f"Question: What is the value for {target_key}?\n"
            "Answer with only the value."
        )
        examples.append(
            LongContextExample(
                example_id=f"kv-{index}",
                prompt=prompt,
                gold_answer=target_value,
                metadata={
                    "family": LongContextFamily.KV.value,
                    "context_len": context_len,
                    "target_key": target_key,
                },
            )
        )

    return examples


def _build_filler(*, target_units: int, rng: random.Random) -> str:
    fragments: list[str] = []
    num_units = 0
    while num_units < target_units:
        fragment_words = [rng.choice(_FILLER_VOCAB) for _ in range(min(16, target_units - num_units))]
        fragments.append(" ".join(fragment_words))
        num_units += len(fragment_words)
    return " ".join(fragments)


def _normalize_text(text: str) -> str:
    normalized = _NON_WORD_RE.sub(" ", text.lower()).strip()
    return re.sub(r"\s+", " ", normalized)
