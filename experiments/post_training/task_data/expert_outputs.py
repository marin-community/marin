# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expert completion sampling for the M1 datasets (WP1's fixed-completion rows).

One served expert (E_X or E_R behind an OpenAI-compatible endpoint) samples one
completion per task prompt at explicitly recorded temperature and max_tokens. A
completion whose rendered text, reasoning included, exceeds the task's rendered-token
cap (4,096 for X, 16,384 for R) is dropped, never truncated, and reported, so over-cap
rows never reach the M1 loader. X completions are graded by the direction-dispatched
execution grader and R completions by the math verifier; only graded-correct
completions become M1 rows.

Rows are written in WP1's tokenized format: the whole conversation rendered once
through the shared chat template with the student's tokenizer, plus an assistant mask
covering the reasoning, the answer and the closing turn token. The text format would
drop ``reasoning_content``, and re-rendering at load time would let the M1 arm and the
M4 arm render the protocol differently.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from experiments.post_training.chat_template import FAMILY_TOKEN_MAPS, shared_template
from experiments.post_training.task_data.code_x import grade_x_task
from experiments.post_training.task_data.expert_traces import (
    ChatClient,
    ExpertResponse,
    chat_completion,
    rendered_completion_text,
)
from experiments.post_training.task_data.math_r import verify_final_answer


@dataclass(frozen=True)
class CompletionRow:
    """One sampled expert completion with its grade."""

    task_id: str
    prompt: list[dict[str, str]]
    response: ExpertResponse
    graded_correct: bool


@dataclass(frozen=True)
class CompletionBatch:
    """Kept rows plus the over-cap drop count the build manifest records."""

    rows: list[CompletionRow]
    dropped_over_cap: int


def sample_completions(
    tasks: list[dict[str, Any]],
    *,
    family: str,
    temperature: float,
    max_tokens: int,
    token_cap: int,
    client: ChatClient,
    model: str,
    grade: Callable[[dict[str, Any], str], bool],
    token_count: Callable[[str], int],
    max_concurrent: int,
) -> CompletionBatch:
    """Sample one completion per task at the given sampling parameters and grade it.

    Completions over ``token_cap`` on the rendered text, or truncated at
    ``max_tokens``, are dropped and counted. ``grade`` sees only the answer content
    (the direction-dispatched execution grader for X, the math verifier for R).
    """
    if max_tokens <= token_cap:
        raise ValueError(f"max_tokens ({max_tokens}) must exceed the token cap ({token_cap})")

    def sample(task: dict[str, Any]) -> CompletionRow | None:
        prompt = [{"role": "user", "content": task["prompt"]}]
        response = chat_completion(client, model=model, messages=prompt, temperature=temperature, max_tokens=max_tokens)
        if response.truncated or token_count(rendered_completion_text(family, response)) > token_cap:
            return None
        return CompletionRow(
            task_id=task["task_id"], prompt=prompt, response=response, graded_correct=grade(task, response.content)
        )

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        results = list(pool.map(sample, tasks))
    rows = [row for row in results if row is not None]
    return CompletionBatch(rows=rows, dropped_over_cap=len(tasks) - len(rows))


def grade_r_task(task: dict[str, Any], completion: str) -> bool:
    """Grade an R completion: its final answer must verify against the stored answer."""
    return verify_final_answer(task["answer"], completion)


GRADERS: dict[str, Callable[[dict[str, Any], str], bool]] = {"X": grade_x_task, "R": grade_r_task}


def tokenized_conversation(
    tokenizer, family: str, prompt: list[dict[str, str]], response: ExpertResponse
) -> dict[str, list[int]]:
    """Render ``prompt`` plus the response as one sequence with an assistant mask.

    The prompt renders with the generation prompt (the assistant turn opener), so the
    supervised span starts at the first reasoning or answer token and ends at the
    family's closing turn token; the template's trailing newline is not supervised
    and not kept. Raises if tokenizing the prompt alone does not give a prefix of the
    whole conversation, since a boundary merge would misplace the mask.
    """
    template = shared_template(FAMILY_TOKEN_MAPS[family])
    assistant = {"role": "assistant", "content": response.content, "reasoning_content": response.reasoning or None}
    prefix_text = tokenizer.apply_chat_template(
        prompt, chat_template=template, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template([*prompt, assistant], chat_template=template, tokenize=False)
    turn_close = FAMILY_TOKEN_MAPS[family].turn_close
    if not full_text.endswith(turn_close + "\n"):
        raise ValueError(f"rendered conversation does not end with {turn_close!r}")
    full_text = full_text[:-1]
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if full_ids[: len(prefix_ids)] != prefix_ids or len(full_ids) == len(prefix_ids):
        raise ValueError("the rendered prompt does not tokenize as a prefix of the rendered conversation")
    return {"input_ids": full_ids, "assistant_masks": [0] * len(prefix_ids) + [1] * (len(full_ids) - len(prefix_ids))}


def write_m1_rows(
    rows: list[CompletionRow],
    output_path: str,
    *,
    tokenizer,
    family: str,
    data_source: str,
    teacher_route: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Write the graded-correct rows in WP1's tokenized format; ungraded rows are dropped and counted.

    Each row keeps its prompt messages and ``task_id`` beside the tokens, so the M4
    arm can draw the same prompts in the same order.
    """
    kept = 0
    dropped = 0
    with open(output_path, "w") as handle:
        for row in rows:
            if not row.graded_correct:
                dropped += 1
                continue
            record = {
                "task_id": row.task_id,
                "prompt": row.prompt,
                "data_source": data_source,
                "teacher_route": teacher_route,
                **tokenized_conversation(tokenizer, family, row.prompt, row.response),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            kept += 1
    return {
        "kept": kept,
        "dropped_ungraded": dropped,
        "family": family,
        "data_source": data_source,
        "teacher_route": teacher_route,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
