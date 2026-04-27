# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict + loose accuracy scorer.

Ported from `allenai/IFBench:evaluation_lib.py` with one change: the
constraint registry is passed in as a parameter instead of hardcoded, so
the same code scores both IFBench's 58 OOD constraints and IFEvalG's 54
train-eligible constraints. Scoring logic is byte-identical to upstream.

The default registry is the union of both vendored sets.
"""

from __future__ import annotations

import collections
import dataclasses
import json
from typing import Any

from . import INSTRUCTION_DICT_ALL


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[dict[str, str | int | None]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def read_prompt_list(input_jsonl_filename: str) -> list[InputExample]:
    """Read inputs from jsonl in the IFBench_test format."""
    inputs = []
    with open(input_jsonl_filename) as f:
        for line in f:
            example = json.loads(line)
            inputs.append(
                InputExample(
                    key=example["key"],
                    instruction_id_list=example["instruction_id_list"],
                    prompt=example["prompt"],
                    kwargs=example["kwargs"],
                )
            )
    return inputs


def read_prompt_to_response_dict(input_jsonl_filename: str) -> dict[str, str]:
    """Map prompt → response from a jsonl of {"prompt": ..., "response": ...}.

    Stores both the original key and an rstrip-normalised key. Some upstream
    fixtures (IFBench's `sample_output.jsonl` vs `IFBench_test.jsonl`) differ
    in trailing whitespace per row; upstream's `run_eval.py` crashes on this.
    Normalising trailing whitespace gives the intended match without
    deviating from upstream semantics on the byte-equal cases.
    """
    out: dict[str, str] = {}
    with open(input_jsonl_filename) as f:
        for line in f:
            example = json.loads(line)
            prompt = example["prompt"]
            out[prompt] = example["response"]
            out[prompt.rstrip()] = example["response"]
    return out


def _lookup_response(prompt_to_response: dict[str, str], prompt: str) -> str | None:
    """Look up a response, falling back to rstrip-normalised key."""
    if prompt in prompt_to_response:
        return prompt_to_response[prompt]
    return prompt_to_response.get(prompt.rstrip())


def test_instruction_following_strict(
    inp: InputExample,
    response: str | None,
    registry: dict[str, type] = INSTRUCTION_DICT_ALL,
) -> OutputExample:
    """Strict: run the verifier on the raw response, no surface tweaks."""
    if response is None:
        response = ""
    is_following_list: list[bool] = []
    for index, instruction_id in enumerate(inp.instruction_id_list):
        instruction_cls = registry[instruction_id]
        instruction = instruction_cls(instruction_id)
        # Drop None-valued kwargs upstream-style before passing to build_description.
        clean_kwargs = {k: v for k, v in (inp.kwargs[index] or {}).items() if v is not None}
        instruction.build_description(**clean_kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response and response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp: InputExample,
    response: str | None,
    registry: dict[str, type] = INSTRUCTION_DICT_ALL,
) -> OutputExample:
    """Loose: also accept the response with markdown-bold stripped and/or first/last lines removed."""
    if response is None:
        return OutputExample(
            instruction_id_list=inp.instruction_id_list,
            prompt=inp.prompt,
            response="",
            follow_all_instructions=False,
            follow_instruction_list=[False] * len(inp.instruction_id_list),
        )

    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    is_following_list: list[bool] = []
    for index, instruction_id in enumerate(inp.instruction_id_list):
        instruction_cls = registry[instruction_id]
        instruction = instruction_cls(instruction_id)
        # Drop None-valued kwargs upstream-style (strict does this via in-place mutation;
        # we do it explicitly so the two paths are independent).
        clean_kwargs = {k: v for k, v in (inp.kwargs[index] or {}).items() if v is not None}
        instruction.build_description(**clean_kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r_variant in all_responses:
            if r_variant.strip() and instruction.check_following(r_variant):
                is_following = True
                break
        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def aggregate(outputs: list[OutputExample]) -> dict[str, Any]:
    """Compute the canonical IFBench/IFEval report numbers."""
    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0
    tier0_total: dict[str, int] = collections.defaultdict(int)
    tier0_correct: dict[str, int] = collections.defaultdict(int)
    tier1_total: dict[str, int] = collections.defaultdict(int)
    tier1_correct: dict[str, int] = collections.defaultdict(int)

    for ex in outputs:
        prompt_total += 1
        if all(ex.follow_instruction_list):
            prompt_correct += 1
        instruction_total += len(ex.instruction_id_list)
        instruction_correct += sum(ex.follow_instruction_list)
        for instruction_id, ok in zip(ex.instruction_id_list, ex.follow_instruction_list, strict=True):
            tier0 = instruction_id.split(":")[0]
            tier0_total[tier0] += 1
            if ok:
                tier0_correct[tier0] += 1
            tier1_total[instruction_id] += 1
            if ok:
                tier1_correct[instruction_id] += 1

    return {
        "prompt_level_accuracy": prompt_correct / prompt_total if prompt_total else 0.0,
        "instruction_level_accuracy": instruction_correct / instruction_total if instruction_total else 0.0,
        "tier0": {k: tier0_correct[k] / tier0_total[k] for k in tier0_total},
        "tier1": {k: tier1_correct[k] / tier1_total[k] for k in tier1_total},
        "n_prompts": prompt_total,
        "n_instructions": instruction_total,
    }


def score_jsonl(
    input_jsonl: str,
    response_jsonl: str,
    registry: dict[str, type] = INSTRUCTION_DICT_ALL,
) -> dict[str, dict[str, Any]]:
    """Run both strict and loose scoring against IFBench-formatted jsonl files."""
    inputs = read_prompt_list(input_jsonl)
    prompt_to_response = read_prompt_to_response_dict(response_jsonl)
    out: dict[str, dict[str, Any]] = {}
    for label, fn in (("strict", test_instruction_following_strict), ("loose", test_instruction_following_loose)):
        results = [fn(inp, _lookup_response(prompt_to_response, inp.prompt), registry) for inp in inputs]
        out[label] = aggregate(results)
    return out
