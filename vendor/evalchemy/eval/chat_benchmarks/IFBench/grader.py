"""IFBench grading entry point.

This module is the single evaluation surface used by ``eval_instruct.py``. It loads
the per-prompt response file written during generation, runs every constraint listed
in each row's ``instruction_id_list`` against the model's response via the matching
:class:`instructions.Instruction` subclass, and reports:

  * ``strict`` / ``loose`` variants -- loose strips markdown emphasis (``*``) and
    re-runs each checker against the response with the first line, last line, or
    both removed, scoring a constraint as followed if ANY variant passes (the same
    upper-bound protocol IFEval and the AI2 IFBench reference implementation ship).
  * ``prompt-level``        -- fraction of prompts where ALL constraints hold.
  * ``instruction-level``   -- fraction of individual constraints satisfied.
  * ``per_type``            -- per-constraint-type accuracy (the part before the
    first ``":"`` of the instruction id, e.g. ``count`` / ``format`` / ``ratio``).

The constraint checkers themselves are vendored verbatim from
``allenai/IFBench`` (Apache-2.0, copyright 2025 Allen Institute for AI) under
``instructions.py`` / ``instructions_registry.py`` / ``instructions_util.py`` --
the canonical graders for the ``allenai/IFBench_test`` dataset -- so the scores
this benchmark reports are directly comparable to the IFBench paper's.
"""

import collections
import dataclasses
import json
import logging
from typing import Dict, List, Optional, Union

from .instructions_registry import INSTRUCTION_DICT

logger = logging.getLogger(__name__)
PER_PROMPT_OUTCOMES = "per_prompt_outcomes"


@dataclasses.dataclass
class InputExample:
    """One IFBench prompt plus its verifiable constraints."""

    key: int
    instruction_id_list: List[str]
    prompt: str
    kwargs: List[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    """Per-prompt evaluation outcome."""

    instruction_id_list: List[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: List[bool]


def read_prompt_list(input_jsonl_filename):
    """Read prompt + constraint metadata from a jsonl of IFBench examples."""
    inputs = []
    with open(input_jsonl_filename, "r") as f:
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


def read_prompt_to_response_dict(input_jsonl_filename):
    """Map each prompt to the model's response, keyed by the prompt text."""
    return_dict = {}
    with open(input_jsonl_filename, "r") as f:
        for line in f:
            example = json.loads(line)
            return_dict[example["prompt"]] = example["response"]
    return return_dict


def _build_instruction(instruction_id, kwargs_for_instruction, prompt):
    """Instantiate and parameterize the checker for one constraint.

    ``None``-valued kwargs are dropped before ``build_description`` (the AI2 strict
    path does the same) so the checker only sees the parameters that are actually
    set for this constraint.
    """
    instruction_cls = INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)
    filtered = {k: v for k, v in kwargs_for_instruction.items() if v is not None}
    instruction.build_description(**filtered)
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
        instruction.build_description(prompt=prompt)
    return instruction


def test_instruction_following_strict(inp, prompt_to_response):
    """Strict: score each constraint against the raw response only."""
    response = prompt_to_response[inp.prompt]
    is_following_list = []
    for index, instruction_id in enumerate(inp.instruction_id_list):
        instruction = _build_instruction(instruction_id, inp.kwargs[index], inp.prompt)
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


def test_instruction_following_loose(inp, prompt_to_response):
    """Loose: also accept any of the markdown-stripped / line-trimmed variants."""
    response = prompt_to_response[inp.prompt]
    if response is None:
        return OutputExample(
            instruction_id_list=inp.instruction_id_list,
            prompt=inp.prompt,
            response="",
            follow_all_instructions=False,
            follow_instruction_list=[False] * len(inp.instruction_id_list),
        )

    lines = response.split("\n")
    response_remove_first = "\n".join(lines[1:]).strip()
    response_remove_last = "\n".join(lines[:-1]).strip()
    response_remove_both = "\n".join(lines[1:-1]).strip()
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

    is_following_list = []
    for index, instruction_id in enumerate(inp.instruction_id_list):
        instruction = _build_instruction(instruction_id, inp.kwargs[index], inp.prompt)
        is_following = False
        for candidate in all_responses:
            if candidate.strip() and instruction.check_following(candidate):
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


def _summarize(outputs):
    """Aggregate per-prompt, per-constraint, and per-type accuracies from outputs."""
    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0
    per_type_total = collections.defaultdict(int)
    per_type_correct = collections.defaultdict(int)

    for example in outputs:
        follow_instruction_list = example.follow_instruction_list
        instruction_id_list = example.instruction_id_list

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)

        for instruction_id, followed in zip(
            instruction_id_list, follow_instruction_list
        ):
            category = instruction_id.split(":")[0]
            per_type_total[category] += 1
            if followed:
                per_type_correct[category] += 1

    return {
        "prompt-level": prompt_correct / prompt_total if prompt_total else 0.0,
        "instruction-level": instruction_correct / instruction_total
        if instruction_total
        else 0.0,
        "per_type": {
            category: per_type_correct[category] / per_type_total[category]
            for category in sorted(per_type_total)
        },
    }


def grade_accuracy(response_filename):
    """Check each response and retain its strict and loose instruction verdicts."""
    inputs = read_prompt_list(response_filename)
    prompt_to_response = read_prompt_to_response_dict(response_filename)
    return [
        {
            "prompt": inp.prompt,
            "instruction_id_list": inp.instruction_id_list,
            "strict_instruction_pass": test_instruction_following_strict(inp, prompt_to_response).follow_instruction_list,
            "loose_instruction_pass": test_instruction_following_loose(inp, prompt_to_response).follow_instruction_list,
        }
        for inp in inputs
    ]


def aggregate_accuracy(graded):
    """Reduce instruction verdicts using prompt, instruction, and constraint-type denominators."""
    reports = {
        protocol: _summarize([
            OutputExample(
                instruction_id_list=row["instruction_id_list"],
                prompt=row["prompt"],
                response="",
                follow_all_instructions=all(row[f"{protocol}_instruction_pass"]),
                follow_instruction_list=row[f"{protocol}_instruction_pass"],
            )
            for row in graded
        ])
        for protocol in ("strict", "loose")
    }
    return {
        f"{protocol}_{metric}": reports[protocol][source]
        for protocol in ("strict", "loose")
        for metric, source in (
            ("prompt_accuracy", "prompt-level"),
            ("instruction_accuracy", "instruction-level"),
            ("per_type", "per_type"),
        )
    }


def evaluate_accuracy(response_filename):
    graded = grade_accuracy(response_filename)
    return {**aggregate_accuracy(graded), PER_PROMPT_OUTCOMES: graded}
