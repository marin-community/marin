"""Per-prompt strict and loose instruction checks."""

import dataclasses
from typing import Dict, List, Optional, Union

from .instructions_registry import INSTRUCTION_DICT


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
