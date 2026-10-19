import logging
import collections
import dataclasses
import json

from .evaluation_main import (
    OutputExample,
    test_instruction_following_strict,
    test_instruction_following_loose,
    read_prompt_to_response_dict,
    read_prompt_list,
)


PER_PROMPT_FOLLOW_RATE = "per_prompt_follow_rate"
"""Report field mapping each prompt to its strict and loose follow rates."""


def get_report(outputs):
    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0

    tier0_total = collections.defaultdict(int)
    tier0_correct = collections.defaultdict(int)

    tier1_total = collections.defaultdict(int)
    tier1_correct = collections.defaultdict(int)

    for example in outputs:
        follow_instruction_list = example.follow_instruction_list
        instruction_id_list = example.instruction_id_list

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)

        for instruction_id, followed_or_not in zip(instruction_id_list, follow_instruction_list):
            instruction_id = instruction_id.split(":")[0]
            tier0_total[instruction_id] += 1
            if followed_or_not:
                tier0_correct[instruction_id] += 1

        for instruction_id, followed_or_not in zip(instruction_id_list, follow_instruction_list):
            tier1_total[instruction_id] += 1
            if followed_or_not:
                tier1_correct[instruction_id] += 1

    for instruction_id in sorted(tier0_total.keys()):
        tight_accuracy = tier0_correct[instruction_id] / tier0_total[instruction_id]
        logging.info(f"tier0 accuracy {instruction_id} {tight_accuracy}")

    for instruction_id in sorted(tier1_total.keys()):
        loose_accuracy = tier1_correct[instruction_id] / tier1_total[instruction_id]
        logging.info(f"tier1 accuracy {instruction_id} {loose_accuracy}")

    return {"prompt-level": prompt_correct / prompt_total, "instruction-level": instruction_correct / instruction_total}


def grade_accuracy(response_filename):
    """Check each response and retain all instruction verdicts."""
    inputs = read_prompt_list(response_filename)
    responses = read_prompt_to_response_dict(response_filename)
    return [
        {
            "prompt": inp.prompt,
            "instruction_id_list": inp.instruction_id_list,
            "strict_instruction_pass": test_instruction_following_strict(inp, responses).follow_instruction_list,
            "loose_instruction_pass": test_instruction_following_loose(inp, responses).follow_instruction_list,
        }
        for inp in inputs
    ]


def aggregate_accuracy(graded):
    """Reduce the loose instruction verdicts with their original prompt and instruction denominators."""
    outputs = [
        OutputExample(
            instruction_id_list=row["instruction_id_list"],
            prompt=row["prompt"],
            response="",
            follow_all_instructions=all(row["loose_instruction_pass"]),
            follow_instruction_list=row["loose_instruction_pass"],
        )
        for row in graded
    ]
    return get_report(outputs)


def evaluate_accuracy(response_filename):
    graded = grade_accuracy(response_filename)
    return {
        **aggregate_accuracy(graded),
        PER_PROMPT_FOLLOW_RATE: {
            row["prompt"]: {
                protocol: float(all(row[f"{protocol}_instruction_pass"]))
                for protocol in ("strict", "loose")
            }
            for row in graded
        },
    }
