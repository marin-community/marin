# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The pinned Ultra release's explicit DAPO/Skywork reconstruction contract."""

import copy
import json
from typing import Any

from taskcompendium.pipeline.models import NormalizationChange

DAPO = "BytedTsinghua-SIA/DAPO-Math-17k"
DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem."
)
DAPO_SUFFIX = 'Remember to put your answer on its own line after "Answer:".'


def restore_placeholder(data: dict[str, Any]) -> tuple[dict[str, Any], tuple[NormalizationChange, ...]]:
    """Apply fill_placeholders.py from Ultra pin482392c to an acquired immutable source row."""
    placeholder = data["_hf_question_placeholder"]
    source = data["placeholder_source"]
    if (source["dataset"], source["split"], source["row_index"]) != (
        placeholder["dataset"],
        placeholder["split"],
        int(placeholder["row"]),
    ):
        raise ValueError("Placeholder source identity does not match its declared recipe")
    record = source["record"]
    bare = record["prompt"][0]["content"]
    if placeholder["dataset"] == DAPO:
        if DAPO_PREFIX in bare:
            bare = bare.split(DAPO_PREFIX, 1)[1]
        if DAPO_SUFFIX in bare:
            bare = bare.rsplit(DAPO_SUFFIX, 1)[0]
    bare = bare.strip()
    if placeholder.get("mode") == "canonical":
        question = placeholder.get("lead", "") + bare + placeholder.get("trail", "")
    else:
        question = placeholder.get("prefix", "") + bare + placeholder.get("suffix", "")
    raw = record["reward_model"]["ground_truth"]
    if isinstance(raw, list) and raw:
        answer = str(raw[0])
    elif isinstance(raw, str):
        answer = raw.strip()
        if answer.startswith(("[", "{")) and answer.endswith(("]", "}")):
            try:
                parsed = json.loads(answer)
            except json.JSONDecodeError:
                parsed = answer
            answer = str(parsed[0]) if isinstance(parsed, list) and parsed else str(parsed)
    else:
        raise ValueError("Placeholder source lacks a supported ground_truth")
    if not bare or not answer:
        raise ValueError("Placeholder source question and answer must be nonempty")
    restored = copy.deepcopy(data)
    restored.pop("_hf_question_placeholder")
    restored.pop("placeholder_source")
    restored["placeholder_provenance"] = {key: value for key, value in source.items() if key != "record"}
    restored["question"] = question
    restored["expected_answer"] = answer
    restored["responses_create_params"]["input"][0]["content"] = question
    for matched in restored.get("matched_sources", []):
        if "expected_answer" in matched:
            matched["expected_answer"] = answer
    changes = (
        NormalizationChange(
            field="question",
            reason="Apply the pinned release's explicit external-source placeholder recipe",
            original=json.dumps(placeholder, ensure_ascii=False),
            replacement=question,
        ),
        NormalizationChange(
            field="expected_answer",
            reason="Restore the placeholder source reward_model.ground_truth as directed by the release",
            original=str(data.get("expected_answer", "")),
            replacement=answer,
        ),
    )
    return restored, changes
