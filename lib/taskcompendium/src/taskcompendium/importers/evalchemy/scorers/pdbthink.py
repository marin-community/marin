# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned PDBThink parsing, provider failure policy, and per-render report evidence."""

import json

from pdbthink.evaluation.response_protocol import MARIN_THINKING, answer_content, response_tool_events
from pdbthink.scoring import score_response
from verifyit.grade import Reward, scored


def grade_pdbthink(config: dict, response: str) -> Reward:
    envelope = json.loads(response)
    message = envelope["message"]
    finish_reason = envelope["finish_reason"]
    text = answer_content(message, MARIN_THINKING)
    tool_call = finish_reason in ("tool_calls", "function_call") or bool(
        response_tool_events({"choices": [{"message": message}]}, MARIN_THINKING)
    )
    outcome = score_response(
        text,
        config["answer_schema"],
        config["gold_answer"],
        parameters=config["parameters"],
        truncated=finish_reason == "length",
        provider_refusal=bool(message.get("refusal")) or finish_reason == "content_filter",
    )
    score = 0.0 if tool_call else float(outcome["score"]["score"])
    correct = bool(outcome["score"].get("correct")) and not tool_call
    metrics = {"macro_accuracy": float(correct), "micro_accuracy": float(correct), "score": score}
    return scored(
        float(correct),
        metrics=metrics,
        metric_weights={"macro_accuracy": config["family_weight"]},
        pdbthink_row={
            **config["report_metadata"],
            "score": score,
            "correct": correct,
            "detail": outcome["score"],
            "parsed_answer": outcome["parsed"],
            "format_error": outcome["format_error"],
            "refusal": outcome["refusal"],
            "truncated": outcome["truncated"],
            "api_error": False,
            "tool_call": tool_call,
        },
    )
