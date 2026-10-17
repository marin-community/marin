# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the historical first-match selector used for E_2's RS demonstrations.

This is NOT the ae362881 evaluator: reasoning can win over the final answer.
Fourteen of the 1,400 selected demos are wrong under the evaluator's rule.
Keep the selection and serialized grade metadata unchanged to audit the trained arm.
"""

import re


def extracted_answer(text: str) -> str | None:
    match = re.search(r"answer is \(?([A-J])\)?", text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"Answer:\s*([A-J])", text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return match.group(0) if match else None


def grade_response(record: dict, gold: str) -> dict:
    """Retain failures and truncations; final-only extraction is diagnostic only."""
    base = {"custom_id": record["custom_id"], "prompt_id": record["prompt_id"], "gold": gold}
    if not 200 <= record["status_code"] < 300:
        return base | {"status": "request_failed", "correct": False, "pred": None}
    choice = record["response"]["choices"][0]
    message = choice["message"]
    if message.get("reasoning_content") or message.get("reasoning"):
        raise ValueError("Expected parser-free native completion; preserve separate reasoning explicitly")
    text = message.get("content") or ""
    prediction = extracted_answer(text)
    has_boundary = "<|end_think|>" in text
    final_text = text.rsplit("<|end_think|>", 1)[-1] if has_boundary else None
    final_prediction = extracted_answer(final_text) if final_text is not None else None
    return base | {
        "status": "ok",
        "pred": prediction,
        "correct": prediction == gold,
        "finish_reason": choice["finish_reason"],
        "native_end_think_count": text.count("<|end_think|>"),
        "content_only_available": has_boundary,
        "content_only_pred": final_prediction,
        "content_only_disagrees": prediction != final_prediction if has_boundary else None,
        "grading_text_policy": "raw native content; inferred equivalent to evaluator COMBINE, not byte-verified",
        "content_only_policy": "diagnostic: text after last native end_think; unavailable without boundary",
    }
