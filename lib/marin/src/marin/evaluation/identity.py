# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native-token identity views and reproducible mechanical pre-screening.

These labels are observations, not role-aware factual judgments. Preserve raw
responses and protocol flags for manual adjudication; do not grade reasoning as
if the model delivered it in the final answer.
"""

import copy
import re
from dataclasses import dataclass
from typing import Any

import requests


def boundary_view(ids: list[int], *, start_id: int, end_id: int) -> dict[str, Any]:
    """Split only at the first generated END, never a marker in the prompt."""
    starts = [i for i, token in enumerate(ids) if token == start_id]
    ends = [i for i, token in enumerate(ids) if token == end_id]
    first = ends[0] if ends else None
    closed = len(starts) == len(ends) == 1 and starts[0] < ends[0]
    return {
        "generated_start_positions": starts,
        "generated_end_positions": ends,
        "duplicate_generated_boundaries": len(starts) > 1 or len(ends) > 1,
        "end_without_preceding_generated_start": bool(ends and (not starts or ends[0] < starts[0])),
        "model_opened_and_closed": closed,
        "canonical_leading_start": bool(closed and starts[0] == 0),
        "no_generated_end": first is None,
        "first_generated_end": first,
        "reasoning_ids": None if first is None else ids[:first],
        "answer_ids": None if first is None else ids[first + 1 :],
        "pre_start_ids": ids[: starts[0]] if starts else None,
    }


@dataclass(frozen=True)
class IdentityRules:
    name_pattern: str
    creator_pattern: str
    foreign_self_claim_pattern: str | None = None


def prescreen(answer: str | None, rules: IdentityRules) -> dict[str, Any]:
    """Literal regex observations only; a missing native final is its own bucket."""
    if answer is None:
        return {
            "answer_available": False,
            "name_match": None,
            "creator_match": None,
            "foreign_self_claim_match": None,
            "empty": None,
            "needs_manual_review": True,
        }
    foreign = (
        re.search(rules.foreign_self_claim_pattern, answer, re.IGNORECASE) if rules.foreign_self_claim_pattern else None
    )
    return {
        "answer_available": True,
        "name_match": bool(re.search(rules.name_pattern, answer, re.IGNORECASE)),
        "creator_match": bool(re.search(rules.creator_pattern, answer, re.IGNORECASE)),
        "foreign_self_claim_match": foreign.group(0) if foreign else None,
        "empty": not answer,
        "needs_manual_review": True,
    }


def decode_views(tokenizer, ids: list[int], *, thinking: bool, start_id: int, end_id: int) -> dict[str, Any]:
    """Keep raw, native-final and whole-ID proxy views distinctly named."""
    boundary = boundary_view(ids, start_id=start_id, end_id=end_id)
    final_ids = boundary["answer_ids"] if thinking else ids
    decode_args = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    return {
        "boundary": boundary,
        "raw_generated_text": tokenizer.decode(ids, **decode_args),
        "native_final_text": None if final_ids is None else tokenizer.decode(final_ids, **decode_args),
        "whole_id_proxy_text": tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
        "native_final_available": final_ids is not None,
    }


class NativeCaptureError(ValueError):
    def __init__(self, message: str, evidence: dict[str, Any]):
        super().__init__(message)
        self.evidence = evidence


def _token_ids(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(type(i) is int and i >= 0 for i in value)


def capture_native_response(
    session: requests.Session,
    base_url: str,
    request: dict[str, Any],
    *,
    expected_prompt_ids: list[int],
    timeout: float = 900,
) -> dict[str, Any]:
    """One tokenize and one completion POST; preserve failures, never retry or repair.

    The completion request changes only the two native-capture flags. Prefix IDs
    must match the independent caller tokenization both before and after decoding.
    Provider content/reasoning and the complete raw response remain unchanged.
    """
    if not _token_ids(expected_prompt_ids):
        raise ValueError("Expected nonempty native prompt IDs")
    original = copy.deepcopy(request)
    body = {**copy.deepcopy(original), "skip_special_tokens": False, "return_token_ids": True}
    token_body = {
        name: copy.deepcopy(body[name])
        for name in ("model", "messages", "chat_template_kwargs", "tools")
        if name in body
    }
    token_body.update(
        add_generation_prompt=body.get("add_generation_prompt", True),
        continue_final_message=body.get("continue_final_message", False),
    )
    evidence = {
        "original_request": original,
        "request": body,
        "expected_prompt_ids": list(expected_prompt_ids),
        "tokenize_request": token_body,
    }

    def post(route: str, payload: dict, key: str):
        try:
            response = session.post(base_url.rstrip("/") + route, json=payload, timeout=timeout)
        except requests.RequestException as error:
            evidence[key] = {"error_type": type(error).__name__}
            raise NativeCaptureError("Native capture request failed", evidence) from None
        evidence[key] = {"status_code": response.status_code, "body": response.text}
        if not response.ok:
            raise NativeCaptureError("Native capture returned an HTTP error", evidence)
        try:
            result = response.json()
        except ValueError:
            raise NativeCaptureError("Native capture returned invalid JSON", evidence) from None
        if not isinstance(result, dict):
            raise NativeCaptureError("Native capture returned a non-object response", evidence)
        return result

    tokens = post("/tokenize", token_body, "tokenize_response")
    if not _token_ids(tokens.get("tokens")) or tokens["tokens"] != expected_prompt_ids:
        raise NativeCaptureError("Server/native prompt prefix mismatch; no completion submitted", evidence)
    result = post("/v1/chat/completions", body, "completion_response")
    evidence["response"] = copy.deepcopy(result)
    if not _token_ids(result.get("prompt_token_ids")) or result["prompt_token_ids"] != expected_prompt_ids:
        raise NativeCaptureError("Completion returned a different prompt prefix", evidence)
    choices = result.get("choices", [])
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
        or not _token_ids(choices[0].get("token_ids"))
    ):
        raise NativeCaptureError("Missing or invalid generated token IDs", evidence)
    evidence["generated_ids"] = list(choices[0]["token_ids"])
    return evidence
