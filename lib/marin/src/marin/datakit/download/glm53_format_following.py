# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 format-following answers paired with their original WildChat prompts."""

import hashlib
import re

from marin.datakit.download.referenced_completion import ReferencedCompletion, chat_normalize_steps, completion_document
from marin.execution.step_spec import StepSpec

WILDCHAT_NAME = "wildchat-glm53-format-completions"
WILDCHAT_REPO = "open-athena/" + WILDCHAT_NAME
WILDCHAT_REVISION = "c20a530940c3b23c832ac89e7b3dc6f6d38b76a9"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def wildchat_document(row: dict, source: dict) -> dict:
    """Recover the exact single user turn and verify its pinned WildChat identity."""
    index = row["source_turn_index"]
    if index < 0:
        raise ValueError("Negative WildChat turn index")
    message = source["conversation"][index]
    prompt = message["content"].strip()
    if (
        source["conversation_hash"] != row["conversation_hash"]
        or message["role"] != "user"
        or str(message.get("turn_identifier")) != row["turn_identifier"]
        or _sha256(prompt) != row["prompt_sha256"]
        or _sha256(re.sub(r"\s+", " ", prompt).strip().casefold()) != row["source_id"]
    ):
        raise ValueError("WildChat source reference mismatch")
    return completion_document(prompt + "\n\n" + row["format_instruction"], row["answer"], WILDCHAT_REPO, row["pair_id"])


WILDCHAT = ReferencedCompletion(
    WILDCHAT_NAME,
    WILDCHAT_REVISION,
    WILDCHAT_REPO,
    "allenai/WildChat-4.8M",
    "c827c6df8fcf008219ffaffa4d1dd77491099367",
    ("conversation_hash", "conversation"),
    wildchat_document,
    splits=("train", "validation"),
)


def glm53_format_following_chat_normalize_steps() -> tuple[StepSpec, ...]:
    return chat_normalize_steps(WILDCHAT, "train", prompt_hash_attrs={})


def glm53_format_following_validation_chat_normalize_steps() -> tuple[StepSpec, ...]:
    return chat_normalize_steps(WILDCHAT, "validation", prompt_hash_attrs={})
