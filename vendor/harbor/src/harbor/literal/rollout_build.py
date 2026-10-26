"""Build ``RolloutDetail`` objects from RecordProxy literal captures.

Shared by the in-trial opencode path (``agents/installed/opencode.py``) and the
RL-loop per-trial correlation glue (MarinSkyRL). CLI agents such as opencode talk
to vLLM over their own transport, bypassing harbor ``Chat``, so their
``context.rollout_details`` is never populated natively. A co-located
:class:`~harbor.literal.proxy.RecordProxy` captures the per-turn token IDs /
logprobs; these helpers turn those captures into the same ``RolloutDetail`` shape
terminus-2 produces via ``Chat.rollout_details``, so the MarinSkyRL extract
functions (``extract_{logprobs,token_ids,prompt_token_ids}_from_rollout_details``)
consume opencode rollouts unchanged.

Two entry points:
  * :func:`build_rollout_details_from_pairs` — from an ordered list of per-turn
    literal dicts (``{prompt_token_ids, completion_token_ids, logprobs}``), the
    in-trial path (one trial's own ``literal.jsonl``).
  * :func:`build_rollout_details_for_trial` — from the shared controller proxy
    log's JSONL entries, selecting a single trial's turns by the per-trial
    correlation id stamped by the proxy (:data:`~harbor.literal.proxy.CORRELATION_ID_HEADER`).
    This is the concurrency-safe path: many trials interleave in one shared log,
    and the id (not content) attributes each turn to its trial — immune to the
    GRPO identical-seed ambiguity that defeats content correlation.

The per-turn length-parity guard mirrors ``Chat._accumulate_rollout_details``
(commit ``8737426c``): a turn whose logprobs length != its ``completion_token_ids``
length (or whose logprobs are absent) records an EMPTY logprob list for that turn
so it stays index-aligned with ``completion_token_ids`` — downstream TIS flags the
turn rather than mis-pairing logprobs to tokens. Never synthesize/re-tokenize.
"""

from __future__ import annotations

import logging
import json
import math
from pathlib import Path
from typing import Any

from upath import UPath

from harbor.models.agent.context import AgentContext
from harbor.models.agent.rollout_detail import RolloutDetail

logger = logging.getLogger(__name__)


def populate_literal_context(
    context: AgentContext, log_path: Path | UPath, correlation_id: str | None
) -> None:
    """Attach captured token evidence and trial identity to the agent result."""
    if correlation_id:
        context.metadata = {
            **(context.metadata or {}),
            "rollout_correlation_id": correlation_id,
        }
    details = build_rollout_details_from_pairs(read_literal_pairs(log_path))
    if details:
        context.rollout_details = details


def read_literal_pairs(log_path: Path | UPath) -> list[dict[str, Any]]:
    """Read ordered literal pairs from a RecordProxy JSONL log."""
    if not log_path.exists():
        return []

    pairs: list[dict[str, Any]] = []
    malformed_records = 0
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            malformed_records += 1
            continue
        literal = entry.get("literal")
        if isinstance(literal, dict):
            pairs.append(literal)

    if malformed_records:
        logger.warning(
            "Skipped %d malformed RecordProxy literal record(s) in %s",
            malformed_records,
            log_path,
        )
    return pairs


def build_rollout_details_from_pairs(
    literal_pairs: list[dict[str, Any]],
) -> list[RolloutDetail]:
    """Build a single :class:`RolloutDetail` from ordered per-turn literal pairs.

    ``literal_pairs`` is one trial's per-turn dicts, in call order, each with any
    of ``prompt_token_ids`` / ``completion_token_ids`` / ``logprobs``. Mirrors
    ``harbor.llms.chat.Chat.rollout_details``: one ``RolloutDetail`` carrying
    per-turn lists, with the ``8737426c`` per-turn parity guard on logprobs.

    Returns a list with a single ``RolloutDetail`` (a linear main session), or an
    empty list if no usable token data was captured.
    """
    prompt_token_ids_list: list[list[int]] = []
    completion_token_ids_list: list[list[int]] = []
    logprobs_list: list[list[float]] = []

    for pair in literal_pairs:
        prompt_ids = pair.get("prompt_token_ids")
        completion_ids = pair.get("completion_token_ids")
        logprobs = pair.get("logprobs")

        if prompt_ids:
            prompt_token_ids_list.append(prompt_ids)

        if completion_ids:
            completion_token_ids_list.append(completion_ids)
            # Keep logprobs index-aligned with completion_token_ids per turn.
            n_ids = len(completion_ids)
            n_lp = len(logprobs) if logprobs else 0
            if logprobs and n_lp == n_ids:
                logprobs_list.append(logprobs)
            else:
                if logprobs:
                    logger.warning(
                        "literal logprob/token-id length mismatch for this turn "
                        "(%d logprobs vs %d completion_token_ids); recording an "
                        "empty logprob list so downstream TIS flags it instead of "
                        "mis-pairing logprobs to tokens.",
                        n_lp,
                        n_ids,
                    )
                logprobs_list.append([])

    if (
        not prompt_token_ids_list
        and not completion_token_ids_list
        and not logprobs_list
    ):
        return []

    rollout_detail: RolloutDetail = {}
    if prompt_token_ids_list:
        rollout_detail["prompt_token_ids"] = prompt_token_ids_list
    if completion_token_ids_list:
        rollout_detail["completion_token_ids"] = completion_token_ids_list
    if logprobs_list:
        rollout_detail["logprobs"] = logprobs_list
    return [rollout_detail]


def _normalize_pi_message(message: dict[str, Any]) -> dict[str, str]:
    """Flatten the served OpenAI message without importing traces_utils' agent factory."""
    content = message.get("content")
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    elif (
        isinstance(content, list)
        and content
        and isinstance(content[0], dict)
        and "text" in content[0]
    ):
        value = content[0]["text"]
        text = (
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        )
    else:
        text = json.dumps(content, ensure_ascii=False)
    if not text and isinstance(message.get("tool_calls"), list):
        text = "\n".join(
            json.dumps(call.get("function", call), ensure_ascii=False, sort_keys=True)
            for call in message["tool_calls"]
            if isinstance(call, dict)
        )
    return {"role": message["role"], "content": text}


def _unique_pi_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate Pi tool argument: {key}")
        value[key] = item
    return value


def _same_pi_tool_call(original: dict[str, Any], repeated: dict[str, Any]) -> bool:
    original_fn, repeated_fn = original.get("function"), repeated.get("function")
    if not isinstance(original_fn, dict) or not isinstance(repeated_fn, dict):
        return False
    if original.get("id") != repeated.get("id") or original.get("type") != repeated.get(
        "type"
    ):
        return False
    if original_fn.get("name") != repeated_fn.get("name"):
        return False
    try:
        original_args = json.loads(
            original_fn["arguments"], object_pairs_hook=_unique_pi_json_object
        )
        repeated_args = json.loads(
            repeated_fn["arguments"], object_pairs_hook=_unique_pi_json_object
        )
    except (KeyError, TypeError, ValueError):
        return False
    return isinstance(original_args, dict) and original_args == repeated_args


def build_pi_trial_evidence(
    entries: list[dict[str, Any]],
) -> tuple[list[RolloutDetail], list[dict[str, str]]]:
    """Hydrate one Pi trial from its private recorder, rejecting incomplete model calls.

    Use served messages rather than Pi's rendered text. Student logprobs and
    IDs belong to the same recorded HTTP exchange; teacher-authored actions
    need exact IDs but may lack unused teacher probabilities. Consecutive
    requests must replay the preceding assistant action before continuing.
    """
    if not entries:
        raise ValueError("Pi trial has no literal model records")
    literal_pairs: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        literal = entry.get("literal")
        request = entry.get("request")
        if (
            entry.get("status_code") != 200
            or not isinstance(literal, dict)
            or not isinstance(request, dict)
        ):
            raise ValueError(
                f"Pi model call {index} has no successful request/response"
            )
        ids = literal.get("completion_token_ids")
        probs = literal.get("logprobs")
        prompt_ids = literal.get("prompt_token_ids")
        assistant = literal.get("assistant_message")
        if (
            not isinstance(prompt_ids, list)
            or any(type(value) is not int or value < 0 for value in prompt_ids)
            or not isinstance(ids, list)
            or not ids
            or any(type(value) is not int or value < 0 for value in ids)
            or not (
                (entry.get("actor") == "teacher" and probs is None)
                or (
                    isinstance(probs, list)
                    and len(ids) == len(probs)
                    and all(
                        type(value) in (int, float) and math.isfinite(value)
                        for value in probs
                    )
                )
            )
            or not isinstance(assistant, dict)
            or assistant.get("role") != "assistant"
        ):
            raise ValueError(f"Pi model call {index} has incomplete literal evidence")
        literal_pairs.append(literal)
        if index + 1 < len(entries):
            following = entries[index + 1].get("request")
            previous_messages = request.get("messages")
            next_messages = (
                following.get("messages") if isinstance(following, dict) else None
            )
            if (
                not isinstance(previous_messages, list)
                or not isinstance(next_messages, list)
                or len(next_messages) <= len(previous_messages)
                or next_messages[: len(previous_messages)] != previous_messages
            ):
                raise ValueError(
                    f"Pi model call {index + 1} is not a continuation of call {index}"
                )
            replay = next_messages[len(previous_messages)]
            if not isinstance(replay, dict) or replay.get("role") != "assistant":
                raise ValueError(
                    f"Pi model call {index + 1} is missing the preceding assistant action"
                )
            if replay.get("content") != assistant.get("content"):
                raise ValueError(
                    f"Pi model call {index + 1} changed the preceding assistant content"
                )
            served_calls, replay_calls = (
                assistant.get("tool_calls"),
                replay.get("tool_calls"),
            )
            if (served_calls is None) != (replay_calls is None):
                raise ValueError(
                    f"Pi model call {index + 1} changed the preceding tool calls"
                )
            if served_calls is not None:
                if (
                    not isinstance(served_calls, list)
                    or not isinstance(replay_calls, list)
                    or len(served_calls) != len(replay_calls)
                ):
                    raise ValueError(
                        f"Pi model call {index + 1} changed the preceding tool calls"
                    )
                for original, repeated in zip(served_calls, replay_calls, strict=True):
                    if not isinstance(original, dict) or not isinstance(repeated, dict):
                        raise ValueError("Pi tool calls must be objects")
                    if not _same_pi_tool_call(original, repeated):
                        raise ValueError(
                            f"Pi model call {index + 1} changed the preceding tool arguments"
                        )

    final_request = entries[-1]["request"]
    final_messages = final_request.get("messages")
    if not isinstance(final_messages, list) or not final_messages:
        raise ValueError("Pi trial has no final served conversation")
    if any(
        not isinstance(message, dict) or not isinstance(message.get("role"), str)
        for message in final_messages
    ):
        raise ValueError("Pi trial contains malformed served messages")
    messages = [_normalize_pi_message(message) for message in final_messages]
    messages.append(_normalize_pi_message(literal_pairs[-1]["assistant_message"]))
    if sum(message["role"] == "assistant" for message in messages) != len(entries):
        raise ValueError(
            "Pi trial's assistant turns do not match its literal model calls"
        )
    return build_rollout_details_from_pairs(literal_pairs), messages


def build_rollout_details_for_trial(
    entries: list[dict[str, Any]],
    trial_id: str,
) -> list[RolloutDetail]:
    """Select one trial's turns from a shared proxy log and build its RolloutDetail.

    ``entries`` are parsed JSONL records from the shared controller RecordProxy log
    (each ``{timestamp, status_code, trial_id, request, literal, ...}``). Records
    whose stamped ``trial_id`` matches are ordered by ``timestamp`` (call order,
    spanning any opencode auto-compaction boundary — the id is per-trial, stable
    across the trial's own compaction) and their ``literal`` dicts are built into
    a single ``RolloutDetail`` via :func:`build_rollout_details_from_pairs`.

    Returns ``[]`` when the trial has no matching, status-200, completion-bearing
    records (verify-or-skip: an unattributable trial contributes NO literal columns
    rather than borrowing another trial's — TIS then flags it honestly).
    """
    if not trial_id:
        return []
    matched = [
        e
        for e in entries
        if e.get("trial_id") == trial_id
        and e.get("status_code") == 200
        and isinstance(e.get("literal"), dict)
    ]
    matched.sort(key=lambda e: e.get("timestamp") or 0.0)
    pairs = [e["literal"] for e in matched if e["literal"].get("completion_token_ids")]
    return build_rollout_details_from_pairs(pairs)
