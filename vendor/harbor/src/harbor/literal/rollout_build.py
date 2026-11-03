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
