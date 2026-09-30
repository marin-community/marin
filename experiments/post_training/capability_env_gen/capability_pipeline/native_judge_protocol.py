"""Bounded protocol repairs for the pinned TaskCompendium judge client."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
import time
from typing import Any

_INLINE_TERMINAL_SCORE = re.compile(
    r"(?P<prefix>\S)[ \t]+(?P<score>SCORE: (?:0|0\.5|1))[ \t]*\Z"
)
_NATIVE_TERMINAL_SCORE = re.compile(r"(?:^|\n)SCORE: (0|0\.5|1)\s*\Z")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class TerminalScoreLineClient:
    """Repair only a missing line break before one exact terminal score token.

    The pinned parser already defines the semantic vocabulary.  This wrapper does
    not infer a score from prose or alter a score value; it only supplies the line
    separator required by that parser and records every repair.
    """

    def __init__(self, client: Any):
        self.client = client
        self.repairs: list[dict[str, str]] = []

    def complete(self, prompt: str, policy: Any, timeout: float) -> Any:
        reply = self.client.complete(prompt, policy, timeout)
        text = reply.text
        match = _INLINE_TERMINAL_SCORE.search(text)
        if match is None or "SCORE:" in text[: match.start("score")]:
            return reply
        repaired = text[: match.start("prefix") + 1] + "\n" + match.group("score")
        self.repairs.append(
            {
                "rule": "missing_newline_before_exact_terminal_score",
                "original_text": text,
                "original_sha256": _digest(text),
                "repaired_sha256": _digest(repaired),
            }
        )
        return dataclasses.replace(reply, text=repaired)


class RetryingTerminalScoreClient:
    """Retry only syntactically malformed native judge replies.

    Each native ``complete`` call has one total timeout budget.  A malformed or
    missing terminal score may be re-asked with the identical prompt and policy;
    score values never influence retry.  Request/transport failures are retained
    and propagated immediately so the native grader keeps them ungraded.
    """

    def __init__(
        self,
        client: Any,
        *,
        max_reasks: int = 3,
        clock=time.monotonic,
    ):
        if type(max_reasks) is not int or max_reasks < 0:
            raise ValueError("max_reasks must be a nonnegative integer")
        self.client = client
        self.max_reasks = max_reasks
        self.clock = clock
        self.calls: list[dict[str, Any]] = []
        self.repairs: list[dict[str, Any]] = []
        self.request_count = 0

    def complete(self, prompt: str, policy: Any, timeout: float) -> Any:
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("judge timeout must be a positive finite number")
        call_index = len(self.calls) + 1
        call = {
            "call_index": call_index,
            "prompt_sha256": _digest(prompt),
            "timeout_seconds": float(timeout),
            "attempts": [],
        }
        self.calls.append(call)
        deadline = self.clock() + float(timeout)
        for request_index in range(1, self.max_reasks + 2):
            remaining = deadline - self.clock()
            if remaining <= 0:
                call["outcome"] = "timeout_budget_exhausted"
                raise TimeoutError("judge protocol retry timeout exhausted")
            self.request_count += 1
            terminal = TerminalScoreLineClient(self.client)
            try:
                reply = terminal.complete(prompt, policy, remaining)
                text = reply.text
                if not isinstance(text, str):
                    raise TypeError("judge reply text is not a string")
            except Exception as error:
                call["attempts"].append(
                    {
                        "request_index": request_index,
                        "outcome": "request_or_reply_error",
                        "error_type": type(error).__name__,
                    }
                )
                call["outcome"] = "request_or_reply_error"
                raise
            repair = terminal.repairs[0] if terminal.repairs else None
            raw_text = repair["original_text"] if repair else text
            attempt = {
                "request_index": request_index,
                "raw_text": raw_text,
                "raw_sha256": _digest(raw_text),
                "model": getattr(reply, "model", None),
                "revision": getattr(reply, "revision", None),
            }
            if repair:
                retained_repair = {
                    **repair,
                    "call_index": call_index,
                    "request_index": request_index,
                }
                self.repairs.append(retained_repair)
                attempt.update(
                    outcome="inline_terminal_score_repaired",
                    repaired_sha256=repair["repaired_sha256"],
                )
            elif _NATIVE_TERMINAL_SCORE.search(text) is not None:
                attempt["outcome"] = "valid_terminal_score"
            else:
                attempt["outcome"] = "malformed_terminal_score"
            call["attempts"].append(attempt)
            if attempt["outcome"] != "malformed_terminal_score":
                call["outcome"] = attempt["outcome"]
                call["selected_request_index"] = request_index
                return reply
            if request_index > self.max_reasks:
                call["outcome"] = "malformed_exhausted"
                call["selected_request_index"] = request_index
                return reply
        raise AssertionError("unreachable judge protocol retry state")

    def evidence(self) -> dict[str, Any]:
        """Return credential-free raw protocol attempts for verifier evidence."""
        return {
            "schema_version": "capability-native-judge-protocol-v2",
            "max_reasks": self.max_reasks,
            "actual_request_count": self.request_count,
            "calls": self.calls,
            "repairs": self.repairs,
        }
