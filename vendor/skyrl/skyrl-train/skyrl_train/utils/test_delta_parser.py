"""In-trajectory test-result parser (Stage C / F2) for loop-behavior reward shaping.

The agent runs the test suite itself *inside* the rollout loop (the hero reads
``"1 failed, 140 passed"`` from a real pytest invocation). Stage C's
potential-based shaping (PBS, F6) needs the per-test-run ``(n_pass, n_fail)``
state so it can credit the *edit* that moved the suite toward green.

This module extracts that signal — and ONLY that signal — from the **real
test-runner stdout** that comes back in the tool observations of the chat
history. Hard safety rules (mirroring the reward-hacking analysis in the plan):

  * **Parse real command output ONLY.** We read *observation / tool* messages
    (the sandbox's stdout), NEVER assistant messages (the model's prose, e.g.
    "all tests pass"). Crediting model claims would be trivially hackable.
  * **Recognized frameworks only.** pytest / unittest / jest summaries are
    regular and reliable; we try those parsers in priority order. We do NOT use
    the generic keyword-counting parser here — counting "PASS"/"FAIL" substrings
    in arbitrary observation text would fabricate deltas from non-test output.
  * **Graceful no-signal fallback.** A trajectory with no recognized test-runner
    output yields an EMPTY run list → the PBS layer produces an all-zero shaping
    vector for it → that trajectory stays pure-RLOO-N (outcome-reward-only). We
    never fabricate a delta.

Pure / CPU-only; no torch dependency. Unit-testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

from skyrl_train.utils.reward_shaping import get_output_parser

# Frameworks tried, in priority order. unittest FIRST: its "Ran N tests in Ys"
# banner is unambiguous and pytest never emits it, whereas pytest's lenient
# line-count fallback would mis-claim a unittest "FAILED (...)" line. jest's
# "Tests:" line is likewise distinct. Generic keyword-counting is intentionally
# EXCLUDED — see the module docstring.
_FRAMEWORKS: List[str] = ["unittest", "pytest", "jest"]

# Roles whose message content is REAL command output (the sandbox stdout fed
# back to the agent). Assistant messages are excluded by construction.
_OBSERVATION_ROLES = frozenset({"tool", "user", "ipython", "observation", "function"})


@dataclass
class TestRunResult:
    """One recognized test-run observation in the trajectory.

    Attributes:
        message_index: index of the observation message in ``chat_history`` that
            produced this result (so the PBS layer can locate the *preceding*
            edit turn).
        passed: number of tests that passed (effective: passed + xfailed).
        failed: number of tests that did not behave as expected
            (failed + errors + xpassed).
        total_runnable: tests that actually ran (excludes skipped); 0 ⇒ no-signal.
        framework: which parser recognized it ("pytest"/"unittest"/"jest").
    """

    # Not a pytest test class (the "Test" prefix would otherwise trip collection).
    __test__ = False

    message_index: int
    passed: int
    failed: int
    total_runnable: int
    framework: str

    @property
    def frac_passing(self) -> float:
        """Fraction of runnable tests passing in [0, 1]; 0.0 if nothing ran."""
        if self.total_runnable <= 0:
            return 0.0
        return max(0.0, min(1.0, self.passed / self.total_runnable))


def _message_text(msg: Dict[str, Any]) -> str:
    """Best-effort extraction of an observation message's text content.

    Tool/observation content can be a plain string or a list of content parts
    (OpenAI-style ``[{"type": "text", "text": ...}]``). We concatenate the text.
    """
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                t = part.get("text") or part.get("content")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return str(content)


def _parse_one_observation(text: str) -> Optional[TestRunResult]:
    """Try each framework parser on one observation's stdout.

    Returns a TestRunResult (message_index filled by the caller) for the FIRST
    framework that both ``can_parse`` and yields a result with runnable tests,
    or None (no recognized test output / nothing ran).
    """
    if not text:
        return None
    for fw in _FRAMEWORKS:
        parser = get_output_parser(fw)
        # can_parse is a cheap framework-presence gate; it stops e.g. the
        # unittest "Ran N tests" pattern from firing on incidental jest text.
        if not parser.can_parse(text):
            continue
        parsed = parser.parse(text)
        if parsed is None:
            continue
        runnable = parsed.runnable_total
        if runnable <= 0:
            # Recognized framework but nothing actually ran (e.g. collection
            # error already returns None; an all-skipped run is no-signal).
            continue
        return TestRunResult(
            message_index=-1,  # filled by caller
            passed=parsed.effective_passed,
            failed=parsed.effective_failed,
            total_runnable=runnable,
            framework=fw,
        )
    return None


def extract_test_runs(chat_history: Optional[List[Dict[str, Any]]]) -> List[TestRunResult]:
    """Extract the ordered sequence of recognized test-run observations.

    Walks ``chat_history`` in order; for each OBSERVATION message (role tool /
    user / etc. — never assistant), tries the framework parsers on its stdout.
    Returns the list of recognized runs in trajectory order (each tagged with
    its ``message_index``). Empty list ⇒ no-signal (PBS stays outcome-only).

    Robust to malformed input (None / non-list / non-dict messages) — those are
    skipped, never raised on (a single bad trajectory must not crash the batch).
    """
    if not chat_history or not isinstance(chat_history, list):
        return []
    runs: List[TestRunResult] = []
    for idx, msg in enumerate(chat_history):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            # NEVER parse the model's own prose as a test result.
            continue
        if role not in _OBSERVATION_ROLES:
            continue
        try:
            result = _parse_one_observation(_message_text(msg))
        except Exception as e:  # pragma: no cover - defensive; never crash a batch
            logger.warning("test_delta_parser: parse failed at msg {} ({}); skipping", idx, e)
            result = None
        if result is not None:
            result.message_index = idx
            runs.append(result)
    return runs
