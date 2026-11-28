# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Answer extraction and reward functions from NVIDIA NeMo Gym litmus_agent
at e87f88065db6105548194e1f12feea720c129cca."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

FLOAT = "float"
BOOL = "bool"
STRING = "string"
_SUPPORTED_ANSWER_TYPES = {FLOAT, BOOL, STRING}

_TRUE_TOKENS = {"1", "1.0", "true", "yes", "present", "t", "y"}
_FALSE_TOKENS = {"0", "0.0", "false", "no", "absent", "f", "n"}

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

# Answer-format wrapper regexes locate the answer in arbitrary wrapper text.
_ANSWER_FORMAT_REGEXES: dict[str, re.Pattern[str]] = {
    "fmt_00": re.compile(r"\(\((.*?)\)\)", re.S),
    "fmt_01": re.compile(r"\(Answer:\s*(.+?)\)", re.S),
    "fmt_02": re.compile(r"Final answer:\s*\((.+?)\)", re.S),
    "fmt_03": re.compile(r"Answer is\s*\[(.+?)\]", re.S),
    "fmt_04": re.compile(r"\[Answer:\s*(.+?)\]", re.S),
    "fmt_05": re.compile(r"\[\[(.+?)\]\]", re.S),
    "fmt_06": re.compile(r"Correct Answer:\s*\[(.+?)\]", re.S),
    "fmt_07": re.compile(r"\\boxed\{(.+?)\}", re.S),
    "fmt_08": re.compile(r"\\boxed\{(.+?)\}", re.S),
    "fmt_09": re.compile(r"\{\{(.+?)\}\}", re.S),
    "fmt_10": re.compile(r"Answer Value:\s*\{(.+?)\}", re.S),
    "fmt_11": re.compile(r"<<(.+?)>>", re.S),
    "fmt_12": re.compile(r"<<(.+?)>>", re.S),
    "fmt_13": re.compile(r"<(.+?)>", re.S),
    "fmt_14": re.compile(r"<Answer:\s*(.+?)>", re.S),
    "fmt_15": re.compile(r"<final_answer>\s*(.+?)\s*</final_answer>", re.S),
    "fmt_16": re.compile(r"Final Answer:\s*\|\|(.+?)\|\|", re.S),
    "fmt_17": re.compile(r"The answer is:\s*\|(.+?)\|", re.S),
    "fmt_18": re.compile(r"\*\*Answer:\s*(.+?)\*\*", re.S),
    "fmt_19": re.compile(r"\*\*Final answer is:\s*(.+?)\*\*", re.S),
    "fmt_20": re.compile(r"Answer:\s*\*(.+?)\*", re.S),
    "fmt_21": re.compile(r"## Answer:\s*(.+?)\s*##", re.S),
    "fmt_22": re.compile(r"ANSWER IS\s*(.+)"),
    "fmt_23": re.compile(r"Response:\s*(.+)"),
    "fmt_24": re.compile(r"Final Answer\s*->\s*(.+)"),
    "fmt_25": re.compile(r"Final value is:\s*(.+)"),
    "fmt_26": re.compile(r"Correct Answer >>\s*(.+)"),
    "fmt_27": re.compile(r"Answer Value:\s*(.+)"),
    "fmt_28": re.compile(r"Final Answer\s*=\s*(.+)"),
    "fmt_29": re.compile(r"Correct answer is\s*(.+)"),
    "fmt_30": re.compile(r"Final Answer:\s*(.+)"),
}


def resolve_answer_type(answer_type: str) -> str:
    """Validate the row's explicit answer type before scoring."""
    if answer_type not in _SUPPORTED_ANSWER_TYPES:
        raise ValueError(f"Unsupported answer_type={answer_type!r}")
    return answer_type


def _capture_with_pattern(text: str, pattern: re.Pattern[str]) -> str | None:
    """Return the last match of a compiled answer regex, or None.

    Only the last match is considered, so a self-correcting response ("...3,
    actually 5") scores on its final answer. Multi-group patterns collapse to
    their first non-empty group.
    """
    matches = pattern.findall(text)
    if not matches:
        return None
    match = matches[-1]
    if isinstance(match, tuple):
        match = next((group for group in match if group), "")
    return match.strip()


def _raw_capture(text: str, answer_format: str) -> str | None:
    """Return the last match of the named ``fmt_XX`` regex, or None."""
    pattern = _ANSWER_FORMAT_REGEXES.get(answer_format)
    if pattern is None:
        raise ValueError(f"Unsupported answer_format={answer_format!r}")
    return _capture_with_pattern(text, pattern)


def _compile_output_regex(output_regex: str) -> re.Pattern[str]:
    """Compile a per-row ``output_regex``, requiring exactly one capture group.

    Fails loudly on an invalid pattern or a group count other than one so bad
    dataset regexes surface at scoring time instead of silently mis-extracting.
    """
    try:
        pattern = re.compile(output_regex, re.S)
    except re.error as exc:
        raise ValueError(f"Invalid output_regex={output_regex!r}: {exc}") from exc
    if pattern.groups != 1:
        raise ValueError(f"output_regex={output_regex!r} must have exactly one capture group, got {pattern.groups}")
    return pattern


def _parse_bool(inner: str) -> float | None:
    """Coerce a capture into 1.0/0.0, accepting word tokens or 0/1 numerics."""
    token = inner.strip().lower()
    if token in _TRUE_TOKENS:
        return 1.0
    if token in _FALSE_TOKENS:
        return 0.0
    num = _parse_numeric(token)
    if num is None:
        return None
    return 1.0 if num != 0.0 else 0.0


def _parse_numeric(inner: str) -> float | None:
    """Parse a number from a capture; fall back to the last number-like token."""
    inner = inner.strip()
    try:
        return float(inner)
    except (ValueError, TypeError):
        pass
    nums = _NUMBER_RE.findall(inner)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass
    return None


def extract_predicted_value(
    response: str,
    answer_type: str,
    *,
    output_regex: str | None = None,
    answer_format: str | None = None,
) -> float | str | None:
    """Extract the model's predicted value from its response text.

    Locates the answer, then coerces it by ``answer_type``: numeric types parse
    to float, ``bool`` to 1.0/0.0, ``string`` returns the raw captured text.

    The answer is located by the first available of, in order:

    1. ``output_regex`` (preferred): a regex carried directly on the row. Must
       compile and have exactly one capture group, else ``ValueError``.
    2. ``answer_format``: look up a regex by ``fmt_XX`` name in the registry
       kept for rows exported without an ``output_regex``. Unknown names raise
       ``ValueError`` so bad data fails loudly.
    3. Double parentheses (``fmt_00``) when neither extractor is specified.

    Returns None when nothing can be extracted.
    """
    if not isinstance(response, str):
        return None
    text = response.strip()

    if output_regex is not None:
        raw = _capture_with_pattern(text, _compile_output_regex(output_regex))
    else:
        fmt = answer_format or "fmt_00"
        raw = _raw_capture(text, fmt)
    if raw is None:
        return None
    if answer_type == STRING:
        return raw
    if answer_type == BOOL:
        return _parse_bool(raw)
    return _parse_numeric(raw)


# ---------------------------------------------------------------------------
# Reward rules
# ---------------------------------------------------------------------------
#
# A reward rule scores a predicted value against the expected one and returns a
# reward in [0.0, 1.0]. Rules are looked up by name in REWARD_RULES; the rule is
# chosen per-row via ``match`` or defaults to _DEFAULT_RULE[answer_type]. Each
# rule accepts **params (so unknown kwargs from a row are tolerated) and the
# server's float tolerances are always supplied as rel_tol/abs_tol defaults.


def _rule_exact(predicted: float, expected: Any, **_: Any) -> float:
    return 1.0 if round(float(predicted)) == round(float(expected)) else 0.0


def _rule_isclose(predicted: float, expected: Any, *, rel_tol: float = 1e-6, abs_tol: float = 1e-6, **_: Any) -> float:
    return 1.0 if math.isclose(float(predicted), float(expected), rel_tol=rel_tol, abs_tol=abs_tol) else 0.0


def _rule_abs_window(predicted: float, expected: Any, *, abs_tol: float = 0.0, **_: Any) -> float:
    return 1.0 if abs(float(predicted) - float(expected)) <= abs_tol else 0.0


def _rule_rel_window(predicted: float, expected: Any, *, rel_tol: float = 0.0, **_: Any) -> float:
    return 1.0 if abs(float(predicted) - float(expected)) <= abs(rel_tol * float(expected)) else 0.0


def _rule_bool_eq(predicted: float, expected: Any, **_: Any) -> float:
    expected_bool = _parse_bool(str(expected))
    return 1.0 if expected_bool is not None and float(predicted) == expected_bool else 0.0


def _rule_string_eq(predicted: Any, expected: Any, **_: Any) -> float:
    return 1.0 if _normalize_str(predicted) == _normalize_str(expected) else 0.0


# Registry of named reward rules. Register a custom rule by adding an entry here.
REWARD_RULES: dict[str, Callable[..., float]] = {
    "exact": _rule_exact,
    "isclose": _rule_isclose,
    "abs_window": _rule_abs_window,
    "rel_window": _rule_rel_window,
    "bool_eq": _rule_bool_eq,
    "string_eq": _rule_string_eq,
}

# Default reward rule per answer_type, used when a row supplies no ``match``.
# ``float`` defaults to numeric equality; rows wanting rounded-integer matching
# opt in with match={"rule": "exact"}.
_DEFAULT_RULE: dict[str, str] = {
    FLOAT: "isclose",
    BOOL: "bool_eq",
    STRING: "string_eq",
}


def _normalize_str(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())



def resolve_reward_rule(answer_type: str, match: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """Resolve (rule_name, params) from a per-row ``match`` or the type default.

    A ``match`` overrides the default and must carry a ``rule`` key; its other
    keys are passed to the rule as params. Raises ValueError on a malformed
    ``match`` so misconfigured rows fail loudly rather than scoring 0.0.
    """
    if match:
        rule_name = match.get("rule")
        if not rule_name:
            raise ValueError(f"match={match!r} must include a 'rule' key")
        params = {k: v for k, v in match.items() if k != "rule"}
        return rule_name, params
    return _DEFAULT_RULE[answer_type], {}


def compute_reward(
    predicted: float | str | None,
    expected: str | float | int,
    answer_type: str,
    *,
    match: dict[str, Any] | None = None,
    float_rel_tol: float = 1e-6,
    float_abs_tol: float = 1e-6,
) -> float:
    """Score a predicted value against the expected one with the resolved rule.

    The comparison rule is resolved independently of ``answer_type``: a per-row
    ``match`` wins, otherwise the answer_type's default rule applies. Returns 0.0
    for an unextractable (None) or NaN prediction.
    """
    if predicted is None:
        return 0.0
    if isinstance(predicted, float) and math.isnan(predicted):
        return 0.0

    rule_name, params = resolve_reward_rule(answer_type, match)
    rule = REWARD_RULES.get(rule_name)
    if rule is None:
        raise ValueError(f"Unsupported reward rule={rule_name!r}; choose from {sorted(REWARD_RULES)}")
    call_params = {"rel_tol": float_rel_tol, "abs_tol": float_abs_tol, **params}
    return float(rule(predicted, expected, **call_params))
