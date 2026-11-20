# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ZeroEval multiple-choice answer extraction.

Adapted from WildEval/ZeroEval (Apache-2.0) at 8c1485edf12c6efb5f69135a562927c5ad484059:
``extract_first_complete_json`` and ``extract_values_from_json`` in ``src/evaluation/eval_utils.py``
and the matching rule in ``src/evaluation/mcqa_eval.py``. The response's first complete JSON object
supplies the ``answer`` field; a regex over ``"answer": ...`` is the fallback. An answer is correct when
it equals the gold letter or contains the letter followed by ``)``, so ``"(B)"`` matches but ``"b"`` and
``"B. text"`` do not. ZeroEval's Llama-3.1-only boxed-answer fallback is omitted.
"""

import json
import re

from verifyit.adapters.completion import final_text
from verifyit.grade import Reward, scored

ANSWER_KEYS = ("reasoning", "answer")


def extract_first_complete_json(text: str) -> dict | None:
    """Parse the first balanced ``{...}`` span with newlines removed; None if it is not valid JSON."""
    depth = 0
    start = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : index + 1].replace("\n", ""))
                except json.JSONDecodeError:
                    return None
    return None


def extract_json_values(text: str) -> dict[str, str]:
    """Recover quoted, broken-quote, or unquoted ``key: value`` pairs from malformed JSON."""
    values = {}
    for key in ANSWER_KEYS:
        patterns = (
            (f'"{key}"\\s*:\\s*"([^"]*?)"', 0),
            (f'"{key}"\\s*:\\s*"(.*?)"', re.DOTALL),
            (f'"{key}"\\s*:\\s*([^,\\s]*)', 0),
            (f"{key}\\s*:\\s*([^,\\s]*)", 0),
        )
        for pattern, flags in patterns:
            match = re.search(pattern, text, flags)
            if match:
                values[key] = match.group(1)
                break
    return values


def zeroeval_answer(response: str) -> object | None:
    """Return the response's ``answer`` value, or None when ZeroEval counts it as no answer."""
    parsed = extract_first_complete_json(response)
    if not isinstance(parsed, dict) or "answer" not in parsed:
        parsed = extract_json_values(response)
    answer = parsed.get("answer")
    return None if answer is None or answer == "" else answer


def grade_zeroeval_mcqa(expected: str, response: str) -> Reward:
    answer = zeroeval_answer(final_text(response))
    correct = isinstance(answer, str) and (answer == expected or f"{expected})" in answer)
    extracted = answer if answer is None or isinstance(answer, str) else json.dumps(answer)
    return scored(float(correct), extracted=extracted, expected=expected)
