"""
The logic in this file largely borrows from Qwen2.5-Math codebase at https://github.com/QwenLM/Qwen2.5-Math:
"""

import re


def get_multiple_choice_answer(pred: str):
    # Try to pull out “Answer: X”, “Answer: {X}” or “Answer: \boxed{X}”
    m = re.search(r"(?:Exact\s+)?Answer:\s*(?:\\boxed)?\{?([A-Z])\}?", pred, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Fallback: isolate any single capital letter
    candidates = re.findall(r"(?<![A-Z])([A-Z])(?![A-Z])", pred.upper())
    if candidates:
        return candidates[-1]

    # Final fallback: return the whole trimmed string
    return pred.strip().rstrip(".").rstrip("/")
