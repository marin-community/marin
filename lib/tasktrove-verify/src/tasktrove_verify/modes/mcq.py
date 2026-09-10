# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mode mcq: the option letter the candidate wrote on its ``Answer:`` line.

Extraction is the nemotron_gym MCQA pattern, taking the last ``Answer: X`` in the output. Half of
the Nemotron prompts ask for ``Answer: \\boxed{X}`` and models also write ``**Answer:** (X)``, so
``\\boxed{}``, markdown emphasis, backticks, and brackets around the letter are dropped before
matching. The original verifier fell back to any trailing single letter when that pattern missed;
the fallback scores prose that never states an answer, so it is not reproduced here. Output with no
``Answer:`` line scores zero, as does a letter outside ``A``..the last option.
"""

import re
import string
from pathlib import Path

from tasktrove_verify.output import read_output
from tasktrove_verify.reward import InvalidTask, Reward, scored
from tasktrove_verify.spec import McqSpec

ANSWER = re.compile(r"Answer\s*:\s*(?!Answer)\s*([A-Za-z0-9])\s*")
BOXED_LETTER = re.compile(r"\\boxed\{\s*([A-Za-z0-9])\s*\}")
WRAPPERS = re.compile(r"[*`_()\[\]]")
MAX_OPTIONS = len(string.ascii_uppercase)


def answer_letters(text: str) -> list[str]:
    """Every letter stated on an ``Answer:`` line, wrappers around the letter removed."""
    return ANSWER.findall(WRAPPERS.sub("", BOXED_LETTER.sub(r"\1", text)))


def grade(spec: McqSpec, tests_dir: Path, workspace: Path) -> Reward:
    if not 1 <= spec.options <= MAX_OPTIONS:
        raise InvalidTask(f"mcq options must be 1..{MAX_OPTIONS}, got {spec.options}")
    letters = string.ascii_uppercase[: spec.options]
    expected = spec.expected.strip().upper()
    if expected not in letters:
        raise InvalidTask(f"mcq expected {spec.expected!r} is not one of {letters!r}")

    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")
    matches = answer_letters(text)
    if not matches:
        return scored(0.0, reason="no_answer_line", expected=expected)
    extracted = matches[-1].strip().upper()
    if extracted not in letters:
        return scored(0.0, reason="out_of_range", extracted=extracted, expected=expected)
    return scored(float(extracted == expected), extracted=extracted, expected=expected)
