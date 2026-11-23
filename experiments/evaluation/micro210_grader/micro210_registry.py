# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grader registry for the embedded Micro210 grader.

Replaces the runtime-v4 wheel's ``eval.graders.specs`` registry with exactly the grader
names Micro210 rows select, mapped to the vendored implementations. The wrapper keeps the
``.function`` attribute the frozen driver calls, so the driver flow is unchanged.
"""

from collections.abc import Callable
from dataclasses import dataclass

from eval.graders import instructions, judged, math_specs, program_suites, structured
from micro210_crux import crux


@dataclass(frozen=True)
class RegisteredGrader:
    """One grader name bound to its vendored function."""

    function: Callable


GRADERS = {
    "boxed_math": RegisteredGrader(math_specs.boxed_math),
    "letter_choice": RegisteredGrader(math_specs.letter_choice),
    "mmlu_choice": RegisteredGrader(math_specs.mmlu_choice),
    "math_equivalence": RegisteredGrader(math_specs.math_equivalence),
    "nupa": RegisteredGrader(math_specs.nupa),
    "program_suite": RegisteredGrader(program_suites.program_suite),
    "instructions": RegisteredGrader(instructions.instruction_following),
    "nonce_similarity": RegisteredGrader(structured.nonce_similarity),
    "crux": RegisteredGrader(crux),
    "judge_failed": RegisteredGrader(judged.judge_failed),
    "equivalence_judge": RegisteredGrader(judged.equivalence_judge),
    "equivalence_label": RegisteredGrader(judged.equivalence_label),
}
