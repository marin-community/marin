# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from taskforge.stage30_author.loader import load_program_source

STEP = """\
from taskforge.stage30_author.program import rl_env_build_program, rl_env_build_step


@rl_env_build_step
async def only(context) -> str:
    return "only"
"""


@pytest.mark.parametrize(
    ("source", "refusal"),
    [
        (STEP, "defines no PROGRAM"),
        (f"{STEP}\nPROGRAM = only\n", "PROGRAM is a RLEnvBuildStep, not an RLEnvBuildProgram"),
    ],
    ids=["missing", "a step"],
)
def test_a_source_without_a_program_in_PROGRAM_is_refused(source, refusal):
    with pytest.raises(ValueError, match=refusal):
        load_program_source(source)
