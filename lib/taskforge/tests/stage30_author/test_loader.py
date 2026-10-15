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
