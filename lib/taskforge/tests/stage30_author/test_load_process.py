# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A program loads in a child process: the manifest comes back, a refusal comes back, a spinning body is cut off."""

import inspect
import time

import pytest

from taskforge.stage30_author.author import program_manifest
from taskforge.stage30_author.load_process import ProgramManifest, load_program_in_process
from taskforge.stage30_author.loader import load_program_source
from taskforge.stage40_build_with_program.template import standard

SPINNING_CLASS = "class Spin:\n    while True:\n        pass\n\n\n"
"""An output model whose class body never returns: the loader runs it when it loads the module."""


def spinning_program() -> str:
    return inspect.getsource(standard).replace("PROGRAM = ", f"{SPINNING_CLASS}PROGRAM = ")


async def test_the_child_returns_the_manifest_the_loader_gives_in_process():
    source = inspect.getsource(standard)

    manifest = await load_program_in_process(source, wall_timeout=60)

    assert manifest == ProgramManifest.model_validate(program_manifest(load_program_source(source)))


async def test_the_loaders_refusal_comes_back_as_the_error():
    source = inspect.getsource(standard).replace("PROGRAM = ", "def helper():\n    return 1\n\n\nPROGRAM = ")

    with pytest.raises(ValueError, match="the function `helper` is not allowed at module level"):
        await load_program_in_process(source, wall_timeout=60)


async def test_a_module_body_that_never_returns_is_cut_off_at_the_wall_timeout():
    started = time.monotonic()

    with pytest.raises(ValueError, match="module body ran for more than 2 seconds"):
        await load_program_in_process(spinning_program(), wall_timeout=2)

    assert time.monotonic() - started < 15
