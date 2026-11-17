# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a build program from the Python source an author wrote.

``load_program_source`` is the plain loader. It runs the source with the full builtins and any
import the source asks for. The author PR replaces it with a restricted loader, which allows a
fixed set of imports and runs the source with safe builtins only.
"""

import linecache
import sys
import types

from taskforge.content_hash import sha256_hex
from taskforge.stage30_author.program import RLEnvBuildProgram

PROGRAM_ATTR = "PROGRAM"
"""The module attribute holding the source's program."""
MODULE_PREFIX = "taskforge_program_"


def load_program_source(source: str) -> RLEnvBuildProgram:
    """Run ``source`` as a fresh module and return its ``PROGRAM``.

    The loader names the module ``taskforge_program_`` plus the first 16 hex digits of the sha256
    of ``source``. It registers the module in ``sys.modules`` and its text in ``linecache``, so
    ``rl_env_build_step`` and ``rl_env_build_sink`` can read each step's source with
    ``inspect.getsource``.

    Raises:
        ValueError: the module defines no ``PROGRAM``, or its ``PROGRAM`` is not an ``RLEnvBuildProgram``.
    """
    name = f"{MODULE_PREFIX}{sha256_hex(source.encode())[:16]}"
    filename = f"<{name}>"
    module = types.ModuleType(name)
    module.__file__ = filename
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    sys.modules[name] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    if PROGRAM_ATTR not in vars(module):
        raise ValueError(f"{name} defines no {PROGRAM_ATTR}")
    program = vars(module)[PROGRAM_ATTR]
    if not isinstance(program, RLEnvBuildProgram):
        raise ValueError(f"{name}.{PROGRAM_ATTR} is a {type(program).__name__}, not an RLEnvBuildProgram")
    return program
