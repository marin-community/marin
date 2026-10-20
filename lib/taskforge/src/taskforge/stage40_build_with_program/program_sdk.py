# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The build SDK's names as the program loader reads them, and the loader bound to them.

``stage30_author.loader`` belongs to an earlier stage than the build SDK, so it imports none of the
SDK's modules: ``sdk_exports`` lists the names they define here, and every caller of the loader in
this stage (the build child, the GLM author and the agentic author's tools) passes them in.
``load_program`` is the loader with these names, the ``StageFunctions.load_program`` of a run.
"""

import ast
import functools
import importlib
import inspect
from types import ModuleType

from taskforge.stage30_author.loader import ALLOWED_IMPORTS, SdkExports, load_program_source
from taskforge.stage30_author.program import RLEnvBuildProgram


def _defined_names(module: ModuleType) -> set[str]:
    """The public names ``module`` defines: its ``__all__``, or else its public names minus its top-level imports."""
    declared = getattr(module, "__all__", None)
    if declared is not None:
        return set(declared)
    public = {n for n, value in vars(module).items() if not n.startswith("_") and not isinstance(value, ModuleType)}
    try:
        tree = ast.parse(inspect.getsource(module))
    except (OSError, TypeError):
        # A module without Python source, such as math, has no import statements to subtract.
        return public
    imported = {
        (alias.asname or alias.name).split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    return public - imported


@functools.cache
def sdk_exports() -> SdkExports:
    """Each public name the ``ALLOWED_IMPORTS`` modules define, with the modules defining it.

    A module re-exporting an imported name does not count, so ``DOCKER_IMAGE_REQUIREMENTS`` maps to
    ``taskforge.stage40_build_with_program.sdk`` alone, although the standard template imports it as well.
    """
    modules: dict[str, list[str]] = {}
    for module in sorted(ALLOWED_IMPORTS):
        for name in _defined_names(importlib.import_module(module)):
            modules.setdefault(name, []).append(module)
    return SdkExports(defining_modules={name: tuple(defining) for name, defining in modules.items()})


def load_program(source: str) -> RLEnvBuildProgram:
    """``load_program_source`` with the build SDK's names."""
    return load_program_source(source, sdk_exports())
