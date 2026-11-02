# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Author diagnostics of the program loader: refusals that name each mistake and the closest correct name.

None of this decides what a program may do; ``loader`` does. A program that reads a name it never
binds, through a missing import or a misspelling, would still fail without these checks, but with a
``NameError`` in the middle of a build, after the steps before it spent their model calls.
``check_names`` refuses it before ``loader.load_program_source`` runs it, and so does
``check_module_attributes`` for an attribute read from an imported module that lacks it, such as
``draft.GRADER_TIMEOUT``. Both ignore scope, so they miss a name bound in one function and read in
another.

A refusal names the closest name, if one is close, out of the program's bound names and the
caller's ``SdkExports``; for an SDK name it also gives the modules to import it from. The loader's
import hook asks ``import_suggestion`` for the same help when a program imports a name its module
lacks. It looks first among the names that module defines, then in the other modules, so a
misspelt name comes back with its spelling in that module, and a name imported from the wrong
module comes back with the module defining it.
"""

import ast
import difflib
import importlib
from collections.abc import Iterable

from pydantic import BaseModel

RENAMED_SDK_NAMES = {"try_grader": "grade_candidate"}
"""Former SDK names and the names replacing them. A program using a former name is told its replacement."""


class SdkExports(BaseModel, frozen=True):
    """Each public name the ``loader.ALLOWED_IMPORTS`` modules define, with the modules defining it.

    The refusals suggest from these names. The build SDK belongs to a later stage, so the loader
    never imports it: the build lists its names (``stage40_build_with_program.program_sdk.sdk_exports``)
    and passes them to ``load_program_source``.
    """

    defining_modules: dict[str, tuple[str, ...]]


def sentence(text: str) -> str:
    """``text`` closed as a sentence: unchanged when a suggestion's question mark ends it, else with a period."""
    return text if text.endswith("?") else f"{text}."


def suggestion(name: str, candidates: Iterable[str]) -> str:
    """``"; did you mean 'x'?"`` for the closest of ``candidates`` to ``name``, or ``""`` when none is close."""
    matches = difflib.get_close_matches(name, list(candidates), n=1)
    return f"; did you mean {matches[0]!r}?" if matches else ""


def _name_suggestion(name: str, bound: Iterable[str], sdk: SdkExports) -> str:
    """``"; did you mean 'x'?"`` for the closest of ``bound`` and the SDK exports to ``name``.

    A former SDK name in ``RENAMED_SDK_NAMES`` gets the name replacing it. An SDK export the
    program has not bound comes with its modules, as in ``"; did you mean 'research' from
    taskforge.stage40_build_with_program.sdk?"``. The result is ``""`` when no name is close.
    """
    exports = sdk.defining_modules
    renamed = RENAMED_SDK_NAMES.get(name)
    if renamed is not None:
        return f"; did you mean {renamed!r} from {' or '.join(exports[renamed])}?"
    local = set(bound)
    matches = difflib.get_close_matches(name, [*local, *exports], n=1)
    if not matches:
        return ""
    match = matches[0]
    if match in local:
        return f"; did you mean {match!r}?"
    return f"; did you mean {match!r} from {' or '.join(exports[match])}?"


def import_suggestion(name: str, module: str, sdk: SdkExports) -> str:
    """The suggestion for ``from module import name``, looking first among the names ``module`` defines.

    A close name from ``module`` comes first, and an exact name another module defines follows it:
    ``machine`` imported from ``taskforge.stage40_build_with_program.sandbox_effects`` gets ``build_machine``
    there, or ``machine`` from ``taskforge.spec.draft``. Without a close name in ``module``, or for
    a former SDK name, the suggestion is ``_name_suggestion``'s.
    """
    if name not in RENAMED_SDK_NAMES:
        exports = sdk.defining_modules
        own = sorted(export for export, defining in exports.items() if module in defining)
        matches = difflib.get_close_matches(name, own, n=1)
        elsewhere = [defining for defining in exports.get(name, ()) if defining != module]
        if matches and elsewhere:
            return f"; did you mean {matches[0]!r} from {module}, or {name!r} from {' or '.join(elsewhere)}?"
        if matches:
            return f"; did you mean {matches[0]!r} from {module}?"
    return _name_suggestion(name, (), sdk)


def _bound_names(tree: ast.Module) -> set[str]:
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.alias):
            bound.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler | ast.MatchAs | ast.MatchStar) and node.name is not None:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest is not None:
            bound.add(node.rest)
        elif isinstance(node, ast.Global | ast.Nonlocal):
            bound.update(node.names)
        elif isinstance(node, ast.TypeVar | ast.ParamSpec | ast.TypeVarTuple):
            bound.add(node.name)
    return bound


def check_names(
    tree: ast.Module,
    sdk: SdkExports,
    *,
    safe_builtins: frozenset[str],
    unsafe_builtins: frozenset[str],
    module_globals: frozenset[str],
) -> None:
    """Refuse a program that reads an unsafe builtin, or a name nothing binds.

    A name is bound when the program imports, defines or assigns it, takes it as a parameter, or
    finds it in ``safe_builtins`` or ``module_globals``, the names the loader binds before the run.

    Raises:
        ValueError: listing the unsafe builtins, or each unbound name with its first line and suggestion.
    """
    bound = _bound_names(tree)
    unsafe: set[str] = set()
    unbound: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load) or node.id in bound:
            continue
        if node.id in unsafe_builtins:
            unsafe.add(node.id)
        elif node.id not in safe_builtins and node.id not in module_globals:
            unbound[node.id] = min(node.lineno, unbound.get(node.id, node.lineno))
    if unsafe:
        raise ValueError(f"the program uses builtins that do not exist for build programs: {sorted(unsafe)}")
    if unbound:
        problems = [
            f"`{name}` (line {line}){_name_suggestion(name, bound | safe_builtins, sdk)}"
            for name, line in sorted(unbound.items(), key=lambda item: item[1])
        ]
        listed = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(
            f"the program reads names that no import, definition, assignment or builtin binds:\n{listed}\n"
            "Import each SDK name the program uses from its module."
        )


def _imported_modules(tree: ast.Module, allowed_imports: frozenset[str]) -> dict[str, str]:
    """The allowed modules the program binds as names, by the name: ``{"draft": "taskforge.spec.draft"}``."""
    modules: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update({alias.asname or alias.name: alias.name for alias in node.names})
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for alias in node.names:
                if f"{node.module}.{alias.name}" in allowed_imports:
                    modules[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return {name: path for name, path in modules.items() if path in allowed_imports}


def check_module_attributes(tree: ast.Module, sdk: SdkExports, allowed_imports: frozenset[str]) -> None:
    """Refuse an attribute the program reads from an imported module of ``allowed_imports`` that does not define it.

    Raises:
        ValueError: listing each missing attribute with its first line and suggestion.
    """
    modules = _imported_modules(tree, allowed_imports)
    missing: dict[tuple[str, str], int] = {}
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Attribute)
            or not isinstance(node.value, ast.Name)
            or not isinstance(node.ctx, ast.Load)
        ):
            continue
        path = modules.get(node.value.id)
        if path is None or hasattr(importlib.import_module(path), node.attr):
            continue
        read = (node.value.id, node.attr)
        missing[read] = min(node.lineno, missing.get(read, node.lineno))
    if not missing:
        return
    problems = [
        f"`{name}.{attr}` (line {line}): {modules[name]!r} has no name {attr!r}"
        f"{import_suggestion(attr, modules[name], sdk)}"
        for (name, attr), line in sorted(missing.items(), key=lambda item: item[1])
    ]
    listed = "\n".join(f"- {sentence(problem)}" for problem in problems)
    raise ValueError(f"the program reads attributes that the modules it imports do not define:\n{listed}")
