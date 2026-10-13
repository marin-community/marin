# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load the source of an authored build program as an ``RLEnvBuildProgram``.

``load_program_source`` checks the source in three passes, and each refusal is a ``ValueError``
whose message an author can act on. The caller turns it into the next round's revision detail, and
the GLM author also sends it back within its structured call.

1. Before running anything it reads the module's top level. A step's memo key covers its own
   source alone, so a helper function or a constant beside the steps could change without
   changing any key. The top level may therefore hold only imports, classes (the output models of
   the steps), ``@rl_env_build_step`` steps, the ``@rl_env_build_sink`` sink and, last,
   ``PROGRAM = rl_env_build_program(..., sink=...)``.
   A step reads whatever else it needs from its inputs, its own body or the build SDK
   (``taskforge.stage40_build.sdk`` and its sibling modules), which ``SDK_VERSION`` covers. The
   pass also refuses a name the program reads without binding it anywhere, by an import, a
   definition, an assignment or a parameter, unless ``SAFE_BUILTINS`` holds it. A missing import
   or a misspelt name would otherwise fail in the middle of a build, after the steps before it
   spent their model calls. The refusal names the closest name, if one is close, out of the
   program's bound names and the names ``sdk_exports`` finds defined in the allowed modules; for
   an SDK name it also gives the modules to import it from. The check ignores scope, so it misses
   a name bound in one function and read in another.
2. It runs the module with imports limited to ``ALLOWED_IMPORTS`` and builtins to
   ``SAFE_BUILTINS``, which keeps a program to the SDK's functions; programs are trusted model
   output in the builder's process, so this is no security boundary. The build SDK belongs to a
   later stage than this module, so the program imports it by name at run time rather than this
   module importing it. Before running the module, the pass registers the source in ``linecache``
   under the module's file name, so the step decorator's ``inspect.getsource`` reads each step's
   text. An import of a name the module lacks gets a suggestion from ``sdk_exports`` as well. The
   loader looks first among the names the imported module defines, then in the other modules, so
   a misspelt name comes back with its spelling in that module, and a name imported from the
   wrong module comes back with the module defining it. The DAG checks
   of ``rl_env_build_program`` run here and refuse, among others, an input named after no step.
3. It reads ``PROGRAM`` and requires a step named ``GRADER_STEP`` returning ``GRADER_OUTPUT``, a
   sink made by ``@rl_env_build_sink`` returning ``TaskSpec``, and every step and sink the module
   defines to be part of ``PROGRAM``.

The module is registered in ``sys.modules`` as ``taskforge_program_<sha256 prefix>``, one name per
distinct source, so pydantic resolves the program's own output models while the store decodes
them, and loading a source twice returns the same program. A refused source is unregistered.
"""

import ast
import builtins
import difflib
import functools
import importlib
import inspect
import linecache
import sys
import types
from collections.abc import Iterable, Mapping, Sequence
from types import ModuleType

from taskcompendium.models import TaskSpec

from taskforge.content_hash import sha256_hex
from taskforge.stage30_author.program import RLEnvBuildProgram, RLEnvBuildSink, RLEnvBuildStep

PROGRAM_NAME = "PROGRAM"
STEP_DECORATOR = "rl_env_build_step"
SINK_DECORATOR = "rl_env_build_sink"
SINK_KEYWORD = "sink"
PROGRAM_CONSTRUCTOR = "rl_env_build_program"
MODULE_PREFIX = "taskforge_program_"
GRADER_STEP = "grader"
"""The step whose output validation reads the reference answer and the prototyped candidates from."""
GRADER_OUTPUT = "taskforge.stage40_build.outputs.Grader"
"""The qualified name of the type the grader step returns."""

ALLOWED_IMPORTS = frozenset(
    {
        "taskforge.stage10_proposal.model",
        "taskforge.stage30_author.program",
        "taskforge.stage40_build.context",
        "taskforge.stage40_build.outputs",
        "taskforge.stage40_build.sandbox_effects",
        "taskforge.stage40_build.sdk",
        "taskforge.stage40_build.step_store",
        "taskforge.stage40_build.template.standard",
        "taskforge.spec.draft",
        "taskforge.spec.verdict",
        "taskforge.spec.controls",
        "taskforge.llm.agent",
        "taskforge.llm.policy",
        "taskforge.sandbox.images",
        "taskcompendium.grader",
        "taskcompendium.models",
        "taskcompendium.runtime.resources",
        "taskcompendium.grading",
        "taskcompendium.grading_result",
        "taskcompendium.submission",
        "verifyit.spec",
        "pydantic",
        "__future__",
        "asyncio",
        "base64",
        "hashlib",
        "collections",
        "collections.abc",
        "dataclasses",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "itertools",
        "json",
        "math",
        "re",
        "shlex",
        "statistics",
        "string",
        "textwrap",
        "typing",
    }
)
UNSAFE_BUILTINS = frozenset(
    {"open", "exec", "eval", "compile", "input", "breakpoint", "globals", "locals", "vars", "help", "exit", "quit"}
)
RENAMED_SDK_NAMES = {"try_grader": "grade_candidate"}
"""Former SDK names and the names replacing them. A program using a former name is told its replacement."""
MODULE_RULE = (
    "An authored program defines only imports, output models (classes), steps (`@rl_env_build_step async def`), "
    "the sink (`@rl_env_build_sink async def ... -> TaskSpec`) and, last, "
    "`PROGRAM = rl_env_build_program(<steps>, sink=<sink>)` at module level. A step's memo key covers its own "
    "source and its output model alone, so whatever else a step reads belongs in its body or its inputs, or comes "
    "from an SDK function."
)


def _restricted_import(
    name: str,
    globals: Mapping[str, object] | None = None,  # noqa: A002 - __import__ signature
    locals: Mapping[str, object] | None = None,  # noqa: A002 - __import__ signature
    fromlist: Sequence[str] | None = None,
    level: int = 0,
) -> ModuleType:
    submodules = [f"{name}.{item}" for item in fromlist or ()]
    allowed = name in ALLOWED_IMPORTS or (bool(submodules) and all(module in ALLOWED_IMPORTS for module in submodules))
    if level != 0 or not allowed:
        refusal = _sentence(f"build programs may not import {name!r}{_suggestion(name, ALLOWED_IMPORTS)}")
        raise ImportError(f"{refusal} Allowed: {sorted(ALLOWED_IMPORTS)}")
    if "." in name and not fromlist:
        raise ImportError(f"use 'from {name} import ...' rather than 'import {name}'")
    if name not in ALLOWED_IMPORTS:
        for module in submodules:
            importlib.import_module(module)
    module = importlib.import_module(name)
    for item in fromlist or ():
        if item != "*" and not hasattr(module, item):
            raise ImportError(f"{name!r} has no name {item!r}{_import_suggestion(item, name)}")
    return module


def _sentence(text: str) -> str:
    """``text`` closed as a sentence: unchanged when a suggestion's question mark ends it, else with a period."""
    return text if text.endswith("?") else f"{text}."


def _suggestion(name: str, candidates: Iterable[str]) -> str:
    """``"; did you mean 'x'?"`` for the closest of ``candidates`` to ``name``, or ``""`` when none is close."""
    matches = difflib.get_close_matches(name, list(candidates), n=1)
    return f"; did you mean {matches[0]!r}?" if matches else ""


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
def sdk_exports() -> dict[str, tuple[str, ...]]:
    """Each public name the ``ALLOWED_IMPORTS`` modules define, with the modules defining it.

    The loader imports those modules here, when it first builds a suggestion, since the build SDK
    belongs to a later stage. A module re-exporting an imported name does not count, so
    ``DOCKER_IMAGE_REQUIREMENTS`` maps to ``taskforge.stage40_build.sdk`` alone, although the
    standard template imports it as well.
    """
    modules: dict[str, list[str]] = {}
    for module in sorted(ALLOWED_IMPORTS):
        for name in _defined_names(importlib.import_module(module)):
            modules.setdefault(name, []).append(module)
    return {name: tuple(defining) for name, defining in modules.items()}


def _name_suggestion(name: str, bound: Iterable[str]) -> str:
    """``"; did you mean 'x'?"`` for the closest of ``bound`` and the SDK exports to ``name``.

    A former SDK name in ``RENAMED_SDK_NAMES`` gets the name replacing it. An SDK export the
    program has not bound comes with its modules, as in ``"; did you mean 'research' from
    taskforge.stage40_build.sdk?"``. The result is ``""`` when no name is close.
    """
    exports = sdk_exports()
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


def _import_suggestion(name: str, module: str) -> str:
    """The suggestion for ``from module import name``, looking first among the names ``module`` defines.

    A close name from ``module`` comes before an exact one from another module: ``machine``
    imported from ``taskforge.stage40_build.sandbox_effects`` gets ``build_machine`` there, although
    ``taskforge.spec.draft`` defines a ``machine``. Without a close name in ``module``, or for a
    former SDK name, the suggestion is ``_name_suggestion``'s.
    """
    if name not in RENAMED_SDK_NAMES:
        own = sorted(export for export, defining in sdk_exports().items() if module in defining)
        matches = difflib.get_close_matches(name, own, n=1)
        if matches:
            return f"; did you mean {matches[0]!r} from {module}?"
    return _name_suggestion(name, ())


SAFE_BUILTINS: dict[str, object] = {
    **{name: value for name, value in vars(builtins).items() if name not in UNSAFE_BUILTINS},
    "__import__": _restricted_import,
}


def module_name(source: str) -> str:
    """The ``sys.modules`` name of the program compiled from ``source``."""
    return f"{MODULE_PREFIX}{sha256_hex(source.encode())[:16]}"


def load_program_source(source: str) -> RLEnvBuildProgram:
    """Check and run ``source`` as a build program module and return its ``PROGRAM``.

    Raises:
        ValueError: the source does not parse, breaks the module-level rule, uses an unsafe
            builtin, fails while it runs (an import outside ``ALLOWED_IMPORTS`` or a DAG check of
            ``rl_env_build_program``, for example), or its ``PROGRAM`` lacks a ``grader`` step
            returning the build SDK's ``Grader`` or a sink made by ``@rl_env_build_sink``.
    """
    name = module_name(source)
    if name in sys.modules:
        return sys.modules[name].PROGRAM
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError(f"the program does not parse: {error}") from error
    _check_module_level(tree)
    _check_names(tree)
    filename = f"<{name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    module = types.ModuleType(name)
    module.__dict__.update({"__builtins__": SAFE_BUILTINS, "__file__": filename})
    sys.modules[name] = module
    try:
        try:
            exec(compile(tree, filename, "exec"), module.__dict__)
        except Exception as error:
            raise ValueError(f"the program failed to load: {type(error).__name__}: {error}") from error
        return _checked_program(module)
    except ValueError:
        del sys.modules[name]
        raise


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _is_step(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.AsyncFunctionDef)
        and len(node.decorator_list) == 1
        and isinstance(node.decorator_list[0], ast.Name)
        and node.decorator_list[0].id in (STEP_DECORATOR, SINK_DECORATOR)
    )


def _is_program_assignment(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == PROGRAM_NAME
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == PROGRAM_CONSTRUCTOR
    )


def _statement(node: ast.stmt) -> str:
    if isinstance(node, ast.FunctionDef):
        return f"the function `{node.name}`"
    if isinstance(node, ast.AsyncFunctionDef):
        return (
            f"the async function `{node.name}`, which is not a single-decorator "
            f"`@{STEP_DECORATOR}` step or `@{SINK_DECORATOR}` sink"
        )
    if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign | ast.TypeAlias):
        return f"an assignment (`{ast.unparse(node).splitlines()[0][:80]}`)"
    return f"a statement (`{ast.unparse(node).splitlines()[0][:80]}`)"


def _check_module_level(tree: ast.Module) -> None:
    statements = tree.body[1:] if tree.body and _is_docstring(tree.body[0]) else tree.body
    if not statements or not _is_program_assignment(statements[-1]):
        raise ValueError(
            f"the program must end with `{PROGRAM_NAME} = {PROGRAM_CONSTRUCTOR}(<steps>, sink=<sink>)`. "
            f"{MODULE_RULE}"
        )
    assignment = statements[-1]
    assert isinstance(assignment, ast.Assign) and isinstance(assignment.value, ast.Call)
    call = assignment.value
    if not any(keyword.arg == SINK_KEYWORD for keyword in call.keywords):
        raise ValueError(
            f"line {call.lineno}: pass the final step to `{PROGRAM_CONSTRUCTOR}` as `{SINK_KEYWORD}=`, "
            f"decorated with `@{SINK_DECORATOR}`. {MODULE_RULE}"
        )
    for node in statements[:-1]:
        if isinstance(node, ast.Import | ast.ImportFrom | ast.ClassDef) or _is_step(node):
            continue
        raise ValueError(f"line {node.lineno}: {_statement(node)} is not allowed at module level. {MODULE_RULE}")


MODULE_GLOBALS = frozenset({"__file__"})
"""Names ``load_program_source`` binds in the module before it runs the program."""


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


def _check_names(tree: ast.Module) -> None:
    bound = _bound_names(tree)
    unsafe: set[str] = set()
    unbound: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load) or node.id in bound:
            continue
        if node.id in UNSAFE_BUILTINS:
            unsafe.add(node.id)
        elif node.id not in SAFE_BUILTINS and node.id not in MODULE_GLOBALS:
            unbound[node.id] = min(node.lineno, unbound.get(node.id, node.lineno))
    if unsafe:
        raise ValueError(f"the program uses builtins that do not exist for build programs: {sorted(unsafe)}")
    if unbound:
        problems = [
            f"`{name}` (line {line}){_name_suggestion(name, bound | SAFE_BUILTINS.keys())}"
            for name, line in sorted(unbound.items(), key=lambda item: item[1])
        ]
        listed = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(
            f"the program reads names that no import, definition, assignment or builtin binds:\n{listed}\n"
            "Import each SDK name the program uses from its module."
        )


def _checked_program(module: ModuleType) -> RLEnvBuildProgram:
    program = module.__dict__.get(PROGRAM_NAME)
    if not isinstance(program, RLEnvBuildProgram):
        raise ValueError(f"`{PROGRAM_NAME}` must be what `{PROGRAM_CONSTRUCTOR}(...)` returns")
    grader = program.steps.get(GRADER_STEP)
    if grader is None or _qualified_name(grader.output_type) != GRADER_OUTPUT:
        raise ValueError(f"the program needs a step named {GRADER_STEP!r} that returns {GRADER_OUTPUT}")
    sink = program.sink
    if not isinstance(sink, RLEnvBuildSink):
        raise ValueError(f"the sink {sink.name!r} must be decorated with `@{SINK_DECORATOR}`, not `@{STEP_DECORATOR}`")
    if sink.output_type is not TaskSpec:
        raise ValueError(f"the sink {sink.name!r} returns {sink.output_type!r}; it must return TaskSpec")
    defined = {
        value.name
        for value in vars(module).values()
        if isinstance(value, RLEnvBuildStep | RLEnvBuildSink) and value.fn.__module__ == module.__name__
    }
    unused = sorted(defined - {*program.steps, sink.name})
    if unused:
        raise ValueError(f"steps {unused} are defined but not passed to `{PROGRAM_CONSTRUCTOR}`")
    return program


def _qualified_name(output_type: object) -> str:
    return f"{getattr(output_type, '__module__', '')}.{getattr(output_type, '__qualname__', '')}"
