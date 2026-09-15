# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import the self-contained Python and native TaskTrove coding families."""

import ast
import tomllib
from typing import Any

from tasktrove_verify.spec import Mode

from taskcompendium.importers.tasktrove import TaskArchive, semantic_verifier
from taskcompendium.models import (
    AnswerRequirements,
    Capability,
    ContainerRuntime,
    Embedded,
    Rejected,
    RejectionReason,
    Resource,
    ResourceRole,
    StepSpecification,
    TaskMetadata,
    TaskRequirements,
    TaskSpec,
    WorkspaceState,
    relative_path,
)

PYTEST_FAMILY = "unit-test-gen"
STDIO_FAMILY = "competitive-programming"
PYTEST_CONVERTER = "python_unit_tests"
STDIO_CONVERTER = "codeforces"
PYTEST_PATH = "/tests/test_curriculum.py"
PYTEST_INTERPRETER = "/opt/tasktrove-pytest/bin/python"
_STDIO_EVALUATION_NOTE = (
    " The verifier compiles your solution (if needed), runs it against test cases, and compares stdout "
    "(a special judge is used where the problem allows multiple valid outputs)."
)


_PYTEST_PROMPT_REWRITES: dict[str, tuple[tuple[str, str], ...]] = {
    "1487": (
        ("exposes an imx.img API used by tests", "exposes the imx.img API"),
        (
            "Import style: tests will use from imx import img, so ensure __init__.py exposes the submodule.",
            "Import style: support `from imx import img` by exposing the submodule from __init__.py.",
        ),
        ("Deterministic encoding suitable for tests:", "Deterministic encoding:"),
        (
            "Tests that check export length for CmdWriteData and CmdNop (e.g., 12 for 1 entry, "
            "20 for 2 entries, 4 for CmdNop) should pass with the encoding described above.",
            "Export lengths are 12 for one CmdWriteData entry, 20 for two entries, and 4 for CmdNop.",
        ),
        ("6) Acceptance criteria (easy)", "6) Required behavior"),
        (
            "__init__.py exposes img so tests importing from imx import img work",
            "__init__.py exposes img so `from imx import img` works",
        ),
        ("All implemented parts are deterministic and test-friendly", "All implemented parts are deterministic"),
    ),
    "1488": (
        ("# Easy: Minimal QR code interface (tests-focused)", "# Easy: Minimal QR code interface"),
        ("prints a specific terminal prefix used by tests", "prints a specific terminal prefix"),
        ("provides a minimal API compatible with the tests", "provides the required minimal API"),
        (
            "produce the exact terminal-prefix output expected by the tests for the provided input",
            "produce the exact terminal-prefix output for the provided input",
        ),
        ("(e.g., io.StringIO in tests)", "(e.g., io.StringIO)"),
        ("2) Terminal rendering behavior (what tests check)", "2) Terminal rendering behavior"),
        (
            "The test will read the start of out.getvalue() and compare it to this prefix, so emit this sequence "
            "as the initial portion of the output.",
            "The output must begin with this prefix, so emit this sequence as its initial portion.",
        ),
        ("The module must be importable by tests and located at", "The module must be importable and located at"),
        (
            "The tests only require a specific prefix for the given input",
            "The required behavior is the specific prefix for the given input",
        ),
        ("sufficient for the provided tests to pass", "sufficient to provide the required API and output"),
    ),
    "1489": (
        (
            "support the pytest test that uses a TensorFlow-based SineLearner defined in the tests",
            "support a TensorFlow-based SineLearner defined by the caller",
        ),
        (
            "compatible with a TensorFlow-based SineLearner defined in the tests",
            "compatible with a TensorFlow-based SineLearner defined by the caller",
        ),
        ("Required functionality (simplified but sufficient for tests)", "Required functionality"),
        ("API (minimal, for tests to subclass)", "API (minimal, for subclasses)"),
        ("a deterministic, test-friendly flow", "a deterministic flow"),
        ("returned by learner.get_batch() in tests", "returned by learner.get_batch()"),
        ("for test reuse", "for reuse"),
        ("so tests can access dc.metalearning.MetaLearner", "so callers can access dc.metalearning.MetaLearner"),
        (
            "The test may define a TensorFlow-based SineLearner subclass",
            "A caller may define a TensorFlow-based SineLearner subclass",
        ),
        ("What the tests will do (summary)", "Usage"),
        ("to verify the outputs are reproducible", "and the outputs remain reproducible"),
        ("compatible with the provided tests", "compatible with the required API"),
        ("compatible with the provided pytest test", "compatible with the required API"),
        (
            "the test\u2019s SineLearner can still rely on TF if present",
            "a caller\u2019s SineLearner can still rely on TF if present",
        ),
    ),
}


def _clean_pytest_instructions(source_row: str, instructions: str) -> str:
    """Rewrite known pytest source boilerplate while retaining API requirements."""
    for original, replacement in _PYTEST_PROMPT_REWRITES.get(source_row, ()):
        instructions = instructions.replace(original, replacement)
    return instructions


def _metadata(archive: TaskArchive) -> dict[str, Any]:
    raw = archive.files.get("task.toml")
    if raw is None:
        raise ValueError("missing task.toml")
    metadata = tomllib.loads(raw.decode()).get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("task metadata is not a table")
    return metadata


def _tags(metadata: dict[str, Any]) -> tuple[str, ...]:
    tags = metadata.get("tags", ())
    return tuple(tag for tag in tags if isinstance(tag, str)) if isinstance(tags, list) else ()


def _resource(path: str, roles: tuple[ResourceRole, ...], data: bytes, *, executable: bool = False) -> Resource:
    relative_path(path)
    return Resource(path, roles, Embedded(data), executable=executable)


def _resources(archive: TaskArchive, mode: Mode) -> tuple[Resource, ...]:
    resources: list[Resource] = []
    if mode is Mode.PYTEST:
        path = PYTEST_PATH.removeprefix("/tests/")
        data = archive.files.get(path) or archive.files.get("tests/" + path)
        if data is None:
            raise ValueError(f"missing pytest verifier file: {path}")
        resources.append(_resource(path, (ResourceRole.VERIFIER,), data))
        for path, data in archive.files.items():
            if path.startswith("setup_files/"):
                resources.append(_resource(path, (ResourceRole.VERIFIER,), data))
    else:
        for path, data in archive.files.items():
            if path.startswith("tests/cases/"):
                resources.append(_resource(path.removeprefix("tests/"), (ResourceRole.VERIFIER,), data))
    # The cleanup converter's optional solution is validation-only material.  It
    # is intentionally never copied into the agent-visible environment.
    oracle = archive.files.get("solution/solution.py")
    solve = archive.files.get("solution/solve.sh") or archive.files.get("solve.sh")
    if oracle is not None:
        resources.append(_resource("oracle/solution.py", (ResourceRole.ORACLE,), oracle))
    if solve is not None:
        resources.append(_resource("oracle/solve.sh", (ResourceRole.ORACLE,), solve, executable=True))
    return tuple(resources)


def _validate_pytest(archive: TaskArchive, verifier: Any) -> None:
    if verifier.mode is not Mode.PYTEST:
        raise ValueError("python unit-test family does not have a pytest verifier")
    parameters = verifier.parameters
    if parameters.get("paths") != (PYTEST_PATH,) or parameters.get("python") != PYTEST_INTERPRETER:
        raise ValueError("pytest verifier must retain its absolute tests and interpreter paths")
    test = archive.files.get("tests/test_curriculum.py")
    if test is None:
        raise ValueError("missing tests/test_curriculum.py")
    try:
        tree = ast.parse(test.decode())
    except (SyntaxError, UnicodeDecodeError) as error:
        raise ValueError(f"pytest verifier is not valid Python: {error}") from error
    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
        for node in ast.walk(tree)
    ):
        raise ValueError("pytest verifier defines no test function")


def _validate_stdio(archive: TaskArchive, verifier: Any) -> None:
    if verifier.mode is not Mode.STDIO:
        raise ValueError("competitive-programming family does not have a stdio verifier")
    if archive.instructions.lstrip().startswith("This is an interactive problem."):
        raise LookupError("interactive stdio tasks require an interactive execution adapter")
    parameters = verifier.parameters
    if parameters.get("special_judge") is not None or parameters.get("cases") != "cases":
        raise ValueError("stdio special judges and relocated case trees are unsupported")
    if "solution.py" not in parameters.get("command", "") or "solution_bin" not in parameters.get("command", ""):
        raise ValueError("stdio command does not preserve the Python/native candidate contract")
    if "g++" not in parameters.get("build", "") or "-std=c++17" not in parameters.get("build", ""):
        raise ValueError("stdio build does not preserve the C++17 toolchain contract")
    cases = [path for path in archive.files if path.startswith("tests/cases/input_")]
    if not cases:
        raise ValueError("stdio verifier has no input cases")
    for path in cases:
        number = path.removeprefix("tests/cases/input_").removesuffix(".txt")
        if f"tests/cases/output_{number}.txt" not in archive.files:
            raise ValueError(f"stdio case {number} has no expected output")


def import_task(
    archive: TaskArchive,
    *,
    python_image: str | None = None,
    native_image: str | None = None,
) -> TaskSpec | Rejected:
    """Convert one retained coding archive using caller-resolved immutable images.

    Image resolution is deliberately an input to this importer: source Dockerfile
    tags are mutable and are not treated as a reproducible runtime contract.
    """
    source = archive.source
    try:
        metadata = _metadata(archive)
        converter = metadata.get("converter")
        if converter == PYTEST_CONVERTER and metadata.get("mode") == "pytest":
            family = PYTEST_FAMILY
            image = python_image
            task_shape = "environment-modification"
        elif converter == STDIO_CONVERTER and metadata.get("mode") == "stdio":
            family = STDIO_FAMILY
            image = native_image
            task_shape = "environment-modification"
        else:
            raise LookupError("unsupported coding converter")
        if archive.family not in {family, "coding"}:
            raise LookupError(f"unexpected coding family {archive.family!r}")
        if image is None:
            raise LookupError("an immutable toolchain image must be supplied by the caller")
        # Constructing both fields validates the caller's digest and keeps the
        # grading runtime identical to the agent's declared toolchain image.
        environment = TaskRequirements(
            (Capability.FILESYSTEM, Capability.SHELL, Capability.PROCESS), WorkspaceState(image=image, workdir="/app")
        )
        runtime = ContainerRuntime(image=image)
        verifier = semantic_verifier(
            archive.verifier,
            runtime,
            implementation_revision=archive.release.verifier_revision,
        )
        if verifier.mode == Mode.PYTEST:
            _validate_pytest(archive, verifier)
        else:
            _validate_stdio(archive, verifier)
        resources = _resources(archive, verifier.mode)
    except LookupError as error:
        return Rejected(source, RejectionReason.UNSUPPORTED_ENVIRONMENT, str(error))
    except (KeyError, UnicodeDecodeError, ValueError, tomllib.TOMLDecodeError) as error:
        return Rejected(source, RejectionReason.BROKEN_GRADER, str(error))
    return TaskSpec(
        id=f"tasktrove-{source.row}",
        requirements=environment,
        resources=resources,
        metadata=TaskMetadata(source=source, competencies=_tags(metadata), task_shape=task_shape),
        steps=(
            StepSpecification(
                instructions=(
                    _clean_pytest_instructions(source.row, archive.instructions)
                    if verifier.mode is Mode.PYTEST
                    else archive.instructions.replace(_STDIO_EVALUATION_NOTE, "")
                ).strip(),
                verifier=verifier,
                answer_requirements=AnswerRequirements("final_state"),
            ),
        ),
    )
