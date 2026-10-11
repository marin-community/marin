# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source image recipes retained by the legacy TaskTrove conversion policy."""

import ast
import re
from pathlib import Path

from taskcompendium.models import DockerBuildContext, TaskSpec, VerifyitGrader, verifyit_spec
from taskcompendium.runtime.resources import resource_bytes
from verifyit.spec import Mode, PytestSpec

from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import DOCKERFILE, TASKTROVE_REPO
from experiments.post_training.task_curation.datasets.tasktrove.conversion.verifyit_build import verifyit_build_context

ARC_SOURCES = frozenset(
    {"laion__nemotron-gym-arc-agi-python-inductive-v2", "laion__nemotron-gym-arc-agi-transductive-v3"}
)

JUDGE_SOURCES = frozenset(
    {
        "laion__nemotron-gym-knowledge-openqa-v4",
        "laion__nemotron-gym-science-so-openq-v3",
        "laion__nemotron-gym-multichallenge-advanced-v4",
        "laion__stackexchange-codereview-sandboxes-verified-v2",
        "laion__glaive-code-assistant-sandboxes-verified-v2",
        "laion__nemotron-gym-safety-v3",
        "laion__stackexchange-overflow-sandboxes-verified-v2",
        "laion__stackexchange-superuser-sandboxes-verified-v2",
        "laion__stackexchange-tezos-sandboxes-verified-v2",
        "laion__stackexchange-unix-sandboxes-verified-v2",
        "laion__wizardlm-orca-v4",
    }
)
OLD_JUDGE_INSTALL = re.compile(r"rewardkit|litellm", re.IGNORECASE)
OLD_PUZZLE_INSTALL = re.compile(r"pip install .*\bpytest\b")
OLD_REASONING_INSTALL = re.compile(r"^RUN pip install --no-cache-dir reasoning-gym")
MODE_EXTRAS: dict[str, tuple[str, ...]] = {
    Mode.MATH: ("answer",),
    Mode.JSON_SCHEMA: ("schema",),
    Mode.REASONING_GYM: ("reasoning-gym",),
    Mode.JUDGE: ("judge",),
}


PYTEST_ENV = "/opt/tasktrove-pytest"
PYTEST_INSTALL = (
    f"RUN python3 -m venv --system-site-packages {PYTEST_ENV}"
    f" && {PYTEST_ENV}/bin/pip install --no-cache-dir pytest pytest-json-report{{dependencies}}\n"
)


def legacy_pytest_dockerfile(dockerfile: str, tree: ast.Module) -> str:
    """Install the legacy isolated pytest interpreter and its test-import dependencies."""
    modules = {
        alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    } | {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    dependencies = " mock" if "mock" in modules else ""
    return dockerfile.rstrip() + "\n" + PYTEST_INSTALL.format(dependencies=dependencies)


def source_actor_build(task: TaskSpec, *, source: str, mode: str, package: Path | None) -> DockerBuildContext | None:
    """Recover the old actor recipe from private provenance without changing the TaskSpec."""
    if task.source.dataset != TASKTROVE_REPO:
        return None
    recipe = next((resource for resource in task.resources.oracle if resource.path == DOCKERFILE), None)
    if recipe is None:
        return None
    if package is None:
        raise ValueError("TaskTrove source recipes require the explicit Verifyit source package root")
    dockerfile = resource_bytes(recipe).decode()
    pattern = None
    if source in JUDGE_SOURCES:
        pattern = OLD_JUDGE_INSTALL
    elif source == "laion__all-puzzles-v2":
        pattern = OLD_PUZZLE_INSTALL
    elif source == "laion__nemotron-gym-reasoning-gym-v2":
        pattern = OLD_REASONING_INSTALL
    if pattern is not None:
        dockerfile = "\n".join(line for line in dockerfile.splitlines() if not pattern.search(line)) + "\n"
    if isinstance(task.grader, VerifyitGrader) and mode == Mode.PYTEST:
        spec = verifyit_spec(task.grader)
        assert isinstance(spec, PytestSpec)
        resources = {resource.path: resource for resource in task.resources.verifier}
        test = resources[spec.paths[0].removeprefix("/tests/")]
        dockerfile = legacy_pytest_dockerfile(dockerfile, ast.parse(resource_bytes(test)))
    if source in ARC_SOURCES:
        # NVARC runs with system Python, including its candidate subprocess. Verifyit's
        # isolated tool environment cannot supply these imports or the reward writer's tomli_w.
        dockerfile = dockerfile.rstrip() + "\nRUN python3 -m pip install --no-cache-dir numpy requests 'tomli-w>=1.2'\n"
    # Legacy text converters retained only the source Dockerfile.
    # Declared executable build contexts are handled before this compatibility path.
    extras = ("schema", "judge") if source == "laion__nemotron-gym-structured-outputs-v4" else MODE_EXTRAS.get(mode, ())
    return verifyit_build_context(dockerfile, (recipe,), package=package, extras=extras)
