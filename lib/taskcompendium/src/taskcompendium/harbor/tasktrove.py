# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source image recipes retained by the legacy TaskTrove conversion policy."""

import ast
import re
from pathlib import Path

from verifyit.spec import Mode, PytestSpec

from taskcompendium.convert.tasktrove import DOCKERFILE, TASKTROVE_REPO
from taskcompendium.convert.tasktrove_python_unit_tests import pytest_dockerfile
from taskcompendium.convert.verifyit_build import verifyit_build_context
from taskcompendium.models import DockerBuildContext, TaskSpec, VerifyitGrader, verifyit_spec
from taskcompendium.runtime.resources import resource_bytes

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
        dockerfile = pytest_dockerfile(dockerfile, ast.parse(resource_bytes(test)))
    # Legacy text converters retained only the source Dockerfile.
    # Declared executable build contexts are handled before this compatibility path.
    extras = ("schema", "judge") if source == "laion__nemotron-gym-structured-outputs-v4" else MODE_EXTRAS.get(mode, ())
    return verifyit_build_context(dockerfile, (recipe,), package=package, extras=extras)
