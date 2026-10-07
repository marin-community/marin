# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep text-solving tasks while executing their original private SymPy graders."""

import base64
import hashlib
from collections.abc import Callable
from dataclasses import asdict, replace
from functools import partial

from shellbox.machine import Backend, MachineFactory, MachineSpec, QemuBundle
from taskcompendium.datasets import executable_tasks
from taskcompendium.grader import native_command_package
from taskcompendium.models import EnvironmentRequirements, ProviderRequirement, ResourceGroups, TaskSpec
from taskcompendium.native_grader import NativeCommandSpec
from taskcompendium.pipeline.models import (
    CheckSuite,
    DatasetRecipe,
    ImportFailureKind,
    ImportRejection,
    NormalizedTask,
    RawRow,
    VerificationReport,
)
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.shell import INTERFACE

SOURCES = frozenset({"math_gym", "math_prism", "math_oracle", "math_stack", "math_openreasoning"})
GYM_SCORER = "be1931919ee22ef704f565126353e7edec7b864dbd4a36590ab34593dd2004c7"
SCORER_RUNNERS = {
    GYM_SCORER: "7a92019aeea76076ad02e4bbca717db3c3c9396f068126e08beab34e81c3fa66",
    "703ea4d9abf2eb797c4e23ac6ff26c2f37699a62d9659d5af1475af7e8762f26": (
        "cc69c5b5b676f27249084dd101edfa2ca4dbf96c8d370bebb4f43922bde8943d"
    ),
}
BACKENDS = (Backend.DOCKER, Backend.GVISOR, Backend.QEMU)
ANSWER_PATH = "/app/answer.txt"
RUNTIME_CHECK = """import sys
from importlib.metadata import version

if sys.version_info[:2] != (3, 11):
    raise RuntimeError("Original math scorer requires Python 3.11")
for package, expected in PACKAGES.items():
    observed = version(package)
    if observed != expected:
        raise RuntimeError(f"Original math scorer requires {package}=={expected}; found {observed}")
"""


def normalize(
    row: RawRow,
    *,
    image: str,
    normalize_task: Callable[[RawRow], TaskSpec | NormalizedTask | ImportRejection],
) -> TaskSpec | NormalizedTask | ImportRejection:
    """Retain prompt normalization and bind the pinned source scorer without substitution."""
    files = {path: base64.b64decode(value, validate=True) for path, value in row.data.get("files", {}).items()}
    scorer = files.get("tests/verifier.py", b"")
    runner = files.get("tests/test.sh", b"")
    expected_runner = SCORER_RUNNERS.get(hashlib.sha256(scorer).hexdigest())
    if expected_runner is None or hashlib.sha256(runner).hexdigest() != expected_runner:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="unsupported_math_scorer",
            detail="Unrecognized original math scorer/runner",
        )
    if "tests/verifier_data.json" not in files:
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_verifier_data",
            detail="Original math verifier data is required",
        )
    result = normalize_task(row)
    if isinstance(result, ImportRejection):
        return result
    task = result.task if isinstance(result, NormalizedTask) else result
    # Gym catches scorer exceptions as zero. Check its runtime before that boundary,
    # so a missing dependency cannot make all negative controls pass without a golden.
    versions = {"sympy": "1.13.3", "antlr4-python3-runtime": "4.11.0"}
    if hashlib.sha256(scorer).hexdigest() == GYM_SCORER:
        versions["numpy"] = "2.1.3"
    runtime_check = RUNTIME_CHECK.replace("PACKAGES", repr(versions))
    resources = (
        *(
            inline_resource("source/" + path, content)
            for path, content in files.items()
            if not path.startswith("solution/")
        ),
        inline_resource("verifier.py", scorer),
        inline_resource("verifier_data.json", files["tests/verifier_data.json"]),
        inline_resource("test.sh", runner),
        inline_resource("runtime_check.py", runtime_check.encode()),
    )
    package = native_command_package(
        NativeCommandSpec(
            argv=("bash", "-c", "python3 /tests/runtime_check.py && bash /tests/test.sh"),
            cwd="/app",
            result_format="reward_file",
            result_path="/logs/verifier/reward.txt",
            timeout=600,
        ),
        resources,
    )
    verifier = package.verifier.model_copy(
        update={"environment_requirements": EnvironmentRequirements(docker_image=image, compatible_backends=BACKENDS)}
    )
    task = task.model_copy(
        update={
            "verifier": verifier,
            "output_paths": (ANSWER_PATH,),
            "resources": ResourceGroups(
                verifier=package.resources,
                oracle=tuple(
                    inline_resource(path, content) for path, content in files.items() if path.startswith("solution/")
                ),
            ),
        }
    )
    return replace(result, task=task) if isinstance(result, NormalizedTask) else task


def verification_report(
    task: TaskSpec, *, factory: MachineFactory, machine_spec: MachineSpec, timeout: float
) -> VerificationReport:
    """Give only the private oracle control a shell, keeping the solving task textual."""
    control_task = task.model_copy(
        update={
            "environment_requirements": EnvironmentRequirements(
                docker_image=task.verifier.environment_requirements.docker_image,
                compatible_backends=BACKENDS,
                capabilities=("shell", "filesystem"),
                tool_providers={"shell": ProviderRequirement(action_interface=INTERFACE, initial_state={})},
            )
        }
    )
    return executable_tasks.verification_report(
        control_task, factory=factory, machine_spec=machine_spec, timeout=timeout
    )


def bind(
    recipe: DatasetRecipe,
    *,
    image: str,
    factory: MachineFactory,
    machine_spec: MachineSpec,
    worker_image: str | None,
    timeout: float,
) -> DatasetRecipe:
    """Bind the same private grading image to source verification and serialized tasks."""
    machine = asdict(machine_spec)
    if isinstance(machine_spec.source, QemuBundle):
        machine["source"] = {"path": str(machine_spec.source.path)}
    suite = CheckSuite(
        id="original-sympy-controls",
        revision="1",
        parameters={
            "image": image,
            "backend": factory.backend.value,
            "machine": machine,
            "worker_image": worker_image,
            "timeout": timeout,
        },
        run=partial(verification_report, factory=factory, machine_spec=machine_spec, timeout=timeout),
    )
    return replace(
        recipe,
        policy=replace(
            recipe.policy,
            normalize=partial(normalize, image=image, normalize_task=recipe.policy.normalize),
            check_suite=suite,
        ),
    )
