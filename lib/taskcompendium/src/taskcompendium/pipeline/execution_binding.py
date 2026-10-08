# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind executable grading procedures to explicit Shellbox verification runtimes."""

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

from rigging.secrets import SecretSpec
from shellbox.backends.gvisor.machine import GvisorMachineFactory
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.backends.qemu.machine import QemuMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Backend, DockerImage, MachineFactory, MachineSpec, NetworkPolicy, QemuBundle

from taskcompendium.datasets.executable_tasks import DEFAULT_OUTPUT_PATHS, ExecutableConversion
from taskcompendium.datasets.raw_conversion import RawConverter
from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    DOCKER_IMAGE_PATTERN,
    EnvironmentRequirements,
    OutputDirectory,
    ScriptGrader,
    TaskSpec,
    VerifyitGrader,
)
from taskcompendium.pipeline.inputs import SourceFiles, hub_inputs
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    DatasetRecipe,
    ImportRejection,
    NormalizedTask,
    RawRow,
    TaskPolicy,
    VerificationReport,
)
from taskcompendium.pipeline.recipes import hf_recipe
from taskcompendium.runtime.shell import machine_spec_identity

SANDBOX_TIMEOUT = 120.0
SANDBOX_MEMORY_MB = 512
IRIS_SCHEDULING_TIMEOUT = 180
IRIS_JOB_TTL = 1800
PINNED_IMAGE = re.compile(DOCKER_IMAGE_PATTERN)
VerificationRuntime = Literal["local-gvisor", "iris-gvisor", "qemu"]
BACKENDS = (Backend.DOCKER, Backend.GVISOR, Backend.QEMU)


class PrivateGraderBinder(Protocol):
    def __call__(
        self,
        recipe: DatasetRecipe,
        *,
        image: str,
        factory: MachineFactory,
        machine_spec: MachineSpec,
        worker_image: str | None,
        timeout: float,
    ) -> DatasetRecipe: ...


@dataclass(frozen=True)
class ExecutionAdapter:
    """A source's converter and grading contract, supplied by its declaration."""

    policy: Callable[[ExecutableConversion], TaskPolicy]
    converter: RawConverter
    converter_revision: str
    output_paths: tuple[str, ...] = DEFAULT_OUTPUT_PATHS
    output_directories: tuple[OutputDirectory, ...] = ()
    timeout: float = SANDBOX_TIMEOUT
    memory_mb: int = SANDBOX_MEMORY_MB
    network: NetworkPolicy = NetworkPolicy.DENY
    compatible_runtimes: tuple[VerificationRuntime, ...] = ("local-gvisor", "iris-gvisor", "qemu")
    required_secret_names: tuple[str, ...] = ()


def verification_machine(
    *,
    image: str,
    verification_runtime: Literal["local-gvisor", "iris-gvisor", "qemu"],
    memory_mb: int,
    network: NetworkPolicy = NetworkPolicy.DENY,
    compatible_runtimes: tuple[VerificationRuntime, ...] = ("local-gvisor", "iris-gvisor", "qemu"),
    required_secret_names: tuple[str, ...] = (),
    controller_url: str | None = None,
    qemu_bundle: Path | None = None,
    worker_image: str | None = None,
    verifier_secret_env: Mapping[str, SecretSpec] | None = None,
) -> tuple[MachineFactory, MachineSpec]:
    """Construct the explicitly selected backend with the source's runtime requirements."""
    if PINNED_IMAGE.fullmatch(image) is None:
        raise ValueError("Executable verification requires an immutable grader image")
    if verification_runtime not in compatible_runtimes:
        raise ValueError(f"Source verification requires one of {compatible_runtimes}")
    if required_secret_names and verifier_secret_env is not None and verification_runtime != "iris-gvisor":
        raise ValueError("Private provider binding requires Iris")
    if verification_runtime == "qemu":
        if network == NetworkPolicy.ALLOW:
            raise ValueError("Original module resolution requires network access; select Iris or local gVisor")
        if qemu_bundle is None:
            raise ValueError("QEMU verification requires a bundle path present on each worker")
        if worker_image is None or PINNED_IMAGE.fullmatch(worker_image) is None:
            raise ValueError("QEMU verification requires an immutable worker image containing the bundle")
        return QemuMachineFactory(), MachineSpec(
            QemuBundle(qemu_bundle), network=NetworkPolicy.DENY, memory_mb=memory_mb
        )
    if verification_runtime == "iris-gvisor":
        if controller_url is None or "@sha256:" not in image:
            raise ValueError("Iris verification requires a controller URL and a registry image pinned by digest")
        secrets = verifier_secret_env if required_secret_names else None
        if secrets is not None and set(secrets) != set(required_secret_names):
            raise ValueError(f"Private provider binding requires only these secret references: {required_secret_names}")
        return (
            IrisMachineFactory(
                controller_url=controller_url,
                scheduling_timeout=IRIS_SCHEDULING_TIMEOUT,
                job_ttl=IRIS_JOB_TTL,
                secret_env=secrets,
            ),
            MachineSpec(RegistryImage(image), network=network, memory_mb=memory_mb),
        )
    if verification_runtime == "local-gvisor":
        return GvisorMachineFactory(), MachineSpec(DockerImage(image), network=network, memory_mb=memory_mb)
    raise ValueError("Executable verification requires an explicit verification runtime")


def bind_executable(
    *,
    name: str,
    version: str,
    hf_id: str,
    revision: str,
    config: str,
    split: str,
    files: SourceFiles,
    adapter: ExecutionAdapter,
    image: str,
    verification_runtime: Literal["local-gvisor", "iris-gvisor", "qemu"],
    controller_url: str | None = None,
    qemu_bundle: Path | None = None,
    worker_image: str | None = None,
    verifier_secret_env: Mapping[str, SecretSpec] | None = None,
) -> DatasetRecipe:
    """Bind a source's explicit conversion and grader without name-based dispatch."""
    factory, spec = verification_machine(
        image=image,
        verification_runtime=verification_runtime,
        memory_mb=adapter.memory_mb,
        network=adapter.network,
        compatible_runtimes=adapter.compatible_runtimes,
        required_secret_names=adapter.required_secret_names,
        controller_url=controller_url,
        qemu_bundle=qemu_bundle,
        worker_image=worker_image,
        verifier_secret_env=verifier_secret_env,
    )
    conversion = ExecutableConversion(
        image=image,
        output_paths=adapter.output_paths,
        output_directories=adapter.output_directories,
        converter=adapter.converter,
        converter_revision=adapter.converter_revision,
        timeout=adapter.timeout,
        machine_factory=factory,
        machine_spec=spec,
        worker_image=worker_image,
    )
    return hf_recipe(
        name=name,
        version=version,
        hf_id=hf_id,
        revision=revision,
        config=config,
        split=split,
        inputs=hub_inputs(hf_id, revision, files),
        policy=adapter.policy(conversion),
    )


def bind_private_grader(
    recipe: DatasetRecipe,
    *,
    binder: PrivateGraderBinder,
    image: str,
    verification_runtime: VerificationRuntime,
    timeout: float = SANDBOX_TIMEOUT,
    memory_mb: int = SANDBOX_MEMORY_MB,
    controller_url: str | None = None,
    qemu_bundle: Path | None = None,
    worker_image: str | None = None,
    verifier_secret_env: Mapping[str, SecretSpec] | None = None,
) -> DatasetRecipe:
    """Bind a source's concrete private scorer to a network-disabled Shellbox runtime."""
    factory, machine_spec = verification_machine(
        image=image,
        verification_runtime=verification_runtime,
        memory_mb=memory_mb,
        controller_url=controller_url,
        qemu_bundle=qemu_bundle,
        worker_image=worker_image,
        verifier_secret_env=verifier_secret_env,
    )
    return binder(
        recipe,
        image=image,
        factory=factory,
        machine_spec=machine_spec,
        worker_image=worker_image,
        timeout=timeout,
    )


def grading_environment(image: str) -> EnvironmentRequirements:
    """A grader environment that runs ``image`` on any backend these bindings construct."""
    return EnvironmentRequirements(docker_image=image, compatible_backends=BACKENDS)


def bound_grader_task(task: TaskSpec, package: GraderPackage) -> TaskSpec:
    """Grade in the package's own machine without adding tools or requirements to the worker."""
    grader = package.grader
    if not isinstance(grader, VerifyitGrader | ScriptGrader) or grader.environment is None:
        raise TypeError(f"A {grader.kind} grader without an environment cannot grade in its own machine")
    return task.model_copy(
        update={
            "environment_requirements": EnvironmentRequirements(),
            "interaction_tools": (),
            "grader": grader,
            "resources": task.resources.model_copy(update={"verifier": package.resources}),
        }
    )


def native_runtime_report(result: GradeResult, witness_detail: str) -> VerificationReport:
    """Report diagnostic execution separately from an absent passing witness."""
    if result.status == Outcome.GRADED and result.reward is not None and 0 <= result.reward <= 1:
        status = CheckStatus.PASS
    elif result.status == Outcome.INFRA_ERROR:
        status = CheckStatus.INFRA_ERROR
    else:
        status = CheckStatus.FAIL
    return VerificationReport(
        checks=[
            CheckResult(
                check="native_runtime",
                status=status,
                detail=result.error or f"Diagnostic status={result.status}; reward={result.reward}",
            ),
            CheckResult(check="positive_witness", status=CheckStatus.SKIPPED, detail=witness_detail),
        ]
    )


def bind_grader_recipe(
    recipe: DatasetRecipe,
    *,
    normalize: Callable[[RawRow], TaskSpec | NormalizedTask | ImportRejection],
    verification: Callable[[TaskSpec], VerificationReport],
    suite_id: str,
    grader_bytes: bytes,
    image: str,
    factory: MachineFactory,
    machine_spec: MachineSpec,
    worker_image: str | None,
    timeout: float,
    verifier_revision: str,
) -> DatasetRecipe:
    machine = machine_spec_identity(machine_spec)
    suite = CheckSuite(
        id=suite_id,
        revision="1",
        parameters={
            "verifier_revision": verifier_revision,
            "grader_sha256": hashlib.sha256(grader_bytes).hexdigest(),
            "image": image,
            "backend": factory.backend.value,
            "machine": machine,
            "worker_image": worker_image,
            "timeout": timeout,
        },
        run=verification,
    )
    return replace(recipe, policy=replace(recipe.policy, normalize=normalize, check_suite=suite))
