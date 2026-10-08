# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind observed RewardKit source contracts to a private, credentialed Iris job."""

import asyncio
import base64
import hashlib
import json
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

from rigging.secrets import SecretSpec
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.machine import Backend, MachineSpec, NetworkPolicy
from taskcompendium.grader import grader_config, verifyit_package
from taskcompendium.grading_result import Outcome
from taskcompendium.models import EnvironmentRequirements, ResourceGroups, TaskSpec, TextMessage
from taskcompendium.pipeline.execution_binding import VerificationRuntime, verification_machine
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    DatasetRecipe,
    ImportFailureKind,
    ImportRejection,
    NormalizedTask,
    RawRow,
    VerificationReport,
)
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import ScriptSpec

from experiments.post_training.task_curation.datasets.rewardkit.runtime import (
    SOURCE_JUDGE,
    SOURCE_TIMEOUT,
    VERDICT_FILENAME,
)
from experiments.post_training.task_curation.datasets.shared import grade_final_message

SOURCES = frozenset({"multichallenge"})
TIMEOUT = SOURCE_TIMEOUT + 30.0
RUNTIME_FILES = {
    "tests/test.sh": "db2681b4a2e86cdfd2699e2b12fd14806cf98040473a9e8956c570c78c99b97b",
    "tests/sitecustomize.py": "87e2c2c69264159f7d755442320a8f048f0e9687a2fba5cc8873433736ca1628",
    "environment/Dockerfile": "7f92fb2192bd4721685459bfc1b18954b68dcdb383f3be24f9bb53d649e00e65",
}
SOURCE_TEST_FILES = {"test.sh", "sitecustomize.py", "judge.toml", "conversation.txt", "verifier_data.json"}
DIAGNOSTIC_RESPONSE = "This is a diagnostic response used to exercise the original grading runtime."


def normalize(
    row: RawRow,
    *,
    image: str,
    normalize_task: Callable[[RawRow], TaskSpec | NormalizedTask | ImportRejection],
) -> TaskSpec | NormalizedTask | ImportRejection:
    """Preserve original runtime files and text actor; reject unknown executable variants."""
    files = {path: base64.b64decode(value, validate=True) for path, value in row.data["files"].items()}
    for path, digest in RUNTIME_FILES.items():
        if hashlib.sha256(files.get(path, b"")).hexdigest() != digest:
            return ImportRejection(
                kind=ImportFailureKind.UNSUPPORTED,
                reason="unsupported_rewardkit_runtime",
                detail=f"Unrecognized source {path}",
            )
    test_files = {path.removeprefix("tests/") for path in files if path.startswith("tests/")}
    if test_files != SOURCE_TEST_FILES:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="unsupported_rewardkit_layout",
            detail="Source tests differ from observed layout",
        )
    settings = tomllib.loads(files["task.toml"].decode())["verifier"]
    if settings["timeout_sec"] != SOURCE_TIMEOUT or settings["env"] != {
        "REWARDKIT_JUDGE": SOURCE_JUDGE,
        "TOGETHER_API_KEY": "${TOGETHER_API_KEY}",
    }:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="unsupported_rewardkit_provider",
            detail="Unrecognized source provider settings",
        )
    configuration = tomllib.loads(files["tests/judge.toml"].decode())
    if configuration.get("judge") != {
        "judge": SOURCE_JUDGE,
        "files": ["/tests/conversation.txt", "/app/response.txt"],
        "mode": "individual",
        "reasoning_effort": "low",
        "timeout": 300,
    } or any(
        set(criterion) != {"name", "description", "type", "min", "max"}
        for criterion in configuration.get("criterion", [])
    ):
        # A credentialed private job must not let source files redirect the
        # provider or attach arbitrary guest files to its judge requests.
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="unsupported_rewardkit_judge",
            detail="Unrecognized judge provider or file scope",
        )
    result = normalize_task(row)
    if isinstance(result, ImportRejection):
        return result
    task = result.task if isinstance(result, NormalizedTask) else result
    # RewardKit scans visible child directories instead of the root when any
    # exist. The bridge and archived source must stay in excluded __ directories.
    resources = (
        inline_resource("config.json", json.dumps(grader_config(task)).encode()),
        *(inline_resource(path[6:], content) for path, content in files.items() if path.startswith("tests/")),
        *(
            inline_resource("__source/" + path, content)
            for path, content in files.items()
            if not path.startswith(("tests/", "solution/"))
        ),
        inline_resource("__runtime/bridge.py", (Path(__file__).parent / "runtime.py").read_bytes()),
    )
    package = verifyit_package(
        ScriptSpec(path="__runtime/bridge.py", timeout=TIMEOUT, verdict_file=VERDICT_FILENAME),
        resources,
        environment=EnvironmentRequirements(docker_image=image, compatible_backends=(Backend.GVISOR,)),
    )
    task = task.model_copy(
        update={
            "grader": package.grader,
            "resources": ResourceGroups(
                verifier=package.resources,
                oracle=tuple(
                    inline_resource(path, content) for path, content in files.items() if path.startswith("solution/")
                ),
            ),
        }
    )
    return replace(result, task=task) if isinstance(result, NormalizedTask) else task


async def checks(task: TaskSpec, *, factory: IrisMachineFactory, machine_spec: MachineSpec) -> VerificationReport:
    """Exercise the empty-response guard and judge execution without inventing a semantic oracle."""
    results = []
    for name, response in (("empty_submission", ""), ("runtime_contract", DIAGNOSTIC_RESPONSE)):
        result = await grade_final_message(
            task, TextMessage(role="assistant", content=response), factory, machine_spec, TIMEOUT + 30
        )
        if result.status == Outcome.INFRA_ERROR:
            status = CheckStatus.INFRA_ERROR
        elif name == "empty_submission" and result.status == Outcome.SUBMISSION_FAILURE:
            status = CheckStatus.PASS
        elif result.status != Outcome.GRADED:
            status = CheckStatus.FAIL
        elif name == "runtime_contract" or result.reward == 0:
            status = CheckStatus.PASS
        else:
            status = CheckStatus.FAIL
        results.append(
            CheckResult(
                check=name, status=status, detail=f"{result.status}: reward={result.reward}; {result.error or ''}"
            )
        )
    results.append(
        CheckResult(
            check="oracle",
            status=CheckStatus.UNSUPPORTED if task.resources.oracle else CheckStatus.SKIPPED,
            detail=(
                "Source supplies oracle files whose execution is not bound"
                if task.resources.oracle
                else "Source supplies no golden response; runtime execution does not establish correctness"
            ),
        )
    )
    return VerificationReport(results)


def verification_report(task: TaskSpec, *, factory: IrisMachineFactory, machine_spec: MachineSpec) -> VerificationReport:
    return asyncio.run(checks(task, factory=factory, machine_spec=machine_spec))


def bind(recipe: DatasetRecipe, *, image: str, factory: IrisMachineFactory, machine_spec: MachineSpec) -> DatasetRecipe:
    """Use the same private ScriptSpec for verification and later text rollout grading."""
    if machine_spec.env:
        raise ValueError("Private judge credentials belong in factory secret references, never MachineSpec.env")
    return replace(
        recipe,
        policy=replace(
            recipe.policy,
            normalize=partial(normalize, image=image, normalize_task=recipe.policy.normalize),
            check_suite=CheckSuite(
                id="original-rewardkit-controls",
                revision="1",
                parameters={
                    "image": image,
                    "machine": asdict(machine_spec),
                    "provider_configured": "TOGETHER_API_KEY" in factory.secret_env,
                },
                run=partial(verification_report, factory=factory, machine_spec=machine_spec),
            ),
        ),
    )


def bind_rewardkit(
    recipe: DatasetRecipe,
    *,
    image: str,
    verification_runtime: VerificationRuntime,
    controller_url: str | None = None,
    qemu_bundle: Path | None = None,
    worker_image: str | None = None,
    verifier_secret_env: Mapping[str, SecretSpec] | None = None,
) -> DatasetRecipe:
    """Bind the original private RewardKit judge to a credentialed Iris runtime."""
    if verifier_secret_env is None or "TOGETHER_API_KEY" not in verifier_secret_env:
        raise ValueError("RewardKit verification requires an explicit private Together secret reference")
    factory, spec = verification_machine(
        image=image,
        verification_runtime=verification_runtime,
        memory_mb=4096,
        network=NetworkPolicy.ALLOW,
        compatible_runtimes=("iris-gvisor",),
        required_secret_names=("TOGETHER_API_KEY",),
        controller_url=controller_url,
        qemu_bundle=qemu_bundle,
        worker_image=worker_image,
        verifier_secret_env=verifier_secret_env,
    )
    assert isinstance(factory, IrisMachineFactory)
    return bind(recipe, image=image, factory=factory, machine_spec=spec)
