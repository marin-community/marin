# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the lock-only toolchain binding (the 9:28 AM directive).

A TaskSpec role whose environment is HostImage + packages_lock binds, at serving
time, to its composed digest-pinned toolchain image (lock_satisfied_by_image).
The backend survey must resolve the same binding so the fit shows image backends
(Daytona) as candidates and the role's disk quota rides the machine spec's own
storage sizing — with the task bytes, requirements and grader semantics unchanged.
"""

from pathlib import Path

import pytest
from rigging.filesystem.s3_compat import configure_coreweave_s3
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.requirements import Requirement, Support
from taskcompendium.backend_requirements import FitSupport

from experiments.post_training.task_curation import backend_capabilities, toolchain_binding
from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES, TOOLCHAIN_GRADER_PACKAGES
from experiments.post_training.task_curation.environment import Environment, Placement, placement
from experiments.post_training.task_curation.images.build import compile_lock
from experiments.post_training.task_curation.tests.test_backend_capabilities import (
    PDB_SHAPE,
    RUNTIME,
    _unit_checkers,
)
from experiments.post_training.task_curation.toolchain_binding import (
    matched_environment,
    resolved_toolchain_image,
)

DIGEST_PINNED = "ghcr.io/marin-community/rollout-images@sha256:" + "b" * 64
LOCK_PIN = "six==1.16.0"


@pytest.fixture
def frozen_lock(tmp_path: Path) -> str:
    """A frozen lock file whose content is the fixture environment's compiled lock."""
    environment = Environment(requirements=(LOCK_PIN,))
    lock = tmp_path / "requirements.lock"
    compile_lock(environment.requirements, lock)
    return str(lock)


def test_resolution_names_no_machine_without_a_built_image(frozen_lock: Path):
    """No built image, no binding: the fit keeps the pre-binding verdict."""
    assert (
        matched_environment(
            toolchain_binding.normalized_lock(Path(frozen_lock).read_text()), (Environment(requirements=(LOCK_PIN,)),)
        )
        is not None
    )
    assert resolved_toolchain_image(str(frozen_lock), environments=(Environment(requirements=(LOCK_PIN,)),)) is None


def test_resolution_binds_the_built_digest_pinned_image(frozen_lock: Path, monkeypatch):
    """The matched environment's composed image binds when the build recorded one."""
    environment = Environment(requirements=(LOCK_PIN,))
    monkeypatch.setattr(
        toolchain_binding, "built_image", lambda env, **kwargs: DIGEST_PINNED if env == environment else None
    )
    assert resolved_toolchain_image(str(frozen_lock), environments=(environment,)) == DIGEST_PINNED


def test_resolution_refuses_an_unmatched_lock(tmp_path: Path):
    other = tmp_path / "other.lock"
    compile_lock(("six==1.15.0",), other)
    assert resolved_toolchain_image(str(other), environments=(Environment(requirements=(LOCK_PIN,)),)) is None


def test_toolchain_variant_bakes_the_same_lock_and_the_host_programs():
    """The toolchain variant of a matched environment keeps the compiled lock and adds the host programs."""
    assert TOOLCHAIN_GRADER_PACKAGES.requirements == GRADER_PACKAGES.requirements
    assert placement(TOOLCHAIN_GRADER_PACKAGES) is Placement.BUILT_IMAGE
    assert placement(GRADER_PACKAGES) is Placement.WORKER
    fragment = TOOLCHAIN_GRADER_PACKAGES.dockerfile.read_text()
    for program in ("git", "build-essential", "curl", "ffmpeg"):
        assert program in fragment


def _lock_document(lock_url: str, storage_mb: int) -> dict:
    """One converted task whose grader is the lock-only HostImage shape with a disk quota."""
    return {
        "task": {
            "environment_requirements": {"command_semantics": "linux_process", "docker_image": PDB_SHAPE},
            "grader": {
                "environment": {"command_semantics": "linux_process", "packages_lock": lock_url},
                "timeout": 600,
            },
        },
        "runtime": {
            "verifier_machine": {
                "network": "deny",
                "cpus": 1,
                "memory_mb": 512,
                "storage_mb": storage_mb,
                "gpus": 0,
                "user": None,
                "startup_timeout": 60,
                "cleanup_timeout": 30,
            }
        },
    }


def _mapped(documents: list[dict], monkeypatch, image: str | None):
    # The production validators accept their probes without acquiring machines;
    # the unit lane's stubs reject everything, which would shadow the binding.
    checkers = {
        **_unit_checkers(),
        "daytona": DaytonaMachineFactory().check_environment,
        "iris": IrisMachineFactory(controller_url="http://survey.invalid").check_environment,
    }
    matrix = backend_capabilities.production_matrix(checkers)
    # The unit lane's Daytona stub rejects every probe; the real survey measured
    # the registry-image path's cells as pending live evidence while the
    # host-image and lock cells are hard inapplicabilities ("Daytona requires a
    # registry image or Docker build context"). Mirror that measured state so the
    # test exercises the binding, not the unit stub.
    matrix = [
        (
            cell.model_copy(update={"support": Support.VERIFY})
            if cell.backend == "daytona" and cell.requirement not in (Requirement.HOST_IMAGE, Requirement.PACKAGES_LOCK)
            else cell
        )
        for cell in matrix
    ]
    if image is None:
        monkeypatch.setattr(backend_capabilities, "resolved_toolchain_image", lambda lock_url: None)
    else:
        monkeypatch.setattr(backend_capabilities, "resolved_toolchain_image", lambda lock_url: image)
    return backend_capabilities.source_map(
        "bound", documents, matrix, checkers, {}, {"agent": RUNTIME, "grader": RUNTIME}
    )


def test_survey_binds_a_lock_only_grader_to_its_toolchain_image(monkeypatch):
    """With the binding, Daytona is a candidate for the grader and the quota rides the spec's storage."""
    mapped = _mapped([_lock_document("s3://bucket/requirements.lock", 2048)], monkeypatch, DIGEST_PINNED)
    daytona = mapped.backends["daytona"]
    assert daytona.not_applicable == {}
    assert daytona.roles["grader"] == {FitSupport.VERIFY: 1}
    local = mapped.backends["local"]
    # The quota is now on the machine spec the local checker sees: it rejects a
    # storage demand without a verified disk pool, so the binding enforces it.
    assert local.roles["grader"] == {FitSupport.UNSUPPORTED: 1}
    assert (
        "grader: LOCAL storage_mb requires an explicitly configured, host-verified disk_root" in local.validation_errors
    )
    # The reported demands stay the declaration's.
    assert mapped.demands["grader"].get(Requirement.HOST_IMAGE) == 1


def test_survey_keeps_the_pre_binding_verdict_without_a_built_image(monkeypatch):
    """Without a built image the fit is exactly what the seal recorded: not_applicable on image backends."""
    mapped = _mapped([_lock_document("s3://bucket/requirements.lock", 2048)], monkeypatch, None)
    daytona = mapped.backends["daytona"]
    assert daytona.not_applicable == {"grader:host_image": 1}
    local = mapped.backends["local"]
    assert local.roles["grader"] == {FitSupport.UNSUPPORTED: 1}
    assert (
        "grader: LOCAL storage_mb requires an explicitly configured, host-verified disk_root" in local.validation_errors
    )


def test_survey_demands_stay_declared_while_the_fit_resolves_the_machine(monkeypatch):
    """The bound role's machine-source demand reads as the registry image in the fit path only."""
    mapped = _mapped([_lock_document("s3://bucket/requirements.lock", 2048)], monkeypatch, DIGEST_PINNED)
    daytona = mapped.backends["daytona"]
    # docker_image is a candidate cell on Daytona: no host_image blocker, no exclusion.
    assert "grader:host_image" not in daytona.not_applicable
    assert "grader:host_image" not in daytona.blockers


@pytest.mark.data_integration
def test_nl2bash_frozen_lock_matches_the_declared_grader_environments():
    """NL2Bash's sealed lock content-matches the declared grader environments (the 9:28 evidence)."""
    configure_coreweave_s3()
    lock_url = "s3://marin-us-east-02a/marin/images/env-278169f50750da36/2026.10.08/requirements.lock"
    lock = toolchain_binding.read_lock(lock_url)
    assert lock is not None
    matched = matched_environment(lock)
    assert matched is not None
    # The composed toolchain image for the match is the conversion's deliverable:
    # the survey binds the role through it once the build records one.
    assert placement(matched) in (Placement.WORKER, Placement.BUILT_IMAGE)
