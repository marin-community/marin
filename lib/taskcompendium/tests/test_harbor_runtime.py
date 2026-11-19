# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from taskcompendium.convert.answers import exact_answer_task
from taskcompendium.harbor.runtime import (
    TaskNetworkPolicyReceipt,
    TaskSpecRuntimeBinding,
    network_policy_qualified,
    write_source_runtime,
)
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import Source
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import inline_resource

RUNTIME = {
    "task_machine": {"network": "allow"},
    "verifier_machine": {"network": "allow"},
}


def _binding(runtime: dict | None = None) -> TaskSpecRuntimeBinding:
    runtime = RUNTIME if runtime is None else runtime
    return TaskSpecRuntimeBinding(
        task_sha256="0" * 64,
        runtime_sha256=canonical_sha256(runtime),
        runtime=runtime,
    )


def _receipt(
    binding: TaskSpecRuntimeBinding,
    *,
    actor_required: bool = False,
    actor_effective: str | None = None,
    verifier_effective: str | None = None,
    runtime_sha256: str | None = None,
    verifier_declared: str | None = "allow",
    actor_declared: str | None = "allow",
) -> TaskNetworkPolicyReceipt:
    return TaskNetworkPolicyReceipt(
        source_runtime_sha256=binding.runtime_sha256 if runtime_sha256 is None else runtime_sha256,
        actor_required=actor_required,
        actor_declared=actor_declared,
        verifier_declared=verifier_declared,
        actor_effective=actor_effective,
        verifier_effective=verifier_effective,
    )


@pytest.mark.parametrize("verifier_effective", [None, "allow"])
def test_machine_less_grading_satisfies_the_declared_verifier_policy(
    verifier_effective,
):
    binding = _binding()
    assert network_policy_qualified(binding, _receipt(binding, verifier_effective=verifier_effective))


def test_machine_less_grading_satisfies_a_declared_deny_policy():
    runtime = {
        "task_machine": {"network": "allow"},
        "verifier_machine": {"network": "deny"},
    }
    binding = _binding(runtime)
    assert network_policy_qualified(binding, _receipt(binding, verifier_effective=None, verifier_declared="deny"))


def test_recorded_verifier_policy_drift_still_fails_qualification():
    binding = _binding()
    assert not network_policy_qualified(binding, _receipt(binding, verifier_effective="deny", verifier_declared="allow"))


def test_declared_policy_drift_still_fails_qualification():
    binding = _binding()
    assert not network_policy_qualified(binding, _receipt(binding, verifier_declared="deny"))


def test_runtime_sha256_drift_still_fails_qualification():
    binding = _binding()
    assert not network_policy_qualified(binding, _receipt(binding, runtime_sha256="1" * 64))


def test_missing_actor_observation_under_a_required_policy_still_fails_qualification():
    binding = _binding()
    assert not network_policy_qualified(binding, _receipt(binding, actor_required=True, actor_effective=None))


def test_recorded_actor_policy_drift_still_fails_qualification():
    binding = _binding()
    assert not network_policy_qualified(binding, _receipt(binding, actor_required=True, actor_effective="deny"))


def test_missing_receipt_with_declared_policies_fails_qualification():
    assert not network_policy_qualified(_binding(), None)


@pytest.mark.parametrize(
    "role,section",
    [("task_machine", "environment"), ("verifier_machine", "verifier.environment")],
)
def test_source_role_budget_rejects_conflicting_original_configuration(tmp_path, role, section):
    task = exact_answer_task(
        RawRow(
            "conflict",
            Source(dataset="fixture", revision="pin", row="0", importer_revision="1"),
            {},
        ),
        prompt="Return 42.",
        answers=("42",),
        ignore_case=False,
    )
    task = task.model_copy(
        update={
            "resources": task.resources.model_copy(
                update={
                    "oracle": (inline_resource("task.toml", f"[{section}]\ncpus = 1\n".encode()),),
                }
            )
        }
    )
    runtime = {role: {"cpus": 2}}
    with pytest.raises(ValueError, match="conflicts"):
        write_source_runtime(task, runtime, tmp_path)
    assert not (tmp_path / "source-runtime.json").exists()
