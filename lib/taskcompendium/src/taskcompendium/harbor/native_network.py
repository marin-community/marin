# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct pinned native Daytona policies from archived configuration, without outcomes."""

import hashlib
import tomllib
from pathlib import PurePosixPath
from typing import Any, Literal

from finestore.rl.curation import ContentEvidenceReference
from pydantic import BaseModel, ConfigDict, Field, model_validator
from shellbox.machine import NetworkPolicy

from taskcompendium.harbor.runtime import TaskNetworkPolicyReceipt, TaskSpecRuntimeBinding
from taskcompendium.identity import canonical_sha256

# These revisions use the same native allow_internet -> MachineSpec.network ->
# _sandbox_parameters mapping. Adding another revision requires checking that path.
NATIVE_DAYTONA_NETWORK_IMPLEMENTATIONS = {
    "3ba52c03e9599b98b38d4851885383043e7568cd": {
        "vendor/harbor/src/harbor_config/models/task/config.py": (
            "a722fece73b44ded382432ab8084073bd11d934f3a25d07d498dffbce18d1bcf"
        ),
        "vendor/harbor/src/harbor/environments/daytona.py": (
            "159ec1504da8c5805419c4f40b459f66de14231a63f76c10bcf32f66f7740fbe"
        ),
        "vendor/harbor/src/harbor/environments/factory.py": (
            "cfea2f511cde24b0cc7a9229c9dcb38cdac771da90b2454d7709674567bd70ae"
        ),
        "lib/shellbox/src/shellbox/backends/daytona/machine.py": (
            "4db013bc5f80328ca3205b996be94f8942ad18d7320a634125f16eedccd83079"
        ),
        "vendor/harbor/uv.lock": "0deef777f30ec9eb58a7165b40225c709c51daa704801a6f420ff9b543335aba",
    },
    "e814b1b34398e27f40bf4eb877343b582c8103e4": {
        "vendor/harbor/src/harbor_config/models/task/config.py": (
            "a722fece73b44ded382432ab8084073bd11d934f3a25d07d498dffbce18d1bcf"
        ),
        "vendor/harbor/src/harbor/environments/daytona.py": (
            "44f6d1e71839e25b97f5bda2e408878c38d521ad000e829f17aa7b34d448966a"
        ),
        "vendor/harbor/src/harbor/environments/factory.py": (
            "cfea2f511cde24b0cc7a9229c9dcb38cdac771da90b2454d7709674567bd70ae"
        ),
        "lib/shellbox/src/shellbox/backends/daytona/machine.py": (
            "7aa81464064303211462543b1e9fdcb154de92c03a4f374b2c42c041ddfc3df8"
        ),
        "vendor/harbor/uv.lock": "60a43c2538c90377df717d432c08fcb999a63650442918ed6e3ba52a89b2104c",
    },
}

NATIVE_DAYTONA_NETWORK_CODE_PATHS = {
    "vendor/harbor/src/harbor_config/models/task/config.py": "EnvironmentConfig.allow_internet defaults to True",
    "vendor/harbor/src/harbor/environments/daytona.py": (
        "machine_spec maps allow_internet to NetworkPolicy; start passes the spec to factory.create"
    ),
    "vendor/harbor/src/harbor/environments/factory.py": (
        "create_environment_from_config forwards the actual TrialResult environment kwargs"
    ),
    "lib/shellbox/src/shellbox/backends/daytona/machine.py": (
        "_sandbox_parameters maps spec.network or explicit DaytonaNetworkPolicy.parameters to network_block_all"
    ),
    "vendor/harbor/uv.lock": "pins the Daytona SDK used to serialize the sandbox creation parameters",
}


class ArchivedNetworkText(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reference: ContentEvidenceReference
    text: str

    @model_validator(mode="after")
    def _exact_bytes(self):
        if hashlib.sha256(self.text.encode()).hexdigest() != self.reference.sha256:
            raise ValueError("Archived network input SHA-256 differs from its bytes")
        return self


class NativeNetworkReconstructionInputs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    source_dataset_id: Literal["Lego-X/Terminal-Lego-15k"]
    source_dataset_revision: Literal["9c197f1c2e87b64cc316b1a5bfcef57b584929f0"]
    archive: ContentEvidenceReference
    execution_root: str
    task_toml: ArchivedNetworkText
    source_runtime: ArchivedNetworkText
    trial_result: ContentEvidenceReference
    trial_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_config: dict[str, Any]
    environment_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_revision: str
    implementation: dict[str, ContentEvidenceReference]
    code_paths: dict[str, str]
    execution_provenance: ContentEvidenceReference

    @model_validator(mode="after")
    def _pinned_inputs(self):
        expected = NATIVE_DAYTONA_NETWORK_IMPLEMENTATIONS.get(self.implementation_revision)
        if (
            expected is None
            or self.implementation.keys() != expected.keys()
            or self.code_paths != NATIVE_DAYTONA_NETWORK_CODE_PATHS
        ):
            raise ValueError("Native network reconstruction requires a qualified pinned implementation")
        for path, digest in expected.items():
            reference = self.implementation[path]
            uri = f"https://raw.githubusercontent.com/marin-community/marin/{self.implementation_revision}/{path}"
            if (reference.uri, reference.sha256) != (uri, digest):
                raise ValueError("Native network implementation identity changed: " + path)
        if canonical_sha256(self.environment_config) != self.environment_config_sha256:
            raise ValueError("Archived trial environment configuration SHA-256 changed")
        reconstructed_native_network_policy(self)
        return self


def reconstructed_native_network_policy(inputs: NativeNetworkReconstructionInputs) -> TaskNetworkPolicyReceipt:
    """Project only the archived native configuration through the cited pinned code path."""
    config = inputs.environment_config
    if config.get("type") != "daytona":
        raise ValueError("Native network reconstruction only supports archived Daytona trials")
    environment = tomllib.loads(inputs.task_toml.text).get("environment", {})
    # Both qualified pinned EnvironmentConfig versions declare this exact default.
    allow = environment.get("allow_internet", True)
    if not isinstance(allow, bool):
        raise ValueError("Native allow_internet must be a boolean")
    network = NetworkPolicy.ALLOW if allow else NetworkPolicy.DENY
    kwargs = config.get("kwargs") or {}
    override = kwargs.get("network_policy")
    if override is not None:
        if set(override) - {"mode", "value"} or override.get("value") is not None:
            raise ValueError("Unsupported archived Daytona network override")
        mode = override.get("mode")
        if mode not in {"unrestricted", "block_all"}:
            raise ValueError("An archived allow-list cannot prove an unrestricted native policy")
        network = NetworkPolicy.DENY if mode == "block_all" else NetworkPolicy.ALLOW
    binding = TaskSpecRuntimeBinding.model_validate_json(inputs.source_runtime.text)
    return TaskNetworkPolicyReceipt(
        source_runtime_sha256=binding.runtime_sha256,
        actor_required=True,
        actor_declared=binding.network("task_machine"),
        verifier_declared=binding.network("verifier_machine"),
        actor_effective=network,
        verifier_effective=network,
    )


class NativeNetworkCanaryComparison(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    inputs: NativeNetworkReconstructionInputs
    physical: ArchivedNetworkText

    @model_validator(mode="after")
    def _physical_equals_reconstructed(self):
        expected_uri = self.inputs.trial_result.uri.rsplit("/", 1)[0] + "/network-policy.json"
        if self.physical.reference.uri != expected_uri:
            raise ValueError("Physical policy belongs to another native canary trial")
        physical = TaskNetworkPolicyReceipt.model_validate_json(self.physical.text)
        if physical != reconstructed_native_network_policy(self.inputs):
            raise ValueError("Physical native policy differs from its reconstruction; fresh controls are required")
        return self


class NativeNetworkCanaryResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    recording_revision: Literal["e814b1b34398e27f40bf4eb877343b582c8103e4"]
    reconstruction_revision: Literal["3ba52c03e9599b98b38d4851885383043e7568cd"]
    rows: tuple[NativeNetworkCanaryComparison, ...] = Field(min_length=20)

    @model_validator(mode="after")
    def _unique_inputs(self):
        ids = {row.inputs.task_id for row in self.rows}
        archives = {row.inputs.archive.uri for row in self.rows}
        trials = {row.inputs.trial_result.uri for row in self.rows}
        if (
            len(ids) != len(self.rows)
            or len(archives) != len(self.rows)
            or len(trials) != len(self.rows)
            or any(row.inputs.implementation_revision != self.reconstruction_revision for row in self.rows)
        ):
            raise ValueError("Native canary comparisons require distinct tasks under the pinned reconstruction")
        return self


class NativeNetworkCanaryValidation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence: ContentEvidenceReference
    recording_revision: Literal["e814b1b34398e27f40bf4eb877343b582c8103e4"]
    reconstruction_revision: Literal["3ba52c03e9599b98b38d4851885383043e7568cd"]
    compared_task_ids: tuple[str, ...] = Field(min_length=20)
    mismatched_task_ids: tuple[str, ...] = Field(max_length=0)

    @model_validator(mode="after")
    def _complete_canary(self):
        if len(set(self.compared_task_ids)) != len(self.compared_task_ids):
            raise ValueError("Native network canary must compare distinct physical trials")
        return self


def verify_native_network_canary(validation: NativeNetworkCanaryValidation, payload: bytes) -> NativeNetworkCanaryResult:
    """Recompute every physical comparison from the immutable canary proof."""
    if hashlib.sha256(payload).hexdigest() != validation.evidence.sha256:
        raise ValueError("Native network canary evidence SHA-256 changed")
    observed = NativeNetworkCanaryResult.model_validate_json(payload)
    if (
        tuple(row.inputs.task_id for row in observed.rows) != validation.compared_task_ids
        or observed.recording_revision != validation.recording_revision
        or observed.reconstruction_revision != validation.reconstruction_revision
    ):
        raise ValueError("Native network canary scope differs from its validation binding")
    return observed


class ReconstructedNativeNetworkPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["network_policy_reconstructed_native"] = "network_policy_reconstructed_native"
    inputs: NativeNetworkReconstructionInputs
    canary: NativeNetworkCanaryValidation
    authority: ContentEvidenceReference

    def require_trial(self, task_id: str, result: dict[str, Any], trial_path: str) -> None:
        if (
            self.inputs.task_id != task_id
            or self.inputs.trial_result_sha256 != canonical_sha256(result)
            or self.inputs.environment_config != result.get("config", {}).get("environment")
            or self.inputs.implementation_revision != self.canary.reconstruction_revision
        ):
            raise ValueError("Native policy reconstruction belongs to another archived task/trial")
        root = PurePosixPath(self.inputs.execution_root)
        package = PurePosixPath(result.get("config", {}).get("task", {}).get("path", ""))
        paths = (
            (self.inputs.task_toml.reference, package / "task.toml"),
            (self.inputs.source_runtime.reference, package / "source-runtime.json"),
            (self.inputs.trial_result, PurePosixPath(trial_path)),
        )
        for reference, path in paths:
            if not root.is_absolute() or ".." in path.parts or not path.is_relative_to(root):
                raise ValueError("Native reconstruction archive path escapes its execution root")
            expected = self.inputs.archive.uri + "#" + str(path.relative_to(root))
            if reference.uri != expected:
                raise ValueError("Native reconstruction input belongs to another archive member")

    @property
    def policy(self) -> TaskNetworkPolicyReceipt:
        return reconstructed_native_network_policy(self.inputs)
