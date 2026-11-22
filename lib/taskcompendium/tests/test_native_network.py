# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from copy import deepcopy

import pytest
from finestore.rl.curation import ContentEvidenceReference

from taskcompendium.harbor.native_network import (
    NATIVE_DAYTONA_NETWORK_CODE_PATHS,
    NATIVE_DAYTONA_NETWORK_IMPLEMENTATIONS,
    ArchivedNetworkText,
    NativeNetworkCanaryResult,
    NativeNetworkCanaryValidation,
    NativeNetworkReconstructionInputs,
    ReconstructedNativeNetworkPolicy,
    reconstructed_native_network_policy,
    verify_native_network_canary,
)
from taskcompendium.harbor.runtime import TaskSpecRuntimeBinding, network_policy_qualified
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.pipeline.oracle_filter import OracleTrialReceipt, trial_network_qualified

from .test_source_processing import apple_rows, convert_script_graded

REVISION = "3ba52c03e9599b98b38d4851885383043e7568cd"
ROOT = "/iris/outputs/controls/attempt"
ARCHIVE = "s3://fixtures/control.tar.gz"


def text_evidence(name, text):
    return ArchivedNetworkText(
        reference=ContentEvidenceReference(uri=ARCHIVE + "#" + name, sha256=hashlib.sha256(text.encode()).hexdigest()),
        text=text,
    )


def case(toml="[environment]\n", override=None, deadlines=None):
    result = {
        "config": {
            "task": {"path": ROOT + "/packet"},
            "environment": {"type": "daytona", "kwargs": {"provisioning": "declarative"}},
        }
    }
    if deadlines is not None:
        result["config"]["resolved_timeouts"] = deadlines
    if override is not None:
        result["config"]["environment"]["kwargs"]["network_policy"] = override
    runtime = {"task_machine": {"network": "allow"}, "verifier_machine": {"network": "allow"}}
    binding = TaskSpecRuntimeBinding(task_sha256="a" * 64, runtime_sha256=canonical_sha256(runtime), runtime=runtime)
    inputs = NativeNetworkReconstructionInputs(
        task_id="task",
        source_dataset_id="Lego-X/Terminal-Lego-15k",
        source_dataset_revision="9c197f1c2e87b64cc316b1a5bfcef57b584929f0",
        archive=ContentEvidenceReference(uri=ARCHIVE, sha256="c" * 64),
        execution_root=ROOT,
        task_toml=text_evidence("packet/task.toml", toml),
        source_runtime=text_evidence("packet/source-runtime.json", binding.model_dump_json()),
        trial_result=ContentEvidenceReference(uri=ARCHIVE + "#jobs/oracle/result.json", sha256="b" * 64),
        trial_result_sha256=canonical_sha256(result),
        environment_config=result["config"]["environment"],
        environment_config_sha256=canonical_sha256(result["config"]["environment"]),
        implementation_revision=REVISION,
        implementation={
            path: ContentEvidenceReference(
                uri=f"https://raw.githubusercontent.com/marin-community/marin/{REVISION}/{path}", sha256=digest
            )
            for path, digest in NATIVE_DAYTONA_NETWORK_IMPLEMENTATIONS[REVISION].items()
        },
        code_paths=NATIVE_DAYTONA_NETWORK_CODE_PATHS,
        execution_provenance=ContentEvidenceReference(uri="s3://fixtures/start.json", sha256="d" * 64),
    )
    validation = NativeNetworkCanaryValidation(
        evidence=ContentEvidenceReference(uri="s3://fixtures/20-physical-policies.json", sha256="e" * 64),
        recording_revision="e814b1b34398e27f40bf4eb877343b582c8103e4",
        reconstruction_revisions=(REVISION,),
        compared_task_ids=tuple(str(i) for i in range(20)),
        mismatched_task_ids=(),
    )
    proof = ReconstructedNativeNetworkPolicy(
        inputs=inputs,
        canary=validation,
        authority=ContentEvidenceReference(uri="s3://fixtures/ruling.json", sha256="f" * 64),
    )
    return binding, result, proof


@pytest.mark.parametrize(
    "toml,override,expected",
    [
        ("[environment]\n", None, "allow"),
        ("[environment]\nallow_internet=false\n", None, "deny"),
        ("[environment]\nallow_internet=false\n", {"mode": "unrestricted"}, "allow"),
        ("[environment]\nallow_internet=true\n", {"mode": "block_all"}, "deny"),
    ],
)
def test_native_projection_uses_pinned_default_and_actual_provider_override(toml, override, expected):
    binding, result, proof = case(toml, override)
    policy = reconstructed_native_network_policy(proof.inputs)
    assert policy.actor_effective.value == policy.verifier_effective.value == expected
    assert network_policy_qualified(binding, policy) == (expected == "allow")
    proof.require_trial("task", result, ROOT + "/jobs/oracle/result.json")


@pytest.mark.parametrize(
    "change",
    ["toml_bytes", "environment_digest", "implementation_pin", "archive_member", "trial_environment", "task_id"],
)
def test_reconstruction_rejects_changed_archived_inputs(change):
    _, result, proof = case()
    value = proof.model_dump(mode="json")
    if change == "toml_bytes":
        value["inputs"]["task_toml"]["text"] += "allow_internet=false\n"
    elif change == "environment_digest":
        value["inputs"]["environment_config"]["type"] = "local"
    elif change == "implementation_pin":
        value["inputs"]["implementation_revision"] = "0" * 40
    elif change == "archive_member":
        value["inputs"]["task_toml"]["reference"]["uri"] = ARCHIVE + "#another/task.toml"
    elif change == "trial_environment":
        result = deepcopy(result)
        result["config"]["environment"]["kwargs"]["network_policy"] = {"mode": "block_all"}
    elif change == "task_id":
        value["inputs"]["task_id"] = "another"
    with pytest.raises(ValueError):
        parsed = ReconstructedNativeNetworkPolicy.model_validate(value)
        parsed.require_trial("task", result, ROOT + "/jobs/oracle/result.json")


def test_a_physical_canary_mismatch_or_missing_row_stops_reconstruction():
    _, _, proof = case()
    for change in ["mismatch", "incomplete"]:
        value = proof.model_dump(mode="json")
        if change == "mismatch":
            value["canary"]["mismatched_task_ids"] = ["7"]
        else:
            value["canary"]["compared_task_ids"].pop()
        with pytest.raises(ValueError):
            ReconstructedNativeNetworkPolicy.model_validate(value)


def test_native_reconstruction_qualifies_a_missing_receipt_without_rewriting_the_trial():
    task = convert_script_graded(
        RawRow(
            "task",
            Source(
                dataset="Lego-X/Terminal-Lego-15k",
                revision="9c197f1c2e87b64cc316b1a5bfcef57b584929f0",
                row="0",
                importer_revision="1",
            ),
            apple_rows(1)[0],
        ),
        ConversionContext({}, None),
    )
    task = task.model_copy(update={"output_paths": ("answer.txt",)})
    binding, result, proof = case(
        deadlines={
            "agent": task.execution_requirements.agent_timeout_sec,
            "verifier": task.execution_requirements.verifier_timeout_sec,
        }
    )
    trial = OracleTrialReceipt(
        path=ROOT + "/jobs/oracle/result.json",
        result=result,
        verdict=None,
        oracle_exit_code=0,
        result_sha256=canonical_sha256(result),
        execution_requirements=task.execution_requirements,
    )
    frozen = trial.model_dump_json()
    assert not trial_network_qualified(task, binding.runtime, trial)
    assert trial_network_qualified(task, binding.runtime, trial, proof)
    assert trial.model_dump_json() == frozen and trial.network_policy is None
    observed = trial.model_copy(update={"network_policy": proof.policy})
    with pytest.raises(ValueError, match="physical network receipt"):
        trial_network_qualified(task, binding.runtime, observed, proof)


def test_twenty_physical_comparisons_are_recomputed_before_accepting_a_canary_seal():
    rows = []
    for index in range(20):
        _, _, proof = case()
        inputs = proof.inputs.model_dump(mode="json")
        inputs["task_id"] = str(index)
        archive = f"s3://fixtures/control-{index}.tar.gz"
        inputs["archive"]["uri"] = archive
        for name in ("task_toml", "source_runtime"):
            inputs[name]["reference"]["uri"] = inputs[name]["reference"]["uri"].replace(ARCHIVE, archive)
        inputs["trial_result"]["uri"] = inputs["trial_result"]["uri"].replace(ARCHIVE, archive)
        encoded = proof.policy.model_dump_json()
        rows.append(
            {
                "inputs": inputs,
                "physical": {
                    "reference": {
                        "uri": archive + "#jobs/oracle/network-policy.json",
                        "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                    },
                    "text": encoded,
                },
            }
        )
    document = {
        "recording_revision": "e814b1b34398e27f40bf4eb877343b582c8103e4",
        "reconstruction_revisions": [REVISION],
        "rows": rows,
    }
    payload = json.dumps(document).encode()
    validation = NativeNetworkCanaryValidation(
        evidence=ContentEvidenceReference(uri="s3://fixtures/canary.json", sha256=hashlib.sha256(payload).hexdigest()),
        recording_revision=document["recording_revision"],
        reconstruction_revisions=(REVISION,),
        compared_task_ids=tuple(str(index) for index in range(20)),
        mismatched_task_ids=(),
    )
    assert len(verify_native_network_canary(validation, payload).rows) == 20
    with pytest.raises(ValueError, match="SHA-256"):
        verify_native_network_canary(validation, payload + b" ")
    changed = deepcopy(document)
    policy = json.loads(changed["rows"][7]["physical"]["text"])
    policy["actor_effective"] = "deny"
    encoded = json.dumps(policy)
    changed["rows"][7]["physical"]["text"] = encoded
    changed["rows"][7]["physical"]["reference"]["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    with pytest.raises(ValueError, match="fresh controls"):
        NativeNetworkCanaryResult.model_validate(changed)
    uncovered = deepcopy(document)
    uncovered["reconstruction_revisions"].append("99f62b7e8c50ce15000e3ebed5205f53e1e6016f")
    with pytest.raises(ValueError, match="every reconstructed implementation"):
        NativeNetworkCanaryResult.model_validate(uncovered)
    changed = deepcopy(document)
    changed["rows"][7] = changed["rows"][6]
    with pytest.raises(ValueError, match="distinct tasks"):
        NativeNetworkCanaryResult.model_validate(changed)
