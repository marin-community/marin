# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A diagnosed bootstrap fault can be retried without replacing ordinary source grades."""

import hashlib
import io
import json
import tarfile

import pytest
from finestore.rl.curation import ContentEvidenceReference
from taskcompendium.identity import task_spec_sha256
from taskcompendium.models import ResourceGroups, TaskSpec
from taskcompendium.pipeline.oracle_filter import (
    InfrastructureRetryApproval,
    LocalOwnershipBootstrapFailure,
    ResourceCalibrationReceipt,
    require_infrastructure_retry_parent,
)
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.oracle_bootstrap_failure import PROJECT_ROOT, require_bootstrap_failure
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task
from lib.taskcompendium.tests.oracle_filter_fixtures import oracle_receipt


@pytest.fixture
def bootstrap_case(tmp_path):
    original = json.loads(prebuilt_task("ownership-task")["lowered_task_spec"])["task"]
    oracle = b"#!/bin/bash\nset -e\nbash /setup_files/setup.sh\nprintf payload\n"
    setup = b"#!/bin/bash\nset -euo pipefail\nchown -R 1000:1000 /task_file\n"
    task = TaskSpec.model_validate(original).model_copy(
        update={
            "resources": ResourceGroups(
                worker=(inline_resource("setup_files/setup.sh", setup),),
                oracle=(inline_resource("solution/solve.sh", oracle),),
            )
        }
    )
    before = oracle_receipt([task], failed=frozenset({task.id})).tasks[task.id].calibration.model_dump(mode="json")
    failure = "chown: changing ownership of '/task_file/output': Invalid argument\n"
    before["attempts"][0].update(oracle_exit_code=1, failure_log=failure)
    parent = ResourceCalibrationReceipt.model_validate(before)
    trial = parent.attempts[-1]
    archive = tmp_path / "original.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name, data in {
            "run/result.json": json.dumps(trial.result).encode(),
            "run/agent/exit-code.txt": b"1",
            "run/agent/oracle.txt": failure.encode(),
        }.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))

    def reference(path):
        return ContentEvidenceReference(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    reproduction = tmp_path / "reproduction.json"
    code = {
        "candidate/shellbox/backends/local/"
        + name: (
            hashlib.sha256((PROJECT_ROOT / "lib/shellbox/src/shellbox/backends/local" / name).read_bytes()).hexdigest()
        )
        for name in ("launch.py", "machine.py")
    }
    reproduction.write_text(
        json.dumps(
            {
                "passed": True,
                "inputs": {"files": code},
                "results": [
                    {"variant": "baseline", "image_exit": 1, "image_stderr": failure},
                    {"variant": "candidate", "image_exit": 0, "image_stderr": ""},
                ],
            }
        )
    )
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "classification": "our_infrastructure",
                "approved_parents": {task.id: trial.result_sha256},
                "source_contracts_changed": False,
            }
        )
    )
    proof = LocalOwnershipBootstrapFailure(
        result_sha256=trial.result_sha256,
        source_task_sha256=task_spec_sha256(task),
        oracle_script_sha256=hashlib.sha256(oracle).hexdigest(),
        setup_script_sha256=hashlib.sha256(setup).hexdigest(),
        target_uid=1000,
        target_gid=1000,
        archive=reference(archive),
        oracle_log=ContentEvidenceReference(
            uri=str(archive) + "#run/agent/oracle.txt", sha256=hashlib.sha256(failure.encode()).hexdigest()
        ),
        result_member="run/result.json",
        oracle_exit_member="run/agent/exit-code.txt",
        reproduction=reference(reproduction),
        decision=reference(decision),
    )
    approval = InfrastructureRetryApproval(
        classification="our_infrastructure",
        parent_result_sha256=trial.result_sha256,
        failure_signature="LOCAL image UID 1000 unmapped before oracle payload",
        decision_ref=str(decision),
        bootstrap_failure=proof,
    )
    return task, parent, approval, archive


def test_exact_bootstrap_retry_preserves_the_zero_result_and_requires_the_full_binding(bootstrap_case):
    task, parent, approval, _ = bootstrap_case
    unchanged = parent.model_dump_json()
    with pytest.raises(ValueError, match="cannot replace a scored source result"):
        require_infrastructure_retry_parent(parent, approval.model_copy(update={"bootstrap_failure": None}))
    require_infrastructure_retry_parent(parent, approval)
    require_bootstrap_failure(task, parent.attempts[-1], approval)
    restored = InfrastructureRetryApproval.model_validate_json(approval.model_dump_json())
    require_bootstrap_failure(task, parent.attempts[-1], restored)
    assert parent.model_dump_json() == unchanged
    assert parent.attempts[-1].result["verifier_result"]["rewards"] == {"reward": 0}


@pytest.mark.parametrize("change", ["ordinary-zero", "source-error"])
def test_a_bootstrap_approval_cannot_replace_an_ordinary_source_result(bootstrap_case, change):
    _, parent, approval, _ = bootstrap_case
    data = parent.model_dump(mode="json")
    data["attempts"][-1].update(
        {"oracle_exit_code": 0} if change == "ordinary-zero" else {"failure_log": "AssertionError: wrong answer\n"}
    )
    with pytest.raises(ValueError, match="pre-payload backend failure"):
        require_infrastructure_retry_parent(ResourceCalibrationReceipt.model_validate(data), approval)


def test_changed_source_or_archive_cannot_reuse_a_bootstrap_approval(bootstrap_case):
    task, parent, approval, archive = bootstrap_case
    changed = task.model_copy(
        update={"resources": ResourceGroups(oracle=(inline_resource("solution/solve.sh", b"different\n"),))}
    )
    with pytest.raises(ValueError, match="another frozen source task"):
        require_bootstrap_failure(changed, parent.attempts[-1], approval)
    archive.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="archive bytes changed"):
        require_bootstrap_failure(task, parent.attempts[-1], approval)
