# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Physical nop archives emitted by the offline Harbor provider fixture."""

import hashlib
import io
import json
import tarfile
from copy import deepcopy

import zstandard
from finestore.rl.curation import ContentEvidenceReference
from taskcompendium.pipeline.oracle_filter import NegativeControlApplicability, OracleExclusionReason, OracleTaskReceipt

from experiments.post_training.task_curation.negative_controls import negative_control_from_archive
from experiments.post_training.task_curation.oracle_filter import filter_cohort


def filter_with_negative_controls(cohort, task_map, jobs, output, *args, **kwargs):
    """Supply explicit offline-provider nop evidence to existing source-result cases."""
    pending = output.parent / "nop-fixture-input/tasks.parquet"
    receipt = filter_cohort(cohort, task_map, jobs, pending, *args, **kwargs)
    records = []
    for task_id, entry in receipt.tasks.items():
        if not isinstance(entry, OracleTaskReceipt) or (
            entry.negative_control_applicability is not NegativeControlApplicability.WORKSPACE
            or entry.exclusion_reason is not OracleExclusionReason.NEGATIVE_CONTROL_PENDING
        ):
            continue
        source = entry.calibration.attempts[-1]
        result = deepcopy(source.result)
        result.update(
            id=f"nop-{result['id']}",
            trial_name="nop",
            trial_uri=f"file:///fixture/{task_id}/trials/nop",
            agent_info={"name": "nop"},
            exception_info=None,
            completion={"trajectory_present": False},
            finished_at="2026-11-14T00:00:00Z",
            verifier_result={"rewards": {"reward": 0}, "exit_code": 0},
        )
        result["config"]["agent"] = {"name": "nop"}
        result["config"]["trial_name"] = "nop"
        root = f"{task_id}"
        result_member = f"{root}/trials/nop/result.json"
        result_bytes = json.dumps(result).encode()
        verdict = {"status": "scored", "reward": 0, "detail": {"total": 1, "passed": 0}}
        setup_script = b"true\n"
        setup = {
            "event": "agent_start",
            "trial_id": "nop",
            "source_native_checksum": entry.harbor_task_checksum,
            "setup_sha256": hashlib.sha256(setup_script).hexdigest(),
            "setup_exit_code": 0,
            "source_agent_timeout_sec": source.execution_requirements.agent_timeout_sec,
            "setup_from_original_public_instruction": True,
            "source_commands_outside_trial": 0,
        }
        control = {
            "task_sha256": entry.task_sha256,
            "runtime_sha256": entry.runtime_sha256,
            "task_checksum": entry.harbor_task_checksum,
            "control": "original-public-setup-then-nop",
            "native_package_unchanged": True,
            "setup": setup,
            "physical_verdict": verdict,
            "verifier_exit_code": 0,
            "trial_result_path": f"/fixture/{result_member}",
            "trial_result_file_sha256": hashlib.sha256(result_bytes).hexdigest(),
            "trial_result": result,
            "source_commands_outside_trial": 0,
        }
        members = {
            result_member: result_bytes,
            f"{root}/trials/nop/verifier/verdict.json": json.dumps(verdict).encode(),
            f"{root}/setup-receipt.json": json.dumps(setup).encode(),
            f"{root}/original-public-setup.sh": setup_script,
            f"{root}/negative-control.json": json.dumps(control).encode(),
        }
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for name, payload in members.items():
                item = tarfile.TarInfo(name)
                item.size = len(payload)
                archive.addfile(item, io.BytesIO(payload))
        payload = zstandard.ZstdCompressor().compress(buffer.getvalue())
        archive_path = pending.parent / f"{task_id}.tar.zst"
        archive_path.write_bytes(payload)
        proof = negative_control_from_archive(
            ContentEvidenceReference(uri=str(archive_path), sha256=hashlib.sha256(payload).hexdigest()),
            f"{root}/negative-control.json",
            source_task_sha256=entry.source_task_sha256,
            source_task_checksum=entry.source_task_checksum,
            execution_requirements=source.execution_requirements,
        )
        records.append({"task_id": task_id, "receipt": proof.model_dump(mode="json")})
    controls = pending.parent / "controls.jsonl"
    controls.write_text("".join(json.dumps(record) + "\n" for record in records))
    return filter_cohort(cohort, task_map, jobs, output, *args, negative_controls=controls, **kwargs)
