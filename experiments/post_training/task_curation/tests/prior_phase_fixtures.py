# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserved phase archives for collector boundary regressions."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

import zstandard
from finestore.rl.curation import ContentEvidenceReference
from taskcompendium.pipeline.oracle_filter import PriorPhaseFailureReceipt


def preserved_phase_failure(tmp_path, original, trial, *, archive_root, changed=None):
    member = original["archive_member"]
    config = json.loads(json.dumps(original["config"]))
    if changed == "another_trial":
        config["task"]["path"] = "/another-task"
    exception = original["exception_text"].encode()
    if changed == "provider_failure":
        exception = b"RuntimeError: Machine is closed\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in (
            (member, exception),
            (str(Path(member).with_name("config.json")), json.dumps(config).encode()),
        ):
            item = tarfile.TarInfo(name)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    archive_path = tmp_path / "preserved.tar.zst"
    payload = zstandard.ZstdCompressor().compress(buffer.getvalue())
    archive_path.write_bytes(payload)
    return PriorPhaseFailureReceipt(
        phase="agent",
        timeout_sec=trial.execution_requirements.agent_timeout_sec,
        result_sha256=trial.result_sha256,
        archive=ContentEvidenceReference(uri=str(archive_path), sha256=hashlib.sha256(payload).hexdigest()),
        exception=ContentEvidenceReference(uri=f"{archive_path}#{member}", sha256=hashlib.sha256(exception).hexdigest()),
        archive_member=member,
        archive_root=archive_root,
    )
