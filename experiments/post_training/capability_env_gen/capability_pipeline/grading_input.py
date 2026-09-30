"""Byte-exact fingerprint of the evidence delivered to a private grader."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _identity_bytes(value: Any) -> bytes:
    """Use caller-supplied serialized bytes unchanged at a delivery boundary."""
    return value if isinstance(value, bytes) else _canonical(value)


def grading_input_fingerprint(
    specification: Any,
    protocol: Any,
    response: str | None,
    workspace: Path,
    transcript: Any,
    step_index: int = 0,
    payload: bytes | None = None,
) -> dict[str, Any]:
    """Hash exact candidate evidence, rejecting links and non-regular entries."""
    if response is not None and not isinstance(response, str):
        raise TypeError("grader response must be text or null")
    if type(step_index) is not int or step_index < 0:
        raise ValueError("grader step index must be a nonnegative integer")
    if payload is not None and not isinstance(payload, bytes):
        raise TypeError("grader payload must be bytes when supplied")
    if workspace.is_symlink() or not workspace.is_dir():
        raise ValueError("grader workspace must be a real directory")
    entries: list[dict[str, Any]] = []
    for path in sorted(workspace.rglob("*")):
        relative = path.relative_to(workspace).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("grader workspace contains a symlink")
        if stat.S_ISDIR(mode):
            entries.append(
                {"path": relative, "kind": "directory", "mode": mode & 0o777}
            )
        elif stat.S_ISREG(mode):
            content = path.read_bytes()
            entries.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode": mode & 0o777,
                    "size": len(content),
                    "sha256": _sha(content),
                }
            )
        else:
            raise ValueError("grader workspace contains a non-regular entry")
    response_record = {
        "present": response is not None,
        "sha256": _sha(response.encode("utf-8")) if response is not None else None,
    }
    submission = {"response": response_record, "workspace": entries}
    specification_sha = _sha(_identity_bytes(specification))
    protocol_sha = _sha(_identity_bytes(protocol))
    transcript_sha = _sha(_canonical(transcript))
    payload_sha = _sha(
        payload
        if payload is not None
        else _canonical(
            {
                "specification_sha256": specification_sha,
                "protocol_sha256": protocol_sha,
                "step_index": step_index,
                "response": response_record,
                "transcript_sha256": transcript_sha,
            }
        )
    )
    return {
        "schema_version": "capability-grading-input-fingerprint-v1",
        "submission_sha256": _sha(_canonical(submission)),
        "grading_input_sha256": _sha(
            _canonical(
                {
                    "submission": submission,
                    "specification_sha256": specification_sha,
                    "protocol_sha256": protocol_sha,
                    "transcript_sha256": transcript_sha,
                    "step_index": step_index,
                    "payload_sha256": payload_sha,
                }
            )
        ),
        "specification_sha256": specification_sha,
        "protocol_sha256": protocol_sha,
        "transcript_sha256": transcript_sha,
        "payload_sha256": payload_sha,
        "step_index": step_index,
        "workspace_file_count": sum(entry["kind"] == "file" for entry in entries),
    }
