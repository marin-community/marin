# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full raw audits preserve unsupported rows without claiming conversion."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from taskcompendium.importers.tasktrove.convert import MAX_ARCHIVE_MEMBERS
from taskcompendium.pipeline.contract_audit import audit_tasktrove_contracts
from taskcompendium.pipeline.models import HFSource


def archive(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as output:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
    return stream.getvalue()


def audit(tmp_path: Path, blobs: list[bytes]) -> tuple[dict, list[dict]]:
    source = tmp_path / "raw" / "fixture" / "tasks.parquet"
    source.parent.mkdir(parents=True)
    # Repeated upstream task names must not collapse distinct source rows.
    pq.write_table(
        pa.Table.from_pylist([{"path": "same-task", "task_binary": blob} for blob in blobs]), source, row_group_size=2
    )
    output = tmp_path / "audited"
    result = audit_tasktrove_contracts(
        str(source.parent.parent),
        str(output),
        HFSource("fixture/tasks", "pinned-revision", "fixture", "train"),
        max_workers=1,
    )
    rows = pq.read_table(output / "audit").to_pylist()
    return result, sorted(rows, key=lambda row: row["source_row"])


def test_contract_audit_conserves_duplicates_and_malformed_rows(tmp_path):
    first = archive({"instruction.md": b"first", "tests/test.sh": b"exit 0", "setup_files/a:b": b"public"})
    second = archive({"instruction.md": b"second", "tests/test.sh": b"exit 0", "solution/solve.sh": b"real golden"})
    blobs = [first, second, b"not an archive", first]
    manifest, rows = audit(tmp_path, blobs)

    assert manifest["input_rows"] == manifest["audit_rows"] == 4
    assert manifest["normalized_rows"] == manifest["reviewed_rows"] == 0
    assert manifest["decoded_rows"] == 3
    assert manifest["decode_unavailable_rows"] == 1
    assert manifest["complete_row_accounting"] and not manifest["complete_contract_coverage"]
    assert manifest["dispositions"] == {"defer": 4}
    assert len({row["task_id"] for row in rows}) == 4
    assert all(row["task_json"] is None and row["filter_status"] == "defer" for row in rows)
    assert all(row["normalization_kind"] == "unsupported" for row in rows)
    assert rows[2]["normalization_reason"] == "archive_decode_unavailable"
    assert all(row["source_revision"] == "pinned-revision" for row in rows)
    for index, (row, blob) in enumerate(zip(rows, blobs, strict=True)):
        data = json.loads(row["raw_json"])["data"]
        assert data["archive_sha256"] == hashlib.sha256(blob).hexdigest()
        locator = data["archive_locator"]
        recovered = pq.read_table(locator["parquet"]).to_pylist()[locator["row"]]["task_binary"]
        assert recovered == blob
        assert locator["row"] == index
    data = json.loads(rows[0]["raw_json"])["data"]
    assert (
        next(file for file in data["files"] if file["path"] == "setup_files/a:b")["sha256"]
        == hashlib.sha256(b"public").hexdigest()
    )
    assert json.loads(rows[1]["raw_json"])["data"]["solution_files"] == 1
    assert list(json.loads(Path(manifest["contract_frequencies"]).read_text()).values()) == [3]


def test_contract_audit_over_limit_keeps_locator_without_claiming_complete_decode(tmp_path):
    blob = archive({f"file-{i}": b"" for i in range(MAX_ARCHIVE_MEMBERS + 1)})
    manifest, rows = audit(tmp_path, [blob])

    assert manifest["audit_rows"] == 1
    assert manifest["decoded_rows"] == 0
    assert manifest["decode_unavailable_rows"] == 1
    assert not manifest["complete_contract_coverage"]
    data = json.loads(rows[0]["raw_json"])["data"]
    assert data["archive_sha256"] == hashlib.sha256(blob).hexdigest()
    assert data["files"] == []
    assert data["contract_sha256"] is None
    assert rows[0]["normalization_reason"] == "archive_decode_unavailable"


def test_contract_audit_distinguishes_original_grader_scripts(tmp_path):
    blobs = [archive({"instruction.md": b"prompt", "tests/test.sh": script}) for script in [b"exit 0", b"exit 1"]]
    manifest, rows = audit(tmp_path, blobs)

    assert manifest["complete_contract_coverage"]
    assert manifest["contract_count"] == 2
    assert sorted(json.loads(Path(manifest["contract_frequencies"]).read_text()).values()) == [1, 1]
    assert all(row["normalization_reason"] == "unbound_source_contract" for row in rows)


def test_contract_audit_corrupt_compressed_payload_is_accounted(tmp_path):
    blob = bytearray(archive({"instruction.md": b"x" * 100000}))
    # Damage compressed file data after the archive header, not just its trailer.
    middle = len(blob) // 2
    blob[middle : middle + 5] = b"\xff" * 5
    manifest, rows = audit(tmp_path, [bytes(blob)])

    assert manifest["audit_rows"] == manifest["decode_unavailable_rows"] == 1
    assert not manifest["complete_contract_coverage"]
    assert rows[0]["task_json"] is None
    assert rows[0]["filter_status"] == "defer"
