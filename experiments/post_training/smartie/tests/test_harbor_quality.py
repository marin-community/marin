# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tarfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.post_training.smartie.harbor_quality import prepare


def archive(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as output:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            output.addfile(member, io.BytesIO(content))
    return buffer.getvalue()


@pytest.fixture
def candidates(tmp_path):
    rows = []
    results = tmp_path / "results"
    for name, reward in (("pass", 1.0), ("fail", 0.0)):
        files = {"instruction.md": b"Print one", "tests/test.sh": b"check-output", "setup_files/seed.sh": b"seed-input"}
        solution = {"solution/solve.sh": b"echo 1"}
        staged = tmp_path / "tasks" / name
        for path, content in (files | solution).items():
            target = staged / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        trial = results / name
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "config": {
                        "task": {"path": str(staged)},
                        "agent": {"name": "oracle"},
                        "environment": {"type": "daytona", "kwargs": {"snapshot_template_name": "shared"}},
                    },
                    "verifier_result": {"rewards": {"reward": reward}},
                    "exception_info": None,
                }
            )
        )
        rows.append(
            {"path": name, "source": "source", "task_binary": archive(files), "solution_binary": archive(solution)}
        )
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    return source, results


def test_quality_packets_exclude_failed_oracles_and_preserve_visibility(candidates, tmp_path):
    source, results = candidates
    output = tmp_path / "packets"
    prepare(source, (results,), output, ())
    packets = pq.read_table(output / "packets.parquet").to_pylist()
    assert [row["task_id"] for row in packets] == ["source/pass"]
    evidence = json.loads(packets[0]["packet_json"])
    files = {row["path"]: row for row in evidence["files"] + evidence["oracle_files"]}
    assert files["instruction.md"]["text"] == "Print one"
    assert files["setup_files/seed.sh"]["role"] == "public"
    assert files["tests/test.sh"]["role"] == "verifier"
    assert files["solution/solve.sh"]["role"] == "oracle"
    assert files["solution/solve.sh"]["text"] == "echo 1"
    assert json.loads((output / "manifest.json").read_text())["oracle_passing_rows"] == 1


def test_quality_rejects_oracle_evidence_from_different_task_bytes(candidates, tmp_path):
    source, results = candidates
    (tmp_path / "tasks/pass/tests/test.sh").write_text("weakened checker")
    with pytest.raises(ValueError, match="Executed task differs from source"):
        prepare(source, (results,), tmp_path / "packets", ())
    assert not (tmp_path / "packets").exists()


def test_quality_accepts_infrastructure_retry_but_not_rescoring(candidates, tmp_path):
    source, results = candidates
    original = results / "pass/result.json"
    completed = json.loads(original.read_text())
    failed = completed | {"exception_info": {"exception_type": "DaytonaConnectionError"}, "verifier_result": None}
    original.write_text(json.dumps(failed))
    retry = tmp_path / "retry/pass"
    retry.mkdir(parents=True)
    (retry / "result.json").write_text(json.dumps(completed))
    output = tmp_path / "packets"
    prepare(source, (results, retry.parent), output, ())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["oracle_passing_rows"] == 1
    passing = next(row for row in manifest["decisions"] if row["task_path"] == "pass")
    assert len(set(passing["attempts_sha256"])) == 2
    with pytest.raises(ValueError, match="Cannot replace a scored oracle result"):
        prepare(source, (results, retry.parent, retry.parent), tmp_path / "invalid", ())
