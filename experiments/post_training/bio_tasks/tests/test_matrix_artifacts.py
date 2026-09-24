# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import hashlib
import json
import subprocess
import sys

import pytest
from tasktrove_verify.grade import Status

from experiments.post_training.bio_tasks.build import verifier_archive
from experiments.post_training.bio_tasks.contract import Column, Contract, MatrixMarketContract, grade_files

HEADER = b"%%MatrixMarket matrix coordinate integer general\n"
ENTRIES = b"1 2 5\n2 1 7\n3 2 9\n"
VALID = HEADER + b"3 2 3\n" + ENTRIES


@pytest.fixture(params=["counts.mtx", "counts.mtx.gz"])
def matrix_submission(tmp_path, request):
    contract = Contract(
        columns={"total": Column(kind="integer", description="total observed counts", unit="counts")},
        expected={"summary": {"total": 21}},
        matrices={
            request.param: MatrixMarketContract(
                rows=3,
                columns=2,
                nonzeros=3,
                sha256=hashlib.sha256(ENTRIES).hexdigest(),
                max_bytes=1024,
                max_decoded_bytes=4096,
            )
        },
    )
    reference = tmp_path / "reference.json"
    reference.write_text(contract.model_dump_json())
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps(contract.answer()))
    return reference, answer, tmp_path / request.param


def write_matrix(path, content):
    path.write_bytes(gzip.compress(content, mtime=0) if path.suffix == ".gz" else content)


def test_matrix_entry_order_and_text_representation_do_not_change_grade(matrix_submission):
    reference, answer, artifact = matrix_submission
    for content in (
        VALID,
        b"%%MatrixMarket\tmatrix  coordinate INTEGER general\r\n% annotation\r\n3 2 3\r\n"
        b"03\t+2 009\r\n\n2 1 7\r\n% another comment\r\n1 2 5\r\n",
    ):
        write_matrix(artifact, content)
        assert grade_files(reference, answer).reward == 1


@pytest.mark.parametrize(
    "content",
    [
        VALID.replace(b"1 2 5", b"1 2 6"),  # Changed count with unchanged JSON total.
        VALID.replace(b"1 2 5", b"1 1 5"),  # Same counts and total; different cell.
        VALID.replace(b"3 2 9\n", b""),
        VALID + b"1 1 1\n",
        VALID.replace(b"3 2 9", b"1 2 9"),  # Duplicate coordinates.
        VALID.replace(b"3 2 3\n", b"2 3 3\n"),  # Transposed axes.
        VALID.replace(b"3 2 9", b"4 2 9"),
        VALID.replace(b"1 2 5", b"0 2 5"),
        VALID.replace(b"1 2 5", b"1 0 5"),
        VALID.replace(b"1 2 5", b"1 2 0"),
        VALID.replace(b"1 2 5", b"1 2 -5"),
        VALID.replace(b"1 2 5", b"1 2 5.0"),
        VALID.replace(b"1 2 5", b"1 2 9223372036854775808"),
        VALID.replace(b"1 2 5", b"1 2 nan"),
        VALID.replace(b"integer general", b"integer symmetric"),
        VALID + b"%" + b" " * 4096,  # Expansion limit applies to ignored comments too.
        b"",
    ],
)
def test_wrong_matrix_fails_with_correct_summary(matrix_submission, content):
    reference, answer, artifact = matrix_submission
    write_matrix(artifact, content)
    result = grade_files(reference, answer)
    assert result.status == Status.SCORED and result.reward == 0


def test_missing_symlink_and_corrupt_gzip_matrix_fail(matrix_submission):
    reference, answer, artifact = matrix_submission
    assert grade_files(reference, answer).reward == 0
    external = artifact.with_name("external" + artifact.suffix)
    write_matrix(external, VALID)
    artifact.symlink_to(external)
    assert grade_files(reference, answer).reward == 0
    artifact.unlink()
    if artifact.suffix == ".gz":
        encoded = gzip.compress(VALID, mtime=0)
        for broken in (encoded[:-4], encoded[:-8] + b"\0" * 8, VALID):
            artifact.write_bytes(broken)
            result = grade_files(reference, answer)
            assert result.status == Status.SCORED and result.reward == 0


def test_packaged_matrix_verifier_checks_all_counts(matrix_submission, tmp_path):
    reference, answer, artifact = matrix_submission
    program = tmp_path / "verifier.pyz"
    program.write_bytes(verifier_archive())
    command = [
        sys.executable,
        "-I",
        str(program),
        "--reference",
        str(reference),
        "--answer",
        str(answer),
        "--logs",
        str(tmp_path / "logs"),
    ]
    for content, expected in ((VALID, 1), (VALID.replace(b"3 2 9", b"3 2 8"), 0)):
        write_matrix(artifact, content)
        subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, timeout=30)
        assert json.loads((tmp_path / "logs/reward.json").read_text()) == {"reward": expected}


def test_missing_sort_is_infrastructure_error(matrix_submission, monkeypatch):
    reference, answer, artifact = matrix_submission
    write_matrix(artifact, VALID)
    monkeypatch.setenv("PATH", str(artifact.parent / "no-executables"))
    assert grade_files(reference, answer).status == Status.INFRA_ERROR
