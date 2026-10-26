"""Regression contracts for the four-attempt Pi verifier smoke task pack."""

import io
import tarfile

import pyarrow.parquet as pq

from cloud.iris.pi_verifier_grpo_tasks import INPUTS, build_taskpack


def test_verifier_grpo_tasks_are_distinct_and_keep_expected_answers_private():
    payload = build_taskpack()
    assert payload == build_taskpack()
    table = pq.read_table(io.BytesIO(payload))
    rows = table.to_pylist()
    assert len(rows) == 4
    assert len({row["path"] for row in rows}) == 4
    assert all(row["path"].startswith("pi-grpo-sum-") for row in rows)

    for row, operands in zip(rows, INPUTS, strict=True):
        with tarfile.open(fileobj=io.BytesIO(row["task_binary"]), mode="r:") as archive:
            instruction = archive.extractfile("instruction.md").read().decode()
            private_test = archive.extractfile("tests/test.sh").read().decode()
        assert ", ".join(map(str, operands)) in instruction
        assert str(sum(operands)) not in instruction
        assert f'"{sum(operands)}"' in private_test
        assert "reward.txt" not in instruction
