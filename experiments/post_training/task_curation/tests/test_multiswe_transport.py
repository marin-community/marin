# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from experiments.post_training.task_curation.datasets.prime_intellect.multiswe.host_grade import source_test_output


@pytest.mark.parametrize(
    ("content", "read_exit_code", "stdout", "stderr", "expected"),
    [
        (b"--- PASS: TestUTF8_\xc3\xa9\r\n", 0, "ignored", "ignored", "--- PASS: TestUTF8_é\r\n"),
        (b"", 0, "harness output\r\n", "harness error", "harness output\r\n\nstderr:\nharness error"),
        (b"stale file", 1, "harness output", "", "harness output"),
        (b" \n", 0, "ignored", "ignored", " \n"),
    ],
)
def test_multiswe_copied_output_preserves_source_cat_and_fallback_semantics(
    tmp_path: Path, content: bytes, read_exit_code: int, stdout: str, stderr: str, expected: str
) -> None:
    output = tmp_path / "test-output.txt"
    output.write_bytes(content)
    assert source_test_output(output, read_exit_code, stdout, stderr) == expected
