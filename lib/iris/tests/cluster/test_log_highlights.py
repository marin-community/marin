# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the failure-log highlight extractor."""

import pytest
from iris.cluster.log_highlights import extract_failure_highlights

_TRACEBACK = [
    "starting training loop",
    "step 100/1000",
    "Traceback (most recent call last):",
    '  File "/app/train.py", line 88, in <module>',
    "    train()",
    "RuntimeError: CUDA error: an illegal memory access was encountered",
]


@pytest.mark.parametrize(
    ("lines", "max_lines", "expected"),
    [
        # tqdm progress bars are dropped; the real error survives.
        (
            ["loading dataset", " 45%|####5     | 450/1000 [00:12<00:15,  3.21it/s]", "RuntimeError: shard missing"],
            20,
            ["RuntimeError: shard missing"],
        ),
        # HTTP access-log noise is dropped.
        (
            ['127.0.0.1 - - "GET /metrics HTTP/1.1" 200 512', "ValueError: invalid shape (0,)"],
            20,
            ["ValueError: invalid shape (0,)"],
        ),
        # CPython's post-crash "Extension modules:" tail is dropped.
        (
            [
                "Fatal Python error: Segmentation fault",
                '  File "/app/train.py", line 42 in step',
                "Extension modules: jax",
            ],
            20,
            ["Fatal Python error: Segmentation fault", '  File "/app/train.py", line 42 in step'],
        ),
        # A traceback is kept down to its signal lines (frame bodies drop out).
        (
            _TRACEBACK,
            20,
            [
                "Traceback (most recent call last):",
                '  File "/app/train.py", line 88, in <module>',
                "RuntimeError: CUDA error: an illegal memory access was encountered",
            ],
        ),
        # A barrier-timeout line repeated once per straggler collapses to one.
        (
            [
                "Barrier result: DEADLINE_EXCEEDED; reached: 1/8",
                "Barrier result: DEADLINE_EXCEEDED; reached: 1/8",
                "Barrier result: DEADLINE_EXCEEDED; reached: 1/8",
                "Terminating process because the JAX distributed service detected fatal errors",
            ],
            20,
            [
                "Barrier result: DEADLINE_EXCEEDED; reached: 1/8",
                "Terminating process because the JAX distributed service detected fatal errors",
            ],
        ),
        # A described bar and a totalless bar are noise too, wherever they sit
        # on the line.
        (
            [
                "Loading shards:  45%|####5     | 450/1000 [00:12<00:15,  1.20s/it]",
                "450it [00:12,  3.21it/s]",
                "RuntimeError: shard missing",
            ],
            20,
            ["RuntimeError: shard missing"],
        ),
        # No signal line → fall back to the de-noised tail so output is never empty.
        (["step 1", "step 2", "step 3"], 20, ["step 1", "step 2", "step 3"]),
        # Empty input → empty output.
        ([], 20, []),
        # max_lines caps the result to the most recent lines.
        (
            [f"ValueError: failure {i}" for i in range(30)],
            5,
            [f"ValueError: failure {i}" for i in range(25, 30)],
        ),
    ],
)
def test_extract_failure_highlights(lines, max_lines, expected):
    assert extract_failure_highlights(lines, max_lines=max_lines) == expected
