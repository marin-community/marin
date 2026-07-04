# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the failure-log highlight extractor."""

from iris.cluster.log_highlights import extract_failure_highlights


def test_drops_tqdm_progress_bars():
    lines = [
        "loading dataset",
        " 45%|####5     | 450/1000 [00:12<00:15,  3.21it/s]",
        " 46%|####6     | 460/1000 [00:12<00:15,  3.20it/s]",
        "RuntimeError: dataset shard missing",
    ]
    result = extract_failure_highlights(lines)
    assert result == ["RuntimeError: dataset shard missing"]


def test_drops_http_access_log_lines():
    lines = [
        '127.0.0.1 - - "GET /metrics HTTP/1.1" 200 512',
        '127.0.0.1 - - "GET /metrics HTTP/1.1" 200 512',
        "ValueError: invalid shape (0,)",
    ]
    result = extract_failure_highlights(lines)
    assert result == ["ValueError: invalid shape (0,)"]


def test_drops_extension_modules_crash_dump_tail():
    lines = [
        "Fatal Python error: Segmentation fault",
        "Current thread 0x00007f4a2f9ff700 (most recent call first):",
        '  File "/app/train.py", line 42 in step',
        "Extension modules: numpy, torch, jax, jaxlib, cuda, cudnn, tensorstore, msgpack",
    ]
    result = extract_failure_highlights(lines)
    assert result == [
        "Fatal Python error: Segmentation fault",
        '  File "/app/train.py", line 42 in step',
    ]


def test_keeps_python_traceback():
    lines = [
        "starting training loop",
        "step 100/1000",
        "Traceback (most recent call last):",
        '  File "/app/train.py", line 88, in <module>',
        "    train()",
        "RuntimeError: CUDA error: an illegal memory access was encountered",
    ]
    result = extract_failure_highlights(lines)
    assert result[0] == "Traceback (most recent call last):"
    assert result[-1] == "RuntimeError: CUDA error: an illegal memory access was encountered"


def test_dedupes_repeated_barrier_timeout_lines():
    """A JAX shutdown-barrier timeout repeats near-identically across followers."""
    lines = [
        "Barrier result: DEADLINE_EXCEEDED; # of tasks that reached the barrier: 1/8",
        "Barrier result: DEADLINE_EXCEEDED; # of tasks that reached the barrier: 1/8",
        "Barrier result: DEADLINE_EXCEEDED; # of tasks that reached the barrier: 1/8",
        "Terminating process because the JAX distributed service detected fatal errors",
    ]
    result = extract_failure_highlights(lines)
    assert result == [
        "Barrier result: DEADLINE_EXCEEDED; # of tasks that reached the barrier: 1/8",
        "Terminating process because the JAX distributed service detected fatal errors",
    ]


def test_falls_back_to_denoised_tail_when_no_signal_line_matches():
    lines = ["step 1", "step 2", "step 3"]
    result = extract_failure_highlights(lines)
    assert result == lines


def test_empty_input_returns_empty():
    assert extract_failure_highlights([]) == []


def test_respects_max_lines():
    lines = [f"ValueError: failure {i}" for i in range(30)]
    result = extract_failure_highlights(lines, max_lines=5)
    assert result == lines[-5:]
