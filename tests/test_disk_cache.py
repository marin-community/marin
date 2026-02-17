# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from marin.execution.artifact import Artifact
from marin.execution.disk_cache import disk_cached, distributed_lock
from marin.execution.executor_step_status import STATUS_SUCCESS, StatusFile
from marin.execution.step_model import StepSpec


@dataclass
class TokenizerInfo:
    model: str
    vocab_size: int


def test_disk_cached_runs_and_caches(tmp_path: Path):
    call_count = 0

    def counting_fn(output_path: str) -> TokenizerInfo:
        nonlocal call_count
        call_count += 1
        os.makedirs(output_path, exist_ok=True)
        return TokenizerInfo(model="gpt2", vocab_size=50257)

    result1 = disk_cached("tok", counting_fn, output_path_prefix=tmp_path.as_posix())
    assert call_count == 1
    assert result1 == TokenizerInfo(model="gpt2", vocab_size=50257)

    result2 = disk_cached("tok", counting_fn, output_path_prefix=tmp_path.as_posix())
    assert call_count == 1, "Function should not be called again on cache hit"
    assert result2 == result1


def test_disk_cached_raises_without_type(tmp_path: Path):
    with pytest.raises(TypeError, match="Cannot infer artifact_type"):
        disk_cached(
            "tok",
            lambda output_path: TokenizerInfo(model="x", vocab_size=1),
            output_path_prefix=tmp_path.as_posix(),
        )


def test_composition_runs_and_caches(tmp_path: Path):
    call_count = 0

    def counting_fn(output_path: str) -> TokenizerInfo:
        nonlocal call_count
        call_count += 1
        os.makedirs(output_path, exist_ok=True)
        return TokenizerInfo(model="gpt2", vocab_size=50257)

    result1 = disk_cached("comp", distributed_lock(counting_fn), output_path_prefix=tmp_path.as_posix())
    assert call_count == 1
    assert result1 == TokenizerInfo(model="gpt2", vocab_size=50257)

    result2 = disk_cached("comp", distributed_lock(counting_fn), output_path_prefix=tmp_path.as_posix())
    assert call_count == 1
    assert result2 == result1


def test_composition_skips_when_another_worker_completed(tmp_path: Path):
    """disk_cached should load the artifact when another worker already completed the step."""
    prefix = tmp_path.as_posix()

    # Simulate another worker having completed the step
    spec = StepSpec(name="race", output_path_prefix=prefix)
    os.makedirs(spec.output_path, exist_ok=True)
    Artifact.save(TokenizerInfo(model="other", vocab_size=999), spec.output_path)
    StatusFile(spec.output_path, "other-worker").write_status(STATUS_SUCCESS)

    call_count = 0

    def my_fn(output_path: str) -> TokenizerInfo:
        nonlocal call_count
        call_count += 1
        return TokenizerInfo(model="mine", vocab_size=1)

    result = disk_cached("race", distributed_lock(my_fn), output_path_prefix=prefix)

    assert call_count == 0
    assert result == TokenizerInfo(model="other", vocab_size=999)
