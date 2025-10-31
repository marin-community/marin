# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for TPU interface in Fray."""

import pytest
from fray import LocalClusterContext, TpuRunConfig
from fray.job import JobContext


def test_run_on_tpu_single_slice():
    """Test run_on_tpu with single slice runs correct number of times."""
    cluster = LocalClusterContext()

    call_count = 0

    def test_fn(ctx: JobContext):
        nonlocal call_count
        call_count += 1
        return call_count

    config = TpuRunConfig(tpu_type="v4-32", num_slices=1)
    results = cluster.run_on_tpu(test_fn, config)

    # v4-32 has 4 VMs per slice
    assert len(results) == 4
    assert results == [1, 2, 3, 4]
    assert call_count == 4


def test_run_on_tpu_multiple_slices():
    """Test run_on_tpu with multiple slices."""
    cluster = LocalClusterContext()

    call_count = 0

    def test_fn(ctx: JobContext):
        nonlocal call_count
        call_count += 1
        return call_count

    config = TpuRunConfig(tpu_type="v5p-128", num_slices=2)
    results = cluster.run_on_tpu(test_fn, config)

    # v5p-128 has 16 VMs per slice
    assert len(results) == 32  # 16 * 2
    assert call_count == 32


def test_run_on_tpu_with_job_context():
    """Test that function receives working JobContext."""
    cluster = LocalClusterContext()

    def test_fn(ctx: JobContext):
        # Test that we can use the JobContext
        ref = ctx.create_task(lambda: 42)
        result = ctx.get(ref)
        return result

    config = TpuRunConfig(tpu_type="v4-8", num_slices=1)
    results = cluster.run_on_tpu(test_fn, config)

    # v4-8 has 1 VM per slice
    assert len(results) == 1
    assert results[0] == 42


@pytest.mark.parametrize(
    "tpu_type,expected_vms",
    [
        ("v4-8", 1),
        ("v4-32", 4),
        ("v4-64", 8),
        ("v5p-128", 16),
        ("v5p-256", 32),
        ("v6e-256", 64),
    ],
)
def test_run_on_tpu_vm_counts(tpu_type, expected_vms):
    """Test that different TPU types run correct number of times."""
    cluster = LocalClusterContext()

    call_count = 0

    def test_fn(ctx: JobContext):
        nonlocal call_count
        call_count += 1
        return call_count

    config = TpuRunConfig(tpu_type=tpu_type, num_slices=1)
    results = cluster.run_on_tpu(test_fn, config)

    assert len(results) == expected_vms
    assert call_count == expected_vms


def test_run_on_tpu_preserves_exceptions():
    """Test that exceptions in the function are propagated."""
    cluster = LocalClusterContext()

    def test_fn(ctx: JobContext):
        raise ValueError("Test error")

    config = TpuRunConfig(tpu_type="v4-8", num_slices=1)

    with pytest.raises(ValueError, match="Test error"):
        cluster.run_on_tpu(test_fn, config)
