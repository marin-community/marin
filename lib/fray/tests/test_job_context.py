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

"""Tests for execution contexts."""

import pytest
from fray import RayContext, SyncContext, ThreadContext, fray_job_ctx


@pytest.fixture
def execution_context(context_type):
    """Create execution context based on context_type parameter.

    This fixture provides either a SyncContext, ThreadContext, or RayContext
    instance depending on the context_type parameter.
    """
    if context_type == "sync":
        return SyncContext()
    elif context_type == "thread":
        return ThreadContext(max_workers=2)
    elif context_type == "ray":
        return RayContext()
    else:
        raise ValueError(f"Unknown context type: {context_type}")


def test_context_put_get(execution_context):
    obj = {"key": "value"}
    ref = execution_context.put(obj)
    assert execution_context.get(ref) == obj


def test_context_run(execution_context):
    future = execution_context.run(lambda x: x * 2, 5)
    assert execution_context.get(future) == 10


def test_context_wait(execution_context):
    futures = [execution_context.run(lambda x: x, i) for i in range(5)]
    ready, pending = execution_context.wait(futures, num_returns=2)
    assert len(ready) == 2
    assert len(pending) == 3


def test_fray_job_ctx_invalid():
    with pytest.raises(ValueError, match="Unknown context type"):
        fray_job_ctx("invalid")  # type: ignore
