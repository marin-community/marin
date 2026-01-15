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

"""Tests for FlusterContext and context management."""

import os
from unittest import mock

from fluster.context import create_context_from_env


def test_create_context_from_env_with_attempt_id():
    """Test that create_context_from_env properly reads FLUSTER_ATTEMPT_ID."""
    env = {
        "FLUSTER_JOB_ID": "test-job/sub-job",
        "FLUSTER_ATTEMPT_ID": "3",
        "FLUSTER_WORKER_ID": "worker-123",
    }

    with mock.patch.dict(os.environ, env, clear=True):
        ctx = create_context_from_env()

        assert ctx.job_id == "test-job/sub-job"
        assert ctx.attempt_id == 3
        assert ctx.worker_id == "worker-123"


def test_create_context_from_env_defaults_attempt_id():
    """Test that create_context_from_env defaults FLUSTER_ATTEMPT_ID to 0."""
    env = {
        "FLUSTER_JOB_ID": "test-job",
        "FLUSTER_WORKER_ID": "worker-456",
    }

    with mock.patch.dict(os.environ, env, clear=True):
        ctx = create_context_from_env()

        assert ctx.job_id == "test-job"
        assert ctx.attempt_id == 0
        assert ctx.worker_id == "worker-456"
