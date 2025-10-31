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

"""Shared test fixtures for Fray tests."""

import os
import shutil

import pytest
import ray

# Check if Monarch is available
try:
    from fray.backend.monarch.monarch_helpers import MONARCH_AVAILABLE
except ImportError:
    MONARCH_AVAILABLE = False


@pytest.fixture(scope="session")
def ray_cluster(tmp_path_factory):
    """
    Start a real local Ray cluster for testing.

    This uses a proper Ray cluster (not local_mode) to test realistic
    distributed behavior including async task execution and timeouts.
    """
    # Use temp directory that won't exceed AF_UNIX path length (103 bytes)
    tmp_path = f"/tmp/fray_tests/{os.getpid()}"

    init_args = {
        "address": "local",
        "namespace": "fray",
        "_temp_dir": tmp_path,
        "num_cpus": 8,
        "num_gpus": 2,  # Add GPU resources for testing resource scheduling
        "ignore_reinit_error": True,
    }

    ctx = ray.init(**init_args)

    yield ctx

    # Cleanup
    ray.shutdown()
    shutil.rmtree(tmp_path, ignore_errors=True)
