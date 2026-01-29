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

"""Tests for Fray v2 types."""

import pytest

from fray.v2.types import (
    Entrypoint,
    EnvironmentSpec,
    JobStatus,
    ResourceSpec,
    namespace_from_job_id,
    parse_memory_string,
)


class TestResourceSpec:
    def test_default_values(self):
        spec = ResourceSpec()
        assert spec.cpu == 0
        assert spec.memory == 0
        assert spec.disk == 0
        assert spec.replicas == 1
        assert spec.preemptible is False

    def test_with_tpu(self):
        spec = ResourceSpec.with_tpu("v5litepod-16", replicas=2)
        assert spec.device_type == "tpu"
        assert spec.device_variant == "v5litepod-16"
        assert spec.replicas == 2

    def test_with_gpu(self):
        spec = ResourceSpec.with_gpu("H100", count=4)
        assert spec.device_type == "gpu"
        assert spec.device_variant == "H100"
        assert spec.device_count == 4

    def test_with_cpu(self):
        spec = ResourceSpec.with_cpu(cpu=4, memory="8g")
        assert spec.cpu == 4
        assert spec.memory == "8g"

    def test_memory_bytes(self):
        spec = ResourceSpec(memory="8g")
        assert spec.memory_bytes() == 8 * 1024 * 1024 * 1024

        spec2 = ResourceSpec(memory=1024)
        assert spec2.memory_bytes() == 1024

    def test_disk_bytes(self):
        spec = ResourceSpec(disk="100g")
        assert spec.disk_bytes() == 100 * 1024 * 1024 * 1024


class TestEnvironmentSpec:
    def test_default_values(self):
        spec = EnvironmentSpec()
        assert spec.workspace is None
        assert spec.pip_packages is None
        assert spec.env_vars is None
        assert spec.extras is None

    def test_effective_workspace(self):
        import os

        spec = EnvironmentSpec()
        assert spec.effective_workspace() == os.getcwd()

        spec2 = EnvironmentSpec(workspace="/custom/path")
        assert spec2.effective_workspace() == "/custom/path"

    def test_effective_env_vars(self):
        spec = EnvironmentSpec(env_vars={"CUSTOM_VAR": "value"})
        env_vars = spec.effective_env_vars()

        # Check defaults are applied
        assert env_vars["HF_DATASETS_TRUST_REMOTE_CODE"] == "1"
        assert env_vars["TOKENIZERS_PARALLELISM"] == "false"

        # Check custom var is included
        assert env_vars["CUSTOM_VAR"] == "value"

    def test_effective_env_vars_override(self):
        spec = EnvironmentSpec(env_vars={"TOKENIZERS_PARALLELISM": "true"})
        env_vars = spec.effective_env_vars()

        # Custom value should override default
        assert env_vars["TOKENIZERS_PARALLELISM"] == "true"


class TestEntrypoint:
    def test_from_callable(self):
        def my_func(x, y, z=10):
            return x + y + z

        entry = Entrypoint.from_callable(my_func, 1, 2, z=3)
        assert entry.callable is my_func
        assert entry.args == (1, 2)
        assert entry.kwargs == {"z": 3}

    def test_default_values(self):
        def my_func():
            pass

        entry = Entrypoint(callable=my_func)
        assert entry.args == ()
        assert entry.kwargs == {}


class TestJobStatus:
    def test_is_finished(self):
        assert JobStatus.is_finished(JobStatus.SUCCEEDED) is True
        assert JobStatus.is_finished(JobStatus.FAILED) is True
        assert JobStatus.is_finished(JobStatus.KILLED) is True
        assert JobStatus.is_finished(JobStatus.PENDING) is False
        assert JobStatus.is_finished(JobStatus.RUNNING) is False


class TestNamespace:
    def test_from_job_id_simple(self):
        ns = namespace_from_job_id("abc123")
        assert ns == "abc123"

    def test_from_job_id_hierarchical(self):
        ns = namespace_from_job_id("abc123/worker-0")
        assert ns == "abc123"

        ns2 = namespace_from_job_id("abc123/worker-0/sub-task")
        assert ns2 == "abc123"

    def test_from_job_id_empty_raises(self):
        with pytest.raises(ValueError, match="Job ID cannot be empty"):
            namespace_from_job_id("")


class TestParseMemoryString:
    def test_int_passthrough(self):
        assert parse_memory_string(1024) == 1024

    def test_gigabytes(self):
        assert parse_memory_string("8g") == 8 * 1024 * 1024 * 1024
        assert parse_memory_string("8G") == 8 * 1024 * 1024 * 1024
        assert parse_memory_string("8gb") == 8 * 1024 * 1024 * 1024

    def test_megabytes(self):
        assert parse_memory_string("512m") == 512 * 1024 * 1024
        assert parse_memory_string("512M") == 512 * 1024 * 1024

    def test_empty_string(self):
        assert parse_memory_string("") == 0
        assert parse_memory_string("0") == 0

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            parse_memory_string("invalid")
