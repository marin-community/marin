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

"""Tests for protobuf utilities."""

from iris.rpc import config_pb2
from iris.rpc.proto_utils import (
    accelerator_type_friendly,
    accelerator_type_name,
    format_accelerator_display,
)


def test_accelerator_type_name():
    """Test enum name conversion."""
    assert accelerator_type_name(config_pb2.ACCELERATOR_TYPE_UNSPECIFIED) == "ACCELERATOR_TYPE_UNSPECIFIED"
    assert accelerator_type_name(config_pb2.ACCELERATOR_TYPE_CPU) == "ACCELERATOR_TYPE_CPU"
    assert accelerator_type_name(config_pb2.ACCELERATOR_TYPE_GPU) == "ACCELERATOR_TYPE_GPU"
    assert accelerator_type_name(config_pb2.ACCELERATOR_TYPE_TPU) == "ACCELERATOR_TYPE_TPU"
    assert accelerator_type_name(999).startswith("UNKNOWN(")


def test_accelerator_type_friendly():
    """Test friendly name conversion."""
    assert accelerator_type_friendly(config_pb2.ACCELERATOR_TYPE_UNSPECIFIED) == "unspecified"
    assert accelerator_type_friendly(config_pb2.ACCELERATOR_TYPE_CPU) == "cpu"
    assert accelerator_type_friendly(config_pb2.ACCELERATOR_TYPE_GPU) == "gpu"
    assert accelerator_type_friendly(config_pb2.ACCELERATOR_TYPE_TPU) == "tpu"


def test_format_accelerator_display_with_variant():
    """Test formatted display with variant."""
    assert format_accelerator_display(config_pb2.ACCELERATOR_TYPE_TPU, "v5litepod-16") == "tpu (v5litepod-16)"
    assert format_accelerator_display(config_pb2.ACCELERATOR_TYPE_GPU, "A100") == "gpu (A100)"
    assert format_accelerator_display(config_pb2.ACCELERATOR_TYPE_GPU, "H100") == "gpu (H100)"


def test_format_accelerator_display_without_variant():
    """Test formatted display without variant."""
    assert format_accelerator_display(config_pb2.ACCELERATOR_TYPE_CPU, "") == "cpu"
    assert format_accelerator_display(config_pb2.ACCELERATOR_TYPE_CPU) == "cpu"
    assert format_accelerator_display(config_pb2.ACCELERATOR_TYPE_TPU, "") == "tpu"


def test_format_accelerator_display_handles_unknown():
    """Test handling of unknown accelerator types."""
    result = format_accelerator_display(999, "some-variant")
    assert "unknown" in result.lower()
