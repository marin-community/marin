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


def test_accelerator_type_name_handles_unknown():
    """Unknown accelerator types are marked clearly."""
    assert accelerator_type_name(999).startswith("UNKNOWN(")


def test_format_accelerator_display_handles_unknown():
    """Unknown accelerator types are handled gracefully."""
    result = format_accelerator_display(999, "some-variant")
    assert "unknown" in result.lower()


def test_accelerator_types_are_distinguishable():
    """Different accelerator types produce different display names."""
    types = [
        config_pb2.ACCELERATOR_TYPE_UNSPECIFIED,
        config_pb2.ACCELERATOR_TYPE_CPU,
        config_pb2.ACCELERATOR_TYPE_GPU,
        config_pb2.ACCELERATOR_TYPE_TPU,
    ]
    friendly_names = [accelerator_type_friendly(t) for t in types]
    assert len(friendly_names) == len(set(friendly_names)), "All accelerator types must have unique display names"


def test_variant_preserved_in_display():
    """Variant information is preserved when formatting for display."""
    variant = "v5litepod-16"
    result = format_accelerator_display(config_pb2.ACCELERATOR_TYPE_TPU, variant)
    assert variant in result, f"Variant '{variant}' must be visible in display: {result}"


def test_empty_variant_omitted_from_display():
    """Empty variant strings don't clutter the display."""
    no_variant = format_accelerator_display(config_pb2.ACCELERATOR_TYPE_CPU, "")
    default_variant = format_accelerator_display(config_pb2.ACCELERATOR_TYPE_CPU)

    assert "(" not in no_variant, "Empty variant should not add parentheses"
    assert "(" not in default_variant, "Default empty variant should not add parentheses"
    assert no_variant == default_variant, "Empty string and default should produce identical output"
