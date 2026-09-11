# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for protobuf utilities."""

from iris.rpc.proto_display import format_accelerator_display


def test_variant_preserved_in_display():
    """Variant information is preserved when formatting for display."""
    variant = "v5litepod-16"
    result = format_accelerator_display("tpu", variant)
    assert result == "tpu (v5litepod-16)"
    assert variant in result


def test_empty_variant_omitted_from_display():
    """Empty variant strings don't clutter the display."""
    no_variant = format_accelerator_display("cpu", "")
    default_variant = format_accelerator_display("cpu")

    assert no_variant == "cpu"
    assert "(" not in no_variant, "Empty variant should not add parentheses"
    assert "(" not in default_variant, "Default empty variant should not add parentheses"
    assert no_variant == default_variant, "Empty string and default should produce identical output"


def test_unspecified_device_type():
    """An empty device type renders as 'unspecified'."""
    assert format_accelerator_display("", "") == "unspecified"
