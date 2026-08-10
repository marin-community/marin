# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.grug._jax_mixed_memory_donation import _mixed_memory_safe_alias_setup


def test_mixed_memory_donation_preserves_known_kinds_and_isolates_unknowns():
    captured = None

    def original(*args):
        nonlocal captured
        captured = args
        return "result"

    wrapped = _mixed_memory_safe_alias_setup(original)
    result = wrapped(
        None,
        ("in0", "in1", "in2"),
        ("out0", "out1", "out2"),
        (True, True, True),
        ("pinned_host", "device", None),
        ("device", "pinned_host", None),
        None,
        None,
        None,
    )

    assert result == "result"
    assert captured is not None
    arg_kinds = captured[4]
    result_kinds = captured[5]
    assert arg_kinds[:2] == ["pinned_host", "device"]
    assert result_kinds[:2] == ["device", "pinned_host"]
    assert arg_kinds[2] is not result_kinds[2]
    assert arg_kinds[2] is not None
    assert result_kinds[2] is not None


def test_all_unknown_memory_kinds_preserve_stock_jax_behavior():
    captured = None

    def original(*args):
        nonlocal captured
        captured = args

    wrapped = _mixed_memory_safe_alias_setup(original)
    wrapped(None, ("in",), ("out",), (True,), (None,), (None,), None, None, None)

    assert captured is not None
    assert captured[4] == [None]
    assert captured[5] == [None]
