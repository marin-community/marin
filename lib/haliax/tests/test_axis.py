# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import pytest

import haliax
from haliax.axis import eliminate_axes, make_axes, rearrange_for_partial_order, replace_axis, without_axes


def test_eliminate_axes():
    H, W, C = make_axes(H=3, W=4, C=5)

    assert eliminate_axes((H, W), (H,)) == (W,)
    assert eliminate_axes((H, W), (W,)) == (H,)
    assert eliminate_axes((H, W), (H, W)) == ()

    with pytest.raises(ValueError):
        eliminate_axes((H, W), (C,))

    with pytest.raises(ValueError):
        eliminate_axes((H, W), (H, C))

    # test string references
    assert eliminate_axes((H, W), ("H",)) == (W,)
    assert eliminate_axes(("H", W), (H,)) == (W,)
    assert eliminate_axes(("H", W), ("H",)) == (W,)
    assert eliminate_axes(("H", W), ("H", "W")) == ()

    # test shape dicts

    HWC = {"H": 3, "W": 4, "C": 5}
    HW = {"H": 3, "W": 4}

    assert eliminate_axes(HWC, HW) == {"C": 5}
    with pytest.raises(ValueError):
        eliminate_axes(HW, HWC)
    assert eliminate_axes(HW, HW) == {}
    assert eliminate_axes(HWC, {"H": 3}) == {"W": 4, "C": 5}
    assert eliminate_axes(HWC, {"W": 4}) == {"H": 3, "C": 5}
    assert eliminate_axes(HWC, {"C": 5}) == HW

    C2 = make_axes(C=6)

    with pytest.raises(ValueError):
        eliminate_axes(HWC, C2)

    with pytest.raises(ValueError):
        eliminate_axes(C2, HWC)


def test_without_axes():
    H, W, C = make_axes(H=3, W=4, C=5)

    assert without_axes((H, W), (H,)) == (W,)
    assert without_axes((H, W), (W,)) == (H,)
    assert without_axes((H, W), (H, W)) == ()

    assert without_axes((H, W), (C,)) == (H, W)

    assert without_axes((H, W), (H, C)) == (W,)

    # test string references
    assert without_axes((H, W), ("H",)) == (W,)
    assert without_axes(("H", W), (H,)) == (W,)
    assert without_axes(("H", W), ("H",)) == (W,)
    assert without_axes(("H", W), ("H", "W")) == ()

    # test shape dicts

    HWC = {"H": 3, "W": 4, "C": 5}
    HW = {"H": 3, "W": 4}

    assert without_axes(HWC, HW) == {"C": 5}
    assert without_axes(HW, HWC) == {}
    assert without_axes(HW, HW) == {}
    assert without_axes(HWC, {"H": 3}) == {"W": 4, "C": 5}
    assert without_axes(HWC, {"W": 4}) == {"H": 3, "C": 5}
    assert without_axes(HWC, {"C": 5}) == HW

    # test different sizes cause error
    C2 = make_axes(C=6)

    with pytest.raises(ValueError):
        without_axes(HWC, C2)

    with pytest.raises(ValueError):
        without_axes(C2, HWC)


def test_replace_axis():
    H, W, C = make_axes(H=3, W=4, C=5)
    H2 = haliax.Axis("H2", 6)

    assert replace_axis((H, W), W, (C,)) == (H, C)

    with pytest.raises(ValueError):
        replace_axis((H, W), C, (H,))

    with pytest.raises(ValueError):
        replace_axis((H, W), H2, (C, W))

    # test string references
    with pytest.raises(ValueError):
        replace_axis((H, W), "H", (W,))

    with pytest.raises(ValueError):
        assert replace_axis(("H", W), "H", ("C", W))

    with pytest.raises(ValueError):
        assert replace_axis(("H", W), "H", ("W",)) == (W, W)

    assert replace_axis(("H", W), "H", ("C",)) == ("C", W)
    assert replace_axis(("H", W), "W", ("C",)) == ("H", "C")
    assert replace_axis(("H", W), "W", ("C", "D")) == ("H", "C", "D")

    # test shape dicts

    HWC = {"H": 3, "W": 4, "C": 5}
    HW = {"H": 3, "W": 4}
    HC = {"H": 3, "C": 5}

    assert replace_axis(HW, "H", HC) == {"H": 3, "C": 5, "W": 4}

    with pytest.raises(ValueError):
        replace_axis(HW, "H", HWC)

    with pytest.raises(ValueError):
        replace_axis(HW, "C", HC)

    with pytest.raises(ValueError):
        replace_axis(HW, "H", HWC)


def assert_partial_order_respected(partial_order, output):
    positions = {el: i for i, el in enumerate(output)}

    last_pos = -1
    for el in partial_order:
        if el is ...:
            # Reset last_pos for flexible positions
            last_pos = -1
        else:
            # Check if the element is in the correct order
            assert el in positions, f"{el} is missing in the output"
            assert positions[el] > last_pos, f"Partial order not respected for {el}"
            last_pos = positions[el]


def test_basic_order():
    partial_order = ("apple", ..., "banana")
    candidates = ("banana", "apple", "cherry")
    expected_output = ("apple", "cherry", "banana")
    actual_output = rearrange_for_partial_order(partial_order, candidates)
    assert actual_output == expected_output
    assert_partial_order_respected(partial_order, actual_output)


def test_start_with_ellipsis():
    partial_order = (..., "apple", "banana")
    candidates = ("banana", "apple", "cherry")
    actual_output = rearrange_for_partial_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)
    assert actual_output == ("cherry", "apple", "banana")


def test_end_with_ellipsis():
    partial_order = ("apple", ..., "banana", ...)
    candidates = ("banana", "apple", "cherry")
    actual_output = rearrange_for_partial_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)

    # this one could be either but we'll assert the order so we catch changes
    assert actual_output == ("apple", "banana", "cherry")


def test_full_flexibility():
    partial_order = (...,)
    candidates = ("banana", "apple", "cherry")
    actual_output = rearrange_for_partial_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)


def test_no_flexibility():
    partial_order = ("apple", "banana")
    candidates = ("banana", "apple", "cherry")
    with pytest.raises(ValueError):
        rearrange_for_partial_order(partial_order, candidates)


def test_final_ellipsis():
    partial_order = ("apple", "banana", ...)
    candidates = ("banana", "apple", "cherry")
    actual_output = rearrange_for_partial_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)
    assert actual_output == ("apple", "banana", "cherry")


def test_lots_of_ellipsis():
    partial_order = ("apple", ..., "banana", ..., "cherry", ...)
    candidates = ("banana", "orange", "cherry", "apple", "grape")
    actual_output = rearrange_for_partial_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)
    assert actual_output == ("apple", "banana", "orange", "cherry", "grape")


def test_no_ellipsis():
    partial_order = ("apple", "banana", "cherry")
    candidates = ("banana", "apple", "cherry")
    actual_output = rearrange_for_partial_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)
    assert actual_output == ("apple", "banana", "cherry")


def test_no_elements():
    partial_order = (...,)
    candidates = ()
    actual_output = rearrange_for_partial_order(partial_order, candidates)
    assert_partial_order_respected(partial_order, actual_output)
    assert actual_output == ()


def test_missing_elements_errors():
    partial_order = ("qux", ...)
    candidates = ("banana", "apple", "cherry")
    with pytest.raises(ValueError):
        rearrange_for_partial_order(partial_order, candidates)


def test_duplicate_elements_errors():
    partial_order: tuple = ("apple", "apple", ...)
    candidates = ("banana", "apple", "cherry")
    with pytest.raises(ValueError):
        rearrange_for_partial_order(partial_order, candidates)

    candidates = ("banana", "apple", "apple")

    with pytest.raises(ValueError):
        rearrange_for_partial_order(partial_order, candidates)

    partial_order = ("apple", "banana", "grape", ...)

    with pytest.raises(ValueError):
        rearrange_for_partial_order(partial_order, candidates)
