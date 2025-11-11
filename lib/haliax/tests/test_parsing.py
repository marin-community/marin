# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import pytest

from haliax._src.parsing import parse_einsum, parse_rearrangement


def _simplify_captures(expr):
    def simplify_capture(capture):
        if capture == Ellipsis:
            return Ellipsis
        elif (capture.binding == capture.axes[0] or capture.binding is None) and len(capture.axes) == 1:
            return capture.axes[0]
        elif capture.binding is None:
            return capture.axes
        else:
            return {capture.binding: capture.axes}

    return [simplify_capture(capture) for capture in expr.captures]


def test_parse_rearrangement_simple():
    lhs, rhs = parse_rearrangement("a b c d -> b c a d")
    assert lhs.is_ordered
    assert _simplify_captures(lhs) == ["a", "b", "c", "d"]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == ["b", "c", "a", "d"]

    lhs, rhs = parse_rearrangement("a ... c d -> b c a d")
    assert lhs.is_ordered
    assert _simplify_captures(lhs) == ["a", Ellipsis, "c", "d"]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == ["b", "c", "a", "d"]

    # longer identifiers
    lhs, rhs = parse_rearrangement("a_longer b123 c d -> b123 c a_longer d")
    assert lhs.is_ordered
    assert _simplify_captures(lhs) == ["a_longer", "b123", "c", "d"]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == ["b123", "c", "a_longer", "d"]


def test_parse_einsum_ordered():
    # We could support this syntax for dot with something like:
    #
    #     support normal einops syntax, including short name-capture: hax.dot("... c h w, h w d -> ...  c d", a, b)
    #     hax.dot("{h, w} -> ", a, b) means "contract h and w", analogous to hax.dot(a, b, axis=("h", "w"))
    #     hax.dot("{h, w} -> ... channel embed", a, b) means "contract h and w and ensure that the result ends with [channel, embed]" (by transposing/einsum)
    #     hax.dot(" -> batch channel embed", a, b) could mean "contract all but the named dims". Not entirely sure how I feel about that one, but used situationally it's probably ok

    lhses, rhs = parse_einsum("a b c d, b c e f -> a d e f")
    assert lhses is not None
    assert len(lhses) == 2
    assert all(lhs.is_ordered for lhs in lhses)
    lhs0_captures = _simplify_captures(lhses[0])
    lhs1_captures = _simplify_captures(lhses[1])
    assert lhs0_captures == ["a", "b", "c", "d"]
    assert lhs1_captures == ["b", "c", "e", "f"]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == ["a", "d", "e", "f"]

    lhses, rhs = parse_einsum("... c h w, h w d -> ...  c d")
    assert lhses is not None
    assert len(lhses) == 2
    assert all(lhs.is_ordered for lhs in lhses)
    lhs0_captures = _simplify_captures(lhses[0])
    lhs1_captures = _simplify_captures(lhses[1])
    assert lhs0_captures == [..., "c", "h", "w"]
    assert lhs1_captures == ["h", "w", "d"]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == [..., "c", "d"]

    lhses, rhs = parse_einsum("{...} -> batch channel embed")
    assert lhses is not None
    assert len(lhses) == 1
    assert not lhses[0].is_ordered
    assert _simplify_captures(lhses[0]) == [...]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == ["batch", "channel", "embed"]

    # just lhs
    lhses, rhs = parse_einsum("batch channel embed -> ")
    assert lhses is not None
    assert len(lhses) == 1
    assert lhses[0].is_ordered
    assert _simplify_captures(lhses[0]) == ["batch", "channel", "embed"]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == []

    # lhs x 2, empty rhs
    lhses, rhs = parse_einsum("batch channel embed, batch channel embed ->")
    assert lhses is not None
    assert len(lhses) == 2
    assert all(lhs.is_ordered for lhs in lhses)
    assert _simplify_captures(lhses[0]) == ["batch", "channel", "embed"]
    assert _simplify_captures(lhses[1]) == ["batch", "channel", "embed"]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == []


def test_parse_einsum_unordered():
    lhses, rhs = parse_einsum("{a, b} -> ")
    assert lhses is not None
    assert len(lhses) == 1
    assert not lhses[0].is_ordered
    assert _simplify_captures(lhses[0]) == ["a", "b"]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == []

    lhses, rhs = parse_einsum("{...} -> ")
    assert lhses is not None
    assert len(lhses) == 1
    assert not lhses[0].is_ordered
    assert _simplify_captures(lhses[0]) == [...]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == []

    lhses, rhs = parse_einsum("{h, w} -> ... channel embed")
    assert lhses is not None
    assert len(lhses) == 1
    assert not lhses[0].is_ordered
    assert _simplify_captures(lhses[0]) == ["h", "w"]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == [..., "channel", "embed"]


def test_parse_paren_groups():
    lhs, rhs = parse_rearrangement("a (b c) d -> b c a d")
    assert lhs.is_ordered
    assert _simplify_captures(lhs) == ["a", ("b", "c"), "d"]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == ["b", "c", "a", "d"]

    lhs, rhs = parse_rearrangement("a (b: c) (d: e f) -> b c a d")
    assert lhs.is_ordered
    assert _simplify_captures(lhs) == ["a", {"b": ("c",)}, {"d": ("e", "f")}]


def test_parse_unordered():
    lhs, rhs = parse_rearrangement("{a b c d} -> {b c a d}")
    assert not lhs.is_ordered
    assert _simplify_captures(lhs) == ["a", "b", "c", "d"]
    assert not rhs.is_ordered
    assert _simplify_captures(rhs) == ["b", "c", "a", "d"]

    lhs, rhs = parse_rearrangement("{(c: a b) d e} -> (q: a d e) b")
    assert not lhs.is_ordered
    assert _simplify_captures(lhs) == [{"c": ("a", "b")}, "d", "e"]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == [{"q": ("a", "d", "e")}, "b"]


def test_parse_quoted_identifiers():
    lhs, rhs = parse_rearrangement("a \"b c\" d -> 'b c' a d")
    assert lhs.is_ordered
    assert _simplify_captures(lhs) == ["a", "b c", "d"]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == ["b c", "a", "d"]

    lhs, rhs = parse_rearrangement("{a \"b c\" (d: 'hello')} -> b c a d")
    assert not lhs.is_ordered
    assert _simplify_captures(lhs) == ["a", "b c", {"d": ("hello",)}]
    assert rhs.is_ordered
    assert _simplify_captures(rhs) == ["b", "c", "a", "d"]


def test_parse_errors():
    with pytest.raises(ValueError, match="Unexpected end of string"):
        parse_rearrangement("a b")

    with pytest.raises(ValueError, match="Expected }"):
        parse_rearrangement("{ a -> c")

    with pytest.raises(ValueError, match="Unexpected }"):
        parse_rearrangement("a } -> c")

    with pytest.raises(ValueError, match="Unexpected }"):
        parse_rearrangement("(a: b } -> c")

    with pytest.raises(ValueError, match="Unexpected character"):
        parse_rearrangement("(a b ! -> c d e")

    with pytest.raises(ValueError, match="Identifier expected"):
        parse_rearrangement("a b ! -> c d e")
