# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import dataclasses
from types import EllipsisType
from typing import Mapping, NoReturn, Sequence

from haliax.axis import Axis, AxisSelector


@dataclasses.dataclass(frozen=True)
class _AxisCapture:
    binding: str | None = None
    axes: tuple[str, ...] = ()
    char_range: tuple[int, int] | None = None

    def __post_init__(self):
        if len(self.axes) == 0:
            raise ValueError("Empty axes not allowed")


@dataclasses.dataclass(frozen=True)
class Expression:
    captures: Sequence[_AxisCapture | EllipsisType]
    is_ordered: bool


def raise_parse_error(message: str, expression: str, pos: int | tuple[int, int] | None) -> NoReturn:
    """Raise a ValueError with a message and the position in the expression."""
    fmt = f"Error while parsing:\n    {expression}"
    if pos is not None:
        if isinstance(pos, int):
            fmt += f'\n    {" " * pos}^'
        else:
            fmt += f"\n    {' ' * pos[0]}{'^' * max(1, pos[1] - pos[0])}"

    fmt += f"\n{message}"

    raise ValueError(fmt)


def _parse_quoted_string(expression: str, pos: int) -> tuple[str, int]:
    """Parse a quoted string from an einops-style haliax rearrangement string."""

    if expression[pos] not in "'\"":
        raise_parse_error(f"Expected \" or ' at position {pos}", expression, pos)
    quote = expression[pos]
    pos += 1
    ident = ""
    while pos < len(expression):
        if expression[pos] == quote:
            pos += 1
            break
        elif expression[pos] == "\\":
            pos += 1
            if pos >= len(expression):
                raise_parse_error(f"Unexpected end of string at position {pos}", expression, pos)
            ident += expression[pos]
            pos += 1
            continue
        else:
            ident += expression[pos]
            pos += 1
            continue
    if len(ident) == 0:
        raise_parse_error("Empty strings are not valid identifiers", expression, pos)

    return ident, pos


def _parse_ident(expression: str, pos: int) -> tuple[str, int]:
    """parses an identifier or string literal from an einops-style haliax rearrangement string."""
    if expression[pos] in "'\"":
        return _parse_quoted_string(expression, pos)
    else:
        ident = ""
        while pos < len(expression):
            if str.isalnum(expression[pos]) or expression[pos] == "_":
                if len(ident) == 0 and str.isdigit(expression[pos]):
                    raise_parse_error("Identifiers cannot start with a number", expression, pos)
                ident += expression[pos]
                pos += 1
                continue
            else:
                break
        if len(ident) == 0:
            raise_parse_error("Identifier expected", expression, pos)

        return ident, pos


def _parse_group(expression, pos):
    # parses a group of axes like (a b c) or (a: b c)
    pos_in = pos
    if expression[pos] != "(":
        raise ValueError("Expected (")
    pos += 1
    binding = None
    axes: list[str] = []
    current_ident = ""
    while pos < len(expression):
        if expression[pos] == ")":
            pos += 1
            break
        elif expression[pos] == ":":
            if binding is not None:
                raise_parse_error("Only one binding allowed per group", expression, pos)
            if not current_ident:
                raise_parse_error("Binding cannot be empty", expression, pos)
            if len(axes) > 0:
                raise_parse_error("Binding must come before axes", expression, pos)
            binding = current_ident
            current_ident = ""
            pos += 1
            continue
        elif str.isspace(expression[pos]):
            if current_ident:
                axes.append(current_ident)
                current_ident = ""
            pos += 1
            continue
        elif expression[pos] == "(":
            raise_parse_error("Only one level of nesting is allowed", expression, pos)
        elif expression[pos] == "}":
            raise ValueError(f"Unexpected }} at {pos}")
        elif str.isalnum(expression[pos]) or expression[pos] == "_":
            # don't allow numbers at the start of an identifier
            if len(current_ident) == 0 and str.isdigit(expression[pos]):
                raise_parse_error("Identifiers cannot start with a number", expression, pos)
            current_ident += expression[pos]
            pos += 1
            continue
        elif expression[pos] in "'\"":
            # parse quoted string as identifier
            if current_ident:
                axes.append(current_ident)

            ident, pos = _parse_quoted_string(expression, pos)
            current_ident = ident
            continue
        else:
            raise_parse_error(f"Unexpected character {expression[pos]}", expression, pos)

    if current_ident:
        axes.append(current_ident)

    if len(axes) == 0:
        raise_parse_error("No axes found", expression, pos_in)

    # todo: should we allow anonymous/literal
    char_range = (pos_in, pos)
    return _AxisCapture(binding, tuple(axes), char_range), pos


def _parse_expression(expression: str, pos) -> tuple[Expression, int]:
    """Parse one side of an einops-style haliax rearrangement string."""
    captures = []
    is_ordered = True
    seen_char = False
    finished = False

    while pos < len(expression):
        if expression[pos] == "{":
            if seen_char:
                raise_parse_error("Unexpected {", expression, pos)
            seen_char = True
            is_ordered = False
            pos += 1
            continue
        elif expression[pos] == "}":
            if is_ordered:
                raise_parse_error("Unexpected }", expression, pos)
            pos += 1
            finished = True
            continue
        elif expression[pos] == "(":
            if finished:
                raise_parse_error("Unexpected ( after }", expression, pos)
            seen_char = True
            capture, pos = _parse_group(expression, pos)
            captures.append(capture)
            continue
        elif str.isspace(expression[pos]):
            pos += 1
            continue
        elif expression[pos : pos + 3] == "...":
            seen_char = True
            if finished:
                raise_parse_error("Unexpected ... after }", expression, pos)
            captures.append(Ellipsis)
            pos += 3
            continue
        elif expression[pos] == "-":
            # if not seen_char:
            #     _raise_error("Unexpected -", expression, pos)
            if pos + 1 >= len(expression):
                raise_parse_error("Unexpected end of string", expression, pos)
            if expression[pos + 1] != ">":
                raise_parse_error("Expected >", expression, pos)
            break
        elif expression[pos] == ",":
            # this ends the current expression if we're not inside a group or unordered capture
            if not is_ordered:
                # treat as whitespace
                pos += 1
                continue
            if finished:
                break

            break
        else:
            if finished:
                raise_parse_error("Unexpected character after }", expression, pos)
            ident, new_pos = _parse_ident(expression, pos)
            captures.append(_AxisCapture(binding=ident, axes=(ident,), char_range=(pos, new_pos)))
            seen_char = True
            pos = new_pos
            continue

    if not finished and not is_ordered:
        raise_parse_error("Expected }", expression, pos)

    return Expression(captures, is_ordered), pos


class AliasTable:
    bindings: dict[str, AxisSelector]  # names in the string to axes

    def __init__(self, bindings=None):
        if bindings is None:
            self.bindings = {}
        else:
            self.bindings = {**bindings}

    def dealias_binding(self, binding: str) -> AxisSelector | None:
        return self.bindings.get(binding, None)

    def bind_alias(self, alias: str, axis: Axis, expr, char_range):
        if axis.name in self.bindings:
            if self.bindings[alias] != axis:
                raise_parse_error(f"Alias {alias} is assigned to more than one axis", expr, char_range)
        elif alias in self.bindings:
            current = self.bindings[alias]
            if isinstance(current, Axis):
                if current != axis:
                    raise_parse_error(f"Alias {alias} is assigned to more than one axis", expr, char_range)
            elif current != axis.name:
                raise_parse_error(f"Alias {alias} is assigned to more than one axis", expr, char_range)
            else:
                self.bindings[alias] = axis
        else:
            self.bindings[alias] = axis


def _resolve_bindings(array, bindings: Mapping[str, Axis | str | int]) -> AliasTable:
    b: dict[str, AxisSelector] = {}
    for name, selector in bindings.items():
        if isinstance(selector, str):
            try:
                selector = array.resolve_axis(selector)
            except ValueError:
                pass
        elif isinstance(selector, int):
            selector = Axis(name, selector)
        assert not isinstance(selector, int)
        b[name] = selector
    return AliasTable(b)


def parse_rearrangement(expression: str) -> tuple[Expression, Expression]:
    """Parse an einops-style haliax rearrangement string."""
    pos = 0
    lhs, pos = _parse_expression(expression, pos)

    # consume the ->
    if pos + 2 >= len(expression):
        raise_parse_error("Unexpected end of string", expression, pos)
    if expression[pos : pos + 2] != "->":
        raise_parse_error("Expected ->", expression, pos)

    pos += 2
    rhs, pos = _parse_expression(expression, pos)

    # make sure we consumed the whole string
    if pos != len(expression):
        raise_parse_error("Unexpected character", expression, pos)

    return lhs, rhs


def parse_einsum(expression: str) -> tuple[Sequence[Expression], Expression]:
    """
    Parse a haliax-style einsum string. This is an expansion of the einsum syntax
    to support named axes
    """
    pos = 0
    lhses = []
    while pos < len(expression):
        lhs, pos = _parse_expression(expression, pos)
        lhses.append(lhs)
        if pos >= len(expression):
            break
        if expression[pos] == ",":
            pos += 1
            continue
        elif expression[pos] == "-":
            if pos + 1 >= len(expression):
                raise_parse_error("Unexpected end of string", expression, pos)
            if expression[pos + 1] != ">":
                raise_parse_error("Expected >", expression, pos)
            pos += 2
            break
        else:
            raise_parse_error("Expected , or ->", expression, pos)

    rhs, pos = _parse_expression(expression, pos)

    if any(x is None for x in lhses):
        if len(lhses) > 1:
            raise_parse_error("If there are multiple lhs expressions, they must all be ordered", expression, pos)
        lhses = None  # type: ignore

    if pos != len(expression):
        raise_parse_error("Unexpected character", expression, pos)

    return lhses, rhs
