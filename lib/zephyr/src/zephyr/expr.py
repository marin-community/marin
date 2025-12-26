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

"""Expression layer for filter and projection pushdown.

Modeled on Vortex expressions: https://docs.vortex.dev/api/python/expr
"""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pyarrow.compute as pc


class Expr(ABC):
    """Base class for expressions.

    Expressions represent computations over record fields that can be:
    - Evaluated directly against Python dicts
    - Converted to PyArrow expressions for filter pushdown

    Example:
        >>> from zephyr.expr import col, lit
        >>> expr = (col("score") > 0.5) & (col("category") == "A")
        >>> expr.evaluate({"score": 0.8, "category": "A"})
        True
    """

    @abstractmethod
    def evaluate(self, record: dict) -> Any:
        """Evaluate expression against a record."""
        pass

    def __hash__(self) -> int:
        """Hash based on repr for use in sets/dicts."""
        return hash(repr(self))

    # Comparison operators
    def __eq__(self, other: object) -> CompareExpr:  # type: ignore[override]
        return CompareExpr(self, _to_expr(other), "eq")

    def __ne__(self, other: object) -> CompareExpr:  # type: ignore[override]
        return CompareExpr(self, _to_expr(other), "ne")

    def __lt__(self, other: object) -> CompareExpr:
        return CompareExpr(self, _to_expr(other), "lt")

    def __le__(self, other: object) -> CompareExpr:
        return CompareExpr(self, _to_expr(other), "le")

    def __gt__(self, other: object) -> CompareExpr:
        return CompareExpr(self, _to_expr(other), "gt")

    def __ge__(self, other: object) -> CompareExpr:
        return CompareExpr(self, _to_expr(other), "ge")

    # Arithmetic operators
    def __add__(self, other: object) -> ArithmeticExpr:
        return ArithmeticExpr(self, _to_expr(other), "add")

    def __sub__(self, other: object) -> ArithmeticExpr:
        return ArithmeticExpr(self, _to_expr(other), "sub")

    def __mul__(self, other: object) -> ArithmeticExpr:
        return ArithmeticExpr(self, _to_expr(other), "mul")

    def __truediv__(self, other: object) -> ArithmeticExpr:
        return ArithmeticExpr(self, _to_expr(other), "truediv")

    # Logical operators (use & and | since and/or can't be overloaded)
    def __and__(self, other: object) -> LogicalExpr:
        return LogicalExpr(self, _to_expr(other), "and")

    def __or__(self, other: object) -> LogicalExpr:
        return LogicalExpr(self, _to_expr(other), "or")

    def __invert__(self) -> NotExpr:
        return NotExpr(self)

    # Field access for nested structs
    def __getitem__(self, key: str) -> FieldAccessExpr:
        return FieldAccessExpr(self, key)

    # Null checks
    def is_null(self) -> IsNullExpr:
        return IsNullExpr(self)

    def is_not_null(self) -> NotExpr:
        return NotExpr(IsNullExpr(self))


def _to_expr(value: Any) -> Expr:
    """Convert a value to an expression if not already one."""
    if isinstance(value, Expr):
        return value
    return LiteralExpr(value)


@dataclass(eq=False)
class ColumnExpr(Expr):
    """Reference to a column by name."""

    name: str

    def evaluate(self, record: dict) -> Any:
        return record.get(self.name)

    def __repr__(self) -> str:
        return f"col({self.name!r})"


@dataclass(eq=False)
class LiteralExpr(Expr):
    """A literal constant value."""

    value: Any

    def evaluate(self, record: dict) -> Any:
        return self.value

    def __repr__(self) -> str:
        return f"lit({self.value!r})"


_COMPARE_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}

_COMPARE_SYMBOLS: dict[str, str] = {
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
}


@dataclass(eq=False)
class CompareExpr(Expr):
    """Comparison between two expressions."""

    left: Expr
    right: Expr
    op: Literal["eq", "ne", "lt", "le", "gt", "ge"]

    def evaluate(self, record: dict) -> bool:
        left_val = self.left.evaluate(record)
        right_val = self.right.evaluate(record)
        return _COMPARE_OPS[self.op](left_val, right_val)

    def __repr__(self) -> str:
        return f"({self.left} {_COMPARE_SYMBOLS[self.op]} {self.right})"


@dataclass(eq=False)
class LogicalExpr(Expr):
    """Logical AND/OR between expressions."""

    left: Expr
    right: Expr
    op: Literal["and", "or"]

    def evaluate(self, record: dict) -> bool:
        left_val = self.left.evaluate(record)
        if self.op == "and":
            return bool(left_val and self.right.evaluate(record))
        return bool(left_val or self.right.evaluate(record))

    def __repr__(self) -> str:
        op_symbol = "&" if self.op == "and" else "|"
        return f"({self.left} {op_symbol} {self.right})"


@dataclass(eq=False)
class NotExpr(Expr):
    """Logical NOT."""

    child: Expr

    def evaluate(self, record: dict) -> bool:
        return not self.child.evaluate(record)

    def __repr__(self) -> str:
        return f"~({self.child})"


@dataclass(eq=False)
class IsNullExpr(Expr):
    """Check if value is null/None."""

    child: Expr

    def evaluate(self, record: dict) -> bool:
        return self.child.evaluate(record) is None

    def __repr__(self) -> str:
        return f"{self.child}.is_null()"


_ARITHMETIC_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "truediv": operator.truediv,
}

_ARITHMETIC_SYMBOLS: dict[str, str] = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "truediv": "/",
}


@dataclass(eq=False)
class ArithmeticExpr(Expr):
    """Arithmetic operation between expressions."""

    left: Expr
    right: Expr
    op: Literal["add", "sub", "mul", "truediv"]

    def evaluate(self, record: dict) -> Any:
        return _ARITHMETIC_OPS[self.op](self.left.evaluate(record), self.right.evaluate(record))

    def __repr__(self) -> str:
        return f"({self.left} {_ARITHMETIC_SYMBOLS[self.op]} {self.right})"


@dataclass(eq=False)
class FieldAccessExpr(Expr):
    """Access a field of a nested struct."""

    parent: Expr
    field: str

    def evaluate(self, record: dict) -> Any:
        parent_val = self.parent.evaluate(record)
        if isinstance(parent_val, dict):
            return parent_val.get(self.field)
        return None

    def __repr__(self) -> str:
        return f"{self.parent}[{self.field!r}]"


# Convenience functions
def col(name: str) -> ColumnExpr:
    """Create a column reference expression.

    Example:
        >>> col("score") > 0.5
        (col('score') > lit(0.5))
    """
    return ColumnExpr(name)


def lit(value: Any) -> LiteralExpr:
    """Create a literal value expression.

    Example:
        >>> col("a") + lit(10)
        (col('a') + lit(10))
    """
    return LiteralExpr(value)


def to_pyarrow_expr(expr: Expr) -> pc.Expression:
    """Convert a Zephyr expression to a PyArrow compute expression.

    This enables filter pushdown to Parquet readers.

    Example:
        >>> pa_expr = to_pyarrow_expr(col("score") > 0.5)
        >>> # Can be used with pq.read_table(..., filter=pa_expr)
    """
    import pyarrow.compute as pc

    if isinstance(expr, ColumnExpr):
        return pc.field(expr.name)
    elif isinstance(expr, LiteralExpr):
        return pc.scalar(expr.value)
    elif isinstance(expr, CompareExpr):
        left = to_pyarrow_expr(expr.left)
        right = to_pyarrow_expr(expr.right)
        ops = {
            "eq": pc.equal,
            "ne": pc.not_equal,
            "lt": pc.less,
            "le": pc.less_equal,
            "gt": pc.greater,
            "ge": pc.greater_equal,
        }
        return ops[expr.op](left, right)
    elif isinstance(expr, LogicalExpr):
        left = to_pyarrow_expr(expr.left)
        right = to_pyarrow_expr(expr.right)
        if expr.op == "and":
            return pc.and_(left, right)
        return pc.or_(left, right)
    elif isinstance(expr, NotExpr):
        return pc.invert(to_pyarrow_expr(expr.child))
    elif isinstance(expr, IsNullExpr):
        return pc.is_null(to_pyarrow_expr(expr.child))
    elif isinstance(expr, ArithmeticExpr):
        left = to_pyarrow_expr(expr.left)
        right = to_pyarrow_expr(expr.right)
        ops = {
            "add": pc.add,
            "sub": pc.subtract,
            "mul": pc.multiply,
            "truediv": pc.divide,
        }
        return ops[expr.op](left, right)
    elif isinstance(expr, FieldAccessExpr):
        # PyArrow uses struct_field for nested access
        parent = to_pyarrow_expr(expr.parent)
        return pc.struct_field(parent, expr.field)
    else:
        raise ValueError(f"Cannot convert {type(expr).__name__} to PyArrow expression")
