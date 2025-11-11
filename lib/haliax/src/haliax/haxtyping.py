# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


@dataclass(frozen=True)
class DTypeCategory:
    """Represents a dtype category such as ``float`` or ``int``."""

    name: str
    category: Any

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return self.name


if TYPE_CHECKING:
    # ── STATIC ONLY: re‑export jaxtyping’s aliases so mypy/Pyright/PyCharm see them
    from jaxtyping import (
        Float32 as f32,
        Float64 as f64,
        Float16 as f16,
        BFloat16 as bf16,
        Int8 as i8,
        Int16 as i16,
        Int32 as i32,
        Int64 as i64,
        UInt8 as u8,
        UInt16 as u16,
        UInt32 as u32,
        UInt64 as u64,
        Bool as bool_,
        Complex64 as complex64,
        Complex128 as complex128,
        Float as Float,
        Int as Int,
        UInt as UInt,
    )

    # axes‑only helper
    from typing import Annotated as Named

else:
    # ── RUNTIME: custom wrappers for NamedArray, plus delegation to jaxtyping ──
    import jaxtyping as jt
    import jax.numpy as jnp
    from typing import Annotated
    from dataclasses import dataclass, replace

    from .core import (
        NamedArray,
        NamedArrayAxes,
        NamedArrayAxesSpec,
        _parse_namedarray_axes,
    )

    def _with_dtype(axes: NamedArrayAxes, dtype):
        """Attach dtype to axes metadata if not already set."""
        return axes if axes.dtype is not None else replace(axes, dtype=dtype)

    def _make_dtype_wrapper(dtype):
        """Factory for f32, i32, etc."""

        class _Wrapper:
            def __class_getitem__(cls, item):
                # two‑arg form: (BaseType, axes_spec)
                if isinstance(item, tuple) and len(item) == 2:
                    base, axes_spec = item
                else:
                    base, axes_spec = NamedArray, item

                # Delegate non‑NamedArray to jaxtyping
                if base is not NamedArray:
                    # e.g. use jt.Float32 for jnp.float32
                    jaxt = getattr(jt, f"Float{dtype.itemsize*8}") if hasattr(dtype, "itemsize") else jt.Float
                    return jaxt[base, axes_spec]

                # Handle NamedArray path
                axes = _parse_namedarray_axes(axes_spec)
                return Annotated[NamedArray, _with_dtype(axes, dtype)]

        return _Wrapper

    # ── Build all dtype wrappers ─────────────────────────────────────────────
    f32 = _make_dtype_wrapper(jnp.float32)
    f64 = _make_dtype_wrapper(jnp.float64)
    f16 = _make_dtype_wrapper(jnp.float16)
    bf16 = _make_dtype_wrapper(jnp.bfloat16)

    i8 = _make_dtype_wrapper(jnp.int8)
    i16 = _make_dtype_wrapper(jnp.int16)
    i32 = _make_dtype_wrapper(jnp.int32)
    i64 = _make_dtype_wrapper(jnp.int64)

    u8 = _make_dtype_wrapper(jnp.uint8)
    u16 = _make_dtype_wrapper(jnp.uint16)
    u32 = _make_dtype_wrapper(jnp.uint32)
    u64 = _make_dtype_wrapper(jnp.uint64)

    bool_ = _make_dtype_wrapper(jnp.bool_)
    complex64 = _make_dtype_wrapper(jnp.complex64)
    complex128 = _make_dtype_wrapper(jnp.complex128)

    def _make_category_wrapper(name: str, category):
        """Like _make_dtype_wrapper but matches any dtype in the JAX category."""

        class _Wrapper:
            def __class_getitem__(cls, item):
                # same base/axes unpack logic
                if isinstance(item, tuple) and len(item) == 2:
                    base, axes_spec = item
                else:
                    base, axes_spec = NamedArray, item

                # non‑NamedArray → delegate to jaxtyping’s category wrapper
                if base is not NamedArray:
                    return getattr(jt, name)[base, axes_spec]

                # NamedArray path
                axes = _parse_namedarray_axes(axes_spec)
                cat = DTypeCategory(name, category)
                return Annotated[NamedArray, _with_dtype(axes, cat)]

        return _Wrapper

    # Build the category wrappers
    Float = _make_category_wrapper("float", jnp.floating)
    Complex = _make_category_wrapper("complex", jnp.complexfloating)
    Int = _make_category_wrapper("int", jnp.signedinteger)
    UInt = _make_category_wrapper("uInt", jnp.unsignedinteger)

    # ── Named: axes‑only helper ───────────────────────────────────────────────
    class _NamedHelper:
        @classmethod
        def __class_getitem__(self, axes_spec_: tuple[type[NamedArray], NamedArrayAxesSpec]):
            _, axes_spec = axes_spec_
            axes = _parse_namedarray_axes(axes_spec)
            return Annotated[NamedArray, axes]

    Named = _NamedHelper

    __all__ = [
        "Named",
        "f32",
        "f64",
        "f16",
        "bf16",
        "i8",
        "i16",
        "i32",
        "i64",
        "u8",
        "u16",
        "u32",
        "u64",
        "bool_",
        "complex64",
        "complex128",
    ]
