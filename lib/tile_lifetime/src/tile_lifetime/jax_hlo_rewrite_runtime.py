# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX runtime boundary for pre-scheduler HLO rewrites."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib


class HloRewriteRuntimeUnavailable(RuntimeError):
    """Raised when the installed JAX runtime cannot host an HLO rewrite."""


@dataclass(frozen=True)
class HloRewriteRuntimeAudit:
    """Availability evidence for the compiler-IR, parser, and pass hook."""

    jax_version: str
    jaxlib_version: str
    compiler_ir_module_type: str | None
    compiler_ir_proto_roundtrip: bool
    text_parser_backend: str | None
    transformation_api: str | None
    unavailable_reasons: tuple[str, ...]

    @property
    def available(self) -> bool:
        """Whether all capabilities needed by a text HLO rewrite are present."""
        return not self.unavailable_reasons


@dataclass(frozen=True)
class HloRewriteRuntime:
    """Resolved runtime handles for one pre-scheduler HLO rewrite."""

    hlo_module_type: type[Any]
    parse_hlo_text: Callable[[str], Any]
    text_parser_backend: str
    transformation_api: ModuleType

    def module_from_serialized_proto(self, serialized_module: bytes) -> Any:
        """Import a callback proto through the public compiler-IR module type."""
        return self.hlo_module_type.from_serialized_hlo_module_proto(serialized_module)

    def module_from_text(self, hlo_text: str) -> Any:
        """Parse rewritten text through the isolated XLA compatibility boundary."""
        return self.parse_hlo_text(hlo_text)


@cache
def _public_compiler_ir_hlo_module_type() -> type[Any]:
    lowered = jax.jit(lambda value: value).lower(jax.ShapeDtypeStruct((), jnp.float32))
    compiler_ir = lowered.compiler_ir(dialect="hlo")
    module = compiler_ir.as_hlo_module()
    module_type = type(module)
    serialized = module.as_serialized_hlo_module_proto()
    restored = module_type.from_serialized_hlo_module_proto(serialized)
    if restored.name != module.name:
        raise HloRewriteRuntimeUnavailable("public JAX compiler-IR HLO proto roundtrip changed the module identity")
    return module_type


def _resolve_hlo_text_parser() -> tuple[Callable[[str], Any], str]:
    failures: list[str] = []
    for module_name in ("jaxlib._jax", "jaxlib._hlo"):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            failures.append(f"{module_name}: {error}")
            continue
        parser = getattr(module, "hlo_module_from_text", None)
        if parser is not None:
            return parser, f"{module_name}.hlo_module_from_text"
        failures.append(f"{module_name}: hlo_module_from_text is absent")
    raise HloRewriteRuntimeUnavailable(
        "JAX exposes no public HLO text parser; compatible private parser unavailable (" + "; ".join(failures) + ")"
    )


def _resolve_transformation_api() -> ModuleType:
    try:
        transformation_api = importlib.import_module("jax.extend.xla")
    except ModuleNotFoundError as error:
        raise HloRewriteRuntimeUnavailable(
            "jax.extend.xla is unavailable; pre-scheduler HLO transformation registration "
            "requires a JAX/JAXLIB build that provides register_hlo_module_transformation"
        ) from error
    required = ("register_hlo_module_transformation", "clear_hlo_module_transformation")
    missing = tuple(name for name in required if not hasattr(transformation_api, name))
    if missing:
        raise HloRewriteRuntimeUnavailable(f"jax.extend.xla lacks required HLO transformation entry points: {missing}")
    return transformation_api


def audit_hlo_rewrite_runtime() -> HloRewriteRuntimeAudit:
    """Inspect the installed JAX runtime without requiring a GPU."""
    reasons: list[str] = []
    module_type_name: str | None = None
    compiler_ir_proto_roundtrip = False
    try:
        module_type = _public_compiler_ir_hlo_module_type()
        module_type_name = f"{module_type.__module__}.{module_type.__qualname__}"
        compiler_ir_proto_roundtrip = True
    except (AttributeError, HloRewriteRuntimeUnavailable, RuntimeError, TypeError, ValueError) as error:
        reasons.append(f"compiler_ir: {error}")

    text_parser_backend: str | None = None
    try:
        _, text_parser_backend = _resolve_hlo_text_parser()
    except HloRewriteRuntimeUnavailable as error:
        reasons.append(f"text_parser: {error}")

    transformation_api_name: str | None = None
    try:
        transformation_api = _resolve_transformation_api()
        transformation_api_name = transformation_api.__name__
    except HloRewriteRuntimeUnavailable as error:
        reasons.append(f"transformation_api: {error}")

    return HloRewriteRuntimeAudit(
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__,
        compiler_ir_module_type=module_type_name,
        compiler_ir_proto_roundtrip=compiler_ir_proto_roundtrip,
        text_parser_backend=text_parser_backend,
        transformation_api=transformation_api_name,
        unavailable_reasons=tuple(reasons),
    )


def require_hlo_rewrite_runtime() -> HloRewriteRuntime:
    """Resolve the complete HLO rewrite runtime or fail before GPU allocation."""
    audit = audit_hlo_rewrite_runtime()
    if not audit.available:
        detail = "; ".join(audit.unavailable_reasons)
        raise HloRewriteRuntimeUnavailable(
            f"JAX HLO rewrite runtime is unavailable for JAX {audit.jax_version} / "
            f"JAXLIB {audit.jaxlib_version}: {detail}"
        )
    parser, parser_backend = _resolve_hlo_text_parser()
    return HloRewriteRuntime(
        hlo_module_type=_public_compiler_ir_hlo_module_type(),
        parse_hlo_text=parser,
        text_parser_backend=parser_backend,
        transformation_api=_resolve_transformation_api(),
    )
