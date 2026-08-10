# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate and audit the ordinary-JAX StableHLO fixtures in this directory."""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
from jax._src.interpreters.mlir import make_ir_context
from jaxlib.mlir import ir, passmanager
from jaxlib.mlir.dialects import stablehlo

JAX_VERSION = "0.10.1"
JAXLIB_VERSION = "0.10.1"
PINNED_XLA_REVISION = "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69"
FINGERPRINT_PATTERN = re.compile(r"(?m)^([0-9A-Fa-f]{64})$")
NORMALIZER_DIAGNOSTIC_FIELD_LIMIT = 1_024
NORMALIZER_DIAGNOSTIC_MESSAGE_LIMIT = 4_096
ACCEPTANCE_FIXTURE_FILENAMES = frozenset(
    {
        "jax-0.10.1-tanh-dot-forward.mlir",
        "jax-0.10.1-tanh-dot-vjp.mlir",
    }
)


def reference_function(x, w0, w1):
    return jnp.tanh(x @ w0) @ w1


def reference_vjp(x, w0, w1, output_cotangent):
    _, pullback = jax.vjp(reference_function, x, w0, w1)
    return pullback(output_cotangent)


def map_only(a, b, c):
    return jnp.transpose((a * b) + c)


def contract(a, b):
    return a @ b


@dataclass(frozen=True)
class Fixture:
    filename: str
    function: Callable[..., object]
    shapes: tuple[tuple[int, ...], ...]
    expression: str


FIXTURES = (
    Fixture(
        "jax-0.10.1-tanh-dot-forward.mlir",
        reference_function,
        ((2, 3), (3, 4), (4, 5)),
        "reference_function(x, w0, w1) = tanh(x @ w0) @ w1",
    ),
    Fixture(
        "jax-0.10.1-tanh-dot-vjp.mlir",
        reference_vjp,
        ((2, 3), (3, 4), (4, 5), (2, 5)),
        "jax.vjp(reference_function, x, w0, w1)[1](output_cotangent)",
    ),
    Fixture(
        "jax-0.10.1-tanh-dot-forward-alt.mlir",
        reference_function,
        ((3, 2), (2, 6), (6, 4)),
        "reference_function(x, w0, w1) = tanh(x @ w0) @ w1",
    ),
    Fixture(
        "jax-0.10.1-tanh-dot-vjp-alt.mlir",
        reference_vjp,
        ((3, 2), (2, 6), (6, 4), (3, 4)),
        "jax.vjp(reference_function, x, w0, w1)[1](output_cotangent)",
    ),
    Fixture(
        "jax-0.10.1-map-only.mlir",
        map_only,
        ((2, 3), (2, 3), (2, 3)),
        "transpose((a * b) + c)",
    ),
    Fixture(
        "jax-0.10.1-contract-only.mlir",
        contract,
        ((3, 2), (2, 4)),
        "a @ b",
    ),
)


def export_stablehlo(fixture: Fixture) -> str:
    arguments = tuple(jax.ShapeDtypeStruct(shape, jnp.float32) for shape in fixture.shapes)
    module = jax.jit(fixture.function).lower(*arguments).compiler_ir(dialect="stablehlo")
    return f"{module}".rstrip() + "\n"


def normalized_fingerprint(payload: str, normalizer: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="shuttle-fixture-audit-") as directory:
        fixture_path = Path(directory) / "fixture.mlir"
        fixture_path.write_text(payload)
        command = [
            str(normalizer),
            "--shuttle-test-report-normalized-fingerprint",
            str(fixture_path),
        ]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            message = (
                "normalizer subprocess failed: "
                f"exit_code={error.returncode}; argv={_bounded_serialized(command)}; "
                f"stdout={_bounded_serialized(error.stdout or '')}; "
                f"stderr={_bounded_serialized(error.stderr or '')}"
            )
            raise RuntimeError(_bounded_message(message)) from error
    match = FINGERPRINT_PATTERN.search(result.stdout)
    if not match:
        message = (
            "normalizer did not report a structural SHA-256: "
            f"argv={_bounded_serialized(command)}; stdout={_bounded_serialized(result.stdout)}"
        )
        raise RuntimeError(_bounded_message(message))
    return match.group(1).upper()


def _bounded_serialized(value: object) -> str:
    serialized = json.dumps(value)
    if len(serialized) <= NORMALIZER_DIAGNOSTIC_FIELD_LIMIT:
        return serialized
    suffix = "...<serialized field truncated>"
    retained = NORMALIZER_DIAGNOSTIC_FIELD_LIMIT - len(suffix)
    return serialized[:retained] + suffix


def _bounded_message(message: str) -> str:
    if len(message) <= NORMALIZER_DIAGNOSTIC_MESSAGE_LIMIT:
        return message
    suffix = "...<diagnostic truncated>"
    retained = NORMALIZER_DIAGNOSTIC_MESSAGE_LIMIT - len(suffix)
    return message[:retained] + suffix


def xla_hook_boundary_stablehlo(payload: str) -> str:
    """Apply the only pinned XLA pre-hook pass that changes these fixtures."""
    stablehlo.register_stablehlo_passes()
    with make_ir_context():
        module = ir.Module.parse(payload)
        pipeline = passmanager.PassManager.parse("builtin.module(func.func(stablehlo-complex-math-expander))")
        pipeline.run(module.operation)
        return f"{module}".rstrip() + "\n"


def audited_fixture(fixture: Fixture, normalizer: Path) -> str:
    payload = export_stablehlo(fixture)
    raw_digest = hashlib.sha256(payload.encode()).hexdigest().upper()
    normalized_digest = normalized_fingerprint(payload, normalizer)
    hook_boundary_header = ()
    if fixture.filename in ACCEPTANCE_FIXTURE_FILENAMES:
        hook_boundary_payload = xla_hook_boundary_stablehlo(payload)
        hook_boundary_digest = hashlib.sha256(hook_boundary_payload.encode()).hexdigest().upper()
        hook_boundary_normalized_digest = normalized_fingerprint(hook_boundary_payload, normalizer)
        hook_boundary_header = (
            "// XLA hook-boundary preprocessing: stablehlo-complex-math-expander",
            f"// XLA hook-boundary StableHLO SHA-256: {hook_boundary_digest}",
            f"// XLA hook-boundary normalized StableHLO SHA-256: {hook_boundary_normalized_digest}",
        )
    shapes = ", ".join(f"{shape}:f32" for shape in fixture.shapes)
    header_lines = (
        "// Copyright The Marin Authors",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        "// Ordinary-JAX export audit",
        f"// Generator: {Path(__file__).name} ({fixture.filename})",
        '// Export: jax.jit(fixture.function).lower(*f32_shape_structs).compiler_ir(dialect="stablehlo")',
        f"// Expression: {fixture.expression}",
        f"// Inputs: {shapes}",
        f"// JAX: {JAX_VERSION}; jaxlib: {JAXLIB_VERSION}; XLA: {PINNED_XLA_REVISION}",
        f"// Raw StableHLO SHA-256: {raw_digest}",
        f"// Normalized StableHLO SHA-256: {normalized_digest}",
        *hook_boundary_header,
    )
    header = "\n".join(header_lines) + "\n\n"
    return header + payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalizer", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    if jax.__version__ != JAX_VERSION or jaxlib.__version__ != JAXLIB_VERSION:
        raise RuntimeError(
            f"fixture audit requires JAX {JAX_VERSION} and jaxlib {JAXLIB_VERSION}; "
            f"found {jax.__version__} and {jaxlib.__version__}"
        )
    mismatches = []
    for fixture in FIXTURES:
        expected = audited_fixture(fixture, arguments.normalizer)
        path = arguments.output_dir / fixture.filename
        if arguments.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected)
        elif not path.exists() or path.read_text() != expected:
            mismatches.append(fixture.filename)
    if mismatches:
        parser.error("fixture audit failed: " + ", ".join(mismatches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
