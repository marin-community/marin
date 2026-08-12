# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate ordinary-JAX BF16 composed primal and VJP fixtures."""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
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
HOOK_PIPELINE = "builtin.module(func.func(stablehlo-complex-math-expander))"
FINGERPRINT_PATTERN = re.compile(r"(?m)^([0-9A-Fa-f]{64})$")


@dataclass(frozen=True)
class StructuralCase:
    rows: int
    reduction: int
    features: int
    scalar_map: str

    @property
    def case_id(self) -> str:
        record = {
            "features": self.features,
            "reduction": self.reduction,
            "rows": self.rows,
            "scalar_map": self.scalar_map,
        }
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
        return "contract_map_" + hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def filename(self) -> str:
        return f"jax-{JAX_VERSION}-bf16-{self.case_id}-primal-vjp.mlir"

    @property
    def input_shapes(self) -> tuple[tuple[int, int], ...]:
        return (
            (self.rows, self.reduction),
            (self.reduction, self.features),
            (self.features, self.reduction),
            (self.rows, self.reduction),
        )


CASES = (
    StructuralCase(43, 104, 72, "sigmoid_product"),
    StructuralCase(131, 168, 104, "tanh_product"),
    StructuralCase(269, 232, 136, "cubic_mix"),
    StructuralCase(521, 328, 184, "sigmoid_product"),
)


def natural_jax_composed_primal_and_vjp(case: StructuralCase):
    """Return natural JAX source with a JAX-owned VJP for one structural case."""

    def composed_primal_and_vjp(activation, first_weight, second_weight, output_cotangent):
        def forward(x, w0, w1):
            preactivation = jnp.matmul(x, w0, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
            scalar = preactivation.astype(jnp.float32)
            if case.scalar_map == "sigmoid_product":
                hidden = (scalar * jax.nn.sigmoid(scalar)).astype(jnp.bfloat16)
            elif case.scalar_map == "tanh_product":
                hidden = (scalar * jnp.tanh(scalar)).astype(jnp.bfloat16)
            elif case.scalar_map == "cubic_mix":
                square = scalar * scalar
                hidden = (scalar + square * scalar).astype(jnp.bfloat16)
            else:
                raise ValueError(f"unsupported structural scalar Map {case.scalar_map!r}")
            return jnp.matmul(hidden, w1, preferred_element_type=jnp.float32).astype(jnp.bfloat16)

        output, pullback = jax.vjp(forward, activation, first_weight, second_weight)
        input_adjoint, first_weight_adjoint, second_weight_adjoint = pullback(output_cotangent)
        return output, input_adjoint, first_weight_adjoint, second_weight_adjoint

    return composed_primal_and_vjp


def export_stablehlo(case: StructuralCase) -> str:
    arguments = tuple(jax.ShapeDtypeStruct(shape, jnp.bfloat16) for shape in case.input_shapes)
    function = natural_jax_composed_primal_and_vjp(case)
    module = jax.jit(function).lower(*arguments).compiler_ir(dialect="stablehlo")
    return f"{module}".rstrip() + "\n"


def xla_hook_boundary_stablehlo(payload: str) -> str:
    stablehlo.register_stablehlo_passes()
    with make_ir_context():
        module = ir.Module.parse(payload)
        pipeline = passmanager.PassManager.parse(HOOK_PIPELINE)
        pipeline.run(module.operation)
        return f"{module}".rstrip() + "\n"


def normalized_fingerprint(payload: str, normalizer: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="shuttle-bf16-fixture-") as directory:
        fixture_path = Path(directory) / "fixture.mlir"
        fixture_path.write_text(payload)
        result = subprocess.run(
            [str(normalizer), "--shuttle-test-report-normalized-fingerprint", str(fixture_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    match = FINGERPRINT_PATTERN.search(result.stdout)
    if not match:
        raise RuntimeError("normalizer did not report one structural SHA-256")
    return match.group(1).upper()


def audited_fixture(case: StructuralCase, normalizer: Path) -> str:
    payload = export_stablehlo(case)
    hook_payload = xla_hook_boundary_stablehlo(payload)
    generator_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper()
    fields = json.dumps(
        {
            "features": case.features,
            "reduction": case.reduction,
            "rows": case.rows,
            "scalar_map": case.scalar_map,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    inputs = ", ".join(f"{shape}:bfloat16" for shape in case.input_shapes)
    output_shapes = (
        (case.rows, case.reduction),
        (case.rows, case.reduction),
        (case.reduction, case.features),
        (case.features, case.reduction),
    )
    outputs = ", ".join(
        f"{role}={shape}:bfloat16" for role, shape in zip(("forward", "dx", "dw0", "dw1"), output_shapes, strict=True)
    )
    header = "\n".join(
        (
            "// Copyright The Marin Authors",
            "// SPDX-License-Identifier: Apache-2.0",
            "",
            "// Ordinary-JAX composed BF16 fixture inventory; not native acceptance evidence",
            f"// Generator: {Path(__file__).name}",
            f"// Generator source SHA-256: {generator_digest}",
            "// Source: jax.jit(natural JAX primal plus jax.vjp).lower(...).compiler_ir(StableHLO)",
            f"// Case ID: {case.case_id}",
            f"// Structural fields: {fields}",
            f"// Inputs: {inputs}",
            f"// Outputs: {outputs}",
            f"// JAX: {JAX_VERSION}; jaxlib: {JAXLIB_VERSION}; XLA: {PINNED_XLA_REVISION}",
            f"// Raw StableHLO SHA-256: {hashlib.sha256(payload.encode()).hexdigest().upper()}",
            f"// Raw normalized StableHLO SHA-256: {normalized_fingerprint(payload, normalizer)}",
            "// XLA hook-boundary preprocessing: stablehlo-complex-math-expander",
            f"// XLA hook-boundary StableHLO SHA-256: {hashlib.sha256(hook_payload.encode()).hexdigest().upper()}",
            f"// XLA hook-boundary normalized StableHLO SHA-256: {normalized_fingerprint(hook_payload, normalizer)}",
            "",
        )
    )
    return header + payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalizer", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    if jax.__version__ != JAX_VERSION or jaxlib.__version__ != JAXLIB_VERSION:
        raise RuntimeError(
            f"fixture inventory requires JAX {JAX_VERSION} and jaxlib {JAXLIB_VERSION}; "
            f"found {jax.__version__} and {jaxlib.__version__}"
        )
    mismatches = []
    for case in CASES:
        expected = audited_fixture(case, arguments.normalizer)
        path = arguments.output_dir / case.filename
        if arguments.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected)
        elif not path.exists() or path.read_text() != expected:
            mismatches.append(case.filename)
    if mismatches:
        parser.error("BF16 fixture inventory drift: " + ", ".join(mismatches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
