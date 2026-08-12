# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the independent hook-boundary oracle for composed BF16 fixtures."""

import argparse
import ast
import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import jax
import jaxlib
from jax._src.interpreters.mlir import make_ir_context
from jaxlib.mlir import ir, passmanager
from jaxlib.mlir.dialects import stablehlo

JAX_VERSION = "0.10.1"
JAXLIB_VERSION = "0.10.1"
PINNED_XLA_REVISION = "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69"
GENERATOR_FILENAME = "regenerate-jax-bf16-composed-fixtures.py"
ORACLE_FILENAME = "jax-0.10.1-bf16-composed-fixture-oracle.json"
HOOK_PIPELINE = "builtin.module(func.func(stablehlo-complex-math-expander))"
FINGERPRINT_PATTERN = re.compile(r"(?m)^([0-9A-Fa-f]{64})$")
ALLOWED_GENERATOR_IMPORTS = frozenset(
    {"argparse", "dataclasses", "hashlib", "jax", "jaxlib", "json", "pathlib", "re", "subprocess", "tempfile"}
)
INPUT_ROLES = ("activation", "first_weight", "second_weight", "output_cotangent")
OUTPUT_ROLES = ("forward", "dx", "dw0", "dw1")


def _sha256(payload: str | bytes) -> str:
    if isinstance(payload, str):
        payload = payload.encode()
    return hashlib.sha256(payload).hexdigest().upper()


def _case_id(fields: dict[str, Any]) -> str:
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return "contract_map_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _header_value(text: str, field: str) -> str:
    matches = re.findall(rf"(?m)^// {re.escape(field)}: (.+)$", text)
    if len(matches) != 1:
        raise ValueError(f"fixture must contain exactly one {field!r} header")
    return matches[0]


def _payload(text: str) -> str:
    start = text.find("module @")
    if start < 0:
        raise ValueError("fixture does not contain a StableHLO module")
    return text[start:].rstrip() + "\n"


def _hook_boundary(payload: str) -> str:
    stablehlo.register_stablehlo_passes()
    with make_ir_context():
        module = ir.Module.parse(payload)
        pipeline = passmanager.PassManager.parse(HOOK_PIPELINE)
        pipeline.run(module.operation)
        return f"{module}".rstrip() + "\n"


def _normalized_fingerprint(payload: str, normalizer: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="shuttle-bf16-oracle-") as directory:
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


def _cast_kind(source: str, result: str) -> str:
    source_dtype = _dtype(source)
    result_dtype = _dtype(result)
    if source_dtype == "f32" and result_dtype == "bf16":
        return "f32_to_bf16_round_nearest_even"
    if source_dtype == "bf16" and result_dtype == "f32":
        return "bf16_to_f32_exact"
    if source_dtype == result_dtype == "f32":
        return "f32_to_f32_identity"
    return f"{source_dtype}_to_{result_dtype}"


def _dtype(mlir_type: str) -> str:
    match = re.search(r"(bf16|f32)>$", mlir_type)
    if not match:
        raise ValueError(f"unsupported fixture element type: {mlir_type}")
    return match.group(1)


def _header_shape(mlir_type: str) -> str:
    match = re.fullmatch(r"tensor<([0-9]+)x([0-9]+)x(bf16|f32)>", mlir_type)
    if not match:
        raise ValueError(f"unsupported fixture tensor type: {mlir_type}")
    dtype = "bfloat16" if match.group(3) == "bf16" else "float32"
    return f"({match.group(1)}, {match.group(2)}):{dtype}"


def _hook_inventory(payload: str) -> dict[str, Any]:
    with make_ir_context():
        module = ir.Module.parse(payload)
        functions = [operation for operation in module.body.operations if operation.operation.name == "func.func"]
        if len(functions) != 1:
            raise ValueError("fixture must contain exactly one function")
        block = next(iter(functions[0].operation.regions[0].blocks))
        operations = list(block.operations)
        names = tuple(operation.operation.name for operation in operations)
        converts = []
        for ordinal, operation in enumerate(operations):
            if operation.operation.name != "stablehlo.convert":
                continue
            source_type = str(operation.operation.operands[0].type)
            result_type = str(operation.operation.results[0].type)
            converts.append(
                {
                    "kind": _cast_kind(source_type, result_type),
                    "operation": ordinal,
                    "source_type": source_type,
                    "result_type": result_type,
                }
            )
        terminator = operations[-1]
        if terminator.operation.name != "func.return" or len(terminator.operation.operands) != len(OUTPUT_ROLES):
            raise ValueError("fixture must return the four composed primal/VJP outputs")
        output_anchors = []
        for role, value in zip(OUTPUT_ROLES, terminator.operation.operands, strict=True):
            owner = value.owner
            if owner not in operations:
                raise ValueError(f"{role} must be produced by a function-body operation")
            output_anchors.append(
                {
                    "operation": operations.index(owner),
                    "result": value.result_number,
                    "role": role,
                    "type": str(value.type),
                }
            )
        input_signature = [
            {"role": role, "type": str(argument.type)}
            for role, argument in zip(INPUT_ROLES, block.arguments, strict=True)
        ]
        return {
            "cast_boundaries": converts,
            "input_signature": input_signature,
            "operation_count": len(operations),
            "operation_inventory": dict(sorted(Counter(names).items())),
            "operation_sequence": list(names),
            "output_anchors": output_anchors,
        }


def _generator_digest(generator: Path) -> str:
    source = generator.read_text()
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    unexpected = sorted(roots - ALLOWED_GENERATOR_IMPORTS)
    if unexpected:
        raise ValueError("generator imports outside the ordinary-JAX fixture boundary: " + ", ".join(unexpected))
    return _sha256(source)


def _differing_fields(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    differing = []
    for field in sorted(expected.keys() | actual.keys()):
        expected_value = expected.get(field)
        actual_value = actual.get(field)
        if expected_value == actual_value:
            continue
        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            differing.extend(
                f"{field}.{nested}"
                for nested in sorted(expected_value.keys() | actual_value.keys())
                if expected_value.get(nested) != actual_value.get(nested)
            )
        else:
            differing.append(field)
    return differing


def derived_case_record(path: Path, normalizer: Path) -> dict[str, Any]:
    text = path.read_text()
    fields = json.loads(_header_value(text, "Structural fields"))
    payload = _payload(text)
    hook_payload = _hook_boundary(payload)
    return {
        "case_id": _case_id(fields),
        "filename": path.name,
        "hashes": {
            "hook_normalized_sha256": _normalized_fingerprint(hook_payload, normalizer),
            "hook_sha256": _sha256(hook_payload),
            "raw_normalized_sha256": _normalized_fingerprint(payload, normalizer),
            "raw_sha256": _sha256(payload),
        },
        "hook_boundary": _hook_inventory(hook_payload),
        "structural_fields": fields,
    }


def write_oracle(
    fixture_directory: Path,
    oracle_path: Path,
    generator_path: Path,
    normalizer: Path,
) -> None:
    fixture_paths = sorted(fixture_directory.glob("jax-0.10.1-bf16-contract_map_*-primal-vjp.mlir"))
    if len(fixture_paths) != 4:
        raise ValueError("BF16 fixture corpus must contain exactly four structural cases")
    oracle = {
        "schema_version": 1,
        "provenance": {
            "generator": GENERATOR_FILENAME,
            "generator_sha256": _generator_digest(generator_path),
            "hook_pipeline": HOOK_PIPELINE,
            "jax": JAX_VERSION,
            "jaxlib": JAXLIB_VERSION,
            "source": "natural_jax_primal_plus_jax_owned_vjp",
            "xla": PINNED_XLA_REVISION,
        },
        "cases": [derived_case_record(path, normalizer) for path in fixture_paths],
    }
    oracle_path.write_text(json.dumps(oracle, indent=2, sort_keys=True) + "\n")


def verify_fixture_corpus(
    fixture_directory: Path,
    oracle_path: Path,
    generator_path: Path,
    normalizer: Path,
) -> None:
    oracle = json.loads(oracle_path.read_text())
    if oracle.get("schema_version") != 1:
        raise ValueError("BF16 fixture oracle schema drift")
    provenance = oracle.get("provenance")
    expected_provenance = {
        "generator": GENERATOR_FILENAME,
        "generator_sha256": _generator_digest(generator_path),
        "hook_pipeline": HOOK_PIPELINE,
        "jax": JAX_VERSION,
        "jaxlib": JAXLIB_VERSION,
        "source": "natural_jax_primal_plus_jax_owned_vjp",
        "xla": PINNED_XLA_REVISION,
    }
    if provenance != expected_provenance:
        raise ValueError("BF16 fixture generator or toolchain provenance drift")
    if jax.__version__ != JAX_VERSION or jaxlib.__version__ != JAXLIB_VERSION:
        raise RuntimeError(
            f"fixture oracle requires JAX {JAX_VERSION} and jaxlib {JAXLIB_VERSION}; "
            f"found {jax.__version__} and {jaxlib.__version__}"
        )
    expected_cases = oracle.get("cases")
    if not isinstance(expected_cases, list) or len(expected_cases) != 4:
        raise ValueError("BF16 fixture oracle must contain exactly four structural cases")
    expected_names = [record["filename"] for record in expected_cases]
    actual_names = sorted(path.name for path in fixture_directory.glob("jax-0.10.1-bf16-contract_map_*-primal-vjp.mlir"))
    if sorted(expected_names) != actual_names:
        raise ValueError("BF16 fixture corpus identity drift")
    mismatches = []
    pins = f"{JAX_VERSION}; jaxlib: {JAXLIB_VERSION}; XLA: {PINNED_XLA_REVISION}"
    generator_digest = expected_provenance["generator_sha256"]
    for expected in expected_cases:
        path = fixture_directory / expected["filename"]
        text = path.read_text()
        derived = derived_case_record(path, normalizer)
        hook_boundary = expected["hook_boundary"]
        inputs_header = ", ".join(_header_shape(record["type"]) for record in hook_boundary["input_signature"])
        outputs_header = ", ".join(
            f"{record['role']}={_header_shape(record['type'])}" for record in hook_boundary["output_anchors"]
        )
        for field, actual in (
            ("Generator", _header_value(text, "Generator")),
            ("Generator source SHA-256", _header_value(text, "Generator source SHA-256")),
            ("Source", _header_value(text, "Source")),
            ("Case ID", _header_value(text, "Case ID")),
            ("Structural fields", _header_value(text, "Structural fields")),
            ("Inputs", _header_value(text, "Inputs")),
            ("Outputs", _header_value(text, "Outputs")),
            ("JAX", _header_value(text, "JAX")),
            ("Raw StableHLO SHA-256", _header_value(text, "Raw StableHLO SHA-256")),
            ("Raw normalized StableHLO SHA-256", _header_value(text, "Raw normalized StableHLO SHA-256")),
            ("XLA hook-boundary preprocessing", _header_value(text, "XLA hook-boundary preprocessing")),
            ("XLA hook-boundary StableHLO SHA-256", _header_value(text, "XLA hook-boundary StableHLO SHA-256")),
            (
                "XLA hook-boundary normalized StableHLO SHA-256",
                _header_value(text, "XLA hook-boundary normalized StableHLO SHA-256"),
            ),
        ):
            expected_header = {
                "Generator": GENERATOR_FILENAME,
                "Generator source SHA-256": generator_digest,
                "Source": "jax.jit(natural JAX primal plus jax.vjp).lower(...).compiler_ir(StableHLO)",
                "Case ID": expected["case_id"],
                "Structural fields": json.dumps(expected["structural_fields"], sort_keys=True, separators=(",", ":")),
                "Inputs": inputs_header,
                "Outputs": outputs_header,
                "JAX": pins,
                "Raw StableHLO SHA-256": expected["hashes"]["raw_sha256"],
                "Raw normalized StableHLO SHA-256": expected["hashes"]["raw_normalized_sha256"],
                "XLA hook-boundary preprocessing": "stablehlo-complex-math-expander",
                "XLA hook-boundary StableHLO SHA-256": expected["hashes"]["hook_sha256"],
                "XLA hook-boundary normalized StableHLO SHA-256": expected["hashes"]["hook_normalized_sha256"],
            }[field]
            if actual != expected_header:
                mismatches.append(f"{path.name}: {field}")
        if derived != expected:
            differing = _differing_fields(expected, derived)
            mismatches.append(f"{path.name}: oracle fields {', '.join(differing)}")
    if mismatches:
        raise ValueError("BF16 composed fixture drift:\n" + "\n".join(mismatches))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalizer", required=True, type=Path)
    parser.add_argument(
        "--fixture-directory",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "test" / "Inputs",
    )
    parser.add_argument("--oracle", type=Path, default=Path(__file__).with_name(ORACLE_FILENAME))
    parser.add_argument(
        "--generator",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "test" / "Inputs" / GENERATOR_FILENAME,
    )
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    if arguments.write:
        write_oracle(arguments.fixture_directory, arguments.oracle, arguments.generator, arguments.normalizer)
    verify_fixture_corpus(arguments.fixture_directory, arguments.oracle, arguments.generator, arguments.normalizer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
