# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the independent structural oracle for BF16 row Fold fixtures."""

import argparse
import ast
import hashlib
import json
import re
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
JAX_REVISION = "619764c15117fbefc4ba13ab941871cb514c23f6"
XLA_REVISION = "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69"
STABLEHLO_VERSION = "1.17.0"
GENERATOR_FILENAME = "regenerate-jax-bf16-row-fold-scale-fixtures.py"
ORACLE_FILENAME = "jax-0.10.1-bf16-row-fold-scale-fixture-oracle.json"
FIXTURE_PATTERN = "jax-0.10.1-bf16-row_fold_scale_*-*.mlir"
HOOK_PIPELINE = "builtin.module(func.func(stablehlo-complex-math-expander))"
BOUNDARIES = ("forward", "backward", "composed")
EXPECTED_SHAPES = {
    "row_fold_scale_44d152ecc3e9ff18": {
        "features": 4096,
        "rows": 2048,
        "shape_role": "primary_shape_candidate",
    },
    "row_fold_scale_81928ab3539c0f03": {
        "features": 13,
        "rows": 7,
        "shape_role": "structural_shape_mutation",
    },
}
ALLOWED_GENERATOR_IMPORTS = frozenset({"argparse", "dataclasses", "hashlib", "jax", "jaxlib", "json", "pathlib"})


def _sha256(payload: str | bytes) -> str:
    if isinstance(payload, str):
        payload = payload.encode()
    return hashlib.sha256(payload).hexdigest().upper()


def _canonical_sha256(value: object) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")))


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
        passmanager.PassManager.parse(HOOK_PIPELINE).run(module.operation)
        return f"{module}".rstrip() + "\n"


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


def _operation(view):
    return view.operation


def _lookup_anchor(value, anchors: list[tuple[Any, list[Any]]]) -> list[Any]:
    for candidate, anchor in anchors:
        if candidate == value:
            return anchor
    raise ValueError(f"operand has no structural anchor: {value}")


def _attribute_inventory(operation) -> dict[str, str]:
    return {name: str(operation.attributes[name]) for name in sorted(operation.attributes)}


def _boundary_roles(boundary: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if boundary == "forward":
        return ("x", "gamma"), ("y",)
    if boundary == "backward":
        return ("x", "gamma", "dy"), ("dx", "dgamma")
    if boundary == "composed":
        return ("x", "gamma", "dy"), ("y", "dx", "dgamma")
    raise ValueError(f"unsupported fixture boundary {boundary!r}")


def _signature_headers(shape: dict[str, Any], boundary: str) -> tuple[str, str]:
    rows = shape["rows"]
    features = shape["features"]
    input_roles, output_roles = _boundary_roles(boundary)
    dimensions = {
        "dgamma": (features,),
        "dy": (rows, features),
        "dx": (rows, features),
        "gamma": (features,),
        "x": (rows, features),
        "y": (rows, features),
    }

    def render(roles: tuple[str, ...]) -> str:
        return ", ".join(f"{role}={dimensions[role]}:bfloat16" for role in roles)

    return render(input_roles), render(output_roles)


def _block_inventory(
    block,
    *,
    function_ordinal: int,
    block_ordinal: int,
    next_block_ordinal: list[int],
    anchors: list[tuple[Any, list[Any]]],
    reducers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for argument in block.arguments:
        anchors.append(
            (
                argument,
                ["block_argument", function_ordinal, block_ordinal, argument.arg_number],
            )
        )
    records = []
    for operation_ordinal, view in enumerate(block.operations):
        operation = _operation(view)
        result_refs = [
            [function_ordinal, block_ordinal, operation_ordinal, result_ordinal]
            for result_ordinal in range(len(operation.results))
        ]
        record = {
            "attributes": _attribute_inventory(operation),
            "name": operation.name,
            "operands": [_lookup_anchor(operand, anchors) for operand in operation.operands],
            "ordinal": operation_ordinal,
            "result_refs": result_refs,
            "result_types": [str(result.type) for result in operation.results],
        }
        if not result_refs:
            record["operation_ref"] = [function_ordinal, block_ordinal, operation_ordinal]
        for result, source_ref in zip(operation.results, result_refs, strict=True):
            anchors.append((result, ["source_result", *source_ref]))
        region_blocks = []
        for region in operation.regions:
            for nested_block in region.blocks:
                nested_ordinal = next_block_ordinal[0]
                next_block_ordinal[0] += 1
                nested_operations = _block_inventory(
                    nested_block,
                    function_ordinal=function_ordinal,
                    block_ordinal=nested_ordinal,
                    next_block_ordinal=next_block_ordinal,
                    anchors=anchors,
                    reducers=reducers,
                )
                region_blocks.append({"block_ordinal": nested_ordinal, "operations": nested_operations})
        if region_blocks:
            record["region_blocks"] = [nested["block_ordinal"] for nested in region_blocks]
        if operation.name == "stablehlo.reduce":
            if len(region_blocks) != 1:
                raise ValueError("each row Fold fixture reduce must contain exactly one block")
            reducers.append(
                {
                    "block_ordinal": region_blocks[0]["block_ordinal"],
                    "operations": region_blocks[0]["operations"],
                    "source_ref": result_refs[0],
                    "top_level_operation": operation_ordinal,
                }
            )
        records.append(record)
    return records


def _graph_inventory(payload: str, boundary: str) -> dict[str, Any]:
    input_roles, output_roles = _boundary_roles(boundary)
    with make_ir_context():
        module = ir.Module.parse(payload)
        functions = [view for view in module.body.operations if _operation(view).name == "func.func"]
        if len(functions) != 1:
            raise ValueError("fixture must contain exactly one function")
        function = _operation(functions[0])
        blocks = list(function.regions[0].blocks)
        if len(blocks) != 1:
            raise ValueError("fixture function must contain exactly one top-level block")
        block = blocks[0]
        if len(block.arguments) != len(input_roles):
            raise ValueError("fixture input arity does not match its boundary")
        anchors: list[tuple[Any, list[Any]]] = []
        reducers: list[dict[str, Any]] = []
        operations = _block_inventory(
            block,
            function_ordinal=0,
            block_ordinal=0,
            next_block_ordinal=[1],
            anchors=anchors,
            reducers=reducers,
        )
        if not operations or operations[-1]["name"] != "func.return":
            raise ValueError("fixture function must end in func.return")
        return_operands = operations[-1]["operands"]
        if len(return_operands) != len(output_roles):
            raise ValueError("fixture output arity does not match its boundary")
        top_level_names = [record["name"] for record in operations]
        nested_names = [nested["name"] for reducer in reducers for nested in reducer["operations"]]
        return {
            "function_type": str(function.attributes["function_type"]),
            "input_signature": [
                {"role": role, "type": str(argument.type)}
                for role, argument in zip(input_roles, block.arguments, strict=True)
            ],
            "nested_operation_inventory": dict(sorted(Counter(nested_names).items())),
            "output_anchors": [
                {"anchor": anchor, "role": role} for role, anchor in zip(output_roles, return_operands, strict=True)
            ],
            "reducers": reducers,
            "top_level_operation_inventory": dict(sorted(Counter(top_level_names).items())),
            "top_level_operations": operations,
        }


def _oracle_inventory(graph: dict[str, Any]) -> dict[str, Any]:
    reducers = []
    for reducer in graph["reducers"]:
        reducers.append(
            {
                "block_ordinal": reducer["block_ordinal"],
                "operations": [
                    {
                        field: operation[field]
                        for field in ("name", "operands", "operation_ref", "ordinal", "result_refs", "result_types")
                        if field in operation
                    }
                    for operation in reducer["operations"]
                ],
                "source_ref": reducer["source_ref"],
                "top_level_operation": reducer["top_level_operation"],
            }
        )
    return {
        "function_type": graph["function_type"],
        "input_signature": graph["input_signature"],
        "nested_operation_inventory": graph["nested_operation_inventory"],
        "output_anchors": graph["output_anchors"],
        "provenance_scheme": {
            "function_ordinal": 0,
            "result_ref_fields": ["function", "block", "operation", "result"],
            "top_level_block_ordinal": 0,
        },
        "reducers": reducers,
        "top_level_operation_inventory": graph["top_level_operation_inventory"],
        "top_level_operation_sequence": [operation["name"] for operation in graph["top_level_operations"]],
    }


def _case_id(fields: dict[str, Any]) -> str:
    shape = {"features": fields["features"], "rows": fields["rows"]}
    payload = json.dumps(shape, sort_keys=True, separators=(",", ":"))
    return "row_fold_scale_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def derived_case_record(path: Path) -> dict[str, Any]:
    text = path.read_text()
    fields = json.loads(_header_value(text, "Structural fields"))
    boundary = fields["boundary"]
    payload = _payload(text)
    hook_payload = _hook_boundary(payload)
    raw_inventory = _graph_inventory(payload, boundary)
    hook_inventory = _graph_inventory(hook_payload, boundary)
    return {
        "boundary": boundary,
        "case_id": _case_id(fields),
        "filename": path.name,
        "hashes": {
            "hook_normalized_sha256": _canonical_sha256(hook_inventory),
            "hook_sha256": _sha256(hook_payload),
            "raw_normalized_sha256": _canonical_sha256(raw_inventory),
            "raw_sha256": _sha256(payload),
        },
        "hook_boundary": _oracle_inventory(hook_inventory),
        "shape": {
            "features": fields["features"],
            "rows": fields["rows"],
            "shape_role": fields["shape_role"],
        },
    }


def _expected_provenance(generator: Path) -> dict[str, Any]:
    return {
        "artifact_kind": "ordinary_jax_fixture_only",
        "evaluation_oracle_status": "not_pinned",
        "generator": GENERATOR_FILENAME,
        "generator_sha256": _generator_digest(generator),
        "hardware_evidence": False,
        "hook_pipeline": HOOK_PIPELINE,
        "jax_revision": JAX_REVISION,
        "jax_version": JAX_VERSION,
        "jaxlib_version": JAXLIB_VERSION,
        "source": "ordinary_jax_row_fold_scale_plus_jax_owned_vjp",
        "stablehlo_current_version": STABLEHLO_VERSION,
        "xla_revision": XLA_REVISION,
    }


def _fixture_paths(fixture_directory: Path) -> list[Path]:
    return sorted(fixture_directory.glob(FIXTURE_PATTERN))


def write_oracle(fixture_directory: Path, oracle_path: Path, generator: Path) -> None:
    paths = _fixture_paths(fixture_directory)
    if len(paths) != len(EXPECTED_SHAPES) * len(BOUNDARIES):
        raise ValueError("BF16 row Fold fixture corpus must contain exactly six cases")
    document = {
        "cases": [derived_case_record(path) for path in paths],
        "provenance": _expected_provenance(generator),
        "schema_version": 1,
    }
    oracle_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _differing_fields(expected: dict[str, Any], actual: dict[str, Any], prefix: str = "") -> list[str]:
    differing = []
    for field in sorted(expected.keys() | actual.keys()):
        name = f"{prefix}.{field}" if prefix else field
        expected_value = expected.get(field)
        actual_value = actual.get(field)
        if expected_value == actual_value:
            continue
        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            differing.extend(_differing_fields(expected_value, actual_value, name))
        else:
            differing.append(name)
    return differing


def _verify_runtime_pins(generator: Path) -> None:
    if jax.__version__ != JAX_VERSION or jaxlib.__version__ != JAXLIB_VERSION:
        raise RuntimeError(
            f"fixture oracle requires JAX {JAX_VERSION} and jaxlib {JAXLIB_VERSION}; "
            f"found {jax.__version__} and {jaxlib.__version__}"
        )
    if stablehlo.get_current_version() != STABLEHLO_VERSION:
        raise RuntimeError("fixture oracle StableHLO version drift")
    mlir_root = generator.resolve().parents[2]
    if (mlir_root / "jax_patch" / "PINNED_JAX_REVISION").read_text().strip() != JAX_REVISION:
        raise RuntimeError("fixture oracle JAX revision pin drift")
    if (mlir_root / "xla_patch" / "PINNED_XLA_REVISION").read_text().strip() != XLA_REVISION:
        raise RuntimeError("fixture oracle XLA revision pin drift")


def verify_fixture_corpus(fixture_directory: Path, oracle_path: Path, generator: Path) -> None:
    oracle = json.loads(oracle_path.read_text())
    if set(oracle) != {"cases", "provenance", "schema_version"} or oracle.get("schema_version") != 1:
        raise ValueError("BF16 row Fold fixture oracle schema drift")
    if oracle.get("provenance") != _expected_provenance(generator):
        raise ValueError("BF16 row Fold fixture generator or toolchain provenance drift")
    _verify_runtime_pins(generator)
    expected_cases = oracle.get("cases")
    if not isinstance(expected_cases, list) or len(expected_cases) != 6:
        raise ValueError("BF16 row Fold fixture oracle must contain exactly six cases")
    actual_names = [path.name for path in _fixture_paths(fixture_directory)]
    expected_names = sorted(case["filename"] for case in expected_cases)
    if actual_names != expected_names:
        raise ValueError("BF16 row Fold fixture corpus identity drift")
    identities = {(case["case_id"], case["boundary"]) for case in expected_cases}
    required = {(case_id, boundary) for case_id in EXPECTED_SHAPES for boundary in BOUNDARIES}
    if identities != required:
        raise ValueError("BF16 row Fold shape or boundary coverage drift")
    mismatches = []
    generator_digest = oracle["provenance"]["generator_sha256"]
    for expected in expected_cases:
        path = fixture_directory / expected["filename"]
        text = path.read_text()
        inputs_header, outputs_header = _signature_headers(expected["shape"], expected["boundary"])
        fields = {
            "boundary": expected["boundary"],
            "epsilon": 1e-5,
            **expected["shape"],
        }
        expected_headers = {
            "Artifact classification": "ordinary_jax_fixture_only",
            "Case ID": expected["case_id"],
            "Evaluation oracle status": "not_pinned",
            "Generator": GENERATOR_FILENAME,
            "Generator source SHA-256": generator_digest,
            "Hardware evidence": "none",
            "Inputs": inputs_header,
            "JAX": f"{JAX_VERSION}; jaxlib: {JAXLIB_VERSION}; JAX revision: {JAX_REVISION}",
            "Outputs": outputs_header,
            "Raw StableHLO SHA-256": expected["hashes"]["raw_sha256"],
            "Source": "jax.jit(ordinary JAX row Fold plus JAX-owned VJP).lower(...).compiler_ir(StableHLO)",
            "Structural fields": json.dumps(fields, sort_keys=True, separators=(",", ":")),
            "XLA hook-boundary StableHLO SHA-256": expected["hashes"]["hook_sha256"],
            "XLA hook-boundary preprocessing": "stablehlo-complex-math-expander",
            "XLA revision": f"{XLA_REVISION}; StableHLO current version: {STABLEHLO_VERSION}",
        }
        for header, value in expected_headers.items():
            if _header_value(text, header) != value:
                mismatches.append(f"{path.name}: {header}")
        derived = derived_case_record(path)
        if derived != expected:
            mismatches.extend(f"{path.name}: {field}" for field in _differing_fields(expected, derived))
    if mismatches:
        raise ValueError("BF16 row Fold fixture drift:\n" + "\n".join(mismatches))


def main() -> int:
    parser = argparse.ArgumentParser()
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
        write_oracle(arguments.fixture_directory, arguments.oracle, arguments.generator)
    verify_fixture_corpus(arguments.fixture_directory, arguments.oracle, arguments.generator)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
