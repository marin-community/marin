# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Conservative weight-operand provenance through optimized HLO control flow."""

import codecs
import hashlib
import re
import zlib
from dataclasses import dataclass

# Array roles are True (a parameter/layout transform), False (computed activation)
# or "constant". Tuples preserve roles for while carries and get-tuple-element.
type Role = bool | str | tuple[Role, ...]


@dataclass(frozen=True)
class Operation:
    name: str
    opcode: str
    operands: tuple[str, ...]
    line: str
    function: str


@dataclass(frozen=True)
class ArrayResult:
    dtype: str
    shape: tuple[int, ...]

    @property
    def floating(self) -> bool:
        return self.dtype.startswith(("f", "bf", "c"))


def _declared_results(operation: Operation) -> tuple[ArrayResult, ...] | None:
    """Parse the complete declared result, including nested tuples and integer arrays."""
    result = operation.line.split(" = ", 1)[1].split(f" {operation.opcode}(", 1)[0]
    pattern = r"([a-z][a-z0-9]*)\[([0-9,]*)\](?:\{[^}]*\})?"
    matches = list(re.finditer(pattern, result))
    remainder = re.sub(pattern, "", result)
    if not matches or re.sub(r"[(),\s]", "", remainder):
        return None
    types = {"pred", "s8", "s16", "s32", "s64", "u8", "u16", "u32", "u64", "bf16", "f16", "f32", "f64"}
    if any(match[1] not in types for match in matches):
        return None
    return tuple(ArrayResult(match[1], tuple(int(x) for x in match[2].split(",") if x)) for match in matches)


def _parse_functions(text: str) -> tuple[dict[str, list[Operation]], str | None]:
    functions: dict[str, list[Operation]] = {}
    entry = None
    current = None
    for line in text.splitlines():
        header = re.match(r"^(ENTRY )?%([^ (]+) .*\{$", line)
        if header:
            current = header.group(2)
            functions[current] = []
            if header.group(1):
                entry = current
            continue
        if current is None or " = " not in line:
            continue
        instruction = re.match(r"\s*(?:ROOT )?%([^ ]+) = (.*)", line)
        if instruction is None:
            continue
        operation = re.search(r"\b([a-z][a-z0-9-]*)\(([^)]*)\)", instruction.group(2))
        if operation is None:
            continue
        operands = tuple(re.findall(r"%([a-zA-Z_][\w.-]*)", operation.group(2)))
        functions[current].append(Operation(instruction.group(1), operation.group(1), operands, line.strip(), current))
    return functions, entry


def _protobuf_fields(payload: bytes) -> dict[int, list[int | bytes]]:
    """Read the bounded varint/bytes subset of JAX's Triton call protobuf."""
    position = 0

    def varint() -> int:
        nonlocal position
        value = 0
        for shift in range(0, 70, 7):
            if position == len(payload):
                raise ValueError("Truncated Triton protobuf")
            byte = payload[position]
            position += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
        raise ValueError("Oversized Triton protobuf varint")

    fields: dict[int, list[int | bytes]] = {}
    while position < len(payload):
        tag = varint()
        field, wire = tag >> 3, tag & 7
        if field == 0 or wire not in (0, 2):
            raise ValueError("Unsupported Triton protobuf field")
        if wire == 0:
            item: int | bytes = varint()
        else:
            size = varint()
            if size > len(payload) - position:
                raise ValueError("Truncated Triton protobuf bytes")
            item = payload[position : position + size]
            position += size
        fields.setdefault(field, []).append(item)
    return fields


def _single_bytes(fields: dict[int, list[int | bytes]], field: int) -> bytes:
    values = fields.get(field, [])
    if len(values) != 1 or not isinstance(values[0], bytes):
        raise ValueError("Missing or repeated Triton protobuf message")
    return values[0]


def _no_write_transport(operation: Operation, functions: dict[str, list[Operation]]) -> dict | None:
    """Certify only a no-write PTX/TTIR kernel used as a ragged collective destination.

    JAX TritonAnyKernelCall -> TritonKernelCall -> TritonKernel fields are defined
    in jaxlib/gpu/triton.proto. Machine images, autotuning, unknown fields, and
    kernels with any executable work remain unverified. A generic target/name
    never certifies a floating result.
    """
    local = functions[operation.function]
    definitions = {item.name: item for item in local}
    if re.search(r'custom_call_target="triton_kernel_call"(?:,|$)', operation.line) is None:
        return None
    results = _declared_results(operation)
    if results is None or len(results) != 1 or not results[0].floating or len(results[0].shape) != 2:
        return None
    if len(operation.operands) != 1 or operation.operands[0] not in definitions:
        return None
    marker = _declared_results(definitions[operation.operands[0]])
    if marker != (ArrayResult("s32", ()),):
        return None
    consumers = [item for item in local if operation.name in item.operands]
    if not consumers or any(
        item.opcode != "ragged-all-to-all"
        or item.operands.count(operation.name) != 1
        or len(item.operands) < 2
        or item.operands[1] != operation.name
        or _declared_results(item) != results
        for item in consumers
    ):
        return None
    config = re.search(r'backend_config="((?:[^"\\]|\\.)*)"$', operation.line)
    if config is None or "output_to_operand_aliasing" in operation.line or "has_side_effect=true" in operation.line:
        return None
    try:
        compressed = codecs.decode(config[1], "unicode_escape").encode("latin1")
        decoder = zlib.decompressobj()
        payload = decoder.decompress(compressed, 1024 * 1024)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            return None
        wrapper = _protobuf_fields(payload)
        if set(wrapper) != {1}:
            return None
        call = _protobuf_fields(_single_bytes(wrapper, 1))
        if set(call) != {1, 2, 3, 4, 5} or any(call[field] != [1] for field in (2, 3, 4)):
            return None
        if len(call[5]) != 2:
            return None
        for parameter in call[5]:
            if not isinstance(parameter, bytes):
                return None
            argument = _protobuf_fields(parameter)
            if set(argument) != {1}:
                return None
            array = _protobuf_fields(_single_bytes(argument, 1))
            if set(array) != {2} or len(array[2]) != 1 or not isinstance(array[2][0], int) or array[2][0] <= 0:
                return None
        kernel = _protobuf_fields(_single_bytes(call, 1))
        if set(kernel) != {1, 2, 4, 5, 6, 10} or any(len(value) != 1 for value in kernel.values()):
            return None
        if kernel[10] != [1]:
            return None
        name = _single_bytes(kernel, 1).decode()
        ptx = _single_bytes(kernel, 4).decode()
        ttir = _single_bytes(kernel, 5).decode()
    except (ValueError, UnicodeError, zlib.error):
        return None
    # Match the entire module, not an arbitrary embedded function or comment.
    ir = re.sub(
        r'^#loc\d* = loc\((?:"[^"\n]*":\d+:\d+|"[^"\n]*"\(#loc\d*\))\)\s*$',
        "",
        ttir,
        flags=re.MULTILINE,
    )
    ir = re.sub(r'loc\("[^"\n]*"\(#loc\d*\)\)|loc\(#loc\d*\)', "", ir).strip()
    pointer = r"%\w+: !tt.ptr<{}> \{{tt.divisibility = \d+ : i32\}}\s*"
    ir_pattern = (
        r"module\s*\{\s*tt.func public @"
        + re.escape(name)
        + r"\("
        + pointer.format("i32")
        + r",\s*"
        + pointer.format(results[0].dtype)
        + r"\) attributes \{noinline = false\} \{\s*tt.return\s*\}\s*\}"
    )
    if re.fullmatch(ir_pattern, ir) is None:
        return None
    ptx = re.sub(r"//[^\n]*", "", ptx)
    entries = list(re.finditer(r"\.visible\s+\.entry\s+([\w.$]+)\([^)]*\)\s*\.reqntid\s+\d+\s*\{([^{}]*)\}", ptx))
    if len(entries) != 1 or entries[0][1] != name or re.search(r"\.func\b|\.extern\b", ptx):
        return None
    body = re.sub(r"^\s*(?:\.loc\s+\d+\s+\d+\s+\d+|[\w.$]+:)\s*$", "", entries[0][2], flags=re.MULTILINE).strip()
    if body != "ret;":
        return None
    return {
        "certification": "decoded no-write TTIR and PTX; integer marker; ragged collective destination only",
        "ttir_sha256": hashlib.sha256(ttir.encode()).hexdigest(),
        "ptx_sha256": hashlib.sha256(_single_bytes(kernel, 4)).hexdigest(),
    }


def _intersect(left: Role, right: Role) -> Role:
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        return tuple(_intersect(a, b) for a, b in zip(left, right, strict=True))
    return left if left == right else False


def weight_operand_operations(
    text: str, weight_parameter_prefixes: tuple[str, ...]
) -> tuple[list[tuple[Operation, tuple[Role, ...]]], bool]:
    """Evaluate parameter/layout provenance, preserving per-call and loop contexts.

    Computed results lose parameter identity even when they depend on weights.
    A retained parameter operand is diagnostic only: reassociated Wgrad can
    contain another projection's weight. Unknown constructs remain unverified.
    """
    functions, entry = _parse_functions(text)
    if entry is None:
        return [], False
    observations: list[tuple[Operation, tuple[Role, ...]]] = []
    parsed = True
    unary_layout = {
        "bitcast",
        "bitcast-convert",
        "reshape",
        "transpose",
        "slice",
        "dynamic-slice",
        "convert",
        "copy",
        "broadcast",
        "reverse",
        "all-gather",
        "all-to-all",
        "collective-permute",
        "optimization-barrier",
    }
    binary_layout = {"concatenate", "multiply", "divide", "add", "subtract", "pad"}

    def evaluate(function: str, parameters: tuple[Role, ...], depth: int = 0) -> Role:
        nonlocal parsed
        if function not in functions or depth > 20:
            parsed = False
            return False
        values: dict[str, Role] = {}
        definitions = {operation.name: operation for operation in functions[function]}
        root: Role = False
        for operation in functions[function]:
            inputs = tuple(values.get(name, False) for name in operation.operands)
            opcode = operation.opcode
            # Only the synchronous call/control-flow forms below propagate
            # callee contexts. An unvisited reducer, comparator or async body
            # can contain Wgrad even when an unrelated entry dot was inspected.
            attributes = operation.line.split(f" {opcode}(", 1)[1].split(")", 1)[1]
            attributes = re.sub(r'"(?:[^"\\]|\\.)*"', '""', attributes)
            if ("%" in attributes and opcode not in ("fusion", "call", "conditional", "while")) or opcode.endswith(
                ("-start", "-update", "-done")
            ):
                parsed = False
            role: Role = False
            if opcode == "parameter":
                number = int(re.search(r"parameter\((\d+)\)", operation.line).group(1))
                if function == entry:
                    # Certification comes from the compiled training entry's
                    # explicit parameter boundary, never from floating dtype.
                    # Unknown floating inputs may be activation cotangents.
                    metadata = re.search(r'op_name="([^" ]+)"', operation.line)
                    path = metadata.group(1) if metadata else operation.name
                    role = (
                        any(
                            path.startswith(prefix) or operation.name.startswith(prefix.replace(".", "_"))
                            for prefix in weight_parameter_prefixes
                        )
                        and "lora_" not in path
                        and not path.startswith("params.")
                    )
                elif number < len(parameters):
                    role = parameters[number]
            elif opcode in ("constant", "iota"):
                role = "constant"
            elif opcode == "compare" and inputs and all(value == "constant" for value in inputs):
                # Fixed token-ID/vocabulary-column equality maps are the sparse
                # output-delta selector, not learned projection weights.
                role = "selector"
            elif opcode == "tuple":
                role = inputs
            elif opcode == "get-tuple-element":
                index = int(re.search(r"index=(\d+)", operation.line).group(1))
                if inputs and isinstance(inputs[0], tuple) and index < len(inputs[0]):
                    role = inputs[0][index]
            elif opcode in unary_layout and inputs:
                role = inputs[0]
                source = _declared_results(definitions[operation.operands[0]])
                result = _declared_results(operation)
                if (
                    opcode in ("convert", "bitcast-convert")
                    and role is False
                    and source is not None
                    and all(not item.floating for item in source)
                    and result is not None
                    and any(item.floating for item in result)
                ):
                    role = "integer-cast"
            elif opcode in binary_layout and inputs:
                if all(value == "constant" for value in inputs):
                    role = "constant"
                elif all(value is True or value == "constant" for value in inputs) and True in inputs:
                    role = True
                elif "integer-cast" in inputs:
                    role = "integer-cast"
            elif opcode == "select" and len(inputs) == 3:
                role = _intersect(inputs[1], inputs[2])
            elif opcode in ("fusion", "call"):
                callee = re.search(r"(?:calls|to_apply)=%([^, }]+)", operation.line)
                if callee:
                    role = evaluate(callee.group(1), inputs, depth + 1)
                else:
                    parsed = False
            elif opcode == "conditional":
                branches = re.search(r"branch_computations=\{([^}]+)\}", operation.line)
                names = (
                    re.findall(r"%([^, }]+)", branches.group(1))
                    if branches
                    else re.findall(r"(?:true|false)_computation=%([^, }]+)", operation.line)
                )
                if len(names) != len(inputs) - 1:
                    parsed = False
                else:
                    results = [
                        evaluate(name, (value,), depth + 1) for name, value in zip(names, inputs[1:], strict=True)
                    ]
                    role = results[0]
                    for result in results[1:]:
                        role = _intersect(role, result)
            elif opcode == "while" and inputs:
                body = re.search(r"body=%([^, }]+)", operation.line)
                if body:
                    role = inputs[0]
                    condition = re.search(r"condition=%([^, }]+)", operation.line)
                    if condition:
                        evaluate(condition.group(1), (role,), depth + 1)
                    else:
                        parsed = False
                    # Frozen parameters remain unchanged in layer/vocab scans.
                    # Intersect with the body result so modified carries lose certification.
                    for _ in range(3):
                        after = evaluate(body.group(1), (role,), depth + 1)
                        stable = _intersect(role, after)
                        if stable == role:
                            break
                        role = stable
                    else:
                        parsed = False
                else:
                    parsed = False
            # A one-token Wgrad can lower to broadcast/multiply with no dot.
            if opcode in ("dot", "multiply", "custom-call", "scatter") or role == "integer-cast":
                observations.append((operation, inputs))
            values[operation.name] = role
            if operation.line.startswith("ROOT "):
                root = role
        return root

    evaluate(entry, ())
    return list(dict.fromkeys(observations)), parsed


@dataclass(frozen=True)
class GradientAudit:
    parsed_weight_operand_dataflow: bool
    inspected_operations: int
    unverified_weight_shaped_results: list[dict]
    unknown_custom_call_outputs: list[str]
    known_nonfloating_custom_calls: list[str]
    certified_transport_calls: list[dict]

    @property
    def avoids_full_base_gradients(self) -> bool:
        return (
            self.parsed_weight_operand_dataflow
            and self.inspected_operations > 0
            and not self.unverified_weight_shaped_results
            and not self.unknown_custom_call_outputs
        )


def audit_gradient_operations(
    text: str,
    shapes: dict[tuple[int, ...], set[str]],
    *,
    weight_parameter_prefixes: tuple[str, ...] = (),
) -> GradientAudit:
    """Check declared dot/multiply/scatter/custom-call results and tracked casts.

    Unsupported computation forms fail closed. Other operations and opaque
    kernel allocations require a separate source audit.
    """
    forbidden = []
    unknown_calls = []
    nonfloating_calls = []
    transports = []
    functions, _ = _parse_functions(text)
    observations, parsed_dataflow = weight_operand_operations(text, weight_parameter_prefixes)
    for operation, operands in observations:
        line = operation.line
        declared = _declared_results(operation)
        if declared is None:
            parsed_dataflow = False
        results = [item.shape for item in declared or () if item.floating]
        if operation.opcode == "custom-call" and declared is None:
            unknown_calls.append(line)
        elif operation.opcode == "custom-call" and not results:
            nonfloating_calls.append(line)
        for result in results:
            if result not in shapes:
                continue
            match = {"shape": list(result), "matching_frozen_paths": sorted(shapes[result]), "operation": line}
            transport = _no_write_transport(operation, functions) if operation.opcode == "custom-call" else None
            if transport is not None:
                transports.append({**match, **transport})
            else:
                if operation.opcode == "dot" and any(role is True or role == "selector" for role in operands):
                    # Reassociation can expose a parameter operand even in a
                    # genuine Wgrad: (X.T @ dY) @ W1.T. Constant selectors can
                    # likewise produce full vocabulary gradients. Neither is a
                    # proof without axis provenance and compact result flow.
                    match["unverified_reason"] = "known parameter/selector operand without axis and result-flow proof"
                forbidden.append(match)
    return GradientAudit(
        parsed_dataflow,
        len(observations),
        forbidden,
        unknown_calls,
        nonfloating_calls,
        transports,
    )
