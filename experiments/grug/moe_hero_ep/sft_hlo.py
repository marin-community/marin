# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Conservative declared-result inventory across optimized HLO computations."""

import codecs
import hashlib
import re
import zlib
from dataclasses import dataclass


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
    result = re.sub(r"/\*index=\d+\*/", "", result)
    arrays = []
    position = 0
    types = {"pred", "s8", "s16", "s32", "s64", "u8", "u16", "u32", "u64", "bf16", "f16", "f32", "f64"}

    def shape() -> bool:
        nonlocal position
        while position < len(result) and result[position].isspace():
            position += 1
        if position < len(result) and result[position] == "(":
            position += 1
            if result[position : position + 1] == ")":
                position += 1
                return True
            while shape():
                while position < len(result) and result[position].isspace():
                    position += 1
                if result[position : position + 1] == ")":
                    position += 1
                    return True
                if result[position : position + 1] != ",":
                    return False
                position += 1
            return False
        match = re.match(r"([a-z][a-z0-9]*)\[((?:\d+(?:,\d+)*)?)\](?:\{[^{}]*\})?", result[position:])
        if match is None or match[1] not in types:
            return False
        arrays.append(ArrayResult(match[1], tuple(int(x) for x in match[2].split(",") if x)))
        position += match.end()
        return True

    return tuple(arrays) if shape() and not result[position:].strip() else None


# This inventory recognizes syntax, not execution semantics. New opcodes must
# be reviewed before they can contribute to a complete declared-result audit.
_HLO_OPCODES = frozenset(
    """
abs add all-gather all-gather-start all-gather-done all-reduce all-reduce-start
all-reduce-done all-to-all and async-start async-update async-done bitcast
bitcast-convert broadcast call ceil clamp collective-permute compare concatenate
conditional constant convert copy cosine custom-call divide dot dynamic-slice
dynamic-update-slice exponential floor fusion fusion-start fusion-done gather
get-tuple-element iota is-finite log log-plus-one map maximum minimum multiply
negate not optimization-barrier or pad parameter partition-id power
ragged-all-to-all reduce reduce-scatter reduce-window replica-id reshape reverse
round-nearest-afz round-nearest-even rsqrt scatter select shift-left
shift-right-arithmetic shift-right-logical sign sine slice sort sqrt subtract
tanh transpose tuple while xor
""".split()
)


def _unquoted(text: str) -> str:
    return re.sub(r'"(?:[^"\\]|\\.)*"', '""', text)


def _balanced(text: str) -> bool:
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for char in _unquoted(text):
        if char in "([{":
            stack.append(char)
        elif char in pairs and (not stack or stack.pop() != pairs[char]):
            return False
    return not stack and re.fullmatch(r'(?:[^"\\]|"")*', _unquoted(text)) is not None


def _attributes(text: str) -> dict[str, str] | None:
    """Split top-level HLO attributes without interpreting nested metadata/IR."""
    text = _unquoted(text)
    fields = {}
    depth = 0
    start = 0
    parts = []
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    if parts[0].strip():
        return None
    for part in parts[1:]:
        field = re.fullmatch(r"\s*([\w-]+)=(.+)", part)
        if field is None or field[1] in fields:
            return None
        fields[field[1]] = field[2].strip()
    return fields


def _parse_functions(text: str) -> tuple[dict[str, list[Operation]], list[str]]:
    """Inventory every instruction, rejecting unaccounted lines and references."""
    functions: dict[str, list[Operation]] = {}
    errors = []
    current = None
    entry = None
    module = False
    debug_section = None
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("HloModule ") and not module and not functions:
            module = True
            continue
        header = re.fullmatch(r"(ENTRY )?%([\w.-]+) (\(.*\)) -> (.+) \{", line)
        if header:
            if current is not None or header[2] in functions or (header[1] and entry is not None):
                errors.append(f"line {number}: duplicate or nested computation")
            current = header[2]
            functions[current] = []
            if header[1]:
                entry = current
            arguments = re.sub(r"[\w.-]+:\s*", "", header[3])
            for signature in (arguments, header[4]):
                declared = Operation("signature", "parameter", (), f"%signature = {signature} parameter(0)", current)
                if _declared_results(declared) is None or not _balanced(signature):
                    errors.append(f"line {number}: unsupported computation signature")
            debug_section = None
            continue
        if current is None:
            if not functions and line in ("FileNames", "FunctionNames", "FileLocations", "StackFrames"):
                debug_section = line
                continue
            debug_pattern = (
                r'\d+ "(?:[^"\\]|\\.)*"'
                if debug_section in ("FileNames", "FunctionNames")
                else r"\d+ \{(?:[a-z_]+=\d+ ?)+\}"
            )
            if debug_section is not None and re.fullmatch(debug_pattern, line):
                continue
            errors.append(f"line {number}: unparsed module syntax")
            continue
        if line == "}":
            if not functions[current] or not functions[current][-1].line.startswith("ROOT "):
                errors.append(f"line {number}: missing terminal root")
            current = None
            continue
        instruction = re.fullmatch(
            r"(?:ROOT )?%([\w.-]+) = (.+?) ([a-z][a-z0-9-]*)\(([^)]*)\)(.*)",
            re.sub(r"/\*index=\d+\*/", "", line),
        )
        if instruction is None:
            errors.append(f"line {number}: unparsed instruction")
            continue
        name, _, opcode, arguments, attributes = instruction.groups()
        operands = tuple(re.findall(r"%([\w.-]+)", arguments))
        operation = Operation(name, opcode, operands, line, current)
        previous = {item.name for item in functions[current]}
        if name in previous or any(item.line.startswith("ROOT ") for item in functions[current]):
            errors.append(f"line {number}: duplicate instruction or nonterminal root")
        if opcode not in _HLO_OPCODES or _declared_results(operation) is None or not _balanced(line):
            errors.append(f"line {number}: unsupported opcode, result type or delimiters")
        if (opcode == "parameter" and not arguments.isdigit()) or (
            opcode not in ("parameter", "constant")
            and re.fullmatch(r"(?:%[\w.-]+(?:,\s*%[\w.-]+)*)?", arguments) is None
        ):
            errors.append(f"line {number}: unsupported operand syntax")
        if any(operand not in previous for operand in operands):
            errors.append(f"line {number}: missing local operand")
        if _attributes(attributes) is None:
            errors.append(f"line {number}: unsupported attribute syntax")
        functions[current].append(operation)
    if not module or entry is None or current is not None:
        errors.append("incomplete module or entry computation")
    for operations in functions.values():
        for operation in operations:
            raw_attributes = operation.line.split(f" {operation.opcode}(", 1)[1].split(")", 1)[1]
            attributes = _attributes(raw_attributes)
            if attributes is None:
                continue
            controls = attributes.pop("control-predecessors", "")
            if controls:
                if re.fullmatch(r"\{%[\w.-]+(?:,\s*%[\w.-]+)*\}", controls) is None or any(
                    name not in {item.name for item in operations} for name in re.findall(r"%([\w.-]+)", controls)
                ):
                    errors.append(f"{operation.function}/{operation.name}: missing control predecessor")
            references = re.findall(r"%([\w.-]+)", " ".join(attributes.values()))
            if any(name not in functions for name in references):
                errors.append(f"{operation.function}/{operation.name}: missing computation reference")
            required = {
                "fusion": ("calls",),
                "fusion-start": ("calls",),
                "call": ("to_apply",),
                "async-start": ("calls",),
                "while": ("condition", "body"),
                "reduce": ("to_apply",),
                "reduce-window": ("to_apply",),
                "map": ("to_apply",),
                "sort": ("to_apply",),
                "scatter": ("to_apply",),
                "all-reduce": ("to_apply",),
                "all-reduce-start": ("to_apply",),
                "reduce-scatter": ("to_apply",),
            }.get(operation.opcode, ())
            if any(re.fullmatch(r"%[\w.-]+", attributes.get(key, "")) is None for key in required):
                errors.append(f"{operation.function}/{operation.name}: missing computation attribute")
            if operation.opcode == "conditional" and not (
                re.fullmatch(r"\{%[\w.-]+(?:,\s*%[\w.-]+)*\}", attributes.get("branch_computations", ""))
                or all(re.fullmatch(r"%[\w.-]+", attributes.get(f"{key}_computation", "")) for key in ("true", "false"))
            ):
                errors.append(f"{operation.function}/{operation.name}: missing conditional branches")
    return functions, errors


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


@dataclass(frozen=True)
class GradientAudit:
    parsed_instruction_inventory: bool
    inventoried_computations: int
    inventoried_instructions: int
    instruction_inventory_errors: list[str]
    inspected_operations: int
    unverified_weight_shaped_results: list[dict]
    unknown_custom_call_outputs: list[str]
    known_nonfloating_custom_calls: list[str]
    certified_transport_calls: list[dict]

    @property
    def avoids_full_base_gradients(self) -> bool:
        return (
            self.parsed_instruction_inventory
            and self.inspected_operations > 0
            and not self.unverified_weight_shaped_results
            and not self.unknown_custom_call_outputs
        )


def audit_gradient_operations(
    text: str,
    shapes: dict[tuple[int, ...], set[str]],
) -> GradientAudit:
    """Check declared dot/multiply/scatter/custom-call and conversion results.

    Every computation is inventoried independently of execution/parameter roles.
    Unknown syntax, types or references fail closed. Other operations and opaque
    kernel allocations require a separate source audit.
    """
    forbidden = []
    unknown_calls = []
    nonfloating_calls = []
    transports = []
    functions, inventory_errors = _parse_functions(text)
    parsed_inventory = not inventory_errors
    observations = [
        operation
        for operations in functions.values()
        for operation in operations
        if operation.opcode in ("dot", "multiply", "custom-call", "scatter", "convert", "bitcast-convert")
    ]
    for operation in observations:
        line = operation.line
        declared = _declared_results(operation)
        if declared is None:
            parsed_inventory = False
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
                forbidden.append(match)
    return GradientAudit(
        parsed_inventory,
        len(functions),
        sum(map(len, functions.values())),
        inventory_errors,
        len(observations),
        forbidden,
        unknown_calls,
        nonfloating_calls,
        transports,
    )
