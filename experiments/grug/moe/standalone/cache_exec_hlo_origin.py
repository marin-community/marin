# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trace an embedded custom-kernel name back to its optimized HLO instruction.

The JAX persistent cache contains a Riegeli-serialized ``GpuExecutableProto``.
That proto retains both the thunk sequence and the optimized ``HloModuleProto``.
This script uses the protobuf wire format directly so it can run without an XLA
source checkout or generated Python protobuf bindings.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import fsspec

from experiments.grug.moe.standalone.cache_exec_carve import decompress
from experiments.grug.moe.standalone.cache_exec_riegeli import RIEGELI_SIGNATURE, flatten_blocks, iter_records


@dataclass(frozen=True)
class Field:
    number: int
    wire_type: int
    value: int | bytes


@dataclass(frozen=True)
class Instruction:
    computation_id: int
    computation_name: str
    name: str
    opcode: str
    instruction_id: int
    operand_ids: tuple[int, ...]
    called_computation_ids: tuple[int, ...]
    shape: str
    fusion_kind: str
    custom_call_target: str
    backend_config: str
    metadata: dict[str, int | str]


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while pos < len(data) and shift <= 63:
        byte = data[pos]
        value |= (byte & 0x7F) << shift
        pos += 1
        if not byte & 0x80:
            return value, pos
        shift += 7
    raise ValueError(f"invalid varint at offset {pos}")


def _fields(data: bytes) -> list[Field]:
    result = []
    pos = 0
    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        number, wire_type = tag >> 3, tag & 7
        if number == 0:
            raise ValueError("protobuf field number zero")
        if wire_type == 0:
            value, pos = _read_varint(data, pos)
        elif wire_type == 1:
            value = data[pos : pos + 8]
            pos += 8
        elif wire_type == 2:
            size, pos = _read_varint(data, pos)
            value = data[pos : pos + size]
            pos += size
        elif wire_type == 5:
            value = data[pos : pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")
        result.append(Field(number, wire_type, value))
    return result


def _bytes_values(data: bytes, number: int) -> list[bytes]:
    return [
        field.value
        for field in _fields(data)
        if field.number == number and field.wire_type == 2 and isinstance(field.value, bytes)
    ]


def _first_bytes(data: bytes, number: int) -> bytes:
    values = _bytes_values(data, number)
    return values[0] if values else b""


def _first_text(data: bytes, number: int) -> str:
    return _first_bytes(data, number).decode(errors="replace")


def _varint_values(data: bytes, number: int) -> list[int]:
    values = []
    for field in _fields(data):
        if field.number != number:
            continue
        if field.wire_type == 0 and isinstance(field.value, int):
            values.append(field.value)
        elif field.wire_type == 2 and isinstance(field.value, bytes):
            pos = 0
            while pos < len(field.value):
                value, pos = _read_varint(field.value, pos)
                values.append(value)
    return values


def _first_varint(data: bytes, number: int) -> int:
    values = _varint_values(data, number)
    return values[0] if values else 0


def _shape_text(data: bytes) -> str:
    if not data:
        return ""
    element_type = _first_varint(data, 2)
    dimensions = _varint_values(data, 3)
    tuple_shapes = _bytes_values(data, 4)
    if tuple_shapes:
        return "(" + ", ".join(_shape_text(shape) for shape in tuple_shapes) + ")"
    return f"type={element_type}[{','.join(str(dimension) for dimension in dimensions)}]"


def _metadata(data: bytes) -> dict[str, int | str]:
    if not data:
        return {}
    names = {
        1: "op_type",
        2: "op_name",
        3: "source_file",
        12: "deduplicated_name",
        16: "scheduling_name",
    }
    result: dict[str, int | str] = {}
    for number, name in names.items():
        value = _first_text(data, number)
        if value:
            result[name] = value
    for number, name in ((4, "source_line"), (15, "stack_frame_id")):
        value = _first_varint(data, number)
        if value:
            result[name] = value
    return result


def _payload_text(data: bytes, payloads: list[bytes]) -> str:
    if not data:
        return ""
    inline = _first_bytes(data, 1)
    if inline:
        data = inline
    else:
        payload_id = _first_varint(data, 2)
        if payload_id >= len(payloads):
            return f"<invalid payload id {payload_id}>"
        data = payloads[payload_id]
    text = data.decode(errors="replace")
    try:
        return json.dumps(json.loads(text), sort_keys=True)
    except json.JSONDecodeError:
        return text


def _instruction(data: bytes, computation_id: int, computation_name: str, payloads: list[bytes]) -> Instruction:
    backend_config = _first_bytes(data, 43)
    if not backend_config:
        backend_config = _first_bytes(data, 99)
        backend_config_text = _payload_text(backend_config, payloads)
    else:
        backend_config_text = backend_config.decode(errors="replace")
        try:
            backend_config_text = json.dumps(json.loads(backend_config_text), sort_keys=True)
        except json.JSONDecodeError:
            pass
    return Instruction(
        computation_id=computation_id,
        computation_name=computation_name,
        name=_first_text(data, 1),
        opcode=_first_text(data, 2),
        instruction_id=_first_varint(data, 35),
        operand_ids=tuple(_varint_values(data, 36)),
        called_computation_ids=tuple(_varint_values(data, 38)),
        shape=_shape_text(_first_bytes(data, 3)),
        fusion_kind=_first_text(data, 11),
        custom_call_target=_first_text(data, 28),
        backend_config=backend_config_text,
        metadata=_metadata(_first_bytes(data, 7)),
    )


def _gpu_executable_records(raw: bytes) -> list[bytes]:
    outer = decompress(raw)
    ifrt = outer[4:]
    metadata_size, pos = _read_varint(ifrt, 0)
    executable_and_options = ifrt[pos + metadata_size :]
    serialized = _first_bytes(executable_and_options, 1)
    if not serialized.startswith(RIEGELI_SIGNATURE):
        raise ValueError("ExecutableAndOptionsProto has no Riegeli executable")
    for header_at_zero in (True, False):
        stream = flatten_blocks(serialized, header_at_zero)
        if len(stream) > 24 and stream[24] == ord("s"):
            return [record for _, record in iter_records(stream)]
    raise ValueError("Riegeli executable has no file-signature chunk")


def _instructions(records: list[bytes]) -> tuple[list[Instruction], dict[int, str]]:
    for record in reversed(records):
        module_with_config = _first_bytes(record, 1)
        module = _first_bytes(module_with_config, 1)
        if not module:
            continue
        payloads = _bytes_values(module, 22)
        instructions = []
        computations = {}
        for computation in _bytes_values(module, 3):
            computation_id = _first_varint(computation, 5)
            computation_name = _first_text(computation, 1)
            computations[computation_id] = computation_name
            instructions.extend(
                _instruction(value, computation_id, computation_name, payloads)
                for value in _bytes_values(computation, 2)
            )
        return instructions, computations
    raise ValueError("no GpuExecutableProto record with an HLO module")


def _nested_thunks(thunk: bytes) -> list[bytes]:
    nested = []
    for container in _bytes_values(thunk, 2):
        nested.extend(_bytes_values(container, 1))
    for container in _bytes_values(thunk, 6):
        for sequence in _bytes_values(container, 2):
            nested.extend(_bytes_values(sequence, 1))
    for container in _bytes_values(thunk, 7):
        for number in (2, 3):
            for sequence in _bytes_values(container, number):
                nested.extend(_bytes_values(sequence, 1))
    for container in _bytes_values(thunk, 18):
        for sequence in _bytes_values(container, 1):
            nested.extend(_bytes_values(sequence, 1))
    for container in _bytes_values(thunk, 47):
        nested.extend(_bytes_values(container, 2))
    for container in _bytes_values(thunk, 57):
        for sequence in _bytes_values(container, 3):
            nested.extend(_bytes_values(sequence, 1))
    for container in _bytes_values(thunk, 60):
        for sequence in _bytes_values(container, 6):
            nested.extend(_bytes_values(sequence, 1))
    return nested


def _normalized(value: str) -> str:
    return value.replace(".", "_").lower()


def _thunk_summaries(thunk: bytes, path: str, targets: set[str]) -> list[dict[str, object]]:
    info = _first_bytes(thunk, 1)
    annotation = _first_text(info, 1)
    result = []
    impl_fields = [field for field in _fields(thunk) if field.number != 1 and isinstance(field.value, bytes)]
    for field in impl_fields:
        names = [annotation]
        summary: dict[str, object] = {
            "path": path,
            "impl_field": field.number,
            "annotation": annotation,
        }
        if field.number == 8:
            kernel_name = _first_text(field.value, 3)
            names.append(kernel_name)
            summary.update({"kind": "kernel", "kernel_name": kernel_name})
        elif field.number == 36:
            custom_kernel = _first_bytes(field.value, 3)
            custom_name = _first_text(custom_kernel, 1)
            kernel_spec = _first_bytes(custom_kernel, 2)
            kernel_name = _first_text(kernel_spec, 4)
            cubin = _first_bytes(_first_bytes(kernel_spec, 2), 1)
            names.extend((custom_name, kernel_name))
            summary.update(
                {
                    "kind": "custom_kernel",
                    "custom_name": custom_name,
                    "kernel_name": kernel_name,
                    "cubin_bytes": len(cubin),
                }
            )
        else:
            summary["kind"] = f"field_{field.number}"
        normalized_names = [_normalized(name) for name in names if name]
        if any(target in name or name in target for target in targets for name in normalized_names):
            result.append(summary)
    for index, child in enumerate(_nested_thunks(thunk)):
        result.extend(_thunk_summaries(child, f"{path}/{index}", targets))
    return result


def _print_thunk_matches(records: list[bytes], instructions: list[Instruction]) -> None:
    targets = set()
    for instruction in instructions:
        targets.add(_normalized(instruction.name))
        for key in ("deduplicated_name", "scheduling_name"):
            value = instruction.metadata.get(key)
            if isinstance(value, str):
                targets.add(_normalized(value))
    matches = []
    for record_index, record in enumerate(records):
        try:
            thunks = _bytes_values(record, 13)
        except ValueError:
            continue
        for thunk_index, thunk in enumerate(thunks):
            matches.extend(_thunk_summaries(thunk, f"record[{record_index}].thunk[{thunk_index}]", targets))
    for match in matches:
        print("THUNK " + json.dumps(match, sort_keys=True))


def _summary(instruction: Instruction, computations: dict[int, str]) -> dict[str, object]:
    return {
        "computation": instruction.computation_name,
        "name": instruction.name,
        "opcode": instruction.opcode,
        "id": instruction.instruction_id,
        "shape": instruction.shape,
        "fusion_kind": instruction.fusion_kind,
        "custom_call_target": instruction.custom_call_target,
        "operands": list(instruction.operand_ids),
        "called_computations": [
            {"id": computation_id, "name": computations.get(computation_id, "<unknown>")}
            for computation_id in instruction.called_computation_ids
        ],
        "metadata": instruction.metadata,
        "backend_config": instruction.backend_config,
    }


def analyze(path: str, target: str) -> None:
    with fsspec.open(path, "rb") as handle:
        records = _gpu_executable_records(handle.read())
    instructions, computations = _instructions(records)
    by_id = {instruction.instruction_id: instruction for instruction in instructions}
    matches = [
        instruction for instruction in instructions if instruction.name.replace(".", "_") == target.replace(".", "_")
    ]
    if not matches:
        candidates = [
            instruction
            for instruction in instructions
            if instruction.opcode == "fusion"
            and str(instruction.metadata.get("op_name", "")).endswith("/MoEMLP/convert_element_type")
        ]
        print(f"target={target} matches=0 analogous_candidates={len(candidates)}")
        for candidate in candidates:
            print("ANALOGOUS " + json.dumps(_summary(candidate, computations), sort_keys=True))
        _print_thunk_matches(records, candidates)
        return

    users: dict[int, list[Instruction]] = {}
    for instruction in instructions:
        for operand_id in instruction.operand_ids:
            users.setdefault(operand_id, []).append(instruction)

    print(f"artifact={path}")
    print(f"target={target} matches={len(matches)}")
    for match in matches:
        print("MATCH " + json.dumps(_summary(match, computations), sort_keys=True))
        for operand_id in match.operand_ids:
            operand = by_id.get(operand_id)
            if operand is not None:
                print("OPERAND " + json.dumps(_summary(operand, computations), sort_keys=True))
        for user in users.get(match.instruction_id, []):
            print("USER " + json.dumps(_summary(user, computations), sort_keys=True))
        for computation_id in match.called_computation_ids:
            members = [instruction for instruction in instructions if instruction.computation_id == computation_id]
            print(
                "CALLED_COMPUTATION "
                + json.dumps(
                    {
                        "id": computation_id,
                        "name": computations.get(computation_id, "<unknown>"),
                        "instruction_count": len(members),
                    },
                    sort_keys=True,
                )
            )
            for member in members:
                print("MEMBER " + json.dumps(_summary(member, computations), sort_keys=True))
    _print_thunk_matches(records, matches)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} CACHE_ENTRY KERNEL_NAME")
    analyze(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
