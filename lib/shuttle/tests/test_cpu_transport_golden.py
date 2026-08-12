# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

_GOLDEN = Path(__file__).parents[1] / "mlir" / "test" / "Inputs" / "cpu-forward-7x13-transport.hex"
_DISTINCT_GOLDEN = _GOLDEN.with_name("cpu-forward-7x13-epsilon-quarter-transport.hex")
_EXPECTED_SIZE = 6_899
_EXPECTED_SHA256 = "99e63ac5a004f5abce7b88fc12bd0fbf9d8fc14785fc9ae87ca32781165d0c31"
_DIGEST = re.compile(r"[0-9a-f]{64}", re.ASCII)
_MAXIMUM_RECORDS = 4_096


@dataclass(frozen=True)
class Tensor:
    element: int
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Entry:
    ordinal: int
    source_task: int
    code_offset: int
    code_length: int
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    input_accesses: tuple[int, ...]
    output_accesses: tuple[int, ...]
    dependencies: tuple[int, ...]
    predication: int
    reduction_order: int | None
    code_digest: str


@dataclass(frozen=True)
class Slot:
    ordinal: int
    source_buffer: int
    tensor: Tensor
    required_bytes: int
    strides: tuple[int, ...]
    offset: int
    alignment: int
    address_space: int
    access: int
    storage: int
    alias_group: int
    reuse_group: int
    binding: int
    binding_index: int | None


@dataclass(frozen=True)
class Transport:
    device_schema: int
    code_format: int
    policy: int
    schedule_fingerprint: str
    code: bytes
    code_digest: str
    device_fingerprint: str
    entries: tuple[Entry, ...]
    abi_schema: int
    plan_fingerprint: str
    abi_schedule_fingerprint: str
    abi_fingerprint: str
    slots: tuple[Slot, ...]
    bundle_schema: int
    bundle_schedule_fingerprint: str
    bundle_device_fingerprint: str
    bundle_abi_fingerprint: str
    completion: int
    bundle_fingerprint: str


class Cursor:
    def __init__(self, payload: bytes):
        self._payload = payload
        self._position = 0

    def take(self, size: int) -> bytes:
        if type(size) is not int or size < 0 or size > len(self._payload) - self._position:
            raise ValueError("transport field exceeds the payload")
        value = self._payload[self._position : self._position + size]
        self._position += size
        return value

    def u8(self, maximum: int | None = None) -> int:
        value = int.from_bytes(self.take(1), "little")
        if maximum is not None and value > maximum:
            raise ValueError("transport enum is outside its closed range")
        return value

    def u32(self) -> int:
        return int.from_bytes(self.take(4), "little")

    def u64(self) -> int:
        return int.from_bytes(self.take(8), "little")

    def count(self) -> int:
        value = self.u64()
        if value > _MAXIMUM_RECORDS:
            raise ValueError("transport record count exceeds the closed limit")
        return value

    def i64(self) -> int:
        return int.from_bytes(self.take(8), "little", signed=True)

    def byte_array(self) -> bytes:
        return self.take(self.u64())

    def string(self) -> str:
        return self.byte_array().decode("utf-8")

    def digest(self) -> str:
        value = self.string()
        if _DIGEST.fullmatch(value) is None:
            raise ValueError("transport digest is not lowercase SHA-256")
        return value

    def i64_array(self) -> tuple[int, ...]:
        return tuple(self.i64() for _ in range(self.count()))

    def tensor(self) -> Tensor:
        return Tensor(element=self.u8(1), shape=self.i64_array())

    def optional_enum(self, maximum: int) -> int | None:
        present = self.u8(1)
        return self.u8(maximum) if present else None

    def optional_i64(self) -> int | None:
        present = self.u8(1)
        return self.i64() if present else None

    def finish(self) -> None:
        if self._position != len(self._payload):
            raise ValueError("transport has trailing bytes")


def _entry(cursor: Cursor) -> Entry:
    return Entry(
        ordinal=cursor.i64(),
        source_task=cursor.i64(),
        code_offset=cursor.i64(),
        code_length=cursor.i64(),
        inputs=cursor.i64_array(),
        outputs=cursor.i64_array(),
        input_accesses=tuple(cursor.u8(2) for _ in range(cursor.count())),
        output_accesses=tuple(cursor.u8(2) for _ in range(cursor.count())),
        dependencies=cursor.i64_array(),
        predication=cursor.u8(1),
        reduction_order=cursor.optional_enum(1),
        code_digest=cursor.digest(),
    )


def _slot(cursor: Cursor) -> Slot:
    return Slot(
        ordinal=cursor.i64(),
        source_buffer=cursor.i64(),
        tensor=cursor.tensor(),
        required_bytes=cursor.i64(),
        strides=cursor.i64_array(),
        offset=cursor.i64(),
        alignment=cursor.i64(),
        address_space=cursor.u8(1),
        access=cursor.u8(2),
        storage=cursor.u8(1),
        alias_group=cursor.i64(),
        reuse_group=cursor.i64(),
        binding=cursor.u8(2),
        binding_index=cursor.optional_i64(),
    )


def decode_transport(payload: bytes) -> Transport:
    if len(payload) > 16 * 1024 * 1024:
        raise ValueError("transport exceeds the closed size limit")
    cursor = Cursor(payload)
    if cursor.take(8) != b"SHUTCPU\0" or cursor.u32() != 1:
        raise ValueError("transport magic or version mismatch")
    transport = Transport(
        device_schema=cursor.i64(),
        code_format=cursor.u8(0),
        policy=cursor.u8(1),
        schedule_fingerprint=cursor.digest(),
        code=cursor.byte_array(),
        code_digest=cursor.digest(),
        device_fingerprint=cursor.digest(),
        entries=tuple(_entry(cursor) for _ in range(cursor.count())),
        abi_schema=cursor.i64(),
        plan_fingerprint=cursor.digest(),
        abi_schedule_fingerprint=cursor.digest(),
        abi_fingerprint=cursor.digest(),
        slots=tuple(_slot(cursor) for _ in range(cursor.count())),
        bundle_schema=cursor.i64(),
        bundle_schedule_fingerprint=cursor.digest(),
        bundle_device_fingerprint=cursor.digest(),
        bundle_abi_fingerprint=cursor.digest(),
        completion=cursor.u8(0),
        bundle_fingerprint=cursor.digest(),
    )
    cursor.finish()
    if (transport.device_schema, transport.abi_schema, transport.bundle_schema) != (1, 2, 1):
        raise ValueError("transport object schema mismatch")
    if transport.code_digest != hashlib.sha256(transport.code).hexdigest():
        raise ValueError("transport code digest mismatch")
    if not (
        transport.schedule_fingerprint == transport.abi_schedule_fingerprint == transport.bundle_schedule_fingerprint
    ):
        raise ValueError("transport schedule roots disagree")
    if transport.device_fingerprint != transport.bundle_device_fingerprint:
        raise ValueError("transport device roots disagree")
    if transport.abi_fingerprint != transport.bundle_abi_fingerprint:
        raise ValueError("transport ABI roots disagree")
    if tuple(entry.ordinal for entry in transport.entries) != tuple(range(len(transport.entries))):
        raise ValueError("transport entry ordinals are not contiguous")
    if tuple(slot.ordinal for slot in transport.slots) != tuple(range(len(transport.slots))):
        raise ValueError("transport slot ordinals are not contiguous")
    if any(slot.ordinal != slot.source_buffer for slot in transport.slots):
        raise ValueError("transport slot/source binding mismatch")
    return transport


def _golden_bytes() -> bytes:
    return _hex_bytes(_GOLDEN)


def _hex_bytes(path: Path) -> bytes:
    text = path.read_text()
    if re.fullmatch(r"(?:[0-9a-f]{64}\n)*[0-9a-f]+\n", text, re.ASCII) is None:
        raise ValueError("transport golden must be canonical lowercase hex")
    return bytes.fromhex(text)


def test_forward_transport_freezes_canonical_bytes_and_external_bindings() -> None:
    payload = _golden_bytes()
    assert len(payload) == _EXPECTED_SIZE
    assert hashlib.sha256(payload).hexdigest() == _EXPECTED_SHA256

    transport = decode_transport(payload)
    assert (transport.device_schema, transport.abi_schema, transport.bundle_schema) == (1, 2, 1)
    assert transport.code_digest == hashlib.sha256(transport.code).hexdigest()
    assert transport.schedule_fingerprint == transport.abi_schedule_fingerprint
    assert transport.schedule_fingerprint == transport.bundle_schedule_fingerprint
    assert transport.device_fingerprint == transport.bundle_device_fingerprint
    assert transport.abi_fingerprint == transport.bundle_abi_fingerprint
    assert tuple(entry.ordinal for entry in transport.entries) == tuple(range(19))
    assert tuple(slot.ordinal for slot in transport.slots) == tuple(range(21))

    external = tuple(slot for slot in transport.slots if slot.binding != 0)
    assert tuple((slot.ordinal, slot.binding, slot.binding_index) for slot in external) == (
        (0, 1, 0),
        (1, 1, 1),
        (20, 2, 0),
    )
    assert tuple((slot.tensor.element, slot.tensor.shape) for slot in external) == (
        (0, (7, 13)),
        (0, (13,)),
        (0, (7, 13)),
    )
    assert tuple((slot.required_bytes, slot.strides) for slot in external) == (
        (182, (26, 2)),
        (26, (2,)),
        (182, (26, 2)),
    )


def test_distinct_forward_transport_has_same_schema_and_distinct_identity() -> None:
    baseline = _golden_bytes()
    distinct = _hex_bytes(_DISTINCT_GOLDEN)
    assert len(distinct) == _EXPECTED_SIZE
    assert hashlib.sha256(distinct).hexdigest() == ("8613f1c0fef79d343ec5dc161ec3a1ee458342b27dc381b7b371adbad5c9c15d")
    assert distinct != baseline
    decoded = decode_transport(distinct)
    assert len(decoded.entries) == 19
    assert len(decoded.slots) == 21


@pytest.mark.parametrize(
    ("offset", "replacement"),
    [
        (0, ord("X")),
        (8, 2),
        (20, 1),
        (12, 2),
    ],
)
def test_transport_oracle_rejects_closed_header_and_enum_mutations(offset: int, replacement: int) -> None:
    payload = bytearray(_golden_bytes())
    payload[offset] = replacement
    with pytest.raises(ValueError):
        decode_transport(bytes(payload))


def test_transport_oracle_rejects_truncation_and_trailing_bytes() -> None:
    payload = _golden_bytes()
    with pytest.raises(ValueError):
        decode_transport(payload[:-1])
    with pytest.raises(ValueError, match="trailing"):
        decode_transport(payload + b"\0")
