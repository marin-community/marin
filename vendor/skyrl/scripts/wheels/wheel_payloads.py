"""Rewrite wheel metadata while proving preservation of its native payloads.

Derived from MarinSkyRL 604f2c1463032ffc1f47da114f43cf1d550dd473
scripts/wheels/wheel_payloads.py; adds explicit non-native member removal.
"""

import base64
import copy
import csv
import hashlib
import io
import zipfile
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repack(
    wheel: Path,
    source_sha256: str,
    destination: Path,
    old_info: str,
    new_info: str,
    changes: dict[str, bytes],
    machine: int,
    removed_members: frozenset[str] = frozenset(),
) -> dict:
    """Rewrite specified members and RECORD, preserving every other payload byte."""
    assert digest(wheel.read_bytes()) == source_sha256
    assert not destination.exists()
    proof = {
        "input_wheel": wheel.name,
        "input_sha256": source_sha256,
        "output_wheel": destination.name,
        "unchanged_payloads": {},
        "native_members": {},
        "removed_members": {},
    }
    with zipfile.ZipFile(wheel) as source:
        members = source.infolist()
        assert len({item.filename for item in members}) == len(members)
        assert set(changes) <= set(source.namelist())
        assert removed_members <= set(source.namelist())
        assert not removed_members.intersection(changes)
        assert not any(name.startswith(old_info) for name in removed_members)
        assert old_info + "METADATA" in source.namelist()
        assert old_info + "RECORD" in source.namelist()
        assert old_info + "RECORD" not in changes
        payloads = {}
        for item in members:
            if item.filename == old_info + "RECORD":
                continue
            original = source.read(item.filename)
            if item.filename in removed_members:
                assert not original.startswith(b"\x7fELF"), "Cannot remove a native payload"
                proof["removed_members"][item.filename] = digest(original)
                continue
            data = changes.get(item.filename, original)
            name = item.filename.replace(old_info, new_info, 1) if item.filename.startswith(old_info) else item.filename
            if item.filename not in changes:
                proof["unchanged_payloads"][item.filename] = digest(data)
            if original.startswith(b"\x7fELF"):
                assert data == original
                assert data[4:6] == b"\x02\x01"
                assert int.from_bytes(data[18:20], "little") == machine
                proof["native_members"][item.filename] = digest(data)
            payloads[name] = data
        assert proof["native_members"]
        record = io.StringIO(newline="")
        writer = csv.writer(record, lineterminator="\n")
        for name, data in payloads.items():
            encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            writer.writerow((name, "sha256=" + encoded, len(data)))
        writer.writerow((new_info + "RECORD", "", ""))
        payloads[new_info + "RECORD"] = record.getvalue().encode()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w") as target:
            for item in members:
                if item.filename in removed_members:
                    continue
                renamed = copy.copy(item)
                if renamed.filename.startswith(old_info):
                    renamed.filename = renamed.filename.replace(old_info, new_info, 1)
                target.writestr(renamed, payloads[renamed.filename])
        with zipfile.ZipFile(destination) as target:
            assert {item.filename for item in target.infolist()} == set(payloads)
            for item in members:
                if item.filename in removed_members:
                    continue
                new_name = (
                    item.filename.replace(old_info, new_info, 1)
                    if item.filename.startswith(old_info)
                    else item.filename
                )
                assert target.read(new_name) == payloads[new_name]
                assert target.getinfo(new_name).date_time == item.date_time
    proof["output_sha256"] = digest(destination.read_bytes())
    proof["changed_members"] = [*changes, old_info + "RECORD"]
    return proof
