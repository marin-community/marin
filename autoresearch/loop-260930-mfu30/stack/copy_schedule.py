# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Main-level schedule of the train step: while loops and large async copies.

Shows where the scheduler put each large host copy relative to the forward and backward loops. On main the
first momentum H2D (`copy-start.44`, 10.1 GiB) starts after the backward loop; if it sits before the
forward loop it holds 10 GiB through the backward peak.

Usage: copy_schedule.py <train_step.hloproto.pb> [min_gib]
"""

import sys

from google.protobuf.internal import decoder

BYTES_PER_ELEMENT = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 2, 8: 4, 9: 8, 10: 2, 11: 4, 12: 8, 16: 2}
GIB = 2**30


def fields(buf):
    pos = 0
    while pos < len(buf):
        tag, pos = decoder._DecodeVarint(buf, pos)
        num, wire = tag >> 3, tag & 7
        if wire == 0:
            value, pos = decoder._DecodeVarint(buf, pos)
        elif wire == 2:
            size, pos = decoder._DecodeVarint(buf, pos)
            value, pos = buf[pos : pos + size], pos + size
        elif wire == 1:
            value, pos = buf[pos : pos + 8], pos + 8
        elif wire == 5:
            value, pos = buf[pos : pos + 4], pos + 4
        else:
            raise ValueError(wire)
        yield num, value


def packed(value):
    if isinstance(value, int):
        return [value]
    out, pos = [], 0
    while pos < len(value):
        x, pos = decoder._DecodeVarint(value, pos)
        out.append(x)
    return out


def first_array_bytes(shape):
    element_type, dims, tuple_shapes = None, [], []
    for num, value in fields(shape):
        if num == 2:
            element_type = value
        elif num == 3:
            dims += packed(value)
        elif num == 4:
            tuple_shapes.append(value)
    if tuple_shapes:
        return first_array_bytes(tuple_shapes[0])
    n = 1
    for d in dims:
        n *= d
    return n * BYTES_PER_ELEMENT.get(element_type, 0)


def main(path: str, min_gib: float) -> None:
    module = next(v for f, v in fields(open(path, "rb").read()) if f == 1)
    instructions, main_id, schedules = {}, None, {}
    for num, value in fields(module):
        if num == 3:
            comp_fields = list(fields(value))
            name = next(v for f, v in comp_fields if f == 1).decode()
            comp_id = next(v for f, v in comp_fields if f == 5)
            if name.startswith("main"):
                main_id = comp_id
            for f, instr in comp_fields:
                if f != 2:
                    continue
                rec = {"op_name": ""}
                for f3, v3 in fields(instr):
                    if f3 == 1:
                        rec["name"] = v3.decode()
                    elif f3 == 2:
                        rec["opcode"] = v3.decode()
                    elif f3 == 3:
                        rec["bytes"] = first_array_bytes(v3)
                    elif f3 == 35:
                        rec["id"] = v3
                    elif f3 == 7:
                        rec["op_name"] = next((v7.decode() for f7, v7 in fields(v3) if f7 == 2), "")
                instructions[rec["id"]] = rec
        elif num == 7:
            for f, entry in fields(value):
                if f != 1:
                    continue
                key, seq = None, []
                for fk, vk in fields(entry):
                    if fk == 1:
                        key = vk
                    elif fk == 2:
                        for fs, vs in fields(vk):
                            if fs == 1:
                                seq += packed(vs)
                schedules[key] = seq
    seq = schedules[main_id]
    print(f"main: {len(seq)} scheduled instructions")
    for pos, iid in enumerate(seq):
        rec = instructions[iid]
        if rec["opcode"] == "while":
            print(f"  {pos:6d} {rec['name']:24s} while  {rec['op_name'][-60:]}")
        elif rec["opcode"] in ("copy-start", "copy-done") and rec.get("bytes", 0) >= min_gib * GIB:
            print(f"  {pos:6d} {rec['name']:24s} {rec['bytes'] / GIB:6.2f} GiB")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
