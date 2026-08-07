import lldb
import struct
from collections import Counter

CLIENT_POOL = 0xFC7D2DC90000
SERVICE_POOLS = {
    "e9e17e30": 0xFC7E51F50000,
    "fafa90d0": 0xFC7D35F50000,
    "f1f243f0": 0xFC7E4CE70000,
    "f5dce020": 0xFC7E653F0000,
}
NRANKS = 72
MAXPER = 2048
OPSZ = 232
OFF_NEXT = 28
OFF_OPCOUNT = 16
TPRANK = 20


def val(target, expr):
    v = target.EvaluateExpression(expr)
    if v.GetError().Success():
        return v.GetValue()
    print(f"EXPR-FAIL {expr}: {v.GetError()}")
    return None


def __lldb_init_module(debugger, d):
    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    err = lldb.SBError()

    # field offsets inside the pool struct
    off_nextops = int(val(target, "(unsigned long)&((struct ncclProxyOpsPool*)0)->nextOps"))
    off_freeops = int(val(target, "(unsigned long)&((struct ncclProxyOpsPool*)0)->freeOps[0]"))
    print(f"POOLOFF nextOps={off_nextops} freeOps={off_freeops}")

    def fingerprint(addr, label):
        b = process.ReadMemory(addr + off_nextops, 8, err)
        if not err.Success():
            print(f"{label}: READ-FAIL {err}")
            return None
        nxt, nxe = struct.unpack("<ii", b)
        fb = process.ReadMemory(addr + off_freeops, 4 * NRANKS, err)
        frees = struct.unpack(f"<{NRANKS}i", fb)
        nz = {i: f for i, f in enumerate(frees) if f != -1}
        print(f"{label} @0x{addr:x}: nextOps={nxt} nextOpsEnd={nxe} freeOps(non--1)={nz}")
        return (nxt, nxe, tuple(frees))

    fps = {}
    fps["client"] = fingerprint(CLIENT_POOL, "CLIENTPOOL")
    for k, a in SERVICE_POOLS.items():
        fps[k] = fingerprint(a, f"SERVICE {k}")
    for k in SERVICE_POOLS:
        if fps[k] is not None and fps["client"] is not None:
            print(f"MATCH client=={k}: {fps[k] == fps['client']}")

    # partition census at client VA
    lo = TPRANK * MAXPER
    buf = process.ReadMemory(CLIENT_POOL + off_freeops + 4 * NRANKS + 0, 1, err)  # noop warm
    arr = int(val(target, f"(unsigned long)&((struct ncclProxyOpsPool*){CLIENT_POOL})->ops[0]"))
    buf = process.ReadMemory(arr + lo * OPSZ, MAXPER * OPSZ, err)
    if not err.Success():
        print(f"PART-READ-FAIL {err}")
        return
    nxt = {}
    ocs = {}
    for k in range(MAXPER):
        s = lo + k
        rec = buf[k * OPSZ:(k + 1) * OPSZ]
        nxt[s] = struct.unpack_from("<i", rec, OFF_NEXT)[0]
        ocs[s] = struct.unpack_from("<Q", rec, OFF_OPCOUNT)[0]

    pointed = set(v for v in nxt.values() if v != -1)
    heads = [s for s in nxt if s not in pointed]
    print(f"PARTITION[{TPRANK}] slots={MAXPER} heads={len(heads)} pointed={len(pointed)}")
    for h in sorted(heads):
        chain = []
        s = h
        seen = set()
        while s != -1 and s in nxt and s not in seen:
            seen.add(s)
            chain.append(s)
            s = nxt[s]
        tail = "end" if s == -1 else ("cycle" if s in seen else f"offpart:{s}")
        c_ocs = [ocs[c] for c in chain]
        print(f"HEAD {h}: len={len(chain)} tail={tail} opCount[min,max]=({min(c_ocs)},{max(c_ocs)})")
        if len(chain) <= 6:
            print(f"  chain={chain} ocs={c_ocs}")
    print("WALK3-DONE")
