import lldb
import struct

# service-VA pools and the partition their OWN rank appends to (rank84..87 -> tp 20..23)
POOLS = [
    ("fafa90d0/rank84", 0xFC7D35F50000, 20),
    ("e9e17e30", 0xFC7E51F50000, None),
    ("f1f243f0", 0xFC7E4CE70000, None),
    ("f5dce020", 0xFC7E653F0000, None),
]
NRANKS = 72
MAXPER = 2048
OPSZ = 232
OFF_NEXT = 28
OFF_OPCOUNT = 16


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

    off_nextops = int(val(target, "(unsigned long)&((struct ncclProxyOpsPool*)0)->nextOps"))
    off_freeops = int(val(target, "(unsigned long)&((struct ncclProxyOpsPool*)0)->freeOps[0]"))
    off_arr = int(val(target, "(unsigned long)&((struct ncclProxyOpsPool*)0)->ops[0]"))

    for label, pool, hint in POOLS:
        fb = process.ReadMemory(pool + off_freeops, 4 * NRANKS, err)
        frees = struct.unpack(f"<{NRANKS}i", fb)
        # find in-use partitions: head != r*MAXPER (pristine) and head != -1... check all with head != pristine
        for r in range(64):
            head = frees[r]
            pristine = r * MAXPER
            if head == pristine:
                continue  # untouched partition
            # read partition r
            buf = process.ReadMemory(pool + off_arr + r * MAXPER * OPSZ, MAXPER * OPSZ, err)
            if not err.Success():
                print(f"{label} part{r}: READ-FAIL")
                continue
            nxt = {}
            for k in range(MAXPER):
                nxt[r * MAXPER + k] = struct.unpack_from("<i", buf, k * OPSZ + OFF_NEXT)[0]
            # walk free chain from head
            n = 0
            s = head
            seen = set()
            while s != -1 and s in nxt and s not in seen:
                seen.add(s)
                n += 1
                s = nxt[s]
            wild = s != -1 and s not in nxt
            print(f"{label} partition {r}: freeHead={head} freeChainLen={n} "
                  f"deficit={MAXPER - n}{' TAIL-ESCAPES-PARTITION' if wild else ''}")
    print("WALK4-DONE")
