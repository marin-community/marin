import lldb

COMM = "((struct ncclComm*)0xfc812001a130)"


def ev(target, expr):
    v = target.EvaluateExpression(expr)
    err = v.GetError()
    if err.Success():
        return v
    print(f"EXPR-FAIL {expr}: {err}")
    return None


def val(target, expr):
    v = ev(target, expr)
    if v is None:
        return None
    s = v.GetValue()
    if s is None:
        s = v.GetSummary()
    return s


def __lldb_init_module(debugger, d):
    target = debugger.GetSelectedTarget()
    process = target.GetProcess()

    # 1. find the thread stuck in ncclLocalOpAppend
    for th in process:
        names = [fr.GetFunctionName() or "" for fr in th]
        if any("ncclLocalOpAppend" in n for n in names):
            print(f"LAGTHREAD tid={th.GetThreadID()} idx={th.GetIndexID()}")

    # 2. comm-level facts
    for e in ["rank", "nRanks", "localRank", "localRanks", "opCount"]:
        print(f"comm.{e} = {val(target, COMM + '->' + e)}")
    tpl = val(target, COMM + "->topParentLocalRanks[" + (val(target, COMM + "->localRank") or "0") + "]")
    print(f"tpLocalRank = {tpl}")
    n = val(target, COMM + "->proxyState->tpLocalnRanks")
    print(f"tpLocalnRanks = {n}")
    n = int(n) if n else 0

    # 3. per-connection proxyOps table: find the stuck one
    for i in range(n):
        base = f"{COMM}->proxyState->proxyOps[{i}]"
        pool = val(target, base + ".pool")
        cnt = val(target, base + ".count")
        fo = val(target, base + ".freeOp")
        no = val(target, base + ".nextOps")
        ne = val(target, base + ".nextOpsEnd")
        if pool and pool != "0x0000000000000000":
            print(f"proxyOps[{i}]: pool={pool} count={cnt} freeOp={fo} nextOps={no} nextOpsEnd={ne}")

    # 4. for each distinct pool, the shared fields
    seen = set()
    for i in range(n):
        base = f"{COMM}->proxyState->proxyOps[{i}]"
        pool = val(target, base + ".pool")
        if not pool or pool == "0x0000000000000000" or pool in seen:
            continue
        seen.add(pool)
        p = f"((struct ncclProxyOpsPool*){pool})"
        print(f"POOL {pool}: nextOps={val(target, p + '->nextOps')} nextOpsEnd={val(target, p + '->nextOpsEnd')}")
        if tpl is not None:
            print(f"POOL {pool}: freeOps[{tpl}]={val(target, p + '->freeOps[' + tpl + ']')}")

    print("WALK-DONE")
