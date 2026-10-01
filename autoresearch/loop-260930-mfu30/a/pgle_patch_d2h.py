"""Rewrite a PGLE profile so the LHS hides the end-of-step optimizer-state D2H copies.

XLA:GPU's latency-hiding scheduler models async memcpys with unlimited concurrency
(kNumAsyncMemcpy = INT_MAX in gpu_latency_hiding_scheduler.cc). The hero's step ends with ~36 GiB of
pinned-host writebacks (three 10 GiB expert momentum shards, embedding Adam m/v) that in reality
serialize on the C2C link (~155 GB/s, ~70 ms each), so a profile with each copy's own duration lets
the scheduler cover only ~70 ms of the ~250 ms tail. This sets every large D2H `copy-start` in the
train step to the serialized total of the batch, so each copy-start lands early enough for the whole
batch to drain under the optimizer compute. D2H device buffers live until their copy-done either
way, so the change costs no HBM. H2D copies are left alone: issuing those earlier does hold HBM.

    uv run python pgle_patch_d2h.py <rows.pkl from tfop_dump.py> <in.pbtxt> <out.pbtxt> [min_ms=1]
"""

import collections
import pickle
import re
import sys

MARGIN = 1.1


def d2h_copy_costs_us(rows_path: str, min_ms: float) -> dict[str, float]:
    data = pickle.load(open(rows_path, "rb"))
    steps = len(data["launches"])
    busy = collections.Counter()
    for stream, name, start, end, _tf_op, hlo_op, module, *_ in data["rows"]:
        if module == "jit_train_step" and name.startswith("MemcpyD2H") and re.match(r"copy-start\.", hlo_op or ""):
            busy[hlo_op] += end - start
    per_step_us = {op: ps / 1e6 / steps for op, ps in busy.items()}
    return {op: us for op, us in per_step_us.items() if us >= min_ms * 1e3}


def main(rows_path: str, in_path: str, out_path: str, min_ms: str = "1") -> None:
    costs = d2h_copy_costs_us(rows_path, float(min_ms))
    if not costs:
        raise ValueError("no large D2H copy-starts in the trace rows")
    total = sum(costs.values()) * MARGIN
    text = open(in_path).read()
    patched = 0

    def rewrite(match: re.Match) -> str:
        nonlocal patched
        name = match.group(1)
        if name not in costs:
            return match.group(0)
        patched += 1
        return f'name: "{name}"\n  cost_us: {total}'

    out = re.sub(r'name: "([^"]+)"\n\s*cost_us: [0-9.eE+-]+', rewrite, text)
    if patched != len(costs):
        raise ValueError(f"patched {patched} of {len(costs)} D2H copy-starts; profile and trace disagree")
    open(out_path, "w").write(out)
    for name, us in sorted(costs.items(), key=lambda kv: -kv[1]):
        print(f"{name}: {us / 1e3:.1f} ms -> {total / 1e3:.1f} ms")


if __name__ == "__main__":
    main(*sys.argv[1:])
