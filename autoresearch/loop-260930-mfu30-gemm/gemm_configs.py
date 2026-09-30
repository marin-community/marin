"""Extract distinct cuBLAS GEMM configurations (shapes, layouts, dims) from HLO text and join them with
trace rows to get per-step time, kernel names, and achieved PF/s.

Usage: gemm_configs.py <hlo.txt> <rows.pkl> <opnames.pkl> <toolkit dir> <out.json> [steps]
"""
import collections
import json
import pickle
import re
import sys

sys.path.insert(0, sys.argv[4])
from anatomy_lib import intersect, length, merge, phase, scope  # noqa: E402

txt = open(sys.argv[1]).read().splitlines()
steps = int(sys.argv[6]) if len(sys.argv) > 6 else 3
shape_re = re.compile(r"(bf16|f32|f16|f8e4m3fn|f8e5m2|s32|u32|s8|pred|u8|s64)\[([0-9,]*)\](\{[0-9,]*\})?")


def parse(s):
    m = shape_re.search(s)
    if not m:
        return None
    dims = [int(x) for x in m.group(2).split(",") if x]
    layout = [int(x) for x in m.group(3)[1:-1].split(",") if x] if m.group(3) else list(range(len(dims) - 1, -1, -1))
    return m.group(1), dims, layout


shapes = {}
calls = []
for line in txt:
    m = re.match(r"\s*(ROOT )?%?([\w.\-]+) = (.*)", line)
    if not m:
        continue
    name, rest = m.group(2), m.group(3)
    head = rest.split(" ", 1)[0] if not rest.startswith("(") else rest[: rest.index(")") + 1]
    p = parse(head)
    if p:
        shapes[name] = p
    if 'custom_call_target="__cublas' in rest:
        calls.append((name, rest))

cfg = {}
for name, rest in calls:
    args = rest.split("custom-call(", 1)[1].split(")", 1)[0]
    ops = [a.strip().lstrip("%") for a in args.split(",")]
    if ops[0] not in shapes or ops[1] not in shapes:
        continue
    gbc = json.loads(re.search(r"backend_config=(\{.*\})", rest).group(1).rsplit("}", 0)[0])["gemm_backend_config"]
    dd = gbc["dot_dimension_numbers"]
    cfg[name] = dict(
        lhs=shapes[ops[0]],
        rhs=shapes[ops[1]],
        out=shapes[name],
        lc=[int(x) for x in dd["lhs_contracting_dimensions"]],
        rc=[int(x) for x in dd["rhs_contracting_dimensions"]],
        lb=[int(x) for x in dd.get("lhs_batch_dimensions", [])],
        rb=[int(x) for x in dd.get("rhs_batch_dimensions", [])],
        epilogue=gbc.get("epilogue"),
        beta=gbc.get("beta"),
        alg=gbc.get("selected_algorithm"),
        target=re.search(r'custom_call_target="([^"]+)"', rest).group(1),
    )

d = pickle.load(open(sys.argv[2], "rb"))
opn = pickle.load(open(sys.argv[3], "rb"))
rows = [r for r in d["rows"] if r[6] == "jit_train_step"]
ragged = merge([(r[2], r[3]) for r in rows if "RaggedAllToAll" in r[1]])
agg = {}
for r in rows:
    c = cfg.get(r[5])
    if c is None:
        continue
    rec = opn.get(r[5])
    op = rec[1] if rec else ""
    key = json.dumps([c["lhs"], c["rhs"], c["out"], c["lc"], c["rc"], c["lb"], c["rb"], c["epilogue"], c["beta"]])
    a = agg.setdefault(key, dict(cfg=c, t=0, n=0, tov=0, kern=collections.Counter(), scopes=collections.Counter(), tno=0, nno=0))
    dt = (r[3] - r[2]) * 1e-12
    ov = length(intersect([[r[2], r[3]]], ragged)) * 1e-12
    a["t"] += dt
    a["n"] += 1
    a["tov"] += ov
    if ov < 0.05 * dt:
        a["tno"] += dt
        a["nno"] += 1
    a["kern"][r[1][:70]] += dt
    a["scopes"][f"{scope(op)}:{phase(op)}"] += dt

out = []
for key, a in agg.items():
    c = a["cfg"]
    lhs, lc, lb = c["lhs"][1], c["lc"], c["lb"]
    k = 1
    for i in lc:
        k *= lhs[i]
    o = 1
    for x in c["out"][1]:
        o *= x
    flops = 2 * o * k
    out.append(
        dict(
            **{kk: c[kk] for kk in ("lhs", "rhs", "out", "lc", "rc", "lb", "rb", "epilogue", "beta", "alg", "target")},
            flops=flops,
            t_step=a["t"] / steps,
            n_step=a["n"] / steps,
            mean_us=1e6 * a["t"] / a["n"],
            pfs=flops * a["n"] / a["t"] / 1e15,
            pfs_nonovl=(flops * a["nno"] / a["tno"] / 1e15) if a["tno"] else None,
            ovl_frac=a["tov"] / a["t"],
            kernels=a["kern"].most_common(3),
            scopes=a["scopes"].most_common(4),
        )
    )
out.sort(key=lambda x: -x["t_step"])
json.dump(out, open(sys.argv[5], "w"), indent=1)
tot = sum(x["t_step"] for x in out)
print(f"distinct configs {len(out)}; total cuBLAS time/step {tot:.3f}s")
for x in out[:40]:
    print(
        f"{x['t_step']:6.3f}s n={x['n_step']:5.1f} {x['mean_us']:8.0f}us {x['pfs']:.2f}PF/s nonovl={x['pfs_nonovl'] or 0:.2f} ovl={x['ovl_frac']:.2f} "
        f"lhs={x['lhs'][1]}{x['lhs'][2]} rhs={x['rhs'][1]}{x['rhs'][2]} out={x['out'][1]}{x['out'][2]} lc={x['lc']} rc={x['rc']} lb={x['lb']} ep={x['epilogue']} beta={x['beta']}"
    )
    print(f"        {x['kernels'][0][0]}  | {x['scopes'][:2]}")
