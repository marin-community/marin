"""Per-fusion table for memory-bound scopes: time/step, bytes, TB/s, phase, JAX op, operand/output shapes.

Usage: fusion_detail.py <prof dir> <toolkit dir> <scope regex> [top]
Needs rows.pkl, opnames.pkl, train_step.hlo.txt in <prof dir>.
"""
import collections
import pickle
import re
import sys

P, TK, SCOPE = sys.argv[1], sys.argv[2], re.compile(sys.argv[3])
TOP = int(sys.argv[4]) if len(sys.argv) > 4 else 60
SORT = sys.argv[5] if len(sys.argv) > 5 else "time"  # or "excess": time above bytes / 7 TB/s
HBM = 7.0e12
sys.path.insert(0, TK)
from anatomy_lib import phase, scope  # noqa: E402

BY = {"bf16": 2, "f32": 4, "f16": 2, "s32": 4, "u32": 4, "s8": 1, "u8": 1, "pred": 1, "s64": 8, "u64": 8, "s16": 2, "u16": 2}
shape_re = re.compile(r"(bf16|f32|f16|s32|u32|s8|u8|pred|s64|u64|s16|u16)\[([0-9,]*)\]")


def nbytes(sh):
    t = 0
    for dt, dd in sh:
        n = 1
        for x in dd.split(","):
            if x:
                n *= int(x)
        t += n * BY[dt]
    return t


txt = open(f"{P}/train_step.hlo.txt").read().splitlines()
defs = {}
for line in txt:
    m = re.match(r"\s*(ROOT )?%?([\w.\-]+) = (\([^)]*\)|\S+) ([\w\-]+)\((.*)", line)
    if not m:
        continue
    name, shp, opc, rest = m.group(2), m.group(3), m.group(4), m.group(5)
    args = [a.strip().lstrip("%") for a in rest.split(")", 1)[0].split(",") if a.strip()]
    calls = re.search(r"calls=([\w.\-]+)", rest)
    defs[name] = dict(shape=shp, opc=opc, args=args, calls=calls.group(1) if calls else None, kind=(re.search(r"kind=(\w+)", rest) or [None, None])[1])
d = pickle.load(open(f"{P}/rows.pkl", "rb"))
ops = pickle.load(open(f"{P}/opnames.pkl", "rb"))
rows = [r for r in d["rows"] if r[6] == "jit_train_step" and r[0].startswith("Stream #17")]
agg = collections.defaultdict(lambda: [0, 0, 0])
for r in rows:
    df = defs.get(r[5])
    if df is None or df["opc"] != "fusion":
        continue
    rec = ops.get(r[5])
    op = rec[1] if rec else ""
    if not SCOPE.search(scope(op)):
        continue
    a = agg[r[5]]
    a[0] += (r[3] - r[2]) * 1e-12 / 3
    a[1] += 1 / 3
    a[2] = r[1][:40]
tot_t = tot_b = 0
lines = []
for name, (t, n, kern) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
    df = defs[name]
    ins = [(a, defs.get(a, {}).get("shape", "?")) for a in df["args"]]
    b_in = sum(nbytes(shape_re.findall(s)) for _, s in ins)
    b_out = nbytes(shape_re.findall(df["shape"]))
    b = (b_in + b_out) * n
    tot_t += t
    tot_b += b
    rec = ops.get(name)
    op = rec[1] if rec else ""
    key = t - b / HBM if SORT == "excess" else t
    xr = " XLA-remat" if ".remat" in name else ""
    lines.append((key, f"{t:6.3f}s excess {t - b / HBM:6.3f}s n={n:4.0f} {b/1e9:7.1f}GB {b/t/1e12 if t else 0:5.2f}TB/s {scope(op)}:{phase(op)}{xr} {df['kind']} {name}\n"
                  f"      out {df['shape'][:90]}\n      in  {' '.join(s[:40] for _, s in ins)[:200]}\n      op  {op[-150:]}"))
print(f"total {tot_t:.3f} s/step, {tot_b/1e9:.0f} GB/step, {tot_b/tot_t/1e12:.2f} TB/s over {len(lines)} fusions")
lines.sort(key=lambda x: -x[0])
for t, l in lines[:TOP]:
    print(l)
