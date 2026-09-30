"""Summarize block_bench.py RESULT lines by kernel category. Usage: block_bench_summary.py <log> [result index]"""
import sys, re, collections, json
txt = open(sys.argv[1]).read()
# first RESULT line holds the default-flag profiled run with all variants
res = [json.loads(l.split("RESULT ", 1)[1]) for l in txt.splitlines() if l.startswith("RESULT ")]
r = res[int(sys.argv[2]) if len(sys.argv) > 2 else 0]
def cat(k):
    if "nvjet" in k or "cublas" in k: return "gemm"
    if k.startswith("ffi_call"): return "fa4(ffi)"
    if k.startswith("pallas"): return "pallas(sconv)"
    m = re.match(r"([a-z_]+?)(_fusion)?[.\s]", k)
    return (m.group(1) if m else k.split()[0])
tab = {}
for v, d in r.items():
    c = collections.Counter()
    for k, (ms, n) in d.get("kernels", {}).items():
        c[cat(k)] += ms
    tab[v] = c
keys = sorted(set(k for c in tab.values() for k in c), key=lambda k: -tab["baseline"].get(k, 0))
vs = list(tab)
print(f"{'category':32s}" + "".join(f"{v:>11s}" for v in vs))
for k in keys:
    if max(tab[v].get(k, 0) for v in vs) < 0.05: continue
    print(f"{k:32s}" + "".join(f"{tab[v].get(k,0):11.3f}" for v in vs))
print(f"{'TOTAL kernels':32s}" + "".join(f"{sum(tab[v].values()):11.3f}" for v in vs))
print(f"{'step (wall)':32s}" + "".join(f"{r[v]['step_ms']:11.3f}" for v in vs))
