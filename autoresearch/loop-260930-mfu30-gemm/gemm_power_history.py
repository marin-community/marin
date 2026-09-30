"""Correlate each GEMM's in-situ rate with how GEMM-dense the preceding window was (power-cap signature)."""
import pickle, sys, bisect, numpy as np
P = sys.argv[1]
d = pickle.load(open(f"{P}/rows.pkl", "rb"))
gf = pickle.load(open(f"{P}/gemmflops.pkl", "rb"))
rows = sorted([r for r in d["rows"] if r[6] == "jit_train_step"], key=lambda r: r[2])
comp = [r for r in rows if r[0].startswith("Stream #17")]
ragged = sorted((r[2], r[3]) for r in rows if "RaggedAllToAll" in r[1] or "nccl" in r[1].lower())
rs = [a for a, b in ragged]
def overlap_any(s, e):
    i = max(bisect.bisect_left(rs, s) - 1, 0)
    for a, b in ragged[i:]:
        if a >= e: return False
        if b > s: return True
    return False
# tensor-heavy kernels: cuBLAS GEMMs, QuACK (cutlass/quack) and FA4 kernels
def heavy(r):
    n = r[1]
    return r[5] in gf or "nvjet" in n or "quack" in n.lower() or "Gemm" in n or "flash" in n.lower() or "fmha" in n.lower()
hs = [(r[2], r[3]) for r in comp if heavy(r)]
hstart = [a for a, b in hs]
def heavy_frac(t, w):
    lo = t - w
    i = max(bisect.bisect_left(hstart, lo) - 1, 0)
    tot = 0
    for a, b in hs[i:]:
        if a >= t: break
        tot += max(0, min(b, t) - max(a, lo))
    return tot / w
W = [2e9, 10e9, 50e9]  # ps windows: 2 ms, 10 ms, 50 ms
recs = []
for r in comp:
    if r[5] not in gf: continue
    fl, lhs, out = gf[r[5]]
    if fl < 1e12: continue
    if overlap_any(r[2], r[3]): continue
    rate = fl / ((r[3] - r[2]) * 1e-12) / 1e15
    recs.append((rate, *[heavy_frac(r[2], w) for w in W]))
a = np.array(recs)
print("n", len(a))
for j, w in enumerate(W):
    x = a[:, j + 1]
    c = np.corrcoef(x, a[:, 0])[0, 1]
    qs = np.quantile(x, [0, .25, .5, .75, 1])
    bins = np.digitize(x, qs[1:-1])
    means = [a[bins == b, 0].mean() for b in range(4)]
    print(f"window {w/1e9:4.0f} ms: corr(rate, preceding tensor-kernel busy frac) = {c:+.2f}; rate by quartile of busy frac: " + " ".join(f"{m:.3f}" for m in means) + f"  (busy frac quartile edges {np.round(qs,2)})")
# Within-shape control: demean the rate by GEMM shape+kernel before correlating.
import collections
recs2 = []
for r in comp:
    if r[5] not in gf: continue
    fl, lhs, out = gf[r[5]]
    if fl < 1e12 or overlap_any(r[2], r[3]): continue
    rate = fl / ((r[3] - r[2]) * 1e-12) / 1e15
    recs2.append(((tuple(lhs), tuple(out), r[1][:60]), rate, heavy_frac(r[2], 50e9)))
g = collections.defaultdict(list)
for k, rate, h in recs2: g[k].append((rate, h))
xs, ys = [], []
for k, v in g.items():
    if len(v) < 20: continue
    v = np.array(v); xs += list(v[:, 1] - v[:, 1].mean()); ys += list(v[:, 0] - v[:, 0].mean())
print(f"within-shape corr (50 ms window): {np.corrcoef(xs, ys)[0,1]:+.2f}  n={len(xs)}; slope {np.polyfit(xs, ys, 1)[0]:.3f} PF/s per unit busy frac")
