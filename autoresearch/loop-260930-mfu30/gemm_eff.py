import sys, pickle, collections, re
sys.path.insert(0, sys.argv[4])
from anatomy_lib import scope, phase, merge, intersect, length
d = pickle.load(open(sys.argv[1], "rb")); ops = pickle.load(open(sys.argv[2], "rb")); gf = pickle.load(open(sys.argv[3], "rb"))
rows = [r for r in d["rows"] if r[6] == "jit_train_step"]
coll = merge([(r[2], r[3]) for r in rows if re.search(r"nccl|RaggedAllToAll", r[1])])
ragged = merge([(r[2], r[3]) for r in rows if "RaggedAllToAll" in r[1]])
agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0])  # flops, time, time_overlapped_ragged, flops_nonovl, time_nonovl
shapes = collections.defaultdict(collections.Counter)
kern = collections.defaultdict(collections.Counter)
for r in rows:
    if r[5] not in gf: continue
    fl, lhs, outd = gf[r[5]]
    rec = ops.get(r[5]); op = rec[1] if rec else ""
    key = (scope(op), phase(op))
    s, e = r[2], r[3]
    ov = length(intersect([[s, e]], ragged))
    a = agg[key]; a[0] += fl; a[1] += e - s; a[2] += ov
    if ov < 0.05 * (e - s):
        a[3] += fl; a[4] += e - s
    shapes[key][(tuple(lhs), tuple(outd))] += e - s
    kern[key][r[1][:50]] += e - s
PS = 1e-12
print(f"{'scope':22s} {'ph':5s} {'t/step':>7s} {'PF/s':>6s} {'%peak':>6s} {'%ovl':>5s} {'PF/s nonovl':>11s}")
for key, a in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    if a[1] * PS / 3 < 0.01: continue
    pf = a[0] / (a[1] * PS) / 1e15
    pfn = a[3] / (a[4] * PS) / 1e15 if a[4] else float("nan")
    print(f"{key[0]:22s} {key[1]:5s} {a[1]*PS/3:7.3f} {pf:6.2f} {100*pf/2.5:6.1f} {100*a[2]/a[1]:5.1f} {pfn:11.2f}")
    for (sh, t) in shapes[key].most_common(3):
        print(f"      {t*PS/3:6.3f}s lhs={list(sh[0])} out={list(sh[1])}")
    for (k, t) in kern[key].most_common(2):
        print(f"      {t*PS/3:6.3f}s {k}")
