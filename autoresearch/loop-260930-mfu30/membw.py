import sys, pickle, collections, re
sys.path.insert(0, sys.argv[4])
from anatomy_lib import scope, phase
d = pickle.load(open(sys.argv[1], "rb")); ops = pickle.load(open(sys.argv[2], "rb")); fb = pickle.load(open(sys.argv[3], "rb"))
rows = [r for r in d["rows"] if r[6] == "jit_train_step" and r[0].startswith("Stream #17")]
agg = collections.defaultdict(lambda: [0, 0, 0])
for r in rows:
    if r[5] not in fb: continue
    rec = ops.get(r[5]); op = rec[1] if rec else ""
    sc = scope(op)
    agg[sc][0] += fb[r[5]]; agg[sc][1] += r[3] - r[2]; agg[sc][2] += 1
BW = 7.0e12
print(f"{'scope':28s} {'t/step':>7s} {'GB/step':>8s} {'TB/s':>6s} {'SOL@7TB/s':>9s}")
for sc, (b, t, n) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    if t / 1e12 / 3 < 0.02: continue
    print(f"{sc:28s} {t/1e12/3:7.3f} {b/3/1e9:8.1f} {b/(t/1e12)/1e12:6.2f} {b/3/BW:9.3f}")
