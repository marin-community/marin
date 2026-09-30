import sys, pickle, collections, re
sys.path.insert(0, sys.argv[3])
from anatomy_lib import scope, phase, merge, intersect, length
d = pickle.load(open(sys.argv[1], "rb")); ops = pickle.load(open(sys.argv[2], "rb"))
rows = [r for r in d["rows"] if r[6] == "jit_train_step"]
L = d["launches"]
comp = merge([(r[2], r[3]) for r in rows if r[0].startswith("Stream #17") and not re.search(r"nccl|Ragged|memcpy|memset", r[1], re.I)])
coll = merge([(r[2], r[3]) for r in rows if re.search(r"nccl|RaggedAllToAll", r[1])])
cc = merge(comp + coll)
# u32 allreduce detail
u = collections.Counter(); un = collections.Counter()
for r in rows:
    if "AllReduce_u32" in r[1]:
        rec = ops.get(r[5]); u[(r[5], (rec[1] or "")[-90:] if rec else "")] += r[3] - r[2]; un[r[5]] += 1
print("u32 all-reduce instrs:")
for k, v in u.most_common(6): print(f"  {v/1e12/3:.4f}s/step n={un[k[0]]//3} {k}")
# exposed memcpy by stream + name + time position
mc = collections.Counter(); pos = collections.Counter()
for r in rows:
    if re.search(r"memcpy|memset", r[1], re.I):
        ex = (r[3] - r[2]) - length(intersect([[r[2], r[3]]], cc))
        if ex > 0:
            rec = ops.get(r[5])
            mc[(r[0][:12], r[1][:12], (rec[1] or r[5])[-80:] if rec else r[5][:40])] += ex
            # which step fraction
            for i, l in enumerate(L):
                nxt = L[i + 1] if i + 1 < len(L) else 1e30
                if l <= r[2] < nxt:
                    pos[int(10 * (r[2] - l) / ((L[1] - L[0])))] += ex
print("exposed memcpy by (stream,name,op):")
for k, v in mc.most_common(12): print(f"  {v/1e12/3:.4f} {k}")
print("exposed memcpy by step decile:", {k: round(v/1e12/3, 3) for k, v in sorted(pos.items())})
# idle gaps distribution by step decile
allb = merge([tuple(iv) for iv in cc] + [(r[2], r[3]) for r in rows if re.search(r"memcpy|memset", r[1], re.I)])
gp = collections.Counter()
for (a, b), (c, _) in zip(allb, allb[1:]):
    g = c - b
    if g > 0:
        for i, l in enumerate(L):
            nxt = L[i + 1] if i + 1 < len(L) else 1e30
            if l <= b < nxt: gp[int(10 * (b - l) / (L[1] - L[0]))] += g
print("idle by step decile:", {k: round(v/1e12/3, 3) for k, v in sorted(gp.items())})
