"""Convert gemm_configs.py output into canonical (M, N, K, layouts) benchmark configs.

Usage: to_bench_configs.py <gemm_configs.json> <out.json> [min_t_step]
"""
import json
import sys

src = json.load(open(sys.argv[1]))
min_t = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
out = []
for x in src:
    if x["t_step"] < min_t:
        continue
    (_, ld, ll), (_, rd, rl), (_, od, ol) = x["lhs"], x["rhs"], x["out"]
    lb, rb, lc, rc = x["lb"], x["rb"], x["lc"], x["rc"]
    if len(lc) != 1 or len(ld) - len(lb) != 2 or len(rd) - len(rb) != 2:
        print("skip", ld, rd)
        continue
    batch = 1
    for i in lb:
        batch *= ld[i]
    lfree = [i for i in range(len(ld)) if i not in lb and i not in lc][0]
    rfree = [i for i in range(len(rd)) if i not in rb and i not in rc][0]
    m, k, n = ld[lfree], ld[lc[0]], rd[rfree]
    # A layout: physically minor dim is ll[0].
    a = "MK" if ll[0] == lc[0] else "KM"
    b = "NK" if rl[0] == rc[0] else "KN"
    c = "MN" if ol[0] == len(od) - 1 else "NM"
    if lb and not (lb == [0] and rb == [0] and ol[-1] == 0 and ll[-1] == 0 and rl[-1] == 0):
        print("skip nonleading batch", ld, rd, od)
        continue
    out.append(
        dict(
            name=f"b{batch}_m{m}_n{n}_k{k}_{a}_{b}_{c}",
            batch=batch, m=m, n=n, k=k, a=a, b=b, c=c,
            insitu_t_step=round(x["t_step"], 4), insitu_n_step=x["n_step"], insitu_pfs=round(x["pfs"], 3),
            insitu_pfs_nonovl=round(x["pfs_nonovl"], 3) if x["pfs_nonovl"] else None,
            insitu_kernel=x["kernels"][0][0], insitu_scope=x["scopes"][0][0],
        )
    )
json.dump(out, open(sys.argv[2], "w"), indent=1)
for c in out:
    print(c["name"], c["insitu_t_step"], c["insitu_pfs"], c["insitu_kernel"][:45], c["insitu_scope"])
print(len(out), "configs, in-situ time/step", round(sum(c["insitu_t_step"] for c in out), 3))
