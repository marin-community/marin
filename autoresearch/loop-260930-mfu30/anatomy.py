"""Scope-level step anatomy from rows.pkl (tfop_dump.py) + opnames.pkl (hlo_opnames.py).

Compute-stream kernel time is attributed to (scope, phase). Collective-stream kernels are
attributed to their scope too, with exposed time = interval minus union(compute intervals).
"""
import sys, pickle, collections, re, json

d = pickle.load(open(sys.argv[1], "rb")); ops = pickle.load(open(sys.argv[2], "rb"))
COMPUTE_STREAM = sys.argv[3] if len(sys.argv) > 3 else "Stream #17"
rows = [r for r in d["rows"] if r[6] == "jit_train_step"]
launches = d["launches"]
end = max(r[3] for r in rows)
steps = list(zip(launches, launches[1:] + [end]))
NSTEP = len(steps)

def phase(op):
    if "rematted_computation" in op: return "remat"
    if "transpose(" in op: return "bwd"
    if "jvp(" in op: return "fwd"
    return "other"

RULES = [
    ("moe_expert_gemm", r"MoEExpertMlp/moe_mlp/shard_map/moe_chunk_\d+/jit\(call_wrapper\)"),
    ("moe_expert_elementwise", r"MoEExpertMlp/moe_mlp/shard_map/moe_chunk_"),
    ("moe_dispatch", r"moe_mlp/shard_map/dispatch"),
    ("moe_combine", r"moe_mlp/shard_map/combine"),
    ("moe_ep_other", r"MoEExpertMlp/moe_mlp"),
    ("moe_latent_proj", r"MoEMLP/(td,dl->tl|tl,ld->td)"),
    ("router", r"MoEMLP/(top_k|.*router|.*td,de|.*softmax|.*sigmoid)|MoEMLP/[^/]*$"),
    ("attn_kernel", r"CausalSelfAttention/shard_map/jit\(call_wrapper\)"),
    ("attn_proj", r"CausalSelfAttention/.*dot_general"),
    ("attn_other", r"CausalSelfAttention/"),
    ("shared_mlp_gemm", r"DenseMLP/.*dot_general"),
    ("shared_mlp_other", r"DenseMLP/"),
    ("short_conv", r"short_conv|ShortConv"),
    ("norms", r"RMSNorm|GatedNorm"),
    ("loss_lmhead", r"fused_linear_softmax_cross_entropy|lm_head|logsumexp"),
    ("embed", r"[Ee]mbed"),
    ("remat_carry", r"/remat2|while/body/eval_jaxpr/checkpoint/[^B]"),
    ("optimizer", r"^jit\(train_step\)/(shard_map|.*muon|.*newton|.*optax|.*adam|.*update|.*apply)"),
]
RULES = [(n, re.compile(p)) for n, p in RULES]

def scope(op):
    if not op: return "unmapped"
    for n, p in RULES:
        if p.search(op): return n
    return "other:" + "/".join(op.split("/")[1:4])[:70]

MEMCPY = re.compile(r"memcpy|memset", re.I)
def is_coll(name):
    return bool(re.search(r"nccl|RaggedAllToAll", name))

def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return out
def length(iv): return sum(e - s for s, e in iv)
def intersect(a, b):
    i = j = 0; out = []
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e: out.append([s, e])
        if a[i][1] < b[j][1]: i += 1
        else: j += 1
    return out

comp_iv = []; coll = []; mem = []
tab = collections.defaultdict(lambda: collections.Counter())
cnt = collections.Counter()
other_streams = collections.Counter()
for r in rows:
    lname, name, s, e = r[0], r[1], r[2], r[3]
    rec = ops.get(r[5]); op = rec[1] if rec else None
    if MEMCPY.search(name):
        mem.append((s, e, lname)); continue
    if is_coll(name):
        coll.append((s, e, name, op, lname)); continue
    if not lname.startswith(COMPUTE_STREAM):
        other_streams[(lname, name[:60])] += e - s
    comp_iv.append((s, e))
    sc = scope(op); ph = phase(op or "")
    tab[sc][ph] += e - s; cnt[sc] += 1
compm = merge(comp_iv)
span = sum(hi - lo for lo, hi in steps)
PS = 1e-12
print(f"steps={NSTEP} mean span={span*PS/NSTEP:.3f}s compute busy={length(compm)*PS/NSTEP:.3f}s/step")
print("non-main-stream compute:", [(k, round(v*PS/NSTEP, 4)) for k, v in other_streams.most_common(5)])
print(f"\n{'scope':28s} {'total':>7s} {'fwd':>7s} {'remat':>7s} {'bwd':>7s} {'other':>7s}  n/step")
for sc, c in sorted(tab.items(), key=lambda kv: -sum(kv[1].values())):
    t = sum(c.values())
    print(f"{sc:28s} {t*PS/NSTEP:7.3f} {c['fwd']*PS/NSTEP:7.3f} {c['remat']*PS/NSTEP:7.3f} {c['bwd']*PS/NSTEP:7.3f} {c['other']*PS/NSTEP:7.3f}  {cnt[sc]//NSTEP}")
# collectives
ctab = collections.defaultdict(lambda: [[], 0])
for s, e, name, op, lname in coll:
    fam = "ragged_a2a" if "RaggedAllToAll" in name else re.sub(r"\(.*", "", name).replace("ncclSymkDevKernel_", "symk_").replace("ncclDevKernel_", "")
    key = (fam, scope(op), phase(op or ""))
    ctab[key][0].append((s, e))
print(f"\n{'collective':38s} {'scope':22s} {'ph':5s} {'busy':>7s} {'exposed':>7s}")
allc = []
rowsout = []
for key, (iv, _) in ctab.items():
    m = merge(iv); allc += iv
    ex = length(m) - length(intersect(m, compm))
    rowsout.append((ex, key, length(m)))
for ex, key, b in sorted(rowsout, reverse=True)[:30]:
    print(f"{key[0][:38]:38s} {key[1][:22]:22s} {key[2]:5s} {b*PS/NSTEP:7.3f} {ex*PS/NSTEP:7.3f}")
cm = merge(allc)
print(f"collective busy={length(cm)*PS/NSTEP:.3f} exposed={(length(cm)-length(intersect(cm, compm)))*PS/NSTEP:.3f}")
memm = merge([(s, e) for s, e, _ in mem])
busy_all = merge(compm + cm + memm)
print(f"memcpy busy={length(memm)*PS/NSTEP:.3f} exposed(not under compute/coll)={(length(memm)-length(intersect(memm, merge(compm+cm))))*PS/NSTEP:.3f}")
print(f"device idle={(span-length(busy_all))*PS/NSTEP:.3f}")
