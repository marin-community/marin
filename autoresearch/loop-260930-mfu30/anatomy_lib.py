import re
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

