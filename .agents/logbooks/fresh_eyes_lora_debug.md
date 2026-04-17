# Fresh-Eyes Logbook: DPO + LoRA Divergence on v5p-8

**Status:** Open. Owner: next agent picking this up. Created: 2026-04-16.

This logbook is the entry point for a new agent investigating the v5p-8 LoRA-DPO bug. It consolidates the prior handoff, adds critical context that wasn't in it (mesh device count, megacore, L2-vs-sum interpretation), explains the Levanter sharding model end-to-end so you don't have to learn it from scratch, and gives concrete probes with copy-pasteable code.

> **Read this top-to-bottom once before launching anything.** Most of the prior agents' time was spent re-deriving facts that are now in here.

---

## 1. TL;DR for fresh agents (60 seconds)

DPO + LoRA training **diverges on v5p-8** but works on v5p-16, v6e-8, v6e-16. Full fine-tuning works everywhere. After 8 experiments the CE kernel is essentially exonerated (Exp 1b upcast didn't fix; Exp 8 matching `num_b_blocks=16` didn't fix). The active hypothesis is that LoRA-specific **gradient sharding/reduction** breaks under the v5p-8 mesh, and the **single most diagnostic fact** is:

> **At step 0, per-module gradient L2 norms match healthy runs to 0.01–0.05%, but gradient sums diverge by 15–40% on LoRA-adapted modules — `gate_proj` worst at +31.1%.**

L2 is sign-invariant, sum is not. **Magnitudes match, signs/composition don't.** That fingerprint is reduction-axis / partitioning, not numerics.

**Critical context the prior handoff did not surface:** Levanter sees v5p-8 as a **4-device** mesh (because each v5p chip has 2 TensorCores in megacore mode, but JAX reports chips not cores), while every working config has **≥ 8 devices**. So "broken iff v5p-8 + LoRA" is operationally "broken iff `data_axis_size == 4` + LoRA". This is the angle the previous investigation missed. See §6.

**Don't:**
- Don't relitigate the CE kernel. It's been ruled out three different ways. (Confirm once with the v_block ≥ V test in §11.A, then move on.)
- Don't re-run `pd` sweeps — Exp 7 already covered pd ∈ {4, 8} on v5p-8.
- Don't add more bf16-vs-fp32 instrumentation. Exp 1b settled it.

**Do, in order:**
1. Dump sharding specs for LoRA params and grads on v5p-8 vs v6e-8. (§11.B — 30 min, dispositive.)
2. Diff `gate_proj.lora_*` vs `up_proj.lora_*` sharding. They're SwiGLU twins; if specs differ that's the bug.
3. Toy repro on a single LoRA-wrapped Linear without DPO/CE, on v5p-8 vs v6e-8.

---

## 2. Glossary (read this if any term below is unfamiliar)

| Term | Meaning |
|---|---|
| **v5p-8** | TPU v5p, 8 cores. v5p chips have **2 cores per chip ("megacore")**, so v5p-8 = 4 chips = 4 JAX devices. Single host. |
| **v5p-16** | v5p, 16 cores = 8 chips = 8 JAX devices. Single host. |
| **v6e-8** | TPU v6e, 8 cores. v6e has **1 core per chip**, so v6e-8 = 8 chips = 8 JAX devices. Single host. |
| **v6e-16** | v6e, 16 cores = 16 chips = 16 JAX devices. Two hosts (8 chips each). |
| **megacore** | v5p design where 2 cores share a chip and HBM. JAX exposes the chip as a single device. |
| **pd / per_device_parallelism** | Levanter knob: microbatch size per device. `train_batch_size = pd × data_axis_size × accum_steps`. |
| **bs / train_batch_size** | Total batch size per training step. |
| **num_b_blocks** | Number of B-tiles inside the streaming CE kernel: `B_local / b_block_size`. Where `B_local = pd` per device. |
| **LoRA r** | LoRA rank. r=64 in this investigation. |
| **`lora_A` / `lora_B`** | Two factor matrices: `lora_A: (LORA_R, In)`, `lora_B: (Out, LORA_R)`. `lora_B` is **zero-initialized** so adapter starts as identity. |
| **`gate_proj`** | SwiGLU gate projection in Llama MLP: `silu(gate(x)) * up(x) → down`. Twin of `up_proj`. |
| **DP / data axis** | Data-parallel mesh axis. In Levanter usually called `data` (within-host) and `replica_dcn` (cross-host). |
| **FSDP** | Sharding model parameters across the data axis (ZeRO-3 style). Levanter does this by mapping `embed` → `data` in the parameter mapping. |
| **TP / model axis** | Tensor-parallel mesh axis, called `model` in Levanter. Default size 1 for these runs. |
| **L2-vs-sum diagnostic** | If two gradient tensors have the same L2 norm but different element-wise sum, the magnitudes match but signs/composition differ. Strong evidence for a reduction-axis bug, not a numerics bug. |
| **CE kernel** | `linear_softmax_cross_entropy_loss_xla` in `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/xla.py`. The fused vocab-tiled softmax CE. |

---

## 3. Problem statement

Identical config trains cleanly on v5p-16, v6e-8, v6e-16. On v5p-8:

- `train/loss` plateaus at ~0.65–0.69 (~`ln(2)`) instead of dropping to ~0.3
- `train/dpo_accuracy` still climbs to 1.0 (margin is nominally there but tiny)
- `grad/norm/total` stays ~27 instead of dropping to ~17 like healthy runs
- LM-as-judge eval against healthy LoRA: v5p-8 outputs are qualitatively worse

**Crucially, the bug is regime × chip, not chip alone.** Full-FT works on v5p-8-equivalent geometries. So the "v5p-8 is just broken" hypothesis is dead.

---

## 4. The single most diagnostic fact (read carefully)

At step 0, per-module gradient diagnostics on v5p-8 LoRA vs v6e-8 LoRA:

| Module | L2 ratio (broken/healthy) | Sum ratio (broken/healthy) |
|---|---|---|
| `gate_proj.lora_B` | ~1.0001 | **+31.1%** |
| Other LoRA modules | within 0.01–0.05% | 15–40% diverge |
| Non-LoRA params | match | match |

### What this means mathematically

For a tensor `g` with elements `g_i`:

- L2 = `sqrt(Σ g_i²)` — invariant under sign flips of any element, and under permutations.
- Sum = `Σ g_i` — depends on signs.

If broken vs healthy match in L2 but diverge in sum, then **the elements are not the same tensor**, but they have nearly identical *magnitude distributions*. The difference is in **sign / arrangement**. This is the fingerprint of:

- **Different reduction axis or different partition**: the same partial contributions are being combined differently across shards.
- **NOT** an accumulation-order numerics drift: those perturb magnitudes by ULPs, never produce 31% sum divergence.
- **NOT** a missing `psum` magnitude bug: missing psum scales by `1/N`, would also blow L2.

This is why CE-kernel hypotheses are all on borrowed time — the kernel can produce ULP drift, not 31% sign-pattern divergence.

### Why `gate_proj` is the loudest

`gate_proj` and `up_proj` are SwiGLU twins: same shape, same activation source (`x` from prev block), feeding `silu(gate(x)) * up(x)`. They're gradient-coupled. If only `gate_proj` is loudest, that's evidence the bug is **per-parameter sharding-spec dependent**, not algorithm-wide. **Diff `gate_proj.lora_*` sharding against `up_proj.lora_*`** — that's the cheapest pointed test you can do.

(`lora_A` is zero-init — wait, **`lora_B` is the one zero-init in this codebase** (see `lora.py:221-225`). At step 0 only `lora_A` would carry information through the forward and only `lora_B` would receive a gradient via the chain rule? Actually it's the reverse — `lora_B`-zero means forward `z = lora_B(lora_A(x)) = 0`, so the adapter contributes nothing to logits at step 0. But `lora_B` itself gets a nonzero gradient (it's `lora_A(x).T @ dL/dy`), and `lora_A` gets a *zero* gradient because the chain goes through `lora_B.weight = 0`. Verify this matches the observed step-0 grads: `lora_A` should be exactly zero on every chip, `lora_B` should be nonzero. If broken/healthy disagree on `lora_A` being zero, that's also a bug.)

---

## 5. Experiment matrix (8 experiments, prior work)

| # | Config | Chip | Regime | num_b_blocks | Result |
|---|---|---|---|---|---|
| 1 | baseline (orig) | v5p-8 | LoRA r=64 | — | ❌ broken |
| 1 | baseline (orig) | v6e-8 | LoRA r=64 | — | ✅ works |
| 1b | + upcast CE kernel block logic to fp32 | v5p-8 | LoRA | — | ❌ still broken |
| 2 | 8B full-FT sanity check | v5p-16 / v6e-32 | full-FT | — | ✅ match |
| 3 | fixed LoRA rank mismatch (64→16→64) | v5p-8 | LoRA | — | ❌ still broken |
| 4 | matched pd=4, bs=64 geometry | v5p-16 vs v6e-16 | full-FT | 1 vs 16 | ✅ match, grad Δ ~0.1% |
| 5 | different topology, matched pd=2 | v5p-16 vs v6e-16 | LoRA | 1 vs 16 | ✅ match, but grad sums differ 17–39% |
| 6 | bumped pd | v5p-16, pd=4 | LoRA | — | ✅ works |
| 7 | back to v5p-8, varied pd | v5p-8, pd=4 and pd=8 | LoRA | 64 | ❌ still broken |
| 8 | **forced num_b_blocks=16 to match v6e-16** | v5p-8, bs=32, pd=4 | LoRA | 16 | ❌ **still broken** |

Held constant from Exp 3 onward: dataloader returns identical indices in identical order; initial parameter checksums match; seed matched; small dataset.

### Re-reading Exp 5

This row matters and is undersold in the original report: **even on the working v5p-16 vs v6e-16 LoRA comparison**, gradient sums differed 17–39%. That means the L2-vs-sum divergence is a **chronic property of LoRA+Levanter across mesh layouts** — it's just *catastrophic* on the 4-device mesh, *tolerable* on 8/16-device meshes, and *invisible* on full-FT. That is a strong hint the reduction is mis-scoped on a per-axis basis and the magnitude of damage scales with mesh-axis cardinality.

---

## 6. Mesh device count: the angle the prior investigation missed

Levanter only knows about JAX devices. It calls `jax.device_count()` and partitions accordingly (`trainer.py:960`, `mesh.py`). It has no concept of "v5p" vs "v6e".

Consult `.agents/logbooks/levanter_mesh_explained.md` (already in this repo). The relevant table:

| Config | `jax.device_count()` | `data_axis_size` | Working? |
|---|---|---|---|
| **v5p-8** | **4** | **4** | ❌ broken |
| v5p-16 | 8 | 8 | ✅ |
| v6e-8 | 8 | 8 | ✅ |
| v6e-16 | 16 | 16 | ✅ |

**The invariant is `data_axis_size == 4`, not the chip family.** The chip-family framing is a red herring caused by the fact that v5p-8 happens to be the only configuration in the matrix that produces a 4-device mesh.

### Falsifiable prediction from this hypothesis

If the bug is "4-device mesh + LoRA", then:

1. **v5p-4** (a single chip, 1 megacore pair = 2 devices) should *also* be broken or even worse. Untested.
2. **v6e-4** (4 chips = 4 devices) should also be broken. **This is the cleanest disambiguation experiment available.** If v6e-4 LoRA is broken with the same fingerprint, the diagnosis is settled: it's the mesh size, not the chip generation. If v6e-4 works, the chip-family hypothesis is alive.
3. **v5p-32** (16 chips = 16 devices) should work fine. Mostly untested but priors say yes.

### Why a 4-device data axis could be uniquely bad

Several plausible mechanisms; all worth considering when reading code:

- **FSDP all-gather granularity**. With `data=4`, the `embed` axis (e.g. 4096) gets all-gathered across 4 shards; collective layout is 2x2 torus in v5p-8. Some XLA collective lowerings switch strategy below a threshold (typically 8). Different layouts differ at the collective level, not just kernel level.
- **MXU padding pressure**. LoRA `lora_A` has shape `(LORA_R=64, In=embed=4096)`. If `In` is sharded by `data=4`, each shard is `(64, 1024)`. 1024 is `8 × 128` — clean MXU tiles. With `data=8`, each shard is `(64, 512)` — also clean. Probably not the issue.
- **Reduce-scatter for grad of sharded weight**. After backward, `grad(lora_A)` needs to be reduce-scattered back to the FSDP layout. With 4 shards, the reduce-scatter tree differs from 8 shards. If for some reason the reduce-scatter is happening on the wrong mesh axis (e.g. on `model` instead of `data`, or partially), the resulting per-shard grad would have wrong sign composition while having the right total magnitude — *exactly* the L2-vs-sum signature.
- **Implicit broadcast of `LORA_R`**. `LORA_R` is a new axis name (`lora.py:217`) that is **not in any standard `parameter_axis_mapping`**, so it gets replicated. If Levanter or Haliax accidentally maps it to a mesh axis on certain mesh shapes (e.g. assigns it to a leftover axis when `data` is too small), the LoRA factors would shard along their tiny dimension — broken.
- **`hax.dot` over a sharded contraction axis with one operand replicated**. Forward `lora_A(x)` contracts `In=embed`. If `lora_A.weight`'s `In` is FSDP-sharded but the activation `x`'s `In` axis has a different sharding, an implicit re-shard or all-gather happens. Different mesh shapes can land this re-shard on different axes.

---

## 7. What's been ruled out (do not re-litigate)

| Hypothesis | Evidence | Confidence |
|---|---|---|
| Dataloader / data order | indices match across chips | high |
| Initial parameters | checksums match | high |
| LoRA rank mismatch | Exp 3 matched config exactly | high |
| CE kernel fp32 upcast | Exp 1b — no effect | high |
| `pd` alone | v5p-16 works at pd=2 and pd=4; v5p-8 fails at pd=4 and pd=8 | high |
| Microbatch size alone | v5p-8 pd=4 and v5p-16 pd=2 have same microbatch; v5p-8 still broken | high |
| `num_b_blocks` in CE kernel | Exp 8 — matched to working config, still broken | high |
| Full-FT code path | works on all chips at all geometries | high |

**The remaining live region of hypothesis space is exactly: LoRA-specific param/grad partitioning under a 4-device data axis.**

---

## 8. Active hypotheses, ranked

### H1 (~50–60%): LoRA gradient reduction is mis-scoped on small `data` mesh

Mechanism: `grad(lora_A)` or `grad(lora_B)` ends up reduce-scattered/all-reduced across the wrong subset of mesh axes when `data_axis_size = 4`. Magnitudes survive (everything still gets summed), signs don't (the partials are combined in a different grouping).

- Strongest evidence: L2-vs-sum signature (§4).
- Why it's worst on `gate_proj`: per-module sharding spec is per-parameter, and the SwiGLU triple may have a specific spec quirk.
- Why it scales with mesh size: same reduction bug at `data=8`/`data=16` only swaps ~17–39% of signs (Exp 5), at `data=4` it swaps enough to destroy training.

**Fastest test:** §11.B (sharding-spec dump).

### H2 (~15–20%): DPO reference-model gradient leakage into adapter params

Mechanism: DPO does forward through *both* policy and reference. If the reference is implemented as "policy with adapters disabled" (likely), gradient flow from reference logprobs to adapter params should be cut by `stop_gradient`. If `stop_gradient` placement is mesh-shape-dependent (it shouldn't be, but…), broken on v5p-8.

**Fastest test:** Replace DPO loss with plain NLL on the same adapter, run on v5p-8 LoRA. If it trains cleanly, DPO interaction is the cause.

### H3 (~10–15%): Adapter-only optimizer state partitioning

Mechanism: Levanter's optimizer (Optax) has its own sharding rules. For LoRA, only adapter params are trainable, and the optimizer state (Adam m/v moments) is built only over the adapter subtree. If that subtree's sharding spec differs from what was assumed at construction, momentum and grad apply land on wrong axes.

**Fastest test:** Dump optimizer state shardings alongside grad shardings.

### H4 (~5–10%): XLA compiler pathology specific to v5p-8 + LoRA HLO

Hard to diagnose without compiler-level tooling. Defer.

### H5 (~5%): CE kernel after all

Kept on the list out of fairness, **but**: Exp 1b (fp32) and Exp 8 (matched `num_b_blocks`) eliminate the obvious mechanisms, the L2-vs-sum signature is wrong shape for kernel numerics, and full-FT (which uses the same kernel) works on the same chip. **Confirm with the §11.A test then close out.**

---

## 9. Levanter sharding model (read this — it's load-bearing)

Levanter sits on **Haliax named axes** + **JAX `pjit`/`shard_map`** + **two ResourceMappings**. If you don't internalize this, the sharding-spec dumps in §11.B will be unreadable.

### 9.1 Named axes

Every tensor has *named* axes, not positional. `Batch`, `Pos`, `Embed`, `Mlp`, `Heads`, `KVHeads`, `Vocab`, `LORA_R`. Construction: `hax.Axis("embed", 4096)`. Operations cite axes by name: `hax.dot(x, w, axis=Embed)` contracts along `Embed` regardless of position.

### 9.2 Mesh

A `jax.sharding.Mesh` with named physical axes. Levanter defaults (`utils/mesh.py`):

```
ICI (within-host):  data=-1, replica=1, model=1
DCN (cross-host):   replica_dcn=-1
```

`-1` means "absorb whatever's left." Net effect: on a single-host TPU, `data = jax.device_count()`, everything else = 1.

### 9.3 Two ResourceMappings

This is the part that surprises everyone. Levanter has **two** mappings (`trainer.py:346-351`):

- **`compute_axis_mapping`** — how *activation* logical axes map to mesh axes. Typical: `{"batch": ("replica_dcn", "replica", "data")}`.
- **`parameter_axis_mapping`** — how *weight* logical axes map to mesh axes. Typical: `{"embed": "fsdp"-or-"data", "mlp": "model", "vocab": "model", ...}`.

A logical axis **not in the mapping is replicated**.

### 9.4 How sharding gets applied

- `hax.shard(x, mapping)` / `hax.shard_with_axis_mapping(x, mapping)` lowers to `jax.lax.with_sharding_constraint`.
- `named_jit(...)` wraps `pjit` with a default mapping (typically the parameter mapping for inputs/outputs).
- Inside `named_jit`, `hax.dot` infers psums from the mapping: contracting a sharded axis ⇒ inserts an implicit psum over the corresponding mesh axes; contracting a replicated axis ⇒ no psum.

### 9.5 Forward / backward symmetry

- Forward: `y = w @ x`, contract `In` axis. If `In` is FSDP-sharded on weight, the matmul does an all-gather of `w` then a local matmul, or splits and reduces — XLA decides.
- Backward for `w`: `grad_w = x.T @ grad_y`, batch-contract `Batch`. If `Batch` is sharded on `data`, this contraction inserts a psum over `data` — that's how DP gradients get summed across devices.
- The transpose ops, the dot reorderings, the implicit reshards — all derived from the named-axis mapping. **There is no explicit `psum` written anywhere in the train step.** It's all done by the mapping.

### 9.6 Why this can break in subtle ways

If a tensor's actual sharding (post-pjit) doesn't match what you'd derive from the mapping, you can get:
- Silent re-sharding inserted by JAX (correct but slow)
- A `psum` over a wrong axis (correct in magnitude but wrong in element-wise composition — **L2-vs-sum signature**)
- Replication where you wanted sharding (correct, just memory-wasteful)
- Sharding where you wanted replication (broken — different shards see different inputs)

**The L2-vs-sum signature points squarely at "psum over a wrong axis or wrong subset."**

### 9.7 Where to find the relevant code

| Concern | File:line |
|---|---|
| Mesh construction | `lib/levanter/src/levanter/utils/mesh.py` (mesh defaults `DEFAULT_ICI_AXIS_SPEC`, `DEFAULT_DCN_AXIS_SPEC`); `trainer.py:960-1063` (slice/host counting, axis sizes) |
| `compute_axis_mapping` definition | `trainer.py:1089-1091` |
| `parameter_axis_mapping` definition | `trainer.py:1093-1095` |
| `named_jit` sites in train step | `trainer.py:654, 664-665, 673-674` |
| State re-sharding after step | `trainer.py:698, 801, 821-822` |
| Microbatch/grad accumulation | `lib/levanter/src/levanter/grad_accum.py` |
| LoRA module | `lib/levanter/src/levanter/lora.py` |

---

## 10. LoRA in Levanter (the parts that matter for this bug)

From `lib/levanter/src/levanter/lora.py`:

### 10.1 What `loraize` does

Tree surgery: walks the model, finds matching `hnn.Linear` layers, replaces each with `LoraLinear(wrapped=orig, lora=LowRankLinear(...))`. Default match = "all linear modules" (regex `None`).

### 10.2 LowRankLinear factor matrices

```python
# lora.py:217-227
_R = hax.Axis(LORA_R, r)              # r=64; LORA_R is the literal string "LORA_R"
lora_A = hnn.Linear.init(In, _R, ..., out_first=True)    # weight axes: (LORA_R, In)
if zero_init_b:
    zero_weight = hax.zeros((Out..., _R))
    lora_B = hnn.Linear(weight=zero_weight, ..., In=_R, Out=Out)   # weight: (Out, LORA_R)
else:
    lora_B = hnn.Linear.init(_R, Out, ..., out_first=True)
```

- **`lora_A.weight` shape: `(LORA_R, In)`** — `In` is e.g. `Embed`.
- **`lora_B.weight` shape: `(Out, LORA_R)`** — `Out` is e.g. `Mlp`.
- **`LORA_R` is a fresh axis name** that is not present in standard parameter mappings. So under the default mapping, `LORA_R` is replicated.

### 10.3 Forward

```python
# lora.py:191-204
def __call__(self, x, key=None):
    x = self.dropout(x, key=key)
    z = self.lora_A(x)           # contracts In; output has axes (Batch, ..., LORA_R)
    z = self.lora_B(z)           # contracts LORA_R; output has axes (Batch, ..., Out)
    return z * self.scale

# In LoraLinear:
def __call__(self, x, key=None):
    return self.lora(x) + self.wrapped(x)   # add adapter delta to base output
```

Two contractions: `In` (sharded if In is in mapping) then `LORA_R` (replicated, no psum).

### 10.4 Initialization (zero_init_b)

`lora_B` is zero-initialized so `lora(x) = 0` at step 0 → policy = reference. **At step 0, the gradient of the loss w.r.t. `lora_A.weight` is zero** (chain rule: it flows through `lora_B.weight = 0`). The gradient w.r.t. `lora_B.weight` is *not* zero.

**Consistency check for any sharding-spec dump at step 0:**
- `grad(lora_A)` should be (numerically) zero on every device.
- `grad(lora_B)` should match across chips up to the bug's fingerprint.

If `grad(lora_A)` is *not* zero on either chip, that's a separate bug worth chasing first.

### 10.5 Adapter / base partitioning

`partition_lora_params(model)` splits the model PyTree into `(base_model, adapter_model)` so the trainer can mark only the adapter as trainable. This means the optimizer's parameter axis mapping is applied to a *subset* of the model PyTree — and any quirk in how that subset's sharding gets resolved (e.g., a fallback / inferred sharding) is a candidate for the bug.

### 10.6 Where to dump state

If the trainer keeps `state.model` as the combined model and `state.opt_state` as Adam state for adapters only, then for sharding diagnostics you want:
- `model` (the live combined model after `loraize` + sharding)
- `state.opt_state` (Adam moments for adapters)
- The first `grads` PyTree returned from `loss_grad_fn` inside the train step

---

## 11. Concrete next steps (do these in order)

### 11.A (15 min) Confirm the CE kernel one last time, then close

Force the reference (non-streaming) path. Set `v_block_size ≥ V`:

```bash
# Vocab is ~128k for Llama-3.x. Set higher.
export MARIN_DEBUG_CE_V_BLOCK_SIZE=200000
export MARIN_DEBUG_CE_B_BLOCK_SIZE=$((200000))   # any value, both must be set per xla.py:464
```

(The `MARIN_DEBUG_CE_*` env vars are honored by `linear_softmax_cross_entropy_loss_xla` at `xla.py:457-482`.) Re-run the v5p-8 LoRA config. Expected: **still broken**. If it's still broken, the kernel is settled — close out hypothesis H5 and move to 11.B.

If it suddenly works, you've discovered a CE-kernel issue that survived Exp 1b and Exp 8. Stop and write up.

### 11.B (30 min, dispositive) Dump sharding specs for LoRA params and grads

Add this just after `loraize` and again after the first `loss_grad_fn` call inside the train step in `levanter/main/train_dpo.py`:

```python
def _dump_shardings(label, tree):
    import jax
    from jax.sharding import NamedSharding
    def _spec(x):
        if hasattr(x, "sharding"):
            s = x.sharding
            if isinstance(s, NamedSharding):
                return f"{tuple(x.shape)} {s.spec}"
            return f"{tuple(x.shape)} {type(s).__name__}"
        return None
    leaves = jax.tree_util.tree_leaves_with_path(jax.tree_util.tree_map(_spec, tree))
    for path, val in leaves:
        if val is None: continue
        # filter to LoRA-adapted modules
        path_str = jax.tree_util.keystr(path)
        if any(s in path_str for s in ("lora_A", "lora_B", "gate_proj", "up_proj", "down_proj", "q_proj")):
            print(f"DEBUGSHARD[{label}] {path_str} :: {val}")

_dump_shardings("model", model)
# ... after first grad call ...
_dump_shardings("grads", grads)
```

Run on v5p-8 and v6e-8. **Diff the outputs.** Concretely, look for:

1. **`LORA_R` showing up in any `PartitionSpec`** — should never happen. If it does on v5p-8 but not v6e-8, that's the bug.
2. **`gate_proj.lora_*` differing from `up_proj.lora_*`** — these are SwiGLU twins, must be identical. If they differ, that's the bug.
3. **`In`/`Embed` axis on `lora_A.weight` mapped to a different mesh axis** between chips — that means the mapping resolved differently on the 4-device mesh.
4. **Grad sharding ≠ weight sharding** for any LoRA factor — that means gradient apply is going to insert a re-shard, which is exactly where wrong-axis psums hide.

### 11.C (15 min) `gate_proj` vs `up_proj` sign-pattern compare

If 11.B shows identical specs for gate/up but their grads still differ in sum, dump the actual grad arrays (small enough to ship to host on a 4-device mesh):

```python
gA = jax.device_get(grads.transformer.layers.mlp.gate_proj.lora.lora_A.weight)
uA = jax.device_get(grads.transformer.layers.mlp.up_proj.lora.lora_A.weight)
print(f"gate_proj.lora_A: shape={gA.shape}, sum={gA.sum():.4f}, l2={np.linalg.norm(gA):.4f}, sign_balance={(gA > 0).mean():.3f}")
print(f"  up_proj.lora_A: shape={uA.shape}, sum={uA.sum():.4f}, l2={np.linalg.norm(uA):.4f}, sign_balance={(uA > 0).mean():.3f}")
# also dump first 16 elements
print(f"gate_proj.lora_A[0,:16] = {gA[0,:16]}")
print(f"  up_proj.lora_A[0,:16] = {uA[0,:16]}")
```

Run on v5p-8 and v6e-8. If on v6e-8 the elements are identical between chips and on v5p-8 they have flipped signs in a structured way (e.g. every 1024 elements flipped), you've localized the wrong-axis psum.

Note again: at step 0 with `zero_init_b`, `grad(lora_A)` may be exactly zero. In that case use `grad(lora_B)` instead.

### 11.D (30 min) Toy repro

Smallest possible thing that exhibits the gradient-sum divergence. Single-file script, on v5p-8 and v6e-8:

```python
import jax, jax.numpy as jnp
import haliax as hax
import haliax.nn as hnn
from haliax.partitioning import named_jit, ResourceMapping

In = hax.Axis("embed", 1024)
Out = hax.Axis("mlp", 4096)
R   = hax.Axis("LORA_R", 64)
B   = hax.Axis("batch", 32)

mesh = jax.make_mesh((jax.device_count(),), ("data",))
mapping: ResourceMapping = {"batch": "data", "embed": "data"}  # vary this!

with mesh, hax.axis_mapping(mapping):
    base_w = hax.random.normal(jax.random.PRNGKey(0), (Out, In))
    lora_A = hax.random.normal(jax.random.PRNGKey(1), (R, In))
    lora_B = hax.zeros((Out, R))   # zero-init, like LowRankLinear

    x = hax.random.normal(jax.random.PRNGKey(2), (B, In))
    target = hax.random.normal(jax.random.PRNGKey(3), (B, Out))

    @named_jit(in_axis_resources=mapping, out_axis_resources=mapping)
    def loss(lora_A, lora_B):
        z = hax.dot(x, lora_A, axis=In)             # (B, R)
        delta = hax.dot(z, lora_B, axis=R)          # (B, Out)
        y = hax.dot(x, base_w, axis=In) + delta     # (B, Out)
        return hax.sum((y - target) ** 2)

    grads = jax.grad(loss, argnums=(0, 1))(lora_A, lora_B)
    print(f"device_count={jax.device_count()}")
    print(f"grad(lora_A) sum={hax.sum(grads[0]).item():.6f}, l2={(hax.sum(grads[0]**2)**0.5).item():.6f}")
    print(f"grad(lora_B) sum={hax.sum(grads[1]).item():.6f}, l2={(hax.sum(grads[1]**2)**0.5).item():.6f}")
```

Run on v5p-8 (4 devices) and v6e-8 (8 devices). Same input data, same seeds. Compare. If sums diverge with matching L2, the bug is reproduced **without DPO, without CE kernel, without LoRA-as-tree-surgery**, in 50 lines. From there it's a tractable JAX/Haliax issue.

If toy doesn't repro, gradually add: (a) `loraize`-style tree surgery, (b) the `LoraLinear.__call__` add of base output and adapter, (c) microbatching via `grad_accum`, (d) DPO loss. Whichever step makes it repro is the culprit.

### 11.E (one hour) DPO ablation

Replace DPO loss with plain NLL on the same adapter and same data:

```python
# in your train_dpo entrypoint, gate the loss
loss = nll(model, batch) if os.environ.get("ABLATE_DPO") else dpo(model, batch)
```

If v5p-8 LoRA NLL trains cleanly, DPO interaction with sharding is in scope (H2 alive). If still broken, DPO is exonerated and H1/H3 are the only things left.

---

## 12. Files / symbols most likely involved

In rough priority order:

| Path | What to look for |
|---|---|
| `lib/levanter/src/levanter/lora.py` | `LowRankLinear.init` (axis names, `LORA_R` axis); `LoraLinear.__call__` (the `lora(x) + wrapped(x)` add); `partition_lora_params` |
| `lib/levanter/src/levanter/trainer.py` | `parameter_axis_mapping`, `compute_axis_mapping`, `_train_step`, `named_jit` boundaries (lines noted in §9.7) |
| `lib/levanter/src/levanter/utils/mesh.py` | `DEFAULT_ICI_AXIS_SPEC`, `DEFAULT_DCN_AXIS_SPEC`; how axes get sized |
| `lib/levanter/src/levanter/main/train_dpo.py` | DPO-specific train step; how reference model is set up; where `stop_gradient` lives |
| `lib/levanter/src/levanter/dpo.py` | DPO loss math; where logprobs get differenced |
| `lib/levanter/src/levanter/grad_accum.py` | Microbatch loop, where grads are accumulated and re-sharded |
| `lib/levanter/src/levanter/optim/` (or wherever Optax is wired) | Optimizer state construction over the adapter subtree |
| `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/xla.py` | **Probably not the bug.** Already extensively probed. |

---

## 13. References (read these before launching anything)

| File | Why |
|---|---|
| `.agents/logbooks/levanter_mesh_explained.md` | How Levanter constructs the mesh from device count. Critical for understanding why v5p-8 = 4-device mesh. |
| `.agents/logbooks/dpo-lora-claude.md` | Prior Claude-led DPO+LoRA investigation logbook. Likely contains earlier debug output worth grepping before re-running anything. |
| `.agents/logbooks/dpo-lora-codex.md` | Prior Codex-led investigation. Cross-reference for hypothesis lineage. |
| `.agents/logbooks/debug_accum_tpu_type.md` | Largest existing logbook on this whole thread. Contains the Exp R, S, T notes referenced elsewhere. |
| `.agents/logbooks/lora_ckpt.md` | LoRA checkpoint format / partitioning. Sometimes reveals subtree shape assumptions. |
| `.agents/projects/dpo_levanter.md` | Project-level context for the DPO + Levanter integration. |
| `.agents/projects/dpo_sharding_explainer.ipynb` | **Notebook explicitly about DPO sharding.** Likely has runnable examples that touch the same machinery. |
| `.agents/projects/linear_ce_loss.md` | CE kernel deep-dive. Helpful if you want to be thorough on §11.A. |
| `docs/debug-log-per-stmt-dpo-v5p-vs-v6e.md` | Public-facing version of the v5p vs v6e log. May have observations not in the agent logbooks. |
| `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/xla.py` | Has the `MARIN_DEBUG_CE_*` env vars and the `DEBUGCE` print at line 512. |

---

## 14. What to report back when you find it

When you have a fix or a confirmed mechanism, write a section here with:

1. **Mechanism in one sentence.** ("`grad(lora_A)` was psum'd over `model` instead of `data` because…")
2. **Which hypothesis (H1–H5) it confirms or contradicts.**
3. **The minimal repro.** Smallest config that exhibits the bug.
4. **The fix.** Diff or PR link. Whether it's in Levanter, Haliax, the LoRA module, or a config.
5. **Falsifiable confirmations.** Re-run Exp 1, 5, 7 and verify they pass with the fix. Verify v6e-4 (predicted broken pre-fix, predicted working post-fix).
6. **Whether Exp 5's chronic 17–39% sum divergence on healthy runs also disappears.** (If yes, you've also closed the slow-bleed version of the bug.)

---

## 15. Open meta-questions for the user (if you can ask)

These were ambiguous in the handoff and the answer changes the test plan:

1. **Does v6e-4 reproduce the bug?** This is the cleanest disambiguation between "v5p-8" and "4-device mesh". If you have v6e-4 capacity, run it before launching anything else.
2. **What is the actual `parameter_axis_mapping` in the failing config?** Specifically whether `embed` maps to `data` or to a separate `fsdp` axis, and whether `model` is in the mapping at all. This determines which `hax.dot` gets which psum.
3. **Is there any explicit `hax.shard` / `with_sharding_constraint` call inside `LoraLinear`, `LowRankLinear`, or the LoRA loss path?** If yes, that's a high-priority code-read.
4. **Does the failing run use `hax.shard_with_axis_mapping` anywhere on the adapter subtree specifically?** If it does and it's using the wrong mapping (compute vs parameter), that's a candidate.

---

## 16. Anti-patterns to avoid (lessons from prior runs)

- **Don't chase the CE kernel further.** It has been ruled out three different ways. The L2-vs-sum signature is wrong shape for a kernel-numerics bug.
- **Don't run more `pd` sweeps in isolation.** Geometry alone has been disambiguated.
- **Don't add more bf16/fp32 toggles** without a specific hypothesis they would test. Exp 1b settled the broad version.
- **Don't write the dump tooling against `state.model` only** — `state.opt_state` and the live `grads` PyTree are equally important and not reachable through `model`.
- **Don't compare runs without holding seeds, indices, and initial checksums** equal. Prior agents established these are equal; if your repro path can't preserve them, your comparisons are noise.
- **Don't trust "looks the same" on grad printouts.** Use L2 *and* sum *and* sign-balance, because the bug specifically hides under L2 alone.
