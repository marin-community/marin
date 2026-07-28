# Evidence ledger — B200/GB200 MoE throughput at EP32/EP64

Companion to [`README.md`](README.md). One card per change. Every measured number
carries its comparison arms; anything without a matched control says so.

Denominator convention throughout: **GB200 BF16 dense peak = 2.5 PFLOP/s per GPU**
(`lib/fray/src/fray/device_flops.py`, set in `c81a29428bd0`). Where a source
reports a W&B MFU on a different denominator the card says so.

Two measurement hazards apply to every EP number and are restated in the cards
that they distort:

- **Drop-inflated MFU.** Expert GEMMs run on fixed capacity-sized buffers, and a
  dropped assignment gathers a zero pad row. A configuration that drops more
  reads *higher* MFU for less real work, so MFU is only comparable within a
  matched drop regime
  ([#7279 comment 5080435482](https://github.com/marin-community/marin/issues/7279#issuecomment-5080435482)).
  **Calibrated from two 350-step legs: ≈ +0.30pp MFU per +0.10 of drop fraction.**
  The canonical illustration is the QB gain g=2 probe, which posted 23.386% while
  dropping 69%.
- **Reported MFU is inflated ×1.08 at seq 4096 with sliding window 512**, because
  `lm_flops_per_token` counts full O(seq²) attention on all 48 layers when 40 are
  windowed. Every EP64 figure in this document inherits that factor — a *reported*
  24.153% is ≈ **22.4% true**. It is applied uniformly, so A/B deltas hold and
  absolute levels do not. Do not carry these numbers into an external comparison
  without the correction.
- **Two disjoint experiment lines share this issue number but not an arm.** The
  ECHO line (`research/rav/7201-ep64-*`) runs d5120 / L48 / **top-4** of 256 /
  routed i2048 / one shared i5120 / SW512. The ep25 line (`agent/ep25-*`) runs
  d5120 / L48 / **top-8** of 256 / **i1280**. Both are EP64 on 16 × 4 GB200 off the
  same base `rav/ep-2` @ `fe21ea495`. **Their drop metrics are not the same
  quantity**: ECHO reports post-ECHO exact aggregate assignment drop, ep25 reports
  the sender-local fixed-bucket drop fraction. Top-k also moves the statistical
  floor — 0.88% at top-8 with mean-2048 buckets, 1.24–1.25% at top-4 with
  mean-1024. **Do not compare a number from one line against a number from the
  other without saying so.**
- **Schedule-position drop comparisons.** The LR schedule is defined over
  `num_train_steps`, so step 119 of a 120-step run is annealed while step 119 of
  a 350-step run is at ~68% of peak LR. Drop fractions are comparable only at the
  same fraction of the schedule, and a tail window beats any single step
  ([#7279 comment 5084892846](https://github.com/marin-community/marin/issues/7279#issuecomment-5084892846)).

---

## Group A — EP enablement

These buy no MFU on their own. Without them an EP64 job OOMs before step 1, so
every Group B number depends on all of them.

### A1 — Shard non-expert weights and optimizer state over `("data", "expert")`

**Mechanism.** `data = devices / (replica · expert)`, so EP64 collapses `data` to
1 and every non-expert weight (attention, shared MLP, lm_head) plus its optimizer
state replicates on all 64 devices. The fix defines `Pfsdp = ("data", "expert")`
and shards those tensors `P(Pfsdp, "model")` / `P("model", Pfsdp)`. `expert` is
intra-rack, so the all-gather stays on NVLink; the spec is identical to
`P("data", …)` at EP1.

**Evidence.** Without it: ~148 GiB/GPU at d5120/L48, OOM before step 1
([#7201 comment 5048941489](https://github.com/marin-community/marin/issues/7201#issuecomment-5048941489), fix 2).
Restated as a load-bearing fix in
[#7201 comment 5067493906](https://github.com/marin-community/marin/issues/7201#issuecomment-5067493906) (fix 2).

**Code.** `lib/levanter/src/levanter/grug/sharding.py` (the `Pfsdp` definition and
`Plm_head`), plus the expert/non-expert split in `experiments/grug/moe/model.py`.
The `sharding.py` part is 4 changed lines; verified by diffing
`origin/b200-300B-tune` against `agent/ep25-d1-adjoint`, where `sharding.py` is
the only shared-substrate file that differs and it differs by exactly this.

**Tracking issue.** [#7513](https://github.com/marin-community/marin/issues/7513) (open, no PR).

**State.** Branch-only. Not on `origin/main` — `main`'s `sharding.py` still has
`Plm_head = P(Pbatch[0], "model")`.

**Risk.** None known. No-op at EP1.

### A2 — Preserve expert sharding through MuonH 4D Newton–Schulz

**Mechanism.** `optim/grugmuon.py` merges the scanned 4D expert stack `(L,E) → LE`
before Newton–Schulz. The merge declared `PartitionSpec(None, "data", "model")`;
under EP (`data = model = 1`) that fully replicates the `[L·E, D, I]` bf16 stack
on every device before the reshard can redistribute it.

**Evidence — memory.** ~300 GiB at L48, observed OOM at 310 GiB
([#7201 comment 5048941489](https://github.com/marin-community/marin/issues/7201#issuecomment-5048941489), fix 1).

**Evidence — throughput.** This is not purely an enabler. SPMD cannot reshard a
stack onto a multi-axis dim0 in one hop and falls back to involuntary full
rematerialization (XLA b/433785288). A 64-GPU probe trio at d5120 L48 b1024
measured the broken path at **20.22% / 14.84 s per step with 208 SPMD warnings**
against the fixed path at **22.02% / 13.63 s with zero warnings** — **+1.8pp**,
matching NS-disabled speed with loss parity to real NS
([#7279 comment 5012284704](https://github.com/marin-community/marin/issues/7279#issuecomment-5012284704)).

**Two competing implementations — pick one.**

- `75c517148` on `mcwitt/moe-standalone-ep` (+54/−11): *skip the merge entirely*
  under EP. E stays on `"expert"`, only L migrates (single-axis all-to-all), then
  `jax.vmap(jax.vmap(local_ns))`. Also fixes `_newtonschulz_padded_stack_sharded`
  with a two-hop reshard, and refactors `is_w_down`/`trailing`/`orig_4d_spec` out
  of the branch. CPU-validated bit-exact, zero involuntary-remat warnings.
- `54bbe3d23` / `fe21ea495` on the rav lineage (grugmuon hunk +33/−4):
  `jnp.swapaxes(x, 0, 1)` to put E first, reshape into `P("expert", None, None)`,
  reshard, swap back.

They will conflict textually. `75c517148` is the better-argued design and carries
the extra padded-stack fix; the rav variant is the one with a 17.8% MFU 64-GPU
measurement behind it. **Resolve this before writing the commit.**

**Foundation commit.** `b0c7a1b56` (+222/−20) adds `_newtonschulz_4d_distributed`
and routes ndim-4 leaves to MuonH at all. Without it, 4D expert stacks silently
fell through to Adam. Must land first.

**Tracking issue.** [#7512](https://github.com/marin-community/marin/issues/7512) (open, no PR).
Extraction inventory: [#7490](https://github.com/marin-community/marin/pull/7490) (draft).

**State.** Branch-only. `_newtonschulz_4d_distributed` is absent from `main`.

**Risk.** Numerical parity of the NS update must be asserted against the
unsharded path.

### A3 — Keep the batch sharded over the expert axis before EP dispatch

**Mechanism.** In `moe_mlp`, under `if has_expert_axis and expert_axis_size > 1`,
override `_batch_spec_from_x(x, mesh)` with `_batch_spec(mesh)`. Without it the
activation layout at the EP `shard_map` boundary carries no expert axis.

**Evidence.** The in-code comment documents a silent **64× dispatch-buffer blowup
— 320 GiB against 5 GiB at EP64**. Listed as required fix 1 in
[#7201 comment 5067493906](https://github.com/marin-community/marin/issues/7201#issuecomment-5067493906).
No isolated A/B; it is part of the 17.8% reproduction.

**Code.** `lib/levanter/src/levanter/grug/grug_moe.py`. **+11/−0, of which 10 lines
are comment.** `main` calls only `_batch_spec_from_x`. Commit `54bbe3d23`.

**This is the highest value-per-line item in the entire ledger.**

**State.** Branch-only.

### A4 — Capacity-factor knob and overflow telemetry

**Mechanism.** `SCALE_CAPACITY_FACTOR` (default 1.0); receive capacity =
`capacity_factor × tokens_local × top_k`. Emits
`train/router/capacity_overflow_rate_mean`.

**Evidence.** [#7514](https://github.com/marin-community/marin/issues/7514) (open),
originating result
[#7201 comment 5048941489](https://github.com/marin-community/marin/issues/7201#issuecomment-5048941489) (fix 4).

**Note.** The overflow *rate of buckets* is not the token-drop fraction. Capacity
== mean load makes roughly half of all buckets overflow their tail while the
dropped tail is a much smaller fraction; an earlier "65–68%" reading was this
confusion
([#7279 comment 5080435482](https://github.com/marin-community/marin/issues/7279#issuecomment-5080435482) §2).
Ship C4 (the exact drop counter) alongside this, not instead of it.

**Two merge hazards.**

1. **`SCALE_CAPACITY_FACTOR` was implemented twice, independently, under the same
   environment-variable name** — `595958b83` and `3e149490f`. Reconcile before
   landing either; do not double-apply.
2. **The default is inconsistent between layers.** `_DEFAULT_EP_CAPACITY_FACTOR`
   is 1.0 in `experiments/grug/moe/model.py:51` while the library default in
   `lib/levanter/src/levanter/grug/_moe/common.py:19` is **1.25**. Receiver
   envelope factors default to 1.125
   (`_moe/ep_ragged_all_to_all.py:45`). Resolve to one canonical value — this is
   exactly the "centralize defaults in one location" rule in `AGENTS.md`, and a
   silent 1.25 would misprice every drop measurement taken against it.

**The capacity-factor price is a cliff, not a slope.** cf 1.00 → 1.05 costs
**1.179pp** for +0.05; cf 1.05 → 1.15 costs **0.254pp** for +0.10. The natural
tile-alignment hypothesis (capacity 2048 = 16 × 128, 2151 not) was **falsified** —
`SCALE_CAPACITY_TILE=128` at cf1.05, byte-identical to cf1.0625, moved MFU
+0.038pp, which is noise (commits `58ee2cdd0`, `b08d5c5ca`; the knob is left
default-0 and marked unvalidated). **Cause unknown.**

**State.** Branch-only.

### A5 — CUTLASS DSL 4.6.0

**Mechanism.** Removing the obsolete 4.5.2 dependency override restores
`cutlass.cute`, which `gpu_fa4_cute` requires. Also supersedes the wheel-shadowing
patch [#7491](https://github.com/marin-community/marin/issues/7491).

**Evidence.** Listed as required fix 5 in
[#7201 comment 5067493906](https://github.com/marin-community/marin/issues/7201#issuecomment-5067493906).
Landed with a 64-GPU full-stack run at 23.106% MFU.

**State.** **Merged** — [#7587](https://github.com/marin-community/marin/pull/7587), commit
`8f1ba5363`, which removed the `nvidia-cutlass-dsl-libs-base==4.5.2` override from
the root `pyproject.toml` and pinned `==4.6.0` in `lib/levanter/pyproject.toml:90`
and `lib/marin/pyproject.toml:123`. **Do not cherry-pick the branch versions** —
`rav/ep-2` re-does the removal (it branched 3.5 h before the main fix) and the base
branch uses `>=4.6.0,<4.7`. Only the `quack-kernels[cu13]==0.6.1` gpu-extra line
and the `cutlass`/`quack` mypy `ignore-missing-imports` entries still need taking.

**Root cause worth recording.** `nvidia-cutlass-dsl` installs both `-libs-base`
and (via the `cu13` extra) `-libs-cu13`, and the two wheels ship **99 overlapping
files** with different CUDA-12/CUDA-13 builds of the DSL frontend and MLIR
compiler. Whichever won the install silently decided whether any CuTe kernel
compiled, re-rolled on every env sync — which is why it presented for weeks as
random per-node GB200 heterogeneity. Multi-pod compile surveys went from
coin-flip-per-pod to 100% green on the fixed lock
([#7282 comment 5016674287](https://github.com/marin-community/marin/issues/7282#issuecomment-5016674287)).

### A6 — JAX 0.11.0 with the OpenXLA CUBIN discriminator fix

**Mechanism.** `TritonFusion` and `MlirKernelFusion` shared an empty kernel-reuse
discriminator, so an MLIR fusion could hit a Triton cache entry whose binary is
intentionally empty (it compiles later) and serialize that emptiness as an owned
CUBIN. On GB200 this produced intermittent large-program CUBIN load failures that
taxed EP work for weeks. Fix is OpenXLA `4c1b005`, in-tree via the JAX 0.11.0 bump.

**Evidence.** [#7421](https://github.com/marin-community/marin/issues/7421) (closed);
fix confirmation
[#7421 comment 5064478402](https://github.com/marin-community/marin/issues/7421#issuecomment-5064478402).

**State.** **Merged** — [#7436](https://github.com/marin-community/marin/pull/7436).

**Consequence.** Removing the mask exposed what it hid: the hetero-KV nested scan
in [#7407](https://github.com/marin-community/marin/issues/7407) now fails with an
honest NCCL OOM, peaking 18.7 GiB/device above the uniform single-scan path after
rematerialization.

---

## Group B — EP throughput

### B1 — FA4 CuTe attention (`SCALE_ATTN_IMPL=gpu_fa4_cute`)

**Mechanism.** SM100 FlashAttention-4 CuTe kernel replaces reference attention
that materialized the `bf16[batch, heads, seq, seq]` score matrix (~80 GiB at the
operating point).

**Measured.** ~10.5% → ~13% MFU at d5120 8-of-256 EP64, 1 rack, batch 1024, seq
4096 ([#7201 comment 5048941489](https://github.com/marin-community/marin/issues/7201#issuecomment-5048941489), fix 3).
That is a +2.5pp step, but it is the difference between two early configurations
rather than a single-variable A/B.

**State.** **On `main`** — `lib/levanter/src/levanter/grug/attention/_fa4_cute*.py`,
10 files reference `gpu_fa4_cute`. Segmented-attention tracking issue
[#5896](https://github.com/marin-community/marin/issues/5896) remains open.
Outstanding defect: THD FA4 full-causal and sliding-window modes fail during CuTe
compilation ([#7483](https://github.com/marin-community/marin/issues/7483)), fixed
by [#7630](https://github.com/marin-community/marin/pull/7630) (open) — the CuTe
DSL traces both arms of a plain `if`, so window sizes must be resolved before
tracing.

### B2 — Fixed-capacity `jax.lax.all_to_all` replacing `ragged_all_to_all`

**Mechanism.** The ragged collective lowered to many small `SendRecv` kernels.
A fixed-capacity layout (64 sender shards × 256 experts, capacity
`cf × tokens_local × topk` per bucket) uses one static `all_to_all` per leg.
`SCALE_A2A_CHUNKS` bounds activation memory.

**Measured.** ~13% typical → **17.823% p50 / 276.1K tok/s / 15.2 s per step**,
d5120 8-of-256 48L EP64, 1 rack, batch 1024, seq 4096, 30 steps
([#7201 comment 5067493906](https://github.com/marin-community/marin/issues/7201#issuecomment-5067493906)).
The pinned reference snapshot reached 17.220% / 266.6K tok/s. That comment also
notes the fixed path's *own* measured gain at seq 4096 is small because the layer
is compute-bound; most of the ~13% → ~17.8% move is the bundle of Group A fixes
plus removing the small-kernel storm.

**Caveat.** The ~13% baseline is placement-noisy: the same config swung to 16.9%
purely on node placement, and a direct repro landed back at 13.5%
([#7201 comment 5048941489](https://github.com/marin-community/marin/issues/7201#issuecomment-5048941489)).
Single-run A/Bs at EP64 need several placement draws.

**Matched control never run.** Fixed-a2a was adopted without a matched ragged
control ([#7279 comment 5074952738](https://github.com/marin-community/marin/issues/7279#issuecomment-5074952738), direction 2).
The closest thing on record is the later one-draw comparison in B4's table.

**Also required for the overlap ceiling.** The pipelined decomposition family is
only implementable on the fixed-capacity layout (static per-peer buckets; the
ragged path has no compile-time peer slices) — so the fixed path wins on option
value even at head-to-head parity. That family has since measured negative (E3,
E4), which weakens but does not void the argument.

**State.** Branch-only. The only implementation is `fe21ea495` (`origin/rav/ep-2`):
**+164/−0** in `lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py`, plus
`test_grugformer_moe.py` +75/−2 and `experiments/grug/moe/test_model.py` +42. It is
additive alongside the ragged path, but the carrying commit *also* bundles the
`Pfsdp` change, the MuonH fix, a cutlass revert and dispatch env forwarding —
**it needs splitting, not cherry-picking.** Key lines: `:43`
`chunks = max(int(os.environ.get("SCALE_A2A_CHUNKS", "1")), 1)`; `:101`
`capacity = ceil(capacity_factor * assignments_per_shard / num_experts)`;
`:137`/`:151` `jax.lax.all_to_all` where `main` has `jax.lax.ragged_all_to_all` at
`:241`/`:269`.

**Do not cherry-pick from `research/rav/7201-ep64-drop3{,-handoff}` or
`7201-ep64-muon-pad`** — they carry the same knobs but are entangled with ~14–16k
lines of `scripts/hybridep_build_probe/` CUDA/FFI scaffolding, patches and
logbooks.

**Chunking tuning.** `SCALE_A2A_CHUNKS=2 → 1` is an improvement, and is part of
the 17.17% → 17.55% delta reported alongside B3.

**Related but narrower than it looks:**
[#7494](https://github.com/marin-community/marin/pull/7494) (`41ce33431`, +38/−1,
3 files) flags and warns on XLA's slow one-shot ragged path
(`--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false`). At 4×GB200
d2560 EP4 that flag alone collapsed step time **52.8 s → 6.9 s (1.88% → 14.4%
MFU)**, corroborated by openxla/xla#33386 (~2% of NVLink bandwidth one-shot vs
~100% wire utilisation on NCCL)
([#7012 comment 4995418725](https://github.com/marin-community/marin/issues/7012#issuecomment-4995418725)).
**But it only matters single-host** — multi-host `ragged_all_to_all` always takes
the NCCL path automatically, confirmed in the 512-expert EP64 profile
(`ncclDevKernel_SendRecv`, not `RaggedAllToAllKernelImpl`). At rack scale this is
a guard against a footgun, not a speedup.

### B3 — Gather dispatch (int32 assignment scatter + activation gather)

**Mechanism.** The fixed-a2a dispatch built its send buffer by repeating each
token's bf16 activation row once per top-k assignment (an 8× bf16 row repeat at
top-8). Instead, scatter int32 assignment indices and then gather activations:
`send_x = padded_x[token_sources]`.

**Measured.** Matched 120-step A/B on 64 GB200, d5120 8-of-256 48L EP64, batch
1024, seq 4096, `SCALE_A2A_CHUNKS=1` held on both sides:

| dispatch construction | p10 | p50 | p90 | mean | p50 tok/s | p50 step |
|---|--:|--:|--:|--:|--:|--:|
| repeated bf16 activation scatter | 17.501% | **17.552%** | 17.603% | 17.550% | 271.9K | 15.43 s |
| int32 assignment scatter + gather | 20.467% | **20.558%** | 20.662% | 20.563% | 318.5K | 13.17 s |

**+3.006pp / +17.13% relative**, bands non-overlapping
([#7201 comment 5073017396](https://github.com/marin-community/marin/issues/7201#issuecomment-5073017396)).

**Fidelity caveat.** Both arms route with QB off, i.e. in the router-collapse
regime — see the drop-inflated-MFU hazard at the top and
[#7201 comment 5080459722](https://github.com/marin-community/marin/issues/7201#issuecomment-5080459722).
The A/B is matched, so the *delta* is sound; the absolute level is not a
shippable figure.

**Correctness.** Gather and scatter produce exactly equal forward outputs and
identical dropped-token counts in the kernel test; gradients match at
rtol = atol = 1e-5. **Open loose end:** the two distributed 120-step jobs did not
track pointwise — steps 100–119 mean loss 5.81466 (scatter) vs 5.87887 (gather),
Δ +0.064, attributed to independent-run RNG divergence but never confirmed.

**State.** Branch-only. Reconstruction commit `45ce02d20` on local
`agent/ep25-d1-adjoint` (**not pushed to origin**), based on `rav/ep-2` @
`fe21ea495`. Env knob `SCALE_A2A_GATHER_DISPATCH=1`.

**Complexity: trivial — 1 file, `ep_ragged_all_to_all.py`, +17/−2.** This is the
best benefit-to-complexity ratio in the ledger by a wide margin.

**Dependency.** B4 requires this — the custom adjoint transposes *these* gathers.

### B4 — Custom scatter-add adjoint for the dispatch and combine gathers

**Mechanism.** Under autodiff both gathers transpose to a generic scatter-add,
which XProf flagged as one of the two largest backward costs. Both have exact
structured transposes expressible from the forward int32 index composition: the
dispatch backward is a segment-sum over each token's top-k send slots, and the
combine backward is an injective gather along the slot→assignment inverse (no
accumulation). A `custom_vjp` routes both through those forms. On the
operating-point shape the backward HLO drops from **544 scatter ops to zero**.

**Measured.** Matched 120-step A/B, back-to-back, checkpointing disabled, same
shape as B3:

| dispatch backward | p10 | p50 | p90 | loss @ 119 |
|---|--:|--:|--:|--:|
| XLA autodiff (scatter-add) | 20.51 | **20.61** | 20.69 | 5.738 |
| custom adjoint | 23.73 | **24.04** | 24.75 | 5.711 |

**+3.43pp / +16.6%**, p10/p90 bands non-overlapping
([#7279 comment 5080435482](https://github.com/marin-community/marin/issues/7279#issuecomment-5080435482) §1).

**Correctness.** Kernel-level gradient parity vs autodiff at rtol = atol = 1e-5
for the input, the combine weights, and both expert weight tensors, with identical
dropped-token counts. The 0.027 loss difference at step 119 is between two
separate jobs and is consistent with RNG divergence.

**Transport comparison at the same shape** (QB off, single draw each):

| transport | MFU | drop @119 |
|---|--:|--:|
| fixed + gather + adjoint | 24.04 (p50) | collapsed |
| ragged, one-shot kernel off | 12.38 (mean) | 0.433 |
| `ring_cute` EP64 | DNF — OOM 141.79 GiB in `jit_train_step` | — |

**State.** Branch-only. `custom_vjp` commit `c9e30f848` on local
`agent/ep25-d1-adjoint`; env knob `SCALE_A2A_CUSTOM_ADJOINT=1` (requires
`SCALE_A2A_GATHER_DISPATCH=1`). *"No pushes have happened; landing the adjoint and
the drop metric needs a PR from this branch."*

**Complexity: 2 files, +234/−10** (`ep_ragged_all_to_all.py` +127,
`test_grugformer_moe.py` +117).

**This is the single largest EP-specific win on record.**

### B5 — Leg-batched expert GEMMs

**Mechanism.** Batch the per-local-expert GEMM legs instead of looping. Related:
the per-local-expert `all_to_all` loop is four collectives per layer per direction
and is a candidate for the same treatment.

**This is the workstream's clearest contradiction: the same idea measured +1.35pp
and −3.66pp.** Do not treat it as a pending win.

- **Positive, QB-off:** 25.39% p50 on a 120-step run
  (`rav/ep64-batched-expert-stability-120-v1-20260724-2353`), against B4's 24.04%.
  **Not a matched A/B** — two runs on the same lineage — and taken at ~85% early
  drops, so per the drop-inflation calibration above a large part of that gap is
  the drop regime, not the kernel
  ([#7279 comment 5080435482](https://github.com/marin-community/marin/issues/7279#issuecomment-5080435482) §1, §5).
- **Negative, QB-on and matched drops:** an independent reconstruction
  (`SCALE_A2A_BATCH_EXPERTS` / `SCALE_A2A_BATCH_GROUP=2`, commits `65e3ca50d`,
  `0789a8482`) measured control 22.66% p50 at 0.088 drops against batched G=2
  **19.00% p50 at 0.092 drops — −3.66pp, bands non-overlapping.** The batched path
  is bit-exact against the loop at `expert_axis=2` (max abs diff 0.0), so this is a
  performance regression, not a correctness bug. **Full batching at G=4 never
  produced a step** — two gang aborts. The implementing agent dropped confidence
  from 6/10 to 2/10.

The ECHO line has its own `SCALE_A2A_BATCH_EXPERT_GEMMS=1` (hard-requiring
`SCALE_A2A_PACK_DISPATCH=1` and `SCALE_A2A_PACK_COMBINE=1`) with parity tests, but
**no rack MFU number exists for it and it is not set in the production runner.**

**Status: unresolved, two incompatible implementations, neither pushed.** The
25.39% and the −3.66pp are different code. Nobody has run leg-batching with QB on
in the implementation that produced the 25.39%.

**State.** Branch-only.

### B6 — XLA collective-overlap flags (config-only, no code change)

**Mechanism.** Two independent settings. (a) Manual PGLE feeds the latency-hiding
scheduler *measured* collective latencies rather than heuristic ones; this takes
the gradient reduce-scatter from 65.4% to 100% hidden. (b)
`xla_gpu_experimental_parallel_collective_overlap_limit` defaults to **1**;
raising it to **4** moves every MoE all-to-all onto the async stream
(compute-stream collective time 2961 ms → 0).

**Measured.** **+0.47pp together**, at the EP64 operating point
([#7279 comment 5084892846](https://github.com/marin-community/marin/issues/7279#issuecomment-5084892846)).
On the FSDP line PGLE alone was worth **+1.1pp** (23.70% → ~24.8%) at 1 rack and
**+1.0pt** at 2 racks — see D1.

**The more important result is the ceiling it exposes.** With collectives async,
exposed collective time (4.29 s) almost exactly fills compute idle (4.44 s of a
33.2 s span). The step is collective-**volume**-bound, not schedule-bound. That
caps the entire scheduling family — rotation, prefetch, token-chunk pipelining,
PGLE and the overlap limit all measured null or negative — and leaves *reducing
collective bytes* as the only remaining lever in that family. The finding
reproduced at the d6144 hero shape: span 42,198 ms per 3 steps, compute busy
38,700 ms (91.7%), compute idle 3,498 ms, exposed collective 4,126 ms
([#7279 comment 5095217108](https://github.com/marin-community/marin/issues/7279#issuecomment-5095217108)).

**Root cause of the inline collectives, and a free diagnostic.**
`GpuCompiler::RunPostSchedulingPipelines` runs `GpuConvertAsyncCollectivesToSync`,
which tags any async-start whose matching done is separated only by no-ops as
`is_sync=true`. A schedule-dump harness
(`experiments/grug/moe/schedule_report.py`) reproduces the census in about three
minutes on one node:

| overlap limit | MoE SYNC a2a | reshard SYNC |
|--:|--:|--:|
| 1 (default) | **10** | 6 |
| 2 | 3 | 14 |
| **4** | **0** | 14 |
| 8 | 1 | 14 |

**This is the same phenomenon as the "three of twelve all-to-all ops on the compute
stream" lead** recorded independently at the d6144 shape (F1) — and
`overlap_limit=4` is the fix for that class. Note `overlap_limit=2` measured
*worse* than 1 on the ECHO line (21.910%), so the setting is not monotone; use 4.

**Measured separately on the two lines.** ep25 (120-step, drop-corrected):
PGLE + LHS +0.34pp raw / +0.32pp corrected, `overlap_limit=4` a further +0.12pp,
**+0.47pp combined**; two control legs 33 minutes apart agreed to 0.000pp, so
allocation noise is ≤0.05pp. ECHO: LHS **+0.599pp**, then `overlap_limit=4`
**+0.427pp** — but **manual PGLE was rejected there** (it matched only 217 of 535
instructions and came in 0.235pp *below* the AutoPGLE leg), and AutoPGLE's CUPTI
profiles came back empty on some attempts. **PGLE is not a reliable win on the EP
stack the way it is on the FSDP stack.**

**A hard requirement on JAX 0.11.**
`--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false` is
mandatory: without it a 64-process run initialises, compiles the train step, and
then **segfaults in NCCL `ncclDevCommCreate` through XLA's
`NcclDeviceCommunicator` before step 0** (fix commit `cbd960569`). Separately,
**the JAX 0.11 baseline is 1.217pp below the 0.10.1-era baseline** (21.815%
against 23.032%) — do not borrow pre-0.11 baselines into a post-0.11 comparison.

**Two flags that break this build:** `xla_gpu_enable_custom_fusions` and
`xla_gpu_enable_address_computation_fusion` kill the process before distributed
init. `xla_gpu_experimental_collective_start_as_early_as_possible` does not exist
on jaxlib 0.10.1.

**The ceiling this implies.** ECHO's own arithmetic: perfectly hiding *all*
remaining exposed communication moves 23.286% to only **~25.7%**, and 30% would
need ~7.41 s/step against the 9.545 s measured.

**State.** Config-only, usable today on any branch — the highest-leverage
already-available item. Not in `main`'s default submit path.

### B7 — Padded, stack-sharded non-expert Muon (`SCALE_MUON_PAD_NONEXPERT=1`)

**Mechanism.** Five 48-layer non-expert Muon parameter stacks are zero-padded to
64 rows so the Newton–Schulz batch can shard across the 64-way expert mesh. The
padded result is resharded directly to the original parameter layout before
slicing, so the padded stack is never replicated on every device. Dummy rows stay
zero and are removed before the update is applied.

**Measured.** Matched one-rack A/B changing only `SCALE_MUON_PAD_NONEXPERT` 0→1,
d5120 4-of-256 (i2048) 48L EP64, batch 1024, seq 4096, sliding window 512, 20
steps, PGLE off, over steps 5–19:

| run | median MFU | median step | median tok/s | mean drop | final loss |
|---|--:|--:|--:|--:|--:|
| matched control | 22.374% | 9.795 s | 428,217 | 2.918% | 7.943 |
| padded, stack-sharded | **24.153%** | 9.073 s | **462,267** | 2.765% | 7.945 |

**+1.78pp / +7.9% tok/s**, and drops move in the favourable direction (2.918% →
2.765%), so this is not a drop-inflated gain
([#7201 comment 5088824573](https://github.com/marin-community/marin/issues/7201#issuecomment-5088824573)).

**Caveats.** 20 steps is a performance screen, not a stability or drop-rate
qualification. No XProf/HLO capture was taken after the optimizer change.

**Correctness.** Local BF16 checks over square and both rectangular orientations
produced identical padded/unpadded updates; repository tests assert the padded
result returns with the original shape and sharding.

**State.** Branch-only. Implementation
[`497423bc6`](https://github.com/marin-community/marin/commit/497423bc6716ad102fc53ebdebd7bfdb38304dcc)
on `research/rav/7201-ep64-muon-pad`; snapshot `1a88e1f3b`.

**Complexity: 2 files, +54/−1** (`optim/grugmuon.py` +19,
`tests/test_grugmuon.py` +36). **The best measured-gain-per-line item after B3.**

**Provenance nuance that matters for the cherry-pick.** The flag and the function
`_newtonschulz_padded_stack_sharded` **already exist** on the shared base
`rav/ep-2` and on all seven `agent/ep25-*` branches. What `497423bc6` adds is the
`target_sharding=` argument — without it the padded result reshards to fully
replicated `P(None, None, None)` before slicing, which is exactly the cost the
mechanism is meant to remove. **The +1.78pp belongs to the fixed variant only**,
and the ep25 fan-out never measured the flag at all. Take `497423bc6`, not the
flag.

**Guards in the shipped code.** The new test asserts no replicated-padded reshard
appears in the jaxpr, and the code raises `ValueError` if the parameter layer axis
is not replicated.

**Dependency.** Needs an expert axis at least as wide as the number of non-expert
parameter stacks, and MuonH. It also overlaps A2's code — land A2 first and
resolve them together.

### B8 — QuACK SM100 grouped expert GEMMs (`sonic_cute`)

**Mechanism.** SM100 grouped-GEMM backend for the expert matmuls via
`cutlass_call`, plus grouped weight-gradient GEMMs.

**Evidence.** Improved matched B200 whole-model MFU while preserving loss and
gradient parity. PoC
[`a3b19c5ff`](https://github.com/marin-community/marin/commit/a3b19c5ff0351a2b6d59f2a130a67be50d99d488);
validation [#7012 comment 4919640071](https://github.com/marin-community/marin/issues/7012#issuecomment-4919640071);
independent reproduction [#7012 comment 4960848663](https://github.com/marin-community/marin/issues/7012#issuecomment-4960848663).
Under the ring EP backend: `107476c8d`, validated in
[#7012 comment 4994519151](https://github.com/marin-community/marin/issues/7012#issuecomment-4994519151) and
[#7279 comment 5028967210](https://github.com/marin-community/marin/issues/7279#issuecomment-5028967210).

**State.** Branch-only; extraction marker
[#7488](https://github.com/marin-community/marin/pull/7488) (draft, branch
`sonic-cute-moe-b200`). Files:
`lib/levanter/src/levanter/grug/_moe/sonic_cute.py` (+272),
`quack_moe_cute.py` (+172), `quack_symmetric_cute.py` (+117). Byte-identical on
`origin/b200-300B-tune` and `agent/ep25-d1-adjoint` — this is shared substrate for
both the FSDP and EP lines.

**Note.** `sonic_cute` is the *local/FSDP* expert backend. At EP64 it supplies the
grouped GEMM that runs on the received buckets; `SCALE_MOE_EXPERT_CHUNKS` (D5) is
the FSDP-only overlap layered on top and does not apply under EP.

### B9 — QuACK grouped weight-gradient GEMM at 256×256 tiles

**Mechanism.** `SCALE_QUACK_GROUPED_WGRAD=1` — route the expert weight-gradient
GEMMs through the QuACK grouped kernel with the autotuned 256×256 tile.

**Measured. +0.861pp** on the ECHO line (leg v134).

**State.** Branch-only, `research/rav/7201-ep64-drop3`.

**Note the tension with E22**, which records the `sonic_cute` varlen-k wgrad shim
at only +0.06–0.08pp at the d2560 row-13 scale. These are different kernels at
different shapes; the EP64 figure is the relevant one for this operating point,
but neither has a matched control against the other.

### B10 — Sonic slot gather (the ECHO line's equivalent of B3 + B4)

**Mechanism.** `SCALE_A2A_SONIC_DISPATCH`, `SCALE_A2A_CLONE_SONIC_SLOT_GATHER`,
`SCALE_A2A_SONIC_COMBINE`, `SCALE_A2A_CLONE_SONIC_CUTE` — replaces the same generic
gather adjoints that B4 targets, by a different route.

**Measured. 18.82% → 20.67% (+1.85pp)** with no routing or clipping change;
30-step qualification 20.51% p50 at 1.28% mean / 1.94% max drop.

**State.** Branch-only, ECHO line.

**Do not stack this with B4 without measuring.** They address the same cost by
different mechanisms and the record contains no arm with both.

**Rejected sibling.** The Sonic *clone-weight reduction* adjoint
(`SCALE_A2A_CLONE_SONIC_WEIGHT_GRAD`) microbenchmarked **3.24× (W2) / 4.63× (W13)**
at zero error, and the matched 20-step rack A/B **regressed −0.167pp** (23.032%
against 22.865%). Microbenchmark-overstates-e2e, again — see the pattern note in
[`README.md`](README.md).

---

---

## Interlude — what EP degree actually buys, and why EP32 is not an option

The request framed the target as "the EP32 or EP64 regime". The record does not
support EP32 at production shape.

**EP32 is essentially unmeasured, and the two data points are bad.**

- 32-GPU d2560 sweep: `ring_cute` EP32 = **9.5%** (decaying steeply from EP4's
  14.3%); `ragged_all_to_all_cute` EP32 = **12.3%** (roughly flat in EP degree,
  12.1 → 12.3 from EP4 → EP32)
  ([#7012 comment 4996624949](https://github.com/marin-community/marin/issues/7012#issuecomment-4996624949)).
- At the d5120 L48 b1024 reference config, **both EP32 arms OOM** on a single
  ~104 GiB temporary — the SPMD involuntary-full-rematerialization fallback (XLA
  b/433785288) for resharding the microbatch input into the `(data, expert)` mesh,
  not a dispatch-buffer limit
  ([#7279 comment 4998828714](https://github.com/marin-community/marin/issues/7279#issuecomment-4998828714)).
  This is the same root cause A2 fixes for the optimizer stack; whether A2's fix
  also unblocks EP32 input resharding is untested.

**EP64 = exactly one NVLink domain.** The grug mesh is
`("replica_dcn", "data", "expert", "model")` with `expert` innermost
(`lib/levanter/src/levanter/grug/sharding.py`), so EP64 keeps every MoE all-to-all
inside one rack and never on InfiniBand. EP32 buys none of the memory relief that
motivates EP while still paying dispatch overhead, and EP > 64 would put the
all-to-all on IB. **Treat EP64 as the only EP operating point.**

**EP pays off in proportion to expert count.** At E512/top-8, d4096 L32 seq8192,
64× GB200, bf16: EP1 6.05% → EP8 12.56% → EP16 13.66% → **EP64 14.39%** — monotone,
a 2.4× jump end to end
([#7332 comment 5007041962](https://github.com/marin-community/marin/issues/7332#issuecomment-5007041962)).
That inverts the E64 result where FSDP beats EP. The practical read: **EP is the
right parallelism for the 256-expert and 512-expert candidates and the wrong one
for 64 experts.** Adding unscaled e4m3 on the wire and into QuACK takes E512/EP64
to 18.2% / 417K tok/s, but that FP8 arm is MFU-measured only and **not numerically
validated** (~8% relative GEMM error against bf16's 0.5%).

**Placement variance is a first-class confound.** Same binary, different gang
draw: ring EP4 moved −10% step time; a2a EP8 flipped from wedge to a clean 19.51%;
the EP64 ragged config swung 13.5% ↔ 16.9%. Multi-rack GB200 gangs get only a
SOFT `nvlink.domain.preferred` packing constraint. **Any margin claim under ~2pp
needs repeated placement draws**
([#7279 comment 5012284704](https://github.com/marin-community/marin/issues/7279#issuecomment-5012284704),
[#7201 comment 5048941489](https://github.com/marin-community/marin/issues/7201#issuecomment-5048941489)).

---

## Group C — Fidelity: routed-token drops

The user-facing bar is that a configuration dropping more than a few percent of
routed assignments is not shippable. `~3%` is the known-acceptable rate at 8
buckets from a prior 1e23 run. Everything in Group B was benchmarked with QB off,
where the router collapses; these are the mechanisms that make an EP64 number
honest.

### C1 — QB (aux-loss-free quantile-balancing) routing, `SCALE_MOE_QB=1`

**Mechanism.** DeepSeek-V3-style router-bias load balancing. Grug's implementation
is an implicit proportional controller: it applies a 1× router-bias residual per
step, not DeepSeek's ±γ integral accumulation.

**Measured.** 120-step frontier at d5120 8-of-256 EP64, batch 1024, seq 4096:

| config | p50 MFU | drop @119 | loss tail |
|---|--:|--:|--:|
| QB off, cf1.0 | 24.04 | collapsed, 0.17–0.79 over run | 5.711 |
| QB off, cf1.15 | 22.13 | 0.649 | — |
| QB on, cf1.0 | 22.60 | 0.083 | 5.767 |
| QB on, cf1.15 | 20.85 | 0.037 | 5.788 |

QB costs **at most −1.44pp**; capacity factor alone buys no fidelity (cf1.15
QB-off pays −1.91pp and still drops 0.649). These are cross-drop-regime gaps, so
the costs are upper bounds
([#7279 comment 5080435482](https://github.com/marin-community/marin/issues/7279#issuecomment-5080435482) §2).

**Steady state.** A 350-step QB-on cf1.0 run (loss healthy to 3.335) gives drops
0.885 @5, 0.271 @60, 0.175 @119, 0.089 @250, 0.064 @349, tail-100 mean **7.3%**.
cf1.0 with QB alone levels toward ~6–7% and does **not** cross a 3% bar.

**The whole controller family is now closed. Four variants, all measured, all at
or below `g=1`:**

| variant | knob / commit | result |
|---|---|---|
| Over-relaxed, g=2 | `SCALE_QB_GAIN`, `58c9a19eb` | p50 23.386% but drops pinned **0.675–0.793 for all 350 steps** (overshoot limit cycle), loss **+0.091 worse**. Categorical negative. |
| Damped, g<1 | same knob | **Cannot beat g=1 by construction** — `pending ← g·beta + (1−g)·pending` has the same fixed point; gain sets only the approach rate. The one g=0.5 rack leg ran 350/350 clean but its metrics were permanently lost to a concurrent GB200 log-shipping outage. |
| DeepSeek-style integral | `SCALE_QB_INTEGRAL`, `3f10dcc6a` | **Clean negative at both gammas**: γ=0.001 plateaus at ~0.60, γ=0.01 at ~0.46, against g=1's 0.073; losses 3.452 / 3.472 against 3.335. The first-5-step collapse (peak 0.89–0.91 in *every* draw) outruns any fixed rate — DeepSeek's rule works over 100k+ steps by *preventing* drift and cannot *reverse* an established collapse in 350. |
| **Sender-local bias** | `SCALE_QB_SENDER`, `50748b995` / `5bf934717` | **Null.** A closed-loop CPU simulation was decisive (global QB stuck at 0.773 over 12 iterations; sender QB 0.758 → 0.016), but the live 350-step leg gave tail-100 **0.0856 against global's 0.0732 — statistically identical**, at parity MFU and loss. |

**The sender-local result overturns the leading hypothesis.** Sender-local bucket
hotspots were the standing explanation for the ~6% residual, and the mechanism
aimed directly at that cause did not move it. The revised conclusion on the record
is that **the residual is batch-stochastic within-batch burstiness, invisible to
any one-step-delayed bias controller** of either kind.

**Statistical floor for context.** At bucket mean 2048, uniform routing floors at
0.91% simulated / 0.88% analytic. Observed 6–8% implies σ 329–411 against a Poisson
45.3 — **routing is 7–9× more clustered than independent-uniform.** The gap is not
controller error.

**State.** `SCALE_MOE_QB` is on the shared `rav/ep-2` base and used in production
submits on both lines. The gain, integral and sender probes are branch-only on
`agent/ep25-d4-pipelined` / `agent/ep25-d3-qbprobes` and were never pushed.

### C2 — Same-step spill (`m = 3`)

**Mechanism.** When a token's chosen expert bucket is full, re-offer the
assignment to the next-ranked expert *that same token already selected*, if that
bucket has headroom, instead of dropping it. Bucket layout, capacity and the
fixed-capacity all-to-all are unchanged.

**Measured.** 350-step legs, same allocation draw, true 100-step tails, QB on,
custom adjoint, d5120 8-of-256 EP64:

| spill m | cf | capacity | p50 MFU | tail-100 drops | ≤3%? |
|--:|--:|--:|--:|--:|:--|
| 0 | 1.0 | 2048 | 22.062 | 7.10% | no |
| 3 | 1.0 | 2048 | 21.849 | 3.66% | no |
| 3 | 1.05 | 2151 | 20.670 | 1.72% | yes |
| **3** | **1.0625** | 2176 | **20.708** | **1.44%** | **yes** |
| 0 | 1.15 | 2356 | 20.416 | 2.60% | yes |

Spill **halves drops for 0.213pp** with loss at parity or slightly better. It is
cheap for a structural reason: expert GEMMs run on capacity-sized buffers whether
or not a slot is filled, so spill adds index work and never matmul work. Combining
spill with a small capacity bump beats buying capacity alone on both axes —
20.708% at 1.44% versus 20.416% at 2.60%
([#7279 comment 5084892846](https://github.com/marin-community/marin/issues/7279#issuecomment-5084892846),
[#7201 comment 5084895357](https://github.com/marin-community/marin/issues/7201#issuecomment-5084895357)).

**Two hard constraints this establishes.**

1. **Spill is capped at `top_k − 1` attempts**, so drop-recovery headroom is a
   function of routing granularity. Measured through the shipping kernel: at
   4-of-256 the drop fraction is flat from m=3 through m=7 (2.88% throughout),
   while 8-of-256 keeps improving (2.73% → 1.57%). Top-4 shapes also carry a
   higher statistical floor because the per-bucket mean load halves. **This is an
   architecture-selection input, not just a kernel detail.**
2. **Capacity factor below 1.0 cannot be rescued by any routing or spill
   mechanism.** Total capacity is exactly `cf × total_assignments`, so drops are
   bounded below by `1 − cf`. cf1.0 is the fastest feasible operating point under
   any drop bar.

**State.** Branch-only. Commit `1224ccb02` (`SCALE_A2A_SPILL`) on local
`agent/ep25-d1-adjoint` / `agent/ep25-d5-d6144`. **2 files, +147/−12.**

**Dependency.** Fixed-capacity a2a (B2); composes with the custom adjoint (B4).

### C3 — Receiver-ECHO with same-expert clones

**Mechanism.** Receiver-pooled placement with same-expert clones, sparse clone
weights, two padding experts, and at most ten receiver expert segments, plus two
sender-balanced pipeline chunks. An accepted routed assignment executes the
router-selected expert either on its home shard or on a same-expert receiver
clone. The path is **not dropless** — fixed receiver envelopes remain bounded and
assignments exceeding them are dropped.

**Measured.** d5120 4-of-256 (i2048) 48L EP64, 1 rack, batch 1024, seq 4096,
sliding window 512, 20 steps, PGLE off, XLA latency-hiding scheduler with
collective-overlap limit 4, CUDA command buffers on:
**24.153% median MFU, 462,267 tok/s, 9.073 s/step, 2.765% mean aggregate
post-ECHO assignment drop, final loss 7.945**
([#7201 comment 5088824573](https://github.com/marin-community/marin/issues/7201#issuecomment-5088824573)).

**Longer, better-qualified sibling.** A stable BF16 **120-step** reference on the
8-of-256 variant (v143) reached **22.299% tail-30 MFU at 1.711% tail-30 drop**,
loss → 5.995. That is the strongest ECHO evidence, and it beats C2's 350-step
compliant point (20.708% @ 1.44%) on MFU at a comparable drop rate.

**This is the best measured EP64 point at a compliant drop rate.** The 24.153%
figure is also the least qualified of the headline numbers: a 20-step performance
screen, not a stability or drop-rate qualification, and the 2.765% is an exact
aggregate *assignment* drop, not a whole-token drop rate.

**State.** Branch-only. Commits `24ee86090` (+`073ae3b35`, `53cd5fd30`) on
`research/rav/7201-ep64-drop3` / `7201-ep64-muon-pad`.

**Complexity: the largest item in the ledger by a wide margin.** `24ee86090`
alone is +725/−39 in `ep_ragged_all_to_all.py` plus +79 of tests. The branch as a
whole is +11,598/−788 across `lib` and `experiments`: `ep_ragged_all_to_all.py`
reaches +3,175 lines, and it carries
`lib/levanter/src/levanter/kernels/hybridep.py` (+315) and a 680-line MNNVL
fabric-transport CUDA FFI. The related HybridEP probe commit `2eac7cc45` is 34
files / +4,749. Related transport A/B:
[#7670](https://github.com/marin-community/marin/issues/7670) (closed).

**Blockers.** No long-run drop qualification; the available profile predates the
B7 Muon-pad change. **A clean 120-step run of the best recipe — let alone with
padded Muon and `overlap_limit=4` — remains unrun. That is the single
highest-value missing measurement in the workstream.**

**Two sub-knobs are the real drop dials, and both are cliff-shaped.**

- `SCALE_A2A_CLONE_MAX_RECEIVER_EXPERTS` (production **10**): 16 → 10 takes
  20.99% → 21.23% (+0.24pp) at 1.88% mean / 2.60% max drop. **8 segments → 3.50%
  max drop, rejected.** **6 segments → `CUDA_ERROR_ILLEGAL_ADDRESS` on first
  execution** — the envelope math has no proven lower bound.
- `SCALE_A2A_CLONE_TOKEN_PADDING_EXPERTS` (production **2**): padding 1 gains only
  +0.049pp and takes drops to **3.49%**, rejected under the 3% gate.

**Why ECHO exists at all.** The alternative — buying the same drop reduction with
capacity factor — measured at roughly **3.4pp of MFU** (350-step sweep, cf ≈ 1.15).
ECHO buys it without that tax. That is the case for the complexity.

**Transport A/B at top-8, PGLE off, pipeline chunks = 1**
([#7670](https://github.com/marin-community/marin/issues/7670)):

| transport | MFU | tok/s | exposed comm | drop |
|---|--:|--:|--:|--:|
| NCCL static a2a | **19.98%** | 279K | 31.4% | 1.32% |
| ECHO-ragged (`SCALE_A2A_ECHO_RAGGED_TRANSPORT=1`) | 18.99% | 279K | 30.9% | **0.02%** |
| HybridEP (DeepEP FFI, `SCALE_A2A_HYBRID_EP=1`) | ~4–6% est | — | ~85% | — |

**ECHO-ragged buys near-zero drop for about 1pp of MFU** — a genuine option if the
drop bar tightens. HybridEP trained correctly at top-8 for the first time but is
not competitive and died at step ~2 on all seven attempts.

### C4 — Exact drop metric plus tracker logging

**Mechanism.** `SCALE_REPORT_DROPS=1` sums the per-layer capacity-overflow count
and divides by the global assignment count `B·S·topk·L`. The grug training loop
logs `train/loss` through callbacks and never logs the returned metrics dict, so
the count was computed and discarded; the fix is an explicit `tracker.log`.

**Validation.** Exact, verified two ways: an integer cross-check at the operating
point (1,438,043,460 dropped / 1,610,612,736 total = 0.8929, matching the logged
fraction) and shard-invariance against a numpy reference of the true global drops
(ratio 1.000 at both 2 and 4 expert shards).

**Why it is load-bearing, not telemetry hygiene.** The drop fractions of the two
prior FSDP figures used as comparison baselines (23.1% and 19.2%) are
**unmeasured**, so both EP-vs-FSDP comparisons are one-sided on fidelity. This is
named as the largest uncertainty on both results
([#7201 comment 5084895357](https://github.com/marin-community/marin/issues/7201#issuecomment-5084895357), caveats).
It is a one-line fix that would settle it permanently for one job's cost.

**State.** Branch-only. Metric `4fbc89152`; tracker-logging fix `2d4a87395`
(exists in two worktrees, unlanded).

**Ship this first.** It is the cheapest item in the ledger and it is what makes
every other number in it interpretable.

---

## Group D — Stack-wide levers, proven on the FSDP line, unmeasured under EP

Larry's read on this is explicit: *"Maybe if it incorporates all the optimizations
on the FSDP stack it already is (none of the FSDP stack optimizations are specific
to the MoE)? (FSDP is ~25 MFU)"*
([#7201 comment 5093296092](https://github.com/marin-community/marin/issues/7201#issuecomment-5093296092)).
That is the single largest untested hypothesis in this document.

### D1 — PGLE (profile-guided latency scheduling)

**Config.** `JAX_ENABLE_PGLE=true`, `JAX_PGLE_PROFILING_RUNS=5`,
`XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true`.

**Measured on the FSDP line, d6144 4-of-128, 1 rack, 64 GB200:**

| change | p50 MFU | tok/s |
|---|---|---|
| baseline (e128, top-4, 1 shared, batch 1024) | 23.17% | — |
| + 2 shared experts (split into 2 × hidden/2) | 23.46% | 261.9K |
| + Muon shape-grouping (shared-only) | 23.55% | — |
| + batch 1024 → 1152 | 23.70% | 263.0K |
| **+ PGLE** | **~24.8%** | **276.8K** |

**PGLE alone is +1.1pp and is config-only.** Enabling the latency-hiding scheduler
*without* PGLE gives nothing (23.5%) — the win is the measured latencies, not the
flag. Reproduced across two runs (24.99%, 24.77%)
([#7201 comment 5076593050](https://github.com/marin-community/marin/issues/7201#issuecomment-5076593050)).

At 2 racks (d6144 4-of-256, 128 GPUs, batch 2304) the same three tunings gave
+1.0pt (18.6% → 19.6%, 412K → 435K tok/s) — smaller, because 2 racks is
cross-rack InfiniBand comm-bound and PGLE can reschedule the weight-gather but
not shrink it
([#7201 comment 5076727526](https://github.com/marin-community/marin/issues/7201#issuecomment-5076727526)).

**Under EP the picture differs.** B6 measured manual PGLE + overlap-limit=4
together at only **+0.47pp** at the EP64 operating point, and the ECHO screen (C3)
ran with **PGLE off**. Whether D1's FSDP-scale gain transfers to EP is open.

### D2 — Split the shared expert into two half-width experts

+0.29pp (23.17% → 23.46%), params-neutral (2 × `hidden_dim/2`).
`SCALE_NUM_SHARED_EXPERTS=2`. Same source as D1.

### D3 — Muon shape-grouping for non-expert Newton–Schulz

Same-shape/same-sharding leaves are concatenated into a single NS call instead of
one per leaf. +0.09pp on top of D2 (23.46% → 23.55%).
`SCALE_MUON_GROUP_NONEXPERT=1 SCALE_MUON_GROUP_SHARED_ONLY=1`. Same source as D1.
Distinct from B7 (padding for stack-sharding) though they touch the same code.

### D4 — Host offload of MuonH optimizer state

**Mechanism.** `SCALE_OFFLOAD_OPT_STATE=1` parks the MuonH optimizer state
(~22 GB/GPU, dominated by expert momentum, idle through forward/backward) on
pinned host memory, freeing the HBM the memory-bound scheduler needs to overlap
the backward weight re-gathers. Viable on GB200 via the ~900 GB/s Grace↔Blackwell
NVLink-C2C link; it would lose on a PCIe host.

**Measured.** 253.0K tok/s @ 22.7% → 255.3K @ 23.1%, i.e. **+0.4pp**, d6144
4-of-128, 1 rack, batch 1024, chunk-2
([#7201 comment 5036748495](https://github.com/marin-community/marin/issues/7201#issuecomment-5036748495), update section).
That arm also added QB routing, XSA, attn-gate and gated-norm, so +0.4pp is the
net of the whole "full-feature + host offload" bundle rather than offload alone.

**Already used under EP** — the d6144 4-of-128 EP64 reference legs run with host
offload ([#7201 comment 5084895357](https://github.com/marin-community/marin/issues/7201#issuecomment-5084895357)).

### D5 — Chunked expert FSDP all-gather (`SCALE_MOE_EXPERT_CHUNKS`)

**Mechanism.** Dispatch runs once, then each expert chunk all-gathers only its
slice of the FSDP-sharded expert weights and runs the QuACK grouped GEMM over its
token segment, so chunk *k*+1's gather overlaps chunk *k*'s compute.

**Measured.** 1 rack, d6144 4-of-128: 245.9K @ 21.8% (unchunked) → 253.0K @ 22.7%
(chunk-2), **+0.9pp**
([#7201 comment 5036748495](https://github.com/marin-community/marin/issues/7201#issuecomment-5036748495)).
2 racks, d6144 4-of-256, batch 2304: chunk-2 402K @ 18.1% → chunk-4 412K @ 18.6%,
**+0.5pt**
([#7201 comment 5054192424](https://github.com/marin-community/marin/issues/7201#issuecomment-5054192424)).

**Does not apply under EP.** Chunking overlaps an FSDP expert-weight all-gather
that EP does not perform. It is the one FSDP-overlap lever that landed a win at
scale, and it is also the *only* chunking-family win on record — token-chunk
pipelining of dispatch/FFN under EP measured −1.96pp (E4).

### D6 — Replica-local embedding gather

**Mechanism.** `Pembed_vocab` goes from `P("model", Pbatch[0])` to `P(None, None)`
— the embedding table is fully replicated so the token lookup is a replica-local
`shard_map` gather rather than an all-to-all. The old layout sharded the table's
hidden dim over `("replica_dcn", "data")`, which forced the gather to assemble
each token's vector across all devices — an all-to-all spanning racks whose NCCL
first-call rendezvous wedges at 8+ racks. The replicated table's gradient is a
normal DDP all-reduce.

**Evidence.** Fixed the 8-rack wedge. PoC
[`bdf61d7ed`](https://github.com/marin-community/marin/commit/bdf61d7ed6c01180964d57a3f3156f54d3cd4a88);
[#7012 comment 5007009232](https://github.com/marin-community/marin/issues/7012#issuecomment-5007009232).
Extraction marker [#7493](https://github.com/marin-community/marin/pull/7493) (draft).

**Correctness/reliability, not throughput.** Required for any run beyond a few
racks. Two lines in `sharding.py` plus the `_embedding_gather` `shard_map`.

### D7 — Slim Sonic CuTe residuals and MoE-aware rematerialization

**Mechanism.** Slim custom-VJP residuals combined with an `all_but_moe`
rematerialization split, reducing recomputation.

**Evidence.** PoC
[`01b8e7c92`](https://github.com/marin-community/marin/commit/01b8e7c92f22faf486bf66a4d3e4b6d1aa7f0236);
[#7012 comment 4984144891](https://github.com/marin-community/marin/issues/7012#issuecomment-4984144891).
Extraction marker [#7489](https://github.com/marin-community/marin/pull/7489)
(draft, branch `mcwitt/sonic-cute-wgrad`, stacked on #7488). Measured at the
d2560 row-13 scale, not at the d5120/d6144 EP64 operating point — naive remat
splits OOM there because `custom_vjp` × `scan` pins live values.

**Explicitly out of scope in the extraction:** the weight-gradient and SM-carveout
experiments (`SONIC_CUTE_SM_CARVEOUT`) on that branch.

---

## Group G — Precision (MXFP8 / FP8): conditional, and the sign depends on EP degree

This is the one family where the record contains an apparent contradiction that
resolves cleanly once you read it correctly. **Nothing here is merged.** `main`
carries only `Fp8DotGeneralOp` (dense per-tensor delayed scaling, `lib/haliax`,
from #6660).

### G1 — The three end-to-end MXFP8 measurements

| config | topology | result | strength |
|---|---|---|---|
| d5120 / L48 / E128 top-5 / B1024, 50 steps | 64 GB200, ring **EP8**, NS on | **1.308×** (392,287 vs 299,894 tok/s) | 1 run per arm, self-labelled exploratory |
| d2560 / L26 / E128 top-4 / B512, **31,474 steps / 66.006B tokens** | 32 GB200, ring **EP8**, NS on | **+7.220%** (845,920 vs 788,954 tok/s) | strongest evidence in the workstream |
| d6144 / L48 / E128 top-4, 30 steps | 64 GB200, **EP1** (FSDP only) | **0.749×** — MXFP8 **25.15% slower** | 1 run per arm |

Sources: [#7282 c5017713906](https://github.com/marin-community/marin/issues/7282#issuecomment-5017713906),
[#7271 c5036746436](https://github.com/marin-community/marin/issues/7271#issuecomment-5036746436),
[#7282 c5037422197](https://github.com/marin-community/marin/issues/7282#issuecomment-5037422197)
(cross-posted as [#7201 c5037417823](https://github.com/marin-community/marin/issues/7201#issuecomment-5037417823)).

**These are not reconciled anywhere in the record.** The two EP8 arms differ
1.308× vs 1.072× at different model sizes; the plausible but unmeasured
explanation is that the expert-GEMM share of the step is much larger at
d5120/L48 than d2560/L26. The EP1 arm is a different regime entirely — 128 local
experts per device, a grouped-GEMM shape the kernels were never tuned for. **The
MFU-versus-EP-degree curve for MXFP8 expert GEMMs has never been measured.**

**Retracted, do not cite:** the `1.278×`/`1.251×`/`+34.8%` headlines. Their arms
differed on EP topology (EP4 vs EP8) *and* had Newton–Schulz disabled. NS alone
costs 7.2% on the mxfp8 stack, so "our bf16 beats the baseline" was purely a
NO_NS artifact.

### G2 — The measured quality cost (#7271) — the one rigorous result

Preregistered gate, 31,474 steps and 66,005,762,048 tokens per arm, same seed,
data order, schedule, optimizer and 32-GB200 topology, checkpoint-audited:

| metric | BF16 | MXFP8 | Δ |
|---|---:|---:|---:|
| Aggregate eval loss | 2.314181 | 2.315482 | **+0.056%** |
| Paloma macro | 2.611326 | 2.614211 | **+0.111%** |
| Uncheatable macro | 2.052934 | 2.057221 | **+0.209%** |
| Mean tok/s | 788,954 | 845,920 | **+7.220%** |

The signal is persistent, not noise: MXFP8 aggregate eval is worse at **32 of 32**
paired evaluations, and all three metrics favour BF16 for the final 26 consecutive
gates. MXFP8 finished before BF16 reached step 30,000, but **never reaches BF16's
terminal held-out targets within the fixed schedule** — so strict
time-to-BF16-final-loss is censored, not improved. The preregistered hypothesis
("quality-neutral") is answered in the negative. This is a speed/quality trade,
not a free lunch.

**Scope limit.** Validates the *hybrid* recipe (grouped MXFP8 + dense per-tensor
FP8) at d2560/EP8/32 GPUs only. Quality at d5120/d6144 or EP64 is unmeasured.
Uniform (dense + grouped) MXFP8 quality has never been run.

**Unused reserve levers**, neither implemented: stochastic rounding on cotangents
(unbiased gradient quantization) and Hadamard rotation before quantization
(QuaRot/SpinQuant), most valuable for the per-tensor dense legs
([#7271 c5017497844](https://github.com/marin-community/marin/issues/7271#issuecomment-5017497844)).

### G3 — Fused MXFP8 grouped expert kernels (`MxFp8MoeMlpOp`)

**Mechanism.** NVIDIA's MIT-licensed cudnn-frontend ≥1.21 CuTeDSL kernels
(`grouped_gemm_swiglu_quant`, `dswiglu_quant`, quantizing-epilogue GEMM, wgrad),
vendored behind the `cutlass_call` adapter. **Quantization moves into the GEMM
epilogue**, which is what removes the producer tax that killed the earlier
CuTeDSL grouped path (honest layer-quad 0.58× — a net loss — once XLA quantize
producers were counted). Wrapped as one stateless `custom_vjp` implementing the
whole expert MLP: no amax history, no `OverwriteWithGradient`, no train-step state.

**Measured.** Layer pipeline 8.777 ms vs 12.220 ms bf16 = **1.39×** on honest
full-layer accounting. MoE-MLP block per step 1.1 s → 0.49 s. **Step temp arena is
37% smaller than bf16's** (500.82 vs 792.47 GiB at d5120/L48/B128 ring-EP8) —
directly relevant because EP capacity walls are the binding constraint at rack
scale.
([#7282 c4998990686](https://github.com/marin-community/marin/issues/7282#issuecomment-4998990686),
[c5010080429](https://github.com/marin-community/marin/issues/7282#issuecomment-5010080429))

**Complexity: very large.** Vendored kernels are **20,587 lines across 19 files**.
The non-vendored delta is +9,090/−187 across 55 files on
`research/mcwitt/7282-mxfp8-blackwell` @ `0a37854`; the core is
`experiments/grug/moe/mxfp8.py` (468 lines), `model.py` (+431/−47), the
`MoeExpertMlpOp` protocol in `_moe/common.py` (+48). **No PR exists.** #7492 was a
draft extraction marker, closed 2026-07-24 without merging the model path.

**Hard dependency chain.** sm100 only · the cutlass-dsl lockfile fix (A5) ·
scan-over-stacked-blocks (`e8e105d4e3`, mandatory at production shape — without it
the step-0 temp arena is 851.61 GiB) · the EP-aware NS fix (A2) · the `w2` finite
guard (G4).

### G4 — The hybrid `w_down` NaN and its guard

Grouped MXFP8 **combined with** dense per-tensor FP8 produces a finite first
forward and a **non-finite `w_down` weight gradient on the first backward**.
Grouped-only and dense-only controls are both finite; the isolated kernel with
synthetic cotangents is finite; `optimization_barrier` does not fix it; adding a
diagnostic *consumer* of `dw2` does. **Root cause never identified** — it is a
liveness/aliasing/scheduling sensitivity in the hybrid compiled graph
([#7480](https://github.com/marin-community/marin/issues/7480), closed;
[#7271 c5018066223](https://github.com/marin-community/marin/issues/7271#issuecomment-5018066223)).

**Mitigation shipped, not a fix** (`f8be94f87`): consume the fused `dw2` with a
finite reduction and conditionally recompute only an invalid `w2` gradient from
saved BF16 preactivations and cotangents. This is the code that ran the 66B-token
quality pair. It has not been re-validated after any kernel or XLA version change.

**Silent-failure hazard worth carrying:** Iris reported the NaN job as
*succeeded* (the training loop breaks rather than raises on NaN) and it wrote a
poisoned step-2 checkpoint. Neither status is a pass.

### G5 — FP8 forward-dispatch wire (#7665) — the only lever whose gain grows with EP

**Mechanism.** Quantize the dispatch payload to MXFP8 *before* the EP collective
and hand the bytes straight to the MXFP8 grouped GEMM, instead of quantizing after
arrival. Halves dispatch bytes with **no added quantization compute** — the pass is
relocated, not added. Packed `[T, 33H/32]` uint8 buffer (payload + e8m0 scales) in
one collective, exactly 33/64 of bf16 bytes.

**Measured** (GB200, ring EP, XLA producer, 16,384 tokens/device, cf 1.25, 3
barriered draws each, **single MoE layer in isolation**):

| config | EP | GPUs | fwd | fwd+bwd | relfrob vs control |
|---|--:|--:|--:|--:|--:|
| d6144 · 4-of-256 · i3072 | 4 | 4 | 1.044 | 1.005 | 1.5e-4 |
| d5120 · 4-of-128 · i2560 | 4 | 4 | 1.071 | 1.019 | 2.1e-4 |
| d5120 · 4-of-128 · i2560 | 16 | 16 | 1.210 | 1.101 | 1.1e-4 |
| d5120 · 4-of-128 · i2560 | **64** | **64** | **1.286** | **1.144** | 6.5e-5 |

`dw13`/`dw2` relfrob is **exactly 0.0** at every configuration
([#7665 c5093816443](https://github.com/marin-community/marin/issues/7665#issuecomment-5093816443)).

**This reconciles G1 against E5.** The sign of quantizing the wire is decided by
whether a quantized consumer exists downstream: #7279 disabled the GEMMs and
measured −2.02pp; #7282 disabled the wire. `GrugFp8Config` *raises* if
`recipe="mxfp8"` is combined with `wire=True`, so the productive combination was
unrunnable by construction until #7665.

**Complexity: +1,067/−8 across 11 files** on top of the 7282 branch
(`_moe/mxfp8_wire.py` +144, `_moe/ep_ring.py` +47,
`experiments/grug/moe/mxfp8.py` +127). Branch
`research/mcwitt/7279-fp8-dispatch-wire` @ `224a0081`. No PR.

**Dependency that currently blocks it.** It requires G3, and **MXFP8 does not
exist on the EP64 fixed-a2a stack at all** — expert GEMMs there are bf16
`jnp.einsum`. A production home requires porting G3 to that stack first.

**Everything above is one layer in isolation** — no scan, no remat, no optimizer,
no competing collectives. The in-step measurement has not been run, and #7279's
own record is explicit that isolated wins on this stack have repeatedly failed to
survive end-to-end.

**Two silent-corruption traps found and pinned as tests.** The quantized payload
cannot cross an autodiff boundary in either carrier dtype: `uint8` has a float0
tangent and **silently zeroes the cotangent**; `float8_e4m3fn` **downcasts an
incoming bf16 cotangent to unscaled e4m3** and flushes ordinary 1e-6 gradients to
exactly zero. Hence the fused `custom_vjp` with bf16 on both differentiable faces.
Neither failure mode raises. Separately, the wgrad operand must be rebuilt through
bf16, not f32 — the f32 rebuild materializes a 1.7 GB `[81920,5120]` tensor and
consumed the entire forward gain.

### G6 — Uniform (dense + grouped) MXFP8 — not available

A zero-producer oracle passes the throughput gate (0.9900× per-tensor time), but
Marin's real CuTe producer is 1.2125× and projects to 98.56–98.66% of hybrid
full-step throughput, **missing the 99% gate in both replications**. TE 2.16's
fused-projection control projects to 99.39–99.48%, so it is physically possible —
but uniform MXFP8 at EP2 and EP4 **dies with a deterministic XLA GPU backend
abort before execution** (`AllReduceThunk::CheckImplementable(): reduction_kind.has_value()`,
exit 134, uncatchable from Python). Decision: keep dense on delayed per-tensor
FP8. Quality of uniform MXFP8 is completely unmeasured
([#7282 c5028942402](https://github.com/marin-community/marin/issues/7282#issuecomment-5028942402)).

### G7 — Related open FP8 defects

- **CuTe activation-quantizer producer breaks 16-node executable load,
  deterministically** (3/3 vs 2/2). All 16 hosts fail `jit_train_step` load with
  `Failed to load in-memory CUBIN` / `CUDA_ERROR_INVALID_VALUE`, across different
  node sets and a fresh compilation cache; the identical run with the XLA producer
  trains clean. **Mechanism open.** Mitigated by defaulting `mxfp8_producer` to XLA
  (`693124f9b`), which strands the CuTe producer's 2.5× advantage.
- [#7659](https://github.com/marin-community/marin/issues/7659) — gradient
  accumulation drops the FP8 amax history to the last microbatch.
- [#6880](https://github.com/marin-community/marin/pull/6880) (H100 per-tensor
  grouped FP8, 1.38×/1.23× fwd+bwd) and
  [#7079](https://github.com/marin-community/marin/pull/7079) (end-to-end wiring,
  1.055× throughput with a +0.0040 matched-step loss gap over 24k steps) are both
  open and both **H100-relevant only** — sm100 has no working per-tensor grouped
  FP8 path, so MXFP8 supersedes them on B200.

---

## Group E — Refuted, null, or sealed. Do not re-run these.

| # | Direction | Result | Source |
|---|---|---|---|
| E1 | **MXFP8 at EP1, d6144 production config** | **0.749× BF16 (−25.15%)**, same commit, same config, only `SCALE_FP8=mxfp8` changed, FP8 wire disabled. Median rank-0 over steps 2–29. | [#7201 c5037417823](https://github.com/marin-community/marin/issues/7201#issuecomment-5037417823) |
| E1b | MXFP8 quality gate | Preregistered claim "quality-neutral" answered **in the negative**: +7.22% throughput but +0.056% aggregate eval, +0.110% Paloma, +0.209% uncheatable loss, with aggregate eval favouring BF16 at all 32 paired evaluations. 31,474 steps / 66.006B tokens per arm. | [#7271](https://github.com/marin-community/marin/issues/7271) |
| E1c | Earlier "+34.8% / 1.251× MXFP8" headlines | **Retracted.** Confounded on EP topology (EP4 vs EP8) and on Newton–Schulz being disabled in the control. | [#7201 c5015174520](https://github.com/marin-community/marin/issues/7201#issuecomment-5015174520), [c5017713989](https://github.com/marin-community/marin/issues/7201#issuecomment-5017713989) |
| E2 | **NVFP4** | Ruled out on risk grounds; [#7403](https://github.com/marin-community/marin/issues/7403) closed. | week of 2026-07-20 summary |
| E3 | Rotation `ppermute` decomposition of the fixed a2a | **−9.46pp** | [#7279 c5080435482](https://github.com/marin-community/marin/issues/7279#issuecomment-5080435482) §3 |
| E4 | Token-chunk pipelining of dispatch/FFN | **−1.96pp** | same |
| E5 | FP8 permutation-leg wire (QDQ decomposition) | **−2.02pp** (also reported as −1.6 to −2.3pp). Recovers 936 ms/step of exposure exactly as the byte thesis predicts, but quantization compute costs more than the bytes save. Delayed scaling, pre-registered to cut added compute 2239 → ~127 ms/step against a 920 ms/step break-even, measured **2182 ms/step** — both falsification clauses fired at 17× the threshold. | same §3; [c5084892846](https://github.com/marin-community/marin/issues/7279#issuecomment-5084892846) |
| E6 | Weight-prefetch overlap | Null, scheduler-gated (LHS/auto-PGLE inert on this workload) | same §3 |
| E7 | TransformerEngine NCCL_EP | NVIDIA's recommended full fused MoE block **ties** Marin's own seam: 16.94% vs 17.15%, both ~1.1–1.3pp behind the incumbent `a2a_cute`. TE-at-tip: #3231's collective-stream pin crashes 64-GPU first execution; shimmed out, the tip wheel is functionally the old wheel (~17% vs 18.05%). Every remaining knob — scoped command-buffer capture, the collective-overlap flag, an SM-budget sweep — was a wash or a loss. | [#7331 c5073227834](https://github.com/marin-community/marin/issues/7331#issuecomment-5073227834); [#7279 c5080435482](https://github.com/marin-community/marin/issues/7279#issuecomment-5080435482) §3 |
| E8 | fa4-lse as a primal output | **+0.18pp**, below the 0.5pp bar. Control 20.465% (3 draws) vs 20.648% (2 draws). On-device variant dead on memory (+32.7 GiB saved activations do not fit); host offload over Grace C2C is the only viable form, saving ~70 ms of a 13.2 s step. The d2560-derived ~1pp estimate does not transfer to d5120 EP64 where attention is a small slice. | same §3 |
| E9 | `ring_cute` at e256 / EP64 | **DNF** — OOM at 141.79 GiB in `jit_train_step`. Its EP4/EP8 backend-ladder wins (20.83% at 64 GPUs) do not transfer; fitting it would be a memory-engineering project of its own. | same §4 |
| E10 | Ragged a2a with the one-shot kernel off, at EP64 | **12.38% mean** — roughly half of fixed+adjoint — and still drops 43% under the same QB-off collapse, so it is not a fidelity refuge either. | same §4 |
| E11 | **Latent MoE** at d6144 4-of-128 EP64 | Wire mechanism works as predicted (expert-a2a exposed time −54%) and **still loses**: matched-work −0.23pp, param-preserving −1.72pp. The projections add 7.52% of analytic work and landed as +10.8% compute-stream busy. Should get *worse* at 12 racks under the current mesh. | [#7279 c5095217108](https://github.com/marin-community/marin/issues/7279#issuecomment-5095217108) |
| E12 | QB controller gain g=2 | Categorical negative: 0.67–0.72 drops for all 350 steps (overshoot limit cycle), loss +0.091 worse. | [#7279 c5080435482](https://github.com/marin-community/marin/issues/7279#issuecomment-5080435482) §2 |
| E13 | Latency-hiding scheduler without PGLE | Null (23.5% vs a 23.70% baseline) | [#7201 c5076593050](https://github.com/marin-community/marin/issues/7201#issuecomment-5076593050) |
| E14 | MLA | Neutral at matched head dims (+0.8% at qk128·1×), negative otherwise (−4.4% qk192·1×, −16.0% qk128·2×, −22.9% qk192·2×, −87.0% at head dim 256). The full-width 72-head + full-causal + learned-rel-pos variant is ~15.6% true MFU, ~120 days for 20T. Quality gain measured at only ~8% for 2× heads. | [#7201 c5011863858](https://github.com/marin-community/marin/issues/7201#issuecomment-5011863858), [c5060285897](https://github.com/marin-community/marin/issues/7201#issuecomment-5060285897) |
| E15 | Source-push, NVSHMEM/CuTe transport, SM comm/compute partitioning, auto-PGLE/LHS | Sealed on this stack. | [#7333](https://github.com/marin-community/marin/issues/7333), [#7114](https://github.com/marin-community/marin/issues/7114), [#7012 c4997270478](https://github.com/marin-community/marin/issues/7012#issuecomment-4997270478) |
| E16 | JAX-Toolbox container | 25.60% against 25.83% / 25.71% controls — a 0.505% time-adjusted regression. Passes a 2% adoption gate but gives no performance reason to switch. | [#7519](https://github.com/marin-community/marin/issues/7519) / [#7524](https://github.com/marin-community/marin/issues/7524) |
| E17 | Fine expert granularity (8-of-256, i1280) at fixed active params | 19.17% vs a 20.36% 4-of-128 baseline at 2 racks — finer granularity costs throughput on the FSDP line. Note this reverses under EP: 8-of-256 gives spill more headroom (C2). | [#7201 c5010023475](https://github.com/marin-community/marin/issues/7201#issuecomment-5010023475) |
| E18 | **SM comm/compute partitioning** (`max_num_sms`, NCCL CTA pin) | **Falsified three separate times**: EP1 −0.20/−1.00pp; ring EP8 −1.22/−3.19pp; TE `max_num_sms` 32→16 monotone negative. | [#7012 c4985697941](https://github.com/marin-community/marin/issues/7012#issuecomment-4985697941), [c4987885345](https://github.com/marin-community/marin/issues/7012#issuecomment-4987885345), [#7331 c5076118699](https://github.com/marin-community/marin/issues/7331#issuecomment-5076118699) |
| E19 | Auto-PGLE (as opposed to the manual FDO flow) | +0.06pp single-node and **crashes multi-host** — per-host recompilation desynchronizes processes. The manual dump-profile → shared-FDO → replay flow is required. | [#7012 c4984144891](https://github.com/marin-community/marin/issues/7012#issuecomment-4984144891) |
| E20 | Sonic exact clone-weight-gradient adjoint | Microbench 3.24×/4.63× on W2/W13, rack A/B **−0.167pp**. Another isolated win that did not survive end to end. | `7201-ep64-mfu.md` logbook, 2026-07-27 05:27 |
| E21 | Two-chunk dispatch prefetch regroup | **−0.228pp** | same logbook, 2026-07-27 07:11 |
| E22 | Sonic `sonic_cute` varlen-k wgrad shim | **+0.06–0.08pp only** — the weight-gradient GEMM is not a bottleneck. Tile autotune (256,256) is ×1.08–1.28 in isolation but **+0.14pp** end to end. | [#7012 c4919640071](https://github.com/marin-community/marin/issues/7012#issuecomment-4919640071) |
| E23 | Native dense MXFP8 (`jax.nn.scaled_dot_general`) | **0.82–1.00× fwd, 0.64–0.81× fwd+bwd vs bf16** — slower than bf16. GEMM-only is 1.33×, but XLA's block-quantize kernel costs 0.26–0.36 ms per operand and the cotangent is quantized twice in backward. Shelved 2026-07-16. | [#7282 c4997628535](https://github.com/marin-community/marin/issues/7282#issuecomment-4997628535) |

**One thing E7 did buy, and it is worth keeping.** TE's MoE-layer chunking plus
chunk remat (`C=8192`) removed the no-drop capacity wall: **EP-side memory becomes
O(chunk), independent of batch and sequence length.** That let b1024 train at 18.0%
against `a2a_cute`'s 19.1%. NCCL_EP itself has **no drop path** — receive overflow
is an out-of-bounds write that poisons the CUDA context, so capacity must cover the
no-drop worst case (~21.5 GiB/rank unchunked). The wire is bf16-only; `moe.py` has
no quantizer support, so NCCL_EP cannot carry Group G. TE main tip `ea41e08` is a
regression (deterministic 64-GPU death at data8×expert8, `ncclCommSplit … remote
process exited`, 2/2 across 4 arms) — pin `68493d2`. Verdict: `a2a_cute` stays
incumbent for bf16; NCCL_EP is preserved as a derisked fallback.

**Two method traps worth carrying forward:**

- **A side effect inside a rematerialized scan body defeats remat.** A bare
  `jax.debug.print` touching no tensor costs 1.41× compiled temp memory, while two
  real reductions cost 0.97×. At 48 layers that is a 300+ GiB allocation request.
  Instrument in a small probe config or return values through the metrics path.
- **`crash_on_nan` does not catch FP8 underflow, and a stable amax is not evidence
  of underflow safety.** Backward cotangents measured 10.9% of non-zero inputs
  quantizing to exactly zero, drifting to 16.4% within 12 steps, while amax looked
  stable. A scale that is far too large drives every value to exactly zero;
  zeroed tensors are not NaN, not Inf, and not out of range, so every finiteness
  guard stays silent. Demonstrated with an eight-orders-of-magnitude scale error.
  ([#7279 c5084892846](https://github.com/marin-community/marin/issues/7279#issuecomment-5084892846))

---

## Group F — Open leads, not yet measured

| # | Lead | Why it is on the list |
|---|---|---|
| F1 | **Three of twelve all-to-all ops run on the compute stream at 0.0% overlap** | 1,266 ms per 3 steps = **422 ms of every 15.3 s step**, 31% of all exposed collective time, consistent across all four GPUs. Larger than latent MoE's entire best case, needs no architecture change, and is unexplained. ([#7279 c5095217108](https://github.com/marin-community/marin/issues/7279#issuecomment-5095217108)) |
| F2 | Sender-local router balancing | The only unprobed direction aimed at the *hypothesized cause* of the ~6% QB-on drop residual. Kernel-level. Global-bias gain tuning is falsified (E12). |
| F3 | Leg-batching (B5) composed with QB-on | Both measured alone, never stacked. |
| F4 | **FSDP-line levers on the EP stack** | D1–D4 are unmeasured under EP; Larry asked for exactly this. PGLE was worth +1.1pp on FSDP but only +0.47pp (with overlap-limit) under EP, and the ECHO screen ran PGLE-off. |
| F5 | **Multi-rack EP** | EP64 has **no multi-rack measurement** behind its ~65–75-day 20T projections. Under the current mesh (`replica_dcn, data, expert, model`, `expert` innermost) holding EP64 while adding racks keeps the MoE all-to-all inside a rack and grows data-axis traffic instead. |
| F6 | FP8 forward-dispatch wire feeding MXFP8 expert GEMMs | [#7665](https://github.com/marin-community/marin/issues/7665) — reconciles E1 against E5: each thread disabled exactly the half the other tested. Quantizing is pure overhead when nothing downstream consumes the FP8, and may pay when an MXFP8 grouped GEMM does. Preregistered; success criterion is a positive layer-level A/B inside the real step with remat on. |
| F7 | DeepSeek-style integral or damped (g<1) QB | Cheaper than F2, unprobed. |
| F8 | Fused FP8 epilogues | Listed as a follow-up in the 25% ledger; unstarted. |

---

## Architecture-selection inputs that fall out of the perf work

- **4-of-256 at EP64 does not fit one rack.** Measured on all 16 tasks: 92.02 GiB
  resident plus a single 106.63 GiB temp arena against 184.3 GiB physical, so no
  memory fraction closes it. The resident set decomposes exactly into expert
  parameters, MuonH momentum and the replicated embedding trio — nothing is
  mis-sharded. The 707B 4-of-256 candidate already assumes two racks plus offload.
  ([#7201 c5084895357](https://github.com/marin-community/marin/issues/7201#issuecomment-5084895357))
- **Drop recovery is bounded by routing granularity** — see C2's constraint 1.
  Neither top-4 candidate reaches a 3% bar by spill alone at cf1.0; they need
  capacity headroom on top, at roughly −0.58pp per +0.05 of capacity factor.
- **8k context is nearly free; 65k costs −35% tok/s.** At constant ~4.19M
  tokens/step on the final d6144 stack: 4k → 246K tok/s, 8k → 241.6K (−1.8%),
  65k → 160.7K (−35%, true MFU 17.0%). Sliding window 512 means the 40 local
  layers do not grow — only the 8 global layers pay.
  ([#7201 c5097482159](https://github.com/marin-community/marin/issues/7201#issuecomment-5097482159))
- **Reported MFU is window-blind and the error grows with sequence length.**
  `lm_flops_per_token` counts full O(seq²) attention on all 48 layers although 40
  are windowed to 512, overcounting FLOPs by ×1.08 at 4k, ×1.17 at 8k and ×2.14 at
  65k. Reported MFU therefore *rises* (22 → 24 → 36) while true efficiency *falls*
  (20.6 → 20.3 → 17.0). Trust tok/s; treat MFU as inflated whenever seq ≠ 4096.
  The fix is a window-aware attention term (`min(seq, window)` on local layers,
  full on global). Same source.
