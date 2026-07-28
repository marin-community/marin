# Commit sequence

The plan for turning the branch-only work in [`evidence.md`](evidence.md) into a
reviewable series against `origin/main` @ `1c631c4c0`.

Each entry is one PR-sized commit. "From" gives the source commit; where the source
is entangled the entry says what to extract rather than cherry-pick. Sizes are the
functional diff against `main`, excluding vendored code and research scaffolding.

**None of this has been executed.** The commits below are a plan, not a record.

## Branch topology

```
main@696eb370d
  └─ origin/grug/embedding-gather-shard-map (23 commits)      ← THE SHARED BASE
       ├─ +13 ─► origin/chunk-moe-fsdp        (c241f31d7)
       │    └─ +21 ─► origin/b200-minimal     (8823246ef; snapshot cff962d730)
       │          └─ ...  ─► origin/b200-300B-tune (fd3e9bc5b)  ← FSDP line
       ├─ +5  ─► origin/b200_mla              (de1cd14db)
       ├─ +1  ─► origin/rav-grug-moe-ep64     (54bbe3d23)
       └─ +1  ─► origin/codex/per-layer-kv-heads-static-fa4 (46fbff3173)

main@51964171c
  └─ replay of the b200-minimal stack + 1 ─► origin/rav/ep-2 (fe21ea495)
       └─ (local) agent/ep25-d1-adjoint, ep25-d4/d5/d6           ← EP line
main@a51da194d
  └─ 25 commits ─► origin/mcwitt/moe-standalone-ep (75c517148)   ← independent
```

`rav/ep-2` is a **rewritten replay**, so its SHAs do not match `b200-minimal`'s even
where content is identical.

**Push the local-only branches before doing anything else.** All four
`agent/ep25-*` branches exist only in this clone and are 92–147 commits ahead of
`origin/main`. If this clone is lost, so are the custom adjoint (+3.43pp), same-step
spill — the only ≤3% mechanism with a true tail-100 qualification — and the
drop-metric fix that makes every fidelity claim in this project checkable. Their
logbooks live at `AGENT_LOG.md` in the repo root rather than under `.agents/`, so a
naive `.agents/`-scoped copy would miss them.

## Prerequisite decisions

Resolve these before writing commit 1. They are not implementation details; each
changes what the series contains.

1. **Which MuonH 4D Newton–Schulz fix.** `75c517148` (skip the merge; also fixes
   `_newtonschulz_padded_stack_sharded` with a two-hop reshard; CPU-validated
   bit-exact) or the rav transpose (`54bbe3d23` grugmuon hunk; carries the 17.8%
   64-GPU measurement). They conflict textually. Recommendation: take `75c517148`
   for the design and re-run the 64-GPU probe against it.
2. **Whether Receiver-ECHO (#13 in the brief) is in scope.** It posts the best
   compliant EP64 number and it is an order of magnitude more code than everything
   else combined. Recommendation: out of scope here; separate project.
3. **Whether MXFP8 is in scope.** Recommendation: no — see the brief, Tier 5.
4. **Which `sonic_cute.py` lineage is canonical.** Chunking and slim residuals both
   edit `_expert_mlp` and have never been combined.

## The series

### Phase A — instrumentation and configuration. No behaviour change to the model.

| # | Commit | From | Size |
|--:|---|---|--:|
| A1 | Log the MoE capacity-overflow metric through the tracker | `2d4a87395`, `4fbc89152` | ~+30 |
| A2 | Expose `SCALE_CAPACITY_FACTOR` on the EP launcher and reconcile the default | `54bbe3d23` (extract) | ~+8 |
| A3 | Document the required XLA flag set | new | docs only |

A2 carries two hazards. `SCALE_CAPACITY_FACTOR` was implemented **twice
independently under the same environment-variable name** (`595958b83` and
`3e149490f`) — reconcile, do not double-apply. And the default is inconsistent
between layers: 1.0 in `experiments/grug/moe/model.py:51` against **1.25** in
`lib/levanter/src/levanter/grug/_moe/common.py:19`, with receiver envelope factors
at 1.125. Pick one canonical value; a silent 1.25 would misprice every drop
measurement taken against it.

A3 must document that
`--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false` is
**mandatory on JAX 0.11** — without it a 64-process run initialises, compiles, and
then segfaults in NCCL `ncclDevCommCreate` before step 0. It should also record
that `overlap_limit` is **not monotone** (2 measured worse than 1; use 4), that
`xla_gpu_enable_custom_fusions` and `xla_gpu_enable_address_computation_fusion`
kill the process before distributed init on this build, and that the JAX 0.11
baseline sits **1.217pp below** the 0.10.1-era baseline, so pre-0.11 numbers cannot
be borrowed as controls.

A1 first, always. It is one line of the fix plus the metric, and it is what makes
every subsequent measurement interpretable — including settling the unmeasured drop
rates of the FSDP figures the EP work is being compared against.

A3 is documentation because the flags are per-job environment, not repo config;
auto-PGLE must not be enabled (it crashes multi-host).

### Phase B — shared substrate. Needed by both the FSDP and EP lines.

| # | Commit | From | Size |
|--:|---|---|--:|
| B1 | Add the `sonic_cute` QuACK SM100 grouped-GEMM backend and `SCALE_MOE_IMPL`/`SCALE_ATTN_IMPL` | `5cf76b64a` | +309 |
| B2 | Add the `quack-kernels[cu13]` dependency and cutlass/quack mypy ignores | `538381606` (extract) | ~+10 |
| B3 | Replica-local embedding gather | `bdf61d7ed` | +32 / −6 |
| B4 | Precompute FA4 per-layer segment bounds outside the scan | `a33e16ced` | +156 / −30 |

B1 is purely additive and lazily imported, so CPU and H100 paths are unaffected.
**Drop** `538381606`'s `nvidia-cutlass-dsl>=4.6.0,<4.7` hunk — `main` pins `==4.6.0`
via #7587. **Drop** the temp smoke script (`458f647b5`, `d7decc466`) and the
`device_flops` commits (`c81a29428`, `7f504d9cb`) — `main` already carries an
equivalent `"b200"` entry with identical numbers under a different key.

Check whether #7587 already applied `5833e329e`'s
`make_fragment` → `make_rmem_tensor` change in `_fa4_cute_segmented_bwd.py`; if so,
skip it.

### Phase C — EP enablement. Nothing here is measurable alone; together they make EP64 run.

| # | Commit | From | Size |
|--:|---|---|--:|
| C1 | Route 4D scanned expert stacks through distributed Newton–Schulz | `b0c7a1b56` (strip the `SCALE_OPTIMIZER` launcher hunk) | +222 / −20 |
| C2 | Preserve expert sharding through the MuonH 4D Newton–Schulz merge | decision 1 | +54 or +33 |
| C3 | Shard non-expert weights and optimizer state over `("data","expert")` | `54bbe3d23` (extract) | +23 / +8 |
| C4 | Force the expert-axis batch spec at the EP dispatch boundary | `54bbe3d23` (extract) | **+11** |

C3 resolves the `Plm_head` conflict in favour of `P(Pfsdp, "model")`; that spec is
identical to the base branch's `P("data","model")` at EP1, so it is a strict
generalisation. Split `54bbe3d23` into C3, C4 and A2 — the same commit also carries
`moe_latent_dim`, latent dispatch and a2a remat markers, which are separate features
and should be dropped from this series (latent MoE measured negative — evidence E11).

C2 closes [#7512](https://github.com/marin-community/marin/issues/7512), C3 closes
[#7513](https://github.com/marin-community/marin/issues/7513), A2 closes
[#7514](https://github.com/marin-community/marin/issues/7514).

### Phase D — the EP64 throughput core. This is where the measured wins are.

| # | Commit | From | Size | Measured |
|--:|---|---|--:|---|
| D1 | Fixed-capacity `lax.all_to_all` dispatch and combine, with `SCALE_A2A_CHUNKS` | `fe21ea495` (extract the `ep_ragged_all_to_all.py` +164 and its two test files only) | +164, +117 tests | ~13% → 17.8% with C1–C4 |
| D2 | Build the dispatch send buffer by index scatter and activation gather | `45ce02d20` | **+17 / −2** | **+3.01pp**, matched 120-step A/B |
| D3 | Structured `custom_vjp` for the dispatch and combine gathers | `c9e30f848` | +117 code, +117 tests | **+3.43pp**, matched 120-step A/B |
| D4 | Shard the non-expert Newton–Schulz batch by zero-padding to the expert-mesh width | `497423bc6` | **+18 / −1**, +36 tests | **+1.78pp**, matched 20-step A/B |
| D5 | Route expert weight-gradient GEMMs through the QuACK grouped kernel at 256×256 tiles | `SCALE_QUACK_GROUPED_WGRAD` on the ECHO branch | flag + kernel path | **+0.861pp** (ECHO leg v134) |

D1 must be extracted, not cherry-picked: `fe21ea495` also carries C2, C3, a cutlass
revert and dispatch env forwarding.

D4 touches the same file as C2 — land C2 first and reconcile. **Take `497423bc6`,
not the flag:** `_newtonschulz_padded_stack_sharded` and `SCALE_MUON_PAD_NONEXPERT`
already exist on the shared base and on all seven `agent/ep25-*` branches, but
without `497423bc6`'s `target_sharding=` argument the padded result reshards to
fully replicated `P(None, None, None)` before slicing — which is precisely the cost
the mechanism removes. The +1.78pp belongs to the fixed variant only.

D5 sits at +0.861pp for a flag, but note the tension with the `sonic_cute` varlen-k
wgrad shim measured at +0.06–0.08pp at the d2560 row-13 scale. Different kernels,
different shapes, no matched control between them — verify at the EP64 shape before
counting it.

Default `SCALE_A2A_CHUNKS` to 1; chunks=2 measured worse.

### Phase E — fidelity. Required before any Phase D number is quotable.

| # | Commit | From | Size | Measured |
|--:|---|---|--:|---|
| E1 | Enable QB routing by default on the EP launcher path | `cff962d730` (extract — see below) | wiring | drops: collapse → ~6–7% steady; ≤ −1.44pp |
| E2 | Same-step spill to the next-ranked selected expert on bucket overflow | `1224ccb02` | **+147 / −12** | **−0.213pp for half the drops**; 20.708% at 1.44% with cf1.0625 |

E1 is the sharpest item in the series and it is not a code change so much as a
default change: `qb_routing` defaults to `False` and no recorded EP64 submit command
set it. Extracting it from `cff962d730` requires manual surgery — that commit
bundles `SCALE_ATTN_GATE`, `SCALE_XSA`, `SCALE_MOE_QB`, `SCALE_OFFLOAD_OPT_STATE`
and `SCALE_NO_HYPERBALL`, and the QB code changes `GrugTrainState` (adding
`pending_qb_betas`) and `next_token_loss`'s return signature, updating all callers.
Budget real time for E1; it is small in effect and awkward in mechanics.

**Ship E1 and E2 in the same PR as D2–D4, or ship them first.** Landing the
throughput work without the fidelity work reproduces exactly the situation the
record spent a week correcting.

### Phase F — FSDP-line levers, small and cheap.

| # | Commit | From | Size | Measured (FSDP, 1 rack) |
|--:|---|---|--:|---|
| F1 | Split the shared expert into two half-width experts | `b200-300B-tune` | small | +0.29pp |
| F2 | Group same-shape non-expert Newton–Schulz leaves into one call | `b200-300B-tune` | small | +0.09pp |
| F3 | Offload MuonH optimizer state to pinned host memory | `cff962d730` (extract) | ~+45 / −6 | +0.4pp (bundled) |

F2 overlaps D4's code; sequence F2 after D4. F3 shares the extraction problem with
E1 — do them together.

### Explicitly out of scope

- **Receiver-ECHO** (`24ee86090`, +725 in one file, plus HybridEP and a 680-line
  MNNVL CUDA FFI). Best compliant number on record, largest cost by far, and its
  headline is a 20-step screen. Separate project.
- **MXFP8 / FP8** (evidence Group G). Speed/quality trade with a measured price and
  an unresolved sign at the production shape; the fused kernels are 20,587 vendored
  lines. Separate, explicitly-priced decision.
- **`SCALE_MOE_EXPERT_CHUNKS`** — does not apply under EP, and its branch history is
  four revert pairs. If it lands for the FSDP line, take the squashed end state.
- **Slim Sonic residuals** (`59e5fe25f`) — conflicts with chunking; both edit
  `_expert_mlp`, never combined.
- **Latent MoE**, **`ring_cute` at EP64**, **TransformerEngine NCCL_EP** — all
  measured at or below parity; see evidence Group E.
- The `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false` guard
  ([#7494](https://github.com/marin-community/marin/pull/7494)) is worth landing on
  its own merits but is a single-host footgun guard, not a rack-scale speedup —
  multi-host `ragged_all_to_all` already takes the NCCL path automatically.

## Verification per phase

Every phase needs at least a shape/HLO assertion, because most of these changes are
invisible to a numerical test:

- **C3/C4:** assert the resulting `PartitionSpec` on non-expert weights and on the
  dispatch-boundary activation, and assert the dispatch buffer size does not scale
  with the expert-axis width.
- **C1/C2/D4:** assert the Newton–Schulz update matches the unsharded path, and
  assert zero SPMD involuntary-remat warnings.
- **D2/D3:** the existing kernel gradient-parity tests at rtol = atol = 1e-5, plus
  an HLO assertion that the backward contains no `scatter` (the whole point is
  544 → 0).
- **B3:** an HLO assertion that embedding lookup contains no global token exchange —
  named as the required regression guard in
  [#7493](https://github.com/marin-community/marin/pull/7493).
- **E2:** assert a spilled assignment computes its router-endorsed expert at its own
  combine weight, and that total capacity is unchanged.
