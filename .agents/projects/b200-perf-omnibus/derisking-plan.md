# Derisking plan

Triage of the eleven-item queue in [`derisking.md`](derisking.md) against the code
that actually exists, done 2026-07-28 in
`/home/marin/projects/marin/.worktrees/b200-perf-omnibus` (branch
`b200-perf-omnibus`). Nothing was submitted, pushed, or modified to produce this;
every classification below rests on a SHA, a `file:line`, or a URL.

The standing protocol in [`derisking.md`](derisking.md) §"Standing protocol"
applies unchanged to every leg here — locked denominator, drops reported beside
every MFU figure, tail windows at matched LR position, repeated placement draws
under ~2pp, `cuda_async` plus the two mandatory XLA flags on every arm.

## 1. Classification

| Item | Class | Blocker | Branch to run from |
|---|---|---|---|
| D-1 Drop rate of the FSDP baselines | **blocked (build)** | The A1 drop metric does not exist on the FSDP line: `git grep -c SCALE_REPORT_DROPS origin/b200-300B-tune` returns nothing, and `report_capacity_overflow=False` is hardcoded at `origin/b200-300B-tune:experiments/grug/moe/model.py:714`. The port is **not** the plumbing job the item assumes — see §2.1. | `origin/b200-300B-tune` @ `fd3e9bc5b` + hand-port of `4fbc89152` and `2d4a87395` |
| D-2 Composed EP64 stack at compliant fidelity | **blocked (build, small)** | No branch carries the composed stack. `agent/ep25-d1-adjoint` @ `f53f781ce` is missing exactly two things: `497423bc6` (padded Muon `target_sharding=`) and the manual-PGLE artifacts. | `agent/ep25-d1-adjoint` @ `f53f781ce` + `497423bc6` |
| D-3 Leg-batching contradiction | **sealed — resolved off-rack** | None. The contradiction is resolved by code recovery and an accounting correction, not by a run. A narrower successor (D-3′) is blocked; see §5.1. | — |
| D-3′ Does `SCALE_A2A_BATCH_EXPERT_GEMMS` pay on the ep25 stack? | **blocked (port)** | The original mechanism requires `SCALE_A2A_PACK_DISPATCH`/`PACK_COMBINE`, which do not exist on `agent/ep25-d1-adjoint`; porting it is a transport change, not a flag. | `98737aecf` (rav) → port onto D-2's build |
| D-4 Multi-rack EP64 | **blocked (on D-2)** | Config-only, not code: `agent/ep25-d1-adjoint:experiments/grug/moe/launch_cw_scale.py:169-217` already derives `data_axis` from replicas/expert_axis. It needs D-2's 1-rack denominator on the identical build. | Same build as D-2, `SCALE_GPU_REPLICAS=32` / `64` |
| D-5 `overlap_limit=4` census at d6144 | **not_an_experiment** (0 rack-hours) | None, but three of the item's claims are wrong — it is not CPU-only, it has never been run at d6144, and its prize is ~+0.1pp not ~2.8%. See §5.2. | `agent/ep25-d4-pipelined` @ `62b026409` |
| D-6a GatedNorm / attn-gate / XSA trio | **runnable_now** | None. `SCALE_GATED_NORM`, `SCALE_ATTN_GATE`, `SCALE_XSA` are wired at `agent/ep25-d1-adjoint:experiments/grug/moe/launch_cw_scale.py:154-156`. `README.md:173` ("absent from the EP runner entirely") is true of the recorded submit commands only. | `agent/ep25-d1-adjoint` @ `f53f781ce` |
| D-6b Muon shape-grouping | **blocked (port, after D-2)** | `_grouped_nonexpert_transform` and `SCALE_MUON_GROUP_NONEXPERT` exist only at `origin/b200-300B-tune:lib/levanter/src/levanter/optim/grugmuon.py:156,311`. ~60 lines into the same function region `497423bc6` edits, so it sequences after D-2's cherry-pick (`sequence.md:212` already says this). | D-2's build + port |
| D-7 Spill-ceiling sweep (m=5, m=7) | **runnable_now** | None. `SCALE_A2A_SPILL` is a free integer at `agent/ep25-d1-adjoint:lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py:76`. The other two D-7 directions are not experiments — burstiness-at-source has no design, and "accept 1.4%" is a decision. | `agent/ep25-d1-adjoint` @ `f53f781ce` |
| D-8 Does the EP-aware NS fix unblock EP32? | **runnable_now** | None. C2 is present via `fe21ea495`; the launcher accepts `expert_axis=32` (`launch_cw_scale.py:209` validates `256 % 32 == 0`). It does **not** need D-2 — see §5.3. | `agent/ep25-d1-adjoint` @ `f53f781ce` |
| D-9 FP8 dispatch wire in the real step | **blocked (port + a consumer that does not exist)** | `research/mcwitt/7279-fp8-dispatch-wire` @ `224a00811` carries `fp8_wire.py` but none of the EP64 training stack (zero hits for `SCALE_A2A_GATHER_DISPATCH`, `SCALE_A2A_CUSTOM_ADJOINT`, `SCALE_A2A_SPILL`, `SCALE_MOE_QB`, `SCALE_REPORT_DROPS`). The success criterion needs a quantized consumer; the only one measured −2.582pp p50 at this exact operating point (`24d411b38`). That branch is also on **zero** origin refs. | None today |
| D-10 MFU-vs-EP-degree for hybrid MXFP8 | **blocked (Tier-5 precondition unmet)** | The recorded reopening condition is a *materially different* mechanism plus a new matched all-QB-on pair (`derisking.md:221-228`); none exists. Separately, no single branch runs the hybrid recipe across EP1/8/16/64 — `research/mcwitt/7282-mxfp8-blackwell` disagrees between local (`0a3785463`) and origin (`c3cb334f8`). | None identified |
| D-11 Hybrid `w_down` NaN root cause | **not_an_experiment** (0 rack-hours) | None. Static HLO analysis plus a single-node repro; a repro branch already exists. Downstream of a Tier-5 decision that has gone the other way, so its value is a correctness audit of the guard, not throughput. | `origin/agent/b200-repro-mxfp8-wdown-nan-full` |

Verified: all SHAs cited above resolve in this clone
(`98737aecf`, `75c517148`, `497423bc6`, `c9e30f848`, `45ce02d20`, `1224ccb02`,
`2d4a87395`, `4fbc89152`, `65e3ca50d`, `0789a8482`, `fe21ea495`, `54bbe3d23`,
`5833e329e`, `f53f781ce`, `62b026409`, `fd3e9bc5b`, `3cff00adf`, `224a00811`,
`c3cb334f8`, `1a88e1f3b`).

## 2. Critical path — what must be built, in order

Four build items gate everything blocked. They are independent of each other
except where noted, and none is large.

### 2.1 A1 on the FSDP line — gates D-1 (both legs)

`sequence.md:70` sizes A1 at ~+30 lines and calls it tracker logging. Two
corrections.

**The size is wrong.** `2d4a87395` is +10/−0 and `4fbc89152` is +36/−20, so A1 is
+46/−20, roughly 50% larger than stated.

**More importantly, one of the two FSDP backends has a real drop bug that A1 must
fix, not merely report.** `origin/b200-300B-tune:lib/levanter/src/levanter/grug/_moe/sonic_cute.py:210`
returns `_zero_dropped_assignments()` from `_moe_mlp_local_sonic_cute_chunked`
(defined at :108), but that path *does* drop: each chunk takes a static
`cap = total_assignments * size // num_experts` (:160) and masks combine weight
with `valid = jnp.arange(cap) < jnp.minimum(count, cap)` (:194), so every
assignment past `cap` is silently zeroed while the next chunk's window starts at
`cu[hi]` and never picks them up. The 23.1% d6144 baseline ran with
`SCALE_MOE_EXPERT_CHUNKS=2`, i.e. on exactly this path. Returning a literal zero
there is incorrect, and it is what makes D-1a a measurement rather than a
tautology. The fix is `sum_c max(0, (cu[hi_c] − cu[lo_c]) − cap_c)` accumulated in
the existing loop.

By contrast the **unchunked** path is dropless by construction —
`_moe_mlp_local_sonic_cute` routes through `_prepare_moe_dispatch`, which argsorts
every assignment and takes `group_sizes = jnp.bincount(expert_ids)` with no
capacity anywhere, then scatters `mode="drop"` on indices that are in-bounds by
construction. `_zero_dropped_assignments()` at
`origin/b200-300B-tune:.../sonic_cute.py:105` is literally correct there. That
asymmetry is what makes D-1b a structural-zero prediction rather than a
measurement.

Do **not** `git apply` `4fbc89152`'s `model.py` hunks. That file differs by 805
lines between `origin/b200-300B-tune` and `agent/ep25-d1-adjoint` (MTP, sconv,
over-encoding), and a clean-looking apply into a wrong-but-similar function is the
recorded backport hazard. Flip `report_capacity_overflow=False` at
`origin/b200-300B-tune:experiments/grug/moe/model.py:714`, thread the count through
`MoEMLP.__call__` → `Block` → the layer scan → `next_token_loss`'s aux (that
branch's `next_token_loss` currently returns `(loss, qb_beta_per_layer)`), and
apply `2d4a87395` verbatim to `experiments/grug/moe/train.py`, which does apply
cleanly.

Maps to **Phase A, A1** in [`sequence.md`](sequence.md). `sequence.md:92` already
says "A1 first, always" — that remains right, and the chunked-path bug is an
additional reason.

### 2.2 `497423bc6` onto `agent/ep25-d1-adjoint` — gates D-2, then D-4 and D-6b

One cherry-pick, verified clean: the entire diff of `grugmuon.py` between
`agent/ep25-d1-adjoint` and `origin/research/rav/7201-ep64-muon-pad` is
`497423bc6`'s two hunks plus a docstring reword (20 insertions, 3 deletions).
Without the `target_sharding=` argument the padded result reshards to
`P(None,None,None)` before slicing, which is the cost the +1.78pp removes
(`sequence.md:164-169`).

Maps to **Phase D, D4**. `sequence.md:164` says land C2 first and reconcile; on
this branch C2 is already present via `fe21ea495`, so the reconciliation is the
one in §4.1 below.

Also copy `experiments/grug/moe/pgle/ep64-qb-adjoint-prefetch.pb` and
`experiments/grug/moe/pgle_convert.py` from `agent/ep25-d4-pipelined` @
`62b026409` — they exist on no other branch. **Thin evidence:** that profile was
captured on the d4-pipelined stack, and adding padded Muon changes the instruction
set. Manual PGLE already matched only 217 of 535 instructions on the ECHO line and
came in 0.235pp *below* the auto-PGLE leg (`derisking.md:148-156`), so budget a
re-capture rather than assuming reuse. The overlap-limit-4 half of the recipe is an
XLA flag and needs no code.

### 2.3 Pin the capacity-factor default — gates the interpretation of every drop figure

Two constants with the same name and different values on `origin/main`:
`experiments/grug/moe/model.py:52` = `1.0` and
`lib/levanter/src/levanter/grug/_moe/common.py:18` = `1.25`. (`sequence.md:78-79`
cites 51 and 19; both are off by one.) A silent 1.25 misprices every drop number in
D-1, D-2 and D-7. The extraction check also found `SCALE_CAPACITY_FACTOR`
implemented **three** times independently, not two as `sequence.md:74-76` says:
`595958b83` (config-field route, +20/−2), `3e149490f` (`model.py` env override,
+3/−1), and `54bbe3d23` itself. Reconcile to one; do not double-apply.

Maps to **Phase A, A2**. Every leg in §3 carries a provenance gate that greps the
logged hyperparameters for the capacity factor it believes it ran.

### 2.4 Muon shape-grouping port — gates D-6b only

~60 lines from `origin/b200-300B-tune:lib/levanter/src/levanter/optim/grugmuon.py:156-183,311-313`.
Sequence it after §2.2 because both edit the same function region. Its FSDP
evidence is one unreplicated screen at +0.09pp (`README.md:169`, `README.md:179-184`),
so this is the lowest-priority build item in the list and it is reasonable never to
do it.

Not on the path: D-9's `fp8_wire.py` port and D-10's hybrid-recipe unification.
Both sit behind a Tier-5 precondition that is not met, and neither should consume
build time before the precondition changes.

## 3. Costed run plan

Cheapest-high-value first. Rack = 16 GB200 nodes × 4 GPUs = 64 GPUs, one NVLink
domain. All legs on `cw-us-east-08a`, all with
`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`,
`--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false` (mandatory
on JAX 0.11) and `--xla_gpu_experimental_parallel_collective_overlap_limit=4`.

| Order | Leg | Build gate | Draws × steps | Rack-hours | What it decides |
|--:|---|---|---|--:|---|
| 0 | **D-5** census at d6144 | none | 1 node, 1 compile × 4 overlap limits | **0** | Closes the "three of twelve at 0.0% overlap" observation. Needs 4 GPUs, not CPU. |
| 1 | **D-8** EP32 diagnosis | none | 1 × 120 (dies at step 0 if predicted) | **1.0** | Whether EP32's ~104 GiB OOM shares C2's root cause. Predicted outcome is a fast death, which is the cheap one. |
| 2 | **D-1a** FSDP d6144 4-of-128 chunk-2 drops | §2.1 | 1 × 350 | **1.9** | The largest single uncertainty on both EP-vs-FSDP comparisons. Also the positive control for D-1b. |
| 3 | **D-1b** FSDP d5120 8-of-256 drops | §2.1 (same build) | 1 × 120 | **0.8** | Converts "unmeasured" to a number for the 19.17% baseline. Predicted structural zero. |
| 4 | **D-7 control** (spill m=3, cf1.0625) | none | 2 × 350 | **3.2** | Shared with D-6a. Submit once. |
| 5 | **D-6a** trio treatment | none | 2 × 350 | **3.2** | Whether item 22 of the Tier-4 ledger (`README.md:173`) is real. |
| 6 | **D-7 sweep** m=5, m=7 | none | 4 × 350 | **6.4** | Whether spill "keeps improving through m=7" — the stated architecture-selection argument for top-8 over top-4 (`derisking.md:194-197`). |
| 7 | **D-2** composed stack | §2.2 | 3 × 350 | **4.5** | Whether the Phase D gains compose. Gates D-4, D-6b, D-3′. |
| 8 | **D-4** 2-rack EP64 | D-2 | 2 draws × 2 racks | **5.0** | The largest unquantified schedule risk (`derisking.md:116`). |

**Committed total: 26.0 rack-hours.**

Conditional add-ons, not committed:

- D-6a stage 2 (three single-variable arms, 2 draws each): **+6.4**, only if the
  trio clears a ≥ +1.0% tok/s gate with non-overlapping two-draw bands.
- D-4 at 4 racks: **+10.0**, only if the 2-rack point lands inside the projection's
  assumed 7% weak-scaling penalty.
- D-6b shape-grouping: **+3.2**, after §2.4. Low priority given a +0.09pp
  unreplicated screen.
- D-1b escalation to 350 steps if the structural zero is falsified: **+0.8**.

Two scheduling constraints that change the cost if ignored. The D-7 control legs
are shared with D-6a — submitting them twice adds 3.2 rack-hours for nothing. And
legs 4–6 must be submitted **sequentially**, one job at a time, so the six legs are
six independent placement draws rather than one co-scheduled block; multi-rack and
single-rack GB200 gangs get only a soft `nvlink.domain.preferred` constraint and
placement swings measured ±2-4pp at EP64 (`derisking.md:25-27`).

Reporting rule that applies to D-6a specifically: XSA and the attention gate change
the model and add FLOPs that `lm_flops_per_token` does not account for, so MFU is
confounded across those arms in the same way it is confounded across sequence
lengths (`README.md:352-358`). Judge D-6a on tok/s; report MFU beside it and do not
rank on it.

## 4. Resolved prerequisite decisions

### 4.1 Decision 1 — which MuonH 4D Newton–Schulz fix: `75c517148`

`sequence.md:52-56` recommends `75c517148` and adds "re-run the 64-GPU probe
against it". **Take `75c517148`; drop the action item.** That probe already exists
and it is `75c517148`'s own:
[#7279 c5012284704](https://github.com/marin-community/marin/issues/7279#issuecomment-5012284704)
— the source `evidence.md:91-93` cites for 20.22% → 22.02% and 208 SPMD warnings →
0 — names `75c517148` explicitly and describes both of its mechanisms. The 17.8%
figure attributed to the rav variant is not an isolation of the NS fix at all;
`evidence.md:131` concedes "No isolated A/B; it is part of the 17.8%
reproduction."

So the evidentiary ordering in `sequence.md:52-56` and `README.md:295-299` — "better
design" versus "the one with the measurement" — is backwards:

| | `75c517148` | rav transpose (`54bbe3d23`/`fe21ea495`) |
|---|---|---|
| isolated 64-GPU A/B of this mechanism | yes (20.22 → 22.02%, 208 → 0 warnings) | no |
| bundle reproduction | no | yes (17.8%, D1+C1–C4) |
| guard can silently fall back to the ~300 GiB path | no | yes — `"expert" in best_axes` at `54bbe3d23` post-image :457 |
| restores `orig_4d_spec` on exit | yes (:420) | no — exits at `P(None,"expert",None,None)` |
| fixes the padded-stack inbound reshard | yes (:581-586) | no |

Both are value-equivalent (Newton–Schulz is applied per `(D, last)` matrix under
`vmap`, so it commutes with any batch permutation) and both give 192 matrices per
device at single-rack EP64. They differ in layout, in guard safety, and in
multi-rack behaviour.

Three consequences:

1. **`75c517148` and `497423bc6` are complementary, not competing.**
   `75c517148` fixes the reshard immediately *before* `jax.vmap(local)(Xd)` in
   `_newtonschulz_padded_stack_sharded`; `497423bc6` adds `target_sharding=` to the
   reshard immediately *after*. The changed lines do not overlap, so the merge is
   mechanical — but they are on disjoint lineages
   (`git merge-base 75c517148 54bbe3d23` = `696eb370d`, plain `main`) and **have
   never run together**. The +1.78pp was measured without the inbound two-hop.
2. **At single-rack EP64 the inbound two-hop does not fire** — `batch_axis` is all
   size>1 mesh axes, which at one rack is just `("expert",)`, so the one-hop was
   already single-axis. Take it anyway; it is six lines and it removes a real
   multi-rack remat, which matters for D-4.
3. **Porting hazard.** `75c517148`'s EP path has no `SCALE_MUON_SYRK` branch,
   because `SCALE_MUON_SYRK` does not exist on the mcwitt lineage. On the
   rav/`b200-minimal` lineage the SYRK `jax.shard_map` sits between the forward and
   inverse transforms (`54bbe3d23` post-image :465-476). A naive conflict
   resolution leaves the EP path silently bypassing `SCALE_MUON_SYRK`. Either
   thread SYRK into the double-vmap branch or state in the commit that the flag is
   FSDP-only.

C2's size line should read **+54/−11**, not `sequence.md:137`'s "+54 or +33" —
that entry compares a gross diff against a net one.

What *does* need re-running as a consequence: the 17.8% bundle, since it was
measured with the rav variant in place. That is subsumed by D-2.

### 4.2 The leg-batching implementation is recovered, and `derisking.md:96-99` is wrong about it

`derisking.md:96` and `evidence.md` B5 both state the patch behind the 25.39% run
was never committed. It was — the next morning, on rav's branch:
`98737aecf` "[grug] Snapshot 27% EP64 research path", tag
`ep64-27pct-sender-clipped-baseline-20260725`, on
`origin/research/rav/7201-ep64-drop3` (also `…-drop3-handoff`), carrying
`SCALE_A2A_BATCH_EXPERT_GEMMS` in
`lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py`. An authoritative
second copy is the Iris job bundle for
`/rav/ep64-batched-expert-stability-120-v1-20260724-2353`
(`bundle_id 0483b2f207323fb3cd79ec326b7592546aabb0812ef8c058be95bd6c8049cd43`,
content hash verified against the download). Extracted artifacts are in this
session's scratchpad, not in the repo.

The recovery makes D-3's headline question answerable without rack time. See §5.1.

### 4.3 The two leg-batching numbers are not comparable, and the +1.35pp is arithmetic on mismatched denominators

rav's bundle `train.py` computes `attention_seq_len = min(sliding_window,
max_seq_len)`; the `agent/ep25-d1-adjoint` tree and its base `fe21ea495` do not —
they pass the full `seq_len`. Both ran `SCALE_SLIDING_WINDOW=2048`,
`SCALE_SEQ_LEN=4096`. Analytic `lm_flops_per_token`: 3.8205e10 at seq 4096 against
3.6180e10 at seq 2048, ratio 1.0560.

So 25.39% under the ep25-d1 accounting is 26.81%; equivalently the adjoint legs'
20.61/24.04 under honest sliding-window accounting are 19.52/22.77. Either way the
leg-batching gap is ≈ +2.6-2.8pp, not the +1.35pp the record computes — and the
whole 24.04 → 22.60 → 20.85 QB ladder in the #7279 milestone comment is inflated
~5.6% relative to rav's numbers. This is the same window-blind MFU defect
`README.md:352-358` identifies, appearing as a *cross-branch* denominator mismatch
rather than a cross-sequence-length one.

### 4.4 "Leg-batching stacks on the custom adjoint" describes a configuration nobody ran

The 25.39% job's environment (`job_config.environment_json`) sets
`SCALE_A2A_BATCH_EXPERT_GEMMS=1`, `SCALE_A2A_PACK_DISPATCH=1`,
`SCALE_A2A_PACK_COMBINE=1`, `SCALE_A2A_GATHER_DISPATCH=1`,
`SCALE_A2A_CUSTOM_DISTRIBUTED_COMBINE=1`, the four `SCALE_A2A_SONIC_*` flags and
`SCALE_A2A_NO_BARRIER=1`. It does **not** set `SCALE_A2A_CUSTOM_ADJOINT`, and
`SCALE_MOE_QB` is unset. The milestone comment's "the adjoint and leg-batching are
independent and stack" is unsupported.

### 4.5 The local-only-branch alarm is stale for the EP line and live for the FP8 line

`sequence.md:34-40` says all four `agent/ep25-*` branches exist only in this clone.
There are eight; six are byte-identical with their `origin/` counterparts, and the
two that differ (`agent/ep25-d3-fa4lse`, `agent/ep25-d3-te-ncclep`) still have
remotes at older commits. Every specifically named at-risk artifact is pushed:
`c9e30f848`, `1224ccb02`, `2d4a87395`, `4fbc89152` are each on three remote
branches including `origin/agent/ep25-d1-adjoint`.

The FP8 half of the same paragraph is correct and unchanged: `224a00811` is on zero
remote branches, and `0a3785463` is not an ancestor of
`origin/research/mcwitt/7282-mxfp8-blackwell` (`c3cb334f8`).

### 4.6 Two `sequence.md` Phase B corrections that gate the build

Not derisking items, but they will stop an implementer.

- **B1's source is incomplete.** `538381606` contains no mypy hunk — only the two
  `pyproject.toml` dependency blocks and `uv.lock`. The `cutlass`/`quack`
  `ignore-missing-imports` entries are in `5cf76b64a`'s root `pyproject.toml`
  (+5), which is otherwise the commit `sequence.md:113` tells you not to
  cherry-pick.
- **`5833e329e` is required and has no slot.** `sequence.md:128` asks whether
  #7587 already applied it. It did not:
  `origin/main:lib/levanter/src/levanter/grug/attention/_fa4_cute_segmented_bwd.py`
  still has 28 occurrences of `make_fragment` and zero of `make_rmem_tensor`, while
  `main` pins `nvidia-cutlass-dsl[cu13]==4.6.0`. `main` carries the 4.6.0 pin with
  the pre-4.6.0 API in the FA4 backward. `5833e329e` (+5/−5, one file) needs a
  numbered slot ahead of anything exercising that path.

Also: `C4`'s source commit `54bbe3d23` contains **two** distinct +11/−0 hunks. The
one C4 wants is in `grug_moe.py` (`batch_spec = _batch_spec(mesh)` plus a 10-line
comment). The other is the two `tree_checkpoint_name` a2a remat markers in
`ep_ragged_all_to_all.py`, which the plan elsewhere says to drop. Matching on
"+11 in `54bbe3d23`" takes the wrong one.

## 5. What this triage changed about the queue

Net effect: one P0 item sealed without rack time, one P0 item's cost roughly
doubled, one P1 item's prize repriced down by an order of magnitude, and two items
moved from blocked to runnable today. The committed queue is 26.0 rack-hours
against a pre-triage reading that implied roughly 40 plus an unbounded code-recovery
task.

### 5.1 D-3 is sealed, and it was never the experiment it looked like

The two arms are **different changes**, which is why they disagree.

In the original (`SCALE_A2A_BATCH_EXPERT_GEMMS`, `98737aecf`) the collective
restructuring is a *precondition*, not part of the patch: the knob hard-requires
`SCALE_A2A_PACK_DISPATCH`/`PACK_COMBINE`, which already collapse four
per-local-expert `all_to_all` calls into one dispatch and one combine over
`split_axis=1`/`concat_axis=1`. Turning the knob on changes **only compute** — it
replaces `local_experts` separate `(bucket, H) @ (H, I2)` matmuls with one batched
`jnp.matmul` over a leading expert axis. Wire traffic, a2a shapes and axis
semantics are bit-identical to its control.

The reconstruction (`SCALE_A2A_BATCH_EXPERTS`, `65e3ca50d`) fuses both concerns
behind one flag on a baseline with **no packing at all** — that file has zero
references to `PACK_DISPATCH`, `PACK_COMBINE` or `SONIC`. It simultaneously moves
`split_axis`/`concat_axis` from 0 to 1 (4 collectives → 1) *and* batches the GEMM,
composed with `SCALE_A2A_CUSTOM_ADJOINT` and QB on. `0789a8482` then re-splits into
`local_experts/G` groups, so the measured `SCALE_A2A_BATCH_GROUP=2` treatment runs
2 dispatch + 2 combine a2a of a new shape plus a `jnp.concatenate` — a *third*
collective schedule present in neither the original nor its own control.

So the −3.66pp prices a collective-schedule change the +25.39% path never made.
Combined with §4.3 (the +1.35pp is invalid arithmetic) and §4.4 (the composition
claim describes an unrun configuration), the "open disagreement" framing in
`README.md:333` and `derisking.md:88-94` is retired. `25.39%` should still not
appear as an achievable number — it is QB-off at ~85% early drops
(`README.md:32`) — but for fidelity reasons, not because the mechanism is
disputed.

What survives is D-3′: whether the *original* mechanism pays on the ep25 stack.
That is a port of the packed-dispatch transport, not a flag flip, and it sits behind
D-2. It is not scheduled.

### 5.2 D-5's prize is ~+0.1pp, not ~2.8%, and the item is a new measurement

Three corrections to `derisking.md:122-138`.

**It is not CPU-only and not "about three minutes on one node" from cold.**
`experiments/grug/moe/schedule_report.py` (149 lines, plus a 72-line test, on
`agent/ep25-d4-pipelined` only — `evidence.md:460`, `evidence.md:1333` and
`README.md:339` all cite the path with no branch attribution, so anyone following
from `main` will not find it) is a pure-text stdlib parser over an `--xla_dump_to`
directory. Producing the dump is a real multi-GPU compile: the recorded harness is
`/mwittmann/ep25d4-schedump-ep4-v1-20260726`, 1 node × 4 GB200 with
`GRUG_RUN_INLINE=1`, and the AGENT_LOG qualifies the three minutes as "with a warm
compile cache". It needs `expert_axis ≥ 2` real devices to produce any a2a legs.
Levanter's `log_xla_hlo` writes pre-optimization StableHLO and is useless here.

**The 10 → 3 → 0 → 1 curve was taken at `local_experts = 4` (d5120), not at
d6144.** The d6144 hero shape is 4-of-128 at EP64, i.e. `local_experts = 2`. The
script has never been run at d6144, so D-5 as written ("verify, do not
re-investigate") is a new measurement. It is still cheap. Also: the "14 reshard
SYNC" column in that census is an artifact of the 4-shard mesh and does not exist
at EP64, and the cover column was wrong before `54809714c`.

**The phenomenon is confirmed but the prize is not.** Four lines of evidence tie
the census to the d6144 "three of twelve at 0.0% overlap" observation, the
strongest being an exact structural match: at `local_experts = 2` the same code
yields 2 + 2 + 4 + 4 = **12** distinct a2a ops, exactly the "12 distinct
`all_to_all.N.1` HLO ops, 144 events each" in
[#7279 c5095217108](https://github.com/marin-community/marin/issues/7279#issuecomment-5095217108),
and the harness correctly predicts that fwd combine is the one leg with nothing
inline. But the rack leg that cleared them
(`/mwittmann/ep25d4-ovlim4-120-v2-20260726`) moved 2961 ms/3-steps of collective
off the compute stream and recovered only **463 ms** of net exposed collective —
about 16% — because the same GEMMs then had to hide 40% more async collective time
(hidden fraction 86.8% → 64.3%). MFU moved **+0.12pp**. Applying that ratio at
d6144 gives roughly 200 ms/3-steps ≈ 66 ms of a 15.3 s step ≈ **+0.1pp**, and that
is generous: the d6144 dense control is more collective-bound than d5120 (compute
idle 3,498 ms against 4,126 ms exposed collective, versus 4.44 s against 4.29 s).

`README.md:129-130`'s "worth more than the entire latent mechanism's best case" and
`derisking.md:129-130`'s framing both inherit this error. **422 ms of every 15.3 s
step is not 422 ms of recoverable step time.** Write D-5 up as closing an
unexplained observation, not as opening a recoverable win. It stays in the queue
because it costs nothing.

One thing the census cannot settle: the d4 log records that the overlap-limit knob
"only bites with LHS on", and the `XLA_FLAGS` of the d6144 legs were never recorded
on `agent/ep25-d5-d6144` or `agent/ep25-d6-latent` (`grep -rn
parallel_collective_overlap_limit` returns zero hits outside prose, so those legs
compiled at the default limit of 1). Set the latency-hiding scheduler and the
overlap limit **together**, as B6 already specifies.

### 5.3 Two items moved from blocked to runnable today

**D-6's first half needs no code.** The trio is wired at
`agent/ep25-d1-adjoint:experiments/grug/moe/launch_cw_scale.py:154-156` and
implemented in `experiments/grug/moe/model.py`. Only the shape-grouping half is
blocked. This matters for the Tier-4 headline: `README.md:176` counts the trio
inside "roughly +1.5pp of reported FSDP gain that EP64 has not collected". If the
trio does not pay, that total drops to at most +0.09pp — shape-grouping alone,
itself an unreplicated single screen — and the "EP just needs the FSDP
optimizations" hypothesis loses its largest remaining component.

**D-8 does not depend on D-2.** `derisking.md:212` scopes it as "one job on the D-2
stack", but the mechanism under test is C2 (`_newtonschulz_4d_distributed`, present
via `fe21ea495`), and padded Muon is a throughput lever that does not touch the
microbatch input-resharding path where the ~104 GiB OOM occurs
(`evidence.md:638-644`). Running it now removes a dependency edge from the critical
path for free. Caveat that should be stated in the writeup: this arm runs
`cuda_async` and the recorded EP32 failures may not have, so a *pass* is ambiguous
between "C2 fixed it" and "the allocator fixed it". The disambiguating follow-up is
one leg with the default allocator, not more EP32 tuning.

### 5.4 One item got more expensive, and one got cheaper

**D-1 is not "one line of code" (`derisking.md:61`).** It is +46/−20 of hand-ported
plumbing across two files that differ by 805 lines between branches, plus a real
drop-accounting fix in the chunked `sonic_cute` backend (§2.1). It remains the
highest-value run in the queue and should still go first among the rack legs.

**D-1's second leg got cheaper and sharper.** The 19.17% baseline is a *two-rack*
measurement (`evidence.md:1280`), which the item does not say. But the quantity
being measured — capacity overflow in the local expert path — has no dependence on
data-axis width, because that path has no capacity at all. So D-1b runs one rack at
120 steps for a predicted structural zero, at a fifth of the cost of reproducing
the baseline, and no MFU number from it may be quoted against 19.17%. D-1a is its
positive control: if D-1a reports exactly 0.000000, the port is unproven and
D-1b's zero means nothing.

### 5.5 Confirmed blocked, no change

D-9 and D-10 stay blocked on the Tier-5 precondition (`derisking.md:221-228`), and
D-9 additionally on a port that does not exist plus a quantized consumer whose only
instance measured −2.582pp p50 at this exact operating point (`24d411b38`). D-11
stays a desk task. No rack time for any of the three.
