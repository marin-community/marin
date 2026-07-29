# B200 performance omnibus — consolidating the proven EP64 throughput work

Compiled 2026-07-28 against `origin/main` @ `1c631c4c0`.

This is an implementation brief, not a result. It collects the changes that are
**measured** to improve GB200 MoE training throughput at a production operating
point with expert parallelism, orders them by benefit against implementation cost,
and says what must be derisked before any of it is trusted at hero-run scale.

Three companion files:

- [`evidence.md`](evidence.md) — one card per change: mechanism, measured deltas
  with their comparison arms, citations, code locations, complexity, dependencies,
  risk. **Every number in this file is sourced there.**
- [`sequence.md`](sequence.md) — the ordered commit plan, with the branch topology
  and the known merge conflicts.
- [`derisking-plan.md`](derisking-plan.md) — the triage of that queue against the code that exists: what is runnable, blocked, or sealed, and the costed run plan.
- [`derisking.md`](derisking.md) — the experiment queue. Nothing in this brief
  was measured by this work; the queue is what would make it safe to act on.

---

## 1. What "~25% MFU with EP64" actually refers to

The request cited a ~25% EP64 figure from
[#7201](https://github.com/marin-community/marin/issues/7201). Four different
numbers in that thread get called ~25%, and they are not interchangeable. All are
one GB200 rack (64 GPUs), all on the 2.5 PFLOP/s GB200 bf16-dense denominator.

| MFU | configuration | drops | evidence quality |
|--:|---|--:|---|
| **25.50%** | d5120 4-of-256, ECHO, **shared intermediate widened 5,120 → 21,504** | **2.02%** | 200-step tail-100 — best-qualified row, but a *capacity* change, not a kernel result |
| **25.39%** | d5120 8-of-256, custom adjoint + leg-batched GEMMs, **QB off** | ~85% early | bench only — QB-off; the mechanism is now sealed, not disputed |
| **24.84 / 24.59%** | d6144 4-of-128, QB on cf1.0, custom adjoint + host offload | **8.9–13%** | 120-step, above the 6% bar |
| **24.15%** | d5120 4-of-256, ECHO + padded stack-sharded Muon | **2.77%** | 20-step screen only |
| **22.30%** | d5120 4-of-256, ECHO (v143) | **1.71%** | 120-step tail-30 (tok/s not recorded) |
| **21.05%** | d5120 4-of-256, ECHO, sh5120 QB baseline (v128) | **1.78%** | 200-step tail-50 |
| **20.71%** | d5120 **8**-of-256, QB + same-step spill m=3 at cf1.0625 | **1.44%** | 350-step true tail-100 |

Four things have to be read alongside that table.

**These are reported MFU, inflated ×1.08** at seq 4096 with sliding window 512,
because `lm_flops_per_token` counts full O(seq²) attention on all 48 layers when 40
are windowed. A reported 24.153% is ≈ **22.4% true**. The factor is uniform, so A/B
deltas hold and absolute levels do not.

**Drop rate inflates MFU.** Expert GEMMs run on fixed capacity-sized buffers and a
dropped assignment gathers a zero pad row, so a configuration that drops more reads
*higher* MFU for less real work. Every cross-drop-regime gap in the record is an
upper bound, including the "QB costs 1.44pp" figure.

**Rows 1 and 4–6 are one arm; row 7 is another. Do not rank across them.** The
receiver-ECHO legs run **top-4** of 256 at routed i2048; the spill/adjoint leg runs
**top-8** of 256 at i1280. Their drop metrics are different quantities — post-ECHO
aggregate assignment drop against sender-local bucket drop — and top-k moves the
statistical floor (0.88% at top-8, 1.24% at top-4). Comparing them confounds top-k,
dispatch mechanism and drop metric at once.

**Rows 2 and 3 are not shippable configurations.** The 25.39% and the d6144 legs
route with quantile balancing off — the research branches gate it behind
`SCALE_MOE_QB`, which no recorded EP64 submit command set (QB is hardcoded on
`main`, but those branches diverged from an older main) — so
the router collapses, dropping 85–89% of assignments early and oscillating 17–79%
over a full run, invisibly, because the always-on shared expert keeps loss
descending
([#7201 c5080459722](https://github.com/marin-community/marin/issues/7201#issuecomment-5080459722)).
Row 2 is doubly weak, though not for the reason earlier drafts gave. The
implementation *was* committed (`98737aecf`, tag
`ep64-27pct-sender-clipped-baseline-20260725`, on
`origin/research/rav/7201-ep64-drop3`), and the −3.66pp reconstruction is a
**different change** — it also moves the collective schedule, which the original
never did. What disqualifies row 2 is fidelity: QB-off at ~85% early drops.

**Row 1 is real, well-qualified, and still not an optimization.** It is a 200-step
QB-on run at 2.02% drops — the best-qualified figure in the table — but it buys its
MFU by widening the dense shared expert more than fourfold, which changes the model.
Its own logbook entry seals it as *"reproducible capacity option, not a
matched-shape EP-kernel improvement."* It belongs to an architecture decision, not
to this ledger; it appears here so nobody reads the band below as a hard ceiling.

**At the production-candidate architectures, what is honestly achievable at EP64
today at a compliant drop rate is 21–24% on one rack**, and the top of that band
rests on a 20-step screen. Nothing at EP64 has been measured beyond one rack.

For comparison, the FSDP line (EP1) reaches **25.2%** on one rack at d6144
4-of-128 with PGLE, two shared experts, Muon shape-grouping, gated-norm, attention
gate and XSA — but its drop rate is **unmeasured**
([#7201 c5093392733](https://github.com/marin-community/marin/issues/7201#issuecomment-5093392733)).
So the EP-versus-FSDP comparison is currently one-sided on fidelity in EP's
disfavour. Closing it is **not** the one-line change earlier drafts assumed: the
drop metric does not exist on the FSDP line at all, and the chunked `sonic_cute`
backend that the 23.1% baseline ran on returns a literal zero dropped-assignment
count while that path genuinely drops. See
[`derisking-plan.md`](derisking-plan.md) §2.1.

### EP32 is not an operating point

The request framed the target as "EP32 or EP64". EP32 has two measurements, both
bad: `ring_cute` EP32 = 9.5% and `ragged_all_to_all_cute` EP32 = 12.3% at d2560/32
GPUs, and at the d5120 L48 b1024 reference config **both EP32 arms OOM** on a
single ~104 GiB temporary. EP64 is exactly one NVLink domain, which is what keeps
every MoE all-to-all off InfiniBand; EP32 pays dispatch overhead without the memory
relief that motivates EP. **Treat EP64 as the only EP operating point.** See the
interlude in [`evidence.md`](evidence.md).

---

## 2. The ledger, ordered by benefit against complexity

Dependencies come before dependents even where that violates strict ratio order;
those cases are marked. "LOC" is the functional diff against `origin/main`,
excluding vendored code, tests and research scaffolding.

### Tier 0 — Free or nearly free. Do these regardless of anything else.

| # | Change | Benefit | LOC | Notes |
|--:|---|---|--:|---|
| 1 | **Exact drop metric + tracker logging** (C4) | none directly — but it is what makes every other number here interpretable, and it is the largest single uncertainty on both EP-vs-FSDP comparisons | ~1 + metric | `2d4a87395` + `4fbc89152`. Exists in two worktrees, unlanded. **Do this first.** |
| 2 | **XLA collective-overlap flags** (B6) | **+0.47pp** at EP64; **+1.1pp** (PGLE alone) on the FSDP line at 1 rack | **0** | `JAX_ENABLE_PGLE=true`, `JAX_PGLE_PROFILING_RUNS=5`, `--xla_gpu_enable_latency_hiding_scheduler=true`, `--xla_gpu_experimental_parallel_collective_overlap_limit=4` (default is 1). Use the *manual* FDO flow — auto-PGLE crashes multi-host (E19). |
| 3 | **Keep the batch sharded over `expert` before EP dispatch** (A3) | enabler; prevents a **64× dispatch-buffer blowup — 320 GiB against 5 GiB** | **+11**, 10 of them comment | Highest value-per-line item in the ledger. |
| 4 | **Capacity-factor env knob** (A4) | enabler for every fidelity result in Tier 2 | ~+8 | The mechanism is already on `main`; only the env override is missing. Ship with item 1, not instead of it. |

### Tier 1 — The EP64 core. Large measured wins, small diffs.

| # | Change | Benefit | LOC | Depends on |
|--:|---|---|--:|---|
| 5 | **Non-expert FSDP sharding over `("data","expert")`** (A1) | enabler — without it, ~148 GiB/GPU and OOM before step 1 | +23 / +8 | — |
| 6 | **MuonH 4D Newton–Schulz expert-sharding fix** (A2) | enabler (~300 GiB replication) **and +1.8pp measured** (20.22% → 22.02%, 64-GPU probe) | +222 foundation, then +54 or +33 | Two competing designs — **resolve first** (§4) |
| 7 | **Fixed-capacity `lax.all_to_all`** replacing `ragged_all_to_all` (B2) | ~13% → **17.8%** with the Tier-1 bundle | +164 | 3, 5, 6 |
| 8 | **Gather dispatch** — int32 assignment scatter + activation gather (B3) | **+3.01pp / +17.1%** (17.55% → 20.56%), matched 120-step A/B, non-overlapping bands | **+17 / −2, one file** | 7 |
| 9 | **Custom scatter-add adjoint** for both gathers (B4) | **+3.43pp / +16.6%** (20.61% → 24.04%), matched 120-step A/B, 544 backward scatter ops → 0 | +234 (117 of them tests) | 8 |
| 10 | **Padded, stack-sharded non-expert Muon** (B7) | **+1.78pp** (22.37% → 24.15%), matched 20-step A/B, and drops improve 2.92% → 2.77% | **+54 (36 tests)** | 6 |

Items 8 and 9 together are **+6.4pp for ~134 lines of implementation** — both land
in `_moe/ep_ragged_all_to_all.py`, with a further ~117 lines of tests. They are the
single best trade in the document and everything else should be sequenced around
landing them. Note the two deltas were measured against different baselines
(17.55% and 20.61%), so +6.4pp is the sum of two matched A/Bs on the same lineage,
not a single measurement of the pair.

### Tier 2 — Fidelity. Without these the Tier-1 numbers are not shippable.

| # | Change | Benefit | LOC | Depends on |
|--:|---|---|--:|---|
| 11 | **QB routing on** (C1) | takes drops from router collapse to ~6–7% steady; costs **at most −1.44pp** | pre-existing feature, wiring only | 4 |
| 12 | **Same-step spill, m = 3** (C2) | **halves drops for −0.213pp** — 7.10% → 3.66% at cf1.0. With cf1.0625: **20.708% at 1.44%** | **+147** | 7, 11 |
| 13 | **Receiver-ECHO** with same-expert clones (C3) | **no on/off delta exists.** The one matched isolation (#7670) takes drops 1.32% → **0.02%** for about **−1pp** MFU | **+725 in one file**, plus HybridEP and a 680-line MNNVL CUDA FFI on the same branch | 7, 11, **14** |

Item 12 is cheap, well-qualified (350-step legs, true tail windows) and should
land.

**Item 13 cannot be ranked on the 24.15% figure.** That number is the *treatment
arm of the padded-Muon A/B* — both arms already ran receiver-ECHO — so it measures
item 10, not item 13. On its own measured merits ECHO is a fidelity trade costing
about 1pp of MFU, bought with the highest complexity in the ledger. It is a
separate project, not part of this consolidation, and it also depends on the
`sonic_cute`/QuACK substrate (item 14) because its kernel path uses the SM100
grouped expert GEMMs.

### Tier 3 — Shared substrate. Needed by both the EP and FSDP lines.

| # | Change | Benefit | LOC | Notes |
|--:|---|---|--:|---|
| 14 | **QuACK SM100 grouped expert GEMM** (`sonic_cute`) (B8) | `sonic` → `sonic_cute` 12.5% → 14.9% at d2560, 16.2% → 17.8% at d5120 (8×B200 single node); ring EP8 +0.38pp; a2a EP4 +1.34pp | **~+560** additive, lazily imported | The **branch-tip** file set is byte-identical on the FSDP and EP branches; the 105-line PoC in `5cf76b64a` is *not* that version. Five divergent `sonic_cute.py` blobs exist — see §4 |
| 15 | **FA4 per-layer bounds precomputed outside the scan** (B1) | fixes the 8-rack device-0 remat wedge | +156 / −30 | The FA4 kernel itself is already on `main` |
| 16 | **Replica-local embedding gather** (D6) | fixes the 8-rack NCCL rendezvous wedge | **+32 / −6** | `bdf61d7ed`, validated at 512 GPU. Cleanest cherry-pick in the set |

### Tier 4 — Proven on FSDP, unmeasured under EP. Highest-value open question.

Larry's read, stated directly on the thread: *"Maybe if it incorporates all the
optimizations on the FSDP stack it already is (none of the FSDP stack
optimizations are specific to the MoE)?"*
([#7201 c5093296092](https://github.com/marin-community/marin/issues/7201#issuecomment-5093296092)).

| # | Change | Benefit on FSDP | Status under EP |
|--:|---|---|---|
| 17 | Two shared experts, split (D2) | +0.29pp, **unreplicated single screen** | **memory-blocked**, not merely untested — the one attempt (two 8192-wide) failed before step 0 at 89.49 GiB; splitting the shared width does not reduce the FSDP gather peak |
| 18 | Muon shape-grouping for non-expert NS (D3) | +0.09pp, **unreplicated single screen** | **unmeasured**; overlaps item 10's code |
| 19 | Host offload of optimizer state (D4) | +0.4pp (bundled) | **split by model size** — in use on the d6144 EP64 legs, but *rejected* at d5120 (needed a 135 GiB pinned-host arena, landed at 19.694%) |
| 20 | Chunked expert FSDP all-gather (D5) | +0.9pp 1 rack, +0.5pt 2 racks | **N/A** — overlaps an all-gather EP does not perform |
| 21 | Slim Sonic residuals + `all_but_moe` remat (D7) | +0.51pp at d2560 | unmeasured; **conflicts with item 20's code** (§4) |
| 22 | GatedNorm + attention gate + XSA | part of the 25.2% stack | **absent from the EP runner entirely** — and an explicit caveat on the EP-vs-FSDP claim |

Item 18 and item 22 are the genuinely open ones; together with the rest they are
roughly **+1.5pp of reported FSDP gain that EP64 has not collected**, and they are
architecture-level rather than MoE-specific.

**Do not sequence items 17 and 18 into the commit series yet.** Their +0.29pp and
+0.09pp come from a single stacked progression; the source explicitly reports
replication only for PGLE (*"Reproduced across two runs (24.99%, 24.77%)"*), not
for these. Both sit far below the ~2pp threshold that this document's own protocol
says needs repeated placement draws. They are unreplicated screens, and their sign
is not established.

**PGLE (item 2) is the one that would matter most, and it does not transfer
cleanly.** It was +1.1pp on FSDP but only +0.47pp under EP combined with the
overlap limit; manual PGLE was *rejected* on the ECHO line (it matched 217 of 535
instructions and came in 0.235pp below the AutoPGLE leg); and the headline
padded-Muon A/B ran with PGLE **off**, because the ~16-minute compile made
preemption certain.

### Tier 5 — Conditional. Precision. Do not sequence this into the consolidation.

**The expert-only MXFP8 port has been tested at the EP64 operating point and it
lost.** Matched 120-step A/B, d5120/i1280 8-of-256 EP64, QB on, cf1.0, drops
reported and essentially matched (0.0885 against 0.0847, so not a drop artifact):
BF16 22.345% p50 against MXFP8 19.763% — **−2.582pp p50, −2.832pp mean, −12.46%
relative**. A fatter-shape check at d6144/i3072 stayed negative (−0.313pp, bands
non-overlapping). The recorded verdict is *"do not adopt MXFP8 expert GEMMs at this
operating point"* (`24d411b38`, local-only). See
[`evidence.md` Group G](evidence.md).

That closes the question for the current mechanism. What remains open is narrower:
a *materially different* mechanism — fused quantization epilogues, or the full
hybrid grouped-plus-dense recipe at a shape not yet tested — plus a new matched
all-QB-on end-to-end pair.

The rest of the precision record, for context:

- Three end-to-end measurements of the same **hybrid** recipe: **1.308×** (d5120,
  EP8, 1 run/arm), **+7.22%** (d2560, EP8, 66B tokens/arm — the strongest evidence
  in the workstream), and **0.749×** (d6144, **EP1**, 1 run/arm). The hybrid recipe
  still has no EP-degree curve.
- The preregistered quality gate answered **in the negative**: +0.056% aggregate
  eval, +0.110% Paloma, +0.209% uncheatable loss, with BF16 favoured at **32 of 32**
  paired evaluations. MXFP8 never reaches BF16's terminal held-out targets within
  the fixed schedule.
- The fused kernels do buy a **37% smaller step temp arena**, which is directly
  relevant to EP capacity walls.
- The FP8 dispatch wire (#7665) is **the only lever whose gain grows with EP
  degree** — 1.286× fwd / 1.144× fwd+bwd at EP64, weight gradients bit-exact. But
  it is one layer in isolation, and it only pays when a quantized consumer exists
  downstream — and that consumer is the expert-GEMM port that just measured
  −2.582pp. The wire would have to more than cover that loss.
- A hybrid `w_down` NaN is masked by a guard whose root cause was never found.

**Recommendation: keep BF16 for the consolidation.** MXFP8 is not an unpriced
option awaiting a decision; at this operating point it is a measured loss.

---

## 3. What the record says will not work

Do not spend on these. Full table with citations in
[`evidence.md` Group E](evidence.md).

**The scheduling family is closed.** With collectives on the async stream, exposed
collective time (4.29 s) almost exactly fills compute idle (4.44 s of a 33.2 s
span) — the step is collective-**volume**-bound, not schedule-bound. This was
reproduced independently at the d6144 hero shape. Consequently: rotation `ppermute`
decomposition **−9.46pp**, token-chunk pipelining **−1.96pp**, weight-prefetch
overlap **null**, and PGLE/LHS beyond item 2 inert. **Reducing collective bytes is
the only remaining lever in that family.**

Also sealed: `ring_cute` at e256/EP64 (**DNF**, OOM at 141.79 GiB); ragged a2a with
the one-shot kernel off at EP64 (**12.38%**, roughly half of fixed+adjoint);
TransformerEngine NCCL_EP (**ties the Marin seam, both ~1.1–1.3pp behind
`a2a_cute`**); latent MoE at d6144 EP64 (**−0.23pp to −1.72pp** despite the wire
mechanism working exactly as predicted); SM comm/compute partitioning (**falsified
three separate times**); QB gain g=2 (**diverges**); fa4-lse as a primal output
(**+0.18pp**, below bar); MLA (neutral at best); native dense MXFP8 (**0.64–0.81×**);
NVFP4 (ruled out on risk).

**A pattern worth naming.** Isolated microbenchmark wins on this stack have failed
to survive end-to-end repeatedly and in both directions: the exact clone-weight-
gradient adjoint was 3.24×/4.63× in a microbench and **−0.167pp** on a rack; tile
autotune was ×1.08–1.28 in isolation and **+0.14pp** e2e; the varlen-k wgrad shim
was **+0.06–0.08pp**. Every candidate in the derisking queue is specified as a
layer-level-or-better A/B **inside the real step with remat on**.

---

## 4. Structural obstacles to the consolidation

**Almost nothing is merged.** `origin/main` has the grug MoE skeleton, the EP
`ring`/`ragged_all_to_all`/`deepep` backends, the FA4 CuTe kernel, capacity-factor
plumbing, CUTLASS DSL 4.6.0 (#7587) and JAX 0.11.0 with the OpenXLA CUBIN
discriminator fix (#7436). It has **none** of `sonic_cute`, `SCALE_MOE_IMPL`, `Pfsdp`,
`_embedding_gather`, `_newtonschulz_4d_distributed`, `SCALE_MOE_EXPERT_CHUNKS`,
`SCALE_OFFLOAD_OPT_STATE`, `SCALE_MUON_SYRK`, or `SCALE_A2A_CHUNKS`. It does carry
`quack-kernels` 0.5.0 transitively (via `flash-attn-4` in `uv.lock`, imported by
`grug/attention/_fa4_cute_segmented_bwd.py`); what is missing is the direct pinned
`quack-kernels[cu13]==0.6.1` the MoE backend needs. `main`'s `launch_cw_scale.py` is a 238-line skeleton with about
20 knobs; the research branches carry 60+.

**But there is one shared base, which makes this tractable.**
`origin/grug/embedding-gather-shard-map` (23 commits off `main@696eb370d`) is the
merge base of `chunk-moe-fsdp` → `b200-minimal`, `b200_mla`, `rav-grug-moe-ep64`,
and `codex/per-layer-kv-heads-static-fa4`. `rav/ep-2` is a rewritten replay of the
`b200-minimal` stack onto a newer main; its SHAs do not match even where content
does. `mcwitt/moe-standalone-ep` is an independent lineage.

Verified directly by blob hash: `sonic_cute.py` (272 lines, `4d53627060`),
`quack_moe_cute.py`, `quack_symmetric_cute.py` (`628f77fdb2`), `loss.py` and
`_fa4_cute.py` are **byte-identical at the branch tips** between the FSDP-tuning
branch (`b200-300B-tune`) and the EP branch (`agent/ep25-d1-adjoint`).
`sharding.py` differs by exactly four lines — the `Pfsdp` change. The two
production lines share a real substrate; Tier 3 is that substrate. **The identity
holds at the tips, not at the PoC commit** — `5cf76b64a` carries a 105-line
`sonic_cute.py` and no symmetric-GEMM file at all.

**Four conflicts to resolve before writing commits.**

1. **Two incompatible MuonH 4D NS designs.** `75c517148` (mcwitt) skips the merge
   entirely under EP and additionally fixes `_newtonschulz_padded_stack_sharded`
   with a two-hop reshard; `54bbe3d23`/`fe21ea495` (rav) transposes E to the front.
   `75c517148` is the better-argued design; the rav variant is the one with a 17.8%
   64-GPU measurement behind it. They will conflict textually.
2. **`Plm_head`.** The base branch sets `P("data","model")` (intra-rack lm_head
   gather, `ac6364557`); rav sets `P(Pfsdp,"model")`. Same line, different values.
   rav's subsumes the base at EP1 — take rav's.
3. **Five divergent `sonic_cute.py` blobs** across branches, diverging along two
   orthogonal axes: chunking (`+86`, `_moe_mlp_local_sonic_cute_chunked`) and slim
   residuals (`+86/−21`, the `custom_vjp` rework). **Both edit `_expert_mlp` and
   they have never been combined.**
4. **Host offload is entangled with QB routing.** `cff962d730` is a six-file
   omnibus bundling `SCALE_ATTN_GATE`, `SCALE_XSA`, `SCALE_MOE_QB` (which changes
   `GrugTrainState` with a `pending_qb_betas` field and changes `next_token_loss`'s
   return signature), `SCALE_OFFLOAD_OPT_STATE` and `SCALE_NO_HYPERBALL`, with the
   offload code interleaved line-by-line with the QB code in the same `train_step`
   hunks. This needs manual surgery, not a cherry-pick.

**Do not cherry-pick:** the temp smoke script (`458f647b5`, `d7decc466`); the
`device_flops` commits (`c81a29428`, `7f504d9cb` — `main` already has an equivalent
`"b200"` entry with identical numbers, and the branch uses the key `"gb200"`, so a
conflict is guaranteed); `538381606`'s cutlass version hunks (superseded by #7587);
and `docs/debug-log-per-long-layer-kv-cond.md` on
`codex/per-layer-kv-heads-static-fa4`, which violates the `AGENTS.md` rule against
`docs/debug-log-*` and belongs in `.agents/ops/`.

The ordered commit plan is in [`sequence.md`](sequence.md).

---

## 5. Open leads worth more than most of the above

| Lead | Why |
|---|---|
| **Multi-rack EP** | EP64 has **no multi-rack measurement at all** behind its ~65–75-day 20T projections, and the measured 1→2-rack drop on the FSDP line was ~19%, not the 7% the projections assume. The largest unquantified schedule risk. |
| **A clean 120-step run of the best ECHO recipe** | The 24.15% headline is a 20-step screen; the settled 120-step figure (22.30%) predates both padded Muon and `overlap_limit=4`. Nobody has run the current best recipe long enough to qualify it. |
| **FSDP-line levers under EP** | Tier 4, and mostly *not* ported. Muon shape-grouping and the GatedNorm/attn-gate/XSA trio are absent from the EP runner entirely — roughly **+1.5pp of measured FSDP gain EP64 has not collected**. |
| **A cross-branch MFU denominator mismatch** | rav's stack computes `attention_seq_len = min(sliding_window, seq_len)`; the ep25 stack passes full `seq_len`. At sw2048/seq4096 that is a **1.056× denominator gap**, so the entire 24.04 → 22.60 → 20.85 QB ladder is inflated ~5.6% relative to rav's figures. Every cross-branch comparison in this document inherits it. |
| **The capacity-factor cliff** | cf 1.00 → 1.05 costs 1.179pp for +0.05, while 1.05 → 1.15 costs 0.254pp for +0.10. The tile-alignment hypothesis was falsified. **Cause unknown**, and it prices every fidelity decision. |

**Closed since the earlier framing, so that effort is not re-spent:** the "three of
twelve all-to-all ops on the compute stream" lead is explained —
`GpuConvertAsyncCollectivesToSync` tags async-starts whose done is separated only by
no-ops, and a schedule census shows MoE SYNC all-to-all going 10 → 0 as
`overlap_limit` goes 1 → 4. Likewise the router-controller family is exhausted:
over-relaxed, damped, DeepSeek-integral, **and sender-local** bias are all measured
at or below `g=1` — with the damped arm *unavailable* rather than measured, since
its one clean leg lost its metrics to a log-shipping outage — and the sender-local
null overturned the hotspot hypothesis. The
revised reading is that the ~6% residual is batch-stochastic within-batch
burstiness — routing is 7–9× more clustered than independent-uniform.

---

## 6. Two reporting problems to fix while consolidating

**Reported MFU is window-blind and the error grows with sequence length.**
`lm_flops_per_token` counts full O(seq²) attention on all 48 layers although 40 are
windowed to 512, overcounting FLOPs ×1.08 at 4k, ×1.17 at 8k and ×2.14 at 65k.
Reported MFU therefore *rises* (22 → 24 → 36) while true efficiency *falls*
(20.6 → 20.3 → 17.0). Trust tok/s; treat MFU as inflated whenever seq ≠ 4096. The
fix is a window-aware attention term
([#7201 c5097482159](https://github.com/marin-community/marin/issues/7201#issuecomment-5097482159)).

**The record mixes MFU denominators** (W&B's convention against 2.5 PF/s GB200
bf16-dense) and architectures (top-8/256 against top-4/64/128), and several
headline claims were later reversed — the "EP8 hard ceiling", the first CUBIN root
cause, and the MXFP8 1.251×. **Any result feeding the hero-run decision should
lock denominator, config, drop regime and placement draws up front.**
