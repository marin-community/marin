# Ragged EP hero MFU campaign — results and promotion dossier

Branch `research/mcwitt/8753-mfu-loop`, based on PR #8753 head (04026e94).
Protocol, evidence trail, and every arm: `DESIGN.md`, `LOGBOOK.md`, `results.tsv`,
`arms.tsv` in this directory. All measurements are single-rack GB200 restores from
the hero's step-30000 permanent checkpoint, scored as the run median of
`throughput/mfu` over restore steps +5..+19, with drops and the pointwise loss
series as guards.

## Headline

**23.34 → 24.04 MFU (+0.70, +3.0% throughput), measured same-night, same protocol,
both sides fidelity-clean.** Peak HBM also falls 116.57 → 112.75 GiB.

Re-anchored on the **wgrad-corrected** tree — the stack you would actually ship,
and now the headline of record:

**Delta: +0.81 MFU (+3.5% throughput), ~12.6σ.** Zero pair 23.183 / 23.120
(sd 0.045) against same-tree control pair 24.016 / 23.906 (sd 0.078), se 0.064.
Peak 116.57 on both zero draws, matching the pre-fix zero exactly. The pre-fix
measurement (+0.70) agrees within error.

**Absolute level of the deployed configuration: 24.03 ± 0.09 MFU**, pooling all
six draws of it — 24.161 / 24.081 / 24.073 / 24.016 / 23.969 / 23.906, four of
six above 24.0. Quote the pooled figure for the level and the same-tree pair for
the delta; do NOT read the control pair's 23.961 as the level. Those two draws
are two of the three lowest in the set, and pairing them against pre-fix draws
would be the cross-night comparison this protocol rules out (placement varies
±2.8% night to night; same-night pairs only).

The post-fix draws sit 0.07 below the pre-fix ones, which is inside the 0.090
spread and bounded far below it by arithmetic: the correctness fix adds ~768
padded rows out of ~300k on a 496 ms/step weight-gradient call, ≈0.002 MFU.

One caveat to carry with it: zero runs the branch-pinned PJRT because
``--pjrt-wheel`` postdates that commit, where the controls run the H11 wheel with
the env var set. The wheel is bit-comparable with the var unset — that is why the
patch was written env-gated with the default preserving today's behavior — so this
is a build difference, not a treatment difference.

That end-to-end number is a direct measurement (campaign-zero tree vs the full
stack on the same night), not a sum of parts: two zero draws at 23.356 / 23.326
(sd 0.02) against three deployment draws at 24.081 / 24.073 / 23.969 (sd 0.06),
a ~12σ separation. The parts add to +0.81, so the two keeps are mildly
sub-additive — they overlap in what they fix.

## Fidelity of the keeps, re-verified on the corrected stack

The keeps' original fidelity evidence predates the wgrad fix, which changes
gradients — so it was re-checked against the two zero and two control draws on
the corrected tree, pointwise over the full scored window:

| comparison | pointwise max |Δloss| |
|---|---|
| within-zero (z–z) | 5.08e-05 |
| within-control (c–c) | 6.19e-05 |
| across-arm (z–c) | **1.31e-04** |

Across-arm sits at ~2.1x the same-arm noise floor, inside the campaign's
calibrated ~1e-4 null band. The difference IS systematic — the keeps show lower
loss at 15 of 15 steps (sign test p≈3e-5) — but at ~8e-5 on a loss of 1.28 that
is bf16 reduction-order sensitivity from the changed output-init and CTA count,
and it runs in the favorable direction. Drops are identical at 3.3e-5.

## What to promote

### 1. Ragged-a2a zero-init deletion — code, hot-swap safe (+0.41 measured alone)

Commits `3e2afe8d8f` (+ hardening `5ef8defcd6`). Every ragged-all-to-all
output-init is built inside the layer loop from a per-site salted expression
instead of a hoisted `jnp.zeros`, and the two transport a2as sit behind
`custom_vjp` wrappers that replicate jax's transpose rule with private inits.

Why it works: `ragged_all_to_all` writes in place into its output-init operand, so
a loop-invariant zeros constant forced CopyInsertion to mint a fresh multi-GB copy
per layer per step — 785 ms/step of compute-stream `MemcpyD2D` at the hero shape.
The change deletes the copies rather than relocating them (the relocation variant,
H9, was measured and lost).

Evidence: compute-stream D2D 785 → ~4 ms/step; `#async_start_instructions`
unchanged (proving deletion, not relocation, which was H9's failure signature);
wall −272 ms/step in the trace, cross-validating the scored +0.41; pointwise loss
vs control at control-control levels; drops identical. Backward replication of
jax's transpose rule verified bitwise by two independent reviewers, including
under the hero's `offload_carry` checkpoint policy.

Checkpoint compatibility: unchanged — no state layout or dtype is touched.

### 2. Ragged-a2a device-kernel CTA cap — needs a wheel (+0.40 measured alone)

Requires the one-header XLA patch in `h11-dk-cta-cap.patch` (env-gated
`XLA_RAGGED_A2A_DK_CTAS_PER_SM`, default 8 = today's behavior, so a wheel carrying
it is bit-comparable until the env var is set) plus
`XLA_RAGGED_A2A_DK_CTAS_PER_SM=1` in the hero launcher's environment.

Why it works: the device kernel launches 8 CTAs on every SM (1216 on GB200) and
holds them — including through its barrier spins — which starves co-running GEMMs.
Measured: GEMMs overlapping the transport window run 3.0× slower at the default
grid; at 1 CTA/SM that collapses to 1.21×. The transport itself is only 6.6% slower
at one-eighth the grid, because at these message sizes it is link-bound, not
SM-bound.

Ladder (all engaged, `cta_count` confirmed in logs): 8 → 23.67, 4 → 23.93,
2 → 23.94, 1 → 24.08. Monotone toward fewer CTAs; deployment record at 1/SM is
24.081 / 24.073 / 23.969 (mean 24.04, sd 0.06).

Deployment note: the campaign built this wheel as a branch artifact only
(`marin-community/xla` branch `mcwitt/adhoc-ragged-dk-cta-cap`, wheel
`0.11.1+marin.ce6db0d2c555`, staged under
`s3://marin-us-east-02a/marin/research/mcwitt-mfuloop/pjrt-h11-cta-cap/`). No
release was published and nothing was pushed to any main branch. Promotion needs a
real fork release, which is a production act left to you.

## Also committed, default-inert (no measured gain, kept for other reasons)

- `--remat-save cheap_recompute` (`7228b5e066`): saves cheap recomputed
  intermediates by name. Measured engaged-null (+0.03): the recompute it deletes
  was not on the critical path. Default `()` lowers byte-identically.
- `--opt-resident-leaves` / `--opt-resident-donation` (`9c66d1ecc4`, `3572ccb3c1`):
  optimizer-state residency plus a corruption canary. **Do not enable** — see the
  hazard below. Kept as reusable forensics; defaults are byte-identical.
- `--expert-chunks` (`30b100e9a1`): makes the chunk count configurable and corrects
  a wrong in-code rationale.

## A validated option you may want, separate from the keeps

`--expert-chunks 1` makes the hero **exactly dropless** — 0.0 token drops on both
measured draws, against 3.3e-5 today — at a cost of about **0.45 MFU (~1.9%
throughput)**. Losses stay in family, peak rises only 5 GiB, and rematerialization
pressure actually falls. The accepted token set at one chunk is provably a superset
of the chunked one (the capacity gate is a lexicographic greedy and the chunk
capacities divide exactly), so nothing routed today is dropped there; it is strictly
less clipping, not different clipping.

It is not a campaign keep because the mandate was throughput at equal fidelity, and
this trades the other way. But if drop-freedom is ever worth ~1.9% to the hero, the
knob is committed, reviewed, and measured. Mechanism for the cost: one chunk moves
twice the rows per transport op, so each op runs ~2x longer and is correspondingly
harder to hide — and the transport is link-bound at that size, so the CTA cap cannot
recover it.

## A silent correctness defect, found and fixed (not a live incident)

The cuDNN grouped-Wgrad wrapper padded expert groups to 8 rows while the
vendored kernel declares ``FIX_PAD_SIZE = 256`` and its own ``can_implement``
rejects ``k % 256``. Marin imported the private kernel class and bypassed that
check, so every ragged-backend step computed expert weight gradients from an
over-read. Three independent lines converged: a fresh Opus auditor derived the
mechanism from vendor source and reproduced issue #8339's GB200 numbers to three
decimals *uniquely* at ``cta_tile_k=64``; a codex auditor found the declared
contract; and a controlled single-variable GB200 experiment measured it. Same
gate, same seeds, only the padding changed:

| seed | ragged grad error before | after fix | ring reference |
|---|---|---|---|
| 0 | 0.0259 | 0.000807 | 0.000673 |
| 1 | 0.0301 | 0.000443 | 0.000604 |
| 2 | 0.0229 | 0.000539 | 0.000738 |
| 3 | 0.0255 | 0.000515 | 0.000532 |

A 30–70x reduction to the ring floor. **Scope: the live hero is NOT affected** —
it was deployed on the pooled-wave backend, whose expert MLP is a plain
``jnp.einsum`` with no grouped GEMM. This is a **ragged-migration blocker**, and
it corrupted every ragged arm in this campaign (throughput is unaffected —
identical FLOPs — but loss-quality claims on ragged arms predate the fix).

Fix isolated as ``2ca4c1e046`` for cherry-picking, and measured **throughput-free**
(24.016 fixed-tree control vs 24.07 pre-fix, inside noise). The 4-GPU gate that
would have caught this **is not run by anything**: #8605 deleted the cluster-smoke
workflow after 38 consecutive timeouts, and #8704 tracks a replacement.

## Hazard found, worth fixing regardless of this campaign

Keeping an optimizer-state leaf device-resident **and donated** under the async
offload schedule corrupts training: bitwise-clean restore, +7.3e-4 divergence on
the first update, NaN by step 8. Excluding that leaf from donation eliminates it
(clean through the horizon, zero canary mismatches across the whole run). This
extends the #8317 hazard family — donated device-resident optimizer-state
read-modify-write leaves adjacent to async offload traffic are unsafe on this
stack. The campaign's own fidelity gates caught it within five rack-minutes.

## Two upstream-facing findings (nothing posted anywhere)

1. **openxla/xla PR #47928's fixed 8×SM grid is over-provisioned for overlapped MoE
   training** — see keep 2. The balanced-CTA assignment in that PR is sound and not
   in question; the fixed grid size is what costs. Under overlap it buys 6.6% of
   transport latency for ~535 ms/step of critical-path compute. The natural upstream
   fix is a tunable or occupancy-aware per-SM factor rather than a hardcoded 8.
2. **jax 0.11.1's `ragged_all_to_all` transpose mis-masks the passthrough cotangent
   when a group is empty** (`jax/_src/lax/parallel.py:1705`), draft report in
   `upstream-jax-transpose-bug-draft.md` with a two-shard counterexample. Latent at
   hero scale (groups average ~2048 tokens) but real at small scale, early training,
   or with a skewed router.

## What was ruled out, with attribution

Every negative here cost rack time or review time and is recorded so nobody re-runs
it: XLA memory-limit slop factor 85→90 (engaged, −0.12: the flag drives both the
scheduler budget and remat, and no remat-only knob exists at this revision);
async-wrapping the multi-GB D2D copies (engaged, −0.13, fully attributed: the
on-stream copies were accidental SM-immune overlap fillers, and freeing that window
fed GEMMs into 3.3× transport contention); pipelined host offloading (OOM-blocked at
the old stack, −0.25 at the new one); marin_ep transport port (declined: kMaxPeers=32
wheel crash at EP64, plus it is a posture package projecting below the current
stack); targeted save-policy (engaged-null); optimizer-state residency (unsafe, see
hazard). Three further candidates were desk-killed by verification before reaching
the rack — a dispatch-chain fusion whose premise was a misread trace family, and two
"dead code" deletions that HLO evidence showed were either never emitted or
load-bearing.

Closed in the second phase, each with a mechanism rather than a number:

- **C15, pre-aligned receiver layout** (-0.173, ~3σ, two draws). The mechanism
  *worked* — it deleted 180 ms/step of loop-fusion copies, reproduced in both
  draws — but gave it back through a reproducible **+3% transport slowdown on a
  payload that grew 0.20%** (bf16[301466,3072] → bf16[302080,3072], 614 rows),
  17x more than size explains. So the ragged kernel's cost tracks per-peer
  message granularity, not volume, matching the wave-quantization finding from
  the dk-native work — and it retroactively explains why the padding-fusion
  follow-up from the CuTe FFI campaign never materialized. The ``h`` mask it
  needs is not removable: the pre-aligned kernel genuinely reads ~40k rows past
  ``cu[-1]`` holding uninitialized memory.
- **Expert chunking, now closed on BOTH sides.** chunks=1 → -0.45 (but exactly
  dropless); chunks=2 (default) → baseline at 3.3e-5 drops; chunks=3 → -0.15 AND
  6.5x the drops (2.15e-4), a fidelity failure independent of throughput. The
  default is a genuine two-sided optimum.
- **Desk-killed before spending rack time**, by pricing each candidate against
  the measured budget of the leg it attacks: a combine-weight epilogue fusion
  (+7-10% in Megatron-Core / ERNIE 4.5 / Ling 2.0 — but ``_unpermute_from_global_expert``
  already fuses exactly that into one ``sonic_gather_sum``); a QB-histogram
  allreduce deferral (premise real and beta genuinely deferred, but capped at
  ~0.036 MFU against a 0.18 bar); a Newton-Schulz padding deletion (capped at
  ~0.07 MFU); and the whole remat direction (1065 ms/step of recompute, but the
  families are 9-160 GiB against ~26 GiB of headroom, and the one that fits
  deletes only ~12-17 ms/step).

The recurring lesson: on this stack, levers that **reschedule** work engaged and
lost; the two that **deleted** work paid. Nine independent perturbations — slop
factor, async copy threshold, pipelined offload (twice), command-buffer subsets,
save policy, receiver alignment, chunking in both directions, allocator fraction
— all engaged and all lost or came back null. **The published configuration is a
strongly defended local optimum**, and that is itself the campaign's most
reusable result: it says where NOT to spend the next engineer-week.
