# Ragged EP hero MFU campaign — results and promotion dossier

Branch `research/mcwitt/8753-mfu-loop`, based on PR #8753 head (04026e94).
Protocol, evidence trail, and every arm: `DESIGN.md`, `LOGBOOK.md`, `results.tsv`,
`arms.tsv` in this directory. All measurements are single-rack GB200 restores from
the hero's step-30000 permanent checkpoint, scored as the run median of
`throughput/mfu` over restore steps +5..+19, with drops and the pointwise loss
series as guards.

## Headline

**23.36 → 24.04 MFU (+0.68, +2.9% throughput), measured same-night, same protocol,
both sides fidelity-clean.** Peak HBM also falls 116.57 → 112.75 GiB.

That end-to-end number is a direct measurement (campaign-zero tree vs the full
stack on the same night), not a sum of parts. The parts add to +0.81, so the two
keeps are mildly sub-additive: they overlap in what they fix.

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
  a wrong in-code rationale. The chunks=1 arm is a capacity-policy change, not a
  fidelity-preserving optimization, and was deliberately not run as a keep
  candidate.

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

The recurring lesson: on this stack, levers that **reschedule** work engaged and
lost; the two that **deleted** work paid.
