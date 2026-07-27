# Fixed nested MoE: QB target mismatch

Run true E16 within E128 within E256 expert nesting without allowing the router
load balancer to distort the fixed expert subsets.

## Initial status

The EP=16 fixed50 scratch pilot remained finite through its 512-step learning
rate warmup, but its median router-bias norm reached roughly 1,803 by update
599. In a matched promoted wave, fixed50 reached roughly 469 by update 339,
versus 36--37 for fixed25 and E256.

## Hypothesis 1

QB used one uniform assignment target for all 256 experts. Fixed nesting makes
that target impossible: E16 experts are eligible on full, E128, and E16 rows,
while outer E256 experts are eligible only on full rows. The resulting bias
growth is a router-control artifact, not useful nested-model pressure.

## Changes to make

Compute eligibility-weighted per-expert assignment targets and select each
expert's corresponding local quantile.

## Results

The eligibility-weighted implementation passed a deterministic small-shape
target-count test, but the 64-GB200 pilot failed before the first optimizer
update on four consecutive attempts. Rank 0 exited during first-step
compilation/runtime; only secondary XLA coordination cancellation reached the
persisted task diagnostic. The implementation was removed rather than promoted.

## Hypothesis 2

A matched no-QB comparison can isolate whether fixed nested training itself is
viable without conflating it with an eligibility-incompatible router
controller.

## Changes to make

Add an explicit router-balance mode. Preserve QB as the default; the `none`
mode emits zero pending QB betas. Run E256, fixed25, and fixed50 from scratch
with the same no-QB mode and otherwise identical optimizer, topology, data, and
evaluation settings.

## Results

The compact behavioral test and changed-file lint/type checks passed. The
matched r39 jobs were submitted on 2026-07-27 at 19:20 UTC. The gate failed:
capacity overflow reached 50.14% in E256, 16.77% in fixed25, and 1.91% in
fixed50 before update 600. Router biases remained zero. Disabling assignment
control is therefore not viable.

## Hypothesis 3

An auxiliary load-balance loss conditioned on each token's fixed eligibility
set can control collapse without imposing an impossible uniform assignment
target across routing modes.

## Changes to make

Compute the standard assignment-frequency/probability auxiliary loss separately
for full E256, fixed E128, and fixed E16 rows, then combine those losses in
proportion to group token count. Use coefficient 0.01 in a matched 600-update
three-arm gate.

## Results

The numerical reference and nested train-step lowering tests pass. The r40
three-arm gate was submitted at 19:30 UTC.

## Future work

- [ ] Verify r40 finiteness, overflow, router entropy, and assignment CV through
  the 512-step warmup.
- [ ] Compare no-QB full-model and extracted E128/E16 Paloma at 2,048-update
  gates.
- [ ] If fixed nesting is promising, design a production router controller
  whose assignment targets condition on eligibility without large static
  top-k compilation.

## Coefficient gates

Coefficient `0.01` remained finite but ended at `1.235%` capacity overflow for
fixed25 and `0.940%` for fixed50, narrowly failing the frozen below-`1%` gate.
At coefficient `0.02`, E256 ended at `0.132%` and fixed50 at `0.034%`, but
fixed25 became non-finite at update 3. This is an optimizer/controller failure,
not a model-quality result.

A final fixed25-only bracket at coefficient `0.015` was submitted as r43. It
must remain finite through update 599 and end below `1%` overflow. If it fails,
do not continue coefficient searching in this investigation.

## Long-schedule failure

Coefficient `0.015` passed the 600-step gate but failed on the actual
8,192-step schedule. The pilot cooled immediately after its update-512 peak;
the long schedule sustained high LR. E256, fixed25, and fixed50 reached
14.2%--14.5% overflow by updates 1,189--1,219. All three r44 arms were stopped
before evaluation.

## Eligibility-specific QB

The bounded structural follow-up stores one QB bias row per routing mode.
E256, E128, and E16 tokens compute their quantiles independently and apply only
their matching bias row. This removes the impossible shared assignment target
without adding an auxiliary gradient. Extraction selects and compacts the
matching nested bias.

Six focused finite-beta, masking, extraction, and lowering tests pass. The r45
three-arm long schedule uses updates 0--1,600 as the controller gate. Stop the
fixed-chain experiment if any arm is non-finite, exceeds 1% overflow, or
exceeds 10% treatment step overhead in that interval.

The first r45 attempt failed before update 0 because the group-beta scatter
started from `router_logits[0]`. That slice has length one on a dimension
sharded over 64 devices, so explicit sharding rejected the output shape. This
was an implementation failure with no experimental observation. The repair
allocates the 256-expert beta vector directly before scattering compact group
values. Lint and the focused eligibility-QB contract pass. The unchanged arms
were relaunched as r46; the same updates 0--1,600 gate applies.

r46 exposed a second explicit-sharding ambiguity before update 0: dynamically
gathering the routing-mode bias row required an explicit token-by-expert output
partition. The repair preserves batch-axis sharding and replicates the expert
dimension. Focused checks pass, and the unchanged arms were relaunched as r47.
Another controller-specific lowering failure ends this follow-up rather than
consuming the remaining experiment window.

r47 compiled and trained successfully. At roughly update 700, post-warmup
median step time was 160.16 ms for E256, 159.70 ms for fixed25, and 162.61 ms
for fixed50. E256 briefly reached 1.063% overflow at update 293, while fixed25
and fixed50 remained below 0.53%. All arms returned near zero and stayed
finite. This technically misses the strict threshold in the untreated control,
but it is neither sustained nor treatment-specific. Continue to the
1,600-update gate and record the deviation rather than treating it as evidence
against fixed nesting.

The sustained-peak gate passed. Across updates 512--1,600, E256, fixed25, and
fixed50 median step times were 159.94, 159.80, and 162.50 ms. Maximum overflow
in that interval was 0.0480%, 0.0367%, and 0.0489%. Losses and gradients stayed
finite while the Muon rate remained near peak. Continue all arms in place.
