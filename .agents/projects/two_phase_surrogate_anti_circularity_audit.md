# Two-Phase Surrogate Anti-Circularity Audit

Date: 2026-07-30

## Verdict

The research process has useful local safeguards but did not adequately preserve
global mechanism provenance across compaction and handoff boundaries.

The WSD80 logbook is append-only, preregisters many candidates, and increasingly
uses algebraic and outcome-free gates. Those controls prevented several recent
reparameterizations from consuming BPB outcomes. They did not link the current
series to the 99-route historical registry, so earlier mechanisms could re-enter
under a new base model or local approximation.

## Confirmed Reopenings

### Fisher phase information

WSD80-SUR-020 included a Fisher-quadratic phase-information cost. WSD80-SUR-032
later established that it is exactly the second-order Taylor approximation of
the phase-label Jensen-Shannon information tested and rejected as historical
route `prior_AB`.

This is a direct reopening. Provenance caught it before a second dedicated
outcome fit, but only after the coordinate had entered the SUR-020 candidate.

### Continuous retained-state dynamics

WSD80-SUR-015 used an acquisition-and-forgetting ODE. Historical routes
`prior_A`, `prior_J`, and `PLAFK` had already tested closely related retained
state and power-law acquisition-forgetting transitions. SUR-015 grafted that
transition onto the RPL response, so it was not algebraically identical. It
was a legitimate reopening that should have named the earlier routes and
explained why their StarCoder shape failures might not apply.

### Replay and finite-corpus terms

WSD80-SUR-019 tested phase-local replay, collision, and Jensen terms. The exact
factorization was new, but the mechanisms overlap historical routes `prior_B`,
`prior_T`, `prior_W`, `prior_Y`, `prior_AH`, `RMR`, `FSCR`, and `JARA`. The
logbook did not record that relationship before fitting.

## Controls That Worked

- WSD80-SUR-032 blocked the Fisher reopening before another outcome fit.
- WSD80-SUR-033 and SUR-036 failed outcome-free identification tests.
- WSD80-SUR-037 recognized that family churn reduced to an exposed Hellinger
  divergence and blocked it before fitting.
- WSD80-SUR-038 froze a Stage-0 directional test and rejected quality-pair
  churn before fitting its hazard amplitude.
- WSD80-SUR-040 through SUR-042 were independently reviewed and blocked before
  asymmetric BPB fitting.

These later entries show that provenance and mechanism-first review can stop
circular work when the historical relationship is made explicit.

## Root Cause

The 2026-07-29 handoff instructed the next session to read the WSD80 logbook and
`AGENTS.md`, but not either historical approach registry. The north-star
charter also contained only a generic instruction not to reproduce rejected
dose models. Neither file required a nearest-prior-route field before fitting.

Compaction cannot be proven as the cause of each reopening. The missing
registry link at the handoff boundary is nevertheless sufficient to explain
how already recorded evidence became unavailable to the next derivation.

## Required Preflight

Before fitting a new candidate:

1. Search the historical 99-route registry and the active registry.
2. Record the nearest prior route IDs.
3. State the exact new latent state, invariant, transition, response, or
   identification argument.
4. Explain why the prior rejection does not apply.
5. Specify an outcome-free or cheapest falsification test.
6. Block the route when only notation, divergence, base surrogate, coefficient
   grid, or output calibration changed.

The active registry is
`two_phase_surrogate_active_registry.csv`. The historical source is
`mechanistic_surrogate_discovery_20260719/approach_registry.csv`.
