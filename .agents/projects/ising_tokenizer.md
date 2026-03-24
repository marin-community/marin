# Ising Tokenizer Project: Repo-Aligned Implementation Plan

## Research Framing

This is a controlled test of the TATT-style claim that trajectory statistics can
carry the physics strongly enough for a tokenized autoregressive model to
recover meaningful structure without explicit Hamiltonian-side supervision.

The 2D Ising model is the right first physics probe because:

- the equilibrium story is known,
- the dynamics are cheap to simulate,
- the critical point is known exactly,
- the phase transition gives a sharp interpolation test.

The project is not mainly about building the best Ising simulator. It is about
testing whether a simple sequence model can absorb phase information from the
natural event language of the dynamics.

## Goal

Build a small, explicit experiment stack for continuous-time Ising trajectories
that fits the repo's grug-style conventions:

- explicit experiment files,
- deterministic synthetic data,
- minimal model changes,
- no framework detour,
- fast local iteration before cluster work.

The first question is not "can we publish Tc recovery?" It is:

> can a tiny temperature-conditioned transformer over tokenized BKL trajectories
> learn nontrivial off-critical dynamics on synthetic data at all?

## V0 Principles

1. Keep the physics path simple.
   Use single-spin-flip continuous-time dynamics with a naive full-lattice rate
   recompute. No clever event tree in the first pass.

2. Keep the model gruggy.
   Reuse the grug block stack. Fork only the part needed to inject continuous
   temperature conditioning.

3. Keep the tokenization inspectable.
   Initial state is explicit `[pos][spin]` pairs. Dynamics are explicit
   `[pos][dt_bin]` pairs. No hidden compression tricks.

4. Keep scope local first.
   Use synthetic local runs and focused tests before executor integration or
   larger sweeps.

5. Keep the result falsifiable.
   First gate is not Tc generation. First gate is whether seen and held-out
   off-critical temperatures are learned cleanly.

## V0 Compromises

The first pass makes a few deliberate simplifications:

- waiting times are log-binned into discrete `dt` tokens,
- temperature is injected as a scalar residual modulation, not a richer MLP,
- the trainer is local and lightweight, not yet a Marin executor job,
- evaluation is teacher-forced likelihood first, rollout behavior later.

These are acceptable as long as they stay explicit and easy to replace.

## Proposed Layout

```text
experiments/ising_tokenizer/
  README.md
  base/
    data.py
    model.py
    train.py
    launch.py
```

Notes:

- `base/` is the canonical simple path.
- Future variants should start as copies of `base/`.
- If a change wins, upstream it back into `base/`.

## Initial Scope

### In scope

- tiny rejection-free 2D Ising trajectory generation,
- deterministic tokenization to fixed-length sequences,
- continuous temperature conditioning,
- local smoke training on off-critical temperatures,
- held-out off-critical and near-critical teacher-forced evaluation.

### Out of scope for V0

- cluster-scale executor integration,
- critical exponent estimation,
- rollout sampling studies,
- learned `dt` quantizers,
- large-scale scaling sweeps,
- fancy BKL acceleration data structures.

## Dataset Plan

Use a tiny synthetic ladder:

- train temperatures:
  `1.5`, `1.8`, `2.8`, `3.1`
- held-out off-critical validation temperatures:
  `1.6`, `2.9`
- near-critical probe:
  `Tc = 2.26918531421`

First local shape:

- lattice size:
  `8 x 8`
- burn-in events:
  `96`
- recorded events:
  `48`
- sequence length:
  `2 * 64 + 2 * 48 + 1 = 225`

## Success Gates

Phase 0 is successful if all of the following are true:

1. synthetic trajectory generation is deterministic under a fixed seed,
2. token IDs stay within the declared vocabulary,
3. the conditioned transformer runs end to end,
4. local train loss decreases on a small smoke,
5. held-out off-critical loss is finite and not obviously broken,
6. the code stays small and readable.

## Immediate Next Steps

1. Add the local scaffold and tests.
2. Run a tiny synthetic smoke.
3. Record the first result in a research logbook.
4. Decide whether the next move is:
   local rollout evaluation, better `dt` bins, or Marin/executor wiring.
