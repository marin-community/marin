# Recipe: Changing Grug (Experiment → Canonical)

Grug is meant to be “grug-simple” and easy to hack, but we still want a single, trustworthy “best guess” implementation in `levanter.grug`.

This recipe describes the workflow for:

1) trying changes safely in a speedrun experiment, and
2) upstreaming successful ideas into the canonical core (and cleaning up old experiments).

## Source Of Truth vs Experiments

- **Source of truth:** `lib/levanter/src/levanter/grug/`
  - This is the “best guess” model. It should stay small, readable, and testable.
- **Evolving experiment:** `experiments/speedrun/grugformer_starter/grugformer_speedrun.py`
  - This is the *living* entrypoint that is expected to evolve as we learn.
- **One-off experiments:** under `experiments/speedrun/…`
  - These are snapshots / specialized edit surfaces (e.g. attention sinks).

We try not to let one-off scripts become the canonical implementation.

## When You Want To Try Something

### 1) Decide what you’re changing

Most changes fall into one bucket:

- **Attention** (masking semantics, kernels, sinks/aux, layout/sharding)
- **Block** (residual wiring, normalization order, pre/post-norm)
- **MLP** (activation, GLU variants, gating, dimension choices)
- **Loss** (large-vocab CE, z-loss, label smoothing, logit soft-cap)
- **Optimizer** (Adam, Muon, etc.)

Try to change **one bucket at a time**. Optimizer isn't really (currently) addressed by Grug, but we'll get there.

### 2) Create an experiment entrypoint

Start from:

- `experiments/speedrun/grugformer_starter/grugformer_speedrun.py`

Recommended workflow:

1. Copy the file to a new experiment (or branch the starter if the change is small):
   - Example: `experiments/speedrun/grugformer_<idea>/grugformer_<idea>.py`
2. Keep the edit surface explicit:
   - If you’re changing attention, keep the change in one local `attention()` or `attn_fn` block.
   - If you’re changing the MLP, keep it local to an `mlp()` helper.
3. Avoid introducing new abstractions (this is a speedrun file; copy/paste is fine).

### 3) Register the experiment in the archive

Add an entry to:

- `docs/reports/grug-archive.md`

Record:
- the experiment path,
- the commit SHA (once known),
- what you changed and why,
- the intended “status” (`active`, `superseded`, `deleted`).

## When You Want To Adopt Something As Canonical

### 1) Port to `levanter.grug`

Move the change into one of the core files:

- `lib/levanter/src/levanter/grug/attention.py`
- `lib/levanter/src/levanter/grug/model.py`
- `lib/levanter/src/levanter/grug/loss.py`

Keep the “grug” style:
- top-level functions,
- small dataclasses only for parameter/state containers,
- explicit sharding when needed (and loud failures otherwise).

### 2) Update/extend tests

Add or adjust tests to lock the intended surface:

- `lib/levanter/tests/test_grugformer_core.py`
- `lib/levanter/tests/test_grugformer_model_loss.py`
- `lib/levanter/tests/test_grugformer_fused_loss.py`

The goal is:
- shapes don’t regress,
- `jit` still works,
- sharding doesn’t explode,
- mask semantics remain correct.

### 3) Clean up old experiments

After merging a canonical improvement:

- If an experiment is now redundant and not referenced, **delete it** and mark it `deleted` in `docs/reports/grug-archive.md`.
- If an experiment represents a meaningful historical run, keep it but mark it `superseded`, and point to the canonical change (or the new experiment).
  Do this only if it's not going to be a maintenance burden.

Prefer “archive entry + deletion” over keeping lots of old code in-tree.

### 4) Run repo checks

Before sending the PR:

```sh
uv run python infra/pre-commit.py --all-files
```

## Notes / Inspiration

This workflow is inspired by projects like `modded-nanogpt`: keep a small, readable core, iterate quickly via “hackable” entrypoints, and regularly upstream what works.

