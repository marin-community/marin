# Grug Archive: Experiments and Snapshots

This file is the paper trail for grug experiments.

## Principles

- `experiments/grug/base/` is the canonical template.
- Speedrun files are exploratory and may be deleted after upstreaming.
- Prefer deletion over long-term maintenance of stale experiment code.

## Entry Template

```text
### <experiment-id>
- Path: <repo-relative-path>
- Introduced: <commit-sha>
- Last known-good: <commit-sha>
- Status: active | superseded | deleted
- Purpose: <one line>
- Superseded by: <path or commit; optional>
- Issue: <url/id; optional>
```

## Experiments

### grug-base-template
- Path: `experiments/grug/base/`
- Introduced: TBD
- Last known-good: TBD
- Status: active
- Purpose: canonical grug template (model/train/launch).

### grugformer-vs-hackable-125m
- Path: `experiments/speedrun/grugformer_vs_hackable_125m/grugformer_vs_hackable_125m.py`
- Introduced: TBD
- Last known-good: TBD
- Status: deleted
- Purpose: historical head-to-head comparison.
- Superseded by: template-first workflow centered on `experiments/grug/base/`.

### moe-relative-position
- Path: `experiments/grug/moe_relative_position/`
- Origin: `experiments/grug/moe/` at `ff3cb0282`
- Introduced: local branch `codex/inkling-relative-position`
- Last known-good: local validation on 2026-07-15
- Status: active
- Purpose: replace positional encoding with Inkling-style learned input-dependent relative attention bias and evaluate Gate 1 against the July Baseline in #6882.
- Diff: local report generated at `/private/tmp/grug-moe-relative-position-diff/index.html`; PR diff pending
