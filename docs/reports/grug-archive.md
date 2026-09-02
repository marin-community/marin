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

### moe-norm-preserving-residual
- Path: `experiments/grug/moe_norm_preserving_residual/`
- Origin: `experiments/grug/moe/` at canonical July Baseline commit `52d8a9eb8`
- Introduced: branch `codex/july-norm-preserving-residual`
- Last known-good: focused numerical validation on 2026-09-02
- Status: active
- Purpose: learn one positive norm-preserving residual-mixing coefficient per layer and evaluate Gate 1 against July Baseline #6882.
- Issue: #8860
- Diff: local report pending
