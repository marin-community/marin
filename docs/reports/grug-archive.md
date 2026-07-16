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

### grug-ve-value-embeddings
- Path: `experiments/grug/ve/`
- Origin: `experiments/grug/base/`
- Introduced: TBD
- Last known-good: TBD
- Status: active
- Purpose: value embeddings — a layer-shared, token-indexed table blended into the attention
  value path under a learnable per-layer gate, buying parameters at zero added matmul FLOPs.
  Paired A/B (`--arm ve` vs `--arm base`) at 130M, confirming at 1.3B, on a single H200.

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
