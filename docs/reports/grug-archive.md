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

### grug-1p5b-array-vs-base-h2h-launcher
- Path: `scripts/experiments/grug/launch_1p5b_h2h.py`
- Introduced: `HEAD`
- Last known-good: `HEAD`
- Status: active
- Purpose: ad-hoc launcher for 1.5B base vs array-stacked comparison runs.
- Issue: https://github.com/marin-community/marin/issues/3714
- W&B (base): https://wandb.ai/marin-community/marin/runs/grug-base-h2h-iris-v5p8-20260316-182608
- W&B (array): https://wandb.ai/marin-community/marin/runs/grug-arraystacked-h2h-iris-v5p8-20260316-182608
