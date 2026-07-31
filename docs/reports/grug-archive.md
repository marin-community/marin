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
- Origin: base | moe | <source variant; omit for entries predating the base template>
- Introduced: <commit-sha>
- Last known-good: <commit-sha; superseded and deleted entries only>
- Status: active | superseded | deleted
- Purpose: <one line>
- Superseded by: <path or commit; optional>
- Diff: <CI-posted grug diff comment, or local report path; optional>
- Issue: <url/id; optional>
```

Active entries track `main`, so they omit `Last known-good` rather than carry a SHA that
goes stale on the next commit.

## Experiments

### grug-base-template
- Path: `experiments/grug/base/`
- Origin: base
- Introduced: 8d752a775
- Status: active
- Purpose: canonical grug template (model/train/launch).

### grug-moe
- Path: `experiments/grug/moe/`
- Origin: base
- Introduced: 9181fd753
- Status: active
- Purpose: canonical Mixture-of-Experts variant; carries its own model, optimizer, train loop, and launch wiring so it can iterate independently of the dense template.
- Issue: https://github.com/marin-community/marin/pull/3046

### grugformer-vs-hackable-125m
- Path: `experiments/speedrun/grugformer_vs_hackable_125m/grugformer_vs_hackable_125m.py`
- Introduced: 5efe76834
- Last known-good: 5efe76834
- Status: deleted
- Purpose: historical head-to-head comparison.
- Superseded by: 8d752a775 — template-first workflow centered on `experiments/grug/base/`.
