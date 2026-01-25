# Grug Archive: Experiments and Snapshots

This file is a lightweight “paper trail” for Grug-related experiments, inspired by the idea of keeping a runnable history without letting a pile of one-off scripts become the de facto source of truth.

## Principles

- **`levanter.grug` is the source of truth.** Speedrun files are snapshots/entrypoints, not the canonical implementation.
- **Every experiment should be attributable to a commit.** If an experiment is removed or superseded, it should be clear what replaced it and why.
- **Prefer deletion over permanent snapshots.** If a script is dead, delete it and record the last known-good commit here.
- **Keep diffs small.** When an experiment is kept “alive”, update it to track the current core rather than forking the entire model.

## When Grug Core Changes

When a change in `levanter.grug` is likely to affect results, performance, or semantics:

1. Update the experiment(s) that should track “best guess”.
2. For experiments that no longer make sense:
   - delete them, or
   - mark them superseded and point to the replacement.
3. Update the corresponding entry in this archive (and any linked issue).

## Entry Template

Copy/paste this block for new experiments:

```text
### <experiment-id>
- Path: `<repo-relative-path>`
- Introduced: <commit-sha>
- Last known-good: <commit-sha>
- Status: active | superseded | deleted
- Purpose: <one line>
- Notes: <optional; what changed, how to reproduce, caveats>
- Superseded by: <experiment-id or commit-sha; optional>
- Issue: <url or issue id; optional>
```

## Experiments

### grugformer-attnsink
- Path: `experiments/speedrun/grugformer_attnsink/grugformer_attn_sink.py`
- Introduced: TBD
- Last known-good: TBD
- Status: active
- Purpose: “Hackable” Grug attention-sink variant; intended edit surface for sinks/aux.
- Notes: Keep this file short; copy/paste local modifications rather than growing new abstractions.

### grugformer-starter-speedrun
- Path: `experiments/speedrun/grugformer_starter/grugformer_speedrun.py`
- Introduced: TBD
- Last known-good: TBD
- Status: active
- Purpose: Minimal starter speedrun for Grug; convenient baseline for quick iteration.

### grugformer-vs-hackable-125m
- Path: `experiments/speedrun/grugformer_vs_hackable_125m/grugformer_vs_hackable_125m.py`
- Introduced: TBD
- Last known-good: TBD
- Status: active
- Purpose: Head-to-head comparison between Hackable Transformer and Grugformer (no sinks).

