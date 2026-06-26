# Delphi true-midtraining configs

This directory is a planning surface for true cooldown midtraining on the
Nemotron math-only mix. It is intentionally not a launcher.

The key difference from CPT is checkpoint selection. CPT can derive its init
from the base model registry, but true cooldown must resume full trainer state
from one exact native pretraining checkpoint. A launcher must never infer that
checkpoint silently.

## Files

- `configs/checkpoint_candidates.yaml`: shared candidate table for every
  registered Delphi base from `3e18` through `1e22`, at cooldown ratios
  `0.30`, `0.20`, and `0.10`.
- `configs/p33m67.yaml`: 33% pretrain replay, 67% math.
- `configs/p50m50.yaml`: 50% pretrain replay, 50% math.
- `configs/p67m33.yaml`: 67% pretrain replay, 33% math.

Mix tags follow the existing convention: `p{pretrain}m{math}`.

## Review Contract

Every row in `checkpoint_candidates.yaml` starts with:

```yaml
review_status: needs_human_review
```

A future launcher should require `review_status: approved` before it stages or
launches a cell. This keeps the target step, closest checkpoint, before/after
bracket, and checkpoint path visible to the operator before any TPU job starts.

Generated checkpoint suggestions use:

```bash
uv run python scripts/list_delphi_checkpoints.py --all --cooldown-ratio <ratio>
```

The suggested checkpoint is only the closest available checkpoint by absolute
step distance. It is not a launch decision.
