# Recipe: Write an Experiment Script

## Overview
Use this recipe when adding a new experiment script under `experiments/`.
The goal is consistent naming, searchable W&B metadata, and executor-friendly launch behavior.

## Naming Convention

1. If the work is tracked by an issue, name the file:
   - `experiments/exp<ISSUE_NUMBER>_<short_name>.py`
2. Use the same `exp<number>` token in run metadata:
   - W&B tags
   - run/group prefixes
   - experiment description

Example:
- Issue `2917` -> file `experiments/exp2917_shuffle_block.py`
- Required tag includes `exp2917`

## Script Skeleton

1. Define an issue constant at top-level.
2. Build run names from `exp<number>` + short name + date.
3. Tag every run with `exp<number>`.
4. Launch all arms in one `executor_main` call for ablations.

```python
ISSUE_NUMBER = 2917
EXP_TAG = f"exp{ISSUE_NUMBER}"
EXP_NAME = "shuffle_block"

run_date = os.environ.get("RUN_DATE", dt.date.today().isoformat())
run_prefix = f"{EXP_TAG}_{EXP_NAME}_{run_date}"

step = default_train(
    ...,
    tags=["ablation", EXP_TAG, ...],
    wandb_group=run_prefix,
    wandb_name=f"{run_prefix}-arm_name",
)
```

## Required Metadata

For each training step:
1. `tags` includes `exp<number>` (for example `exp2917`)
2. `wandb_group` set to a shared run prefix for the experiment
3. `wandb_name` unique per arm
4. `description` mentions the issue number

## Ablation Pattern

For multi-arm ablations:
1. Keep shared config in one place (model/train/data defaults)
2. Build arm-specific configs with a small helper
3. Generate `steps` list programmatically
4. Call `executor_main(steps=steps, ...)` once

This keeps runs comparable and avoids drift across arms.

## Do / Don’t

Do:
1. Use issue-based filenames and tags
2. Keep one source of truth for issue number and experiment short name
3. Prefer deterministic run naming (`RUN_DATE` override support)

Don’t:
1. Use unrelated labels (for example "ferry") for one-off ablations
2. Scatter issue number literals throughout the file
3. Launch separate scripts for each arm unless required

## Validation Checklist

1. `uv run python -m compileall experiments/exp<id>_<name>.py`
2. Confirm `tags` include `exp<id>`
3. Confirm run/group prefix contains `exp<id>`
4. Confirm script path follows `exp<id>_<name>.py`
