# Grug Layout and Usage

`experiments/grug/` is template-first. You edit experiment code here directly.

## Directory layout

- `base/model.py`: model config and model implementation (`init` + `__call__` + loss method).
- `base/train.py`: train loop, optimizer step, callbacks, eval/checkpoint wiring.
- `base/launch.py`: experiment config and execution entrypoint (`ExecutorStep` + resources).

## Entry-point guide

- Start in `base/launch.py` for normal run edits.
- `GrugBaseLaunchConfig` is the user-facing knob surface (model/data/optimizer/trainer/eval/run metadata).
- `versioned(...)` marks config values that should affect executor step version/hash.
- `this_output_path()` resolves to the current step's output root.
- `run_grug(...)` in `base/train.py` is the runtime entry point used by the `ExecutorStep`.

## How to use it

1. Copy `experiments/grug/base` to a new variant directory (for example `experiments/grug/moe`).
2. Make model/training changes in that variant, not in shared trainer libraries.
3. Set run knobs in `<variant>/launch.py` (run id, data mix, optimizer, TPU type).
4. Launch from the variant's `launch.py` entrypoint.

## Trainer knobs people ask about

- `z_loss_weight` in `GrugTrainerConfig`: weight on the logsumexp stabilization term in LM loss.
- `ema_beta` in `GrugTrainerConfig`: EMA coefficient for eval/checkpoint model; `None` disables EMA.

## What should stay consistent

- Keep core training/eval metrics aligned with classic Levanter (`train/loss`, `throughput/*`, `eval/*`).
- Prefer shared helpers only for generic infrastructure; keep variant behavior local to the template.

## Further guidance

- Grug principles: [`/.agents/projects/grugformer.md`](../../.agents/projects/grugformer.md)
- Change workflow: [`/docs/recipes/change_grug.md`](../../docs/recipes/change_grug.md)
