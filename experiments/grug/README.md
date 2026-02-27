# Grug Layout and Usage

`experiments/grug/` is template-first. You edit experiment code here directly.

## Directory layout

- `base/model.py`: model config and model implementation (`init` + `__call__` + loss method).
- `base/train.py`: train loop, optimizer step, callbacks, eval/checkpoint wiring.
- `base/launch.py`: experiment config and execution entrypoint (`ExecutorStep` + resources).

## How to use it

1. Copy `experiments/grug/base` to a new variant directory (for example `experiments/grug/moe`).
2. Make model/training changes in that variant, not in shared trainer libraries.
3. Set run knobs in `<variant>/launch.py` (run id, data mix, optimizer, TPU type).
4. Launch from the variant's `launch.py` entrypoint.

## What should stay consistent

- Keep core training/eval metrics aligned with classic Levanter (`train/loss`, `throughput/*`, `eval/*`).
- Prefer shared helpers only for generic infrastructure; keep variant behavior local to the template.

## Further guidance

- Grug principles: [`/.agents/projects/grugformer.md`](../../.agents/projects/grugformer.md)
- Change workflow: [`/docs/recipes/change_grug.md`](../../docs/recipes/change_grug.md)
