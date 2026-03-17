# Grug Variant Notes

Use this file for variant-specific guidance and examples.

## `modular_opt` (`issue #3075`)

`experiments/grug/modular_opt/launch.py` is a concrete template example for [issue #3075](https://github.com/marin-community/marin/issues/3075):

- Uses `optax.multi_transform` to route parameter groups by path pattern.
- Supports parameter-level differences in:
  - learning-rate multiplier
  - weight-decay multiplier
  - Adam `beta1`/`beta2`
- Avoids per-layer re-specification by applying shared path-pattern rules (`embed`/`lm_head`, `scalar`/`gate`/`lambda`, no-decay patterns).

Launch it with:

```bash
uv run python experiments/grug/modular_opt/launch.py
```

Default optimizer class:

- `GrugParamGroupAdamConfig` in `experiments/grug/modular_opt/launch.py`

## `array_stacked` (`issue #3714`)

`experiments/grug/array_stacked/` is the base-template variant that swaps the
transformer block container from a Python tuple loop to
`haliax.nn.ArrayStacked`.

- `model.py`: uses `ArrayStacked.fold_via(...)` for block execution.
- `train.py`: mirrors base train-loop wiring with variant-local model import.
- `launch.py`: runnable trial entrypoint for this variant.

Launch it with:

```bash
uv run python experiments/grug/array_stacked/launch.py
```
