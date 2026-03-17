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
