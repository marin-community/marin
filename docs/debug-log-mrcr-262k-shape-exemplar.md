# Debugging log for the 262K MRCR shape exemplar

The bounded 262K MRCR probe must restore only model parameters under alternative evaluation meshes.

## Initial status

The `v4-32-cp4-fsdp` probe failed before checkpoint restore because the TPU runtime's
`ShapeDtypeStruct` did not expose the optional `manual_axis_type` attribute.

## Hypothesis 1

The context-sharding rewrite copied optional reference/manual-axis metadata that ordinary model
parameter exemplars do not use. The deployed JAX constructor accepts those fields, but its shape
objects do not consistently expose them as readable attributes.

## Changes to make

Construct the rewritten parameter exemplar from the required shape, dtype, sharding, and weak-type
fields only. Add a regression using a minimal shape object without the optional attributes.

## Results

The focused evaluator regression passes without reading `manual_axis_type` or `is_ref`. The active
`v4-64-cp8-ep4` probe does not use the generic context-sharding rewrite and was not restarted.

## Future work

- [ ] Exercise the generic `v4-32-cp4-fsdp` fallback only if the proven CP8/EP4 path cannot run.
