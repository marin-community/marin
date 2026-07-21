# Debugging log for Marin Serve Levanter resource mapping

Restore Levanter serving through `remote_inference()` after automatic KV-cache sizing failed before the server started.

## Initial status

An Iris Levanter worker loaded its model, then failed in `InferenceEngine.from_model_with_config()` with
`ValueError: No resource mapping found`. The exception came from Haliax while estimating the per-device size of
an abstract KV cache.

## Hypothesis 1

`InferenceEngine.from_model_with_config()` accepts the compute `axis_resources` and uses them when allocating the
real cache, but `_infer_max_pages_from_hbm()` sizes the abstract cache without them. Sizing consequently depends on
an ambient thread-local mapping even though the public constructor was given an explicit mapping.

## Changes to make

- Add a public-constructor regression test with automatic page sizing, explicit axis resources, and no ambient
  Haliax mapping.
- Thread `axis_resources` into automatic page sizing and use it for the abstract cache byte calculation.

## Results

The regression test failed before the fix with the reported `ValueError: No resource mapping found` and passed after
`axis_resources` was threaded through automatic sizing. The complete non-slow Levanter inference suite passed: 46
tests passed and 1 was skipped.

## Future work

- [ ] Confirm the repaired branch starts the reported Levanter service on Iris.
