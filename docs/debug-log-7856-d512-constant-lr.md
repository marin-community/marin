# Debugging log for issue #7856 d512 constant-LR TPU runs

Launch the 25 d512 constant-LR cells on TPU without changing the historical
datakit mixture.

## Initial status

`AUG-LRC-TPU-003-d512-30x-lr1` reached a v4-8 and created its W&B run, then
failed before model initialization. `c01q1` had no source or cache because the
historical relative prefix `datakit/store_8ac06c74` resolved under `/app`.

## Hypothesis 1

`mirror://datakit/store_8ac06c74` will resolve the cache in the worker's local
Marin bucket.

## Changes to make

Pass the datakit store prefix into `_datakit_data_config` and use the mirror
prefix only in the new TPU launcher. Keep the historical launcher's relative
prefix unchanged.

## Results

The corrected attempt loaded all 200 cache ledgers from the local mirror. It
then failed while reading the first shard because TensorStore only accepts its
native `gs`, `s3`, and local URI schemes:

```text
ValueError: Unsupported URI scheme for tensorstore: 'mirror' in
'mirror://datakit/store_8ac06c74/cluster=1/quality=3/.../input_ids/offsets'
```

No model initialization, optimizer step, loss, or checkpoint occurred in
either attempt.

## Hypothesis 2

The explicit in-region URI
`gs://marin-us-central2/datakit/store_8ac06c74` will work for fsspec metadata
and TensorStore shard reads. Training is pinned to `us-central2-b`, so this is
not a cross-region read.

## Changes to make

Use the explicit GCS prefix in the TPU launcher and add a focused test for the
materialized `c01q1` cache URI.

## Results

Pending live verification on `AUG-LRC-TPU-003-d512-30x-lr1`.

## Future work

- [ ] Start the remaining 24 cells only after the representative run reports
  finite loss and constant post-warmup LR.
