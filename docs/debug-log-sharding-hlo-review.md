# Debugging log for sharding/HLO review

Review the actual sharding dumps and HLO artifacts for the `v5p-8` vs `v6e-8`
LoRA DPO investigation, without relying on the existing logbook analysis.

## Initial status

The repository contains experiment scripts and a long logbook claiming:

- sharding layouts match between `v5p-8` and `v6e-8`
- HLO is identical except for collective width
- the bad `v5p-8` behavior is tied to the width-4 collective path

The user asked for an independent review of the dumped sharding and HLO.

## Hypothesis 1

The existing sharding summary may be overstated or incorrect; inspect the actual
dumped `DEBUGJ SHARDING` output directly.

## Changes to make

None yet. Read artifacts and compare against code paths.

## Future Work

- [ ] Confirm the exact LoRA param and optimizer-state shardings on both TPU types
- [ ] Confirm whether any array-level sharding differs beyond mesh width
- [ ] Confirm which collectives appear in optimized HLO and with what dtypes
- [ ] Check whether the HLO evidence directly supports the strongest causal claims

## Results

### Sharding dump

I fetched the actual `output.log` files from the two Exp Y runs and inspected the
raw `DEBUGJ SHARDING` lines emitted by
`lib/levanter/src/levanter/main/train_dpo.py`.

What the dump proves:

- The LoRA param and optimizer-state `PartitionSpec`s match exactly between
  `v5p-8` and `v6e-8`.
- The mesh-level mappings printed in `DEBUGJ SHARDING_MESH` also match:
  `param_mapping={'mlp': 'model', 'heads': 'model', 'embed': 'data'}` and
  `compute_mapping={'mlp': 'model', 'heads': 'model', 'batch': ('replica_dcn',
  'replica', 'data'), 'token': ('replica_dcn', 'replica', 'data'),
  'token_repeat': ('replica_dcn', 'replica', 'data')}`.
- The only difference in the dump is the actual mesh size on the `data` axis:
  `{'data': 4, ...}` on `v5p-8` versus `{'data': 8, ...}` on `v6e-8`.

Important nuance:

- The debug hook prints the array's global `shape` plus its `PartitionSpec`.
  It does not print the local per-device shard shape.
- Because the `data` mesh size differs, the same `PartitionSpec(..., 'data')`
  implies different local shard sizes across the two runs.
- So it is correct to say the logical sharding layout matches, but incorrect to
  infer from that that the optimized HLO should be identical.

### HLO dump

I compared the downloaded
`after_optimizations_before_buffer_assignment` dumps for the two Exp Z4 runs.

What the dump proves:

- The `psum` reduction region attributed to
  `lib/haliax/src/haliax/partitioning.py:909` is a `bf16` scalar add on both
  TPU types.
- The LoRA-gradient tuple all-reduce in the transpose/JVP path exists on both
  TPU types with the same logical tuple structure, but different replica-group
  width:
  - `v5p-8`: `replica_groups=[1,4]<=[4]`
  - `v6e-8`: `replica_groups=[1,8]<=[8]`
- The same "contract embed -> batch, position, LORA_R" logical operation uses
  different local shard sizes before all-gather:
  - `v5p-8`: local input `bf16[1,64,1024]`, then all-gather to
    `bf16[1,64,4096]`
  - `v6e-8`: local input `bf16[1,64,512]`, then all-gather to
    `bf16[1,64,4096]`

What the dump does not support:

- The strong claim that the optimized HLO is "the same except for
  `replica_groups`" is too strong for these exact artifacts.
- The two optimized modules differ substantially in collective lowering:
  `v5p-8` has many more explicit `all-reduce` ops, while `v6e-8` has many more
  `collective-permute` ops.
- This is consistent with the different local shard sizes implied by the same
  `PartitionSpec` under `data=4` versus `data=8`.

### Best current reading

The independent read is:

- I buy the narrow sharding claim: the LoRA layout is logically the same across
  `v5p-8` and `v6e-8`, and the primary sharding-level difference is the width of
  the `data` axis.
- I buy the narrow HLO claim: the relevant LoRA/data-axis reductions are `bf16`
  reductions on both runs, and the corresponding tuple all-reduce is width 4 on
  `v5p-8` versus width 8 on `v6e-8`.
- I do not buy the stronger claim that the optimized HLO is otherwise identical.
  The compiler lowers the two cases differently, and that is expected once the
  local shard sizes differ.

So the surviving conclusion is:

- The evidence supports "same logical sharding, different data-axis width, and a
  matched `bf16` LoRA-gradient all-reduce whose width changes from 4 to 8."
- The evidence does not, by itself, prove that width-4 `bf16` reduction is the
  sole cause of the training pathology. That remains a plausible mechanism, not
  a closed causal proof from HLO alone.
