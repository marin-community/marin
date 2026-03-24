# Debugging log for ising tokenizer TPU mesh

Restore the TPU splash-attention path for the Ising tokenizer smoke run and make the `v5p-8` Iris job start cleanly.

## Initial status

The `us-central1-a` `v5p-8` job `/dlwh/ising-tokenizer-bkl-v5p-8-r1` reached the worker and failed during the launcher smoke pass with:

`RuntimeError: TPU splash attention requires a JAX mesh context.`

The traceback showed the call path from `experiments/ising_tokenizer/base/launch.py` into `experiments/ising_tokenizer/base/train.py`, where `_evaluate_dataset` invoked the jitted loss outside any mesh context. A temporary fallback to `reference_attention` avoided the crash locally but bypassed the intended TPU attention path.

## Hypothesis 1

The smoke loop is building and evaluating the model on TPU arrays that are not attached to a mesh or `NamedSharding`, so `levanter.grug.attention._tpu_splash_attention` rejects them before training starts.

## Changes to make

Update [experiments/ising_tokenizer/base/model.py](/Users/dlwh/.codex/worktrees/86f8/marin/experiments/ising_tokenizer/base/model.py) to restore `attention(...)`, and update [experiments/ising_tokenizer/base/train.py](/Users/dlwh/.codex/worktrees/86f8/marin/experiments/ising_tokenizer/base/train.py) to:

- create a simple repo-native `data` mesh with Levanter's mesh helper
- enter `haliax.partitioning.set_mesh(mesh)` for the full train/eval loop
- replicate model and optimizer state onto `NamedSharding`
- shard batches on the leading axis when divisible, otherwise replicate them

## Future Work

- [ ] Replace the ad hoc local smoke loop with a proper trainer path if this experiment graduates past proof-of-concept
- [ ] Add a TPU-only regression test once there is a stable lightweight way to exercise splash attention in CI
- [ ] Decide whether the Ising scaffold should adopt Grug parameter sharding instead of full replication

## Results

Patch applied locally. Pending verification: repo lint, scaffold tests, and a resubmitted `v5p-8` Iris run.
