# Belay Ckpt 1 on TPU

This variant copies `fast_track` from commit `26921126e` on the local Marin
branch `codex/belay-ckpt1-tpu`. Its model and recipe originate at upstream Marin
commit `8415c3ce8411051c1e22bc13184c272b88ec2b6c`. It targets the original Belay
Ckpt 1 training recipe: width 512, six layers, global batch 128, sequence length
4,096 and 8,821 updates from initialization.

The optimizer uses batched JAX products for `X @ X.T` and `A @ A.T` in the source
BF16 Newton–Schulz iteration. This replaces the GPU-only QuACK/CUTLASS Gram
kernel. The coefficients, iteration count, normalization and MuonH update remain
unchanged. Module imports and the optimizer registration name are local to this
variant. The original `fast_track` optimizer is preserved for comparison.

Run the full-model continuation diagnostic with
`python -m experiments.grug.belay_tpu.belay_ckpt1_tpu_update_check` on TPU using
`JAX_PLATFORMS=tpu,cpu`, `BELAY_TPU=v4-64` and a fresh `BELAY_RUN_ID`. It first
compares the batched Newton–Schulz operator against the source's separate JAX
per-matrix reference on TPU. It then saves a newly trained checkpoint after one
update and requires exact equality of the next batch, state and metrics between
continuous training and a restored replay. It does not restore an upstream
checkpoint artifact.
The CPU backend is required by the native data shuffler.

The full replication entrypoint is
`python -m experiments.grug.belay_tpu.belay_ckpt1_tpu`. Its `--run` flag starts
the original full schedule; `--profile-start-step 100 --profile-steps 10` enables
trace capture and compiled HLO output.

TPU operator parity, full updates, checkpoint continuation, the complete training
run and the target of over 20% useful-model MFU against physical BF16 peak are
unvalidated for this variant. Evidence and the MFU definition are tracked in
[research issue 253](https://github.com/WhenWen/astra-d8-research/issues/253).
