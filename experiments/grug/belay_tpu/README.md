# Belay Ckpt 1 on TPU

This variant copies `fast_track` from commit `26921126e` on the local Marin
branch `codex/belay-ckpt1-tpu`. Its model and recipe originate at upstream Marin
commit `8415c3ce8411051c1e22bc13184c272b88ec2b6c`. It targets the original Belay
Ckpt 1 training recipe: width 512, six layers, global batch 128, sequence length
4,096 and 8,821 updates from initialization.
This checkpoint recipe has no copy head.

The optimizer uses batched JAX products for `X @ X.T` and `A @ A.T` in the `fast_track`
BF16 Newton–Schulz iteration. This replaces the GPU-only QuACK/CUTLASS Gram
kernel. The coefficients, iteration count, normalization and MuonH update remain
unchanged. Module imports and the optimizer registration name are local to this
variant. The original `fast_track` optimizer is preserved for comparison.

Run the full-model continuation diagnostic with
`python -m experiments.grug.belay_tpu.belay_ckpt1_tpu_update_check` on TPU using
`JAX_PLATFORMS=tpu,cpu`, `BELAY_TPU=v4-64`, a fresh `BELAY_RUN_ID`, and
`--xla_allow_excess_precision=false` in `XLA_FLAGS`. The rounding flag is required
by the TPU training and validation entrypoints. It preserves declared BF16
rounding across forward and backward compiler fusions. The diagnostic first
compares the batched Newton–Schulz operator against `fast_track`'s separate JAX
per-matrix reference on TPU. It then saves a newly trained checkpoint after one
update and requires exact equality of the restored state and next data batch.
Executing that next update on both paths must also produce identical state and
metrics. It does not restore an upstream
checkpoint artifact.
The CPU backend is required by the native data shuffler.

The adapter sets `attn_res_remat_attention=False`. The update diagnostic attempts
the original d512/B128/S4096 workload, saves lowered MLIR before compilation, and
logs memory analysis when compilation succeeds. `--compile-only` stops after
compilation without updating parameters.

The candidate kernels retain forward intermediates for backward. `kda_tpu.py`
uses the source 16-token chunk preparation and a Pallas state pass ported from
`belay-base16-tpu-v4` commit `aef63f29d`. It saves FP32 chunk states and corrected
values. `inkling_tpu.py` and `inkling_pallas.py` implement Inkling attention with
saved FP32 probabilities and aligned bias slices. They avoid the source's dense
bias gather indices. `model.py` selects saved-state Pallas KDA and saved-probability
XLA attention on TPU. XLA attention used less temporary memory and was slightly
faster than the Pallas candidate at the target shape. CPU and GPU retain their
existing paths. The full-update diagnostic now exercises these replacements.

The `kernel_check`, `attention_kernel_check`, and `distributed_kernel_check`
modules in this package validate values and gradients without optimizer updates.
The distributed check uses the actual v4-64 mesh and checks explicit `shard_map`
boundaries. Numerical failures remain failures after the harness records the
remaining cases. `attention_kernel_check --benchmark-source` also remeasures the
dense `reference_attention` baseline; its values and gradients are always evaluated.

The full replication entrypoint is
`python -m experiments.grug.belay_tpu.belay_ckpt1_tpu`. Its `--run` flag starts
the original full schedule; `--profile-start-step 100 --profile-steps 10` enables
trace capture and compiled HLO output.

Set `BELAY_PACKED_KDA_CHECK=1` to compare chunked KDA with its sequential recurrence
before training, at head width 128 and with document boundaries inside and between
chunks. It checks outputs, final states and gradients of q, k, v, decay and beta.
Absolute and relative tolerances are both `1e-4` for FP32 and `2e-2` for BF16.

Set `BELAY_MEASURE_MFU=1` to measure zero-based update windows 200–499, 500–799
and 800–1099. `belay_ckpt1_tpu.build_config` supplies the frozen recipe and retains
`config.trainer.pipeline_depth=0`. `useful_mfu.MODEL_CONFIG_SHA256` identifies the
accepted model configuration; a hash mismatch raises an error.
Each window waits for all model/optimizer state and synchronizes all processes at
both endpoints. The reported duration is the maximum across processes. It includes
optimizer work, communication, host callbacks and the counting instrumentation.
Native step timing ends when the loss is ready and does not explicitly drain
optimizer state. The numerator counts selected dense matrices, expert products
retained after observed routing drops, and causal-attention products at valid
queries. Only the numerator excludes padding, recomputation and the omitted
operations listed in `useful_mfu.py`. Dividing by the measured time and the BF16
peak of all allocated v4 chips gives a conservative useful-MFU lower bound.
Per-update counts, per-process window times and available current/peak memory
usage and memory limits are saved as `useful-mfu-rankN.jsonl` under `IRIS_OUTPUT_DIR`.
Window metrics also go to W&B. The launcher uses one process per TPU host.
The MFU measurement has not yet run during training.

The Muon port matched the source per-matrix `_zeropower_via_newtonschulz_local`
reference exactly on three matrix fixtures on all eight v4-64 processes.
The source packed-KDA check also passed on those processes. These results preceded
the declared-rounding requirement and must be rechecked under it.
The full-update compiler report identified a replicated
`s32[128,4,4096,4096]` Inkling index occupying 32 GiB per chip. Total temporary
buffers required 38.89 GiB against 30.75 GiB available HBM.

Seven CPU attention operator tests pass with declared rounding. With the same
flag, `attention-v6` passes all 16 attention value/gradient comparisons, including
the 4096-token shape. `kernel-v3` passes all 12 candidate-versus-source KDA
comparisons. Its separate FP32 recurrent-oracle gate still fails for both the
source and candidate in two BF16 cases (`repeated` and `ckpt1_packed`, beta
gradients); that run remains failed. Source equivalence does not establish that
the BF16 source meets the oracle bound. `distributed-v1` passed on all eight
v4-64 processes with the actual EP16/model2 mesh.
Full updates, checkpoint continuation, the complete training run and
over 20% useful-model MFU remain unvalidated.
Evidence and the MFU definition are tracked in
[research issue 253](https://github.com/WhenWen/astra-d8-research/issues/253).
