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

The TPU adapter enables `attn_res_remat_attention`: each attention branch is
recomputed during backward instead of retaining its intermediate tensors.
This adds one attention-branch forward per layer and preserves its computation.
The update diagnostic first compares loss and all gradients with attention
rematerialization enabled versus disabled on a reduced L6/d64/head128,
16-expert/top10, B128/S128 FP32 fixture. This fixture retains
KDA, MLA, Inkling, AttnRes and MoE, but does not establish full-shape parity.
After that check passes, the diagnostic attempts a complete update on the original
d512/B128/S4096 workload. It saves lowered MLIR before compilation and logs memory
analysis when compilation succeeds.

The full replication entrypoint is
`python -m experiments.grug.belay_tpu.belay_ckpt1_tpu`. Its `--run` flag starts
the original full schedule; `--profile-start-step 100 --profile-steps 10` enables
trace capture and compiled HLO output.

Set `BELAY_PACKED_KDA_CHECK=1` to compare chunked KDA with its sequential recurrence
before training, at head width 128 and with document boundaries inside and between
chunks. It checks outputs, final states and gradients of q, k, v, decay and beta.
Absolute and relative tolerances are both `1e-4` for FP32 and `2e-2` for BF16.

Set `BELAY_MEASURE_MFU=1` to measure zero-based update windows 200–499, 500–799
and 800–1099. This requires pipeline depth zero and the frozen Ckpt 1 model
configuration; a hash mismatch raises an error.
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
These measurement and packed-document checks have not yet run on TPU.

The Muon port matched the source per-matrix `_zeropower_via_newtonschulz_local`
reference exactly on three matrix fixtures
on all eight v4-64 processes. The following full-update compilation failed:
temporary buffers required 38.89G against 30.75G available HBM. The attention
rematerialization check, full updates, checkpoint continuation, the complete
training run and the target of over 20% useful-model MFU remain unvalidated.
Evidence and the MFU definition are tracked in
[research issue 253](https://github.com/WhenWen/astra-d8-research/issues/253).
