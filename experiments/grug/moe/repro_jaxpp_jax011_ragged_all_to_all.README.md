# JaxPP and JAX 0.11 ragged-all-to-all regression

This reproducer establishes the primitive-level boundary around a failure seen
when pinned JaxPP executes a JAX 0.11.1 nightly training graph containing
`jax.lax.ragged_all_to_all`. It uses the minimum two MPMD ranks. Each stage has
two H100s because the receiving stage needs a two-device collective axis.

The base cases use four `int32` values:

```text
[[0], [1], [100], [101]]
```

Stage 0 owns the payload with two rows per device and transfers it to stage 1.
The `jaxpp-transfer` control checks the transferred payload unchanged. The
`jaxpp-ragged` case sends one row from each source to each destination and
checks the exact result:

```text
destination 0: [[0], [100]]
destination 1: [[1], [101]]
```

Both cases require zero mismatches and checksum `202`. `direct-ragged` runs the
same collective on two H100s without JaxPP.

`jaxpp-four-stage-bf16-vjp-ragged` casts the same values to BF16 and removes the
Marin model, optimizer, and training state. Ranks 0-2 each run one
differentiable ragged forward task. Rank 3 runs a fourth ragged forward inside
a scalar mean-square loss and forms its VJP. Ranks 2-0 then run reverse VJP
tasks. The program has seven tasks, three forward DIME transfers, three reverse
DIME transfers, and no final scalar transfer. Rank 3 checks loss `5050.5`; rank
0 checks the exact input gradient `0.5 * input`.

## Pinned environment

- JAX, JAXLIB, CUDA 13 plugin, and PJRT: `0.11.1.dev20260725`
- NCCL Python wheel: `2.30.7`
- JaxPP: `7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9`
- JaxPP compatibility patch:
  `experiments/grug/moe/jaxpp_jax_0_11_inline.patch`
- jax-tvm-ffi: `e238a28483123efc8f56b9de358c2fb8b8de77e5`
- jax-tvm-ffi multi-device patch:
  `experiments/grug/moe/jax_tvm_ffi_multidevice.patch`

The script rejects any other package versions or an unpatched JaxPP runtime.

## H100 command

Run the six base cases and the 16-microbatch four-stage pair on one eight-H100 Iris task. The direct and transfer-only
controls must pass before interpreting the combined cases. The two four-stage
cases isolate rank count and bidirectional transfer structure before adding a
device ragged collective to every stage task. The BF16 VJP case then replaces
the synthetic reverse transform with reverse-mode differentiation.

```bash
STAMP=$(date -u +%Y%m%d-%H%M%S)
uv run --package marin-iris --extra controller iris --cluster=cw-rno2a \
  job run --no-wait --enable-extra-resources --gpu=H100x8 \
  --cpu=16 --memory=128G --disk=128G --extra=gpu --timeout=900 \
  --max-retries=0 --priority=interactive \
  --job-name="jaxpp-jax011-ragged-minimal-${STAMP}" \
  -- bash -c '
    set -euxo pipefail
    cd "$IRIS_WORKDIR"
    JAX_VERSION=0.11.1.dev20260725
    JAX_INDEX=https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
    JAXPP_REVISION=7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9
    JAX_TVM_FFI_REVISION=e238a28483123efc8f56b9de358c2fb8b8de77e5

    uv pip install --link-mode=symlink --prerelease=allow --index "$JAX_INDEX" \
      "jax==$JAX_VERSION" "jaxlib==$JAX_VERSION" \
      "jax-cuda13-plugin[with-cuda]==$JAX_VERSION" \
      "jax-cuda13-pjrt==$JAX_VERSION" "nvidia-nccl-cu13==2.30.7"
    uv pip install --link-mode=symlink cupy-cuda13x

    rm -rf /tmp/jax-tvm-ffi /tmp/jaxpp
    git clone --quiet --filter=blob:none \
      https://github.com/NVIDIA/jax-tvm-ffi.git /tmp/jax-tvm-ffi
    git -C /tmp/jax-tvm-ffi checkout --quiet "$JAX_TVM_FFI_REVISION"
    git -C /tmp/jax-tvm-ffi apply \
      "$IRIS_WORKDIR/experiments/grug/moe/jax_tvm_ffi_multidevice.patch"
    uv pip install --link-mode=symlink --force-reinstall --no-deps \
      /tmp/jax-tvm-ffi

    git clone --quiet --filter=blob:none https://github.com/NVIDIA/jaxpp.git \
      /tmp/jaxpp
    git -C /tmp/jaxpp checkout --quiet "$JAXPP_REVISION"
    git -C /tmp/jaxpp apply --unidiff-zero \
      "$IRIS_WORKDIR/experiments/grug/moe/jaxpp_jax_0_11_inline.patch"
    uv pip install --link-mode=symlink --force-reinstall --no-deps /tmp/jaxpp

    export JAXPP_SOURCE=/tmp/jaxpp
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.35
    export XLA_FLAGS="--xla_gpu_autotune_level=0 \
      --xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true \
      --xla_gpu_ragged_all_to_all_mode=symmetric \
      --xla_enable_nccl_symmetric_buffers_for_collectives=RaggedAllToAll \
      --xla_gpu_nccl_termination_timeout_seconds=120"
    SCRIPT=experiments/grug/moe/repro_jaxpp_jax011_ragged_all_to_all.py

    CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -u "$SCRIPT" \
      --case direct-ragged --timeout 180 --stack-after 30
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -u "$SCRIPT" \
      --case jaxpp-transfer --coordinator-port 5831 \
      --timeout 180 --stack-after 30
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -u "$SCRIPT" \
      --case jaxpp-ragged --coordinator-port 5832 \
      --timeout 180 --stack-after 30
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python -u "$SCRIPT" \
      --case jaxpp-four-stage-transfer --coordinator-port 5833 \
      --timeout 180 --stack-after 30
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python -u "$SCRIPT" \
      --case jaxpp-four-stage-ragged --coordinator-port 5834 \
      --timeout 180 --stack-after 30
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python -u "$SCRIPT" \
      --case jaxpp-four-stage-bf16-vjp-ragged --coordinator-port 5837 \
      --timeout 180 --stack-after 30
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python -u "$SCRIPT" \
      --case jaxpp-four-stage-transfer --microbatches 16 \
      --coordinator-port 5835 --timeout 300 --stack-after 30
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python -u "$SCRIPT" \
      --case jaxpp-four-stage-ragged --microbatches 16 \
      --coordinator-port 5836 --timeout 300 --stack-after 30
  '
```

Each process emits JSON lines for environment validation, phase entry and
return, watchdog stacks, exact checks, barriers, shutdown, and worker exit.
Exit `124` is the watchdog timeout. A native signal is reported by the parent
as the corresponding shell exit code, such as `139` for `SIGSEGV`.

## Result

Corrected job
`/dlwh/jaxpp-jax011-ragged-minimal-r2-20260726-121552` succeeded on
`cw-rno2a` in one attempt:

- `direct-ragged` returned in `0.863855s` with zero mismatches and checksum
  `202`.
- `jaxpp-transfer` returned from `eval_local` in `1.242563s` on rank 0 and
  `2.206070s` on rank 1. The transferred payload had zero mismatches and
  checksum `202`.
- `jaxpp-ragged` returned from `eval_local` in `1.425364s` on rank 0 and
  `2.447640s` on rank 1. The exchanged payload had zero mismatches and checksum
  `202`.
- No watchdog, phase failure, nonzero exit, retry, or signal occurred.

The four-stage extensions also succeeded:

- `/dlwh/jaxpp-jax011-ragged-four-stage-r1-20260726-124203` passed the
  one-microbatch transfer and ragged cases exactly with checksum `202`.
- `/dlwh/jaxpp-jax011-ragged-four-stage-m16-r2-20260726-124630` passed both
  16-microbatch cases exactly with checksum `3,232`. The transfer control
  completed 96 logical transfers, and the ragged treatment completed 128 stage
  tasks.
- Every rank returned from initialization, lowering, evaluation, barrier, and
  shutdown. Neither job emitted a watchdog stack, phase failure, nonzero exit,
  retry, or signal.

The primitive compositions do not reproduce the L8 training deadlock, even
with four bidirectional ranks, repeated receive-buffer reuse, 16 microbatches,
96 transfers, and 128 stage tasks.
The smallest known failing programs remain the four-stage L8/d2560/e64/top-k4
jobs `/dlwh/iris-run-job-20260726-111423` (GPipe) and
`/dlwh/iris-run-job-20260726-113121` (standard 1F1B with transfer priority).
Both use the pinned environment above and stop with JaxPP transfer ranks in
`ncclGroupEnd` or receive waits while another rank remains in compilation or
PJRT execution. Neither emits a finite loss.

This test is the lower-bound regression package: it verifies package identity,
the JAX 0.11 inline patch, direct ragged-all-to-all, JaxPP task transfer, and
their smallest composition. Any smaller case would remove either an MPMD stage
or the two-device collective axis.

The four-stage transfer control runs four two-H100 MPMD ranks through three
forward transfers and three reverse transfers. Each rank executes one forward
and one backward stage task. The ragged treatment replaces every identity stage
task with an exact two-device ragged exchange. Eight exchanges restore the
original payload. The `--microbatches 16` pair repeats the full chain,
accumulates all returned payloads on stage 0, and forces 96 transfers plus 128
stage tasks to remain live. The base cases require checksum `202`; the repeated
cases require checksum `3,232`. Every case requires zero mismatches.

The standalone BF16 VJP reduction also passed in
`/dlwh/jaxpp-bf16-vjp-ragged-min-r1-20260726-1812`:

- Rank 3 produced exact loss `5050.5` with absolute error `0.0`.
- Rank 0 produced the exact `0.5 * input` gradient with maximum absolute error
  `0.0`.
- All four ranks lowered in `0.032-0.033s` and returned from `eval_local` in
  `6.108-6.972s` without a watchdog stack or timeout.

Therefore differentiable BF16 ragged forward/VJP tasks and bidirectional DIME
transfers are not sufficient to reproduce the training deadlock.

Part of #7024.
