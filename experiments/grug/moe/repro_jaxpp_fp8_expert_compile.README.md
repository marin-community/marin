# JaxPP FP8 expert-backward compile reproducer

This bounded reproducer isolates the compile stall seen in
`grug_1f1b_mb0_stage3_loss_backward`. It retains one grouped expert MLP, the
real Haliax Mosaic FP8 ragged GEMMs and their delayed-scaling overwrite state,
`value_and_grad` over parameters and the incoming activation, and optional
microbatch gradient accumulation. It removes attention, routing, expert
parallel collectives, optimizer state, and the language-model head.

Nothing has been filed upstream. The H100 result still needs to be collected.

## Smallest RNO2A allocation

This exact command requests one fractional two-H100 allocation and runs the
wrapper control, the single-process FP8 control, the matched two-rank direct
control, and the JaxPP FP8 case. It intentionally does not submit unless run by
the parent investigation owner.

```bash
uv run --package marin-iris --extra controller iris --cluster=cw-rno2a \
  job run --no-wait --enable-extra-resources --gpu=H100x2 \
  --cpu=16 --memory=128G --disk=128G --extra=gpu --timeout=2400 \
  --job-name=jaxpp-fp8-expert-compile-minimal \
  -- bash -lc '
    set -euxo pipefail
    uv pip install --link-mode=symlink cupy-cuda13x
    uv pip install --link-mode=symlink --no-deps \
      "jaxpp @ git+https://github.com/NVIDIA/jaxpp.git@7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9"
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
    CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -u \
      experiments/grug/moe/repro_jaxpp_fp8_expert_compile.py \
      --runtime jaxpp --kernel bf16 --timeout 600 --stack-after 120
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u \
      experiments/grug/moe/repro_jaxpp_fp8_expert_compile.py \
      --runtime direct --kernel fp8 --timeout 600 --stack-after 120 \
      --dump-dir /tmp/jaxpp-fp8-repro/direct
    CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -u \
      experiments/grug/moe/repro_jaxpp_fp8_expert_compile.py \
      --runtime distributed_direct --kernel fp8 --timeout 600 --stack-after 120 \
      --dump-dir /tmp/jaxpp-fp8-repro/distributed-direct
    CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -u \
      experiments/grug/moe/repro_jaxpp_fp8_expert_compile.py \
      --runtime jaxpp --kernel fp8 --timeout 600 --stack-after 120 \
      --dump-dir /tmp/jaxpp-fp8-repro/jaxpp
  '
```

## Smallest test matrix

Run direct JAX on one H100 first:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.50 \
  uv run python -u experiments/grug/moe/repro_jaxpp_fp8_expert_compile.py \
  --runtime direct --kernel fp8 --dump-dir /tmp/jaxpp-fp8-repro/direct
```

The matched distributed control initializes those same two ranks and devices,
but runs the ordinary JAX loss/backward only on the compute rank:

```bash
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=.50 \
  uv run python -u experiments/grug/moe/repro_jaxpp_fp8_expert_compile.py \
  --runtime distributed_direct --kernel fp8 \
  --dump-dir /tmp/jaxpp-fp8-repro/distributed-direct
```

Then run the same backward through JaxPP on two local H100 ranks:

```bash
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=.50 \
  uv run python -u experiments/grug/moe/repro_jaxpp_fp8_expert_compile.py \
  --runtime jaxpp --kernel fp8 --dump-dir /tmp/jaxpp-fp8-repro/jaxpp
```

The one-flag compiler control replaces only the FP8 grouped GEMMs and overwrite
state with BF16 grouped `einsum` operations:

```bash
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=.50 \
  uv run python -u experiments/grug/moe/repro_jaxpp_fp8_expert_compile.py \
  --runtime jaxpp --kernel bf16
```

The defaults are the minimum kernel-valid shape: one layer, one expert, 128
tokens, hidden size 128, intermediate size 128, one microbatch, and 16 amax
history entries. If that passes, increase only these dimensions in order:

```text
--layers 2
--experts 8 --tokens 1024
--hidden 640 --intermediate 384
--hidden 2560 --intermediate 1280
--microbatches 4
```

`--stop-after lower` proves that Python/JaxPP tracing and MPMD localization
finish without entering XLA compilation. The full path logs distinct direct
lower/compile/execute events. JaxPP's `eval_local` compiles and executes each
localized `apply_task`, so its combined boundary is named
`jaxpp_eval_local_compile_execute_entered`. The JaxPP logger identifies the
individual task being compiled. A watchdog dumps Python stacks periodically,
emits `watchdog_timeout`, and returns status 124 instead of hanging forever.

## Runtime pins

The production observation used JAX/JAXLIB 0.10.1 and NVIDIA/JaxPP revision
`7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9`. The FP8 kernel uses E4M3 for both
forward and reverse because mixed E4M3/E5M2 Mosaic WGMMA requires JAX 0.11 or
newer. A CUDA toolchain containing `ptxas` and `nvlink` must be on `PATH`.

## Interpretation

- Direct FP8 and distributed-direct FP8 pass while JaxPP FP8 stalls: JaxPP
  wrapping/localized `apply_task` compilation is essential.
- Distributed-direct FP8 stalls: the failure is below JaxPP in distributed
  JAX, Mosaic, or XLA compilation.
- Single-process direct FP8 stalls: the failure is below distributed runtime
  setup in the Mosaic/XLA compiler path.
- JaxPP BF16 passes while JaxPP FP8 stalls: the FP8 custom VJP/kernel graph is
  essential.
- The one-microbatch case stalls: gradient accumulation is not causal. If only
  `--microbatches 4` stalls, the overwrite add/max reduction is causal.
