# Concurrent QuACK TVM-FFI hang reproducer

This is an upstream-ready, bounded reproducer for a concurrent multi-device
call to one QuACK function through `jax-tvm-ffi`. It imports neither Marin nor
Levanter. Nothing has been filed upstream.

## Environment

The failure was reproduced on Linux x86_64 with Python 3.12.13, NVIDIA H100
GPUs, CUDA 13, and these exact Python packages:

```text
jax==0.10.1
jaxlib==0.10.1
jax-tvm-ffi==0.1.3
quack-kernels==0.5.0
```

Install the direct-JAX reproducer into an isolated Python 3.12 environment:

```bash
uv venv --python 3.12 .venv-quack-repro
uv pip install --python .venv-quack-repro/bin/python \
  'jax[cuda13]==0.10.1' \
  jax-tvm-ffi==0.1.3 quack-kernels==0.5.0
```

`quack-kernels` supplies its CUTLASS DSL dependencies. A CUDA 13 driver/runtime
and Hopper-class GPUs remain external prerequisites.

## One-command reproducer

The default arguments are the minimum observed failure: three devices, three
experts, one token per expert, 8x8 BF16 matrices, and replicated inputs.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 .venv-quack-repro/bin/python -u \
  experiments/grug/moe/repro_jaxpp_quack_minimal.py
```

Expected result: after the default 180-second watchdog, the process prints a
JSON event containing `"verdict": "hang"` and exits with status 124. Python
stacks are dumped after 60 seconds and every 60 seconds thereafter.

The nearest passing control changes only the concurrent device/expert count:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv-quack-repro/bin/python -u \
  experiments/grug/moe/repro_jaxpp_quack_minimal.py --fsdp 2 --experts 2
```

Expected result: a `direct_eval_returned` event followed by
`{"event": "verdict", "verdict": "pass", ...}` and exit status 0.

Use shorter diagnostics when iterating:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 .venv-quack-repro/bin/python -u \
  experiments/grug/moe/repro_jaxpp_quack_minimal.py \
  --timeout 30 --stack-after 10
```

## Optional JaxPP mode

JaxPP is not required to reproduce the defect. To demonstrate the same
boundary through JaxPP MPMD, install the exact tested revision without changing
the direct-JAX package pins:

```bash
uv pip install --python .venv-quack-repro/bin/python --no-deps \
  'jaxpp @ git+https://github.com/NVIDIA/jaxpp.git@7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9'
uv pip install --python .venv-quack-repro/bin/python cupy-cuda13x==14.1.1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 .venv-quack-repro/bin/python -u \
  experiments/grug/moe/repro_jaxpp_quack_minimal.py \
  --runtime jaxpp --transfer scalar
```

The JaxPP mode uses two local ranks of three GPUs each. Its passing control uses
four visible GPUs plus `--fsdp 2 --experts 2`.

## Controls and interpretation

- `--operation plain` replaces QuACK with `jax.numpy.einsum` and passes.
- `--operation opaque` replaces QuACK with a Pallas call and passes.
- `--transform gradient` places the QuACK call in a custom-VJP backward pass.
- `JAXPP_QUACK_ALLOW_CUDA_GRAPH=false` disables QuACK CUDA graphs; fsdp3 still
  hangs.

Direct JAX and JaxPP have the same fsdp2/pass and fsdp3/hang boundary. That
places the failure below JaxPP: concurrent device execution threads invoke one
shared compiled QuACK TVM-FFI function. Serializing each registered
`JAXTVMFFIHandler` call is a validated workaround, but is intentionally not
applied by this reproducer.
