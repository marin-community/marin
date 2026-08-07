# Generated StatefulScan H100 checkpoint

This directory preserves the first H100 execution of Shuttle's generated
bounded-rank affine recurrent-scan skeleton. The benchmark lowers recovered
tensor algebra to a generic Triton kernel with this transition:

```text
decayed = diagonal * state
residual[r] = scale[r] * (additive[r] - right[r]^T @ decayed)
next = decayed + sum_r outer(left[r], residual[r])
output = read^T @ next
```

The environment did not install FLA or FlashQLA. The two generated benchmark
files contain no FLA, FlashQLA, GDN-kernel, or KDA-kernel imports or calls. The
correctness reference is a local direct PyTorch interpretation of the generic
affine update, not an architecture kernel or external oracle.

## Revisions and source identity

- Shuttle checkpoint: `fae336fd48143fb70a9be3257ac45223a710d675`
- generated Triton skeleton SHA-256:
  `c410d304e941cfe5292bd1f727ba2af7485d33255f2952d5d728ffc2cfbe4d3b`
- H100 harness SHA-256:
  `5df1c4091cbbd05d43f12b3f2acfee343b135c76ee50ede0e7e3c0b15c6b7f8d`
- affine recovery SHA-256:
  `d0a4d72fb1ea9b3397e44d0630c0a112f81a97d42bc73d7da3456478a3039e41`
- generic update-expression fixture SHA-256:
  `d3b97ccb4424041561f22bdb38adc2de55511a434a747800fe5a6e659f368f79`
- plan records SHA-256:
  `093b8bd9360453c41aeeded1cac272b34cbe34aa6da92e0af53e56fb1c897612`

The rank-two run includes the pre-run semantic fix that computes all low-rank
residuals against the same decayed state, accumulates the corrections, and
applies their sum once. No source patch was required on the H100.

## Environment

- host: `g83d0f6`
- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0
- driver: 595.71.05
- CUDA runtime: 12.8
- PyTorch: `2.8.0+cu128`
- Triton: 3.4.0
- Python: 3.12.13
- HBM clock: 2619 MHz
- power cap: 700 W

One GPU was used from an eight-H100 holder. The SM clock was not locked; the
JSON files retain the clock sampled by each process.

## Results

The production shape is `B=1, T=64, H=32, K=V=128`, BF16 factors and output,
and FP32 persistent state. Each production and mutation row contains 50 CUDA
event samples after ten warmups.

| case | block_v | median ms | minimum ms | mean ms | maximum ms | output max abs | state max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scalar decay, rank 1 | 8 | 0.157120 | 0.155392 | 0.157999 | 0.169440 | 0.00006104 | 1.49e-8 |
| scalar decay, rank 1 | 16 | 0.149424 | 0.145888 | 0.150637 | 0.177376 | 0.00024414 | 1.49e-8 |
| scalar decay, rank 1 | 32 | 0.138544 | 0.137056 | 0.139366 | 0.149280 | 0.00024414 | 1.49e-8 |
| per-key decay, rank 1 | 32 | 0.138000 | 0.136480 | 0.138776 | 0.148992 | 0.00006104 | 1.86e-8 |
| scalar decay, rank 2 | 32 | 0.183376 | 0.181184 | 0.183866 | 0.193280 | 0.00012207 | 1.86e-8 |

All outputs and final states are finite and bitwise equal across repeated
invocations for identical inputs. The raw JSON files include full samples,
output/state SHA-256 hashes, recovered transition structure, low-rank bound,
diagonal axes, and term signatures.

The smoke case `B=1, T=4, H=2, K=V=16, R=1, block_v=16` used three warmups
and 20 samples. It produced an exact BF16 output, state maximum absolute error
`5.59e-9`, and median latency `0.04888 ms`.

The bounded physical choice matters even in this first kernel: block 32 was
the fastest rank-one production candidate, 11.8% faster than block 8 by
median latency. More importantly, per-key diagonal structure and a simultaneous
rank-two correction executed through the same recovered representation and
physical kernel without workload-specific dispatch.

## Reproduction

The H100 was allocated using the current Iris client from detached Marin commit
`839e5e9e18299d750e6009b58f4535f77f9edafa`:

```bash
uv sync --all-packages
uv run scripts/iris/dev_gpu.py \
  --config lib/iris/config/cw-us-east-02a.yaml \
  --name dlwh-generated-affine-scan-20260807 \
  allocate --gpu-variant h100 --gpus-per-node 8 --priority interactive
```

Remote environment setup:

```bash
uv venv --python 3.12 /tmp/shuttle-generated/venv
uv pip install --python /tmp/shuttle-generated/venv/bin/python setuptools==82.0.1
uv pip install --python /tmp/shuttle-generated/venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
uv pip install --python /tmp/shuttle-generated/venv/bin/python \
  numpy jax==0.10.1
uv pip install --python /tmp/shuttle-generated/venv/bin/python \
  --no-deps -e /tmp/shuttle-generated/tile_lifetime
```

The resolver installed setuptools 83.0.0 for the explicit setuptools command.
Before running, the environment check asserted that `find_spec("fla")` and
`find_spec("flash_qla")` both returned `None`.

Representative production invocation:

```bash
CUDA_VISIBLE_DEVICES=0 /tmp/shuttle-generated/venv/bin/python \
  /tmp/shuttle-generated/tile_lifetime/benchmarks/h100_generated_affine_scan.py \
  --batch-size 1 --sequence-length 64 --heads 32 \
  --key-dimension 128 --value-dimension 128 --update-rank 1 \
  --decay-axes scalar --gate-operation exp --block-v 32 \
  --warmups 10 --repeats 50 \
  --shuttle-revision fae336fd48143fb70a9be3257ac45223a710d675 \
  --json-output /tmp/shuttle-generated/results/production_b1_t64_h32_k128_v128_r1_scalar_exp_bv32.json
```

The holder was released after copying the raw artifacts. Iris reported the job
`killed`, no matching pod remained, and `dev_gpu status` reported no active
session.
