# StatefulScan H100 checkpoint

This directory preserves the first H100 backend measurements for Shuttle's
generic `StatefulScan` lowering of the Gated DeltaNet recurrence. The benchmark
uses the production-like shape `B=1, Hq=16, Hv=32, K=V=128`, BF16 inputs, an
FP32 persistent state, decay scale `0.1`, and chunk size 64.

## Revisions and environment

- Shuttle checkpoint: `fae336fd48143fb70a9be3257ac45223a710d675`
- FLA: `9c8e42e762fce087c27b673af4922795d9edb85e` (`0.5.2`)
- FlashQLA: `050c6bbee9e03efbbfe41063fe4e33742c4a87cb` (`0.1.2`)
- benchmark source SHA-256:
  `de4b8746b4b8cdeabff254a037f9584fff4214a014f268d939a001c11ca5b36d`
- run-time `stateful_scan.py` SHA-256:
  `8a629380c85df437ece1b6ca504a1b48a511308430c1ced9e1e0ad28ce392fb8`
- run-time `gated_delta_scan.py` SHA-256:
  `2433b19659824586e68e3ba66d4a15b234c9575ec603ab793c390a2abb7fecc4`
- Python 3.12.13
- PyTorch `2.8.0+cu128`; CUDA runtime 12.8
- Triton 3.4.0
- TileLang 0.1.9; `apache-tvm-ffi` 0.1.9
- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0
- driver: 595.71.05
- HBM clock sampled at 2619 MHz; power cap 700 W
- host: `g83d0f6`

The SM clock was not locked. The JSON files preserve the clock sampled during
each process; it was 345 MHz while idle. The benchmark itself records CUDA
events after ten warmups, so all comparisons use the raw warmed distributions
rather than the idle clock sample.

The working tree advanced while the remote run was active. The two run-time
source hashes above identify the exact copies sent to the H100, independently
of the later local files.

## Correctness

At `B=1, T=64, Hq=16, Hv=32, K=V=128`:

| execution form | output max abs | output mean abs | state max abs | state mean abs | repeated output/state |
| --- | ---: | ---: | ---: | ---: | --- |
| FLA recurrent | 0.0002441 | 0.00001407 | 0.0000005364 | 0.00000003657 | bitwise equal |
| FLA chunk 64 | 0.0004427 | 0.00004214 | 0.005543 | 0.0004110 | bitwise equal |

Both results are finite. The recurrent backend is the source-ordered oracle;
the chunk backend uses the bounded-reassociation numerical contract represented
in the candidate plan saved in each JSON file.

## Timing results

Each row contains 50 CUDA-event measurements after ten warmups. Full samples,
candidate plans, hashes, and environment records are in `raw/*.json`.

| workload | backend | median ms | minimum ms | mean ms | maximum ms |
| --- | --- | ---: | ---: | ---: | ---: |
| decode B1, T1 | FLA recurrent | 0.073168 | 0.067520 | 0.073228 | 0.093152 |
| decode B4, T1 | FLA recurrent | 0.070048 | 0.067136 | 0.071617 | 0.100672 |
| decode B16, T1 | FLA recurrent | 0.073728 | 0.070112 | 0.075419 | 0.104416 |
| prefill B1, T64 | FLA recurrent | 0.084960 | 0.084416 | 0.086532 | 0.156544 |
| prefill B1, T64 | FLA chunk 64 | 0.515104 | 0.488512 | 0.514723 | 0.547808 |
| prefill B1, T256 | FLA recurrent | 0.321792 | 0.319488 | 0.322891 | 0.391456 |
| prefill B1, T256 | FLA chunk 64 | 0.532176 | 0.510464 | 0.533355 | 0.558464 |
| prefill B1, T2048 | FLA recurrent | 3.940768 | 3.937408 | 3.942070 | 4.020800 |
| prefill B1, T2048 | FLA chunk 64 | 0.510624 | 0.488800 | 0.512533 | 0.553696 |
| prefill B1, T8192 | FLA chunk 64 | 0.703536 | 0.699744 | 0.708307 | 0.911712 |

The empirical execution-form choice is real. Recurrent is 6.06x faster at
T64 and 1.65x faster at T256. Chunk is 7.72x faster at T2048. The bounded
candidate planner therefore needs a length-dependent crossover instead of a
fixed "decode recurrent, all prefill chunk" rule.

A separate no-warmup sanity process returned output shape
`[1, 2048, 32, 128]` and final-state shape `[1, 32, 128, 128]`, both finite.
Its single CUDA-event sample was 1.034 ms. It is not included in the timing
table; it confirms that the measured operation returns the complete grouped
value-head output and persistent state.

## FlashQLA result

FlashQLA installed and imported at the pinned revision, and its upstream static
function-signature test passed. Its kernel JIT did not run in this holder image:

1. The image had no system `nvcc`.
2. A split cached CUDA 13.2 compiler/runtime attempt failed because the runtime
   header tree lacked the matching `crt/host_config.h` search path. The full
   error is `raw/flash_qla_split_cuda_toolkit_failure.log`.
3. A complete cached CUDA 13.2 tree reached compilation but CCCL rejected its
   compiler/header combination as incompatible. The final error is
   `raw/correctness_flash_qla_b1_t64_hq16_hv32_k128_v128_c64.stderr.log`.

No FlashQLA timing is reported. The source revision, TileLang pin, and PyTorch
pin were not changed to work around the holder image.

The optional upstream FLA forward/backward test reached the repository's
explicit Hopper guard and failed because Triton 3.4 is in the known incorrect
backward range (`>=3.4,<3.7.1`). That complete diagnostic is preserved in
`raw/fla_upstream_gdn_chunk64_test.log`. The forward-only Shuttle correctness
and timing measurements above do not exercise that backward path.

## Reproduction commands

The H100 was reserved with the current Iris client from detached Marin commit
`839e5e9e18299d750e6009b58f4535f77f9edafa`:

```bash
uv sync --all-packages
uv run scripts/iris/dev_gpu.py \
  --config lib/iris/config/cw-us-east-02a.yaml \
  --name dlwh-stateful-gdn-i-20260807 \
  allocate --gpu-variant h100 --gpus-per-node 8 --priority interactive
```

Remote setup:

```bash
git clone https://github.com/fla-org/flash-linear-attention.git fla
git -C fla checkout --detach 9c8e42e762fce087c27b673af4922795d9edb85e
git clone https://github.com/QwenLM/FlashQLA.git flash-qla
git -C flash-qla checkout --detach 050c6bbee9e03efbbfe41063fe4e33742c4a87cb

uv venv --python 3.12 /tmp/shuttle-gdn/venv

# This first index-only attempt failed because the CUDA wheel index did not
# provide torch's setuptools requirement.
uv pip install --python /tmp/shuttle-gdn/venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0

uv pip install --python /tmp/shuttle-gdn/venv/bin/python setuptools==82.0.1
uv pip install --python /tmp/shuttle-gdn/venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
uv pip install --python /tmp/shuttle-gdn/venv/bin/python \
  numpy einops transformers pytest
uv pip install --python /tmp/shuttle-gdn/venv/bin/python \
  --no-deps -e /tmp/shuttle-gdn/tile_lifetime
uv pip install --python /tmp/shuttle-gdn/venv/bin/python \
  --no-deps -e /tmp/shuttle-gdn/repos/fla
uv pip install --python /tmp/shuttle-gdn/venv/bin/python jax==0.10.1
uv pip install --python /tmp/shuttle-gdn/venv/bin/python \
  tilelang==0.1.9 apache-tvm-ffi==0.1.9
uv pip install --python /tmp/shuttle-gdn/venv/bin/python \
  --no-deps -e /tmp/shuttle-gdn/repos/flash-qla

# This wheel supplied ptxas but not nvcc; the JIT attempts therefore used the
# cached CUDA 13.2 trees described above.
uv pip install --python /tmp/shuttle-gdn/venv/bin/python \
  nvidia-cuda-nvcc-cu12==12.8.93
```

The resolver installed setuptools 83.0.0 for the explicit setuptools command.

Representative benchmark invocation:

```bash
CUDA_VISIBLE_DEVICES=0 /tmp/shuttle-gdn/venv/bin/python \
  /tmp/shuttle-gdn/tile_lifetime/benchmarks/h100_stateful_scan.py \
  --mode prefill --backend fla_chunk \
  --batch-size 1 --sequence-length 2048 \
  --query-heads 16 --value-heads 32 \
  --key-dimension 128 --value-dimension 128 \
  --chunk-size 64 --decay-scale 0.1 \
  --warmups 10 --repeats 50 \
  --fla-root /tmp/shuttle-gdn/repos/fla \
  --shuttle-revision fae336fd48143fb70a9be3257ac45223a710d675 \
  --json-output /tmp/shuttle-gdn/results/prefill_fla_chunk_b1_t2048_hq16_hv32_k128_v128_c64.json
```

The holder was released after copying the artifacts. Iris reported the holder
job `killed`, no matching pod remained, and `dev_gpu status` reported no active
session.
