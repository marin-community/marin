# Generated partitioned Contract H100 failure checkpoint

This artifact is a non-accepted failure checkpoint. It contains no reportable
latency result.

The single authorized H100 invocation compiled and loaded the generated
typed-FFI library, executed the generated and ordinary-JAX paths, passed the
ordered CPU-reference, determinism, and monotone host-handler-count gates, and
completed the counterbalanced timing loops. It then failed while constructing
the JSON result because the harness called `git rev-parse HEAD` from a source
archive without a `.git` directory. Timing samples existed only in process
memory and were lost when result serialization failed. The invocation was not
retried.

The harness now requires an explicit `--shuttle-revision`, but that correction
is unmeasured.

## Revisions and allocation

- Generated source revision: `70581434c4`.
- Source archive SHA256:
  `abbbcc773bd72677c2489a56c36a3210867187b4140cf327021ac3383f79cb30`.
- Iris holder revision: `eafa4d49f7c55fbf2abb26b5d92c1ac7d093f9fb`.
- Iris job: `/dlwh/dev-gpu-dlwh-shuttle-partitioned-gemm-95bb`.
- Request: one H100, batch priority, one CPU, 32 GB memory, 50 GB disk.
- GPU attempts: one benchmark invocation, no retries, no tuning.

The detached holder project explicitly contained:

```toml
[dependency-groups]
dev = []
```

Local preallocation validation ran:

```text
uv sync --package marin-iris --extra controller --group dev
```

and confirmed that a nonexistent holder session returned the expected clean
absence before allocation.

## Environment and command

The remote environment used JAX, jaxlib, the CUDA plugin, and the CUDA PJRT
plugin at version `0.11.0`. The CUDA 13 extra supplied the runtime libraries and
NVCC; `nvidia-cuda-cccl` supplied compiler headers. JAX reported one CUDA device.

The one benchmark invocation was:

```text
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PWD/lib/tile_lifetime/src:$PWD" \
/app/.venv/bin/python \
lib/tile_lifetime/benchmarks/xla_partitioned_contract_map_gpu_custom_call.py \
  --artifact-directory /tmp/shuttle-partitioned-run \
  --nvcc /app/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc \
  --architecture sm_90a \
  --seed 20260809 \
  --warmup 10 \
  --iterations 1000 \
  --samples 30
```

This command intentionally predates the new required
`--shuttle-revision` argument.

## Preserved evidence

- `raw/generated_partitioned_contract.cu` is the exact compiled source.
- `raw/post-gated-pre-scheduler-hlo.txt.gz` is the exact recovered natural-Grug
  input after the previously generated low-rank gated-product replacement.
- `raw/benchmark.stderr` contains the NVCC warnings and final traceback.
- `raw/benchmark.stdout` is empty because JSON serialization never completed.
- `failure.json` records hashes, the failure boundary, and release status.

The holder was released immediately after copying these files. A subsequent
holder-status query reported no active session, and the Kubernetes pod was
absent.
