# Plan: Run vLLM on TPU via Docker sidecar

Goal: avoid making `vllm-tpu` a Python dependency inside the Ray runtime env. Instead, run `vllm serve` in a sibling Docker container on the TPU VM and talk to it over HTTP (OpenAI-compatible API). This reduces uv/pip pin conflicts and makes “stand up a vLLM server on a TPU node” a reusable primitive.

## Status (as of 2026-01-12)

Implemented (evaluators only; **no RL hot-reload changes**):

- Sidecar container manager:
  - `lib/marin/src/marin/vllm/docker_server.py` starts a sibling `vllm serve` container and polls `/v1/models`.
  - Leaves containers around on failure (no `--rm`) so `docker logs`/`docker inspect` are available; error messages redact secrets.
- TPU vLLM evaluator supports **docker sidecar (default)** and **native** modes:
  - `lib/marin/src/marin/evaluation/evaluators/vllm_tpu_evaluator.py`
  - Sidecar requires `MARIN_VLLM_DOCKER_IMAGE` and a Ray worker environment with `/var/run/docker.sock` mounted.
  - Model paths can be `gs://` / `s3://` and default to `load_format=runai_streamer` (no FUSE required).
  - Defaults TPU logging noise down via `TPU_MIN_LOG_LEVEL=5` and `TPU_STDERR_LOG_LEVEL=5` (overrideable).
- lm-eval-harness evaluator is wired to talk to the sidecar:
  - `lib/marin/src/marin/evaluation/evaluators/lm_evaluation_harness_evaluator.py`
  - Uses `local-chat-completions` + `/v1/chat/completions` when `apply_chat_template=True`, otherwise `local-completions`.
  - Fixes tokenizer resolution for `gs://...` model ids by:
    - accepting `engine_kwargs["tokenizer"]=<hf_id_or_local_path>`, or
    - staging tokenizer files from the remote checkpoint dir to a temporary local dir.
- Smoke test entry point for repeated bring-up:
  - `python -m marin.vllm.smoke_test ...` (starts server, runs a query, stops server; supports `--repeat` and a stable local cache dir).

Open issues / known rough edges:

- Persistent compilation cache behavior is still being investigated (local stable paths showed speedups; `gs://` cache paths did not seem to hit reliably).
- The historical `gcsfuse_mount/` bucket prefix still appears in some paths, but it does **not** imply a FUSE mount requirement anymore.

## Current state (what we’re building on)

- The Ray cluster already runs inside Docker and supports “docker-alongside-docker” by mounting the Docker socket:
  - `infra/marin-us-central2.yaml` has `-v "/var/run/docker.sock:/var/run/docker.sock"` and `--privileged`.
- There are already cluster configs for a vLLM-flavored image:
  - `infra/marin-us-central2-vllm.yaml` uses `.../marin/marin_vllm:<tag>`.
- Marin currently starts vLLM via the `vllm` CLI directly in-process (not via Python API):
  - `lib/marin/src/marin/evaluation/evaluators/vllm_tpu_evaluator.py` runs `vllm serve ...` and polls `/v1/models`.

## Decisions / updated direction

- The existing “vllm” Dockerfiles/images are **full Ray cluster images** that happen to be vLLM-capable. We should:
  - Rename `docker/marin/Dockerfile.vllm` -> `docker/marin/Dockerfile.vllm_cluster` (and update any build/publish scripts and infra references).
  - Create a **slim, dedicated sidecar image** for running `vllm serve` on TPU workers.
- For the sidecar base, prefer upstream artifacts over reassembling the TPU stack ourselves:
  - Start with `vllm/vllm-tpu` from Docker Hub (nightlies) as the base runtime for serving.
  - If we need full reproducibility and pinned versions, use the `tpu-inference` upstream Dockerfile as a pinned build recipe:
    - `https://github.com/vllm-project/tpu-inference/blob/main/docker/Dockerfile`
  - The plan below assumes we’ll take one of:
    - **Option A (fastest)**: “bring-your-own image”: run `vllm/vllm-tpu:<tag>` directly as the sidecar.
    - **Option B (recommended for long-term)**: build/publish `marin/vllm_tpu_sidecar:<tag>` that is a thin wrapper around upstream (adds minimal tooling + env defaults).

Pinned sidecar image (first milestone):

- `vllm/vllm-tpu:nightly-20260104-4a1e25b-0d4044e`

2) **Where should model weights live for vLLM?**
   - We now prefer a **fuseless** approach:
     - vLLM reads weights directly from object storage (`gs://...` / `s3://...`) using `load_format=runai_streamer` (defaulted for object-store paths).
   - HF repo IDs are also supported (vLLM downloads inside the sidecar container); decide where HF cache should live for resumability.

### Storage strategy (no FUSE)

We do not rely on gcsfuse for serving weights.

- For `gs://` / `s3://` checkpoints, prefer vLLM streaming (`load_format=runai_streamer`) so the sidecar reads directly from object storage.
- For HF repo IDs, vLLM will download into the container; choose whether to persist HF cache (e.g. a dedicated disk, or a separate blob-store cache population process).

3) **Do we need multi-tenant vLLM per VM?**
   - If multiple Ray jobs can land on the same TPU VM concurrently, we should allocate ports dynamically and use unique container names.
   - If scheduling is “one job per VM”, fixed port `8000` is fine and simpler.

4) **Should the sidecar be long-lived (daemon on node) or per-step?**
   - Per-step is simplest and matches “small units of work on preemptible TPUs”.
   - Long-lived could amortize startup but needs robust GC/coordination.

## Design overview

### High-level behavior

- In any TPU job that needs vLLM:
  1) Choose `port` (default: dynamic free port).
  2) Start a sibling Docker container that runs `vllm serve ...` with `--device tpu`.
  3) Poll `http://127.0.0.1:${port}/v1/models` until ready.
  4) Run evaluation by talking to the HTTP endpoint.
  5) Tear down the container on success/failure.

### Why this helps

- Keeps the Ray runtime env (uv export) stable.
- Removes uv/pip fights over hard pins in `vllm-tpu` (torch, torch-xla, setuptools, numba, etc.).
- Makes vLLM server startup a reusable “platform service” in Marin rather than an evaluator-specific dependency.

## Proposed implementation steps

### Step 1: Add a small vLLM “sidecar manager” module

Create something like `lib/marin/src/marin/vllm/docker_server.py`:

- `@dataclass(frozen=True) class VllmDockerServerConfig:`
  - `image: str` (e.g. `vllm/vllm-tpu:<tag>` or `us-<region>-docker.pkg.dev/.../marin/vllm_tpu_sidecar:<tag>`)
  - `model_name_or_path: str`
  - `port: int | None` (if `None`, pick a free port)
  - `host: str = "127.0.0.1"`
  - `container_name: str | None` (if `None`, derive from run id + port)
  - `env: dict[str, str]` (pass `HF_TOKEN`, `WANDB_API_KEY` if needed, plus `TOKENIZERS_PARALLELISM=false`)
  - `volumes: list[tuple[str, str]]` (e.g. mount `/opt/gcsfuse_mount`, `/tmp`, maybe HF cache)
  - `extra_vllm_args: list[str]` (append to `vllm serve`)
  - `docker_run_args: list[str]` (e.g. `--privileged`, `--net=host`, `--shm-size=...`)

- `start_vllm_docker_server(config) -> VllmServerHandle`
  - constructs and runs:
    ```bash
    docker run -d --rm --net=host --name "${name}" \
      --privileged \
      -v /opt/gcsfuse_mount:/opt/gcsfuse_mount \
      -v /tmp:/tmp \
      -e HF_TOKEN=... \
      <image> \
      vllm serve <model> --device tpu --host 127.0.0.1 --port <port> --trust-remote-code ...
    ```
  - polls `/v1/models` with timeout; on failure, captures:
    - `docker logs <name> --tail=200`
    - `docker inspect <name>` (maybe truncated)

- `stop_vllm_docker_server(handle)`:
  - `docker rm -f <name>` in a `finally` block.

Notes:
- Use host networking (`--net=host`) so the Ray process can hit `127.0.0.1:<port>`.
- Prefer unique container names, e.g. `marin-vllm-${run_id}-${port}`.
- Avoid local imports; keep subprocess usage isolated.

### Step 2: Wire `VllmTpuEvaluator` to use the sidecar

In `lib/marin/src/marin/evaluation/evaluators/vllm_tpu_evaluator.py`:

- Add an evaluator-level switch:
  - `vllm_mode: Literal["native", "docker"] = "docker"` (or pick based on env var like `MARIN_VLLM_MODE`).
- Implement:
  - `start_vllm_server_in_background(..., mode="docker")` to call the new `docker_server` helper instead of `subprocess.Popen([vllm, ...])`.
- Keep the existing HTTP readiness probing logic (it’s good), but adjust failure reporting to show container logs.

This ensures any evaluator that subclasses `VllmTpuEvaluator` automatically benefits.

### Step 3: Make the image configurable via cluster config / env var

We need a consistent way for TPU jobs to know which vLLM image to run.

Options:
- Read the cluster’s configured docker image tag (not great: that’s the *Ray* container image, not the sidecar image).
- Add a new env var in cluster YAML (preferred):
  - e.g. in `infra/marin-*.yaml` `docker.worker_run_options`:
    - `- -e MARIN_VLLM_DOCKER_IMAGE=<region-docker.pkg.dev/.../marin_vllm:<tag>>`
- Or add a “default vllm image” constant in code keyed by region (worse; duplicates infra).

Plan:
- Start with env var `MARIN_VLLM_DOCKER_IMAGE`, fail fast if missing when `mode="docker"`.
- Set it in the cluster YAML `docker.worker_run_options` (the Ray container env), e.g.:
  - `- -e MARIN_VLLM_DOCKER_IMAGE=vllm/vllm-tpu:nightly-20260104-4a1e25b-0d4044e` (fast path)
  - or `- -e MARIN_VLLM_DOCKER_IMAGE=<region>-docker.pkg.dev/.../marin/vllm_tpu_sidecar:<tag>` (preferred once we publish our wrapper)

### Step 4: Standardize mounts and cache layout

To keep resumability and avoid repeated downloads:

- Always mount `/tmp` (host) into the sidecar (`-v /tmp:/tmp`) so we can use stable local paths for:
  - compilation cache (`VLLM_XLA_CACHE_PATH`, `JAX_COMPILATION_CACHE_DIR`)
  - any explicit staging we decide to do later
- If we decide to persist HF cache across runs, mount a dedicated host path into the sidecar and set:
  - `HF_HOME=...`
  - `HF_HUB_CACHE=...`
  - `TRANSFORMERS_CACHE=...`

### Step 5: Provide a generic “stand up a server” executor primitive

Add a small utility function (or executor step wrapper) so any pipeline can:

- Start server
- Return an endpoint URL (as an output artifact, e.g. `endpoint.json`)
- Optionally leave it running for the duration of dependent steps (only if we add a “daemon” concept later)

For now (per-step), the pattern is:
- Step A: (optional) stage model/data
- Step B: run eval (starts/stops server internally)

### Step 6: Rollout plan

1) Add the sidecar manager module + unit tests that validate command construction (no actual docker).
2) Make sidecar the default; keep native path available as a fallback via `MARIN_VLLM_MODE=native`.
3) Update HELMET runner to use `VllmTpuEvaluator` docker mode (it already does).
4) Validate on a small TPU job:
   - Start server, hit `/v1/models`, run a single short eval, ensure cleanup.

## Edge cases / failure modes (handle explicitly)

- **Docker socket missing**: fail fast with a clear error (“docker-alongside-docker not enabled; check cluster yaml mounts”).
- **Port collision**: if fixed port, detect and choose another; prefer dynamic.
- **Container exits early**: surface `docker logs` tail in the exception.
- **Model path not visible in container**: error should include the path and suggested mounts.
- **TPU visibility**: if sidecar can’t see TPU devices, detect via vLLM logs and fail fast.

## Concrete code snippet (what we’d run)

Example `docker run` (final will be built programmatically):

```bash
docker run -d --net=host --name marin-vllm-${RUN_ID}-${PORT} \
  --privileged \
  -v /tmp:/tmp \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e TOKENIZERS_PARALLELISM=false \
  "${MARIN_VLLM_DOCKER_IMAGE}" \
  vllm serve gs://my-bucket/path/to/model \
    --host 127.0.0.1 \
    --port ${PORT} \
    --trust-remote-code
```

## Image work (slim sidecar)

### Rename current cluster Dockerfiles

- Rename:
  - `docker/marin/Dockerfile.vllm` -> `docker/marin/Dockerfile.vllm_cluster`
- Verify infra configs and build scripts still point to the right Dockerfile(s).

### Add a new sidecar Dockerfile

Create something like `docker/vllm-tpu-sidecar/Dockerfile`:

- Base: `FROM vllm/vllm-tpu:<tag>` (or build from `tpu-inference` upstream Dockerfile if we need a pinned build).
- Add only what’s required for Marin’s execution environment:
  - `curl` (optional, but useful for health/debug)
  - set sensible env defaults: `TOKENIZERS_PARALLELISM=false`, HF cache envs
- Entrypoint remains `vllm` so the runner can pass `vllm serve ...` args.

Rollout:
- Publish as `<region>-docker.pkg.dev/hai-gcp-models/marin/vllm_tpu_sidecar:<tag>`.
- Update `infra/*` to set `MARIN_VLLM_DOCKER_IMAGE` accordingly.

## What this plan does *not* include (yet)

- A long-lived node daemon / shared server across steps (possible later).
- Any change to executor semantics (e.g. “don’t aggregate if earlier steps failed”).
- Reworking HELMET itself; this is just plumbing.

## RL hot-reload addendum (`inference_ctx/vllm.py`)

Context: `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py` hot-reloads weights by calling vLLM internals:

- `self.llm.llm_engine.model_executor.driver_worker.sync_weights(...)`
- `self.llm.llm_engine.reset_prefix_cache()`

This is **not** expressible via the OpenAI HTTP API, so moving RL serving into a Docker sidecar requires adding a control plane to the sidecar.

### Milestones

**Milestone 1: Evaluators (no hot-reload)**

- Scope: `VllmTpuEvaluator` + HELMET/other evaluators.
- Use a “plain” sidecar that just runs `vllm serve` from a pinned image:
  - `vllm/vllm-tpu:nightly-20260104-4a1e25b-0d4044e`
- No hot reload; server lifecycle is per evaluation step.

## Next steps

1) End-to-end eval validation:
   - Run `experiments/scratch/exp_vllm_gsm8k_llama31_8b_instruct.py` on a TPU worker with sidecar mode, verify W&B + GCS outputs.
2) Harden infra assumptions:
   - Ensure all Ray TPU worker images/clusters that need sidecar have docker-alongside-docker correctly configured (`/var/run/docker.sock` mount).
   - Standardize where `MARIN_VLLM_DOCKER_IMAGE` is set (cluster YAML vs per-run).
3) Compilation cache:
   - Default `VLLM_XLA_CACHE_PATH` is now `/dev/shm/marin-vllm-xla-cache` (container-local) to avoid filling `/tmp` / Ray session directories.
   - Object-store cache paths like `gs://...` still require adding `gcsfs` (or equivalent) to the vLLM image; otherwise vLLM/JAX will warn and ignore the cache.

**Milestone 2: RL (true hot-reload)**

- Scope: `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py`.
- Goal: preserve true hot-reload semantics (on-policy / high throughput).
- Requires a long-lived sidecar per TPU VM for the duration of training (not per rollout batch).

### Sidecar requirements for RL hot-reload

1) **Custom entrypoint (not just `vllm serve`)**
   - The sidecar must start vLLM in a way that keeps access to `llm_engine` objects in-process.
   - It must expose a control endpoint like:
     - `POST /reload` → calls `sync_weights(...)` + `reset_prefix_cache()`.

2) **Weights transport via shared memory**
   - Standardize on `/dev/shm` for hot-reload payloads (fast, VM-local).
   - Mount it into the sidecar:
     - `-v /dev/shm:/dev/shm`
   - Trainer writes a new “weights payload” into `/dev/shm/...` and then triggers reload via control API.
   - Payload format options:
     - (A) a compact serialized state dict format understood by the sidecar
     - (B) a memory-mapped file protocol (still file-backed, but minimizes copies)
   - The `POST /reload` request should include:
     - path to payload in `/dev/shm`
     - model name (to choose `MODEL_MAPPINGS` / transpose keys)
     - an incrementing version id (for idempotency + debugging)

3) **Concurrency semantics**
   - Define behavior for in-flight requests during reload:
     - either block new requests during `sync_weights`
     - or accept requests but route them to “old weights” until reload completes
   - For correctness, recommend a short “write lock” where inference requests block while weights are being swapped.

4) **Failure handling / observability**
   - Control endpoint returns structured errors and logs reload timing.
   - Sidecar should expose:
     - `/healthz` (server ready)
     - `/metrics` (optional)
     - `/version` (image tag + git SHA + current weights version)

### Notes on feasibility

- True hot-reload remains feasible only because `vllm-tpu` exposes in-process `sync_weights`.
- Docker doesn’t prevent this, but it forces us to provide a control channel into the sidecar process.
- `/dev/shm` helps for throughput, but it’s just the transport; the key requirement is an in-sidecar control API that calls vLLM internals.
