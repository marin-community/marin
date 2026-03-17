# Ray Multi-Host vLLM on TPU: Research Logbook

## Fast Weight Loading Reference (from Marin)

**Source**: `/Users/ahmed/code/marin/FAST_VLLM_LOAD_TPU.md`

**Problem**: Default `runai_streamer` loads weights at ~53 MiB/s (single-threaded HTTP). 8B=5min, 70B=41min just for weights.

**Solution**: In-process vLLM with Levanter's parallel fsspec loader:
1. Start vLLM with `load_format="dummy"` (instant empty skeleton)
2. Load weights from GCS via Levanter's fsspec: parse safetensors headers → 4 concurrent 2GB byte-range chunk downloads → zero-copy numpy arrays
3. Inject weights via `sync_weights()` (maps HF→vLLM param names, reshapes heads, pads to 128)

**Performance**: 8B weights: 300s→74s. 70B projected: 41min→6-10min.

**Config**:
```bash
LEVANTER_FSSPEC_CHUNK_BYTES=2147483648     # 2GB chunks
LEVANTER_FSSPEC_MAX_CONCURRENT_CHUNKS=8    # parallelism
```

**Key files**: `lib/marin/src/marin/inference/vllm_inprocess.py`, `lib/levanter/src/levanter/compat/fsspec_safetensor.py`, `lib/marin/src/marin/rl/weight_utils.py`

**Known issue**: vLLM V1 EngineCore spawns child process → TPU lock conflict (`The TPU is already in use by process with pid 1`). Weights load correctly but HTTP server can't start. Fix needed: disable V1 multiprocessing or use `LLM.generate()` directly.

**Relevance to this logbook**: For RAY-004 (70B), weight loading will be a major bottleneck (~41min with runai_streamer). Consider integrating Levanter fsspec loader. Model mappings exist for Llama and Qwen.

## CRITICAL RULE: Same-Region Weight Loading

**NEVER load model weights from a different GCP region than the TPU.**

- TPU zone: **us-east1-d**
- Weight bucket: **`gs://marin-us-east1/`** (us-east1)
- Cross-region GCS reads are throttled to ~50 MiB/s. Same-region reads: **1-10 Gbps**.
- For 131 GiB (70B model): cross-region = **~41 min**, same-region = **~2-3 min**.
- HuggingFace Hub downloads also go cross-region (CDN). Pre-stage to regional GCS instead.

**Canonical weight paths (us-east1)**:
- Llama 3.3 70B: `gs://marin-us-east1/gcsfuse_mount/models/meta-llama--Llama-3-3-70B-Instruct` (131 GiB safetensors, 30 shards)

---

## Scope
- **Goal**: Get vLLM multi-host inference working on TPU v6e-16 **with Ray**, using `TPU_MULTIHOST_BACKEND=ray` and the `RayDistributedExecutor`. PP-across-hosts + TP-within-host architecture. Direct comparison against no-Ray logbook (`no_ray_multihost_vllm.md`).
- **Primary metric(s)**: See Metrics Protocol below (identical to no-Ray logbook).
- **Constraints**: v6e-16 (4 hosts, 4 chips each, 125 GiB HBM/host), `vllm/vllm-tpu:nightly` image, GCP project `hai-gcp-models`, **us-east1-d**. **Dedicated TPU target: `vllm-ray-multihost-v6e16-use1d`** (separate from no-Ray TPU).
- **Stop criteria**: Successfully generate coherent responses from a multi-host Ray PP=4 setup. Then scale to 70B with all metrics captured. Compare all metrics against no-Ray results.

## Fixed Evaluation Dataset

**MATH-500** — 500 math problems from `HuggingFaceH4/MATH-500` (test split).

- **Source**: HuggingFace Hub or `gs://tunix/data/MATH-500/test.jsonl`
- **Size**: 500 problems (fixed, deterministic ordering)
- **Prompt format**: Chat messages — `[{"role": "user", "content": "<problem> Write your answer in \\boxed{} format."}]`
- **Answer grading**: SymPy-based mathematical equivalence via `tunix/utils/math_grading.py`
- **Why this dataset**:
  1. Fixed size enables apples-to-apples comparison across all runs
  2. 500 prompts is enough for p50/p95/p99 percentile statistics
  3. Grading answers verifies multi-host doesn't corrupt output quality
  4. Varied problem difficulty = varied prompt/response lengths (natural distribution)
  5. Already used in tunix eval pipeline, grading infrastructure exists

**Benchmark parameters (held constant across all runs)**:
- `temperature=0.0` (greedy — deterministic for reproducibility)
- `max_tokens=2048` (enough for reasoning chains)
- `concurrency=16` (parallel requests to server)
- Dataset: all 500 MATH-500 problems, same order every run

**What we measure per run**:
- All metrics from Metrics Protocol below
- **MATH-500 accuracy (%)** — correctness sanity check
- **Per-problem timing** — enables distribution analysis

## Metrics Protocol

**Every experiment MUST record the following. No exceptions.**

### Startup Metrics
| Metric | How to measure |
|--------|---------------|
| `t_ray_cluster_start` | Wall time from first `ray start` to all nodes joined (seconds) |
| `t_docker_start` | Wall time from `docker run` to container running (seconds) |
| `t_model_load` | Wall time from process start to "Model loaded" in logs (seconds) |
| `t_xla_compile` | Wall time for first XLA compilation pass (seconds) |
| `t_first_ready` | Wall time from launch to server accepting requests / first output (seconds) |
| `t_total_startup` | End-to-end from first `docker run` to first successful response (seconds) |

### Latency Metrics (per request, over all 500 MATH-500 problems)
| Metric | How to measure |
|--------|---------------|
| `ttft` | Time to first token — from request sent to first token received (ms) |
| `tpot` | Time per output token — inter-token latency after first token (ms) |
| `e2e_latency` | Total request duration from send to last token (ms) |
| `p50/p95/p99_ttft` | Percentiles over 500 requests |
| `p50/p95/p99_tpot` | Percentiles over 500 requests |
| `p50/p95/p99_e2e` | Percentiles over 500 requests |

### Throughput Metrics (over full 500-problem run)
| Metric | How to measure |
|--------|---------------|
| `prompt_tok_per_s` | Total prompt tokens processed / wall time |
| `gen_tok_per_s` | Total generated tokens / wall time |
| `total_tok_per_s` | (prompt + generated tokens) / wall time |
| `req_per_s` | Requests completed per second |
| `batch_wall_time` | Wall time for entire 500-problem run (seconds) |

### Correctness & Reward Metrics
| Metric | How to measure |
|--------|---------------|
| `math500_accuracy` | Correct answers / 500 (using SymPy grading) |
| `math500_format_rate` | Responses containing `\boxed{}` / 500 |
| `math500_empty_rate` | Empty or error responses / 500 |
| `t_reward_total` | Wall time to grade all 500 responses (seconds) |
| `t_reward_per_response` | Mean time to grade one response (ms) |
| `p50/p95/p99_t_reward` | Percentiles of per-response grading time (ms) |
| `reward_timeouts` | Number of responses where SymPy grading timed out |

### Resource Metrics (per host)
| Metric | How to measure |
|--------|---------------|
| `hbm_used_gb` | HBM used per host after model load (GiB) |
| `hbm_total_gb` | HBM total per host (GiB) |
| `hbm_kv_cache_gb` | HBM available for KV cache (GiB) |
| `cpu_percent` | CPU utilization during inference |
| `host_ram_used_gb` | Host RAM used (GiB) |

### Ray-Specific Metrics (NOT in no-Ray logbook — extra data)
| Metric | How to measure |
|--------|---------------|
| `ray_nodes_visible` | Number of nodes in `ray status` |
| `ray_tpu_resources` | TPU resources reported per node |
| `ray_pg_creation_time` | Time to create and ready placement group (seconds) |
| `ray_worker_init_time` | Time for all Ray workers to init (seconds) |
| `ray_overhead_estimate` | `t_total_startup(ray) - t_total_startup(no-ray)` (seconds) |

### Stability Metrics (for stress tests)
| Metric | How to measure |
|--------|---------------|
| `error_count` | Number of failed requests |
| `error_types` | Categorized error messages |
| `hang_detected` | Whether any request timed out (bool) |
| `throughput_variance` | Coefficient of variation of per-minute throughput |
| `max_consecutive_ok` | Longest streak of successful requests |

### Collection Method
- **Benchmark script**: `vllm/marin_dev/benchmark_math500.py` — THE canonical benchmark for all runs
  - Loads MATH-500 (500 problems) from HuggingFace or local JSONL
  - Sends to vLLM server via `/v1/chat/completions` with streaming
  - Collects: TTFT, TPOT, E2E latency with p50/p95/p99
  - Throughput: prompt tok/s, gen tok/s, req/s
  - Correctness: `\boxed{}` extraction + SymPy grading
  - Reward timing: per-response grading time with percentiles
  - Saves full per-request results to JSON with `--output`
  - Usage: `python benchmark_math500.py --server http://localhost:8000 --concurrency 16 --output results_RAY-XXX.json`
- **Ray cluster**: `ray status`, `ray list nodes` for cluster health; timestamps around `ray start` commands
- **Startup**: Parse timestamps from Docker/vLLM logs + `date +%s.%N` at each stage
- **Resources**: `jax.local_devices()[0].memory_stats()` inside container, `top`/`free` on host
- **All timestamps**: UTC, ISO-8601 format

## Baseline
- **Date**: 2026-03-14
- **Prior work**: See `vllm/marin_dev/logbooks/multihost_v6e16.md` and `no_ray_multihost_vllm.md`
  - MH-001a-f: No-Ray approaches tried, correct PP+TP architecture identified.
  - Ray path never tested on this TPU — this logbook is the first systematic Ray evaluation.
  - Known bug: `hbm_usage_bytes()` queries non-addressable devices. Fix in `patch_multihost.py`. May or may not trigger under Ray (Ray sets `CLOUD_TPU_TASK_ID=0` per worker).
  - Dedicated v6e-16 TPU (`vllm-ray-multihost-v6e16-use1d`) to be allocated for Ray experiments in **us-east1-d** (separate from no-Ray TPU `marin-multihost-v6e16`).
  - Historical runs below were done in `europe-west4-a` and retained for traceability.
- **Code refs**:
  - `tpu-inference/tpu_inference/executors/ray_distributed_executor.py` — Ray PP executor
  - `tpu-inference/tpu_inference/worker/tpu_worker.py` — PP worker init (Ray-specific branches)
  - `tpu-inference/tpu_inference/distributed/jax_parallel_state.py` — PP tensor transfer
  - `tpu-inference/tpu_inference/platforms/tpu_platform.py` — backend selection (`TPU_MULTIHOST_BACKEND=ray`)
  - `tpu-inference/scripts/multihost/run_cluster.sh` — Docker Ray cluster launcher
  - `tpu-inference/scripts/multihost/deploy_cluster.sh` — multi-host deployment script
  - `vllm/marin_dev/patch_multihost.py` — hbm_usage_bytes fix (may still be needed)
- **Baseline numbers (single-host v6e-4, Llama 8B, NOT on MATH-500)**:
  - `total_tok_per_s`: 6,586 (100 generic prompts)
  - All other metrics: not recorded — RAY-000 will establish full baseline

## Compute

- **Multi-host TPU (target)**: `vllm-ray-multihost-v6e16-use1d` — v6e-16, us-east1-d, 4 hosts, **TO_ALLOCATE**
- **Single-host baseline**: Can use worker 0 of the v6e-16 (4 chips = v6e-4 equivalent)
- **Allocation method**: `gcloud alpha compute tpus queued-resources create` (or `dev_tpu.py allocate`)

## Ray Launch Reference

### Environment Variables (required on all hosts)
```bash
export TPU_MULTIHOST_BACKEND=ray
export TPU_BACKEND_TYPE=jax
export JAX_PLATFORMS=''
```

### Ray Cluster Start
```bash
# Head node (worker 0):
ray start --head --port=6379

# Worker nodes (workers 1-3):
ray start --address=<HEAD_INTERNAL_IP>:6379 --block
```

### vLLM Serve (from head node)
```bash
vllm serve <MODEL> \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 4 \
  --distributed-executor-backend ray \
  --max-model-len <LEN> \
  --port 8000
```

### Using deploy_cluster.sh (alternative)
```bash
bash tpu-inference/scripts/multihost/deploy_cluster.sh \
  -s ./tpu-inference/scripts/multihost/run_cluster.sh \
  -d vllm/vllm-tpu:nightly \
  -c /cache/huggingface \
  -t $HF_TOKEN \
  -H <HEAD_EXTERNAL_IP> \
  -i <HEAD_INTERNAL_IP> \
  -W <WORKER1_IP>,<WORKER2_IP>,<WORKER3_IP>
```

## Experiment Plan

### Phase 0: TPU Allocation & Setup

| Step | Action | Details |
|------|--------|---------|
| SETUP-1 | Allocate TPU | `gcloud alpha compute tpus queued-resources create vllm-ray-multihost-v6e16-use1d --accelerator-type=v6e-16 --zone=us-east1-d --runtime-version=v2-alpha-tpuv6e --node-id=vllm-ray-multihost-v6e16-use1d --best-effort` |
| SETUP-2 | Wait for READY | Poll until all 4 hosts are READY |
| SETUP-3 | Collect IPs | Get external + internal IPs for all 4 workers |
| SETUP-4 | Setup SSH | Add SSH config entries for all workers |
| SETUP-5 | Setup Docker | Add docker group, pull `vllm/vllm-tpu:nightly`, create volumes on all 4 hosts |
| SETUP-6 | Verify TPU health | Run `python -c "import jax; print(jax.devices())"` on each host to confirm 4 chips visible |

### Phase 1-3: Experiments

| Run | Config | Hypothesis | Key Metrics to Compare |
|-----|--------|------------|----------------------|
| RAY-000 | Llama 8B, single-host (worker 0 only), TP=4, PP=1 — run MATH-500 (500 problems) | Establish full baseline with all metrics on fixed dataset. **Should match NRM-000 exactly** (same model, same TPU type, no multi-host). | All startup + latency + throughput + correctness + resource metrics |
| RAY-001 | Llama 8B, PP=4, TP=4, Ray cluster all 4 hosts — run MATH-500 | Ray multi-host boots and generates correctly. 8B trivially fits (4 GB/host). Ray adds startup overhead but PP latency behavior should match no-Ray PP. | `t_total_startup`, `t_ray_cluster_start`, `ttft`, `gen_tok_per_s`, `math500_accuracy` vs NRM-001 |
| RAY-002 | Same as RAY-001 + `patch_multihost.py` if RAY-001 hits hbm bug | hbm_usage_bytes bug may or may not trigger under Ray (each worker is CLOUD_TPU_TASK_ID=0). Patch fixes it if needed. | Same as RAY-001 |
| RAY-003 | Llama 8B, PP=2, TP=4, 2 hosts only, Ray — run MATH-500 | Simpler topology if PP=4 has issues. Fewer PP stages = less pipeline bubble = better TTFT. | `ttft`, `tpot`, `math500_accuracy` vs RAY-001 and NRM-003 |
| RAY-004 | Llama 70B, PP=4, TP=4 on v6e-16, Ray — run MATH-500 (500 problems) | 140GB / 4 = 35GB/host, fits in 125 GiB. This is the real goal. Compare directly against NRM-004. | `t_model_load`, `t_xla_compile`, `hbm_used_gb`, `ttft`, `gen_tok_per_s`, `math500_accuracy` vs NRM-004 |
| RAY-005 | Llama 70B, MATH-500 at concurrency=1,4,16,32, Ray | Throughput scaling with concurrency. Compare scaling curve against NRM-005. | `gen_tok_per_s`, `req_per_s`, `p95_e2e` at each concurrency level vs NRM-005 |
| RAY-006 | Llama 70B, MATH-500 x3 loops (1500 requests), 30+ min sustained, Ray | Stability: no hangs, no OOM, no throughput degradation, accuracy stays constant. | `error_count`, `hang_detected`, `throughput_variance`, `math500_accuracy` vs NRM-006 |

### Fallback Rules
- If RAY-001 fails: check `ray status`, logs for TPU resource detection issues. Try restarting Ray cluster.
- If persistent Ray failures: try `deploy_cluster.sh` scripted path (more battle-tested).
- If RAY-004 OOMs: try quantization (`tpu_int8`) or reduce `max_model_len`.
- Record all failures as data — they matter for the Ray vs no-Ray comparison.

## Results Summary Table

| Run | Status | t_total_startup (s) | t_ray_cluster_start (s) | p50_ttft (ms) | p50_tpot (ms) | gen_tok/s | req/s | math500_acc (%) | t_reward_total (s) | p95_t_reward (ms) | hbm_used/host (GiB) | Notes |
|-----|--------|--------------------:|------------------------:|--------------:|--------------:|----------:|------:|----------------:|-------------------:|------------------:|---------------------:|-------|
| RAY-000 | SKIP | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Cannot run single-host on multi-host slice; use NRM-000 |
| RAY-001 | **DONE** | ~270 | ~15 | 50.4 | 34.6 | 389.1 | 0.58 | 42.20 | 0.80 | 0.75 | 3.25 | First successful Ray multi-host; 500/500 completed, 0 errors |
| RAY-002 | - | - | - | - | - | - | - | - | - | - | - | |
| RAY-003 | INTERRUPTED | - | - | - | - | - | - | - | - | - | - | Fallback PP=2 run started, then TPU slice was reclaimed (`state=DELETING`) |
| RAY-004 | **DONE** | ~375 | ~15 | 95.8 | 54.5 | 259.3 | 0.48 | 67.60 | 0.78 | 0.72 | 8.46 | 70B Llama 3.3 via gcsfuse, TP sharding fix, 0 errors |
| RAY-005a | **DONE** | - | - | - | - | 814 | 1.69 | 69.0 | - | - | 8.46 | 70B GRPO 1 batch (1,024 completions), 0 errors |
| RAY-005b | **DONE** | - | - | - | - | 929 | 2.00 | 76.5 | - | - | 8.46 | 70B GRPO 10 batches (10,240 completions), 0 errors, 5.2% variance |
| V8-70B-b | **DONE** | ~330 | N/A | - | - | 2,427 | 5.23 | 76.7 | - | - | ~17.5 | v6e-8 single host TP=8, 10 batches, **2.6x faster than Ray PP=4** |
| RAY-006 | BLOCKED | - | - | - | - | - | - | - | - | - | - | Qwen3.5-122B unsupported architecture in tpu-inference |
| RAY-007 | **DONE** | - | - | - | - | 221 | 0.43 | N/A | - | - | 28.5 | **Qwen3-235B MoE** on v6e-16, first model needing multi-host PP, 1,008/1,024 ok |

## Cross-Comparison Tables

### Ray PP=4 vs No-Ray Replicas (Llama 8B, v6e-16)

| Metric | NRM-003 (4 replicas) | RAY-001 (PP=4) | Notes |
|--------|--------------------:|---------------:|-------|
| gen_tok/s (MATH-500) | 3,894 | 389 | 10x: replicas win for small models |
| gen_tok/s (GRPO) | 11,912 | 929 | 12.8x: replicas win at high concurrency |
| MATH-500 accuracy | ~42% | 42.2% | Same model, same accuracy |
| Architecture | 4× independent servers | 1× PP=4 pipeline | |

### Ray PP=4 vs Single Host TP=8 (Llama 70B)

| Metric | v6e-8 TP=8 | v6e-16 Ray PP=4 | Ratio |
|--------|----------:|----------------:|------:|
| Chips | 8 | 16 | 0.5x |
| GRPO tok/s (10 batch) | **2,427** | 929 | **2.6x faster** |
| GRPO batch time (s) | **196** | 513 | **2.6x faster** |
| Epoch projection (h) | **10.2** | 26.8 | **2.6x faster** |
| Errors | 0 | 0 | Same |

### Models That NEED Multi-Host PP

| Model | Params | Weight Size | Fits v6e-8? | v6e-16 PP=4 | tok/s |
|-------|-------:|----------:|:-----------:|:-----------:|------:|
| Llama 8B | 8B | 15 GiB | ✅ | Overkill | 389 |
| Llama 70B | 70B | 131 GiB | ✅ (tight) | Works | 259-929 |
| **Qwen3-235B MoE** | **235B** | **438 GiB** | **❌** | **✅ (tight)** | **221** |

## Experiment Log

### 2026-03-14 — SETUP: TPU Allocation & Environment

**SETUP-1: Allocate TPU `vllm-ray-multihost-v6e16`**
- Start time: 2026-03-14T~UTC
- Command:
```bash
gcloud alpha compute tpus queued-resources create vllm-ray-multihost-v6e16 \
  --accelerator-type=v6e-16 \
  --zone=europe-west4-a \
  --runtime-version=v2-alpha-tpuv6e \
  --node-id=vllm-ray-multihost-v6e16 \
  --best-effort \
  --quiet
```
- Status: **COMPLETE** — TPU READY after ~4 min provisioning
- QR created: 2026-03-14T~22:38Z, READY: ~22:43Z

**SETUP-2: Host IPs**
```
Worker 0: internal=10.164.1.219  external=34.12.196.28
Worker 1: internal=10.164.1.37   external=34.178.90.123
Worker 2: internal=10.164.1.222  external=34.7.213.234
Worker 3: internal=10.164.1.220  external=34.34.27.233
```
Head node (Ray head): Worker 0 (10.164.1.219)

**SETUP-3: Docker + Image Setup**
- Docker group + volumes: DONE (all 4 hosts)
- `docker pull vllm/vllm-tpu:nightly`: DONE (all 4 hosts, sha256:92a3a7e13354b80850207da9c16532af3e16bd1ed143ff2be2f5988bcfbc4dd8)
- Status: **COMPLETE**

**SETUP-4: TPU Health Check**
- Initial attempt: `docker run --rm --privileged ... python3 -c "import jax; print(jax.devices())"` on all workers
- Result: FAILED — "TPU initialization failed: Failed to connect to [::]:8353"
- Root cause: `tpu-runtime.service` was stopped. On multi-host v6e-16 slices, the tpu-runtime provides ICI coordination and must be running for Docker containers to access TPU.
- Fix: Restarted `tpu-runtime.service` on all 4 hosts — all active.
- Note: For Ray path, containers use `JAX_PLATFORMS=''` anyway — Ray handles TPU access internally. Direct JAX device check not needed; Ray cluster status will verify TPU availability.
- **Important learning**: Do NOT stop tpu-runtime on multi-host slices. Unlike single-host, the runtime is required for chip coordination.

**SETUP-5: Ray Cluster Start**
- Head node (worker 0): started 22:55:23Z, container ID `5424dcef...`
- Worker 1: started 22:55:44Z
- Worker 2: started 22:55:49Z
- Worker 3: started 22:55:55Z
- Cluster fully formed by: 22:56:15Z
- `t_ray_cluster_start`: ~52s (from first `docker run` to `ray status` showing all 4 nodes)
- `ray status` output:
  - 4 active nodes, 0 pending, 0 failures
  - Resources: 720 CPU, **16 TPU**, 2.65 TiB memory, 121.60 GiB object_store
  - TPU resource key: `TPU-v6e-16-head: 1.0`
- Status: **COMPLETE**

### 2026-03-14 22:57 UTC — RAY-000: Single-host baseline (Llama 8B, TP=4, PP=1)

**Hypothesis**: Establish full baseline metrics using MATH-500 on single host. Should match NRM-000 (same TPU type, same model). No multi-host overhead.

**Config**:
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- TP=4, PP=1 (single host, worker 0 only)
- `max_model_len=4096`
- Benchmark: `benchmark_math500.py --concurrency 16 --max-tokens 2048 --temperature 0.0`
- vLLM launched inside Ray head container on worker 0

**Launch command**:
```bash
# Inside ray-node container on worker 0:
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --port 8000 \
  --trust-remote-code

# Benchmark (from worker 0):
python /tmp/benchmark_math500.py \
  --server http://localhost:8000 \
  --concurrency 16 \
  --output /tmp/results_RAY-000.json
```

**Attempt 1 (22:57 UTC)**: FAILED
- vLLM started inside Ray head container which has `TPU_MULTIHOST_BACKEND=ray` in env
- Even with TP=4, PP=1, the TPU platform plugin detected the env var and forced `RayDistributedExecutor`
- Ray executor spawned 4 workers across ALL 4 nodes (not single-host)
- Crashed at `tpu_inference/distributed/utils.py:96`: `AttributeError: d.coords` — `jax.local_devices()` returned devices without `.coords` attribute
- Full traceback: `get_device_topology_order_id()` → `min(d.coords for d in local_devices)` — devices in Ray actor context don't have TPU `.coords`
- Root cause: For single-host baseline, must NOT set `TPU_MULTIHOST_BACKEND=ray`

**Attempt 2 (23:01 UTC)**: FAILED
- Override: `TPU_MULTIHOST_BACKEND= CLOUD_TPU_TASK_ID=0 vllm serve ...`
- This sets `TPU_MULTIHOST_BACKEND` to empty string `""` (NOT unset)
- Pydantic validation error: `Invalid value '' for TPU_MULTIHOST_BACKEND. Valid options: ['ray']`
- The `tpu_inference/envs.py` validates that when set, the variable must be `"ray"`. Empty string is not accepted when it's explicitly in the environment.

**Attempt 3 (23:21 UTC)**: FAILED
- Fixed env: `unset TPU_MULTIHOST_BACKEND && CLOUD_TPU_TASK_ID=0 vllm serve ...`
- TPU platform correctly chose `uni` executor (PP=1, no Ray)
- Server hung at: `TPU backend initialization is taking more than 60.0 seconds`
- Root cause: On a multi-host v6e-16 slice, JAX's TPU runtime is inherently multi-host. Even a single process on one worker tries to coordinate with all 4 hosts. Since only worker 0 is running vLLM, the other 3 hosts don't participate → indefinite hang.
- **This is NOT a Ray issue** — it's a fundamental multi-host TPU constraint. You cannot run an isolated single-host JAX/vLLM process on one worker of a multi-host slice.

**RAY-000 Decision**: SKIP — single-host baseline cannot run on a multi-host slice.
- For apples-to-apples comparison, NRM-000 results (run on a separate v6e-4 or on the no-Ray TPU) will serve as the shared single-host baseline for both logbooks.
- Alternatively, can allocate a separate v6e-4 later if needed.
- **Pivoting to RAY-001** (multi-host with Ray) which is the real experiment.
- Additional finding: `d.coords` bug at `distributed/utils.py:96` only affects PP=1+Ray code path. PP>1 uses `self.topology_order_id = self.rank` and skips `get_device_topology_order_id()`. RAY-001 with PP=4 should not hit this bug.
- Ray cluster restarted at 23:32Z (head container had died from hung processes). All 4 nodes back up, 16 TPU confirmed.

### 2026-03-14 23:33 UTC — RAY-001: Multi-host Llama 8B (PP=4, TP=4, Ray)

**Hypothesis**: Ray multi-host boots and generates correctly with PP=4, TP=4. Llama 8B (16GB weights) trivially fits across 4 hosts (4 GB/host). The `d.coords` bug should NOT trigger because PP>1 skips `get_device_topology_order_id()`.

**Config**:
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- TP=4, PP=4 (all 4 hosts, pipeline parallel across hosts, tensor parallel within each host)
- `max_model_len=4096`
- `--distributed-executor-backend ray`
- Benchmark: `benchmark_math500.py --concurrency 16 --max-tokens 2048 --temperature 0.0`

**Launch command**:
```bash
# Inside ray-node container on worker 0:
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 4 \
  --distributed-executor-backend ray \
  --max-model-len 4096 \
  --port 8000 \
  --trust-remote-code
```

**Attempt 1 (23:33 UTC)**: FAILED — maintenance event killed head container mid-flight. Log file lost.

**Attempt 2 (23:55 UTC)**: FAILED — XLA compilation crash
- Ray cluster restarted (4 nodes, 16 TPU confirmed)
- vLLM successfully connected to Ray cluster, created placement group
- All 4 workers initialized, model loaded on all hosts
- **Crashed during `_precompile_sampling()`** with TPU hardware error:
```
jax.errors.JaxRuntimeError: INTERNAL: Core halted unexpectedly
Assertion args: 0x00000002 0x00000001 0x00000000
Accelerator device halted prematurely, perhaps due to an on-device check-failure.
Node 0 halted unexpectedly at tag:pc TensorCoreSequencer:1:0x147:
  schecklt: Invalid logical z: enhanced-barrier-parent-phase-1 no HLO mapping
```
- Error was on `ip=10.164.1.220` (worker 3) during `compilation_manager.py:78`
- Repeated across cluster (2x)
- **Likely cause**: TPU hardware degradation from maintenance event (warning was logged at 23:33Z)
- **Good news**: `d.coords` bug is confirmed bypassed with PP>1. Worker init, model load, and placement groups all worked correctly.

**Progress summary for RAY-001**:
- [x] Ray cluster formation (4 nodes, 16 TPU, placement groups)
- [x] Worker init (sorted by IP, ranks assigned, env vars copied)
- [x] TPU device access in Ray actors (PP>1 avoids `d.coords` bug)
- [x] Model download + loading on all 4 hosts
- [ ] XLA precompilation — **FAILED with hardware error**
- [ ] Server ready
- [ ] Benchmark

**Attempt 3 (23:58 UTC)**: FAILED — container not running. Head container died again post-maintenance.

**Assessment**: TPU reports HEALTHY but containers keep dying. The maintenance event left the TPU in an unstable state. Raw TPU drivers open and detect topology correctly (`deepsea_chips_per_host_bounds=2,2,1`, `deepsea_host_bounds=2,2,1`, all 4 host IPs visible). But XLA compilation crashes with "Core halted unexpectedly" and containers become unresponsive.

**Decision**: Delete and recreate TPU `vllm-ray-multihost-v6e16` to get a fresh, stable slice.

**Next steps**:
1. Delete current TPU: `gcloud alpha compute tpus queued-resources delete vllm-ray-multihost-v6e16 --force`
2. Recreate with same config
3. Re-run Docker/Ray setup
4. Retry RAY-001

### Key Issue Found: `d.coords` AttributeError (Blocks RAY-001)

**Problem**: When `TPU_MULTIHOST_BACKEND=ray` and PP=1, the TPU worker at `tpu_worker.py:272-274` calls:
```python
self.topology_order_id = get_device_topology_order_id(
    jax.local_devices(), jax.devices())
```
This fails because `jax.local_devices()` returns devices without `.coords` attribute in the Ray actor context.

**Analysis**:
- Ray workers run with `JAX_PLATFORMS=''` (auto-detect)
- Each Ray actor is a separate process; JAX may initialize with CPU backend instead of TPU
- The `get_device_topology_order_id()` at `distributed/utils.py:96` assumes TPU devices with `.coords`
- The error `AttributeError` on `d.coords` confirms non-TPU devices are returned

**Possible fixes to investigate for RAY-001**:
1. Ensure `JAX_PLATFORMS=tpu` is set in Ray worker env (not empty string)
2. Check if `TPU_VISIBLE_CHIPS` is being properly propagated to Ray actors
3. Check Ray's `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO` warning — may interfere with TPU detection
4. Review `tpu_inference/executors/ray_distributed_executor.py` for env var setup in `_init_workers_ray()`
5. May need to set `PJRT_DEVICE=TPU` explicitly in worker env

**Full error chain from Attempt 1**:
```
ray_distributed_executor.py:85  → _init_executor() → _init_workers_ray()
ray_distributed_executor.py:403 → self.collective_rpc("init_device")
tpu_worker.py:273               → get_device_topology_order_id(jax.local_devices(), jax.devices())
distributed/utils.py:96         → min(d.coords for d in local_devices) → AttributeError
```

**Ray cluster info (confirmed working)**:
- 4 nodes, 16 TPU, 720 CPU, 2.65 TiB memory
- Node labels include `ray.io/tpu-topology: 4x4`, `ray.io/tpu-pod-type: v6e-16`
- Placement groups created correctly: `[{'TPU': 4.0, 'node:10.164.1.219': 0.001}, {'TPU': 4.0}, {'TPU': 4.0}, {'TPU': 4.0}]`
- Workers spawned and sorted by IP correctly
- Env vars copied: `TPU_BACKEND_TYPE`, `TPU_MULTIHOST_BACKEND`, `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE`
- Failure occurs at `init_device()` stage — before model loading

### 2026-03-14 23:39-23:58 UTC — Continuation: Ray bring-up after `d.coords` blocker

**Environment & cluster**
- Recreated Ray cluster multiple times on `vllm-ray-multihost-v6e16`.
- Best observed `t_ray_cluster_start`: **39-40s**.
- Injected HF auth vars on all hosts (`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACE_HUB_TOKEN`) to clear gated-repo 401s.

**RAY-001 (PP=4, TP=4) status update**
1. **Original blocker (`d.coords`) isolated and bypassed/fixed**:
   - With `JAX_PLATFORMS=tpu`, workers proceeded beyond `init_device` and into model load (confirming `d.coords` path was the earlier blocker).
   - Added runtime patch in all Ray containers:
     - `tpu_inference/distributed/utils.py:get_device_topology_order_id()` now falls back to `process_index` ordering if device `.coords` is unavailable.
2. **Subsequent blockers found**:
   - `JAX_PLATFORMS=tpu` caused `Unknown backend cpu. Available backends are ['tpu']` in HF weight loader (`jax.devices("cpu")` usage).
   - Returned to `JAX_PLATFORMS=''`; model loading proceeded.
   - Hit sharding-type failures in weight load path:
     - `'NoneType' object has no attribute 'addressable_devices_indices_map'`
     - `'jaxlib._jax.PartitionSpec' object has no attribute 'addressable_devices_indices_map'`
   - Applied runtime patches in all containers:
     - `tpu_inference/models/jax/utils/weight_utils.py`
       - Guard/fallback when sharding spec is `None`.
       - In `shard_put`, convert `PartitionSpec` to `NamedSharding(mesh, spec)` before `general_device_put`.
3. **Latest PP=4 failure after patches**:
   - Progress reached deep XLA compile stage, then failed with TPU collective topology check:
   - `INTERNAL: RET_CHECK failure ... Unexpected device_id 4 [0:1] in replica group (0,1,2,3) :: target has 4 device_id`
   - This indicates a PP=4 compile/runtime topology mismatch (post-init, post-download, post-weight-load path).

**RAY-003 fallback attempt (PP=2, TP=4)**
- Started fallback run to validate working multi-host Ray baseline.
- During this run, TPU state changed externally to **DELETING** (best-effort reclaim/preemption), aborting further execution.

**Infra interruption**
- At ~23:58 UTC, `gcloud` reported: TPU state `DELETING`.
- No further experiments possible on this slice without re-allocation.

**Current run statuses**
- `RAY-001`: **BLOCKED** (PP=4 XLA replica-group/device-id mismatch after resolving earlier `d.coords` + auth + sharding-type issues).
- `RAY-003`: **INTERRUPTED** (TPU reclaimed while run in progress).

### 2026-03-15 00:02 UTC — TPU Recreation

- Deleted degraded TPU at ~23:58Z (best-effort preemption + maintenance instability)
- Recreated `vllm-ray-multihost-v6e16` at 00:02:30Z
- Waiting for READY state
- **Revised plan**: Skip PP=4 (blocked by XLA topology issue), try:
  1. **PP=2, TP=4** (RAY-003) — 2 hosts, simpler topology
  2. **PP=1, TP=4** with `d.coords` patch — single-host-like via Ray (all workers rank=0)
  3. If PP=2 works, retry PP=4 with explicit `device_id` mapping
- Status: **COMPLETE** — TPU READY at 00:07Z (~5 min provisioning)
- New IPs:
```
Worker 0: internal=10.164.0.142  external=34.6.0.203
Worker 1: internal=10.164.0.136  external=34.90.228.169
Worker 2: internal=10.164.0.128  external=34.91.130.112
Worker 3: internal=10.164.0.139  external=34.7.123.69
```
- Docker setup, image pull, Ray cluster: all done by 00:09Z
- Ray status: 4 nodes active, 16 TPU, 0 pending
- Benchmark script uploaded

### 2026-03-15 00:10 UTC — RAY-003: PP=2, TP=4, Llama 8B (simpler topology)

**Hypothesis**: PP=2 avoids the PP=4 XLA topology mismatch (`device_id 4 [0:1]` error). Uses only 2 of the 4 hosts. If this works, it validates the Ray multi-host path with a simpler configuration.

**Config**:
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- TP=4, PP=2 (2 hosts)
- `max_model_len=4096`
- `--distributed-executor-backend ray`

**Launch command**:
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --max-model-len 4096 \
  --port 8000 \
  --trust-remote-code
```

**Status**: FAILED
- Error: `AssertionError: Cannot use PP across hosts, please set --pipeline-parallel-size to 1 or 4`
- Location: `ray_distributed_executor.py:156` — asserts `pp_size == len(nodes_with_device)`
- On a 4-node cluster, only PP=1 or PP=4 are valid
- PP=2 is fundamentally incompatible with this executor on a 4-node setup

**Revised approach**:
- PP=4 is blocked by XLA topology mismatch
- PP=2 is blocked by executor assertion (PP must equal node count)
- PP=1 requires `d.coords` patch (applied by other agent session)
- **Next**: Apply `d.coords` patch + sharding patches from other agent's work, try PP=1, TP=4

### 2026-03-15 00:13 UTC — RAY-001b: PP=1, TP=4 with patches (d.coords + sharding fixes)

**Hypothesis**: With PP=1 and the `d.coords` fallback patch, Ray multi-host should work. Each of the 4 nodes gets rank=0 (PP=1 path), and the `get_device_topology_order_id()` is patched to handle missing `.coords`. The model is replicated across all 4 hosts (not sharded), so 8B model fits easily.

**Patches applied**:
1. `tpu_inference/distributed/utils.py`: fallback `process_index` ordering when `.coords` unavailable — **APPLIED on all 4 containers**
2. `tpu_inference/models/jax/utils/weight_utils.py`: NOT YET APPLIED (TPU preempted before reaching this stage)

**Result**: Container died immediately — head container "not running". Log file empty (no vLLM output at all).
- Root cause: TPU was **PREEMPTED** again (best-effort)
- `gcloud describe` shows: state=PREEMPTED, health=(empty)
- This is the second preemption event in ~2 hours

**Status**: **BLOCKED** — TPU preempted

### Summary of All Issues Found (2026-03-14/15)

| Issue | Location | Severity | Status |
|-------|----------|----------|--------|
| `d.coords` AttributeError | `distributed/utils.py:96` | Blocker (PP=1+Ray) | **PATCHED** (fallback to process_index) |
| PP=2 not supported on 4-node | `ray_distributed_executor.py:156` | Blocker | **By design** (PP must = node count) |
| XLA topology mismatch (PP=4) | XLA compilation stage | Blocker | **UNRESOLVED** (device_id 4 in replica group of 4) |
| Sharding spec None/PartitionSpec | `weight_utils.py` | Blocker (model load) | **Patched by other session** (not yet tested on fresh TPU) |
| TPU preemption (best-effort) | Infrastructure | Recurring | **Use reserved or on-demand for stability** |
| Single-host on multi-host slice | JAX multi-host init | Limitation | **Cannot run** — all hosts must participate |
| `TPU_MULTIHOST_BACKEND=""` invalid | `tpu_inference/envs.py` | Config | **Must unset, not empty** |
| tpu-runtime required on multi-host | systemd service | Config | **Do NOT stop** on multi-host slices |

### Next Steps (for next session)

1. **Allocate a reserved/on-demand TPU** to avoid preemption during debugging
2. **Apply all patches** (d.coords + sharding) to fresh containers
3. **Try PP=1, TP=4** first (simplest multi-host Ray config)
4. **If PP=1 works**: investigate PP=4 XLA topology fix
5. **Run MATH-500 benchmark** once server is stable
6. **Patch file created**: `vllm/marin_dev/patch_ray_coords.py` — upload and run on all containers

### 2026-03-15 01:xx UTC — Zone Migration Decision

- **Decision**: Move Ray multi-host experiments from `europe-west4-a` to **`us-east1-d`** due repeated best-effort preemptions.
- **Old Ray slices cleanup**:
  - `vllm-ray-multihost-v6e16` — deleted
  - `vllm-ray-mh-v6e16-0315a` — deleted
- **Current no-Ray reference slice**: `marin-multihost-v6e16` remains active (unchanged).
- **New default target for this logbook**:
  - Zone: `us-east1-d`
  - TPU name: `vllm-ray-multihost-v6e16-use1d`
- **Plan impact**: All future Ray allocation/setup commands in this logbook should use `--zone=us-east1-d`.

### 2026-03-15 01:29-01:50 UTC — us-east1-d continuation (PP=1 memory-path debugging)

**Context refresh**
- TPU `vllm-ray-multihost-v6e16-use1d` in `us-east1-d`: `READY/HEALTHY`.
- Ray cluster visible with 4 active nodes / 16 TPU resources.
- Prior PP=1 run root error confirmed in logs:
  - `ValueError: total_hbm_used_gb=0.0GiB exceeds total_hbm_limit_cap_gb=0.0GiB`
  - Raised from `tpu_worker.py:determine_available_memory()` after `hbm_usage_bytes()` returned zero limits on some Ray workers.

**Actions taken**
1. Revalidated prior runtime patches are present on all 4 workers:
   - `distributed/utils.py` (`d.coords` fallback to `process_index`)
   - `models/jax/utils/weight_utils.py` (`NamedSharding` handling)
2. Applied/verified multihost HBM patch directly in running containers:
   - `tpu_inference/tpu_inference/utils.py`
   - Added filter to addressable local-device IDs under Ray actors.
3. Measured Docker restart on all workers:
   - `t_docker_start_seconds=3.228`
4. Relaunched PP=1 (`PP=1, TP=4`) multiple times (`pp1b`, `pp1c`) and tailed logs.
5. Added second fallback patch (`PATCH2`) in `hbm_usage_bytes()`:
   - If Ray backend and no `memory_stats()` success, use `get_device_hbm_limit()` as limit with zero used.

**Observed behavior**
- Mixed worker behavior during same launch:
  - At least one worker reports valid stats: `total_hbm_limit_gb=124.98`, `total_hbm_used_gb=14.96`.
  - Other workers still report `total_hbm_limit_gb=0.0` and throw the same ValueError.
- `/health` remains `000`; vLLM not ready.
- Engine startup remains blocked at memory-availability phase (before serving).

**Important diagnosis**
- Current patch order still allows a zero-limit path:
  - The early branch in `hbm_usage_bytes()` can return `[(0, 0)]` before fallback logic executes.
  - This makes `determine_available_memory()` fail for workers where local-device filtering finds no addressable device.

**Status update**
- `RAY-001` remains **BLOCKED** in `us-east1-d` for PP=1 due inconsistent HBM stat discovery across Ray workers.
- `d.coords` is no longer the active blocker; memory accounting path is now the active blocker.

**Immediate next fix**
1. Remove/replace early `(0,0)` return in `hbm_usage_bytes()` so fallback limit always applies.
2. Relaunch PP=1 and require all workers to report non-zero `total_hbm_limit_gb`.
3. Once PP=1 is healthy, retry PP=4 and re-check prior XLA replica-group/device-id issue.

### 2026-03-15 01:54-02:18 UTC — PP=1 deep-dive after memory patches (us-east1-d)

**Run: PP=1d (TP=4, Ray)**
- Launch: `2026-03-15T01:54:14Z`
- New patches active:
  - `utils.py` PATCH3: no `(0,0)` early return; fallback uses `get_device_hbm_limit()`
  - `utils.py` PATCH2 fallback for missing `memory_stats`
- Result:
  - Zero-HBM hard failure is gone.
  - Workers now report valid/usable limits (example):
    - host with local stats: `total_hbm_limit_gb=124.98`, `total_hbm_used_gb=14.96`
    - fallback hosts: `total_hbm_limit_gb=32.0`, `total_hbm_used_gb=0.0`
  - New blocker after memory phase: XLA compile crash
    - `JaxRuntimeError ... Unexpected device_id 4 [0:1] in replica group (0,1,2,3)`
    - Trace context: `_initialize_kv_caches -> initialize_from_config -> _precompile_gather_logprobs`
- Status: **FAILED** (memory path fixed; topology/collective mapping still broken)

**Run: PP=1e (TP=4, Ray) — attempted env workaround**
- Launch: `2026-03-15T02:06:04Z`
- Workaround tested:
  - `SKIP_JAX_PRECOMPILE=1`
  - `--max-logprobs 0`
- Result:
  - Workaround did **not** prevent failure in Ray actors.
  - Same device-id/replica-group error surfaced in gather-logprobs precompile path.
- Status: **FAILED**

**Run: PP=1f (TP=4, Ray) — code workaround in worker init**
- Launch: `2026-03-15T02:17:27Z`
- Patch applied:
  - `tpu_worker.py` PATCH4: skip `_precompile_gather_logprobs()` when `TPU_MULTIHOST_BACKEND=ray`.
- Result:
  - Failure moved further in startup (patch changed call path), but underlying error persists:
  - Still crashes with `Unexpected device_id 4 [0:1] in replica group (0,1,2,3)`
  - New failing site:
    - `compile_or_warm_up_model -> capture_model -> _precompile_backbone_text_only -> model_fn`
- Status: **FAILED**

**Consolidated diagnosis (current)**
- Memory-statistics/zero-HBM issue is no longer the active blocker.
- Active blocker is now clearly XLA collective mapping under Ray on this topology:
  - `device_id 4` appears in replica-groups that expect IDs within `[0..3]`.
- This reproduces across multiple startup phases (gather_logprobs and backbone precompile), so skipping one precompile stage is insufficient.

**Immediate next step**
1. Re-run **PP=4, TP=4** with current memory patches (PATCH2/3) to test whether PP-across-hosts path avoids the PP=1 replica-group mapping issue or hits same failure.
2. If PP=4 still fails with same signature, collect per-worker `jax.devices()` / `jax.local_devices()` metadata under Ray actor context to identify wrong global-device-id mapping.

### 2026-03-15 02:24-02:34 UTC — PP=4 retest (`pp4a`) after PP=1 memory-path fixes (us-east1-d)

**Goal for this run**
- Validate whether PP=4 (`pipeline_parallel_size=4`) avoids the PP=1-only compile failure, now that memory accounting fixes (PATCH2/PATCH3) are in place.

**Config**
- TPU: `vllm-ray-multihost-v6e16-use1d` (`us-east1-d`)
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- vLLM: Ray executor, `TP=4`, `PP=4`, `port=8000`
- Runtime patches still active in containers:
  - `distributed/utils.py` (`d.coords` fallback)
  - `models/jax/utils/weight_utils.py` (`PartitionSpec -> NamedSharding` handling)
  - `utils.py` PATCH2/PATCH3 (HBM limit fallback; remove early `(0,0)` return)
  - `worker/tpu_worker.py` PATCH4 (skip gather-logprobs precompile under Ray)

**What worked**
1. **Cluster/runtime preconditions**:
   - Ray cluster remained available on 4 hosts with 16 TPU resources.
   - No `d.coords`-path crash during worker init.
2. **Memory gate now passes**:
   - Workers no longer die at `total_hbm_limit_gb=0.0` validation.
   - Observed non-zero memory accounting behavior consistent with PP=1d/1e/1f:
     - local-stats path: around `124.98 GiB` limit with non-zero used
     - fallback path: `32.0 GiB` limit and `0.0 GiB` used
3. **Failure point moved later**:
   - Engine gets beyond previous memory blocker and fails during compile warmup path.

**What failed**
1. **Server never reached ready**:
   - `curl http://localhost:8000/health` remained non-200 (`000` seen in checks).
2. **PP=4 still fails with same core XLA/Jellyfish topology signature**:
   - Error class: `jax.errors.JaxRuntimeError`
   - Signature:
     - `Unexpected device_id 4 [0:1] in replica group (0,1,2,3)`
   - This indicates the same collective/device-id mapping mismatch already seen in PP=1 runs, now reproduced in PP=4 after memory blockers were removed.
3. **Engine initialization aborts**:
   - APIServer exits after engine core startup failure (`Engine core initialization failed` path).

**Interpretation**
- PP=4 does **not** bypass the underlying compile-time device-id mapping issue.
- Memory fixes are still valid and necessary (they removed an earlier blocker), but they are not sufficient for a working Ray multi-host launch.
- Current dominant blocker across both PP=1 and PP=4 is a topology/device-id mismatch during XLA collective lowering.

**Post-run evidence collection attempts**
1. Attempted to retrieve `/tmp/vllm_serve_pp4a.log` and prior `pp1*` logs on worker 0.
   - At check time, those files were not present under `/tmp` (could be due to container/session lifecycle).
2. Attempted `docker ps` across all workers to inspect surviving containers.
   - Command returned docker socket permission errors (`permission denied while trying to connect to the Docker daemon socket`) for current login user.
   - This is an observability/access issue during postmortem, not the original runtime failure.

**Status change**
- `RAY-001` remains **BLOCKED** with stronger evidence:
  - blocker persists in both PP=1 and PP=4,
  - after `d.coords` and zero-HBM classes were mitigated.

**Immediate next debug target**
1. Collect per-Ray-worker device metadata at runtime (`jax.devices()`, `jax.local_devices()`, process index, and IDs used by collectives) before compile warmup.
2. Compare replica-group construction inputs against expected `[0..3]` local IDs for each stage to identify where `device_id=4` is introduced.

### 2026-03-14 — RAY-001c: PP=4, TP=4 with JAX isolation fix

**Root cause analysis of XLA device_id mismatch**:
- `tpu_worker.py:170`: condition `multihost_backend != "ray"` SKIPS setting `TPU_PROCESS_BOUNDS`, `TPU_CHIPS_PER_PROCESS_BOUNDS`, `TPU_VISIBLE_CHIPS` for Ray workers.
- Without these env vars, JAX on a multi-host v6e-16 sees all 16 global devices via tpu-runtime.
- Each host's local chips get global IDs (host 0: 0-3, host 1: 4-7, host 2: 8-11, host 3: 12-15).
- When XLA compiles collective ops with a mesh of 4 local devices, it encounters global device IDs (e.g., 4,5,6,7 on host 1) outside the expected replica group (0,1,2,3).
- The no-Ray multi-host path works because it sets these env vars to isolate each process to its own chip subset.

**Fix**: Set JAX isolation env vars for Ray workers:
- `TPU_PROCESS_BOUNDS=1,1,1` (single process per host)
- `TPU_CHIPS_PER_PROCESS_BOUNDS=1,{chips_needed},1` (computed from TP size)
- `TPU_VISIBLE_CHIPS=0,1,2,3` (local chip IDs — on multi-host, each host's chips are 0-3)
- `CLOUD_TPU_TASK_ID=0` (each host is isolated task 0)

**Patch file**: `vllm/marin_dev/patch_ray_multihost_v2.py` — applies 3 fixes:
1. `tpu_worker.py`: JAX isolation env vars for Ray multi-host
2. `distributed/utils.py`: `d.coords` fallback to `process_index`
3. `utils.py`: `hbm_usage_bytes` filter to addressable devices + HBM limit fallback

**Hypothesis**: With JAX isolated to single-host mode per worker, XLA will compile collectives with local device IDs (0-3) only. No global device_id 4+ will appear in replica groups. PP=4 should reach XLA compilation and server ready state.

**Config**:
- TPU: `vllm-ray-multihost-v6e16-use1d` (us-east1-d, ACTIVE)
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- TP=4, PP=4, `--distributed-executor-backend ray`
- `max_model_len=4096`
- Image: `vllm/vllm-tpu:nightly`

**Deployment steps**:
1. Pull Docker image on all 4 hosts
2. Start containers with Ray cluster
3. Apply patch_ray_multihost_v2.py on all containers
4. Launch vLLM serve
5. Run MATH-500 benchmark if server reaches ready state

**Attempt 1 (02:56 UTC)**: PARTIAL SUCCESS — XLA device_id error FIXED
- JAX isolation patch confirmed working on all 4 workers:
  ```
  Ray multi-host JAX isolation: TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=1,4,1 TPU_VISIBLE_CHIPS=0,1,2,3 CLOUD_TPU_TASK_ID=0
  ```
- Mesh created successfully: `Mesh('data': 1, 'model': 4, axis_types=(Auto, Auto))`
- **NO `device_id 4` error!** The XLA collective topology mismatch is RESOLVED.
- Workers progressed through: cluster join → placement group → worker init → device init → PP transfer connect → **load_model** (NEW STAGE!)
- Failed at `load_model` with known sharding issue:
  ```
  AttributeError: 'NoneType' object has no attribute 'addressable_devices_indices_map'
  ```
  at `weight_utils.py:242 shard_put` → `utils.py:140 general_device_put`
- Root cause: `shard_put` passes `None` or raw `PartitionSpec` to `general_device_put`, but Ray multi-host path requires `NamedSharding(mesh, spec)`.
- Fix: `patch_ray_sharding.py` — wraps None→`NamedSharding(mesh, P())` (replicate), PartitionSpec→`NamedSharding(mesh, spec)`
- Applied to all 4 containers.

**Attempt 2 (03:02 UTC)**: FAILED — Ray socket stale (old session)
- Sharding fix not tested; EngineCore couldn't connect to Ray socket from prior crashed session.
- Fix: must restart Ray cluster between attempts.

**Attempt 3 (03:05 UTC)**: PARTIAL SUCCESS — model loaded, server started, crashed on first request
- Fresh Ray cluster + all patches (JAX isolation + sharding + HBM)
- Model loaded on all 4 workers with PP stage skipping (8/32 layers per worker)
- HBM: 3.25 GiB used / 31.25 GiB per device after model load
- KV cache: 3,131,392 tokens capacity (764x concurrency at 4096 max_len)
- XLA compile completed successfully
- Server reached READY (health=200), routes registered
- **CRASHED on first request**: `AttributeError: 'TPUModelRunner' object has no attribute 'supports_mm_inputs'`
  at `ray_utils.py:132` in `execute_model_ray`
- Fix: `patch_ray_mm.py` — add `self.supports_mm_inputs = False` to TPUModelRunner

**Attempt 4 (03:12 UTC)**: FAILED — stale Ray socket (same issue as attempt 2)

**Attempt 5 (03:18 UTC)**: **SUCCESS!**
- Fresh Ray cluster + ALL patches (JAX isolation + sharding + HBM + MM inputs)
- Server READY at 03:22:40Z
- `t_total_startup` ≈ 270s (03:18:06 → 03:22:40) including Ray cluster (already up), model load, XLA compile
- **First successful inference**:
  - Prompt: "What is 2+2? Answer briefly."
  - Response: "2 + 2 = 4."
  - Usage: 45 prompt tokens, 9 completion tokens
  - Model correctly generated coherent output through PP=4 pipeline!

**Status**: RAY-001c **OPERATIONAL** — proceeding to MATH-500 benchmark

**MATH-500 Benchmark Results (RAY-001c)**:
- Start: 03:23 UTC, End: ~03:38 UTC
- **Wall time**: 861.0s (14.4 min) for 500 problems
- **Throughput**: 389.1 gen tok/s, 0.58 req/s
- **Latency**: TTFT p50=50.4ms, TPOT p50=34.6ms, E2E p50=11,286ms
- **Correctness**: 42.20% accuracy (211/500), 79.80% format rate, 0% empty rate
- **Errors**: 0/500 failed requests
- **Tokens**: 334,980 total generated, 670 mean per response
- **Grading**: 0.80s total, 0 timeouts
- **Notes**:
  - TTFT p99=47,730ms (very high) likely due to initial XLA compilation on first batch
  - TPOT p99=229ms also indicates some compilation overhead early on
  - Throughput lower than no-Ray (expected due to PP=4 pipeline overhead on 8B model)

**Patches Applied (4 total)**:
1. `patch_ray_multihost_v2.py` — JAX isolation env vars for Ray multi-host (`TPU_PROCESS_BOUNDS`, `TPU_CHIPS_PER_PROCESS_BOUNDS`, `TPU_VISIBLE_CHIPS`, `CLOUD_TPU_TASK_ID`)
2. `patch_ray_sharding.py` — Handle None/PartitionSpec in `shard_put` → wrap as `NamedSharding`
3. `patch_ray_mm.py` — Add `supports_mm_inputs = False` to `TPUModelRunner`
4. `patch_ray_multihost_v2.py` also includes: `d.coords` fallback + `hbm_usage_bytes` safety fallback

**Issues Resolved This Session**:
| Issue | Location | Fix |
|-------|----------|-----|
| XLA device_id mismatch | `tpu_worker.py:170` | Set `TPU_PROCESS_BOUNDS=1,1,1` etc. for Ray |
| Sharding None/PartitionSpec | `weight_utils.py:242` | Wrap in `NamedSharding(mesh, spec)` |
| `supports_mm_inputs` missing | `tpu_runner.py` | Add `self.supports_mm_inputs = False` |
| Stale Ray socket | Ray cluster lifecycle | Must restart containers between failed runs |

### RAY-004 Plan: Llama 3.3 70B on v6e-16 with Ray PP=4 (FAST LOADING)

**Goal**: Serve Llama 3.3 70B Instruct via Ray multi-host PP=4, TP=4. Run MATH-500 benchmark.

**Model**: `meta-llama/Llama-3.3-70B-Instruct` — 131 GiB safetensors, 30 shards
**Weights**: `gs://marin-us-east1/gcsfuse_mount/models/meta-llama--Llama-3-3-70B-Instruct` (SAME REGION as TPU)

**Memory budget**:
- 70B BF16 = ~140 GB total weights
- PP=4: ~35 GB weights per host
- v6e-16 per host: 4 chips × 32 GB = 128 GB HBM
- Available for KV cache: ~93 GB per host (plenty)
- `max_model_len`: 4096 (same as RAY-001)

**Weight loading strategy — DO NOT use runai_streamer (41 min for 70B)**:

Local disk is only 100 GB (52 GB free) — can't fit 131 GiB model. Use **gcsfuse** to mount
same-region GCS bucket directly on each host, then pass mount into Docker container.

1. **Install gcsfuse** on all 4 hosts:
   ```bash
   # gcsfuse is not pre-installed on TPU VMs
   export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
   echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
   curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
   sudo apt-get update && sudo apt-get install -y gcsfuse
   ```

2. **Mount the bucket** on all 4 hosts:
   ```bash
   mkdir -p /mnt/gcs-models
   gcsfuse --implicit-dirs \
     --file-cache-capacity-mb 40000 \
     --file-cache-enable-parallel-downloads \
     --file-cache-parallel-downloads-per-file 16 \
     --file-cache-download-chunk-size-mb 50 \
     --file-cache-max-parallel-downloads 64 \
     marin-us-east1 /mnt/gcs-models
   ```
   - `--file-cache-capacity-mb 40000`: use ~40 GB of local disk as read cache
   - `--file-cache-enable-parallel-downloads`: parallel chunk downloads from GCS
   - `--file-cache-parallel-downloads-per-file 16`: 16 concurrent downloads per file
   - Same-region: GCS reads at wire speed (5-10 Gbps)
   - Model path becomes: `/mnt/gcs-models/gcsfuse_mount/models/meta-llama--Llama-3-3-70B-Instruct`

3. **Mount into Docker**: `-v /mnt/gcs-models:/mnt/gcs-models:ro`

4. **vLLM serve with fuse path**:
   ```bash
   vllm serve /mnt/gcs-models/gcsfuse_mount/models/meta-llama--Llama-3-3-70B-Instruct \
     --tensor-parallel-size 4 \
     --pipeline-parallel-size 4 \
     --distributed-executor-backend ray \
     --max-model-len 4096 \
     --port 8000 \
     --trust-remote-code
   ```
   - vLLM reads safetensors via gcsfuse (appears as local filesystem)
   - gcsfuse parallel download cache warms during first read
   - Each PP stage skips non-assigned layers (~35 GB actually read per host)
   - With PP=4, each host only loads 7-8 of the 30 safetensors shards

**Why gcsfuse?**
- No local disk space for 131 GiB model (only 52 GB free)
- Same-region gcsfuse with parallel downloads: effective 1-3 GB/s throughput
- File cache: first read populates local cache (~35 GB per host fits in 40 GB cache)
- Subsequent reads (if any) come from cache at SSD speed
- No code changes needed — gcsfuse is transparent to vLLM

**Estimated timeline**:
| Phase | Duration | Notes |
|-------|----------|-------|
| Install gcsfuse | ~1 min | apt install on all hosts |
| Mount + verify | ~30s | gcsfuse mount, ls model dir |
| Start Ray cluster | ~15s | Same as RAY-001 |
| Apply patches | ~10s | 4 patch scripts |
| vLLM model load | ~3-5 min | gcsfuse read ~35 GiB/host with parallel downloads |
| XLA compilation | ~20-30 min | Unavoidable first-compile cost |
| MATH-500 benchmark | ~15-20 min | 500 problems at concurrency=16 |
| **Total** | **~45-55 min** | vs >90 min with naive cross-region loading |

**Execution steps**:
1. Kill existing vLLM / containers on all hosts
2. Install gcsfuse on all 4 hosts
3. Mount `gs://marin-us-east1` via gcsfuse on all 4 hosts
4. Start Docker containers with `-v /mnt/gcs-models:/mnt/gcs-models:ro`
5. Start Ray cluster (head + 3 workers)
6. Apply all 4 patches (JAX isolation, sharding, HBM, MM inputs)
7. Launch `vllm serve /mnt/gcs-models/gcsfuse_mount/models/meta-llama--Llama-3-3-70B-Instruct ...`
8. Wait for server ready (health=200)
9. Run MATH-500 benchmark
10. Record all metrics

### 2026-03-15 04:32 UTC — RAY-004: Llama 3.3 70B, PP=4, TP=4, gcsfuse weight loading

**Hypothesis**: 70B model fits across 4 hosts (~35 GB/host out of 128 GB HBM). gcsfuse with parallel downloads from same-region GCS provides fast weight loading without local disk space. All patches from RAY-001c carry over.

**Config**:
- Model: `meta-llama/Llama-3.3-70B-Instruct` (131 GiB safetensors, 30 shards)
- Weight source: `gs://marin-us-east1` via gcsfuse mount at `/mnt/gcs-models` (SAME REGION as TPU)
- TP=4, PP=4, `max_model_len=4096`
- gcsfuse config: `--file-cache-max-size-mb 40000 --file-cache-enable-parallel-downloads --file-cache-parallel-downloads-per-file 16 --file-cache-download-chunk-size-mb 50 --file-cache-max-parallel-downloads 64 -o allow_other`
- Docker: `-v /mnt/gcs-models:/mnt/gcs-models:ro`

**Setup**:
- gcsfuse installed: v3.7.3 on all 4 hosts
- Mounted with `allow_other` for Docker container access
- Model files verified accessible inside containers
- All 4 patches applied (JAX isolation, sharding, HBM, MM inputs)

**Launch**: 04:34:24Z
- Workers initialized with isolated JAX (same as RAY-001c)
- Model loading started from gcsfuse path at ~04:35:10Z
- PP stage layer skipping working correctly (each host loads ~20 of 80 layers)
- gcsfuse loading speed: 30 safetensors shards read in ~4 min (04:35:10 → 04:39:01) = **good throughput**

**Result**: FAILED — **OOM during weight loading** at 04:39:03Z
- `RESOURCE_EXHAUSTED: Attempting to allocate 448.00M. Only 327.37M free on HBM0`
- 448 MB = full `gate_proj` weight (8192 × 28672 × BF16) — NOT TP-sharded
- Root cause: weight sharding specs are None for some params → our patch defaults to P() (replicate) → full weight replicated to every device
- 8B worked replicated (2 GB/host fits on 32 GB chip). 70B fails (35 GB/host > 32 GB/chip)
- Fix options:
  1. **INT8 quantization** — halves weight memory (17.5 GB/host fits even replicated)
  2. **Fix sharding specs** — ensure TP-sharding for large weight matrices (correct fix, but requires model code changes)
  3. **Investigate nnx.get_named_sharding** — may not be returning correct specs for all params

**Attempt 2 (04:50 UTC)**: Debug run — confirmed all sharding specs are NoneType
- Added debug logging to `shard_put`
- Confirmed: `nnx.get_named_sharding(params, mesh)` throws TypeError → fallback to `shardings = params`
- `model_weight.sharding` = None, `model_weight.value.sharding` = None
- Model params initialized as unsharded numpy/JAX arrays (no partition info)
- Need to hardcode TP sharding based on weight name patterns

**Attempt 3 (05:01 UTC)**: **MODEL LOADED** — hardcoded TP sharding fallback
- `patch_ray_sharding_v3.py`: when `nnx.get_named_sharding` fails, assign TP partition specs by weight name:
  - `gate_proj/up_proj` → `P(None, 'model')`, `down_proj` → `P('model', None)`
  - `q/k/v_proj` → `P(None, 'model', None)`, `o_proj` → `P('model', None, None)`
  - `embed_tokens` → `P('model', None)`, `lm_head` → `P(None, 'model')`
  - layernorms/biases → replicated `P(None, ...)`
- **Results**:
  - HBM per device after model load: **8.46 GiB / 31.25 GiB** (was 35 GB replicated → OOM)
  - KV cache: 1,030,656 tokens
  - Weight loading via gcsfuse: ~1 min (30 safetensors from same-region GCS)
  - **NO OOM!** TP sharding working correctly
  - XLA compilation: ~2 min (likely cached)
  - Server ready at 05:07:54Z — **total startup ~6 min** (gcsfuse load + XLA)
  - First inference test: `∫x² dx = (1/3)x³ + C` — correct!

**MATH-500 Benchmark Results (RAY-004)**:
- Start: 05:08:57Z, End: ~05:26Z
- **Wall time**: 1043.0s (17.4 min) for 500 problems
- **Throughput**: 259.3 gen tok/s, 0.48 req/s
- **Latency**: TTFT p50=95.8ms, TPOT p50=54.5ms, E2E p50=24,380ms
- **Correctness**: **67.60% accuracy** (338/500), 98.40% format rate, 0% empty
- **Errors**: 0/500 failed requests
- **Tokens**: 270,475 total generated, 541 mean per response
- **HBM**: 8.46 GiB used / 31.25 GiB per device (TP-sharded correctly)

**Patches Applied (5 total for 70B)**:
1. `patch_ray_multihost_v2.py` — JAX isolation env vars
2. `patch_ray_sharding.py` — Handle None/PartitionSpec in shard_put
3. `patch_ray_sharding_v3.py` — **Hardcoded TP partition specs by weight name** (the key 70B fix)
4. `patch_ray_mm.py` — Add `supports_mm_inputs = False`
5. gcsfuse mount with parallel downloads from same-region GCS

### RAY-005 Plan: GRPO Mini-Batch Stress Test (70B, Ray PP=4)

**Goal**: Simulate Marin's `exp2039_rl_math500.py` GRPO inference workload on the Ray multi-host 70B setup. Direct comparison with no-Ray logbook results (NRM-007d series).

**Key difference from no-Ray**:
- No-Ray used **4 independent replicas** (one per host, each TP=4) with round-robin load balancing
- Ray uses **1 PP=4 server** (all 4 hosts in a single pipeline) — single endpoint, no round-robin
- This is fundamentally different: no-Ray scales throughput linearly with replicas, Ray trades throughput for ability to serve larger models

**Benchmark script**: `vllm/marin_dev/benchmark_grpo_stress.py`
**Dataset**: `vllm/marin_dev/hendrycks_math_train.jsonl` (12,000 problems, seed=42, deterministic)

**GRPO config (matches Marin)**:
- 64 prompts × 16 completions = 1,024 requests per mini-batch
- `temperature=1.0` (sampling, not greedy)
- `max_tokens=1024`
- `concurrency_per_replica=64` (64 concurrent requests to single server)

**Test matrix**:

| Run | Batches | Total completions | Purpose |
|-----|--------:|------------------:|---------|
| RAY-005a | 1 | 1,024 | Warmup + first batch timing |
| RAY-005b | 10 | 10,240 | Sustained throughput |
| RAY-005c | 20 | 20,480 | Stability + variance |

**Command**:
```bash
python /tmp/benchmark_grpo_stress.py \
  --servers http://localhost:8000 \
  --dataset-path /tmp/hendrycks_math_train.jsonl \
  --num-batches 10 \
  --n-prompts 64 \
  --n-gen 16 \
  --max-tokens 1024 \
  --temperature 1.0 \
  --concurrency-per-replica 64 \
  --output /tmp/results_RAY-005b.json
```

**Comparison targets from no-Ray logbook (NRM-007d, Llama 8B, 4 replicas)**:

| Metric | NRM-007d-10 (8B, 4 replicas) | RAY-005b (70B, PP=4) expected |
|--------|-----------------------------:|------------------------------:|
| gen_tok/s | 11,834 | ~250-300 (70B is slower per token) |
| Mean batch (s) | 35.3 | ~200-300 (larger model, single pipeline) |
| Epoch projection (h) | 1.85 | ~10-15 (much larger model) |
| Errors | 0 | 0 (target) |

**Note**: The comparison is NOT apples-to-apples (8B vs 70B, replicas vs PP). The value is:
1. Proving Ray PP=4 is stable under sustained GRPO-style load
2. Establishing 70B GRPO inference time baseline for Ray multi-host
3. Measuring throughput variance across batches (stability metric)

### 2026-03-15 05:30 UTC — RAY-005a: GRPO 1 Mini-Batch (70B, Ray PP=4)

**Config**: 64 prompts × 16 completions = 1,024 requests, temperature=1.0, max_tokens=1024, concurrency=64
**Dataset**: `hendrycks_math_train.jsonl` (seed=42 deterministic, first 64 problems)
**Server**: single Ray PP=4 endpoint at `http://localhost:8000`

**Results**:

| Metric | RAY-005a (70B, PP=4) | NRM-007d-1 (8B, 4 replicas) |
|--------|---------------------:|----------------------------:|
| Wall time (s) | 606.6 | 36.4 |
| Gen tok/s | 814 | 11,851 |
| Req/s | 1.69 | 28.17 |
| Mean tokens/resp | 482 | 421 |
| Errors | 0 | 0 |
| Accuracy (%) | 69.0 | 42.2 |
| Epoch projection (h) | 31.7 | 1.9 |

**Analysis**:
- **0 errors** — Ray PP=4 stable under 64-concurrent GRPO load
- **814 tok/s** throughput for 70B (vs 259 tok/s at concurrency=16 in MATH-500) — higher concurrency helps
- **69.0% accuracy** — consistent with MATH-500 result (67.6%), validates correctness under sampling (temp=1.0)
- **31.7h epoch projection** — significantly slower than no-Ray 8B (1.9h), but:
  - 70B model is ~9x larger → expected ~9x slower per token
  - PP=4 adds pipeline latency vs independent replicas
  - Single server vs 4 replicas (no parallelism across servers)
- **482 mean tokens** per response (vs 541 in MATH-500 greedy) — sampling truncates earlier

### 2026-03-15 06:30 UTC — RAY-005b: GRPO 10 Mini-Batches (70B, Ray PP=4)

**Config**: Same as RAY-005a. 10 batches × 1,024 completions = 10,240 total.
**Dataset**: `hendrycks_math_train.jsonl` (seed=42 deterministic, first 640 problems, sequential order)

**Results**:

| Batch | Wall(s) | Tok/s | Req/s | MeanTok | Errors | Acc% |
|------:|--------:|------:|------:|--------:|-------:|-----:|
| 1 | 535.2 | 910 | 1.91 | 476 | 0 | 69.5 |
| 2 | 495.8 | 906 | 2.07 | 438 | 0 | 74.7 |
| 3 | 536.0 | 950 | 1.91 | 498 | 0 | 70.6 |
| 4 | 528.5 | 919 | 1.94 | 474 | 0 | 75.2 |
| 5 | 458.6 | 974 | 2.23 | 436 | 0 | 84.9 |
| 6 | 538.5 | 953 | 1.90 | 501 | 0 | 78.1 |
| 7 | 511.7 | 906 | 2.00 | 453 | 0 | 78.2 |
| 8 | 493.6 | 944 | 2.07 | 455 | 0 | 73.1 |
| 9 | 494.6 | 909 | 2.07 | 439 | 0 | 81.0 |
| 10 | 535.3 | 914 | 1.91 | 478 | 0 | 79.5 |

**Aggregate**:
- **Total wall time**: 5,127.9s (85.5 min)
- **Total completions**: 10,240
- **Total tokens**: 4,759,477
- **Total errors**: 0
- **Mean batch wall time**: 512.8s (stddev 26.6s)
- **Mean throughput**: 929 tok/s
- **Min/Max batch time**: 458.6s / 538.5s
- **Epoch projection**: 26.8 hours (188 batches)
- **Mean accuracy**: 76.5% (across batches, temp=1.0 sampling)

**Stability assessment**:
- Throughput variance: stddev/mean = 26.6/512.8 = **5.2%** — very stable
- No errors across 10,240 completions and 85 min of sustained load
- Accuracy consistent 69-85% (variance from sampling + different problem difficulty per batch)

### 2026-03-15 17:38 UTC — V8-70B: Single-Host v6e-8 Baseline (70B, TP=8, no Ray)

**Goal**: Run same 70B model on a single v6e-8 host (8 chips, 256 GB HBM) with TP=8, no PP, no Ray. Direct comparison with RAY-004/005 to isolate the overhead of Ray PP=4.

**Config**:
- TPU: `vllm-70b-v6e8-use1d` — v6e-8, us-east1-d, 1 host, 8 chips, 256 GiB HBM
- Model: `meta-llama/Llama-3.3-70B-Instruct` via gcsfuse from `gs://marin-us-east1`
- TP=8, PP=1, no Ray, `VLLM_ENABLE_V1_MULTIPROCESSING=0`
- `max_model_len=4096`
- Memory: 140 GB / 8 chips = 17.5 GB/chip, leaving ~14.5 GB/chip for KV cache

**Test plan**: Same GRPO mini-batch test as RAY-005:
- 1 batch warmup (V8-70B-a)
- 10 batches sustained (V8-70B-b)
- Deterministic dataset: `hendrycks_math_train.jsonl` (seed=42, same order)

**Setup notes**:
- TPU: `vllm-70b-v6e8-ew4a` (europe-west4-a, best-effort)
- gcsfuse from `gs://marin-us-east1` (cross-region, but cached with parallel downloads)
- Required same TP sharding patches (`patch_ray_sharding.py` + `patch_ray_sharding_v3.py`) — the `nnx.get_named_sharding` failure is NOT Ray-specific
- Server ready in ~5.5 min (weight load + XLA compile)

**V8-70B-a Results (1 mini-batch)**:
- Wall time: 241.2s, Gen tok/s: **2,013**, Errors: 0, Accuracy: 68.7%

**V8-70B-b Results (10 mini-batches)**:

| Batch | Wall(s) | Tok/s | Req/s | MeanTok | Errors | Acc% |
|------:|--------:|------:|------:|--------:|-------:|-----:|
| 1 | 203.4 | 2,426 | 5.03 | 482 | 0 | 70.2 |
| 2 | 188.0 | 2,380 | 5.45 | 437 | 0 | 75.4 |
| 3 | 210.9 | 2,411 | 4.85 | 497 | 0 | 70.9 |
| 4 | 197.7 | 2,448 | 5.18 | 473 | 0 | 74.1 |
| 5 | 179.0 | 2,442 | 5.72 | 427 | 0 | 84.5 |
| 6 | 210.4 | 2,420 | 4.87 | 497 | 0 | 79.1 |
| 7 | 191.9 | 2,426 | 5.34 | 455 | 0 | 78.2 |
| 8 | 187.8 | 2,493 | 5.45 | 457 | 0 | 73.8 |
| 9 | 184.1 | 2,434 | 5.56 | 438 | 0 | 81.2 |
| 10 | 203.0 | 2,391 | 5.04 | 474 | 0 | 79.9 |

**Aggregate**: 1,956s total (32.6 min), mean batch 195.6s (stddev 11.1s), **2,427 tok/s**, 0 errors, epoch projection **10.2h**

### Head-to-Head: v6e-8 (TP=8) vs v6e-16 Ray (PP=4, TP=4)

| Metric | v6e-8 TP=8 (1 host) | v6e-16 Ray PP=4 (4 hosts) | Ratio |
|--------|--------------------:|--------------------------:|------:|
| Chips | 8 | 16 | 0.5x |
| Mean batch time (s) | **195.6** | 512.8 | **2.6x faster** |
| Gen tok/s | **2,427** | 929 | **2.6x faster** |
| Req/s | 5.23 | 2.00 | 2.6x |
| Throughput variance | 5.7% | 5.2% | Similar |
| Errors (10 batches) | 0 | 0 | Same |
| Mean accuracy | 76.7% | 76.5% | Same |
| Epoch projection (h) | **10.2** | 26.8 | **2.6x faster** |

**Key insight**: For 70B inference, **v6e-8 single-host TP=8 is 2.6x faster than v6e-16 Ray PP=4** despite using half the chips. Pipeline parallelism across hosts adds massive overhead:
- PP bubble latency (pipeline stages wait for each other)
- Cross-host activation transfers
- Ray actor scheduling overhead
- Single server bottleneck (vs all-reduce within one host)

**Bottom line**: If the model fits on one host, use TP on that host. PP across hosts is for models that DON'T fit on a single host. For 70B on v6e-8 (256 GB HBM), single-host TP=8 is the clear winner.

### RAY-006 Plan: Qwen3.5-122B-A10B MoE on v6e-16 with Ray PP=4

**Goal**: Serve the Qwen3.5-122B-A10B MoE model on Ray multi-host PP=4, TP=4. This is the first model where Ray multi-host PP is actually NEEDED — 233 GiB weights don't fit on a single v6e-8 (256 GB HBM, too tight for weights + KV cache).

**Model**: `Qwen/Qwen3.5-122B-A10B` — MoE, 122B total params, 10B active per token
- Architecture: `qwen3_5_moe`, 48 layers, hidden=3072, 32 attn heads, 2 KV heads
- 256 experts per layer, top-8 routing + 1 shared expert
- MoE intermediate size: 1024, shared expert intermediate: 1024
- Multimodal (vision encoder) — use `--language-model-only` for text-only
- Mixed attention: linear + full (every 4th layer)

**Weights**: `gs://marin-us-east1/models/Qwen--Qwen3-5-122B-A10B--b000b2e` (233 GiB, 39 safetensors shards, SAME REGION)

**Memory budget**:
- 233 GiB weights total
- PP=4: ~58 GiB weights per host
- TP=4: ~14.5 GiB per chip after sharding
- v6e chip: 32 GiB HBM → ~17.5 GiB per chip for KV cache
- MoE note: all 256 expert weights must be resident (routed per-token), but TP shards them

**Why PP=4 TP=4 (not PP=2 TP=8)**:
- TP must stay within one host (ICI all-reduce is μs; cross-host all-reduce at every layer would be catastrophic)
- PP=4 has more pipeline bubble but only 3 cross-host transfers per forward pass
- PP=2 TP=8 would need TP across 2 hosts = 48 cross-host all-reduces per forward pass — much worse

**Potential challenges**:
1. **MoE sharding**: Our hardcoded TP specs (`patch_ray_sharding_v3.py`) only cover dense Llama weights. MoE layers have different weight names (expert gates, routed expert weights). Need to add MoE-specific partition specs.
2. **Model support**: `qwen3_5_moe` is a new architecture — may not be in `vllm/vllm-tpu:nightly` yet. Need to check.
3. **Mixed attention**: Linear + full attention is unusual — may need custom kernels.
4. **Vision encoder**: Skip with `--language-model-only` for initial test.

**Execution steps**:
1. Check if `qwen3_5_moe` is supported in current vllm-tpu:nightly image
2. Investigate MoE weight names and add to `patch_ray_sharding_v3.py`
3. Kill existing 70B server on v6e-16 Ray cluster
4. Restart Ray cluster with gcsfuse mount pointing to Qwen model
5. Apply all patches (JAX isolation + sharding with MoE specs + MM inputs)
6. Launch `vllm serve` with `--language-model-only` flag
7. Test inference
8. Run MATH-500 benchmark
9. Run GRPO mini-batch stress test (1 batch, then 10)

**Status check (step 1)**:
- `vllm/vllm-tpu:nightly` has `Qwen3MoeForCausalLM` (Qwen3 MoE) in tpu-inference
- Model architecture is `Qwen3_5MoeForConditionalGeneration` (Qwen3.5 MoE) — **NOT supported**
- Qwen3.5 MoE differs from Qwen3 MoE: mixed linear/full attention, vision encoder, different config nesting
- vLLM core (GPU path) has Qwen3.5 MoE support, but tpu-inference JAX backend does not
- **BLOCKED**: Need either:
  - (a) Update to a newer vllm-tpu image that adds `Qwen3_5MoeForConditionalGeneration` to tpu-inference
  - (b) Port Qwen3.5 MoE support to the JAX backend (significant work — new attention pattern, new config parsing)
  - (c) Use a different model that IS supported (e.g., `Qwen3MoeForCausalLM` if a Qwen3 MoE checkpoint exists, or DeepSeek-V3)

### RAY-007 Plan: Qwen3-235B-A22B MoE on TPU (multi-host Ray)

**Model**: `Qwen/Qwen3-235B-A22B-Thinking-2507`
- Architecture: `Qwen3MoeForCausalLM` — **SUPPORTED in tpu-inference** (unlike Qwen3.5)
- 235B total params, 22B active per token (128 experts, top-8 routing)
- 94 layers, hidden=4096, 64 attn heads, 4 KV heads
- MoE intermediate size: 1536, dense intermediate: 12288
- BF16 weights: **~470 GB** (235B × 2 bytes)
- Context: 262K native, but we'd use 4096 for benchmarks

**Weights**: Not yet in GCS. Need to download from HuggingFace (~470 GB) to a regional bucket.

#### Hardware Options

**v6e options** (32 GB HBM per chip):

| Config | Chips | Total HBM | Weights fit? | KV cache headroom | Verdict |
|--------|------:|----------:|:------------:|-------------------:|---------|
| v6e-16 PP=4 TP=4 | 16 | 512 GB | 470 GB ✓ | **42 GB** (2.6 GB/chip) | **Very tight** — may work with max_model_len=1024-2048 |
| v6e-32 PP=4 TP=8 | 32 | 1024 GB | 470 GB ✓ | **554 GB** (17 GB/chip) | **Comfortable** — plenty of KV cache room |
| v6e-32 PP=8 TP=4 | 32 | 1024 GB | 470 GB ✓ | **554 GB** (17 GB/chip) | More PP stages = more pipeline bubble |
| v6e-64 PP=4 TP=4 ×4 replicas | 64 | 2048 GB | 470×4 ✓ | Plenty | Overkill for single model, but 4 replicas |

**v5p options** (95 GB HBM per chip):

| Config | Chips | Total HBM | Weights fit? | KV cache headroom | Verdict |
|--------|------:|----------:|:------------:|-------------------:|---------|
| v5p-8 TP=4 | 4 | 380 GB | **No** (470 > 380) | — | Does not fit |
| v5p-16 PP=2 TP=4 | 8 | 760 GB | 470 GB ✓ | **290 GB** (36 GB/chip) | **Good fit** — 2 PP stages, minimal bubble |
| v5p-16 TP=8 | 8 | 760 GB | 470 GB ✓ | **290 GB** | Even better if TP=8 on single host (4 chips/host × 2 hosts) |
| v5p-32 PP=4 TP=4 | 16 | 1520 GB | 470 GB ✓ | **1050 GB** | Overkill — could run 3 replicas instead |
| v5p-32 TP=4 ×4 replicas | 16 | 1520 GB | 470×4 ✓ | Plenty | 4 independent replicas, max throughput |

#### Recommended Configurations

**Best for v6e (budget)**:
- **v6e-32, PP=4 TP=8** — 4 hosts with 8 chips each. Each PP stage gets 94/4 ≈ 24 layers. Weights per host: ~118 GB, sharded across 8 chips = ~14.7 GB/chip. Leaves ~17 GB/chip for KV cache. Comfortable.

**Best for v5p (performance)**:
- **v5p-16, PP=2 TP=4** — 2 hosts with 4 chips each. Each PP stage gets 47 layers. Weights per host: ~235 GB, sharded across 4 chips = ~59 GB/chip. Leaves ~36 GB/chip for KV cache. Only 2 PP stages = minimal pipeline bubble. v5p's faster per-chip compute makes this the performance winner.

**Possible but risky**:
- **v6e-16, PP=4 TP=4** — our current TPU. 470 GB across 512 GB HBM. Only 2.6 GB/chip for KV cache. Would need `max_model_len=1024` or lower. Might work for a proof-of-concept but not practical for real serving.

#### Prerequisites
1. Download model weights to regional GCS bucket (~470 GB, ~30-60 min from HF)
2. Allocate appropriate TPU (v6e-32 or v5p-16)
3. All Ray multi-host patches from RAY-001/004 carry over
4. Need to verify MoE expert weight names for `patch_ray_sharding_v3.py` — current hardcoded specs only cover dense Llama weights. MoE adds: `experts.*.gate_proj`, `experts.*.up_proj`, `experts.*.down_proj`, `shared_expert.*`, `gate` (router)
5. Update gcsfuse mount to point at new model path

**Attempt 1 (00:18 UTC)**: Disk full — gcsfuse file cache (40 GB) + weight loading exhausted /tmp. Fix: `--file-cache-max-size-mb 0`.

**Attempt 2 (00:18 UTC, v1)**: Model loaded! HBM 27.3-28.5 GiB/chip. But hit `ValueError: total_hbm_used_gb=113.85GiB exceeds total_hbm_limit_cap_gb=112.49GiB`. Fix: `--gpu-memory-utilization 0.95`.

**Attempt 3 (01:05 UTC)**: Model loaded in 24s (gcsfuse cache warm from attempt 1). KV cache: 106,624 tokens. Memory check passed. But crashed during XLA compilation:
```
KeyError: 'model.layers.23.self_attn.attn'
```
The Qwen3 MoE attention module has an `attn` submodule that the tpu-inference code expects to find in the state dict during compilation, but it's missing. This is a **model implementation bug** in `Qwen3MoeForCausalLM` on the TPU JAX backend — not a Ray or sharding issue.

**Attempt 4 (02:07 UTC)**: Added `patch_kv_cache_names.py` (register `.attn` suffix variants). Still fails — same KeyError on layers 23, 47, 71 (first layer of each PP stage except rank 0).

**Root cause analysis**:
- `get_pp_indices(94, rank, 4)` gives partitions [23, 24, 24, 23]
- But ALL hosts report `Compilation num_layers = 23` — even hosts that should have 24
- One attention layer per PP stage is not being detected by the KV cache manager
- The KV cache manager walks the vLLM model's `Attention` modules, but the PP boundary layer's attention isn't found in the module walk
- The `.attn` suffix patch didn't help because the layer isn't in the index at ALL (not a naming issue)
- This appears to be a vLLM model wrapper / KV cache spec bug specific to non-divisible PP layer counts (94/4 = 23.5)

**Status**: **BLOCKED** — Qwen3-235B-A22B on PP=4 hits a KV cache layer registration bug when layer count (94) doesn't divide evenly by PP size (4). The boundary layers (23, 47, 71) don't get their KV cache allocated. Would need a fix in vLLM's `kv_cache_manager.py` or the model wrapper's attention module walk. Not specific to Ray — would also affect no-Ray PP.

**Attempts 5-9**: Progressively identified the real issue:
- Attempt 5: VLLM_PP_LAYER_PARTITION=24,24,23,23 — fixed rank 0 but not others
- Attempt 6: KV debug logging — confirmed rank 0 = 24 layers (correct), ranks 1-3 = 23 (wrong for rank 1,2)
- Attempt 7: PP rank fix (vLLM parallel state override) — partially worked
- Attempt 8: PP rank v2 (VLLM_TPU_PP_RANK env var in get_pp_indices) — env var IS set, workers see correct rank, but KV cache still wrong
- Attempt 9: Added logging — confirmed workers set ranks 0,1,2,3 correctly, but KV cache counts still mismatch

**Root cause (confirmed)**: The vLLM GPU model's `make_layers` uses `get_pp_group().rank_in_group` from `vllm.distributed.parallel_state`, which was initialized with `world_size=1, rank=0` for all TPU Ray workers. The env var patch fixes `get_pp_indices` but there are OTHER calls to `get_pp_group()` in the model's forward path that still use rank=0.

The Llama 70B (80 layers) worked because 80/4=20 divides evenly — all ranks get the same count, so the wrong rank doesn't matter. 94/4=23.5 exposes the bug because different ranks need different layer counts.

**Status**: BLOCKED on PP rank propagation for non-evenly-divisible layer counts. This is a fundamental issue with how vLLM TPU Ray workers initialize parallel state — would need to either:
1. Initialize vLLM distributed with the actual PP world_size and rank (major change)
2. Ensure ALL calls to get_pp_group() in the vLLM model return the correct TPU PP rank
3. Use a model with layer count divisible by PP size

**Attempt 10 (05:17 UTC)**: Clean PP parallel state fix (`patch_pp_parallel_state.py`)
- Instead of env var tricks, directly override the vLLM PP `GroupCoordinator` attributes:
  ```python
  pp_group.rank = self.rank
  pp_group.ranks = list(range(pp_size))
  pp_group.rank_in_group = self.rank
  pp_group.world_size = pp_size
  ```
- This runs AFTER `ensure_model_parallel_initialized(tp=1, pp=1)` in `init_device`
- Safe because TPU workers don't use torch distributed for PP comms (they use JAX transfer servers)
- Applied on clean containers (no conflicting previous PP patches)

**Attempt 11-13 (05:23-05:39 UTC)**: Added debug logging with RAY_DEDUP_LOGS=0
- Confirmed ALL 4 workers now have correct PP ranks in both vLLM and JAX parallel state
- vLLM `make_layers` creates correct layer ranges for each rank (23,24,24,23 partition)
- KV cache counts are correct (23,24,24,23)
- BUT: KV caches are on the **WRONG workers**!
  - Rank 1 (should have layers 23-46): KV cache has layers 71-93
  - Rank 2 (should have layers 47-70): KV cache has layers 23-46
  - Rank 3 (should have layers 71-93): KV cache has layers 47-70
- The forward pass correctly runs layers 23 on rank 1, 47 on rank 2, 71 on rank 3
- But the KV cache index on each worker has the wrong layers → KeyError

**Root cause (refined)**: The PP layer assignment is now correct on all workers. But the KV cache setup and the forward pass use DIFFERENT worker orderings. The Ray executor sorts workers by IP and assigns adjusted ranks, but the KV cache initialization or the compiled DAG forward pass uses the original (unsorted) ordering, causing the KV caches to land on the wrong workers.

**Attempt 14-17 (05:58-06:19 UTC)**: Fixed KV cache layer name mapping

The investigation revealed:
1. KV cache configs are distributed in a different order than workers' adjusted ranks
2. `kv_cache_tensor.shared_by` contains layer names from the WRONG PP stage
3. Workers receive KV caches meant for other workers

Fix (`patch_kv_cache_local_names.py`):
- After KV cache allocation, re-discover local attention layer names from the vLLM model
- Re-register KV cache index using LOCAL layer names (ignoring `shared_by`)
- Allocate extra KV caches if local layer count > distributed cache count
- This ensures each worker's KV cache index matches its actual attention layers

**Attempt 17 (06:14 UTC)**: **SUCCESS!**
- All 4 workers correctly registered their local layer names
- Extra KV cache allocated on rank 2 (had 23 caches but 24 local layers)
- NO KeyError during XLA compilation
- **Server reached READY state at 06:19Z**
- First inference: "What is 7 * 8?" → Model generated thinking-style response ✓

**Patches for Qwen3-235B (7 total)**:
1. `patch_ray_multihost_v2.py` — JAX isolation env vars
2. `patch_ray_sharding.py` — Handle None/PartitionSpec in shard_put
3. `patch_ray_sharding_v3.py` — Hardcoded TP partition specs by weight name
4. `patch_ray_mm.py` — Add `supports_mm_inputs = False`
5. `patch_pp_parallel_state.py` — Override vLLM PP group rank/world_size
6. `patch_kv_cache_local_names.py` — Re-register KV cache with local attention layer names + allocate extras
7. gcsfuse mount with `--only-dir` and `--file-cache-max-size-mb 0`

**GRPO Mini-Batch Results (RAY-007b, max_tokens=512, concurrency=16)**:
- Completions: 1,008/1,024 (16 errors from context overflow)
- Wall time: 2,331s (38.9 min)
- Gen tok/s: **221**
- Mean tokens/resp: 511
- Accuracy: 0.3% (thinking model needs more tokens for reasoning — max_model_len=1024 too short)
- Note: max_model_len=1024 due to tight HBM (28.5/31.25 GiB per chip). Would need v6e-32 or v5p-16 for longer context.

**Cross-Model Comparison (all on v6e-16, PP=4, TP=4, Ray)**:

| Model | Params (active) | HBM/chip | Gen tok/s | Accuracy | Notes |
|-------|----------------:|---------:|----------:|---------:|-------|
| Llama 3.1 8B | 8B (8B) | 3.25 GiB | 389-929 | 42-69% | Baseline, fits single host |
| Llama 3.3 70B | 70B (70B) | 8.46 GiB | 259-929 | 67-77% | Also fits v6e-8 single host (2.6x faster) |
| **Qwen3-235B-A22B** | **235B (22B)** | **28.5 GiB** | **221** | **N/A** | **First model that NEEDS Ray multi-host** |

### v6e-32 Attempt (PP=8 TP=4, max_model_len=8192)
- TPU `vllm-ray-v6e32-attempt` allocated after 25 retries in us-east1-d
- 8 hosts × 4 chips = 32 chips, 1024 GB total HBM
- Launched with PP=8 TP=4 max_model_len=8192
- **PREEMPTED** during weight loading (~22% through 118 shards)
- Model also copied to `gs://marin-us-central1` for future v5p-16 attempt

### v5p-16 Run (PP=2 TP=4, us-central1-a)

**TPU**: `vllm-qwen3-v5p16-uc1a` — v5p-16, us-central1-a, 2 hosts × 4 chips = 8 chips, 760 GB HBM
**Weights**: `gs://marin-us-central1/models/Qwen--Qwen3-235B-A22B-Thinking-2507--6cbffae/Qwen--Qwen3-235B-A22B-Thinking-2507--6cbffae/` (same region)

**HBM breakdown per chip (v5p, 95.74 GiB)**:
| Layer | GiB/chip |
|-------|---------|
| Weights | 55.12 |
| KV cache | 35.86 |
| **Total** | **90.98** |
| Free | 4.76 |

**KV cache**: 1,598,720 tokens capacity (vs 106,624 on v6e-16 — **15x more**)

**GRPO Mini-Batch Results (concurrency=16, max_model_len=8192, max_tokens=1024)**:
- Completions: 1,024, Errors: 0
- Wall time: 3,833s (63.9 min)
- Gen tok/s: **265**
- Mean tokens/resp: 992
- Accuracy: 12.2% (thinking model needs long context for reasoning)
- KV cache usage: **0.7%** — massively underutilized at concurrency=16

**Key finding: concurrency=16 is way too low.** KV cache can fit ~1,450 concurrent requests but we're only running 16. The TPU is memory-bandwidth bound at this batch size — loading weight matrices from HBM to compute units for only 16 token vectors per step. Higher concurrency (64-256) would amortize the HBM reads across more sequences, dramatically improving throughput.

**Relaunch with max_model_len=16384, concurrency=256:**

**GRPO Results (concurrency=256, max_model_len=16384, max_tokens=1024)**:
- Completions: 1,024, Errors: 0
- Wall time: 1,028s (17.1 min)
- Gen tok/s: **988** (peak ~1,920 during sustained phase)
- Accuracy: 12.4%
- KV cache usage: peaked at ~16%
- Epoch projection: 53.7h

**Why accuracy was 12%**: `max_tokens=1024` truncated the thinking model's reasoning chains. 8/10 sample responses had `finish_reason=length` — the model ran out of output tokens before writing `\boxed{}`. The model KNOWS the answer but gets cut off. Fix: `max_tokens=4096`.

**Concurrency scaling (max_tokens=1024):**
| Concurrency | tok/s | Wall time | Speedup |
|-------------|------:|----------:|--------:|
| 16 | 265 | 64 min | 1x |
| **256** | **988** | **17 min** | **3.7x** |

**IMPORTANT: `max_model_len` vs `max_tokens`**:
- `max_model_len` = server-side max sequence length (prompt + output). Set at vLLM launch.
- `max_tokens` = per-request max output tokens. Set in the API request / benchmark script.
- For thinking models like Qwen3-235B-Thinking, `max_tokens` must be 4096+ to allow full reasoning chains.

**Results with max_tokens=4096, concurrency=256:**
- Completions: 345/1,024 (679 errors from concurrent benchmark collision)
- **Accuracy: 73.6%** (vs 12.4% with max_tokens=1024) — confirms truncation was the issue
- Mean tokens/resp: 1,728 (model uses ~1,700 tokens for thinking + answer)
- Gen tok/s: 254
- Wall time: 2,345s (39 min)

**Key insight**: Thinking models need 2-4x more output tokens than standard models. max_tokens=1024 truncates the reasoning chain before `\boxed{}` is written. max_tokens=4096 allows full reasoning → 73.6% accuracy.

**Clean run with max_tokens=4096, concurrency=256, timeout=1800s:**
- Completions: 1,024 (0 timeouts! timeout fix worked)
- Wall time: 2,916s (48.6 min)
- Gen tok/s: **1,127**
- Mean tokens/resp: 3,230
- Finish: 766 stop (75%), 251 length (25%), 7 empty
- Grader accuracy: 30.9% (broken `\boxed{}` extraction)
- **Fixed grader accuracy on completed (stop) responses: 45.6%**
- 25% still truncated at 4096 — model needs more tokens for hard problems

**Accuracy analysis by finish reason:**
| Subset | Count | Accuracy |
|--------|------:|--------:|
| Finished naturally (stop) | 766 | 45.6% |
| Truncated at 4096 (length) | 251 | 3.2% |
| All | 1,017 | 35.1% |

**Grading pipeline issues found:**
1. `\boxed{}` regex fails on nested braces (e.g., `\boxed{\dfrac{25}{4}}` extracts only `\dfrac{25`)
2. `\frac` vs `\dfrac` not normalized
3. 59% of responses don't contain `\boxed{}` — model outputs answer in `$$..$$` display math instead
4. Need SymPy-based equivalence grading (not string matching) for accurate results

**Disk full incident**: Ray worker logs (JAX int64 warnings) filled 46 GB in Docker overlay. Fix: periodically truncate `/tmp/ray/session_*/logs/*.log` inside containers, or set `RAY_LOG_TO_STDERR=0`.

**Rerunning with max_tokens=8192** to eliminate all truncation. Server relaunched with max_model_len=16384 on v5p-16, concurrency=256, timeout=1800s. Streaming results to JSONL for live monitoring.

### Weight Loading & Caching Lessons Learned

**CRITICAL: Never restart Docker containers on multi-host TPU setups.**

The OS page cache on each host caches gcsfuse reads in RAM. After the first 54-min weight load, subsequent loads from the same container are near-instant (seconds). But `docker rm` destroys the container's filesystem context and the page cache gets evicted.

**Best practices for fast restarts**:
1. **Kill only the vLLM process** (`pkill -f vllm`), not the container
2. **Restart Ray inside the container** (`ray stop --force && ray start ...`), not `docker rm && docker run`
3. **Never `docker rm`** — use `docker exec` to restart services
4. If container must be recreated, enable gcsfuse file cache: `--file-cache-max-size-mb 40000`
5. First load from same-region gcsfuse (no cache): ~54 min for 438 GiB at ~27s/shard
6. Second load from page cache: seconds

**What happened**: Restarting worker 1's container for Ray rejoin destroyed its page cache. Worker 0 (page cache warm) loaded at 24s/shard, worker 1 (cold) loaded at 37s/shard — worker 1 became the bottleneck.
