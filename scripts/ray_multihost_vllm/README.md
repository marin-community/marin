# Ray Multi-Host vLLM on TPU

Serve large models (70B-235B+) across multiple TPU hosts using vLLM with Ray pipeline parallelism.

## Architecture

```
Host 0: 4 TPU chips, TP=4 → PP stage 0 (layers 0-N/PP)
Host 1: 4 TPU chips, TP=4 → PP stage 1 (layers N/PP-2N/PP)
...
Host K: 4 TPU chips, TP=4 → PP stage K (layers ...)
```

- **Pipeline Parallelism (PP)** across hosts — each host runs a subset of layers
- **Tensor Parallelism (TP)** within each host — weight matrices sharded across chips
- **Ray** manages worker placement and communication
- PP communication via JAX transfer servers (not Ray data plane)

## Quick Start

```bash
# 1. Allocate TPU
gcloud alpha compute tpus queued-resources create my-tpu \
  --accelerator-type=v5p-16 --zone=us-central1-a \
  --runtime-version=v2-alpha-tpuv5 --node-id=my-tpu --best-effort

# 2. Launch (Qwen3-235B on v5p-16, PP=2 TP=4)
bash scripts/ray_multihost_vllm/launch.sh my-tpu us-central1-a \
  gs://marin-us-central1/models/Qwen--Qwen3-235B-A22B-Thinking-2507--6cbffae/Qwen--Qwen3-235B-A22B-Thinking-2507--6cbffae \
  2 4 16384

# 3. Wait for health=200 (30-60 min first load, seconds on restart)
# 4. Send requests to http://<HEAD_IP>:8000/v1/chat/completions
```

## Required Patches (6 total)

The `vllm/vllm-tpu:nightly` image has bugs that prevent Ray multi-host from working. These patches are applied at runtime inside Docker containers:

| Patch | What it fixes |
|-------|---------------|
| `patch_ray_multihost_v2.py` | Sets `TPU_PROCESS_BOUNDS`, `TPU_CHIPS_PER_PROCESS_BOUNDS`, `TPU_VISIBLE_CHIPS` to isolate each Ray worker to single-host JAX mode. Without this, XLA sees all global devices and compilation fails with `Unexpected device_id`. |
| `patch_ray_sharding.py` | Handles `None` and `PartitionSpec` in `shard_put` by wrapping in `NamedSharding(mesh, spec)`. Without this, the Ray multi-host path in `general_device_put` crashes. |
| `patch_ray_sharding_v3.py` | Hardcodes TP partition specs by weight name when `nnx.get_named_sharding` fails (which it always does under Ray). Maps `gate_proj`→`P(None, 'model')`, `q_proj`→`P(None, 'model', None)`, etc. Without this, weights are replicated instead of sharded → OOM for 70B+. |
| `patch_ray_mm.py` | Adds `self.supports_mm_inputs = False` to `TPUModelRunner`. The Ray executor checks this attribute but TPU runner doesn't define it. |
| `patch_pp_parallel_state.py` | Overrides vLLM's `GroupCoordinator` PP rank/world_size to match the actual TPU worker rank. Without this, all workers think they're PP rank 0 and create the same layer assignment. |
| `patch_kv_cache_local_names.py` | Re-registers KV cache indices using local attention layer names instead of global `shared_by` names from the distributed config. Also allocates extra KV caches when local layer count exceeds distributed count. Fixes PP boundary layer KeyError. |

## Hardware Sizing

### Model fits on single host → DON'T use Ray PP
| Model | Weights | v6e-8 (256 GB) | Recommendation |
|-------|--------:|:---------------:|----------------|
| Llama 8B | 15 GB | ✅ easy | Single host TP=4 or TP=8 |
| Llama 70B | 131 GB | ✅ fits | Single host TP=8 (2.6x faster than PP=4) |

### Model needs multi-host → Use Ray PP
| Model | Weights | Config | HBM/chip | Notes |
|-------|--------:|--------|----------|-------|
| Qwen3-235B MoE | 438 GB | v6e-16 PP=4 TP=4 | 28.5/31.25 GiB | Very tight, max_model_len=1024 |
| Qwen3-235B MoE | 438 GB | **v5p-16 PP=2 TP=4** | **55/95 GiB** | **Recommended.** Plenty of KV cache. |
| Qwen3-235B MoE | 438 GB | v6e-32 PP=8 TP=4 | 14.7/32 GiB | Comfortable but needs 8 hosts |

## Critical Rules

### Same-Region Weight Loading
**NEVER load weights from a different GCP region than the TPU.** Cross-region GCS: ~50 MiB/s. Same-region: 1-10 Gbps.

```
TPU in us-central1-a → weights in gs://marin-us-central1/
TPU in us-east1-d    → weights in gs://marin-us-east1/
```

### Never Restart Docker Containers
OS page cache caches gcsfuse reads. First load: ~55 min. Second load from same container: seconds. `docker rm` destroys the cache. Always use `docker exec` to restart services inside the container.

### Ray Log Cleanup
JAX int64 warnings spam Ray logs at ~1 GB/hour. Periodically truncate:
```bash
docker exec ray-node find /tmp/ray -name '*.log' -size +100M -delete
```

## Benchmark Scripts

| Script | Purpose |
|--------|---------|
| `benchmark_math500.py` | MATH-500 eval (500 problems, greedy) |
| `benchmark_grpo_stress.py` | GRPO mini-batch stress test (64 prompts × 16 gen = 1024/batch) |
| `prepare_hendrycks_math.py` | Generate deterministic JSONL dataset (seed=42) |
| `view_responses.html` | Browser-based response viewer |

### GRPO Benchmark
```bash
python benchmark_grpo_stress.py \
  --servers http://localhost:8000 \
  --dataset-path hendrycks_math_train.jsonl \
  --num-batches 1 \
  --max-tokens 8192 \
  --concurrency-per-replica 256 \
  --stream-output /tmp/stream.jsonl \
  --output /tmp/results.json
```

Tail live results:
```bash
tail -f /tmp/stream.jsonl.batch1 | python3 -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    print(f'{r[\"prompt_idx\"]:>3} | {\"✓\" if r[\"correct\"] else \"✗\"} | {r[\"tokens\"]:>5} tok | {r[\"ground_truth\"]:>10} | {r[\"model_answer\"][:30]}')
"
```

## Performance Results

### Qwen3-235B-A22B on v5p-16 (PP=2, TP=4)

| Concurrency | max_tokens | tok/s | Wall time (1024 req) | Accuracy |
|-------------|-----------|------:|--------------------:|---------:|
| 16 | 1024 | 265 | 64 min | 12% (truncated) |
| 256 | 1024 | 988 | 17 min | 12% (truncated) |
| 256 | 4096 | 1,127 | 49 min | 45.6% (stop) |

### Llama 3.3 70B on v6e-16 (PP=4, TP=4) vs v6e-8 (TP=8)

| Config | tok/s | Epoch projection |
|--------|------:|-----------------:|
| v6e-16 Ray PP=4 | 929 | 26.8h |
| **v6e-8 single TP=8** | **2,427** | **10.2h** |

**Key insight**: If the model fits on one host, single-host TP is always faster than multi-host PP.
