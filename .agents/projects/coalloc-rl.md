# Coalloc-RL: On-Demand Multi-Host RL Training Without Ray

## Synopsis

On-demand RL is an infrastructure mode that runs RLOO reinforcement learning
(trainer + sampler) on **two independent TPU VMs** coordinated entirely through
GCS — no Ray cluster required. This eliminates cluster bootstrap overhead,
reduces cost (pay for exactly 2 TPUs), and simplifies multi-region deployment.

The work was developed on the `on-demand-rl` branch and validated end-to-end
with `exp2039` (Llama 3.1-8B RLOO on MATH-500).

---

## Problem

Ray clusters add management complexity, startup latency, and cost overhead that
is unnecessary for RL workloads where only two roles exist (trainer and sampler).
Creating, maintaining, and debugging a Ray cluster for two machines is overkill.

**Goals:**
- Launch trainer and sampler as independent TPU VMs (no shared cluster)
- Coordinate via GCS (the only surface both VMs can access)
- Preserve full compatibility with the existing Ray-based path
- Support preemption recovery, graceful shutdown, and fast restart

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Shared GCS Bucket                       │
│   gs://<bucket>/tmp/exp2039/<run_id>/                        │
│  ├── checkpoints/             (trainer saves)                │
│  ├── rollouts/                (sampler writes, trainer reads) │
│  └── weight_transfer/                                        │
│      └── arrow_flight_coordinator.json  (discovery + status) │
└──────────────────────────────────────────────────────────────┘
         ▲                                         ▲
         │                                         │
    ┌────┴───────────────┐          ┌──────────────┴────┐
    │   TRAINER TPU VM   │          │   SAMPLER TPU VM  │
    │                    │          │                    │
    │ 1. Load model      │          │ 1. Load model     │
    │ 2. Publish weights ├─────────▶│ 2. Init vLLM      │
    │    (Arrow Flight)  │          │ 3. Fetch weights   │
    │ 3. Wait for        │          │ 4. Generate text   │
    │    rollouts        │◀─────────┤ 5. Compute rewards │
    │ 4. Train step      │          │ 6. Write rollouts  │
    │ 5. Repeat 2-4      │          │ 7. Repeat 3-6      │
    │ 6. Signal done     │          │ 8. Exit on signal  │
    └────────────────────┘          └───────────────────┘
       Arrow Flight Server            Arrow Flight Client
       (gRPC, port 34829)
```

---

## Key Coordination Mechanisms

### 1. Filesystem Arrow Flight Coordinator

Replaces the Ray actor-based coordinator with a GCS-persisted JSON file:

```json
{
  "weight_id": 42,
  "server_addresses": ["grpc://10.128.0.5:34829"],
  "param_names": ["transformer.layers.0.attn.q_proj.weight", ...],
  "status": "running"
}
```

- **Trainer** publishes this JSON after each weight serve
- **Sampler** polls it every ~1s, discovers Arrow Flight servers, connects via gRPC
- **Stale weight rejection**: `weight_id` monotonically increases; `weight_id=-1`
  is the sentinel for initial/restart weights, bypassing stale checks

**File:** `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py` (~825 lines)

### 2. Rollout Storage (File-Based)

Serialized rollout data (prompts, generations, rewards) as pickle files on GCS.
Multiple sampler workers can write in parallel without locking. The trainer reads
and filters by `max_rollout_step_delay`.

**File:** `lib/marin/src/marin/rl/rollout_storage.py` (~568 lines)

### 3. Weight Transfer via Arrow Flight

Apache Arrow Flight (gRPC-based) transfers model weights between VMs:
- ~7 GB/s on TPUv5
- State dict → numpy → Arrow RecordBatches → gRPC → JAX arrays
- `convert_to_bfloat16=True` halves transfer size

**Fallback:** GCS checkpoint mode if Arrow Flight is unavailable.

### 4. Fast vLLM Bootstrap

vLLM normally downloads the full model from HuggingFace (30s-2min). With
`load_format=dummy`, we stage just metadata locally, then inject weights from
the trainer via Arrow Flight.

- Stage metadata (`config.json`, `tokenizer.json`) from checkpoint: ~1s
- Initialize vLLM with `load_format="dummy"`: ~5s
- Inject weights via `sync_weights`: ~10s
- **Total: ~16s vs 60s+ standard bootstrap**

**File:** `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py`

---

## Experiment Entry Point

**File:** `experiments/exp2039_rl_math500.py` (524 lines)

Supports four modes via `--mode`:

| Mode | Purpose |
|------|---------|
| `executor` | Use existing Ray cluster (backward compatible, default) |
| `trainer` | Run only the trainer worker on this TPU VM |
| `sampler` | Run only the sampler worker on this TPU VM |
| `launch-plan` | Print exact launch commands for review |

### Deployment Presets

| Preset | Zone | Trainer | Sampler |
|--------|------|---------|---------|
| `v5p_east5a` | us-east5-a | v5p-8 | v5p-8 |
| `v5p_central1a` | us-central1-a | v5p-8 | v5p-8 |
| `v6e_euw4a` | europe-west4-a | v6e-16 | v6e-8 |
| `v6e_east1d` | us-east1-d | v6e-16 | v6e-8 |

---

## What Works (Validated End-to-End)

1. **Manual deployment flow** — trainer and sampler launch independently, find
   each other via GCS, train successfully
2. **Filesystem Arrow Flight coordinator** — discovery, connection, stale weight
   rejection, preemption recovery (`weight_id=-1` bypass)
3. **Graceful sampler shutdown** — trainer signals `status=completed`, sampler
   exits cleanly, W&B runs close properly
4. **Fast vLLM bootstrap** — 16s total startup vs 60s+ standard
5. **Multi-zone deployments** — four presets across EU and US zones, same-region
   GCS enforcement
6. **Preemption recovery** — trainer restarts with sentinel weight ID, sampler
   reconnects automatically

---

## Known Issues / WIP

### Inflight Weight Updates (Background Fetch)
- Currently disabled due to JAX + `os.fork()` deadlock
- `AsyncvLLMInferenceContext` forks after JAX init → deadlock on device locks
- Sync mode works but adds ~10s/cycle overhead
- **Options:** (a) delay JAX init until after vLLM fork, (b) background fetch
  thread without async context
- **Logbook:** `.agents/logbooks/debug_inflight_weight_updates.md`

### Sampler Timeout on Dead Servers
- If trainer gets SIGKILL before `mark_failed()`, GCS retains stale server
  addresses and sampler polls dead servers indefinitely
- Needs timeout-based fallback (not yet implemented)

### Stale Rollout Cleanup
- Old rollout files persist on GCS from previous runs
- Filtered out by `max_rollout_step_delay` but waste storage
- Low priority

---

## File Map

| Component | File | Purpose |
|-----------|------|---------|
| Experiment | `experiments/exp2039_rl_math500.py` | Entry point, modes, presets |
| Trainer | `lib/marin/src/marin/rl/train_worker.py` | RLOO training, weight serving |
| Sampler | `lib/marin/src/marin/rl/rollout_worker.py` | vLLM inference, rollout writing |
| Coordinator | `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py` | GCS coordinator, Arrow Flight |
| Interfaces | `lib/marin/src/marin/rl/weight_transfer/base.py` | Abstract weight transfer API |
| vLLM Boot | `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py` | Fast bootstrap |
| RL Job | `lib/marin/src/marin/rl/rl_job.py` | Ray-mode job config |
| Launcher | `on-demand-rl-scripts/_run_exp2039.sh` | Main launch script |
| Presets | `on-demand-rl-scripts/exp2039_*.sh` | Zone-specific wrappers |

---

## Logbooks & Design Docs

- **Design overview:** `ON_DEMAND_RL.md` (top-level)
- **Detailed research log:** `.agents/logbooks/on-demand-rl.md` (1473 lines)
- **Inflight weight debug:** `.agents/logbooks/debug_inflight_weight_updates.md`
- **Graceful exit design:** `.agents/projects/graceful_sampler_exit.md`
- **Preemption design:** `.agents/projects/rl_preemption.md`
- **Inflight weight design:** `.agents/projects/inflight_weight_updates.md`

---

## Multi-Host "Data Parallel" vLLM (Independent Replicas)

A key discovery from the `no_ray_multihost_vllm` research: on multi-host TPUs,
**running independent model replicas per host massively outperforms scaling
tensor parallelism across hosts**. This is the approach we use for the sampler
in on-demand RL.

### Why Not Cross-Host TP or PP?

- **Cross-host PP without Ray failed**: vLLM's TPU `MultiprocExecutor` spawns
  PP workers per host regardless of `--nnodes`. Each worker hits
  `INTERNAL: Failed to get global TPU topology`. Cross-host PP requires Ray.
- **TP across hosts is worse**: TP=8 on a v6e-8 (8 chips, 1 host) is only 3%
  faster than TP=4 on a v6e-4 (4 chips). Communication overhead dominates for
  models that fit in a single host's HBM.

### The Pattern: One vLLM Server Per Host, Round-Robin Routing

Each host in a multi-host TPU runs its own independent vLLM server with TP
equal to the number of local chips. A load balancer (or the sampler itself)
distributes prompts round-robin across replicas. Each host is an isolated JAX
cluster — no cross-host coordination needed.

```
Multi-host TPU (e.g. v6e-16 = 4 hosts × 4 chips)

  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │  Host 0      │  │  Host 1      │  │  Host 2      │  │  Host 3      │
  │  vLLM TP=4   │  │  vLLM TP=4   │  │  vLLM TP=4   │  │  vLLM TP=4   │
  │  Llama 8B    │  │  Llama 8B    │  │  Llama 8B    │  │  Llama 8B    │
  │  :8000       │  │  :8000       │  │  :8000       │  │  :8000       │
  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
         │                 │                 │                 │
         └────────────┬────┴────────┬────────┘─────────────────┘
                      │  Round-Robin Load Balancer / Sampler   │
                      └────────────────────────────────────────┘
```

### How to Launch Each Replica

Each host runs Docker with environment variables that isolate it as a
single-host JAX cluster:

```bash
docker run -d --privileged --net=host --shm-size=32gb \
  -v /tmp:/tmp \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -e TPU_BACKEND_TYPE=jax \
  -e PJRT_DEVICE=TPU \
  -e CLOUD_TPU_TASK_ID=0 \
  -e TPU_PROCESS_BOUNDS=1,1,1 \
  -e TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  -e TPU_VISIBLE_CHIPS=0,1,2,3 \
  -e HF_TOKEN=$HF_TOKEN \
  vllm/vllm-tpu:nightly \
  vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 4 --max-model-len 4096 --port 8000
```

**Critical env vars** that make each host an independent JAX cluster:
- `CLOUD_TPU_TASK_ID=0` — each host thinks it's the only host
- `TPU_PROCESS_BOUNDS=1,1,1` — single-process topology (no cross-host comm)
- `TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1` — all local chips belong to this process
- `TPU_VISIBLE_CHIPS=0,1,2,3` — expose all 4 local chips
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` — in-process engine (no subprocess fork)

### Benchmarked Results

#### Cross-Hardware Comparison (Llama 8B, optimal concurrency=256/replica)

| Config | Chips | Replicas | tok/s | Batch time (1024 completions) | Epoch projection |
|--------|------:|---------:|------:|------------------------------:|-----------------:|
| v6e-4 (1 host, TP=4) | 4 | 1 | ~4,900 | ~85s | ~4.4h |
| v6e-8 (1 host, TP=8) | 8 | 1 | ~4,000 | ~104s | ~5.5h |
| **v6e-16 (4 hosts, 4×TP=4)** | **16** | **4** | **19,708** | **21.1s** | **1.1h** |
| v5p-8 (1 host, TP=4) | 4 | 1 | 7,940 | 52.9s | 2.8h |
| **v5p-16 (2 hosts, 2×TP=4)** | **8** | **2** | **14,303** | **29.3s** | **1.5h** |

#### Key Findings

1. **v6e-8 (TP=8) is slower than v6e-4 (TP=4)** for 8B models — TP
   communication overhead dominates when the model already fits in 4 chips
2. **4 replicas on v6e-16 give 4x the throughput of 1 replica** — near-linear
   scaling because replicas are fully independent
3. **v5p is ~1.45x faster per chip than v6e** at optimal concurrency, but v6e
   wins on total throughput when you have more replicas
4. **Concurrency=256/replica saturates all tested hardware** — going higher
   gives zero additional throughput (confirmed on v6e-16, v5p-16, v5p-8)
5. **Scaling rule**: for throughput-bound workloads, run as many independent
   replicas as possible on the smallest host that fits your model

#### Concurrency Sweep (v6e-16, 4 replicas)

| Concurrency/replica | tok/s | Batch time | Epoch |
|--------------------:|------:|-----------:|------:|
| 32 | 7,239 | 60.1s | 3.1h |
| 64 | 11,870 | 35.5s | 1.85h |
| 128 | 15,651 | 26.6s | 1.4h |
| **256 (optimal)** | **19,708** | **21.1s** | **1.1h** |
| 512+ | ~19,700 | ~21s | (saturated) |

### Implications for RL

This is how the **sampler** in on-demand RL should work on multi-host TPUs:
each host runs an independent vLLM replica, and the sampler distributes prompts
across them. Combined with the fast vLLM bootstrap (16s startup via
`load_format=dummy` + Arrow Flight weight injection), each replica can be
online and generating within seconds of weight publication.

For a v6e-16 sampler with 4 replicas at concurrency=256: **~19,700 tok/s,
~21s per 1024-completion mini-batch, ~1.1h per epoch** of pure inference.

---

## Prior Art: Why Tunix Disaggregated RL Failed on Multi-Host TPU

Full details: [TUNIX_RL_MASTER.md](../../TUNIX_RL_MASTER.md)

Before building on-demand RL in Marin, we attempted multi-host async RL using
**Tunix** — a JAX/Flax NNX post-training library with pluggable rollout engines
(vanilla, vLLM, SGLang) and explicit role-to-mesh assignment.

### What Tunix Does

Tunix decomposes RL into roles (actor, reference, rollout, critic, reward) each
assigned a JAX mesh. In **collocated mode**, all roles share the same mesh and
execution is sequential over shared hardware — this works reliably on
multi-host TPU. In **disaggregated mode**, roles get different meshes so
training and rollout can overlap on dedicated chips.

The RL loop: generate rollouts → score → compute advantages → compute
logprobs → train actor → sync weights back to rollout engine → repeat.
Weight sync from actor to vLLM rollout goes through:
`VllmRollout → VllmSampler.update_params → transfer_state_with_mappings →
reshard_pytree → jax.device_put`.

### Three Attempts, All Failed

**1. Host-separated split** (host 0 = rollout mesh, host 1 = actor mesh):
Crashed during initial vLLM weight sync. `jax.device_put` can't transfer
between disjoint meshes on different hosts in normal JAX SPMD — both hosts
must participate in every compiled program.

**2. Interleaved-mesh split** (devices `[0,1,4,5]` vs `[2,3,6,7]` spread
across hosts): Still breaks — non-contiguous device sets aren't valid physical
TPU rectangles. XLA can't plan collectives over them.

**3. Transfer-mesh proposal** (a third full mesh just for weight sync): Clever
but doesn't fix the fundamental issue — JAX SPMD isn't a Pathways-style
central dispatcher that can independently schedule disjoint role meshes.

### Root Cause

The problem is **architectural, not a bug**:

- JAX multi-controller SPMD requires all hosts to enter the same compiled
  program together
- You can't have one host "doing rollout" while another "does training"
  independently within a single JAX process group
- The OSS reshard fallback (`jax.device_put`) crashes on cross-host disjoint
  mesh transfers
- Pathways references in Tunix code are aspirational — the actual runtime is
  ordinary JAX
- LoRA reduces sync payload but doesn't solve the runtime model mismatch
- CPU offload helps memory but doesn't fix cross-mesh execution

### What This Means for Coalloc-RL

The Tunix doc's own conclusion recommends the **multi-process architecture** as
the only credible fix: separate actor process, separate rollout process,
external coordinator, explicit weight sync transport. That is exactly what
Marin's on-demand RL implements — separate TPU VMs coordinated via GCS with
Arrow Flight weight transfer.

---

## How to Run

```bash
# Launch with a preset (creates both TPU VMs):
bash on-demand-rl-scripts/exp2039_euwest4.sh

# With overrides:
CAPACITY_MODE=on-demand bash on-demand-rl-scripts/exp2039_useast5a.sh

# Preview commands without launching:
uv run python experiments/exp2039_rl_math500.py \
  --mode launch-plan \
  --deployment-preset v6e_euw4a \
  --run-id my-run \
  --shared-root gs://my-bucket/tmp/exp2039/my-run
```
