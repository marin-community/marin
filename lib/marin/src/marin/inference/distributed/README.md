# `marin.inference.distributed`

Multi-region distributed inference for vLLM-on-TPU. One stateless top-level
entry point that hides the per-region, per-worker plumbing behind a single
function call.

```python
from marin.inference.distributed import (
    InferenceConfig, ModelSpec, SamplingParams, inference,
)

result = inference(
    model=ModelSpec(model="Qwen/Qwen3-0.6B", engine_kwargs={"tensor_parallel_size": 4}),
    dataset=[{"id": "1", "payload": {"kind": "text", "prompt": "Hello"}}],
    config=InferenceConfig(regions=["us-central1"], results_region="us-central1"),
)
for record in result.iter_records():
    print(record.id, record.response)
```

For the design rationale (what's actually happening under the hood, and the
key decisions that shaped it), see [DESIGN.md](DESIGN.md).

## What this gives you

- One regional Zephyr job per entry in `config.regions`; each region's
  workers run vLLM on the configured TPU shape.
- All shard outputs land in a **single canonical `results_region`** so
  downstream callers reading from one bucket pay no cross-region egress.
- The JAX/vLLM XLA compile cache lives at
  `gs://marin-{region}/tmp/ttl=30d/vllm-cache/{model_hash}/`, shared across
  every worker in a region and across runs.
- Model output is preserved **verbatim** — `<think>` / `<reasoning>`
  markers are not stripped. Downstream callers post-process as they see fit.

## Quickstart (single region, public HF model)

```python
from marin.inference.distributed import (
    InferenceConfig, ModelSpec, SamplingParams, inference,
)

prompts = [
    {"id": f"p{i}", "payload": {"kind": "text", "prompt": f"Question {i}:"}}
    for i in range(100)
]

cfg = InferenceConfig(
    regions=["us-central1"],
    results_region="us-central1",
    tpu_shapes=("v5p-8",),
    max_workers_per_region=4,
    shard_size=500,
    sampling=SamplingParams(temperature=0.0, max_tokens=512),
    job_name="my-inference-run",
)

result = inference(
    model=ModelSpec(
        model="Qwen/Qwen3-0.6B",
        engine_kwargs={"tensor_parallel_size": 4},
    ),
    dataset=prompts,
    config=cfg,
)

print(f"Outputs at: {result.results_uri}")
print(f"Missing shards: {result.missing_shards}")
for record in result.iter_records():
    print(record.id, record.response)
```

The call blocks until every regional job has terminated. Inspect
`result.missing_shards` to confirm completeness — see "Failure model" below.

## Input shapes

`dataset` accepts either:

- **An in-memory `Sequence[dict]` of `PromptRecord` dicts.** The library
  materializes these to gzipped JSONL files in the results bucket so all
  regional workers can read them. The chunk size is `config.shard_size`
  records per file; one file becomes one content shard downstream.
- **A path or glob string** (e.g. `"gs://marin-us-central1/.../prompts-*.jsonl.gz"`)
  pointing at pre-written JSONL files. Each file is one content shard;
  `shard_size` is not used in this case.

Each input record has two fields:

```python
{"id": "any-string", "payload": <payload>}
```

Two payload kinds:

- `{"kind": "text", "prompt": "..."}` — raw completion via `engine.generate()`.
- `{"kind": "messages", "messages": [{"role": ..., "content": ...}, ...]}` —
  chat completion via `engine.chat()`.

Mixing kinds within a single shard is rejected at runtime; spread the kinds
across separate runs (or input files) if you need both.

## Model paths

`ModelSpec.model` supports three shapes:

- **HF id** (e.g. `"meta-llama/Llama-3-8B"`) — vLLM downloads at startup.
- **`marin://checkpoints/...`** — resolves per worker region to
  `gs://marin-{worker_region}/checkpoints/...`. Use this for Marin-trained
  checkpoints that have been replicated across regions.
- **Explicit `gs://...`** — hard-pinned. Workers scheduled in a different
  region crash on the pre-flight `check_gcs_paths_same_region` check
  before vLLM loads (avoids a ~16 GB cross-region weight download).

## Multi-region with cross-region aggregation

```python
cfg = InferenceConfig(
    regions=["us-central1", "europe-west4"],
    results_region="us-central1",          # canonical sink
    tpu_shapes=("v6e-4",),
    max_workers_per_region=8,
    shard_size=500,
    sampling=SamplingParams(temperature=0.7, max_tokens=2048, top_p=0.9),
)
prompts = [
    {"id": f"chat-{i}", "payload": {
        "kind": "messages",
        "messages": [{"role": "user", "content": "..."}],
    }}
    for i in range(N)
]
result = inference(model=..., dataset=prompts, config=cfg)
```

The two regions process disjoint shards via deterministic per-region
rotation; if both regions reach the same shard concurrently, the
`skip_existing` check on the shared output prefix arbitrates the race.

## Multi-shape TPU fallback

`tpu_shapes` is a tuple — you can pass multiple variants so the scheduler
can land on whichever has capacity. The variants must share **both**
`vm_count` **and** `chips_per_vm` (enforced by `fray.ResourceConfig.with_tpu`).

```python
tpu_shapes=("v6e-4", "v5litepod-4")  # both 1 VM × 4 chips → valid alternatives
tpu_shapes=("v6e-4", "v6e-8")        # chips_per_vm differs → rejected
```

## Tuning knobs

For long-running runs that exceed the upstream Zephyr defaults, raise the
per-context fields on `InferenceConfig`:

```python
cfg = InferenceConfig(
    ...,
    heartbeat_timeout=1800,          # default 120 — raise for cold XLA compile
    max_shard_infra_failures=200,    # default 20 — raise for preemption-heavy runs
)
```

vLLM throughput tuning belongs in `ModelSpec.engine_kwargs` (passed through
unchanged to `vllm.LLM(...)`):

```python
ModelSpec(
    model="...",
    engine_kwargs={
        "tensor_parallel_size": 4,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.92,
        "enable_prefix_caching": True,
        "max_model_len": 32768,
    },
)
```

The compile-cache key includes `engine_kwargs`, so changing TP size or any
other compile-affecting kwarg automatically gets its own cache prefix.

To disable the compile cache entirely (e.g. when debugging):

```python
cfg = InferenceConfig(..., compile_cache_uri_template="")
```

## Failure model

Per-region failures are **non-fatal at the run level**. The meta-coordinator
waits for every regional job to terminate, then checks the canonical output
prefix and returns an `InferenceResult` with `missing_shards` populated
for any shard no region completed. Surviving regions cover for failed ones
via the per-region rotation + `skip_existing` semantics.

```python
result = inference(...)
if not result.is_complete:
    print(f"Missing shards: {result.missing_shards}")
    # decide whether to retry, fail loudly, accept the partial result, etc.
```

The pre-flight `check_gcs_paths_same_region` catches misconfigured GCS
paths early — e.g. pointing a worker in `us-east5` at a
`gs://marin-us-central1/...` model URI. That regional job crashes before
vLLM loads.

## What's intentionally out of v1

- **Multi-host TPU** (e.g. v5p-16, v5p-32). Single-host shapes only.
- **`SamplingParams.n > 1`**. Rejected up front so we don't silently drop
  completions. Multi-completion fans out via either nested-list or fan-out;
  landing as a follow-up.
- **Non-vLLM backends.** Levanter / remote / litellm are not wired.
- **`token_ids` / `prompt_token_ids` in output extras** (large; re-tokenize
  from `text` if needed). A `capture_token_ids` config can be added if a
  real workflow demands it.
