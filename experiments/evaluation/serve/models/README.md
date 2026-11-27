# Model serve catalog

One `<org>/<model>.yaml` per model. Each file deserializes (via draccus) into a
`marin.evaluation.model_config.ModelConfig`; unknown fields or mistyped values fail at load, not at
serve time. `experiments.evaluation.models.models()` scans this directory once and merges the entries
with the Python factory entries defined in `models.py`. Files and directories whose names start with
`_` or `.` are skipped.

The launcher also accepts one file with this schema directly, without adding it to the catalog:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model-config /path/to/model.yaml \
  --evals smoke \
  --dry-run
```

Exactly one of `--model <registry-key>` or `--model-config <path>` is required. `record.json` stores
the normalized schema under `model.config`.

## Schema

```yaml
name: qwen3-32b                 # slash-free launch identity; catalog entries must be unique
location: Qwen/Qwen3-32B        # HF repo id, or gs://|s3:// HF-format export dir
revision: null                  # pin an immutable checkpoint (base models); optional
tokenizer: null                 # required only when location is an object-store path
apply_chat_template: true       # select curated chat or completion corpus variants

resource_hint:                  # ResourceHint -> experiment fleet placement
  hbm_gb: 84                    # serving HBM budget the hardware selector sizes a slice from
  gpu: {}                       # alternatively, accepted exact GPU shapes, e.g. {H100: 8}
  cpu: null                     # optional inference-worker host CPU override
  memory: null                  # override; the default is sized from the checkpoint's weight files
  disk: null                    # optional inference-worker host disk override

serve:                          # ServeConfig -> model-server behavior
  tensor_parallel_size: 2
  data_parallel_size: null
  pipeline_parallel_size: 1
  pipeline_layer_partition: []  # VLLM_PP_LAYER_PARTITION; layers per pipeline stage, e.g. [35, 37]
  gpu_memory_utilization: null
  max_model_len: 32768
  max_num_batched_tokens: null  # vLLM prefill-token budget
  max_num_seqs: null            # concurrent sequence limit
  hf_overrides: null            # JSON string, e.g. rope-scaling overrides
  limit_mm_per_prompt: null     # JSON string, e.g. '{"image":0,"video":0}' for text-only eval
  tool_call_parser: hermes      # enables --enable-auto-tool-choice + --tool-call-parser
  reasoning_parser: qwen3
  vllm_batch_invariant: null    # VLLM_BATCH_INVARIANT; true/false is preserved
  vllm_use_flashinfer_sampler: null # VLLM_USE_FLASHINFER_SAMPLER; false uses the built-in sampler
  vllm_use_nccl_symm_mem: null  # VLLM_USE_NCCL_SYMM_MEM; all-reduce, all-gather and reduce-scatter
  vllm_source: null             # GPU default is marin_fork; upstream requires vllm_version
  vllm_version: null            # exact upstream version, e.g. "0.29.0"
  vllm_nightly: null            # {commit, torch_backend, cuda_toolchain_version} for an unreleased vllm_version
  vllm_plugin_requirements: []  # pinned packages installed with GPU vLLM in its isolated environment
  vllm_extra_args: ["--enable-prefix-caching"]   # escape hatch for flags without a typed field
  chat_template: null           # jinja served in place of the tokenizer's own
  auto_overrides: true          # derive remaining flags + clamp max_model_len from config.json

generation:                     # per-model generation behavior
  max_gen_toks: null            # Curated generation limit and Harbor agent output budget
  extra_gen_kwargs:             # forwarded to curated requests, e.g. for a thinking model
    skip_special_tokens: "false"
  chat_template_kwargs:         # boolean or string template arguments on curated chat requests
    enable_thinking: false

agent:                          # AgentConfig -> the Harbor/agentic agent
  agent_kwargs:                 # forwarded to the agent's request against the served endpoint
    extra_body: '{"chat_template_kwargs":{"enable_thinking":true}}'
```

`auto_serve_overrides` fills unset serve fields from the model's `config.json` and may clamp
`max_model_len` to the model's native limit. `resource_hint.hbm_gb` is portable across TPU and GPU;
`resource_hint.gpu` declares that the model requires one of the listed exact GPU shapes.

For a GPU model whose architecture is supplied by a vLLM plugin, set `vllm_source: upstream`,
an exact `vllm_version`, and a plugin requirement pinned to a release or commit. These fields are
part of the model catalog and the recorded effective serving configuration. The plugin is installed
in the same isolated environment as vLLM; TPU serving does not accept these GPU-specific settings.

A GPU model that needs upstream support not yet in a vLLM release can pin a nightly build. Set
`vllm_source: upstream`, `vllm_nightly.commit` to the full upstream `main` commit,
`vllm_nightly.torch_backend` to the CUDA build its wheels target (for example `cu132`),
`vllm_nightly.cuda_toolchain_version` to the CUDA toolkit that PyTorch build pins (for example
`13.2.78` for `torch==2.14.0+cu132`), and
`vllm_version` to the exact version that commit's index at `https://wheels.vllm.ai/<commit>/vllm/`
publishes, for example `0.34.1rc1.dev13+g12eb1d4ee`. Not every `main` commit has published wheels;
pick one whose index exists. Replace the pin with a release once one includes the support.

## vLLM eager execution

vLLM defaults `enforce_eager` to false. `--enforce-eager` disables both `torch.compile` and CUDA
graphs. In a controlled Snowball Grug-67B run on eight H100s, eager execution took 44.025 ms per
decode token, versus 6.523 ms with the vLLM default (6.75x faster). See [issue #9339](https://github.com/marin-community/marin/issues/9339)
for the setup, startup costs, and limits of that measurement.

Keep compilation and CUDA graphs enabled for serving, evaluation, and inference smoke tests.
Do not copy eager-mode flags from debugging runs or historical ablations into a launch configuration.
Shorter startup alone does not justify disabling these optimizations.

Agents must not add `--enforce-eager` or `--i-know-i-am-making-vllm-slow` unless the user explicitly
requests eager execution. If an observed compilation or CUDA-graph failure suggests eager execution
is needed, report the failure and performance cost and obtain that instruction before launching.
Record the failure and the user's requested exception in a configuration or launch-file comment,
or in the task's durable launch record for a CLI-only invocation.

The launch guard rejects `--enforce-eager` without the acknowledgement flag
`--i-know-i-am-making-vllm-slow`. Treat that error as a reason to investigate the configuration;
do not satisfy it automatically. Marin removes the acknowledgement flag before launching vLLM
and rejects `--config` files and abbreviated eager flags.

## NCCL symmetric memory

Set `serve.vllm_use_nccl_symm_mem: true` to pass `VLLM_USE_NCCL_SYMM_MEM=1` to GPU vLLM workers.
`false` disables it; an unset field leaves the worker environment unchanged and is omitted from
`record.json`. vLLM ignores the request under `VLLM_BATCH_INVARIANT=1` or when its NCCL allocator
fails to build. For Hero BF16 on 16 GB200 GPUs with full decode graphs, it raised throughput from
1,321 to 1,399 output tokens/s/GPU at 1,024 requests in flight
([issue #11430](https://github.com/marin-community/marin/issues/11430#issuecomment-6876476443)).

## Hero serving settings

Hero BF16 was measured on 16 GB200 GPUs at TP1/DP16/EP16; see
[issue #11430](https://github.com/marin-community/marin/issues/11430) for the results. ShortConv
models need `cudagraph_mode: PIECEWISE` with `vllm::unified_attention_with_output`,
`vllm::unified_kv_cache_update` and `vllm::grug_moe_short_conv` in
`compilation_config.splitting_ops`. On the pinned wheel, any other graph mode or a missing split op
runs without CUDA graphs. Use `max_num_seqs: 128` for 1K outputs and about 96 for 4K outputs, which
avoids KV-cache waiting. The catalog cannot launch this topology yet: it needs data parallelism
across hosts.

## Multi-node GPU serving

Set `serve.instances` to start independent model servers behind Marin's inference broker. Each
instance receives the configured `resource_hint`, tensor/data parallelism, and pipeline task count.
For example, `resource_hint.gpu: {H100: 8}`, `data_parallel_size: 8`, and `instances: 2` starts two
DP8 servers on two H100x8 Iris tasks and exposes one load-balanced endpoint. This provides 16
independent data-parallel replicas; it does not create one 16-GPU model replica.

Set `serve.pipeline_parallel_size` to the number of Iris tasks in one vLLM serving gang.
Each task runs one pipeline stage. For example, `resource_hint.gpu: {H100: 8}` with
`tensor_parallel_size: 8`, `pipeline_parallel_size: 2`, and `data_parallel_size: 1`
requests 16 GPUs across two tasks and exposes one endpoint.

vLLM splits layers evenly across pipeline stages by default. A model whose layers share KV caches across
a stage boundary fails at startup and needs an explicit split: set `serve.pipeline_layer_partition` to
one layer count per stage, which Marin passes to vLLM as `VLLM_PP_LAYER_PARTITION`.

Multi-node serving requires the vLLM backend, an explicit GPU resource hint, and an explicit
tensor parallel size. Tensor parallel size times data parallel size must equal GPUs per task.
Set `serve.backend: vllm` (the default); an unset `data_parallel_size` means 1.
Pipeline parallelism requires `serve.instances: 1`; the inference API also requires `broker=None`.
Host memory and CPU hints apply to each task; automatic host memory sizing budgets the full
checkpoint on each task.

For multi-node serving, the launcher owns topology flags; do not put tensor, data, or pipeline
parallel sizes, node ranks, device IDs, rendezvous addresses, or `--headless` in `vllm_extra_args`.
`gpu_memory_utilization` accepts values in `(0, 1]` for vLLM. Set either the typed field or its
raw flag, not both. Unset values retain vLLM's default.

Dry runs show per-task hardware and serving geometry. `record.json` retains requested model settings
and reports resolved settings under `serving` after endpoint startup; `serving.effective` is false
when startup did not produce an endpoint. Multi-node support and context capacity depend on the
pinned vLLM version and model architecture and must be checked on the target hardware.

## Session routing for native vLLM DP

`marin-serve local` and `marin-serve iris` accept `--session-router`. With prefix caching enabled,
reuse `X-Session-ID` on later turns to reach the same DP rank and KV cache. Direct serving remains the default.

Two local replicas:

```bash
uv run marin-serve local Qwen/Qwen3-0.6B --launcher cuda \
  --tensor-parallel-size 1 --data-parallel-size 2 --session-router \
  --vllm-arg=--enable-prefix-caching
```

16 replicas on four 4-GPU Iris tasks:

```bash
uv run marin-serve iris Qwen/Qwen3-0.6B --cluster cw-us-east-08a --gpu GB200x4 \
  --tensor-parallel-size 1 --data-parallel-size 16 \
  --session-router --proxy-timeout 14400 \
  --vllm-source marin-fork --vllm-release marin-vllm-gpu-manifest.json \
  --vllm-arg=--enable-prefix-caching
```

`--vllm-release` reads `marin-vllm-gpu-manifest.json` from a [published release](https://github.com/marin-community/vllm/releases).
Omitting it uses the normal runtime.
The isolated `vllm-router==0.1.16` process uses vLLM's global internal DP frontend;
external/hybrid balancing is unsupported. `/metrics` reports vLLM worker counters.

Without a session ID, [upstream hashing](https://github.com/vllm-project/router/blob/v0.1.16/docs/load_balancing/README.md#consistent-hash)
applies; identical bodies can select the same busy rank.

HTTP retries are disabled; client SDK retries are separate. The router timeout is 14,400 seconds.
Iris `--proxy-timeout` defaults to 600 seconds and raises the router timeout when larger.
Streaming disconnects are detected on the next upstream chunk; stalled streams wait for timeout.

Rank failures can remap sessions and lose cached prefixes. If a worker or router exits, cancel
with `iris --cluster CLUSTER job cancel JOB`, wait for tasks to stop, then resubmit.

In Python, set `VllmEngineConfig(router=VllmRouterConfig())` and `IrisConfig.serving_geometry`:

```python
ServingGeometry(
    tensor_parallel_size=1,
    data_parallel_size=16,
    pipeline_parallel_size=1,
    gpus_per_task=4,
)
```

This gives four tasks, four local ranks per task and 16 router targets. Omitted model TP uses
geometry; explicit TP must match. Multi-task or routed geometry requires `instances=1`, `broker=None`.
`ServedModelConfig.object_store_load_mode` defaults to `STREAM`. With `STAGE_LOCAL`, every task stages its
own local copy of the weights before vLLM starts.
