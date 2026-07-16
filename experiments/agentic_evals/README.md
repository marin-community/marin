# agentic-evals

Standalone agentic evaluation package for running Harbor-based evals
(SWE-bench, Terminal-Bench, etc.) on Iris TPU/GPU clusters.

Extracted from [OpenThoughts-Agent](https://github.com/open-thoughts/openthoughts-agent) (#6958) as a self-contained package with no
`hpc.*` or `scripts.*` dependencies.

## Install

```bash
# Base (harbor + argparse + YAML)
pip install -e .

# With HuggingFace trace upload
pip install -e ".[hf]"

# With Iris TPU/GPU cluster backend
pip install -e ".[iris]"

# With local GPU serving (Ray + vLLM)
pip install -e ".[serve]"

# Everything for development
pip install -e ".[dev,hf,iris,serve]"
```

## Quickstart

### Submit an eval job to Iris

```bash
python -m agentic_evals.launch \
    --harbor_config harbor.yaml \
    --model Qwen/Qwen3-32B \
    --dataset_path ./tasks \
    --preset tb2 \
    --tpu v6e-4 \
    --cluster-config /path/to/iris.yaml
```

### Run an eval locally (single-node Ray + vLLM)

```bash
python -m agentic_evals.run_eval \
    --harbor_config harbor.yaml \
    --model Qwen/Qwen3-32B \
    --dataset_path ./tasks \
    --agent terminus-2 \
    --n_concurrent 16
```

### Dry run (print commands without executing)

Append `--dry_run` to either command.

## Architecture

```
agentic_evals/
  presets/          Benchmark preset catalog (dataset, concurrency, parser)
    *.yaml          One per benchmark (tb2, swebench, aider, ...)
  serve/            vLLM serve config construction
    vllm_args.py    Config dict -> CLI args + env vars
    model_config.py Per-model YAML resolver (max_model_len, TP, parsers, ...)
    tpu.py          TPU-specific serve-flag stripping/defaults
    models/         Per-model vLLM config YAMLs (the data registry)
  harness/          Harbor harness wiring
    config.py       Harbor config load + agent-kwargs merge
    command.py      Build harbor jobs start command + CLI execution
    job_config.py   JobConfig loading + metric filtering
    trial_prune.py  Prune infra-errored trials for auto-resume
    _compat.py      Harbor legacy/unified API shims
  runtime/          Worker runtime
    args.py         Reusable argparse groups
    runner.py       LocalHarborRunner (Ray+vLLM lifecycle + harbor exec)
    vllm_server.py  VLLMServer context manager (start, health, warmup, stop)
    docker.py       Docker/Podman runtime detection
  results/          Pluggable result sinks
    __init__.py     ResultSink Protocol + NoOpResultSink
    local.py        LocalResultSink (writes result.json)
    hf_upload.py    HFResultSink (uploads traces to HuggingFace)
    infra_errors.py Infrastructure-error classification (INFRA_ERROR_TYPES)
  backends/         Pluggable cluster backends
    __init__.py     EvalBackend Protocol
    iris.py         IrisBackend (Marin Iris TPU/GPU adapter)
  launch.py         Launcher CLI entry point
  run_eval.py       Worker CLI entry point (EvalRunner)
```

## Key Concepts

### Presets
Benchmark presets (`presets/<name>.yaml`) seed `--dataset_path`,
`--n_concurrent`, agent parser, and agent kwargs. Explicit CLI flags always
override preset values.

### Model Config
Per-model YAMLs (`serve/models/<org>/<slug>.yaml`) define vLLM serve
parameters (TP/DP, `max_model_len`, `tool_call_parser`, `reasoning_parser`,
`hf_overrides`, etc.). The resolver merges base intrinsics -> subsystem
overlays (eval/datagen/iris) -> hardware variants.

### Result Sinks
Pluggable post-run handling via the `ResultSink` protocol. The runner
selects the sink based on args: `HFResultSink` when `--upload_hf_repo` is
set, `LocalResultSink` otherwise. Custom sinks implement `publish()`.

### Backends
Pluggable cluster submission via the `EvalBackend` protocol. The launcher
delegates to `IrisBackend` (Marin Iris) by default; custom backends implement
`submit()`, `query()`, `logs()`.

## Dependencies

- **harbor** (>=0.7.0) — the agentic eval harness
- **pyyaml** — config loading
- Optional: **vllm**, **ray**, **huggingface_hub**, **iris-client**, **vllm-tpu**, **jax**

## License

Apache-2.0
