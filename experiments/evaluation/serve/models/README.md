# Model serve catalog

One `<org>/<model>.yaml` per model. Each file deserializes (via draccus) into a
`marin.evaluation.model_config.ModelConfig`; unknown fields or mistyped values fail at load, not at
serve time. `experiments.evaluation.models.models()` scans this directory once and merges the entries
with the Python factory entries defined in `models.py`. Files and directories whose names start with
`_` or `.` are skipped.

## Schema

```yaml
name: qwen3-32b                 # registry key (must be unique across catalog + factories)
location: Qwen/Qwen3-32B        # HF repo id, or gs://|s3:// HF-format export dir
revision: null                  # pin an immutable checkpoint (base models); optional
tokenizer: null                 # required only when location is an object-store path
apply_chat_template: true       # whether Evalchemy formats requests with the tokenizer chat template

serve:                          # ServeConfig -> vllm serve
  hbm_gb: 84                    # serving HBM budget the hardware selector sizes a slice from
  fixed_gpu: null               # [gpu_type, count] to pin an exact shape instead of sizing
  gpu_only: false               # force the GPU path (quantized / fork-only architectures)
  tensor_parallel_size: 2
  data_parallel_size: null
  max_model_len: 32768
  swap_space_gb: 32             # CPU KV offload (GPU-only; the TPU serve path strips it)
  trust_remote_code: true
  hf_overrides: null            # JSON string, e.g. rope-scaling overrides
  limit_mm_per_prompt: null     # JSON string, e.g. '{"image":0,"video":0}' for text-only eval
  tool_call_parser: hermes      # enables --enable-auto-tool-choice + --tool-call-parser
  reasoning_parser: qwen3
  vllm_extra_args: ["--enable-prefix-caching"]   # escape hatch for flags without a typed field
  chat_template: null           # jinja served in place of the tokenizer's own
  serve_memory: null            # host-memory request override for the serve child
  target_cluster: null          # CoreWeave peer a GPU job routes to
  auto_overrides: true          # derive remaining flags + clamp max_model_len from config.json
  variants:                     # per-hardware overrides, keyed by accelerator label
    gh200:
      tensor_parallel_size: 1

generation:                     # GenerationConfig -> evalchemy --gen_kwargs
  max_gen_toks: null            # per-model generation budget override
  extra_gen_kwargs:             # forwarded verbatim, e.g. for a thinking model
    skip_special_tokens: "false"

agent:                          # AgentConfig -> the Harbor/agentic agent
  agent_kwargs:                 # forwarded to the agent's request against the served endpoint
    extra_body: '{"chat_template_kwargs":{"enable_thinking":true}}'
```

`auto_serve_overrides` fills unset serve fields from the model's `config.json` and may clamp
`max_model_len` to the model's native limit. `variants` carry per-hardware overrides; they apply only
when the served slice's label matches (the Marin H100/GB200/TPU slices do not match `gh200`).
