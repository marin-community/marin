---
license: openmdw-1.1
base_model: {SOURCE_REPO}
library_name: mlx
tags:
- mlx
- grug-moe
- apple-silicon
---

Research artifact. This model has not been properly tested or evaluated and is not necessarily secure. Use at your own risk in production settings.

# {MODEL_NAME}: MLX {RECIPE}

MLX build of [`{SOURCE_REPO}`](https://huggingface.co/{SOURCE_REPO}) (revision `{REVISION}`) for
Apple Silicon. {RECIPE_DESCRIPTION} Affine quantization, group size 64. Gated norms and the attention
gate are unquantized.

## Run it

```bash
uvx --from mlx-lm==0.32.2 mlx_lm.server --model {MLX_REPO} --trust-remote-code --max-tokens 32768
```

Keep the `mlx-lm==0.32.2` pin. The server listens on `http://127.0.0.1:8080/v1`; use
`"model": "default_model"` in requests. Reasoning comes back in the `reasoning` message field. Turn
thinking off per request with `"chat_template_kwargs": {"enable_thinking": false}`. Tool calls come
back as OpenAI `tool_calls`.

## Why --trust-remote-code

`config.json` sets `"model_file": "grug_moe_mlx.py"`, so `mlx-lm` loads the Grug MoE architecture from
this repo. The file also teaches `mlx-lm` 0.32.2 the thinking markers `<|start_think|>` and
`<|end_think|>`. That patch depends on `mlx-lm` internals and goes away once `mlx-lm` supports the
markers. `grug_moe_mlx.py` SHA-256: `{MODEL_FILE_SHA256}`.

## Parity

Checked against the Marin vLLM fork serving the source revision in bf16, comparing top-25 logprobs at
every prompt position and 64 greedy continuation tokens per case. Cases: a short passage, chat with and
without thinking, a tool call, a tool result, code, multilingual text, and {LONG_PROMPT_TOKENS} tokens of
unrepeated WikiText that run past the {SLIDING_WINDOW}-token sliding window.

{PARITY_REPORT}

## Known limits

- The base model is a research checkpoint. This parity check is the only correctness evidence for the
  conversion.
- With thinking off, the model sometimes reasons after a tool result and closes with `<|end_think|>`
  without opening it, so that text lands in the answer. vLLM behaves the same way. Keep thinking on in
  agent harnesses.
- Some clients read only `reasoning_content`; `mlx-lm` returns `reasoning`.
