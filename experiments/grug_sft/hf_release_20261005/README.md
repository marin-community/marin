---
license: openmdw-1.1
pipeline_tag: text-generation
tags:
  - marin
  - grug
  - mixture-of-experts
  - sft
  - reasoning
  - tool-use
---

# Grug 67B A2B Datakit SFT 262K — Full A/B mixture

> **Research artifact. This model has not been properly tested or evaluated and is not validated for production use.** The 262K context length is a configuration value; long-context generation has not been validated.

This is a BF16 Grug mixture-of-experts checkpoint after 4,750 supervised fine-tuning updates. The run used packed SFT data from 216 sources and long-context pretraining replay.

## Training data

The run scheduled 318,767,104,000 token positions: 4,750 updates × 256 packed sequences × 262,144 positions. The fixed mixture allocated 253,911,629,824 positions (79.65%) to SFT and 64,855,474,176 positions (20.35%) to pretraining replay. These are scheduled positions, including padding in packed sequences, rather than a count of distinct text tokens. The SFT schedule selected approximately one epoch of the packed source stores. Small sources were pooled so they could be sampled in the 64,000-sequence mixture blocks.

| Source family | Allocated SFT positions | Share of complete run |
| --- | ---: | ---: |
| `nemotron_sft_v3` | 204,759,629,824 | 64.2349% |
| `open_swe_traces` | 34,535,112,704 | 10.8340% |
| `agenttrove` | 5,239,734,272 | 1.6438% |
| `openthoughts4-code-glm-5.2-n4` | 4,401,922,048 | 1.3809% |
| `penfever-traces` | 4,321,181,696 | 1.3556% |
| `ultrachat-persona-conversations` | 326,369,280 | 0.1024% |
| `agenttrove-glm53-compactions` | 198,967,296 | 0.0624% |
| `identity-data` | 79,429,632 | 0.0249% |
| `science-tool-use-conversations` | 20,971,520 | 0.0066% |
| `glm-5.2-kernelgym-rollouts` | 16,777,216 | 0.0053% |
| `wildchat-glm53-format-completions` | 9,699,328 | 0.0030% |
| `synthetic-misconceptions-conversations` | 1,835,008 | 0.0006% |
| Long-context pretraining replay | 64,855,474,176 | 20.3457% |

[`training_data_mix.json`](training_data_mix.json) gives the exact 216 SFT source allocations and schedule. [`pretraining_replay_mix.json`](pretraining_replay_mix.json) gives the replay source weights and stores.

## Model and provenance

| Property | Value |
| --- | --- |
| Architecture | `GrugMoeForCausalLM` (`grug_moe`) |
| Parameters | 67,078,882,816 total; approximately 2B active non-embedding parameters per token |
| Experts | 256 routed experts, top-4 routing, plus a shared expert |
| Layers / hidden size | 26 / 2,560 |
| Attention heads / KV heads | 20 / 5 |
| Vocabulary size | 128,256 |
| Configured context length | 262,144 tokens |
| Sliding window | 2,048 tokens |
| QK multiplier | 1.75 |
| Checkpoint step | 161,750 |
| SFT updates | 4,750, starting from base step 157,000 |
| Export | BF16 safetensors |

The starting point was an unpublished base checkpoint at step 157,000. Training resumed its optimizer state without warmup. The peak base learning rate was 5e-5. Updates to seven special-token rows in both input embeddings and the LM head were scaled by √32, giving an effective peak learning rate of approximately 2.83e-4 for those rows. Router weights remained trainable. Router biases and deferred load-balancing bias updates stayed frozen during SFT. The deferred updates were applied to the exported weights.

Seven LM-head rows were initialized from single-token anchors: 128006 from 3560 (` role`), 128007 from 1984 (` message`), 128002 from 1781 (` think`), 128003 from 4320 (` answer`), 128005 from 5507 (` tool`), 128011 from 842 (` end`), and 128009 from EOS (128001). Only those LM-head optimizer moments were reset. Input embeddings and their optimizer moments were preserved at initialization and remained trainable. BOS and EOS were preserved; no new tokens were added.

Training run: [grug-67b-sft-20260929-special-token-lr-full-ab-4750-grouped](https://wandb.ai/marin-community/marin_moe_sft/runs/grug-67b-sft-20260929-special-token-lr-full-ab-4750-grouped). The training source is [commit `ce7c77a`](https://github.com/marin-community/marin/commit/ce7c77abcb0ada792cd6d76ba3d80db74c33a859).

Seven attention layers use full causal attention; the other 19 use the sliding window.

## Chat and generation settings

Omitting `enable_thinking` defaults to thinking. For a vLLM chat request, select nonthinking mode with:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

For local prompt formatting:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.10.05",
)
prompt_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 17 times 23?"}],
    tokenize=True,
    return_dict=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

The generation configuration stops on both `<|end_of_text|>` (128001) and `<|eot_id|>` (128009). The tokenizer's EOS remains 128001. Preserve both stop IDs when configuring another runtime.

Thinking output uses `<|start_think|>` (128002) and `<|end_think|>` (128003). Decode with `skip_special_tokens=False` when these boundaries are needed. Tools use JSON inside `<tool_call>...</tool_call>`; input arguments may be objects or JSON strings. Tool-call delimiters remain ordinary text.

The [inference template](chat_template.jinja) adds the thinking default to the [training template](training_chat_template.jinja).

## Serving and evaluation

This architecture requires a runtime with Grug MoE support. No serving runtime or benchmark result has been validated for this checkpoint. Training loss and export-format checks do not establish generation quality, safety, or numerical equivalence to the native checkpoint.

## License

The model materials are released under the [OpenMDW License Agreement, version 1.1](LICENSE).
