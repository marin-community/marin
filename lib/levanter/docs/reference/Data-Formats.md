# Dataset Format Reference

This document outlines the supported dataset formats in Levanter and how each format transforms raw data into model-ready tokens. These formats determine how Levanter tokenizes, structures, and masks training data.
For a more directed, tutorial-like guide, see the [Training Data Guide](../guides/Training-Data-Guide.md).

## Overview

Levanter supports three canonical formats:

| Format       | Intended Use                       | Required Fields                               | YAML Spec Example  |
|--------------|------------------------------------|-----------------------------------------------|--------------------|
| `text`       | Language modeling pretraining      | `"text"` → string                             | `type: text`       |
| `chat`       | Conversational fine-tuning (SFT)   | `"messages"` → list of turns in OpenAI format | `type: chat`       |
| `prebuilt`   | Pre-tokenized cache inputs         | `"input_ids"` → list of ints                  | `type: prebuilt`   |

!!! tip

     Extra fields in the JSON are ignored. All input must be valid JSONL (i.e., one JSON object per line).

---

## `text` Format

This is the default format used for pretraining.

**Expected Input:**
```jsonl
{"text": "The quick brown fox jumps over the lazy dog."}
```

#### Configuration

!!! tip

    For `text`, `format` is optional.

```yaml
format:
  type: text
  text_key: text  # optional, default is "text"
```

#### Processing:
- Tokenizes the value in `text_key`
- Appends EOS token and prepends BOS token if not already present

---

## `chat` Format

Used for multi-turn conversation datasets (e.g. ShareGPT, OpenChat, Tulu).

**Expected Input:**
```jsonl
{"messages": [
  {"role": "user", "content": "Hello!"},
  {"role": "assistant", "content": "Hi there!"}
]}
```

#### Configuration:

```yaml
format:
  type: chat
  messages_key: messages  # optional (default)
  pack: true  # optional (default)
  mask_user_turns: true  # optional (default). See below for important details!
  chat_template: |
    {{ bos_token }}
    {%- for message in messages -%}
    {%- if message['role'] == 'assistant' -%}
        <|start_header_id|>{{ message['role'] }}<|end_header_id|>
    {% generation %}{{- message['content'] | trim }}<|eot_id|>{% endgeneration %}\n
    {% else %}
    <|start_header_id|>{{ message['role'] }}<|end_header_id|>
    {{ message['content'] | trim }}<|eot_id|>
    {% endif %}
    {%- endfor -%}
    {%- if add_generation_prompt -%}
    <|start_header_id|>assistant<|end_header_id|>\n{% endif -%}
```

* `pack: true` will pack multiple conversations into a single example if they fit within the context length.
* `pack: false` will produce a single example per conversation. This is very inefficient.

#### Processing:
- Requires a `chat_template`:
  - If not supplied in config, will use `tokenizer.chat_template`
  - If neither is available, raises an error
- Uses template to flatten messages into a single token sequence
- Builds `loss_weight` so that only assistant spans are predicted

### Chat Templates

Chat templates are Jinja2 templates that format a list of messages into a single string.
Hugging Face provides mostly sufficient documentation [here](https://huggingface.co/docs/transformers/main/en/chat_templating_writing)
but **misses one important detail**: the template must contain `{%generation%}` to indicate where the assistant message
should be inserted. (See [here](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1530).)
We need this tag to construct the `loss_weight` for training, unless `mask_user_turns` is set to `false`.

Unfortunately, almost no tokenizers use this format, so you will need to write your own.

Here is an example we use in the [stanford-crfm/marin-tokenizer](https://huggingface.co/stanford-crfm/marin-tokenizer)
tokenizer:

```
{{ bos_token }}
{%- for message in messages -%}
{%- if message['role'] == 'assistant' -%}
    <|start_header_id|>{{ message['role'] }}<|end_header_id|>
{% generation %}{{- message['content'] | trim }}<|eot_id|>{% endgeneration %}\n
{% else %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>
{{ message['content'] | trim }}<|eot_id|>
{% endif %}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|start_header_id|>assistant<|end_header_id|>\n{% endif -%}
```

The key points are:
* Wrap the assistant message in `{% generation %}` and `{% endgeneration %}` to indicate what the model is responsible
for predicting. Jinja's handling of white space is confusing to me, so you'll want to be careful there.
* Use `{{ bos_token }}` to prepend the BOS token.
* Ensure that the generation prompt resembles the format of the training data (e.g. the final `\n`).


---

## `prebuilt` Format

Used when your dataset already contains tokenized (pretokenized) sequences and optional loss weights.
The primary use case is when you have some custom logic for creating a dataset consisting of token IDs
and optionally loss weights, and you want to skip tokenization in Levanter.

`prebuilt` supports being created from jsonl/parquet, similarly to other formats, though
it is primarily intended to be used with [Direct Cache Construction](../guides/Direct-Cache-Construction.md).

**Expected Input:**
```jsonl
{"input_ids": [101, 2023, 2003, 1037, 7099], "loss_weights": [1, 1, 1, 1, 1]}
```

`loss_weights` is optional. When provided, it is multiplied by the standard causal mask.

#### Configuration:
```yaml
format:
  type: prebuilt
  input_ids_key: input_ids  # optional, default is "input_ids"
  loss_weights_key: loss_weights  # optional
```

Note that when being used programmatically, `prebuilt` also supports a `loss_weight_transform` `Callable` that
can be used to modify the loss weights on-the-fly. Primarily this is intended for thresholding or scaling loss weights
without needing to rebuild the cache.

#### Processing:
- Reads `input_ids` directly (no tokenization).
- Optional `loss_weights` are applied and multiplied by the causal mask.

---


# API


## Overall Configs

::: levanter.data.text.LmDataConfig
::: levanter.data.text.DatasetComponent
::: levanter.data.text.LmDatasetSourceConfigBase
::: levanter.data.text.HfDatasetSourceConfig
::: levanter.data.text.UrlDatasetSourceConfig

## Formats

::: levanter.data.text.LmDatasetFormatBase

::: levanter.data.text.ChatLmDatasetFormat
::: levanter.data.text.PrebuiltLmDatasetFormat
::: levanter.data.text.TextLmDatasetFormat

## Datasets


::: levanter.data.text.TokenSeqDataset
::: levanter.data.text.CausalLmDataset
::: levanter.data.text.ChatDataset
