# Dataset Format Reference

This document outlines the supported dataset formats in Levanter and how each format transforms raw data into model-ready tokens. These formats determine how Levanter tokenizes, structures, and masks training data.
For a more directed, tutorial-like guide, see the [Training Data Guide](../guides/Training-Data-Guide.md).

## Overview

Levanter supports three canonical formats:

| Format       | Intended Use                       | Required Fields                               | YAML Spec Example  |
|--------------|------------------------------------|-----------------------------------------------|--------------------|
| `text`       | Language modeling pretraining      | `"text"` → string                             | `type: text`       |
| `chat`       | Conversational fine-tuning (SFT)   | `"messages"` → list of turns in OpenAI format | `type: chat`       |
| `supervised` | Instruction tuning / seq2seq tasks | two string fields (e.g. `prompt`, `answer`)   | `type: supervised` |

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
- Builds `loss_mask` so that only assistant spans are predicted

### Chat Templates

Chat templates are Jinja2 templates that format a list of messages into a single string.
Hugging Face provides mostly sufficient documentation [here](https://huggingface.co/docs/transformers/main/en/chat_templating_writing)
but **misses one important detail**: the template must contain `{%generation%}` to indicate where the assistant message
should be inserted. (See [here](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1530).)
We need this tag to construct the `loss_mask` for training, unless `mask_user_turns` is set to `false`.

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

## `supervised` Format

Used for single-turn instruction following or sequence-to-sequence tasks.

**Expected Input:**
```jsonl
{"prompt": "Translate to French: Hello", "answer": "Bonjour"}
```

#### Configuration:
```yaml
format:
  type: supervised
  input_field: prompt
  output_field: answer
  separate_with: "\n"  # optional separator between input and output
  pack: true  # optional, default is true
  mask_inputs: true  # optional, default is true
```

* `pack: true` will pack multiple examples into a single example if they fit within the context length.
* `pack: false` will produce a single example per conversation. This is very inefficient.

#### Processing:
- Tokenizes `prompt`, then tokenizes `answer` (with optional separator)
- Produces a single `input_ids` sequence
- Computes `sources_len` so that loss is masked on prompt tokens (assuming `mask_inputs: true`)

---


# API


## Overall Configs

::: levanter.data.text.LMMixtureDatasetConfig
::: levanter.data.text.SingleDatasetLMConfigBase

::: levanter.data.text.HfSingleDatasetLMConfig
::: levanter.data.text.UrlSingleDatasetLMConfig

## Formats

::: levanter.data.text.LmDatasetFormatBase

::: levanter.data.text.ChatLmDatasetFormat
::: levanter.data.text.SupervisedLmDatasetFormat
::: levanter.data.text.TextLmDatasetFormat

## Datasets


::: levanter.data.text.CausalLmDataset
::: levanter.data.text.MultiturnChatDataset
::: levanter.data.text.SupervisedDataset
