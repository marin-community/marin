# How to Use Dataset Formats in Levanter

This guide walks you through configuring and using various dataset formats in Levanter.

---

## Step 1: Identify the Format for Each Dataset

Each dataset in your configuration can use a different format depending on its structure and intended use. Currently, the supported formats are:

| Format       | Best For                           | Example HF Dataset                                                                                   |
|--------------|------------------------------------|------------------------------------------------------------------------------------------------------|
| `text`       | Pretraining on plain text          | [`mlfoundations/dclm-baseline-1.0`](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) |
| `chat`       | Multi-turn chat models (SFT)       | [`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)           |
| `prebuilt`   | Pre-tokenized cache inputs         | (custom prebuilt data)                                                                               |

---

## Step 2: Prepare Your Data

All data should either be in [JSONL format](https://jsonlines.org/) — one JSON object per line –
or be a [Hugging Face dataset]](https://huggingface.co/docs/datasets/create_dataset) with appropriate field names.

### `text` format

```jsonl
{"text": "The quick brown fox jumps over the lazy dog."}
```

### `chat` format

```jsonl
{"messages": [ {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"} ]}
```
### `prebuilt` format

```jsonl
{"input_ids": [101, 2023, 2003, 1037, 7099], "loss_weights": [1, 1, 1, 1, 1]}
```

The `prebuilt` format is primarily intended for cases where you build the cache ahead of time using a custom
Zephyr pipeline or `SerialCacheWriter`. For the recommended approach and examples, see
[Direct Cache Construction](./Direct-Cache-Construction.md).

---

## Step 3: Write Your Config

> **URL vs Hugging Face dataset**
>
> - Use `train_urls:` and `validation_urls:` if your data is stored in GCS, S3, or local files.
> - Use `id:` if your dataset is hosted on Hugging Face (e.g. `allenai/tulu-3-sft-mixture`).
>   You can also specify `name:` if the dataset has named subsets.

Use either a single data source:

```yaml
data:
  train_urls: ["gs://bucket/train.jsonl.gz"]
  validation_urls: ["gs://bucket/val.jsonl.gz"]
  format:
    type: prebuilt
    input_ids_key: input_ids
    loss_weights_key: loss_weights
  tokenizer: stanford-crfm/marin-tokenizer
  cache_dir: gs://bucket/cache
```

Or a mixture of multiple datasets:

```yaml
data:
  components:
    owt:
      source:
        type: url
        train_urls: ["gs://openwebtext/train.{1..128}.jsonl.gz"]
    alpaca:
      source:
        type: url
        train_urls: ["gs://bucket/alpaca_pretokenized.jsonl.gz"]
      format:
        type: prebuilt
        input_ids_key: input_ids
        loss_weights_key: loss_weights
    tulu:
      source:
        type: hf
        id: allenai/tulu-3-sft-mixture
      format:
        type: chat
        messages_key: messages
  train_weights:
    owt: 0.5
    alpaca: 0.3
    tulu: 0.2
  tokenizer: stanford-crfm/marin-tokenizer
  cache_dir: gs://bucket/cache
```

!!! tip

     `train_weights` lets you mix datasets with different importance. Values are normalized.


!!! warning

    To use a chat format, your tokenizer must have a `chat_template`, or you must provide one in the config.
    This template must be formatted to work for training (which most are not, and it is not well documented in Hugging Face).
    The `stanford-crfm/marin-tokenizer` has a default template that works. See our [chat template docs](../reference/Data-Formats.md#chat-templates) for more details.

https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1530


### Training Mixture Schedules

Levanter supports training mixtures with different weights at different points in the training schedule.

To do so, you need to specify the transition points in the `train_weights` section:

```yaml
data:
  components:
    owt:
      source:
        type: url
        train_urls: ["gs://openwebtext/train.{1..128}.jsonl.gz"]
    alpaca:
      source:
        type: url
        train_urls: ["gs://bucket/alpaca_pretokenized.jsonl.gz"]
      format:
        type: prebuilt
        input_ids_key: input_ids
        loss_weights_key: loss_weights
    tulu:
      source:
        type: hf
        id: allenai/tulu-3-sft-mixture
      format:
        type: chat
        messages_key: messages
  train_weights:
    - [0, {"owt": 0.5, "alpaca": 0.3, "tulu": 0.2}]
    - [1000, {"owt": 0.2, "alpaca": 0.4, "tulu": 0.4}]
  tokenizer: stanford-crfm/marin-tokenizer
```

(Again, the weights need not sum to 1.)

---

## Step 4: Launch Training

Run with:

```bash
python -m levanter.main.train_lm --config_path my_config.yaml
```

The cache will build on-the-fly using Ray, and training will begin as soon as data is ready.

---

## Tips

- Always validate that your JSONL is well-formed and contains all required fields.
- Each format must include a `format:` block unless you're using plain `text` with default settings.
- Use `train_weights: { mydataset: 0.0 }` to include a dataset for evaluation only.

For more details, see the [Dataset Formats Reference](../reference/Data-Formats.md).
