# Levanter

<a href="https://github.com/stanford-crfm/levanter/actions?query=branch%3Amain++">
    <img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/stanford-crfm/levanter/run_tests.yaml?branch=main">
</a>
<a href="https://levanter.readthedocs.io/en/latest/?badge=latest">
    <img alt="Documentation Status" src="https://readthedocs.org/projects/levanter/badge/?version=latest">
</a>
<a href="">
<img alt="License" src="https://img.shields.io/github/license/stanford-crfm/levanter?color=blue" />
</a>
<a href="https://https://pypi.org/project/levanter/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/levanter?color=blue" />
</a>


<!--levanter-intro-start-->
> *You could not prevent a thunderstorm, but you could use the electricity; you could not direct the wind, but you could trim your sail so as to propel your vessel as you pleased, no matter which way the wind blew.* <br/>
> — Cora L. V. Hatch

Levanter is a framework for training large language models (LLMs) and other foundation models that strives for legibility, scalability, and reproducibility:

1. **Legible**: Levanter uses our named tensor library [Haliax](https://github.com/stanford-crfm/haliax) to write easy-to-follow, composable deep learning code, while still being high performance.
2. **Scalable**: Levanter scales to large models, and to be able to train on a variety of hardware, including GPUs and TPUs.
3. **Reproducible**: Levanter is bitwise deterministic, meaning that the same configuration will always produce the same results, even in the face of preemption and resumption.

We built Levanter with [JAX](https://github.com/jax-ml/jax), [Equinox](https://github.com/patrick-kidger/equinox), and [Haliax](https://github.com/stanford-crfm/haliax).

## Documentation

Levanter's documentation is available at [levanter.readthedocs.io](https://levanter.readthedocs.io/en/latest/).
Haliax's documentation is available at [haliax.readthedocs.io](https://haliax.readthedocs.io/en/latest/).

## Features

* **Distributed Training**: We support distributed training on TPUs and GPUs, including FSDP and tensor parallelism.
* **Compatibility**: Levanter supports importing and exporting models to/from the Hugging Face ecosystem, including tokenizers, datasets, and models via [SafeTensors](https://github.com/huggingface/safetensors).
* **Performance**: Levanter's performance rivals commercially-backed frameworks like MosaicML's Composer or Google's MaxText.
* **Resilience**: Levanter supports fast, distributed checkpointing and fast resume from checkpoints with no data seek, making Levanter robust to preemption and hardware failure.
* **Cached On-Demand Data Preprocessing**: We preprocess corpora online, but we cache the results of preprocessing so
that resumes are much faster and so that subsequent runs are even faster. As soon as the first part of the cache is complete, Levanter will start training.
* **Logging**: Levanter logs a rich and detailed set of metrics covering loss and performance. Levanter also supports a few different logging backends, including [WandB](https://wandb.ai/site) and [TensorBoard](https://www.tensorflow.org/tensorboard). (Adding a new logging backend is easy!) Levanter even exposes the ability
to log inside of JAX `jit`-ted functions.
* **Reproducibility**: On TPU, Levanter is bitwise deterministic, meaning that the same configuration will always produce the same results, even in the face of preemption and resumption.
* **Distributed Checkpointing**: Distributed checkpointing is supported via Google's [TensorStore](https://google.github.io/tensorstore/) library. Training can even be resumed on a different number of hosts, though this breaks reproducibility for now.
* * **Optimization**: Levanter supports the new [Sophia](https://arxiv.org/abs/2305.14342) optimizer, which can be 2x as fast as Adam. We also support [Optax](https://github.com/deepmind/optax) for optimization with AdamW, etc.
* * **Flexible**: Levanter supports tuning data mixtures without having to retokenize or shuffle data.

<!--levanter-intro-end-->

Levanter was created by [Stanford's Center for Research on Foundation Models (CRFM)](https://crfm.stanford.edu/)'s research engineering team.
You can also find us in the #levanter channel on the unofficial [Jax LLM Discord](https://discord.gg/CKazXcbbBm)

## Getting Started

Here is a small set of examples to get you started. For more information about the various configuration options,
please see the [Getting Started](./docs/Getting-Started-Training.md) guide or the [In-Depth Configuration Guide](docs/reference/Configuration.md).
You can also use `--help` or poke around other configs to see all the options available to you.


### Installing Levanter

<!--levanter-installation-start-->

After [installing JAX](https://github.com/google/jax/blob/main/README.md#installation) with the appropriate configuration
for your platform, you can install Levanter with:

```bash
pip install levanter
```

or using the latest version from GitHub:

```bash
pip install git+https://github.com/stanford-crfm/levanter.git
wandb login  # optional, we use wandb for logging
```

If you're developing Haliax and Levanter at the same time, you can do something like.
```bash
git clone https://github.com/stanford-crfm/levanter.git
cd levanter
pip install -e .
cd ..
git clone https://github.com/stanford-crfm/haliax.git
cd haliax
pip install -e .
cd ../levanter
```

<!--levanter-installation-end-->

Please refer to the [Installation Guide](docs/Installation.md) for more information on how to install Levanter.

If you're using a TPU, more complete documentation for setting that up is available [here](docs/Getting-Started-TPU-VM.md). GPU support is still in-progress; documentation is available [here](docs/Getting-Started-GPU.md).

<!--levanter-user-guide-start-->

### Training a GPT2-nano

As a kind of hello world, here's how you can train a GPT-2 "nano"-sized model on a small dataset.

```bash
python -m levanter.main.train_lm --config_path config/gpt2_nano.yaml

# alternatively, if you didn't use -e and are in a different directory
python -m levanter.main.train_lm --config_path gpt2_nano
```

This will train a GPT2-nano model on the [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) dataset.

### Training a Llama-small on your own data

You can also change the dataset by changing the `dataset` field in the config file.
If your dataset is a [Hugging Face dataset](https://huggingface.co/docs/datasets/loading_datasets.html), you can use the `data.id` field to specify it:

```bash
python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.id openwebtext

# optionally, you may specify a tokenizer and/or a cache directory, which may be local or on gcs
python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.id openwebtext --data.tokenizer "NousResearch/Llama-2-7b-hf" --data.cache_dir "gs://path/to/cache/dir"
```

If instead your data is a list of URLs, you can use the `data.train_urls` and `data.validation_urls` fields to specify them.
Data URLS can be local files, gcs files, or http(s) URLs, or anything that [fsspec](https://filesystem-spec.readthedocs.io/en/latest/) supports.
Levanter (really, fsspec) will automatically uncompress `.gz` and `.zstd` files, and probably other formats too.

```bash
python -m levanter.main.train_lm --config_path config/llama_small_fast.yaml --data.train_urls ["https://path/to/train/data_*.jsonl.gz"] --data.validation_urls ["https://path/to/val/data_*.jsonl.gz"]
```

### Customizing a Config File

You can modify the config file to change the model, the dataset, the training parameters, and more. Here's
the `llama_small_fast.yaml` file:

```yaml
data:
  train_urls:
      - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_train.{1..128}-of-128.jsonl.gz"
  validation_urls:
      - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_val.{1..8}-of-8.jsonl.gz"
  cache_dir: "gs://pubmed-mosaic/tokenized/openwebtext/"
model:
  type: llama
  hidden_dim: 768
  intermediate_dim: 2048
  num_heads: 12
  num_kv_heads: 12
  num_layers: 12
  seq_len: 1024
  gradient_checkpointing: true
trainer:
  tracker:
    type: wandb
    project: "levanter"
    tags: [ "openwebtext", "llama" ]

  mp: p=f32,c=bfloat16
  model_axis_size: 1
  per_device_parallelism: 4

  train_batch_size: 512
optimizer:
  learning_rate: 6E-4
  weight_decay: 0.1
  min_lr_ratio: 0.1
```

### Other Architectures

Currently, we support the following architectures:

* GPT-2
* [LLama](https://ai.meta.com/llama/), including Llama 1, 2 and 3
* [Gemma](https://ai.google.dev/gemma), including Gemma 1, 2 and Gemma 3.
* [Qwen2](https://huggingface.co/Qwen/Qwen2.5-7B)
* [Qwen3](https://huggingface.co/Qwen/Qwen3-8B)
* [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
* [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
* [Olmo2](https://huggingface.co/allenai/Olmo-2-1124-7B)

We plan to add more in the future.

For speech, we currently only support [Whisper](https://huggingface.co/openai/whisper-large-v3).

#### Continued Pretraining with Llama

Here's an example of how to continue pretraining a Llama 1 or Llama 2 model on the OpenWebText dataset:

```bash
python -m levanter.main.train_lm --config_path config/llama2_7b_continued.yaml
```


## Distributed and Cloud Training

### Training on a TPU Cloud VM

Please see the [TPU Getting Started](docs/Getting-Started-TPU-VM.md) guide for more information on how to set up a TPU Cloud VM and run Levanter there.

### Training with CUDA

Please see the [CUDA Getting Started](docs/Getting-Started-GPU.md) guide for more information on how to set up a CUDA environment and run Levanter there.

<!--levanter-user-guide-end-->

## Contributing

[![GitHub repo Good Issues for newbies](https://img.shields.io/github/issues/stanford-crfm/levanter/good%20first%20issue?style=flat&logo=github&logoColor=green&label=Good%20First%20issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) [![GitHub Help Wanted issues](https://img.shields.io/github/issues/stanford-crfm/levanter/help%20wanted?style=flat&logo=github&logoColor=b545d1&label=%22Help%20Wanted%22%20issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) [![GitHub Help Wanted PRs](https://img.shields.io/github/issues-pr/stanford-crfm/levanter/help%20wanted?style=flat&logo=github&logoColor=b545d1&label=%22Help%20Wanted%22%20PRs)](https://github.com/stanford-crfm/levanter/pulls?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) [![GitHub repo Issues](https://img.shields.io/github/issues/stanford-crfm/levanter?style=flat&logo=github&logoColor=red&label=Issues)](https://github.com/stanford-crfm/levanter/issues?q=is%3Aopen)

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## License

Levanter is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.
