# Training on Your Own Data

This guide is meant to be a detailed walkthrough of training a model on your own data using Levanter.

The basic steps are:

1. [Configure your environment/cloud](#environment-setup)
1. [Prepare your data and upload to cloud](#data-preparation)
1. [Configure your training run](#configuration)
1. [Upload the training configuration file](#upload-config-to-gcs)
1. [Launch training](#launching-training)
1. [Evaluate](#evaluation)
1. [Export your model to Huggingface](#huggingface-export)

If you're training on data that isn't text (or [audio-to-text](./guides/Training-On-Audio-Data.md)), you'll need to
write a custom cache. See the guide on [Direct Cache Construction](./guides/Direct-Cache-Construction.md).)

## Environment Setup

### TPU Setup

See the [TPU guide](./Getting-Started-TPU-VM.md). You should go through
the installation steps in that guide before continuing. Don't spin up a TPU VM instance yet, though.

### GPU Setup

See the [GPU guide](./Getting-Started-GPU.md).

### WandB Setup

Levanter mainly uses [WandB](https://wandb.ai) for logging. Create a WandB account and [get an API key](https://wandb.ai/authorize).

## Data Preparation

The key ingredient for training an LM is a lot of plain-text data.
We have two top-level ways of consuming training data: a [**single source**](#single-data-source) and [**mixture of sources**](#mixture-of-sources).
Single source is simpler and probably closer to what you're used to, while multiple
source allows you to have multiple evaluation sets or have fine-grained control over the training schedule.

### Data Sources

In Levanter, a data source can either be a list of training and validation URLs pointing to
(possibly compressed) JSONL files, or a Huggingface Dataset. In either case,
we assume there is a single field, by default called `"text"`, that contains the text of the example.

See the [Training Data Guide](./guides/Training-Data-Guide.md) and the [Data Formats Reference](./reference/Data-Formats.md) for more details.

#### Data Format: JSONL

The canonical format for training data in Levanter is (compressed) [JSONL, or JSON Lines](https://jsonlines.org/).
Each line of the file is a JSON object, which is a dictionary of key-value pairs.
The only required key is `"text"`, which should map to a string of plain text.
Other keys are ignored, but you can use them to store metadata about your data.

Once you have done so, you can create the `data` section of your training configuration:

```yaml
    train_urls:
      - "gs://path/to/train_web_{1..32}.jsonl.gz"
      - "gs://path/to/train_web_crawl2.jsonl.gz"
    validation_urls:
      - "gs://path/to/valid_web.jsonl.gz"
```

Levanter uses [fsspec](https://filesystem-spec.readthedocs.io/en/latest/) to read data from files,
so it can transparently handle compressed files and files in cloud storage (like Google Cloud Storage or AWS S3).
Levanter uses [braceexpand](https://pypi.org/project/braceexpand/) to expand the `{1..32}` syntax.
You can also use more than one entry if you have urls that don't follow a naming scheme.

!!! tip

    Levanter's preprocessing pipeline works best if you split your data into at least 1 shard for every machine
    (i.e. every 8 TPUs or GPUs). This isn't a big deal, but it helps.

#### Data Format: Huggingface Datasets

If you have a Huggingface Dataset, such as [The Pile](https://huggingface.co/datasets/EleutherAI/pile), you can use it directly in Levanter. It must
have a `"text"` column, and it must be split into train and validation sets. To use it,
you can specify the dataset name in the `data` section of your training configuration:

```yaml
data:
    id: "EleutherAI/pile"
    # if needed:
    # name: "subset"
```

This will be passed to `datasets.load_dataset`. By default, we use `stream: true` to stream the data, which
is much more efficient, as it avoids downloading the dataset onto every machine. If a streaming dataset is sharded,
we will automatically process each shard in parallel.

### Single Data Source

If you have a single source of data, you can use the `data` section of your training configuration to specify it:

```yaml
data:
    train_urls:
      - "gs://path/to/train.{1..32}.jsonl.gz"
    validation_urls:
      - "gs://path/to/valid.{1..4}.jsonl.gz"
    cache_dir: "gs://path/to/cache"
    tokenizer: "gpt2"  # any HF tokenizer path, or GCS path to an HF tokenizer
```

### Mixture of Sources

If you have multiple sources of data (e.g., multiple domains, or distinct subsets of data), you can use the `data` section of your training configuration to specify them:

```yaml
data:
  configs:
    wikitext:
      id: dlwh/wikitext_103_detokenized
    web:
      train_urls:
        - "gs://path/to/train_web_{1..32}.jsonl.gz"
      validation_urls:
        - "gs://path/to/valid_web.jsonl.gz"
  train_weights:
    wikitext: 0.1
    web: 0.9
  cache_dir: "gs://path/to/cache"
  tokenizer: "gpt2"  # any HF tokenizer path, or GCS path to an HF tokenizer
```

`train_weights` is a dictionary mapping source names to weights. The weights need not sum to 1, but they should be positive.
The weights are normalized to sum to 1. You can include a weight of 0.0 to exclude a source from training,
in which case it will only be used for evaluation (if present).

Evaluation losses are broken down by source, so you can see how each source is performing. Not every source needs to have
validation data.

!!! tip

        If you only have one training source, but you want to use multiple evaluation sources, you can use the
        the mixture of sources mechanism with a single source. Just set the weight of the training source to 1.0
        and the weights of the evaluation sources to 0.0.

## Data Preprocessing

Levanter supports both online and offline preprocessing. Online preprocessing is done on-the-fly
during training. With online preprocessing, you don't need to think about preprocessing your data
except to make sure it's in the right format and where you'd like to store the cached preprocessing
results.

Our data loading pipeline will automatically break and concatenate documents into chunks equal
to the model's `seq_len` parameter. It will also automatically add special tokens to the
end of documents.

### Online Preprocessing

We have a sophisticated caching mechanism using [Ray](https://docs.ray.io/en/latest/)
that builds a cache of preprocessed data on the fly. Online caching happens transparently
in the background, using the mostly-idle CPU-cores of the machine(s) you are training on.

The cache that is built is fully reproducible, and can be used for future training runs.
Training will start as soon as the system has the data it needs.

### Offline Preprocessing

If you want, you can also preprocess your data offline, and then upload the preprocessed data to cloud storage.

Levanter has a script that basically runs the same online preprocessing code, but doesn't do any training.
You can run it like this:

```bash
python -m levanter.main.cache_dataset --config_path my_config.yaml
```

You can actually connect this to a Ray cluster, and use the cluster to do the preprocessing. This lets you
use any of Ray's autoscaling features to scale up the preprocessing job.

To do so:

```bash
python -m levanter.main.cache_dataset \
    --config_path my_config.yaml \
    --address <ray-cluster-address> \
    --start_workers false \
    --auto_start_cluster false
```

### Direct Cache Construction

We also support direct cache construction. This is useful if you want to use a custom cache format.
See our guide on [Direct Cache Construction](./guides/Direct-Cache-Construction.md) for more details.


## Configuration

Levanter uses [Draccus](https://github.com/dlwh/draccus) to configure training runs. It's a YAML-to-dataclass
library that also supports argument parsing via argparse. A detailed guide to configuring Levanter is available
in the [Configuration Guide](./reference/Configuration.md).

This section will cover the basics of configuring a training run.

### TL;DR

Here's a configuration for a 1.4B parameter model with reasonable values for everything:

```yaml
data:
    train_urls:
      - "gs://path/to/train.{1..32}.jsonl.gz" # TODO
    validation_urls:
      - "gs://path/to/valid.{1..4}.jsonl.gz" # TODO
    cache_dir: "gs://path/to/cache"  # TODO
    tokenizer: "gpt2"  # any HF tokenizer path, or GCS path to an HF tokenizer
model:
  type: gpt2
  hidden_dim: 1536
  num_heads: 24
  num_layers: 48
  seq_len: 1024
  gradient_checkpointing: true
  scale_attn_by_inverse_layer_idx: true
trainer:
  tracker:
    type: wandb
    project: "levanter" # TODO
    tags: ["gpt2"]

  mp: p=f32,c=bfloat16
  num_train_steps: 100000  # TODO
  train_batch_size: 512  # you may need to tune this or per_device_parallelism
  per_device_parallelism: -1
  per_device_eval_parallelism: 8

  max_eval_batches: null # set to a number to limit eval batches. useful if your eval set is enormous

  checkpointer:
    base_path: "gs://path/to/checkpoints"  # TODO
    save_interval: 15m
    keep:
      - every: 10000
optimizer:
  learning_rate: 1E-4
  weight_decay: 0.1
  min_lr_ratio: 0.1
# if you want:
hf_save_steps: 10000
hf_save_path: "gs://path/to/hf/checkpoints"  # TODO
hf_upload: null # set to an hf repo if you want to upload automatically. You need to have logged in to hf-cli
```

If you want a different model size or architecture, you can look at the config files in
[levanter/config](https://github.com/stanford-crfm/levanter/tree/main/config).

### Continued Pretraining

Levanter supports starting from an HF pretrained model. To do so, you should set your config like this:

```yaml
model:
  type: mpt
initialize_from_hf: "NousResearch/Llama-2-7b-hf"
use_hf_model_config: true
```

You should probably reduce the learning rate by a factor of 10 or so. TODO: figure out best practices here.

#### Llama 2

While some versions of Llama (and other gated models) require extra steps, `NousResearch/Llama-2-7b-hf` is freely accessible on the Hub, so no gating is required. You can initialize like so:

```yaml
model:
  type: llama
initialize_from_hf: "NousResearch/Llama-2-7b-hf"
use_hf_model_config: true
```

### Checkpointing

See also the [Checkpointing section of the Configuration Guide](./reference/Configuration.md#checkpointing-and-initialization).

Levanter supports checkpointing to both local and Google Cloud Storage, backed by [TensorStore](https://google.github.io/tensorstore/).
If you're using multiple machines, you should probably use cloud storage or NFS.

Levanter saves two kinds of checkpoints:

* **time-based checkpoints**: **temporary** checkpoints that are saved every `save_interval` minutes. The previous
time-based checkpoint is deleted when a new one is saved.
* **step-based checkpoints**: **permanent** checkpoints that are saved according to a policy. These checkpoints are never deleted.

At a minimum, you should set a `base_path` for your checkpoints. This can be a local path or a GCS path.

### Determining Batch Size

We don't have a mechanism for tuning batch size automatically, so you need to do this by hand. Memory usage is controlled
by `per_device_parallelism`, `train_batch_size`, gradient checkpointing, model size, and number of accelerators, and
`model_axis_size`. (For most models `model_axis_size` should be 1, so you can ignore it.)

`per_device_parallelism` is analogous to `per_device_batch_size` in other frameworks. It controls how many examples
are processed at once on a single accelerator, but the name is a bit more "correct" in the presence of tensor or pipeline
parallelism.  A `per_device_parallelism` of `-1` means "use as many as possible to not perform gradient accumulation."

Gradient accumulation is performed whenever `num_accelerators * per_device_parallelism / model_axis_size < train_batch_size`.
Gradient checkpointing is enabled by default and highly recommended.

So, to find your batch size, you should modify either `per_device_parallelism` or `train_batch_size` until
your job runs. Note that, due to FSDP, as you add more TPUs, you can increase the effective parallelism because
you will use less memory per accelerator to store parameters and optimizer states.

### Number of Training Steps

Levanter does not support epochs or number of tokens/examples, so if you want to train for a certain number of epochs or
tokens, you'll need to compute the number of steps yourself. You can use the following formula:

```
num_train_steps = num_epochs * num_tokens_per_epoch / train_batch_size / seq_len
```

Note however that epoch boundaries aren't really respected: our implementation of sharded data loading restarts
from the beginning as soon as any machine finishes its shards.

## Launching Training

### TPU

First, we assume you've gone through the setup steps in [the TPU guide](./Getting-Started-TPU-VM.md), at least through setting up your gcloud account.
We also strongly recommend setting up ssh keys and ssh-agent.

#### Upload Config To GCS

Once you have your config built, you should upload it to GCS. You could also `scp` it to all workers, but this is easier
and works with the TPU babysitting script.

```bash
gsutil cp my_config.yaml gs://path/to/config.yaml
```

#### Using the Babysitting Script with a Preemptible or TRC TPU VM

If you are using a preemptible TPU VM, or a TRC TPU VM, you should use the babysitting script to automatically restart
your VM if it gets preempted. A detailed guide to babysitting is available in the
[babysitting section of the TPU guide](./Getting-Started-TPU-VM.md#using-the-babysitting-script-with-a-preemptible-or-trc-tpu-vm).
Here is the upshot:

```bash
infra/babysit-tpu-vm my-tpu -z us-east1-d -t v3-128 -- \
    WANDB_API_KEY=... levanter/infra/run.sh python levanter/src/levanter/main/train_lm.py --config_path gs://path/to/config.yaml
```

#### Spin up and manual launch

You should probably use the automated setup script, as described in the [relevant section of the TPU guide](./Getting-Started-TPU-VM.md#automatic-setup).
Here's what that looks like:

```bash
bash infra/spin-up-tpu-vm.sh my-tpu -z us-east1-d -t v3-128
```

This will spin up a TPU VM instance and install Levanter on it. You can then run a command like so:


```
cat > .config <<EOF
env:
    WANDB_API_KEY:
    WANDB_ENTITY:
    WANDB_PROJECT:
    HF_TOKEN:
    TPU_STDERR_LOG_LEVEL: 2
    TPU_MIN_LOG_LEVEL: 2
    LIBTPU_INIT_ARGS: <extra args to libtpu>

docker_repository: levanter
zone: us-west4-a
tpu_type: "v5litepod-16"
vm_image: "tpu-ubuntu2204-base"
preemptible: true
autodelete: false
subnetwork: "default"

EOF
```

```bash
python infra/launch.py --tpu_name=my_tpu -- python src/levanter/main/train_lm.py --config_path gs://path/to/config.yaml"
```

## Monitoring

Levanter integrates with WandB for logging. You can view your run on the WandB website. Levanter will also log
to the console, and write logs to `logs/$RUN_ID.log` on each machine. Logs can be pretty verbose.

We recommend monitoring `train/loss` and `eval/loss` in log/log scale. You should be seeing roughly a linear decrease
in loss followed by a gradual flattening. You can also monitor `throughput`.

## Evaluation

Levanter will run evaluation every `trainer.steps_per_eval` steps.

You can also run evaluation manually by running the `levanter/main/eval_lm.py` script:

```bash
python -m levanter.main.eval_lm --config_path gs://path/to/config.yaml --checkpoint_path gs://path/to/checkpoint
```

You can also use this script to evaluate on other datasets by modifying the config.


## Huggingface Export

### Exporting during Training

You can export to HF during training using the `hf_save_steps` and `hf_save_path` options in your config. You can
also set `hf_upload` to an HF repo to automatically upload your model to HF. See the [config above](#tldr) for an example.

Typically, you will have saved checkpoints in a directory like `gs://path/to/checkpoints/hf/my_run/step_10000/`.
Hugging Face Transformers doesn't know how to read these. So, you'll need to copy the files to a local directory:

```bash
gsutil -m cp gs://path/to/checkpoints/hf/my_run/step_10000/* /tmp/my_exported_model
```

Then you can use the model as you would expect:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("/tmp/my_exported_model")
tokenizer = AutoTokenizer.from_pretrained("/tmp/my_exported_model")
```

### Exporting after Training

After training, you can run a separate script to export levanter checkpoints to Huggingface:

```bash
python -m levanter.main.export_lm_to_hf --config_path my_config.yaml --output_dir gs://path/to/output
```
