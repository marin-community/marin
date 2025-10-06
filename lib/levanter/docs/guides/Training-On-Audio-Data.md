# Training on Audio Data

This guide will provide a detailed walkthrough of training a system for Automatic Speech Recognition (i.e. transcribing speech to text) from scratch. ASR systems, such as [Whisper](https://arxiv.org/abs/2212.04356), are a major focus for speech processing research since the same ASR system can be paired with many text-only NLP systems to provide broad capabilities for spoken language understanding. However, it is worth noting that the tooling and scripts referenced here are applicable to any task which expects audio as input and text as output.

For the most part, training Audio models only differs from text-only LLMs in the data formatting stage! For stages that overlap, through common code this guide will point to the more general [Training on Your Own Data](../Training-On-Your-Data.md) guide.


As in that guide, here are the basic steps:

- [ ] [Configure your environment/cloud](#environment-setup)
- [ ] [Prepare your data and upload to cloud](#data-preparation)
- [ ] [Configure your training run](#configuration)
- [ ] [Launch training](#launching-training)
- [ ] [Using Your Model](#using-your-model)


## Environment Setup
The environment setup for training Audio models should require minimal extra work from setting up text-only language model training with Levanter! If this is your first time setting up Levanter, we suggest going through the text focused [Getting Started Training Guide](../Getting-Started-Training.md)

### Audio Specific Dependencies

Once you have your baseline setup, you will need a few Audio focused dependencies which are primarily to simplify interacting with and processing the wide variety of formats Audio data is stored in. In Levanter, we use two tightly linked dependencies [Librosa](https://librosa.org/), which handles common Audio pre-processing functionality, and [Soundfile](https://github.com/bastibe/python-soundfile), which abstracts away the details of different Audio filetypes and codecs. These are not included in Levanter by default, so you'll need to install them before you proceed.

```sh
pip install librosa soundfile
```

## Data Preparation

Unlike text-only LM's, large-scale Audio models are usually multi-modal and both feature extraction from the Audio domain *and* Language Modeling capabilities in the text domain. In Levanter, we currently only support cases where the *input* modality is audio and the *output* modality is text, which is represented through the `AudioIODatasetConfig` class.

### Data Sources

As always in Levanter, a data source can either be a list of training and validation URLs pointing to (possibly compressed) JSONL files, or a Huggingface Dataset. For either of these formats, there must be at least two fields present `text_key` and `audio_key`, which default to `"text"` and `"audio"`. The data stored under `text_key` should contain the text which is expected as output for the example, in this case the transcription.

The data stored under `audio_key`, however, can take a variety of formats discussed below to allow flexibility and avoid the need to pre-process existing datasets into new formats.


#### Data Format: Huggingface Datasets

If you have a Huggingface Dataset, such as [Librispeech](https://huggingface.co/datasets/EleutherAI/pile), you can use it directly in Levanter. To use it,
you can specify the dataset name in the `data` section of your training configuration, along with the training split name, the validation split name, the name of your text column, and the name of your audio collumn:

```yaml
data:
  cache_dir: "gs://diva-flash/processed/mixture"
  # The Whisper Tokenizer is way too large for Librispeech
  tokenizer: "facebook/wav2vec2-base-960h"
  configs:
    librispeech:
      id: WillHeld/librispeech_parquet
      cache_dir: "gs://diva-flash/processed/librispeech"
      train_split: "train.360"
      validation_split: "validation"
  train_weights:
    librispeech: 1.0
```

Levanter directly supports the HuggingFace [Audio](https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/main_classes#datasets.Audio) class. Underlying this class is a simple dictionary, which fits into one of the following 3 modes. The first mode is completely pre-processed audio which provides a time-domain `array` of audio data along with a pre-defined `sampling_rate`. The second mode is data which has been loaded into memory as a sequence of `bytes`, but has not been decoded to raw audio data. Finally, if *only* the `path` of the dictionary is defined this points to where the audio file for that example is stored. Levanter will transparently handle all of these modes and process them uniformly to the `array` and `sampling_rate` which is required for downstream modeling.

#### Data Format: JSONL

The canonical format for training data in Levanter is (compressed) JSONL, or JSON Lines.
Each line of the file is a JSON object, which is a dictionary of key-value pairs.
Unlike for pure-text LMs, this tutorial entries for both `text_key` and `audio_key`.

When loading from a JSONL files, the same HuggingFace Audio format is supported or you can provide a string representing where the audio is stored.

Once you have done so, you can create the `data` section of your training configuration:

```yaml
    train_urls:
      - "gs://path/to/train_web_{1..32}.jsonl.gz"
      - "gs://path/to/train_web_crawl2.jsonl.gz"
    validation_urls:
      - "gs://path/to/valid_web.jsonl.gz"
```

Levanter uses [fsspec](https://filesystem-spec.readthedocs.io/en/latest/) to read data from files,
so it can transparently handle compressed files and files in cloud storage (like Google Cloud Storage or AWS S3). This applies not only to the files listed in the config, but also to all the paths stored in `audio_key`. This means that you do not need to pre-download all the audio files to your local storage and can provide URLs directly to content hosts if there are concerns about hosting audio directly.


### Data Preprocessing
Using Levanter's [Ray](https://docs.ray.io/en/latest/) based pre-processing and caching, you can apply further tokenization and feature extraction in the background while your model is training.

By default, you can define both the `tokenizer` and `preprocessor` from HuggingFace. By default, if no tokenizer is provided Levanter will fall back to the one defined by the `preprocessor`. Regardless of the tokenization, the Pre-Processor will always be used to convert the time-domain audio data into the expected input for your model, such as Log-Mel-Spectrograms for Whisper.

Below is an example which defines the preprocessor using the Whisper pre-processor from OpenAI, but switched the tokenizer to a simple character-level tokenizer used in prior-ASR work.
```yaml
    tokenizer: "facebook/wav2vec2-base-960h"
    preprocessor: "openai/whisper-tiny"
```

## Configuration

Levanter uses [Draccus](https://github.com/dlwh/draccus) to configure training runs. It's a YAML-to-dataclass
library that also supports argument parsing via argparse. A detailed guide to configuring Levanter is available
in the [Configuration Guide](../reference/Configuration.md).

This section will cover the basics of configuring a training run.

### TL;DR

Here's a configuration for a Whisper Tiny model with reasonable values for everything:

```yaml
data:
  cache_dir: "gs://diva-flash/processed/mixture"
  # The Whisper Tokenizer is way too large for Librispeech
  tokenizer: "facebook/wav2vec2-base-960h"
  configs:
    librispeech:
      id: WillHeld/librispeech_parquet
      cache_dir: "gs://diva-flash/processed/librispeech"
      train_split: "train.360"
      validation_split: "validation"
  train_weights:
    librispeech: 1.0
model:
  type: whisper
  vocab_size: 32
trainer:
  tracker:
    - type: wandb
      project: "levanter"
      tags: [ "librispeech", "whisper"]

  mp: p=f32,c=bf16
  model_axis_size: 1
  per_device_parallelism: -1

  train_batch_size: 128
  num_train_steps: 16000
optimizer:
  learning_rate: 3E-3
  weight_decay: 0.1
  warmup: 0.01
```

Right now, Levanter only supports the Encoder-Decoder architecture used in Whisper due to it's popularity, but feel free to contribute or request others!

### Continued Pretraining

Levanter supports starting from existing Whipser pretrained models on HuggingFace. To do so, you should set your config like this:

```yaml
model:
  type: whisper
initialize_from_hf: "openai/whisper-tiny"
use_hf_model_config: true
```

### Checkpointing
Levanter will automatically checkpoint your training run. For more details, see the [Checkpointing section of the Configuration Guide](../reference/Configuration.md#checkpointing-and-initialization).

### Hyperparameter Tuning
The same set of hyperparameters and functionalities that are supported for text-only LLMs, such as [gradient accumulation](../Training-On-Your-Data.md#determining-batch-size), are supported for Audio models! For in-depth walkthrough of this configuration, please see the [Trainer section of the Configuration Guide](../reference/Configuration.md#trainer-and-trainerconfig).

## Launching Training

### TPU

First, we assume you've gone through the setup steps in [the TPU guide](../Getting-Started-TPU-VM.md), at least through setting up your gcloud account.
We also strongly recommend setting up ssh keys and ssh-agent.

On TPU, you will also need to ensure that all hosts installed `librosa` and `soundfile`: `gcloud compute tpus tpu-vm ssh $TPU_NAME   --zone $ZONE --worker=all --command "source venv310/bin/activate; pip install librosa soundfile`

#### Upload Config To GCS

Once you have your config built, you should upload it to GCS. You could also `scp` it to all workers, but this is easier
and works with the TPU babysitting script.

```bash
gsutil cp my_config.yaml gs://path/to/config.yaml
```

#### Using the Babysitting Script with a Preemptible or TRC TPU VM

If you are using a preemptible TPU VM, or a TRC TPU VM, you should use the babysitting script to automatically restart
your VM if it gets preempted. A detailed guide to babysitting is available in the
[babysitting section of the TPU guide](../Getting-Started-TPU-VM.md#using-the-babysitting-script-with-a-preemptible-or-trc-tpu-vm).
Here is the upshot:

```bash
infra/babysit-tpu-vm my-tpu -z us-east1-d -t v3-128 -- \
    WANDB_API_KEY=... levanter/infra/run.sh python levanter/src/levanter/main/train_asr.py --config_path gs://path/to/config.yaml
```

#### Spin up and manual launch

You can start up a TPU VM and launch your instance with `launch.py`. To simplify your command for multiple launches, you can put common parameters into `.config` in your `levanter` directory:

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

```bash

python infra/launch.py --tpu_name=my_tpu -- python levanter/src/levanter/main/train_asr.py --config_path gs://path/to/config.yaml"
```

### GPU
After following the [GPU Setup Guide](../Getting-Started-GPU.md) on your node, you can kick of GPU training directly using the `train_asr.py` script.

```bash
python levanter/src/levanter/main/train_asr.py --config_path gs://path/to/config.yaml
```

## Using Your Model

### Huggingface Export

#### Exporting during Training

You can export to HF during training using the `hf_save_steps` and `hf_save_path` options in your config. You can
also set `hf_upload` to an HF repo to automatically upload your model to HF. See the [config above](#tldr) for an example.

Typically, you will have saved checkpoints in a directory like `gs://path/to/checkpoints/hf/my_run/step_10000/`.
Hugging Face Transformers doesn't know how to read these. So, you'll need to copy the files to a local directory:

```bash
gsutil -m cp gs://path/to/checkpoints/hf/my_run/step_10000/* /tmp/my_exported_model
```

Then you can use the model as you would expect:

```python
from transformers import AutoFeatureExtractor, AutoTokenizer, WhisperForConditionalGeneration
fe = AutoFeatureExtractor.from_pretrained("WillHeld/levanter-whisper-tiny-librispeech")
tokenizer = AutoTokenizer.from_pretrained("WillHeld/levanter-whisper-tiny-librispeech")
model = WhisperForConditionalGeneration.from_pretrained("WillHeld/levanter-whisper-tiny-librispeech")
```

#### Exporting after Training

After training, you can run a separate script to export levanter checkpoints to Huggingface:

```bash
python -m levanter.main.export_lm_to_hf --config_path my_config.yaml --output_dir gs://path/to/output
```

### HuggingFace Inference

Once your model has been exported to HuggingFace, it can be uploaded to the hub and loaded for inference just like any other Whisper model trained directly with HuggingFace. For an example of a Whisper Tiny model trained with the [config above](#tldr), see [WillHeld/levanter-whisper-tiny-librispeech](https://huggingface.co/WillHeld/levanter-whisper-tiny-librispeech).

You can run inference with this model for free using this demonstration [Colab](https://colab.research.google.com/drive/1vrLjZL_ysaT21B0D5g0716v3X80W-unZ#scrollTo=QfwpnJ-WckgK). As you see, the model provides reasonable results for a 17 Million parameter model which trained in just under 8 hours on an A100.
