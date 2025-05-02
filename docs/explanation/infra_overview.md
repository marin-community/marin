# Infrastructure Overview

This document provides an overview of the infrastructure components and their roles within the Marin project.
It serves as a reference for understanding the system's architecture and how different components interact with each other.

Marin relies on a large number of infrastructure components to support its various tasks and operations. We discuss the key components and their roles below.

## Autoscaling Cluster for Marin Data Processing

Marin at its core is basically a distributed Make written using [Ray](https://docs.ray.io/en/latest/index.html) with
Makefiles written in Python. The Executor framework is responsible for managing the execution of [experiments](./experiments.md) and steps.
See the [Executor framework](./executor.md) for more details on how it works conceptually.

The Executor framework runs inside a Ray cluster, and relies on Ray for dependency management, autoscaling, and distributed execution.

### Dependencies and Frameworks

* [**Google Cloud Platform (GCP)**](https://cloud.google.com/): Marin uses GCP for its cloud computing needs. It provides a wide range of services, including compute, storage, and networking.
We are striving to make Marin cloud-agnostic, but for now, GCP is the only cloud we use. We have limited testing on local GPUs. We primarily use TPUs for training.
* [**Ray**](https://docs.ray.io/en/latest/index.html): Ray is a distributed computing framework used for parallelizing tasks and managing distributed resources. We mainly
use Ray for dependency management, data processing, orchestrating training jobs, and autoscaling clusters.
* [**Docker**](https://docs.docker.com/get-started/overview/): Docker is a containerization platform used for packaging and deploying applications. We use Docker to package our code and dependencies for easy deployment with Ray.
 
# Processing Steps

## Data

### Downloading

TODO

### Crawling

XXX

### Filtering

TODO


## Training and Tokenization

### Tokenization

Tokenization is initiated via [experiments.defaults.default_tokenize][], which you can read more about here XXX.
What follows is a high-level overview of how the tokenization system fits into Marin as a whole.

Given processed documents in our [extended Dolma format](./explanations/data_formats.md), tokenization
converts the documents into a format that can be used for training. For text pre-training, documents are tokenized
into a sequence of tokens, which are then used to train the model. For supervised fine-tuning, conversations are tokenized
into a sequence of tokens and a mask indicating which tokens the model is responsible for predicting.

Tokenization is handled by [Levanter's tokenization infrastructure](https://levanter.readthedocs.io/en/latest/dev/cache-construction.html).
The basic idea is that we construct a cache of tokenized documents. While Levanter supports background tokenization, we
typically tokenize offline in Marin. A cache is a directory of [TensorStore](https://google.github.io/tensorstore/) arrays
using [Zarray 3](https://zarr.readthedocs.io/en/stable/spec/v3.html) format. The cache is a column store, where each column
is a different field of the document. Column stores make concat-and-split and sequence packing easy.
Tensorstore with Zarray gives us random access and compression.


### Training

The main entry point is [experiments.defaults.default_train][], which is likewise handled by [Levanter](https://levanter.readthedocs.io/en/latest/index.html).
Training takes a tokenized dataset or a [mixture of datasets]() XXX and runs training given a config.

On TPU, Marin uses Levanter's [Ray-based training orchestration](https://levanter.readthedocs.io/en/latest/dev/ray-job-manager.html).
This component handles requesting TPUs (delegating to Ray), launching jobs, and babysitting jobs, including retrying
jobs that fail due to preemption. It also handles multislice training orchestration. (Actual training communication
is handled by JAX/libtpu/XLA.)

### Supervised Fine-Tuning

Supervised fine-tuning (SFT) is handled by [experiments.defaults.default_sft][]. 
SFT is just training with a different config.
We still use Levanter, but we typically use chat datasets like [Tulu v3](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture).


### Training Frameworks and Libraries

* [**JAX**](https://github.com/google/jax) is the primary framework for training models. We use JAX for its high performance, reproducibility, and easy distributed training.
* [**Levanter**](https://github.com/stanford-crfm/levanter): Levanter is a library for training large language models and other foundation models. 
* [**Haliax**](https://github.com/stanford-crfm/haliax): Haliax is a named tensor library used for building neural networks. 
It is used for training models on TPUs and GPUs.
* [**Equinox**](https://github.com/patrick-kidger/equinox) is a neural network library for JAX.
* [**Tensorstore**](https://google.github.io/tensorstore/): TensorStore is a library for storing and accessing large-scale tensor data. It is used for storing and accessing the preprocessed data used for training models.
Levanter uses TensorStore for sharded checkpointing and to store the preprocessed data used for training models.
* [**HuggingFace Tokenizers**](https://huggingface.co/docs/tokenizers/) is used for tokenizing text data.
* [**WandB**](https://wandb.ai/): WandB is a machine learning experiment tracking platform used for monitoring and visualizing the performance of our models during training. We also make use of WandB for making reports.
* [**HuggingFace Hub**](https://huggingface.co/): The HuggingFace Hub is a platform for sharing and discovering machine learning models. We use it to share our models and download datasets.



## Inference

TODO XXX

### Inference Frameworks and Libraries

* [**VLLM**](https://github.com/vllm-project/vllm) is a library for running large language model inference on GPUs and TPUs.

## Evaluation

We do evaluation in a few different ways in Marin, broadly grouped into three categories:

- Perplexity evals during training
- Multiple choice (MCQA) evals during or after training
- Generation evals after training
 
### Training-time Evaluation (Perplexity and MCQA)

During training, Levanter will compute validation losses on held-out data (as separate data sources, typically [Paloma](https://huggingface.co/datasets/allenai/paloma))
Levanter can also use EleutherAI's LM Evaluation Harness to evaluate the model on multiple choice questions and other
tasks that only need log-probabilities and not inference. See the documentation on XXX training for how
these steps work.


### Post-training Evaluation (MCQA and Generation)

After training, we have a few more options. Of course, we can still of course use Levanter to evaluate the model on MCQA tasks,
but we can also use VLLM to evaluate the model on generation tasks. (Unfortunately, as of this writing, VLLM still
does not support log-probabilities on TPU, so we don't use it for MCQA tasks.)

Using VLLM, we have a few options:

* LM Evaluation Harness: We can use the LM Evaluation Harness to evaluate the model on other tasks that require generation.
* Alpaca Eval: We can run Alpaca Eval to evaluate how well it does with chat discussions.
* HELM: We can use HELM to evaluate the model on other tasks that require generation.

### Evaluation Frameworks and Libraries

* [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) 
* [HELM](https://github.com/stanford-crfm/helm)
* [Alpaca Eval](https://github.com/tatsu-lab/alpaca_eval)

During training, we mainly use the LM Evaluation Harness for evaluation because Levanter lacks generation support and
because HELM is more geared towards larger models than most of our workloads.
