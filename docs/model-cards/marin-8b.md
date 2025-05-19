---
license: apache-2.0
datasets:
- allenai/dolmino-mix-1124
- allenai/olmo-mix-1124
- bigcode/starcoderdata
- EleutherAI/proof-pile-2
- hltcoe/megawika
- mlfoundations/dclm-baseline-1.0
- HuggingFaceTB/finemath
- marin-community/ar5iv-noproblem-markdown
- marin-community/ar5iv-warning-markdown
- marin-community/datashop-science-qa
- marin-community/stackexchange-markdown
- marin-community/wikipedia-markdown
# REMINDER: when the instruct model should add dependencies on the instruct datasets and the base model.
language:
- en
tags:
- text-generation
---

<img alt="Marin Logo" src="https://huggingface.co/datasets/marin-community/blog-images/resolve/main/marin-boat.jpg" width="96" style="margin-left:'auto' margin-right:'auto' display:'block'">


# Model Card for Marin 8B

This is the model card for the Marin 8B Base model. [The Marin Project](https://marin.community) is a collaborative effort to develop open-source foundation models.

## Datasets

### Datasets used in Marin 8B Base

Marin 8B Base was trained on a variety of datasets:

- [Nemotron-CC](https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html)
- [DCLM Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)
- [Starcoder Data](https://huggingface.co/datasets/bigcode/starcoderdata)
- [Proofpile 2](https://huggingface.co/datasets/EleutherAI/proof-pile-2)
- [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) 3+
- [Dolma](https://huggingface.co/datasets/allenai/dolma), including their versions of:
  - [MegaWika](https://huggingface.co/datasets/hltcoe/megawika)
  - [peS2o](https://huggingface.co/datasets/allenai/peS2o)
  - (And most of the rest of it)
- [Dolmino-Mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124), including their versions of:
    - [FLAN](https://arxiv.org/abs/2109.01652)
    - [CodeSearchNet](https://arxiv.org/abs/1909.09436) (with OWM Filter)
    - [GSM8K](https://arxiv.org/pdf/2110.14168v1)
    - [MetaMath](https://arxiv.org/abs/2309.12284)
    - [MathCoder2 Synthetic](https://arxiv.org/abs/2310.03731)


And some new datasets:

- [Marin Markdownified StackExchange](https://huggingface.co/datasets/marin-community/stackexchange-markdown)
- [Marin Markdownified Wikipedia](https://huggingface.co/datasets/marin-community/wikipedia-markdown)
- [Marin Markdownified Ar5iv (No Problem)](https://huggingface.co/datasets/marin-community/ar5iv-noproblem-markdown)
- [Marin Markdownified Ar5iv (Warnings)](https://huggingface.co/datasets/marin-community/ar5iv-warning-markdown)
- [Marin Datashop Science QA](https://huggingface.co/datasets/marin-community/datashop-science-qa)

The `Markdownified` datasets are licensed under the original licenses of the individual documents.
Please refer to [StackExchange](https://stackoverflow.com/help/licensing),
[Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Database_download),
and [arXiv](https://arxiv.org/help/license) for more information.

The Datashop Science QA dataset is licensed under [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

### Datasets used in Marin 8B Instruct

Marin 8B Instruct is currently an SFT-only model. It was trained on the following datasets:

- [TIGER-Lab/AceCode-89K](https://huggingface.co/datasets/TIGER-Lab/AceCode-89K)
- [bespokelabs/Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k)
- [cognitivecomputations/dolphin-r1](https://huggingface.co/datasets/cognitivecomputations/dolphin-r1) (includes both nonreasoning and reasoning subsets)
- [tuenguyen/dolphin_r1_reasoning](https://huggingface.co/datasets/tuenguyen/dolphin_r1_reasoning)
- [facebook/natural_reasoning](https://huggingface.co/datasets/facebook/natural_reasoning)
- [open-r1/OpenThoughts-114k-math](https://huggingface.co/datasets/open-r1/OpenThoughts-114k-math)
- [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
- [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)
- [PrimeIntellect/verifiable-math-problems](https://huggingface.co/datasets/PrimeIntellect/verifiable-math-problems)

It is quite likely that we will release improved versions of this model in the future.

## Checkpoints

We release a large number of checkpoints.

### Base Model Checkpoints

Main Page: [marin-community/marin-8b-base](https://huggingface.co/marin-community/marin-8b-base)

| Name/Link                                                                                      | Training Tokens |
|------------------------------------------------------------------------------------------------|-----------------|
| [`main`](https://huggingface.co/marin-community/marin-8b-base/tree/main)                       | 12.7T           |
| [`kestrel`](https://huggingface.co/marin-community/marin-8b-base/tree/kestrel)                 | 2.7T            |
| [`ocelot`](https://huggingface.co/marin-community/marin-8b-base/tree/ocelot)                   | 3.78T           |
| [`jellyfish`](https://huggingface.co/marin-community/marin-8b-base/tree/jellyfish)             | 4.78T           |
| [`phoenix`](https://huggingface.co/marin-community/marin-8b-base/tree/phoenix)                 | 11.1T           |
| [`starling`](https://huggingface.co/marin-community/marin-8b-base/tree/starling)               | 12.4T           |
| [`deeper-starling`](https://huggingface.co/marin-community/marin-8b-base/tree/deeper-starling) | 12.7T           |

`main` currently refers to `deeper-starling`.
This may change in the future, but we will maintain compatibility at the architecture and tokenizer level,
so the model will remain drop-in compatible with existing tooling.
If you require a specific checkpoint, please use the `revision` argument.

### Instruct Model Checkpoints

Main Page: [marin-community/marin-8b-instruct](https://huggingface.co/marin-community/marin-8b-instruct)

| Name                    | SFT Tokens | Link                                                                                                                     |
|-------------------------|------------|--------------------------------------------------------------------------------------------------------------------------|
| `main`                  | 5.3B       | [marin-community/marin-8b-instruct](https://huggingface.co/marin-community/marin-8b-instruct/tree/deeper-starling-05-15) |
| `deeper-starling-05-15` | 5.3B       | [marin-community/marin-8b-instruct](https://huggingface.co/marin-community/marin-8b-instruct/tree/deeper-starling-05-15) |

`main` currently refers to `deeper-starling-05-15`. This will (probably) change in the future, though we will maintain
architecture and tokenizer compatibility.
If you require a specific checkpoint, please use the `revision` argument.

## Installation

Marin 8B uses the [Llama architecture](https://arxiv.org/abs/2302.13971) and as such should
work out-of-the-box with the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library
and any other library that supports the Llama architecture.


We use a variant of the Llama 3 tokenizer: [stanford-crfm/marin-tokenizer](https://huggingface.co/stanford-crfm/marin-tokenizer/).

## Inference

You can use Marin with the standard Hugging Face Transformers library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
marin = AutoModelForCausalLM.from_pretrained("marin-community/marin-8b-base", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("marin-community/marin-8b-base")
message = ["The Marin wind is"]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
response = marin.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
"The Marin wind is blowing, which is a blessing for the residents and an opportunity for the city's ambitious plans of generating more and more clean and renewable energy from the local area."
```

We released a number of checkpoints of this model. To load a specific checkpoint, add the argument `revision`:

```bash
marin = AutoModelForCausalLM.from_pretrained("marin-community/marin-8b-base", revision="deeper-starling")
```

### Model Description

- **Developed by:** The Marin team at Stanford CRFM.
- **Model type:** a Transformer style autoregressive language model.
- **Knowledge Cutoff:** To the best of our knowledge, the base model has no data from later than July 2024.
- **Language(s) (NLP):** English
- **License:** The code and model are released under Apache 2.0.
- **Contact:** `dlwh at stanford.edu`

### Model Sources

- **Project Page:** https://marin.community
- **Repositories:**
    - Core repo (data and experiment management): https://github.com/marin-community/marin
    - Training code: https://github.com/stanford-crfm/levanter
- **Retrospective:** https://marin.readthedocs.io/en/latest/reports/marin-8b-retro.html
- **W&B Logs:** [Marin 8B](https://wandb.ai/stanford-mercury/marin/reports/Tootsie-8B---VmlldzoxMTY3MzU3OA)


## Evaluation


### Base Model Results

We ran a suite of standard benchmarks to compare our model with [Llama 3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B), and the open-source 7-8B models [Olmo 2 7B](https://huggingface.co/allenai/OLMo-2-1124-7B), and [MAP-Neo 7B](https://huggingface.co/m-a-p/neo_7b).
For all benchmarks, we used [LM Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness) with the default setup for each task. (These numbers may differ from reported results due to differences in setup. LM Eval Harness is usually somewhat stricter than other harnesses.)

| Model                             | Average  | AGI Eval LSAT-AR | ARC Challenge | ARC Easy | BBH      | BoolQ    | CommonSense QA | COPA     | GPQA     | GSM8K    | HellaSwag_1, 10 shot | HellaSwag, 0 shot | lambada_openai | MMLU Pro | MMLU_5shot | MMLU-0shot | OpenBookQA | PIQA     | WinoGrande | WSC      |
|-----------------------------------|----------|------------------|---------------|----------|----------|----------|----------------|----------|----------|----------|----------------------|-------------------|----------------|----------|------------|------------|------------|----------|------------|----------|
| Marin 8B Base   (Deeper Starling) | **66.6** | 20.9             | **63.1**      | **86.5** | **50.6** | **85.9** | 79.1           | **92.0** | 30.3     | 61.3     | **83.6**             | **82.3**          | **74.7**       | **36.5** | **67.6**   | **65.9**   | 44.2       | **84.4** | **74.5**   | 82.1     |
| Llama 3.1 Base                    | 65.3     | 20.4             | 58.9          | 85.8     | 46.4     | 84.2     | 75.2           | **92.0** | **32.3** | 56.8     | 81.9                 | 79.4              | **74.7**       | 33.3     | 66.4       | 65.5       | 45.8       | 82.9     | 74.4       | 83.5     |
| OLMo 2 Base                       | 64.9     | 17.4             | 60.7          | 85.0     | 44.4     | 85.5     | 75.4           | 89.0     | 26.8     | **67.6** | 81.7                 | 80.5              | 73.1           | 30.6     | 63.9       | 61.9       | **46.2**   | 82.5     | 74.3       | **86.1** |
| MAP NEO 7B                        | 59.5     | **23.0**         | 52.0          | 81.1     | 42.4     | 84.7     | **81.7**       | 82.0     | 27.8     | 48.0     | 73.3                 | 72.5              | 64.6           | 25.2     | 58.2       | 56.4       | 39.4       | 79.0     | 66.1       | 73.3     |


Marin 8B Base fares well on most of these tasks.


## Model Details

Please see [our technical retrospective](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro.html) for more details on the pretraining process.

### Architecture Details

- **Architecture:** Llama 3 8B
- **Hidden size:** 4096
- **Feedforward size:** 14336
- **Number of layers:** 32
- **Number of attention heads:** 32
- **Number of KV heads:** 8

### Tokenizer Details

Marin 8B uses a variant of the Llama 3 tokenizer: [stanford-crfm/marin-tokenizer](https://huggingface.co/stanford-crfm/marin-tokenizer/). It has the same vocabulary but bundles a chat template into the base tokenizer for convenience.

### Training Phases

#### Pre-training Phases

- *Kestrel (DCLM WSD-S Phase)*: DCLM+StarCoder+Proofpile2 using [WSD-S](https://arxiv.org/abs/2410.05192) (0->2.7T tokens)
- *Ocelot (DCLM WSD Phase)*: Increased batch size, using WSD. (2.7T->3.78T tokens)
- *Jellyfish (First Cooldown)*: Higher quality data (~Dolmino+Fine Math). (3.78T->4.78T tokens)
- *Phoenix (Reheated)*: Rapid rewarming + [Nemotron-CC](https://arxiv.org/abs/2412.02595) (plus [Starcoder](https://huggingface.co/datasets/bigcode/starcoderdata)). (4.78T->11.1T tokens)
- *Starling (Second Cooldown)*: Another cooldown. We followed a similar process to the first cooldown, but added a few new datasets. (11.1T->12.4 tokens)
- *Deeper Starling*: Somewhat more pretraining. (12.4->12.7T tokens)

All released pre-training checkpoints except Kestrel use an exponential moving average of the model weights.

#### SFT Phase

SFT was comparably simple, consisting of only one phase for 5.3B tokens.

## Bias, Risks, and Limitations

Like any base language model or fine-tuned model without safety filtering, these models can easily be prompted by users to generate harmful and sensitive content. Such content may also be produced unintentionally, especially in cases involving bias, so we recommend that users consider the risks when applying this technology. Additionally, many statements from Marin or any LLM are often inaccurate, so responses should be verified.

Marin 8B has not undergone any safety tuning or evaluation. We strongly recommend that users use this model with caution and consider the risks when applying this technology.
In particular, this model is not intended for fully autonomous use.

## Model Card Contact
For errors in this model card, please open an issue in this repository. For technical inquiries, please contact `dlwh at stanford.edu`.

## Acknowledgements

The compute for this model was generously provided by Google's [TPU Research Cloud](https://sites.research.google/trc/about/).
