---
license: apache-2.0
datasets:
- allenai/dolmino-mix-1124
- allenai/olmo-mix-1124
- bigcode/starcoderdata
- EleutherAI/proof-pile-2
- hltcoe/megawika
- mlfoundations/dclm-baseline-1.0
language:
- en
---
## Model Details

<img alt="Marin Logo" src="https://huggingface.co/datasets/marin-community/blog-images/resolve/main/marin-boat.png" width="242px" style="margin-left:'auto' margin-right:'auto' display:'block'">


# Model Card for Marin 8B

This is the model card for the Marin 8B Base model. The Marin Project is a collaborative effort to develop open-source foundation models.


## Datasets

### Datasets used in Marin 8B Base

Marin 8B Base was trained on a variety of datasets:

- [Nemotron-CC](https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html)
- [DCLM Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)
- [Starcoder Data](https://huggingface.co/datasets/bigcode/starcoderdata)
- [Proofpile 2](https://huggingface.co/datasets/EleutherAI/proof-pile-2)
- [FineMath 3+](https://huggingface.co/datasets/EleutherAI/proof-pile-2)
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

- [Marin Markdownified StackExchange](XXX)
- [Marin Markdownified Wikipedia](XXX)
- [Marin Markdownified Ar5iv](XXX)
- [Marin Datashop Science QA](XXX)

A full report is available on [our ReadTheDocs site](https://marin.readthedocs.org/en/latest/reports/marin-8b-retro.html).

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


## Checkpoints

We release a number of training checkpoints. Other checkpoints may be made available on request.


### Base Model Checkpoints

| Name | Training Tokens | Link |
|------|--------|---------|-------------|
| `deeper-starling` | 13.7T | [marin-community/marin-8b-base](https://huggingface.co/marin-community/marin-8b-base/tree/deeper-starling) |

### Instruct Model Checkpoints

XXX TODO


## Installation

Marin 8B uses the [Llama architecture](https://arxiv.org/abs/2302.13971) and as such should
work out-of-the-box with the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library
and any other library that supports the Llama architecture.

We use a variant of the Llama 3 tokenizer: [stanford-crfm/marin-tokenizer](https://huggingface.co/stanford-crfm/marin-tokenizer/).

## Inference

You can use Marin with the standard HuggingFace Transformers library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
marin = AutoModelForCausalLM.from_pretrained("marin-community/marin-8b-base")
tokenizer = AutoTokenizer.from_pretrained("marin-community/marin-8b-base")
message = ["The Marin wind is "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
response = marin.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
>> 'Language modeling is  a key component of any text-based application, but its effectiveness...'
```

We released a number of checkpoints of this model. To load a specific checkpoint, simply add the argument `revision`:

XXX

```bash
marin = AutoModelForCausalLM.from_pretrained("marin-community/marin-8b-base", revision="deeper-starling")
```

### Model Description

- **Developed by:** Stanford CRFM and the Marin Community
- **Model type:** a Transformer style autoregressive language model.
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

XXX TODO

## Model Details

Please see [our technical retrospective](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro.html) for more details on the pretraining process.

### Architecture Details

- **Architecture:** Llama 3 8B
- **Hidden size:** 4096
- **Feedforward size:** 14336
- **Number of layers:** 32
- **Number of attention heads:** 32
- **Number of KV heads:** 8

### Training Phases

#### Pre-training Phases

- *Kestrel (DCLM WSD-S Phase)*: DCLM+StarCoder+Proofpile2 using [WSD-S](https://arxiv.org/abs/2410.05192) (0->2.7T tokens)
- *Ocelot (DCLM WSD Phase)*: Increased batch size, using WSD. (2.7T->3.78T tokens)
- *Jellyfish (First Cooldown)*: Higher quality data (~Dolmino+Fine Math). (3.78T->4.78T tokens)
- *Phoenix (Reheated)*: Rapid rewarming + [Nemotron-CC](https://arxiv.org/abs/2412.02595) (plus [Starcoder](https://huggingface.co/datasets/bigcode/starcoderdata)). (4.78T->11.1T tokens)
- *Starling (Second Cooldown)*: Another cooldown. We followed a similar process to the first cooldown, but added a few new datasets. (11.1T->12.75T tokens)

All released pre-training checkpoints except Kestrel use an exponential moving average of the model weights.

####  SFT Phase

SFT was comparably simple, consisting of only one phase for 5.3B tokens.

## Bias, Risks, and Limitations
Like any base language model or fine-tuned model without safety filtering, these models can easily be prompted by users to generate harmful and sensitive content. Such content may also be produced unintentionally, especially in cases involving bias, so we recommend that users consider the risks when applying this technology. Additionally, many statements from Marin or any LLM are often inaccurate, so facts should be verified.


## Citation

XXX

## Model Card Contact
For errors in this model card, please open an issue in this repository. For technical inquiries, please contact `dlwh at stanford.edu`.

## Acknowledgements

XXX

(We based this model card on Olmo 2's.)
