---
license: apache-2.0
datasets:
- allenai/dolmino-mix-1124
- allenai/olmo-mix-1124
language:
- en
---
## Model Details

<img alt="Marin Logo" src="https://huggingface.co/datasets/marin-community/blog-images/resolve/main/marin-boat.png" width="242px" style="margin-left:'auto' margin-right:'auto' display:'block'">


# Model Card for Marin 8B Base

This is the model card for the Marin 8B Base model. The Marin Project is a collaborative effort to develop open-source foundation models.


Marin 8b was trained on a variety of datasets, including 

* [Nemotron-CC]()
* [Dolmino-Mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124) and [OLMo-Mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124) datasets. The model is designed to be a general-purpose language model that can be fine-tuned for various tasks.



| Size | Training Tokens | Layers | Hidden Size | Attention Heads | Context Length |
|------|--------|---------|-------------|-----------------|----------------|
| [OLMo 2-7B](https://huggingface.co/allenai/OLMo-2-1124-7B) | 4 Trillion   | 32     | 4096        | 32              |  4096  |
| [OLMo 2-13B](https://huggingface.co/allenai/OLMo-2-1124-13B) | 5 Trillion   | 40     | 5120        | 40              |  4096  |

The core models released in this batch include the following:

| **Stage**           | **OLMo 2 7B**                                                                                          | **OLMo 2 13B**                                                                                         |
|----------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| **Base Model**       | [allenai/OLMo-2-1124-7B](https://huggingface.co/allenai/OLMo-2-1124-7B)                                | [allenai/OLMo-2-1124-13B](https://huggingface.co/allenai/OLMo-2-1124-13B)                             |
| **SFT**              | [allenai/OLMo-2-1124-7B-SFT](https://huggingface.co/allenai/OLMo-2-1124-7B-SFT)                | [allenai/OLMo-2-1124-13B-SFT](https://huggingface.co/allenai/OLMo-2-1124-13B-SFT)              |
| **DPO**              | [allenai/OLMo-2-1124-7B-DPO](https://huggingface.co/allenai/OLMo-2-1124-7B-DPO)                | [allenai/OLMo-2-1124-13B-DPO](https://huggingface.co/allenai/OLMo-2-1124-13B-DPO)              |
| **Final Models (RLVR)** | [allenai/OLMo-2-1124-7B-Instruct](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct)                        | [allenai/OLMo-2-1124-13B-Instruct](https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct)                      |
| **Reward Model (RM)**| [allenai/OLMo-2-1124-7B-RM](https://huggingface.co/allenai/OLMo-2-1124-7B-RM)                                                     | (Same as 7B)                                                     |

## Installation

OLMo 2 will be supported in the next version of Transformers, and you need to install it from the main branch using:
```bash
pip install --upgrade git+https://github.com/huggingface/transformers.git
```

## Inference

You can use OLMo with the standard HuggingFace transformers library:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")
message = ["Language modeling is "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# optional verifying cuda
# inputs = {k: v.to('cuda') for k,v in inputs.items()}
# olmo = olmo.to('cuda')
response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
>> 'Language modeling is  a key component of any text-based application, but its effectiveness...'
```

For faster performance, you can quantize the model using the following method:
```python
AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B", 
    torch_dtype=torch.float16, 
    load_in_8bit=True)  # Requires bitsandbytes
```
The quantized model is more sensitive to data types and CUDA operations. To avoid potential issues, it's recommended to pass the inputs directly to CUDA using:
```python
inputs.input_ids.to('cuda')
```

We have released checkpoints for these models. For pretraining, the naming convention is `stepXXX-tokensYYYB`. For checkpoints with ingredients of the soup, the naming convention is `stage2-ingredientN-stepXXX-tokensYYYB`


To load a specific model revision with HuggingFace, simply add the argument `revision`:
```bash
olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B", revision="step1000-tokens5B")
```

Or, you can access all the revisions for the models via the following code snippet:
```python
from huggingface_hub import list_repo_refs
out = list_repo_refs("allenai/OLMo-2-1124-7B")
branches = [b.name for b in out.branches]
```

### Fine-tuning
Model fine-tuning can be done from the final checkpoint (the `main` revision of this model) or many intermediate checkpoints. Two recipes for tuning are available.
1. Fine-tune with the OLMo repository:
```bash
torchrun --nproc_per_node=8 scripts/train.py {path_to_train_config} \
    --data.paths=[{path_to_data}/input_ids.npy] \
    --data.label_mask_paths=[{path_to_data}/label_mask.npy] \
    --load_path={path_to_checkpoint} \
    --reset_trainer_state
```
For more documentation, see the [GitHub readme](https://github.com/allenai/OLMo?tab=readme-ov-file#fine-tuning).

2. Further fine-tuning support is being developing in AI2's Open Instruct repository. Details are [here](https://github.com/allenai/open-instruct).

### Model Description

- **Developed by:** Allen Institute for AI (Ai2)
- **Model type:** a Transformer style autoregressive language model.
- **Language(s) (NLP):** English
- **License:** The code and model are released under Apache 2.0.
- **Contact:** Technical inquiries: `olmo@allenai.org`. Press: `press@allenai.org`
- **Date cutoff:** Dec. 2023.


### Model Sources

- **Project Page:** https://allenai.org/olmo
- **Repositories:** 
    - Core repo (training, inference, fine-tuning etc.): https://github.com/allenai/OLMo
    - Evaluation code: https://github.com/allenai/OLMo-Eval
    - Further fine-tuning code: https://github.com/allenai/open-instruct
- **Paper:** https://arxiv.org/abs/2501.00656
<!-- - **Technical blog post:** https://blog.allenai.org/olmo-1-7-7b-a-24-point-improvement-on-mmlu-92b43f7d269d  -->
<!-- - **W&B Logs:** [pretraining](https://wandb.ai/ai2-llm/OLMo-7B/groups/OLMo-1.7-7B), [annealing](https://wandb.ai/ai2-llm/OLMo-7B/groups/OLMo-1.7-7B-anneal) -->


## Evaluation
Core model results for OLMo 2 7B and 13B models are found below.

| Model | Train FLOPs | Average | ARC/C | HSwag | WinoG | MMLU | DROP | NQ | AGIEval | GSM8k | MMLUPro | TriviaQA |
|-------------------|------------|---------|--------|--------|--------|-------|-------|-----|----------|--------|-----------|-----------|
| *Open weights models:* |
| Llama-2-13B | 1.6·10²³ | 54.1 | 67.3 | 83.9 | 74.9 | 55.7 | 45.6 | 38.4 | 41.5 | 28.1 | 23.9 | 81.3 |
| Mistral-7B-v0.3 | n/a | 58.8 | 78.3 | 83.1 | 77.7 | 63.5 | 51.8 | 37.2 | 47.3 | 40.1 | 30 | 79.3 |
| Llama-3.1-8B | 7.2·10²³ | 61.8 | 79.5 | 81.6 | 76.6 | 66.9 | 56.4 | 33.9 | 51.3 | 56.5 | 34.7 | 80.3 |
| Mistral-Nemo-12B | n/a | 66.9 | 85.2 | 85.6 | 81.5 | 69.5 | 69.2 | 39.7 | 54.7 | 62.1 | 36.7 | 84.6 |
| Qwen-2.5-7B | 8.2·10²³ | 67.4 | 89.5 | 89.7 | 74.2 | 74.4 | 55.8 | 29.9 | 63.7 | 81.5 | 45.8 | 69.4 |
| Gemma-2-9B | 4.4·10²³ | 67.8 | 89.5 | 87.3 | 78.8 | 70.6 | 63 | 38 | 57.3 | 70.1 | 42 | 81.8 |
| Qwen-2.5-14B | 16.0·10²³ | 72.2 | 94 | 94 | 80 | 79.3 | 51.5 | 37.3 | 71 | 83.4 | 52.8 | 79.1 |
| *Partially open models:* |
| StableLM-2-12B | 2.9·10²³ | 62.2 | 81.9 | 84.5 | 77.7 | 62.4 | 55.5 | 37.6 | 50.9 | 62 | 29.3 | 79.9 |
| Zamba-2-7B | n/c | 65.2 | 92.2 | 89.4 | 79.6 | 68.5 | 51.7 | 36.5 | 55.5 | 67.2 | 32.8 | 78.8 |
| *Fully open models:* |
| Amber-7B | 0.5·10²³ | 35.2 | 44.9 | 74.5 | 65.5 | 24.7 | 26.1 | 18.7 | 21.8 | 4.8 | 11.7 | 59.3 |
| OLMo-7B | 1.0·10²³ | 38.3 | 46.4 | 78.1 | 68.5 | 28.3 | 27.3 | 24.8 | 23.7 | 9.2 | 12.1 | 64.1 |
| MAP-Neo-7B | 2.1·10²³ | 49.6 | 78.4 | 72.8 | 69.2 | 58 | 39.4 | 28.9 | 45.8 | 12.5 | 25.9 | 65.1 |
| OLMo-0424-7B | 0.9·10²³ | 50.7 | 66.9 | 80.1 | 73.6 | 54.3 | 50 | 29.6 | 43.9 | 27.7 | 22.1 | 58.8 |
| DCLM-7B | 1.0·10²³ | 56.9 | 79.8 | 82.3 | 77.3 | 64.4 | 39.3 | 28.8 | 47.5 | 46.1 | 31.3 | 72.1 |
| **OLMo-2-1124-7B** | 1.8·10²³ | 62.9 | 79.8 | 83.8 | 77.2 | 63.7 | 60.8 | 36.9 | 50.4 | 67.5 | 31 | 78 |
| **OLMo-2-1124-13B** | 4.6·10²³ | 68.3 | 83.5 | 86.4 | 81.5 | 67.5 | 70.7 | 46.7 | 54.2 | 75.1 | 35.1 | 81.9 |

## Model Details

### Pretraining
|  | **OLMo 2 7B** | **OLMo 2 13B** |
|-------------------|------------|------------|
| Pretraining Stage 1<br>([OLMo-Mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124)) | 4 trillion tokens<br>(1 epoch) | 5 trillion tokens<br>(1.2 epochs) |
| Pretraining Stage 2<br>([Dolmino-Mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124)) | 50B tokens (3 runs)<br>*merged* | 100B tokens (3 runs)<br>300B tokens (1 run)<br>*merged* |
| Post-training<br>([Tulu 3 SFT OLMo mix](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-mixture)) | SFT + DPO + PPO<br>([preference mix](https://huggingface.co/datasets/allenai/olmo-2-1124-7b-preference-mix)) | SFT + DPO + PPO<br>([preference mix](https://huggingface.co/datasets/allenai/olmo-2-1124-13b-preference-mix)) |

#### Stage 1: Initial Pretraining
- Dataset: [OLMo-Mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124) (3.9T tokens)
- Coverage: 90%+ of total pretraining budget
- 7B Model: ~1 epoch
- 13B Model: 1.2 epochs (5T tokens)

#### Stage 2: Fine-tuning
- Dataset: [Dolmino-Mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124) (843B tokens)
- Three training mixes:
  - 50B tokens
  - 100B tokens
  - 300B tokens
- Mix composition: 50% high-quality data + academic/Q&A/instruction/math content

#### Model Merging
- 7B Model: 3 versions trained on 50B mix, merged via model souping
- 13B Model: 3 versions on 100B mix + 1 version on 300B mix, merged for final checkpoint


## Bias, Risks, and Limitations
Like any base language model or fine-tuned model without safety filtering, these models can easily be prompted by users to generate harmful and sensitive content. Such content may also be produced unintentionally, especially in cases involving bias, so we recommend that users consider the risks when applying this technology. Additionally, many statements from OLMo or any LLM are often inaccurate, so facts should be verified.


## Citation
```
@misc{olmo20242olmo2furious,
      title={2 OLMo 2 Furious}, 
      author={Team OLMo and Pete Walsh and Luca Soldaini and Dirk Groeneveld and Kyle Lo and Shane Arora and Akshita Bhagia and Yuling Gu and Shengyi Huang and Matt Jordan and Nathan Lambert and Dustin Schwenk and Oyvind Tafjord and Taira Anderson and David Atkinson and Faeze Brahman and Christopher Clark and Pradeep Dasigi and Nouha Dziri and Michal Guerquin and Hamish Ivison and Pang Wei Koh and Jiacheng Liu and Saumya Malik and William Merrill and Lester James V. Miranda and Jacob Morrison and Tyler Murray and Crystal Nam and Valentina Pyatkin and Aman Rangapur and Michael Schmitz and Sam Skjonsberg and David Wadden and Christopher Wilhelm and Michael Wilson and Luke Zettlemoyer and Ali Farhadi and Noah A. Smith and Hannaneh Hajishirzi},
      year={2024},
      eprint={2501.00656},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.00656}, 
}
```

## Model Card Contact
For errors in this model card, contact `olmo@allenai.org`.