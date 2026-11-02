<p align="center">
  <a href="https://github.com/Leolty/repobench#gh-light-mode-only">
    <img src="assets/repobench_dark.png" width="318px" alt="repobench logo" />
  </a>
  <a href="https://github.com/Leolty/repobench#gh-dark-mode-only">
    <img src="assets/repobench_light.png" width="318px" alt="repobench logo" />
  </a>

<p align="center">
  <a href="https://arxiv.org/abs/2306.03091">
    RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems
  </a>
  <br></br>
  <a>
    <b>ICLR 2024</b>
  </a>
</p>

<hr>

## 🔥 News

- *Feb 5th, 2024*: **RepoBench v1.1** (with newest code data) is now available on the 🤗 HuggingFace Hub. You can access the datasets for Python and Java using the following links:
  - For Python: [🤗 Repobench Python V1.1](https://huggingface.co/datasets/tianyang/repobench_python_v1.1)
  - For Java: [🤗 Repobench Java V1.1](https://huggingface.co/datasets/tianyang/repobench_java_v1.1)
  > **For more details of RepoBench v1.1, please refer to the [data directory](./data/README.md).**

- *Jan 16th, 2024*: RepoBench is accepted to ICLR 2024! 🎉


## 🛠️ Installation

```bash
git clone https://github.com/Leolty/repobench.git
cd repobench
```

> [!NOTE] 
> There is a `requirements.txt` file, which contains dependencies for reproducing the results in the paper. If you are only interested in the data, you can skip the installation of dependencies.

## ⚙️ Description of Settings

As discussed in the paper, we have three settings for each task:

- `cross_file_first`: Masks the line where a module from a different file is used for the first time.
- `cross_file_random`: Masks a random line where a module from a different file is used (not the first usage).
- `in_file`: Masks a random line that has no cross-file dependency.


## 📥 Load Data

```python
from datasets import load_dataset

dataset = load_dataset("tianyang/repobench_python_v1.1", ignore_verifications=True)
```

For more details, visit the Hugging Face dataset pages:
- Python: [🤗 Repobench Python V1.1](https://huggingface.co/datasets/tianyang/repobench_python_v1.1)
- Java: [🤗 Repobench Java V1.1](https://huggingface.co/datasets/tianyang/repobench_java_v1.1)

## 🚀 Running Experiments

To run experiments on the RepoBench v1.1 dataset, we provide a very basic `run.py` script using the 🤗 Transformers library.

Example usage:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --model_name "deepseek-ai/deepseek-coder-1.3b-base" \
               --dataset_name "tianyang/repobench_python_v1.1" \
               --start_date "2023-12-01" \
               --end_date "2023-12-31" \
               --language "python" \
               --max_token_nums 15800 \
               --levels "2k" "4k" "8k" "12k" "16k" \
               --temperature 0.2 \
               --top_p 0.95 \
               --max_new_tokens 128 \
               --batch_size 1
```

For a full list of available parameters, please refer to the `run.py` file. And it should be super easy to customize the script for your own needs.

## 📊 Evaluation

After generating completions, you can evaluate the results using the `eval.py` script. This script calculates various metrics including Exact Match (EM), Edit Similarity (ES), and CodeBLEU (CB) scores for each setting.

To run the evaluation:

```bash
python eval.py --path "results/deepseek-coder-1.3b-base-python" --language "python"
```

The script will output scores for each level (`cross_file_first`, `cross_file_random`, `in_file`) as well as weighted averages across all levels.

## 📝 Note

This branch of the repository is specifically for RepoBench v1.1. For the results presented in our ICLR 2024 paper, which used the initial version of RepoBench, please refer to the [`archive/v0` branch](https://github.com/Leolty/repobench/tree/archive/v0) of this repository.


## 📝 Citation

If you use RepoBench in your research, please consider citing us:

```bibtex
@misc{liu2023repobench,
      title={RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems}, 
      author={Tianyang Liu and Canwen Xu and Julian McAuley},
      year={2024},
      url={https://arxiv.org/abs/2306.03091},
      booktitle={International Conference on Learning Representations}
}
```






