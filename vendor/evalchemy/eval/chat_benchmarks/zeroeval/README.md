# ZeroEval: A Unified Framework for Evaluating Language Models

ZeroEval is a simple unified framework for evaluating (large) language models on various tasks.
This repository aims to evaluate instruction-tuned LLMs for their zero-shot performance on various reasoning tasks such as MMLU and GSM. We evaluate LLMs with a unified setup by controlling the factors such as prompting, sampling, output parsing, etc. In ZeroEval, we perform **zero-shot** prompting, and instruct LM to output both reasoning and answer in a **json**-formatted output. We are actively adding new tasks. Contributions are welcome! 

- Leaderboard: [https://hf.co/spaces/allenai/ZeroEval](https://huggingface.co/spaces/allenai/ZeroEval)
- [X post](https://x.com/billyuchenlin/status/1814037110577578377)


## Todo

- [ ] Support new tasks (GPPA, AIME, etc.)
- [ ] Add private tests 
- [ ] Prefix-prefill for open models such that the parsing is easier
- [ ] Add other formatting options (e.g. markup language instead of json, etc.)


## Installation 

<details>
  <summary> Click to expand </summary>

```bash
conda create -n zeroeval python=3.10
conda activate zeroeval
# pip install vllm -U # pip install -e vllm 
pip install vllm -U
pip install -r requirements.txt
# export HF_HOME=/path/to/your/custom/cache_dir/ 
```

</details>


## Tasks 

- [MMLU-redux](https://arxiv.org/abs/2406.04127) (`-d mmlu-redux`)
- [ZebraLogic](https://huggingface.co/blog/yuchenlin/zebra-logic) (`-d zebra-grid`)
- [CRUX](https://crux-eval.github.io/) (`-d crux`)
- [MATH (Level 5)](https://huggingface.co/datasets/AI-MO/aimo-validation-math-level-5) (`-d math-l5`)
- [GSM8K](https://openai.com/index/solving-math-word-problems/) (`-d gsm`)

- More tasks will be added soon. (e.g., ARC, MMLU-Pro, etc.)
<!-- - AlpacaEval (`-d alpaca-eval`) -->

## Usage

`zero_eval_local.sh` and `zero_eval_api.sh` are the two main scripts to run the evaluation.

### Examples

- `bash zero_eval_local.sh -d mmlu-redux -m meta-llama/Meta-Llama-3-8B-Instruct -p Meta-Llama-3-8B-Instruct -s 4` (Run Llama-3-8B-Instruct with greedy decoding on `mmlu-redux`)

- `bash zero_eval_api.sh -d gsm -f openai -m openai/gpt-4o-mini-2024-07-18 -p gpt-4o-mini-2024-07-18 -s 8` (Run gpt-4o-mini with greedy decoding on `gsm`)

- `bash zero_eval_api.sh -d zebra-grid -f openai -m deepseek-chat -p deepseek-chat -s 8` (Run deepseek-chat via openai style api, with greedy decoding on `zebra-grid`)


More examples can be found in the `scripts` folder, e.g., the [scripts/_MMLU_redux.md](scripts/_MMLU_redux.md) and [scripts/_GSM.md](scripts/_GSM.md) files as well as [`scripts/local/crux.sh`](ZeroEval/scripts/local/crux.sh).

### Arguments  
 

<details>
<summary>Command Line Arguments</summary>

| Arguments | Description | Default |
|-----|-------------|---------|
| `-d` | DATA_NAME: `mmlu-redux`, `gsm`, `math-l5`, `zebra-grid`, `alpaca_eval`, ... (see [src/task_configs.py](src/task_configs.py)) | |
| `-m` | model_name | |
| `-p` | model_pretty_name | |
| `-s` | number of shards (When `-s 1` we'll use all your GPUs for loading the model and running the inference; When `-s K`, we'll use K GPUs and divide the data into K shards for each GPU to run the inference on a single shard, and merge the results at the end.) | 1 |
| `-f` | engine (`vllm` by default for `zero_eval_local.sh`, can be changed to `hf`; For `zero_eval_api.sh`, we can use `openai`, `anthropic`, ...) | `vllm`/`openai` for `zero_eval_local/api.sh` |
| `-r` | run_name (the results will be saved in a sub folder with the `run_name` when it is specified) | "default" |
| `-t` | temperature | 0 (greedy decoding) |
| `-o` | top_p for nucleus sampling | 1.0 |
| `-e` | repetition penalty | 1.0 |
| `-b` | batch size | 4 |
| `-x` | max_length | 4096 |

</details>

## Results 

🚨 View results on our Leaderboard: [https://hf.co/spaces/allenai/ZeroEval](https://huggingface.co/spaces/allenai/ZeroEval)

- MMLU-Redux: `python src/evaluation/mcqa_eval.py mmlu-redux` --> [Full results](result_dirs/mmlu-redux.summary.md)
- GSM/MATH-L5: `python src/evaluation/math_eval.py math-l5/gsm` --> [Full results](result_dirs/gsm.summary.md)
- ZebraLogic: `python src/evaluation/zebra_grid_eval.py` --> [Full results](result_dirs/zebra-grid.summary.md)
  and [Leaderboard](https://huggingface.co/spaces/allenai/ZebraLogic)
- CRUX: `python src/evaluation/crux_eval.py` --> [Full results](result_dirs/crux.summary.md)
- All: `python src/evaluation/summarize.py` --> [Full results](result_dirs/summary.md) ⬇️




<!-- 
python src/evaluation/mcqa_eval.py mmlu-redux
python src/evaluation/math_eval.py math-l5
python src/evaluation/zebra_grid_eval.py
python src/evaluation/crux_eval.py
python src/evaluation/summarize.py

python src/evaluation/math_eval.py gsm 
 -->


### Changelogs 

- 08/02/2024: added Gemini 1.5 Pro Exp 0801 and CRUX results 
- 07/31/2024: added Meta-Llama-3.1-70B-Instruct and gemma-2-2b-it 
- 07/29/2024: added Llama-3.1-8B, Mistral-Large-2, and deepseek-coder-v2-0724 

## Citation
If you find ZeroEval useful, please cite it as follows in your publication:

```bibtex
@software{Lin_ZeroEval_A_Unified_2024,
    author = {Lin, Bill Yuchen},
    month = jul,
    title = {{ZeroEval: A Unified Framework for Evaluating Language Models}},
    url = {https://github.com/yuchenlin/ZeroEval},
    year = {2024}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yuchenlin/ZeroEval&type=Date)](https://star-history.com/#yuchenlin/ZeroEval&Date)
