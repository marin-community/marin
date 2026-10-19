<div align="center">
    <h1><img height="150px" src="./images/matharena_icon.png" alt="MathArena"><br>MathArena</h1>

  <a href="https://www.python.org/">
<img alt="Build" src="https://img.shields.io/badge/Python-3.12-1f425f.svg?color=blue">
  </a>
  <a href="https://opensource.org/licenses/MIT">
<img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a>
</div>

## 👋 Overview

MathArena is a platform for the evaluation of LLMs on the latest math competitions and olympiads. It is hosted on [matharena.ai](https://matharena.ai/). This repository contains all the code used for model evaluation of the competitions. The README explains how to run your models or add a new competition. 

## 📑 Table of Contents
- [Installation](#installation)
- [Evaluating a New Model](#evaluating-a-new-model)
  - [Model Configuration](#model-configuration)
  - [Running the Model](#running-the-model)
  - [Local VLLM Usage](#running-models-locally-using-vllm)
- [Adding a Competition](#adding-a-competition)
  - [Setting Up Competition Files](#setting-up-competition-files)
  - [Verifying Problem Statements](#verifying-problem-statements)
  - [Running Models on Competitions](#running-models-on-competitions)
  - [Competitions Requiring Grading](#competitions-requiring-grading)
  - [Running LLMs as Judges](#running-llms-as-judges)
- [Viewing Results](#viewing-results)
- [Evaluation logs](#evaluation-logs)

## 🚀 Installation

MathArena uses [UV](https://github.com/astral-sh/uv) to manage dependencies. If you want to run local models, uncomment the vllm installation in `pyproject.toml`.

### Install UV

- **macOS and Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Windows:**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
---

### Alternative installation

As an alternative to UV, you can also create a conda environment and install the package as follows:
```bash
conda create -n matharena python=3.12
conda activate matharena
python -m pip install -e .
```
If you choose this option, disregard `uv run` in all instructions and use python directly instead.

## 🤖 Evaluating a New Model

### Model Configuration

Create a configuration file in the `configs/models` folder. Each config must include:
- **Required:**
  - `model`: Model name. Reasoning effort of OpenAI models can be set by appending `--[low/medium/high]` to the model name, e.g., `o3-mini--high`.
  - `api`: API provider. The API key should be defined as environment variable when using the specified API. The supported options with their corresponding API keys are:
    - **openai**: `OPENAI_API_KEY`
    - **anthropic**: `ANTHROPIC_API_KEY`
    - **together**: `TOGETHER_API_KEY`
    - **google**: `GOOGLE_API_KEY`
    - **deepseek**: `DEEPSEEK_API_KEY`
    - **openrouter**: `OPENROUTER_API_KEY`
    - **vllm**: (runs locally; no API key required)
  - `human_readable_id`: A unique, descriptive identifier.
- **Optional Parameters:**
  - API settings like `temperature`, `top_p`, and `top_k` (default: `temperature` is from competition config, see [Adding a Competition](#adding-a-competition)).
  - `max_tokens`: Max number of tokens for the model (default: from competition config, see [Adding a Competition](#adding-a-competition)).
  - `concurrent_requests`: Number of parallel requests to API (default: 30).
  - `timeout`: Request timeout in seconds (default: 2000).
  - `max_retries`: Retry attempts to API (default: 50).
  - `read_cost` & `write_cost`: Cost per million tokens in USD for input and output tokens (default: 1 each).
  - `date`: Creation date of the model in the format "yyyy-mm-dd". Only affects the website.
  - `batch_processing`: If set to true, the model will be queried using batch processing. Only available for OpenAI and Anthropic models.

### Running the model
Execute the following command to evaluate a model on a competition:
```bash
uv run python scripts/run.py --configs path/to/your/config --comp path/to/competition
```
- `path/to/your/config`: Relative path from the `configs/models` folder to the model configuration (excluding the `.yaml` extension).
- `path/to/competition`: Relative path from the `configs/competition` folder to the competition folder (excluding the `.yaml` extension).

**Example:**
```bash
uv run python scripts/run.py --configs openai/gpt-4o --comp aime/aime_2025_I
```

**Additional Flags:**
- `skip-existing`: Skip problems already processed through the model.
- `n`: Number of runs per problem (default: 4).

*Note*: Errors thrown by the API provider are retried every minute up to 50 times. If no answer is returned after 50 tries, the answer will be counted as incorrect. Running again with `skip-existing` enabled will attempt to run the problems on which this occurred again.

### Running Models Locally Using VLLM

If using a local model with vllm, start the server:
```bash
vllm serve [[model_name]] --dtype auto --api-key token-abc123
```

### Uploading answers to HuggingFace
You can upload the model answers to HuggingFace as follows:
```bash
uv run python scripts/upload_outputs.py --org your_org --repo-name your_repo_name --comp path/to/competition
```
This will upload all model answers to a private repository named `your_org/your_repo_name`. `path/to/competition` is the relative path from the `configs/competition` folder to the competition folder (excluding the `.yaml` extension).

## ➕ Adding a Competition

### Competition Format
MathArena supports the addition of any benchmark or competition uploaded to HuggingFace (or locally saved using the `datasets` library) that has the following columns:
- `problem_idx` (int): The id associated with the problem.
- `problem`(str): The problem statement.
- `answer` (str, Optional): The answer to the problem. Required for competitions with final answers.
- `points` (int, Optional): The number of points associated with the problem. Only required for competitions without final answers.
- `sample_solution` (str, Optional): Sample solution to the problem. Only required for competitions without final answers and during autograding.
- `sample_grading` (str, Optional): Example of how the grading format should look like. Only required for competitions without final answers and during autograding.
- `grading_scheme` (list, Optional): The grading scheme for the problem. Only required for competitions without final answers.
We refer to [the instructions regarding graded competitions](#competitions-requiring-grading) for the specific format of the grading scheme.

### Configuration
To set up MathArena for evaluation on the competition, you should add a competition config file in the `configs/competitions` folder with the following parameters:
- `instruction`: Instructions for the model. *Must* require the final answer be in `\boxed{}`.
- `default_temperature`: Default temperature.
- `default_max_tokens`: Default max tokens.
- `strict_parsing`: `true` for strict format matching (e.g., only `\boxed{43}` is accepted) or `false` for lenient parsing.
- `n_problems`: Total number of problems.
- `date`: Date of the competition, in the format "YYYY-MM-DD".
- `dataset_path`: Path to the dataset uploaded on HuggingFace or stored locally.
- `final_answer` (optional): If set to false, the competition is one that is manually graded with judges. Defaults to true if not set.

### Manual Curation and Creation
To create a pipeline that enables quick curation and easy generation of new competitions, we describe our full process for dataset creation. Note that you do not have to follow these steps if you have another way to generate your benchmark in the appropriate format.

#### Setting Up Competition Files
In the `data/` folder, create a new directory for your competition with the following structure:
1. **Problems:**  
   - Create a subfolder `problems/` and add each problem as a separate LaTeX file named `1.tex`, `2.tex`, ..., `{k}.tex`, where `k` is the number of problems in your competition. You can skip a problem if you want/need to.
2. **Answers:**  
   - If the competition is one based on final answers, add an `answers.csv` file with columns `id` and `answer`.
     - `id`: The problem filename (without the `.tex` extension).
     - `answer`: The integer answer.
   - If the competition is evaluated using human judges, add a `grading_scheme.json` file. This file should consist of a list of dictionaries, each of which contain the following fields:
     - `id`: The problem filename (without the `.tex` extension).
     - `points`: The maximum number of points for the question.
     - `scheme`: A list of dictionaries, each containing substeps for which points are awarded. Each dictionary contains the following keys:
        - `points`: Points associated with this step.
        - `title`: Title of the step. Should be unique across all dictionaries in this scheme.
        - `desc`: Description of the step.

#### Verifying Problem Statements
Ensure your LaTeX problems compile correctly:
```bash
uv run python scripts/curation/check_latex.py --comp path/to/competition
```
Then, build the `latex/main.tex` to generate a PDF and confirm all problems appear as expected.

#### Upload to HuggingFace
Finally, you can upload the competition to HuggingFace:
```bash
uv run python scripts/curation/upload_competition.py --org your_org --repo-name your_repo_name --comp path/to/competition
```
This will upload all answers in the appropriate format to a private repository named `your_org/your_repo_name`. `path/to/competition` is the relative path from the `configs/competition` folder to the competition folder (excluding the `.yaml` extension). Thus, you need to have created the configuration file before uploading to HuggingFace.

### Running Models on Competitions
To run multiple models (possibly across different APIs), use:
```bash
uv run python scripts/curation/run_multiple.py --apis openai google anthropic together --comp path/to/competition
```
This will run models from the same API sequentially and from different APIs concurrently.
**Options:**
- `--simul`: Run all models in parallel, even if they use the same API.
- `models`: Provide space-separated regex patterns to filter models. A model is only run if it matches any of the regexes.
- `skip-existing`: Skip problems already processed through the model.
- `n`: Number of runs per problem (default: 4).

*Note:* For local vllm usage, ensure the vllm server is running as described above. Logs will be found in the `logs/` folder.

### Competitions Requiring Grading
To set up grading of questions, convert the model answers to TeX files: 
```bash
uv run python scripts/judge/answers_to_latex.py --comp path/to/competition
```
This will compile all model answers in a PDF file in the folder `latex/path/to/competition`.

Now, collect all PDFs for all evaluated models in a single folder using:
```bash
uv run python scripts/judge/collect_pdfs.py --comp path/to/competition
```
This will put all PDFs associated with question with idx `i` in the folder `latex/path/to/competition/i`. Each PDF will be given a unique (anonymous) ID. Follow the instructions in `README_judges.md` to grade each PDF. The grading of a single PDF should be placed in `grading/path/to/competition/i/{ID}.json`, where the ID is the ID given to the PDF associated with the grading. In case a PDF is graded by multiple people, you can add more files by naming them `grading/path/to/competition/i/{ID}_{X}.json` where `X` is any suffix. Finally, run
```bash
uv run python scripts/judge/merge_judgments.py --comp path/to/competition
```
This will add the judgments directly in the raw output traces of the models.

### Running LLMs as judges
To run an LLM as a judge, you must first add the solutions of all problems of the competition in `data/path/to/competition/solutions/{i}.text` where `i` is the index of the problem. 

Then, use the following command:
```bash
uv run python scripts/grade.py --grader-config path/to/grader --solver-config path/to/solver/1 path/to/solver/2 --comp path/to/competition
```

**Options:**
- `path/to/grader`: Relative path from the `configs/models` folder to the model configuration for the judge.
- `path/to/solver`: Relative path from the `configs/models` folder to the model configuration of the judged model. Multiple ones can be given by passing space-separated paths.
- `path/to/competition`: Relative path from the `configs/competitions` folder to the competition folder.

**Example:**
```bash
uv run python scripts/grade.py --grader-config openai/o3-mini --solver-config openai/o3-mini anthropic/claude-37 --comp usamo/usamo_2025
```

**Additional Flags:**
- `skip-existing`: Skip problems already processed through the model.
- `n`: Number of runs per problem to evaluate (default: 4). Must be no larger than the amount of generated solutions.

*Notes:* For local vllm usage, ensure the vllm server is running as described above. Logs will be found in the `logs/` folder. We also recommend either using generalist models or either of `o1`, `o3-mini` or `Claude 3.7` as graders due to their robustness with respect to the formatting instructions. 

After obtaining the judgments, you can then process them using the aforementioned `merge_judgments.py` script:
```bash
uv run python scripts/judge/merge_judgments.py --comp path/to/competition --grading-folder autogradings
```
This will add the judgments directly in the raw output traces of the models.

## 📊 Viewing Results

Launch a local web server to inspect the results:
```bash
uv run python scripts/app.py --comp path/to/competition
```
Access the app at [http://localhost:5001/](http://localhost:5001/). Warning signs for solutions indicate a potential problem with the model run and should be manually verified. Any warning is caused by one of the following problems:

* 💀: parser threw an error or encountered something unexpected.
* ⚠️: The correct answer might be present in the model answer, but it was not extracted.
* ❕: Model likely hit max token limit.

If issues are found, delete the corresponding output file or fix the parser and rerun the model with `skip-existing`. If the parser requires a manual overwrite, you can edit `src/matharena/parse_manual.py` and add a key-value pair mapping the model solution to a parseable solution.

## 🪵 Evaluation Logs

You can find logs from our evaluation containing full reasoning traces (if available) and solutions produced by the models at the following link: [https://huggingface.co/MathArena](https://huggingface.co/MathArena).

## 📚 Citation

```
@misc{balunovic_srimatharena_2025,
	title = {MathArena: Evaluating LLMs on Uncontaminated Math Competitions},
  author = {Mislav Balunović and Jasper Dekoninck and Ivo Petrov and Nikola Jovanović and Martin Vechev},
	copyright = {MIT},
	url = {https://matharena.ai/},
	publisher = {SRI Lab, ETH Zurich},
	month = feb,
	year = {2025},
}
```
