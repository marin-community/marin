# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from marin.download.huggingface.download_hf import DownloadConfig, download_hf as _download_hf
from marin.execution.executor import ExecutorStep, executor_main, StepRef, versioned, step, deferred, output
from marin.transform.huggingface.dataset_to_eval import DatasetConversionConfig, OutputFormatOptions, hf_dataset_to_jsonl as _hf_dataset_to_jsonl

download_hf = deferred(_download_hf)
hf_dataset_to_jsonl = deferred(_hf_dataset_to_jsonl)

from experiments.eval_datasets import (
    arc_raw as ai2_arc_raw,
)
from experiments.eval_datasets import (
    bbh_raw,
    boolq_raw,
    commonsense_qa_raw,
    gpqa_raw,
    gsm8k_raw,
    hellaswag_raw,
    humaneval_raw,
    instruction_following_raw,
    lambada_openai_raw,
    math_raw,
    mmlu_pro_raw,
    musr_raw,
    openbookqa_raw,
    truthful_qa_raw,
    winograd_wsc_raw,
)
from experiments.eval_datasets import (
    piqa_baber_raw as piqa_raw,
)

"""
This script downloads HF datasets and converts them to dolma "text" format for decontamination.
We use the decontamination format to identify potential overlaps between pre-training data and evaluation data.

The script follows the pattern from eval_datasets.py but focuses only on the decontamination conversion.
"""

############################################################
# Download datasets
############################################################

############################################################
# Convert datasets to dolma format for decontamination
############################################################

# the one in eval_datasets.py is the wrong flattened version
# for some reason that doesn't have test splits
@step(name="raw/cais/mmlu_raw", override_output_path="raw/cais/mmlu_raw")
def mmlu_raw():
    return download_hf(DownloadConfig(
        hf_dataset_id="cais/mmlu",
        revision=versioned("c30699e"),
        gcs_output_path=output(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ))

# Convert gsm8k to dolma format
@step(name="decontamination/gsm8k-dolma")
def gsm8k_convert_dolma():
    return hf_dataset_to_jsonl(DatasetConversionConfig(
        dataset_name="gsm8k/main",
        subsets=["*"],
        splits=["test"],
        input_path=gsm8k_raw(),
        hf_path="gsm8k/main",
        output_path=output(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question",
        answer_text_key="answer",
    ))

# Convert math dataset to dolma format
@step(name="decontamination/math-dolma")
def math_convert_dolma():
    return hf_dataset_to_jsonl(DatasetConversionConfig(
        dataset_name="hendrycks/math",
        subsets=["*"],
        splits=["test"],
        input_path=math_raw(),
        hf_path="hendrycks/math",
        output_path=output(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="problem",
        answer_text_key="solution",
    ))

# Convert truthful_qa to dolma format
# columns are: question (string), best_answer (string), correct_answers (List[string])
@step(name="decontamination/truthful_qa-dolma")
def truthful_qa_convert_dolma():
    return hf_dataset_to_jsonl(DatasetConversionConfig(
        dataset_name="truthful_qa/truthful_qa",
        subsets=["generation"],
        splits=["validation"],
        input_path=truthful_qa_raw(),
        hf_path="truthful_qa/truthful_qa",
        output_path=output(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question",
        answer_text_key="best_answer",
        options_key="correct_answers",
    ))

# Convert bbh to dolma format
@step(name="decontamination/bbh-dolma")
def bbh_convert_dolma():
    return hf_dataset_to_jsonl(DatasetConversionConfig(
        dataset_name="SaylorTwift/bbh",
        subsets=["*"],
        splits=["test"],
        input_path=bbh_raw(),
        hf_path="SaylorTwift/bbh",
        output_path=output(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="input",
        answer_text_key="target",
    ))


@step(name="decontamination/mmlu")
def mmlu_convert_dolma():
    return hf_dataset_to_jsonl(DatasetConversionConfig(
        dataset_name="cais/mmlu",
        subsets=["*"],
        splits=["test"],
        input_path=mmlu_raw(),
        hf_path="cais/mmlu",
        output_path=output(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B", "C", "D"],
        exclude_subsets=["auxiliary_train"],
    ))

# Convert humaneval to dolma format
@step(name="decontamination/humaneval")
def humaneval_convert_dolma():
    return hf_dataset_to_jsonl(DatasetConversionConfig(
        dataset_name="openai/openai_humaneval",
        subsets=["*"],
        splits=["test"],
        input_path=humaneval_raw(),
        hf_path="openai/openai_humaneval",
        output_path=output(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="prompt",
        answer_text_key="canonical_solution",
    ))

# Convert instruction_following to dolma format (load remotely, no answers)
@step(name="decontamination/instruction_following")
def instruction_following_convert_dolma():
    return hf_dataset_to_jsonl(DatasetConversionConfig(
        dataset_name="wis-k/instruction-following-eval",
        subsets=["*"],
        splits=["train"],
        input_path="wis-k/instruction-following-eval",
        hf_path="wis-k/instruction-following-eval",
        output_path=output(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="prompt",
        options_key="instruction_id_list",
        answer_text_ignore=True,
    ))

# Convert gpqa to dolma format (load from HF hub, single split)
@step(name="decontamination/gpqa")
def gpqa_convert_dolma():
    return hf_dataset_to_jsonl(DatasetConversionConfig(
        dataset_name="Idavidrein/gpqa",
        subsets=["gpqa_main", "gpqa_extended", "gpqa_diamond"],
        splits=["train"],
        input_path="Idavidrein/gpqa",
        hf_path="Idavidrein/gpqa",
        output_path=output(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="Question",
        answer_text_key="Correct Answer",
        options_keys=["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"],
    ))


@step(name="decontamination/mmlu_pro")
def mmlu_pro_convert_dolma():
    return hf_dataset_to_jsonl(DatasetConversionConfig(
        dataset_name="TIGER-Lab/MMLU-Pro",
        subsets=["*"],
        splits=["test"],
        input_path=mmlu_pro_raw(),
        hf_path="TIGER-Lab/MMLU-Pro",
        output_path=output(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question",
        options_key="options",
        answer_idx_key="answer_index",
        answer_labels=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    ))

# Convert musr to dolma format
@step(name="decontamination/musr")
def musr_convert_dolma():
    return hf_dataset_to_jsonl(DatasetConversionConfig(
        dataset_name="WillHeld/MuSRDecontam",
        subsets=[""],
        splits=["test"],
        input_path=musr_raw(),
        hf_path="WillHeld/MuSRDecontam",
        output_path=output(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="narrative",
        options_key="choices",
        answer_idx_key="answer_index",
        answer_text_key="answer_choice",
    ))

# Convert HellaSwag to dolma format
@step(name="decontamination/hellaswag")
def hellaswag_convert_dolma():
    return hf_dataset_to_jsonl(DatasetConversionConfig(
        dataset_name="Rowan/hellaswag",
        subsets=["*"],
        splits=["test"],
        input_path=hellaswag_raw(),
        hf_path="Rowan/hellaswag",
        output_path=output(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="ctx",
        answer_text_ignore=True,
    ))

# Convert AI2-ARC to dolma format
@step(name="decontamination/ai2_arc")
def ai2_arc_convert_dolma():
    return hf_dataset_to_jsonl(DatasetConversionConfig(
        dataset_name="allenai/ai2_arc",
        subsets=["*"],
        splits=["test"],
        input_path=ai2_arc_raw(),
        hf_path="allenai/ai2_arc",
        output_path=output(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question",
        options_key="choices.text",
        answer_label_key="answerKey",
        answer_labels=["A", "B", "C", "D"],
        answer_text_ignore=True,
    ))

# Convert BoolQ to dolma format
@step(name="decontamination/boolq")
def boolq_convert_dolma():
    return hf_dataset_to_jsonl(DatasetConversionConfig(
        dataset_name="google/boolq",
        subsets=["*"],
        splits=["validation"],
        input_path=boolq_raw(),
        hf_path="google/boolq",
        output_path=output(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question",
        answer_text_ignore=True,
    ))

# Insert Tau Commonsense QA conversion step
@step(name="decontamination/commonsense_qa")
def commonsense_qa_convert_dolma():
    return hf_dataset_to_jsonl(DatasetConversionConfig(
        dataset_name="tau/commonsense_qa",
        subsets=["*"],
        splits=["validation"],
        input_path=commonsense_qa_raw(),
        hf_path="tau/commonsense_qa",
        output_path=output(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question",
        options_key="choices.text",
        answer_label_key="answerKey",
        answer_labels=["A", "B", "C", "D", "E"],
        answer_text_ignore=True,
    ))

# Convert Lambada OpenAI to dolma format
@step(name="decontamination/lambada_openai")
def lambada_openai_convert_dolma():
    return hf_dataset_to_jsonl(DatasetConversionConfig(
        dataset_name="EleutherAI/lambada_openai",
        subsets=["*"],
        splits=["test"],
        input_path=lambada_openai_raw(),
        hf_path="EleutherAI/lambada_openai",
        output_path=output(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="text",
        answer_text_ignore=True,
    ))

# Convert AllenAI OpenBookQA to dolma format
@step(name="decontamination/openbookqa")
def openbookqa_convert_dolma():
    return hf_dataset_to_jsonl(DatasetConversionConfig(
        dataset_name="allenai/openbookqa",
        subsets=["*"],
        splits=["test"],
        input_path=openbookqa_raw(),
        hf_path="allenai/openbookqa",
        output_path=output(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question_stem",
        options_key="choices.text",
        answer_label_key="answerKey",
        answer_labels=["A", "B", "C", "D"],
        answer_text_ignore=True,
    ))

# Convert PIQA to dolma format
@step(name="decontamination/piqa")
def piqa_convert_dolma():
    return hf_dataset_to_jsonl(DatasetConversionConfig(
        dataset_name="baber/piqa",
        subsets=["*"],
        splits=["test"],
        input_path=piqa_raw(),
        hf_path="baber/piqa",
        output_path=output(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="goal",
        options_keys=["sol1", "sol2"],
        answer_label_key="label",
        answer_labels=["0", "1"],
        answer_text_ignore=True,
    ))

# Convert Winograd WSC to dolma format
@step(name="decontamination/winograd_wsc")
def winograd_wsc_convert_dolma():
    return hf_dataset_to_jsonl(DatasetConversionConfig(
        dataset_name="marcov/winograd_wsc_wsc273_promptsource",
        subsets=["*"],
        splits=["test"],
        input_path=winograd_wsc_raw(),
        hf_path="marcov/winograd_wsc_wsc273_promptsource",
        output_path=output(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="rendered_input",
        options_key="options",
        answer_label_key="label",
        answer_labels=["0", "1"],
        answer_text_ignore=True,
    ))

############################################################

# List of evaluation dataset conversion steps for train-test overlap detection
EVAL_DATASET_STEPS: list[ExecutorStep] = [
    gsm8k_convert_dolma(),
    math_convert_dolma(),
    truthful_qa_convert_dolma(),
    bbh_convert_dolma(),
    mmlu_convert_dolma(),
    humaneval_convert_dolma(),
    instruction_following_convert_dolma(),
    gpqa_convert_dolma(),
    musr_convert_dolma(),
    mmlu_pro_convert_dolma(),
    hellaswag_convert_dolma(),
    ai2_arc_convert_dolma(),
    boolq_convert_dolma(),
    commonsense_qa_convert_dolma(),
    lambada_openai_convert_dolma(),
    openbookqa_convert_dolma(),
    piqa_convert_dolma(),
    winograd_wsc_convert_dolma(),
]

@step(name="train_test_overlap/eval_overlap/all")
def run_all_eval_overlap():
    """Entry point for eval datasets overlap analysis.

    Downloads raw evaluation datasets and converts them to dolma format
    for train-test overlap detection.
    """
    # Download raw datasets
    mmlu_raw()
    gsm8k_raw()
    math_raw()
    truthful_qa_raw()
    bbh_raw()
    humaneval_raw()
    gpqa_raw()
    instruction_following_raw()
    musr_raw()
    mmlu_pro_raw()
    boolq_raw()
    ai2_arc_raw()
    hellaswag_raw()
    piqa_raw()
    winograd_wsc_raw()
    commonsense_qa_raw()
    lambada_openai_raw()
    openbookqa_raw()

    # Convert to dolma format for decontamination
    gsm8k_convert_dolma()
    math_convert_dolma()
    truthful_qa_convert_dolma()
    bbh_convert_dolma()
    mmlu_convert_dolma()
    humaneval_convert_dolma()
    instruction_following_convert_dolma()
    gpqa_convert_dolma()
    musr_convert_dolma()
    mmlu_pro_convert_dolma()
    hellaswag_convert_dolma()
    ai2_arc_convert_dolma()
    boolq_convert_dolma()
    commonsense_qa_convert_dolma()
    lambada_openai_convert_dolma()
    openbookqa_convert_dolma()
    piqa_convert_dolma()
    winograd_wsc_convert_dolma()

if __name__ == "__main__":
    executor_main(
        steps=[run_all_eval_overlap()],
        description="Download and convert evaluation datasets for train-test overlap analysis",
    )
