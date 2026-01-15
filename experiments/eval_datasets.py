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

import dataclasses

from marin.download.huggingface.download_hf import DownloadConfig
from marin.download.huggingface.download_hf import download_hf as _download_hf
from marin.execution import deferred, executor_main, output, step, versioned
from marin.execution.executor import ExecutorStep
from marin.transform.huggingface.dataset_to_eval import DatasetConversionConfig, OutputFormatOptions
from marin.transform.huggingface.dataset_to_eval import hf_dataset_to_jsonl as _hf_dataset_to_jsonl

from experiments.defaults import default_download

# Mark library functions as deferred
download_hf = deferred(_download_hf)
hf_dataset_to_jsonl = deferred(_hf_dataset_to_jsonl)

"""
This script downloads HF datasets for various tasks and converts them to prompt/response JSONL format for log prob
evaluation. It also converts them to dolma "text" format for decontamination.

To adda new dataset, you need to:
1. Download the dataset (in the download section)
2. Convert the dataset to evaluation format (in the conversion section)
3. Add the dataset to the eval_datasets list
4. (Optional) Convert the dataset to dolma format (in the conversion section)

TODO: group together the download and conversion steps for each dataset
"""

"""
Downloads the following datasets
- mmlu
- boolq
- piqa
- winogrande
- arc
- openbookqa
- hellaswag
- MMLU-Pro
- openai_humaneval
- mbpp

"""
############################################################
# download mmlu dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
@step(name="raw/cais/mmlu", override_output_path="raw/cais/mmluhf")
def mmlu_raw():
    return download_hf(
        DownloadConfig(
            hf_dataset_id="cais/mmlu",
            revision=versioned("c30699e"),
            gcs_output_path=output(),
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.md"],
        )
    )

# download boolq dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
boolq_raw = default_download(
    name="raw/google/boolq",
    hf_dataset_id="google/boolq",
    revision=versioned("35b264d"),
    override_output_path="raw/google/boolqhf",
    hf_urls_glob=["**/*.parquet"],
)

# download hellaswag dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
hellaswag_raw = default_download(
    name="raw/Rowan/hellaswag",
    hf_dataset_id="Rowan/hellaswag",
    revision=versioned("50441ce"),
    override_output_path="raw/Rowan/hellaswaghf",
    hf_urls_glob=["**/*.parquet"],
)

# download piqa dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
piqa_raw = default_download(
    name="raw/ybisk/piqa",
    hf_dataset_id="ybisk/piqa",
    revision=versioned("142c512"),
    override_output_path="raw/ybisk/piqahf",
    hf_urls_glob=["**/*.parquet"],
)

# download winogrande dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
winogrande_raw = default_download(
    name="raw/allenai/winogrande",
    hf_dataset_id="allenai/winogrande",
    revision=versioned("ebf71e3"),
    override_output_path="raw/allenai/winograndehf",
    hf_urls_glob=["winogrande_xl/**/*.parquet"],
)

# download arc dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
arc_raw = default_download(
    name="raw/allenai/ai2_arc",
    hf_dataset_id="allenai/ai2_arc",
    revision=versioned("210d026"),
    override_output_path="raw/allenai/ai2_archf",
    hf_urls_glob=["**/*.parquet", "*.md"],
)

# download openbookqa dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
openbookqa_raw = default_download(
    name="raw/allenai/openbookqa",
    hf_dataset_id="allenai/openbookqa",
    revision=versioned("388097e"),
    override_output_path="raw/allenai/openbookqahf",
    hf_urls_glob=["**/*.parquet", "*.md"],
)

# download MMLU-Pro dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
mmlu_pro_raw = default_download(
    name="raw/TIGER-Lab/MMLU-Pro",
    hf_dataset_id="TIGER-Lab/MMLU-Pro",
    revision=versioned("3373e0b"),
    override_output_path="raw/TIGER-Lab/MMLU-Prohf",
    hf_urls_glob=["**/*.parquet", "*.md"],
)

# download openai_humaneval
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
humaneval_raw = default_download(
    name="raw/openai/openai_humaneval",
    hf_dataset_id="openai/openai_humaneval",
    revision=versioned("7dce605"),
    override_output_path="gs://marin-us-central2/raw/openai/openai_humanevalhf",
    hf_urls_glob=["**/*.parquet", "*.md"],
)

# download mbpp
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
@step(name="raw/google-research-datasets/mbpp", override_output_path="raw/google-research-datasets/mbpphf")
def mbpp_raw():
    return download_hf(
        DownloadConfig(
            hf_dataset_id="google-research-datasets/mbpp",
            revision=versioned("4bb6404"),
            gcs_output_path=output(),
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.md"],
        )
    )

"""
Converts raw to JSON for:
- mmlu
- boolq
- piqa
- winogrande
- arc-easy
- arc-challenge
- openbookqa
- hellaswag
- MMLU-Pro
- openai_humaneval
- mbpp
"""
############################################################
# Convert mmlu to evaluation format (i.e. JSON with "prompt", "response" fields)
# This is the input for internal evaluation which measures PPL model gives to correct responses to prompts


@dataclasses.dataclass(frozen=True)
class EvalDataset:
    """
    A dataset for log prob evaluation. The steps should point to the data in prompt/response JSONL format.
    """

    org: str
    name: str
    steps: list[ExecutorStep]
    tags: list[str] = dataclasses.field(default_factory=list)


# This creates a JSON file representing the auxiliary training data subset of MMLU
@step(name="evaluation/mmlu-eval-aux")
def mmlu_aux_eval():
    raw = mmlu_raw()
    return hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="cais/mmlu",
            subsets=["all"],
            splits=["auxiliary_train"],
            input_path=raw,
            hf_path="cais/mmlu",
            output_path=output(),
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question",
            options_key="choices",
            answer_idx_key="answer",
            answer_labels=["A", "B", "C", "D"],
        )
    )

# This creates one file per subject from MMLU, excluding the all and auxiliary training subsets
@step(name="evaluation/mmlu-eval-subject")
def mmlu_subject_eval():
    raw = mmlu_raw()
    return hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="cais/mmlu",
            subsets=["*"],
            splits=["dev", "validation"],
            input_path=raw,
            hf_path="cais/mmlu",
            output_path=output(),
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question",
            options_key="choices",
            answer_idx_key="answer",
            answer_labels=["A", "B", "C", "D"],
            exclude_subsets=["all", "auxiliary_train"],
        )
    )

# This creates a JSON file representing the train and validation data subset of boolq
@step(name="evaluation/boolq-eval")
def boolq_eval():
    return hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="google/boolq",
            subsets=["*"],
            splits=["train", "validation"],
            input_path=boolq_raw,
            hf_path="google/boolq",
            output_path=output(),
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question",
            answer_label_key="answer",
            answer_labels=[True, False],
            answer_text_ignore=True,
        )
    )

# This creates a JSON file representing the training and validation data subset of piqa
@step(name="evaluation/piqa")
def piqa_eval():
    return hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="ybisk/piqa",
            subsets=["*"],
            splits=["train", "validation"],
            input_path=piqa_raw,
            hf_path="ybisk/piqa",
            output_path=output(),
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="goal",
            options_keys=["sol1", "sol2"],
            answer_idx_key="label",
            answer_labels=["1", "2"],
        )
    )

# This creates a JSON file representing the training and validation data subset of winogrande_xl
@step(name="evaluation/winogrande")
def winogrande_eval():
    return hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="allenai/winogrande",
            subsets=["default"],
            splits=["train", "validation"],
            input_path=winogrande_raw,
            hf_path="allenai/winogrande",
            output_path=output(),
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="sentence",
            options_keys=["option1", "option2"],
            answer_label_key="answer",
            answer_labels=["1", "2"],
        )
    )

# This creates a JSON file representing the train and validation splits of ARC-Easy
@step(name="evaluation/arc-easy")
def arc_easy_eval():
    return hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="allenai/ai2_arc",
            subsets=["ARC-Easy"],
            splits=["train", "validation"],
            input_path=arc_raw,
            hf_path="allenai/ai2_arc",
            output_path=output(),
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question",
            options_key="choices.text",
            answer_labels_key="choices.label",
            answer_label_key="answerKey",
        )
    )

# This creates a JSON file representing the train and validation splits of ARC-Challenge
@step(name="evaluation/arc-challenge")
def arc_challenge_eval():
    return hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="allenai/ai2_arc",
            subsets=["ARC-Challenge"],
            splits=["train", "validation"],
            input_path=arc_raw,
            hf_path="allenai/ai2_arc",
            output_path=output(),
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question",
            options_key="choices.text",
            answer_labels_key="choices.label",
            answer_label_key="answerKey",
        )
    )

# This creates a JSON file for the train and validation subsets of OpenBookQA
@step(name="evaluation/openbookqa-eval")
def openbookqa_eval():
    return hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="allenai/openbookqa",
            subsets=["main"],
            splits=["train", "validation"],
            input_path=openbookqa_raw,
            hf_path="allenai/openbookqa",
            output_path=output(),
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question_stem",
            options_key="choices.text",
            answer_label_key="answerKey",
            answer_labels_key="choices.label",
        )
    )

# This creates a JSON file representing the training and validation splits for hellaswag
@step(name="evaluation/hellaswag-eval")
def hellaswag_eval():
    return hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="Rowan/hellaswag",
            subsets=["*"],
            splits=["train", "validation"],
            input_path=hellaswag_raw,
            hf_path="Rowan/hellaswag",
            output_path=output(),
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="ctx",
            options_key="endings",
            answer_labels=["A", "B", "C", "D"],
            answer_idx_key="label",
        )
    )

# This creates a JSON file representing the test and validation splits for MMLU-Pro
@step(name="evaluation/MMLU-Pro-eval")
def mmlu_pro_eval():
    return hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="TIGER-Lab/MMLU-Pro",
            subsets=["*"],
            splits=["test", "validation"],
            input_path=mmlu_pro_raw,
            hf_path="TIGER-Lab/MMLU-Pro",
            output_path=output(),
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question",
            options_key="options",
            answer_labels=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"],
            answer_idx_key="answer_index",
        )
    )

# This creates a JSON file representing the test and validation splits for openai_humaneval
@step(name="evaluation/humaneval-eval")
def humaneval_eval():
    return hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="openai/openai_humaneval",
            subsets=["*"],
            splits=["test"],
            input_path=humaneval_raw,
            hf_path="openai/openai_humaneval",
            output_path=output(),
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="prompt",
            answer_text_key="canonical_solution",
        )
    )

# This creates a JSON file representing the train, test, and validation splits for mbpp
@step(name="evaluation/mbpp-eval")
def mbpp_eval():
    raw = mbpp_raw()
    return hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="google-research-datasets/mbpp",
            subsets=["*"],
            splits=["train", "test", "validation"],
            input_path=raw.cd("4bb6404/full"),
            hf_path="google-research-datasets/mbpp",
            output_path=output(),
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="text",
            answer_text_key="code",
        )
    )

eval_datasets = [
    # these tags are used to group datasets together for averaging
    EvalDataset("cais", "mmlu", [mmlu_aux_eval(), mmlu_subject_eval()]),
    EvalDataset("google", "boolq", [boolq_eval()], ["core"]),
    EvalDataset("Rowan", "hellaswag", [hellaswag_eval()], ["core"]),
    EvalDataset("ybisk", "piqa", [piqa_eval()], ["core"]),
    EvalDataset("allenai", "winogrande", [winogrande_eval()], ["core"]),
    EvalDataset("allenai", "ai2_arc_easy", [arc_easy_eval()], ["core", "arc"]),
    EvalDataset("allenai", "ai2_arc_challenge", [arc_challenge_eval()], ["core", "arc"]),
    EvalDataset("allenai", "openbookqa", [openbookqa_eval()], ["core"]),
    EvalDataset("openai", "openai_humaneval", [humaneval_eval()]),
    EvalDataset("google-research-datasets", "mbpp", [mbpp_eval()]),
]


############################################################
# Convert mmlu to dolma format (i.e. JSON with "text" field)
# This is used as input to the decontamination pipeline so documents with MMLU content are removed
@step(name="decontamination/mmlu-dolma")
def mmlu_convert_dolma():
    raw = mmlu_raw()
    return hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="cais/mmlu",
            subsets=["all"],
            splits=["dev", "test", "validation"],
            input_path=raw,
            hf_path="cais/mmlu",
            output_path=output(),
            output_format=OutputFormatOptions("decontamination"),
            prompt_key="question",
            options_key="choices",
            answer_idx_key="answer",
            answer_labels=["A", "B", "C", "D"],
        )
    )

@step(name="raw/ambean/lingOly")
def lingoly():
    return download_hf(
        DownloadConfig(
            hf_dataset_id="ambean/lingOly",
            revision=versioned("6aff4c2"),
            gcs_output_path=output(),
            wait_for_completion=True,
        )
    )

# gsm8k raw
@step(name="raw/gsm8k", override_output_path="raw/gsm8k/mainhf")
def gsm8k_raw():
    return download_hf(
        DownloadConfig(
            hf_dataset_id="openai/gsm8k",
            revision=versioned("e53f048"),
            gcs_output_path=output(),
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.md"],
        )
    )

# math dataset raw
@step(name="raw/hendrycks_math", override_output_path="raw/hendrycks/mathhf")
def math_raw():
    return download_hf(
        DownloadConfig(
            hf_dataset_id="EleutherAI/hendrycks_math",
            revision=versioned("21a5633"),
            gcs_output_path=output(),
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.md"],
        )
    )

# truthful_qa raw
@step(name="raw/truthful_qa", override_output_path="raw/truthful_qa/multiple_choicehf")
def truthful_qa_raw():
    return download_hf(
        DownloadConfig(
            hf_dataset_id="truthfulqa/truthful_qa",
            revision=versioned("741b827"),
            gcs_output_path=output(),
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.md"],
        )
    )

# bbh raw
@step(name="raw/bbh", override_output_path="raw/SaylorTwift/bbhhf")
def bbh_raw():
    return download_hf(
        DownloadConfig(
            hf_dataset_id="SaylorTwift/bbh",
            revision=versioned("b5306be"),
            gcs_output_path=output(),
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.md"],
        )
    )

# gpqa raw
@step(name="raw/gpqa", override_output_path="raw/Idavidrein/gpqa")
def gpqa_raw():
    return download_hf(
        DownloadConfig(
            hf_dataset_id="Idavidrein/gpqa",
            revision=versioned("90b8e5b"),
            gcs_output_path=output(),
            wait_for_completion=True,
            hf_urls_glob=["**/*.csv", "*.csv"],
        )
    )

# instruction-following raw
@step(name="raw/instruction_following_eval", override_output_path="raw/wis-k/instruction-following-evalhf")
def instruction_following_raw():
    return download_hf(
        DownloadConfig(
            hf_dataset_id="wis-k/instruction-following-eval",
            revision=versioned("5a5661c"),
            gcs_output_path=output(),
            wait_for_completion=True,
            hf_urls_glob=["**/*.jsonl", "*.jsonl"],
        )
    )

# musr raw
@step(name="raw/musr", override_output_path="raw/WillHeld/MuSRDecontamhf")
def musr_raw():
    return download_hf(
        DownloadConfig(
            hf_dataset_id="WillHeld/MuSRDecontam",
            revision=versioned("39b4f56"),
            gcs_output_path=output(),
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.parquet"],
        )
    )

# winograd WSC raw
@step(name="raw/winograd_wsc", override_output_path="raw/marcov/winograd_wsc_wsc273_promptsourcehf")
def winograd_wsc_raw():
    return download_hf(
        DownloadConfig(
            hf_dataset_id="marcov/winograd_wsc_wsc273_promptsource",
            revision=versioned("63befd8"),
            gcs_output_path=output(),
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.parquet"],
        )
    )

# commonsense_qa raw
@step(name="raw/commonsense_qa", override_output_path="raw/tau/commonsense_qahf")
def commonsense_qa_raw():
    return download_hf(
        DownloadConfig(
            hf_dataset_id="tau/commonsense_qa",
            revision=versioned("94630fe"),
            gcs_output_path=output(),
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.parquet"],
        )
    )

# lambada_openai raw
@step(name="raw/lambada_openai", override_output_path="raw/EleutherAI/lambada_openaihf")
def lambada_openai_raw():
    return download_hf(
        DownloadConfig(
            hf_dataset_id="EleutherAI/lambada_openai",
            revision=versioned("879e19a"),
            gcs_output_path=output(),
            wait_for_completion=True,
            hf_urls_glob=["**/*.jsonl", "*.jsonl"],
        )
    )

# Alternative PIQA variant used only for contamination analysis
@step(name="raw/baber/piqa", override_output_path="raw/baber/piqahf")
def piqa_baber_raw():
    return download_hf(
        DownloadConfig(
            hf_dataset_id="baber/piqa",
            revision=versioned("142f6d7"),
            gcs_output_path=output(),
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.parquet"],
        )
    )

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            mmlu_convert_dolma(),
            *[step for ds in eval_datasets for step in ds.steps],
        ]
    )
