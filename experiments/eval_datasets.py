# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.step_model import StepSpec
from marin.execution.step_runner import StepRunner
from marin.transform.huggingface.dataset_to_eval import DatasetConversionConfig, OutputFormatOptions, hf_dataset_to_jsonl

from experiments.defaults import default_download

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
mmlu_raw = StepSpec(
    name="raw/cais/mmlu",
    hash_attrs={"hf_dataset_id": "cais/mmlu", "revision": "c30699e", "hf_urls_glob": ["**/*.parquet", "*.md"]},
    override_output_path="raw/cais/mmluhf",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="cais/mmlu",
            revision="c30699e",
            gcs_output_path=output_path,
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.md"],
        )
    ),
)

# download boolq dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
boolq_raw = default_download(
    name="raw/google/boolq",
    hf_dataset_id="google/boolq",
    revision="35b264d",
    override_output_path="raw/google/boolqhf",
    hf_urls_glob=["**/*.parquet"],
)

# download hellaswag dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
hellaswag_raw = default_download(
    name="raw/Rowan/hellaswag",
    hf_dataset_id="Rowan/hellaswag",
    revision="50441ce",
    override_output_path="raw/Rowan/hellaswaghf",
    hf_urls_glob=["**/*.parquet"],
)

# download piqa dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
piqa_raw = default_download(
    name="raw/ybisk/piqa",
    hf_dataset_id="ybisk/piqa",
    revision="142c512",
    override_output_path="raw/ybisk/piqahf",
    hf_urls_glob=["**/*.parquet"],
)

# download winogrande dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
winogrande_raw = default_download(
    name="raw/allenai/winogrande",
    hf_dataset_id="allenai/winogrande",
    revision="ebf71e3",
    override_output_path="raw/allenai/winograndehf",
    hf_urls_glob=["winogrande_xl/**/*.parquet"],
)

# download arc dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
arc_raw = default_download(
    name="raw/allenai/ai2_arc",
    hf_dataset_id="allenai/ai2_arc",
    revision="210d026",
    override_output_path="raw/allenai/ai2_archf",
    hf_urls_glob=["**/*.parquet", "*.md"],
)

# download openbookqa dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
openbookqa_raw = default_download(
    name="raw/allenai/openbookqa",
    hf_dataset_id="allenai/openbookqa",
    revision="388097e",
    override_output_path="raw/allenai/openbookqahf",
    hf_urls_glob=["**/*.parquet", "*.md"],
)

# download MMLU-Pro dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
mmlu_pro_raw = default_download(
    name="raw/TIGER-Lab/MMLU-Pro",
    hf_dataset_id="TIGER-Lab/MMLU-Pro",
    revision="3373e0b",
    override_output_path="raw/TIGER-Lab/MMLU-Prohf",
    hf_urls_glob=["**/*.parquet", "*.md"],
)

# download openai_humaneval
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
humaneval_raw = default_download(
    name="raw/openai/openai_humaneval",
    hf_dataset_id="openai/openai_humaneval",
    revision="7dce605",
    override_output_path="gs://marin-us-central2/raw/openai/openai_humanevalhf",
    hf_urls_glob=["**/*.parquet", "*.md"],
)

# download mbpp
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
_mbpp_base = StepSpec(
    name="raw/google-research-datasets/mbpp",
    hash_attrs={
        "hf_dataset_id": "google-research-datasets/mbpp",
        "revision": "4bb6404",
        "hf_urls_glob": ["**/*.parquet", "*.md"],
    },
    override_output_path="raw/google-research-datasets/mbpphf",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="google-research-datasets/mbpp",
            revision="4bb6404",
            gcs_output_path=output_path,
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.md"],
        )
    ),
)
mbpp_raw_path = os.path.join(_mbpp_base.output_path, "4bb6404/full")

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
    steps: list[StepSpec]
    tags: list[str] = dataclasses.field(default_factory=list)


# This creates a JSON file representing the auxiliary training data subset of MMLU
mmlu_aux_eval = StepSpec(
    name="evaluation/mmlu-eval-aux",
    hash_attrs={},
    deps=[mmlu_raw],
    fn=lambda output_path: hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="cais/mmlu",
            subsets=["all"],
            splits=["auxiliary_train"],
            input_path=mmlu_raw.output_path,
            hf_path="cais/mmlu",
            output_path=output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question",
            options_key="choices",
            answer_idx_key="answer",
            answer_labels=["A", "B", "C", "D"],
        )
    ),
)

# This creates one file per subject from MMLU, excluding the all and auxiliary training subsets
mmlu_subject_eval = StepSpec(
    name="evaluation/mmlu-eval-subject",
    hash_attrs={},
    deps=[mmlu_raw],
    fn=lambda output_path: hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="cais/mmlu",
            subsets=["*"],
            splits=["dev", "validation"],
            input_path=mmlu_raw.output_path,
            hf_path="cais/mmlu",
            output_path=output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question",
            options_key="choices",
            answer_idx_key="answer",
            answer_labels=["A", "B", "C", "D"],
            exclude_subsets=["all", "auxiliary_train"],
        )
    ),
)

# This creates a JSON file representing the train and validation data subset of boolq
boolq_eval = StepSpec(
    name="evaluation/boolq-eval",
    hash_attrs={},
    deps=[boolq_raw],
    fn=lambda output_path: hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="google/boolq",
            subsets=["*"],
            splits=["train", "validation"],
            input_path=boolq_raw.output_path,
            hf_path="google/boolq",
            output_path=output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question",
            answer_label_key="answer",
            answer_labels=[True, False],
            answer_text_ignore=True,
        )
    ),
)

# This creates a JSON file representing the training and validation data subset of piqa
piqa_eval = StepSpec(
    name="evaluation/piqa",
    hash_attrs={},
    deps=[piqa_raw],
    fn=lambda output_path: hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="ybisk/piqa",
            subsets=["*"],
            splits=["train", "validation"],
            input_path=piqa_raw.output_path,
            hf_path="ybisk/piqa",
            output_path=output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="goal",
            options_keys=["sol1", "sol2"],
            answer_idx_key="label",
            answer_labels=["1", "2"],
        )
    ),
)

# This creates a JSON file representing the training and validation data subset of winogrande_xl
winogrande_eval = StepSpec(
    name="evaluation/winogrande",
    hash_attrs={},
    deps=[winogrande_raw],
    fn=lambda output_path: hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="allenai/winogrande",
            subsets=["default"],
            splits=["train", "validation"],
            input_path=winogrande_raw.output_path,
            hf_path="allenai/winogrande",
            output_path=output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="sentence",
            options_keys=["option1", "option2"],
            answer_label_key="answer",
            answer_labels=["1", "2"],
        )
    ),
)

# This creates a JSON file representing the train and validation splits of ARC-Easy
arc_easy_eval = StepSpec(
    name="evaluation/arc-easy",
    hash_attrs={},
    deps=[arc_raw],
    fn=lambda output_path: hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="allenai/ai2_arc",
            subsets=["ARC-Easy"],
            splits=["train", "validation"],
            input_path=arc_raw.output_path,
            hf_path="allenai/ai2_arc",
            output_path=output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question",
            options_key="choices.text",
            answer_labels_key="choices.label",
            answer_label_key="answerKey",
        )
    ),
)

# This creates a JSON file representing the train and validation splits of ARC-Challenge
arc_challenge_eval = StepSpec(
    name="evaluation/arc-challenge",
    hash_attrs={},
    deps=[arc_raw],
    fn=lambda output_path: hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="allenai/ai2_arc",
            subsets=["ARC-Challenge"],
            splits=["train", "validation"],
            input_path=arc_raw.output_path,
            hf_path="allenai/ai2_arc",
            output_path=output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question",
            options_key="choices.text",
            answer_labels_key="choices.label",
            answer_label_key="answerKey",
        )
    ),
)

# This creates a JSON file for the train and validation subsets of OpenBookQA
openbookqa_eval = StepSpec(
    name="evaluation/openbookqa-eval",
    hash_attrs={},
    deps=[openbookqa_raw],
    fn=lambda output_path: hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="allenai/openbookqa",
            subsets=["main"],
            splits=["train", "validation"],
            input_path=openbookqa_raw.output_path,
            hf_path="allenai/openbookqa",
            output_path=output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question_stem",
            options_key="choices.text",
            answer_label_key="answerKey",
            answer_labels_key="choices.label",
        )
    ),
)

# This creates a JSON file representing the training and validation splits for hellaswag
hellaswag_eval = StepSpec(
    name="evaluation/hellaswag-eval",
    hash_attrs={},
    deps=[hellaswag_raw],
    fn=lambda output_path: hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="Rowan/hellaswag",
            subsets=["*"],
            splits=["train", "validation"],
            input_path=hellaswag_raw.output_path,
            hf_path="Rowan/hellaswag",
            output_path=output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="ctx",
            options_key="endings",
            answer_labels=["A", "B", "C", "D"],
            answer_idx_key="label",
        )
    ),
)

# This creates a JSON file representing the test and validation splits for MMLU-Pro
mmlu_pro_eval = StepSpec(
    name="evaluation/MMLU-Pro-eval",
    hash_attrs={},
    deps=[mmlu_pro_raw],
    fn=lambda output_path: hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="TIGER-Lab/MMLU-Pro",
            subsets=["*"],
            splits=["test", "validation"],
            input_path=mmlu_pro_raw.output_path,
            hf_path="TIGER-Lab/MMLU-Pro",
            output_path=output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="question",
            options_key="options",
            answer_labels=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"],
            answer_idx_key="answer_index",
        )
    ),
)

# This creates a JSON file representing the test and validation splits for openai_humaneval
humaneval_eval = StepSpec(
    name="evaluation/humaneval-eval",
    hash_attrs={},
    deps=[humaneval_raw],
    fn=lambda output_path: hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="openai/openai_humaneval",
            subsets=["*"],
            splits=["test"],
            input_path=humaneval_raw.output_path,
            hf_path="openai/openai_humaneval",
            output_path=output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="prompt",
            answer_text_key="canonical_solution",
        )
    ),
)

# This creates a JSON file representing the train, test, and validation splits for mbpp
mbpp_eval = StepSpec(
    name="evaluation/mbpp-eval",
    hash_attrs={},
    deps=[_mbpp_base],
    fn=lambda output_path: hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="google-research-datasets/mbpp",
            subsets=["*"],
            splits=["train", "test", "validation"],
            input_path=mbpp_raw_path,
            hf_path="google-research-datasets/mbpp",
            output_path=output_path,
            output_format=OutputFormatOptions("evaluation"),
            prompt_key="text",
            answer_text_key="code",
        )
    ),
)

eval_datasets = [
    # these tags are used to group datasets together for averaging
    EvalDataset("cais", "mmlu", [mmlu_aux_eval, mmlu_subject_eval]),
    EvalDataset("google", "boolq", [boolq_eval], ["core"]),
    EvalDataset("Rowan", "hellaswag", [hellaswag_eval], ["core"]),
    EvalDataset("ybisk", "piqa", [piqa_eval], ["core"]),
    EvalDataset("allenai", "winogrande", [winogrande_eval], ["core"]),
    EvalDataset("allenai", "ai2_arc_easy", [arc_easy_eval], ["core", "arc"]),
    EvalDataset("allenai", "ai2_arc_challenge", [arc_challenge_eval], ["core", "arc"]),
    EvalDataset("allenai", "openbookqa", [openbookqa_eval], ["core"]),
    EvalDataset("openai", "openai_humaneval", [humaneval_eval]),
    EvalDataset("google-research-datasets", "mbpp", [mbpp_eval]),
]


############################################################
# Convert mmlu to dolma format (i.e. JSON with "text" field)
# This is used as input to the decontamination pipeline so documents with MMLU content are removed
mmlu_convert_dolma = StepSpec(
    name="decontamination/mmlu-dolma",
    hash_attrs={},
    deps=[mmlu_raw],
    fn=lambda output_path: hf_dataset_to_jsonl(
        DatasetConversionConfig(
            dataset_name="cais/mmlu",
            subsets=["all"],
            splits=["dev", "test", "validation"],
            input_path=mmlu_raw.output_path,
            hf_path="cais/mmlu",
            output_path=output_path,
            output_format=OutputFormatOptions("decontamination"),
            prompt_key="question",
            options_key="choices",
            answer_idx_key="answer",
            answer_labels=["A", "B", "C", "D"],
        )
    ),
)

lingoly = StepSpec(
    name="raw/ambean/lingOly",
    hash_attrs={"hf_dataset_id": "ambean/lingOly", "revision": "6aff4c2"},
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="ambean/lingOly",
            revision="6aff4c2",
            gcs_output_path=output_path,
            wait_for_completion=True,
        )
    ),
)

# gsm8k raw
gsm8k_raw = StepSpec(
    name="raw/gsm8k",
    hash_attrs={"hf_dataset_id": "openai/gsm8k", "revision": "e53f048", "hf_urls_glob": ["**/*.parquet", "*.md"]},
    override_output_path="raw/gsm8k/mainhf",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="openai/gsm8k",
            revision="e53f048",
            gcs_output_path=output_path,
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.md"],
        )
    ),
)

# math dataset raw
math_raw = StepSpec(
    name="raw/hendrycks_math",
    hash_attrs={
        "hf_dataset_id": "EleutherAI/hendrycks_math",
        "revision": "21a5633",
        "hf_urls_glob": ["**/*.parquet", "*.md"],
    },
    override_output_path="raw/hendrycks/mathhf",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="EleutherAI/hendrycks_math",
            revision="21a5633",
            gcs_output_path=output_path,
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.md"],
        )
    ),
)

# truthful_qa raw
truthful_qa_raw = StepSpec(
    name="raw/truthful_qa",
    hash_attrs={
        "hf_dataset_id": "truthfulqa/truthful_qa",
        "revision": "741b827",
        "hf_urls_glob": ["**/*.parquet", "*.md"],
    },
    override_output_path="raw/truthful_qa/multiple_choicehf",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="truthfulqa/truthful_qa",
            revision="741b827",
            gcs_output_path=output_path,
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.md"],
        )
    ),
)

# bbh raw
bbh_raw = StepSpec(
    name="raw/bbh",
    hash_attrs={"hf_dataset_id": "SaylorTwift/bbh", "revision": "b5306be", "hf_urls_glob": ["**/*.parquet", "*.md"]},
    override_output_path="raw/SaylorTwift/bbhhf",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="SaylorTwift/bbh",
            revision="b5306be",
            gcs_output_path=output_path,
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.md"],
        )
    ),
)

# gpqa raw
gpqa_raw = StepSpec(
    name="raw/gpqa",
    hash_attrs={"hf_dataset_id": "Idavidrein/gpqa", "revision": "90b8e5b", "hf_urls_glob": ["**/*.csv", "*.csv"]},
    override_output_path="raw/Idavidrein/gpqa",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="Idavidrein/gpqa",
            revision="90b8e5b",
            gcs_output_path=output_path,
            wait_for_completion=True,
            hf_urls_glob=["**/*.csv", "*.csv"],
        )
    ),
)

# instruction-following raw
instruction_following_raw = StepSpec(
    name="raw/instruction_following_eval",
    hash_attrs={
        "hf_dataset_id": "wis-k/instruction-following-eval",
        "revision": "5a5661c",
        "hf_urls_glob": ["**/*.jsonl", "*.jsonl"],
    },
    override_output_path="raw/wis-k/instruction-following-evalhf",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="wis-k/instruction-following-eval",
            revision="5a5661c",
            gcs_output_path=output_path,
            wait_for_completion=True,
            hf_urls_glob=["**/*.jsonl", "*.jsonl"],
        )
    ),
)

# musr raw
musr_raw = StepSpec(
    name="raw/musr",
    hash_attrs={
        "hf_dataset_id": "WillHeld/MuSRDecontam",
        "revision": "39b4f56",
        "hf_urls_glob": ["**/*.parquet", "*.parquet"],
    },
    override_output_path="raw/WillHeld/MuSRDecontamhf",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="WillHeld/MuSRDecontam",
            revision="39b4f56",
            gcs_output_path=output_path,
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.parquet"],
        )
    ),
)

# winograd WSC raw
winograd_wsc_raw = StepSpec(
    name="raw/winograd_wsc",
    hash_attrs={
        "hf_dataset_id": "marcov/winograd_wsc_wsc273_promptsource",
        "revision": "63befd8",
        "hf_urls_glob": ["**/*.parquet", "*.parquet"],
    },
    override_output_path="raw/marcov/winograd_wsc_wsc273_promptsourcehf",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="marcov/winograd_wsc_wsc273_promptsource",
            revision="63befd8",
            gcs_output_path=output_path,
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.parquet"],
        )
    ),
)

# commonsense_qa raw
commonsense_qa_raw = StepSpec(
    name="raw/commonsense_qa",
    hash_attrs={
        "hf_dataset_id": "tau/commonsense_qa",
        "revision": "94630fe",
        "hf_urls_glob": ["**/*.parquet", "*.parquet"],
    },
    override_output_path="raw/tau/commonsense_qahf",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="tau/commonsense_qa",
            revision="94630fe",
            gcs_output_path=output_path,
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.parquet"],
        )
    ),
)

# lambada_openai raw
lambada_openai_raw = StepSpec(
    name="raw/lambada_openai",
    hash_attrs={
        "hf_dataset_id": "EleutherAI/lambada_openai",
        "revision": "879e19a",
        "hf_urls_glob": ["**/*.jsonl", "*.jsonl"],
    },
    override_output_path="raw/EleutherAI/lambada_openaihf",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="EleutherAI/lambada_openai",
            revision="879e19a",
            gcs_output_path=output_path,
            wait_for_completion=True,
            hf_urls_glob=["**/*.jsonl", "*.jsonl"],
        )
    ),
)

# Alternative PIQA variant used only for contamination analysis
piqa_baber_raw = StepSpec(
    name="raw/baber/piqa",
    hash_attrs={"hf_dataset_id": "baber/piqa", "revision": "142f6d7", "hf_urls_glob": ["**/*.parquet", "*.parquet"]},
    override_output_path="raw/baber/piqahf",
    fn=lambda output_path: download_hf(
        DownloadConfig(
            hf_dataset_id="baber/piqa",
            revision="142f6d7",
            gcs_output_path=output_path,
            wait_for_completion=True,
            hf_urls_glob=["**/*.parquet", "*.parquet"],
        )
    ),
)

############################################################

if __name__ == "__main__":
    StepRunner().run(
        [
            mmlu_convert_dolma,
            *[step for ds in eval_datasets for step in ds.steps],
        ]
    )
