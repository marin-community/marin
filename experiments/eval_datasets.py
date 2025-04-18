import dataclasses

from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from operations.download.huggingface.download import DownloadConfig
from operations.download.huggingface.download_hf import download_hf
from operations.raw2json.huggingface.qa.raw2json import DatasetConversionConfig, OutputFormatOptions, raw2json

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
mmlu_raw = ExecutorStep(
    name="raw/cais/mmlu",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="cais/mmlu",
        revision=versioned("c30699e"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
    override_output_path="raw/cais/mmluhf",
)

# download boolq dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
boolq_raw = ExecutorStep(
    name="raw/google/boolq",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="google/boolq",
        revision=versioned("35b264d"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet"],
    ),
    override_output_path="raw/google/boolqhf",
).cd("35b264d")

# download hellaswag dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
hellaswag_raw = ExecutorStep(
    name="raw/Rowan/hellaswag",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="Rowan/hellaswag",
        revision=versioned("50441ce"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet"],
    ),
    override_output_path="raw/Rowan/hellaswaghf",
).cd("50441ce")

# download piqa dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
piqa_raw = ExecutorStep(
    name="raw/ybisk/piqa",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="ybisk/piqa",
        revision=versioned("142c512"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet"],
    ),
    override_output_path="raw/ybisk/piqahf",
).cd("142c512")

# download winogrande dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
winogrande_raw = ExecutorStep(
    name="raw/allenai/winogrande",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="allenai/winogrande",
        revision=versioned("ebf71e3"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["winogrande_xl/**/*.parquet"],
    ),
    override_output_path="raw/allenai/winograndehf",
).cd("ebf71e3")

# download arc dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
arc_raw = ExecutorStep(
    name="raw/allenai/ai2_arc",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="allenai/ai2_arc",
        revision=versioned("210d026"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
    override_output_path="raw/allenai/ai2_archf",
).cd("210d026")

# download openbookqa dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
openbookqa_raw = ExecutorStep(
    name="raw/allenai/openbookqa",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="allenai/openbookqa",
        revision=versioned("388097e"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
    override_output_path="raw/allenai/openbookqahf",
).cd("388097e")

# download MMLU-Pro dataset
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
mmlu_pro_raw = ExecutorStep(
    name="raw/TIGER-Lab/MMLU-Pro",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="TIGER-Lab/MMLU-Pro",
        revision=versioned("3373e0b"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
    override_output_path="raw/TIGER-Lab/MMLU-Prohf",
).cd("3373e0b")

# download openai_humaneval
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
humaneval_raw = ExecutorStep(
    name="raw/openai/openai_humaneval",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="openai/openai_humaneval",
        revision=versioned("7dce605"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
    override_output_path="gs://marin-us-central2/raw/openai/openai_humanevalhf",
).cd("7dce605")

# download mbpp
# TODO: Earlier datasets were stored in gcs_output_path/<revision> instead of gcs_output_path.
#   Migrate the dataset and cd can be removed.
mbpp_raw = ExecutorStep(
    name="raw/google-research-datasets/mbpp",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="google-research-datasets/mbpp",
        revision=versioned("4bb6404"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
    override_output_path="raw/google-research-datasets/mbpphf",
).cd("4bb6404/full")


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
mmlu_aux_eval = ExecutorStep(
    name="evaluation/mmlu-eval-aux",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="cais/mmlu",
        subsets=["all"],
        splits=["auxiliary_train"],
        input_path=mmlu_raw,
        hf_path="cais/mmlu",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B", "C", "D"],
    ),
)

# This creates one file per subject from MMLU, excluding the all and auxiliary training subsets
mmlu_subject_eval = ExecutorStep(
    name="evaluation/mmlu-eval-subject",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="cais/mmlu",
        subsets=["*"],
        splits=["dev", "validation"],
        input_path=mmlu_raw,
        hf_path="cais/mmlu",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B", "C", "D"],
        exclude_subsets=["all", "auxiliary_train"],
    ),
)

# This creates a JSON file representing the train and validation data subset of boolq
boolq_eval = ExecutorStep(
    name="evaluation/boolq-eval",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="google/boolq",
        subsets=["*"],
        splits=["train", "validation"],
        input_path=boolq_raw,
        hf_path="google/boolq",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="question",
        answer_label_key="answer",
        answer_labels=[True, False],
        answer_text_ignore=True,
    ),
)

# This creates a JSON file representing the training and validation data subset of piqa
piqa_eval = ExecutorStep(
    name="evaluation/piqa",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="ybisk/piqa",
        subsets=["*"],
        splits=["train", "validation"],
        input_path=piqa_raw,
        hf_path="ybisk/piqa",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="goal",
        options_keys=["sol1", "sol2"],
        answer_idx_key="label",
        answer_labels=["1", "2"],
    ),
)

# This creates a JSON file representing the training and validation data subset of winogrande_xl
winogrande_eval = ExecutorStep(
    name="evaluation/winogrande",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="allenai/winogrande",
        subsets=["default"],
        splits=["train", "validation"],
        input_path=winogrande_raw,
        hf_path="allenai/winogrande",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="sentence",
        options_keys=["option1", "option2"],
        answer_label_key="answer",
        answer_labels=["1", "2"],
    ),
)

# This creates a JSON file representing the train and validation splits of ARC-Easy
arc_easy_eval = ExecutorStep(
    name="evaluation/arc-easy",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="allenai/ai2_arc",
        subsets=["ARC-Easy"],
        splits=["train", "validation"],
        input_path=arc_raw,
        hf_path="allenai/ai2_arc",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="question",
        options_key="choices.text",
        answer_labels_key="choices.label",
        answer_label_key="answerKey",
    ),
)

# This creates a JSON file representing the train and validation splits of ARC-Challenge
arc_challenge_eval = ExecutorStep(
    name="evaluation/arc-challenge",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="allenai/ai2_arc",
        subsets=["ARC-Challenge"],
        splits=["train", "validation"],
        input_path=arc_raw,
        hf_path="allenai/ai2_arc",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="question",
        options_key="choices.text",
        answer_labels_key="choices.label",
        answer_label_key="answerKey",
    ),
)

# This creates a JSON file for the train and validation subsets of OpenBookQA
openbookqa_eval = ExecutorStep(
    name="evaluation/openbookqa-eval",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="allenai/openbookqa",
        subsets=["main"],
        splits=["train", "validation"],
        input_path=openbookqa_raw,
        hf_path="allenai/openbookqa",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="question_stem",
        options_key="choices.text",
        answer_label_key="answerKey",
        answer_labels_key="choices.label",
    ),
)

# This creates a JSON file representing the training and validation splits for hellaswag
hellaswag_eval = ExecutorStep(
    name="evaluation/hellaswag-eval",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="Rowan/hellaswag",
        subsets=["*"],
        splits=["train", "validation"],
        input_path=hellaswag_raw,
        hf_path="Rowan/hellaswag",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="ctx",
        options_key="endings",
        answer_labels=["A", "B", "C", "D"],
        answer_idx_key="label",
    ),
)

# This creates a JSON file representing the test and validation splits for MMLU-Pro
mmlu_pro_eval = ExecutorStep(
    name="evaluation/MMLU-Pro-eval",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="TIGER-Lab/MMLU-Pro",
        subsets=["*"],
        splits=["test", "validation"],
        input_path=mmlu_pro_raw,
        hf_path="TIGER-Lab/MMLU-Pro",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="question",
        options_key="options",
        answer_labels=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"],
        answer_idx_key="answer_index",
    ),
)

# This creates a JSON file representing the test and validation splits for openai_humaneval
humaneval_eval = ExecutorStep(
    name="evaluation/humaneval-eval",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="openai/openai_humaneval",
        subsets=["*"],
        splits=["test"],
        input_path=humaneval_raw,
        hf_path="openai/openai_humaneval",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="prompt",
        answer_text_key="canonical_solution",
    ),
)

# This creates a JSON file representing the train, test, and validation splits for mbpp
mbpp_eval = ExecutorStep(
    name="evaluation/mbpp-eval",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="google-research-datasets/mbpp",
        subsets=["*"],
        splits=["train", "test", "validation"],
        input_path=mbpp_raw,
        hf_path="google-research-datasets/mbpp",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="text",
        answer_text_key="code",
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
mmlu_convert_dolma = ExecutorStep(
    name="decontamination/mmlu-dolma",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="cais/mmlu",
        subsets=["all"],
        splits=["dev", "test", "validation"],
        input_path=mmlu_raw,
        hf_path="cais/mmlu",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B", "C", "D"],
    ),
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            mmlu_convert_dolma,
            *[step for ds in eval_datasets for step in ds.steps],
        ]
    )
