from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from operations.download.huggingface.download import DownloadConfig
from operations.download.huggingface.download_hf import download_hf
from operations.raw2json.huggingface.qa.raw2json import DatasetConversionConfig, OutputFormatOptions, raw2json

"""
Downloads the following datasets
- mmlu
- piqa
- winogrande
"""
############################################################
# download mmlu dataset
mmlu_download_step = ExecutorStep(
    name="raw/cais/mmlu",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="cais/mmlu",
        revision=versioned("c30699e"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet"],
    ),
    override_output_path="gs://marin-us-central2/raw/cais/mmlu",
).cd("c30699e")

# download piqa dataset
piqa_download_step = ExecutorStep(
    name="raw/ybisk/piqa",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="ybisk/piqa",
        revision=versioned("142c512"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet"],
    ),
    override_output_path="gs://marin-us-central2/raw/ybisk/piqa",
).cd("142c512")

# download winogrande dataset
winogrande_download_step = ExecutorStep(
    name="raw/allenai/winogrande",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="allenai/winogrande",
        revision=versioned("ebf71e3"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["winogrande_xl/*.parquet"],
    ),
    override_output_path="gs://marin-us-central2/raw/allenai/winogrande",
).cdd("ebf71e3")

"""
Converts raw to JSON for:
- mmlu
- piqa
- winogrande
"""
############################################################
# Convert mmlu to evaluation format (i.e. JSON with "prompt", "response" fields)
# This is the input for internal evaluation which measures PPL model gives to correct responses to prompts

# This creates a JSON file representing the auxiliary training data subset of MMLU
mmlu_convert_eval_aux = ExecutorStep(
    name="evaluation/mmlu-eval-aux",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="cais/mmlu",
        subsets=["all"],
        splits=["auxiliary_train"],
        input_path=mmlu_download_step,
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
mmlu_convert_eval_subject = ExecutorStep(
    name="evaluation/mmlu-eval-subject",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="cais/mmlu",
        subsets=["*"],
        splits=["dev", "validation"],
        input_path=mmlu_download_step,
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

# This creates a JSON file representing the training and validation data subset of piqa
piqa_convert_eval = ExecutorStep(
    name="evaluation/piqa",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="ybisk/piqa",
        subsets=["*"],
        splits=["train", "validation"],
        input_path=piqa_download_step,
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
winogrande_convert_eval = ExecutorStep(
    name="evaluation/winogrande",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="allenai/winogrande",
        subsets=["default"],
        splits=["train", "validation"],
        input_path=winogrande_download_step,
        hf_path="allenai/winogrande",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="sentence",
        options_keys=["option1", "option2"],
        answer_label_key="answer",
        answer_labels=["1", "2"],
    ),
)
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
        input_path=mmlu_download_step,
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
            mmlu_download_step,
            mmlu_convert_eval_aux,
            mmlu_convert_eval_subject,
            mmlu_convert_dolma,
            piqa_download_step,
            piqa_convert_eval,
            winogrande_download_step,
            winogrande_convert_eval,
        ]
    )
