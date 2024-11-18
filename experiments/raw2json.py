from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from operations.download.huggingface.download import DownloadConfig
from operations.download.huggingface.download_hf import download_hf
from operations.raw2json.huggingface.qa.raw2json import DatasetConversionConfig, OutputFormatOptions, raw2json

"""
Downloads the following datasets
- mmlu
- boolq
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
        hf_url_glob="**/*.parquet",
    ),
    override_output_path="gs://marin-us-central2/raw/cais/mmlu",
)

# download boolq dataset
boolq_download_step = ExecutorStep(
    name="raw/google/boolq",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="google/boolq",
        revision=versioned("35b264d"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_url_glob="**/*.parquet",
    ),
    override_output_path="gs://marin-us-central2/raw/google/boolq",
)


"""
Converts raw to JSON for:
- mmlu
- boolq
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

# This creates a JSON file representing the train and validation data subset of boolq
boolq_convert_eval = ExecutorStep(
    name="evaluation/boolq-eval",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="google/boolq",
        subsets=["*"],
        splits=["train", "validation"],
        input_path=boolq_download_step,
        hf_path="google/boolq",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="question",
        answer_label_key="answer",
        answer_labels=[True, False],
        answer_text_ignore=True,
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
            boolq_download_step,
            boolq_convert_eval,
        ]
    )
