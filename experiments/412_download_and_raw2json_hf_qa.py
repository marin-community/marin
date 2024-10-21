from marin.execution.executor import ExecutorStep, executor_main, this_output_path, output_path_of, versioned
from operations.raw2json.huggingface.qa.raw2json import DatasetConversionConfig, OutputFormatOptions, raw2json
from operations.download.huggingface.download import DownloadConfig, download

"""
Downloads the following datasets
- mmlu
"""
############################################################
# download mmlu dataset
mmlu_download_step = ExecutorStep(
    name="raw/cais/mmlu",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="cais/mmlu",
        revision=versioned("c30699e"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="gs://marin-us-central2/raw/cais/mmlu", 
)

mmlu_download_step_output_path = mmlu_download_step.cd("c30699e/huggingface.co/datasets/cais/mmlu/resolve/c30699e")

"""
Converts raw to JSON for:
- mmlu
"""
############################################################
# Convert mmlu to evaluation format (i.e. JSON with "prompt", "response" fields)
# This is the input for internal evaluation which measures PPL model gives to correct responses to prompts
mmlu_convert_eval_aux = ExecutorStep(
    name="evaluation/mmlu-eval_aux",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="cais/mmlu",
        subsets=["all"],
        splits=["auxiliary_train"],
        input_path="gs://marin-us-central2/raw/cais/mmlu/c30699e/huggingface.co/datasets/cais/mmlu/resolve/c30699e",
        hf_path="cais/mmlu",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B", "C", "D"],
    ),
)

mmlu_convert_eval_subject = ExecutorStep(
    name="evaluation/mmlu-eval-subject",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="cais/mmlu",
        subsets=["*"],
        splits=["dev", "validation"],
        input_path="gs://marin-us-central2/raw/cais/mmlu/c30699e/huggingface.co/datasets/cais/mmlu/resolve/c30699e",
        hf_path="cais/mmlu",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B", "C", "D"],
        exclude_subsets=["all"],
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
        input_path=mmlu_download_step_output_path,
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
            #mmlu_convert_eval_aux,
            #mmlu_convert_eval_subject,
            mmlu_convert_dolma,
        ]
    )
