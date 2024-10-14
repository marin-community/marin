from marin.execution.executor import ExecutorStep, executor_main
from operations.huggingface.qa.raw2json import raw2json, DatasetConversionConfig, OutputFormatOptions

"""
Downloads the following datasets
- mmlu
and saves internal eval and decontamination formats to GCS.
"""
############################################################
# Convert mmlu to evaluation format
mmlu_convert_eval_aux = ExecutorStep(
    name="evaluation/mmlu-eval_aux",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="cais/mmlu",
        subsets=["all"],
        splits=["auxiliary_train"],
        input_path="cais/mmlu",
        hf_path="cais/mmlu",
        output_path="gs://marin-us-central2/eval/mmlu",
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
        input_path="cais/mmlu",
        hf_path="cais/mmlu",
        output_path="gs://marin-us-central2/eval/mmlu",
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B", "C", "D"],
        exclude_subsets=["all"]
    ),
)

# Convert mmlu to dolma format
mmlu_convert_dolma = ExecutorStep(
    name="evaluation/mmlu-dolma",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="cais/mmlu",
        subsets=["all"],
        splits=["dev", "test", "validation"],
        input_path="cais/mmlu",
        hf_path="cais/mmlu",
        output_path="gs://marin-us-central2/dolma/mmlu",
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
            mmlu_convert_eval_aux,
            mmlu_convert_eval_subject,
            mmlu_convert_dolma,
        ]
    )
