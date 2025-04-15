from experiments.datashop.datashop_datasets import datashop_dclm_annotation_subset, datashop_dclm_pretraining_subset
from experiments.datashop.datashop_runner import DatashopRunner, DatashopRunnerConfig
from experiments.datashop.default_configs import default_quality_filter_train_config_kwargs
from marin.execution.executor import executor_main

FINEMATH_3_POINT_DATA_FILTER_PROMPT = """
Evaluate the following text extract for its potential usefulness for studying mathematics up to high school and early undergraduate levels. Use the following 3-point scoring system described below. Points are accumulated based on the satisfaction of
each criterion:
- Add 1 point if the extract contains some mathematical content, even if it’s not very useful for studying or is an academic
paper that is too advanced.
- Add another point if the extract demonstrates logical reasoning in a mathematical context, even if it lacks step-by-step
explanations or is too advanced.
- Award a third point if the extract is at an appropriate level (up to high school and early undergraduate levels) and contains
clear mathematical deductions and step-by-step solutions to mathematical problems.
Question-answer formats (e.g., from educational websites or forums) are acceptable if they meet the criteria. Ignore any
formatting errors or missing equations and make assumptions based on the overall content.
The text extract:
{example}
After examining the extract:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Final score: <total points>"
"""  # noqa: E501, RUF001

quality_filter_train_config_kwargs = default_quality_filter_train_config_kwargs.copy()
quality_filter_train_config_kwargs["training_config"].max_label = 3

datashop_runner = DatashopRunner(
    DatashopRunnerConfig(
        experiment_name="finemath-cascade",
        annotator_model_name="Llama-3.3-70B-Instruct",
        pretraining_data_path=datashop_dclm_pretraining_subset,
        annotator_data_path=datashop_dclm_annotation_subset,
        data_filter_prompt=FINEMATH_3_POINT_DATA_FILTER_PROMPT,
        dataset_output_processor_config_kwargs={"processor_type": "finalscore0-5"},
        quality_train_config_kwargs=quality_filter_train_config_kwargs,
    )
)

# TODO(chris): After obtaining the 3-point filtering model,
# 1. Filter the pretraining data pool to 3+ examples.
# 2. Annotate with 5 point scale.
# 3. Filter rest of the documents.

if __name__ == "__main__":
    executor_main([datashop_runner.filtered_documents])
