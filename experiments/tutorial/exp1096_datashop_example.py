from experiments.datashop.datashop_datasets import datashop_dclm_tutorial_subset
from experiments.datashop.datashop_runner import DatashopRunner, DatashopRunnerConfig
from experiments.exp939_finemath import FINEMATH_DATA_FILTER_PROMPT

datashop_runner = DatashopRunner(
    DatashopRunnerConfig(
        experiment_name="datashop-tutorial",
        annotator_model_name="Llama-3.3-70B-Instruct",
        pretraining_data_path=datashop_dclm_tutorial_subset,
        annotator_data_path=datashop_dclm_tutorial_subset,
        data_filter_prompt=FINEMATH_DATA_FILTER_PROMPT,
        dataset_output_processor_config_kwargs={"processor_type": "finalscore0-5"},
    )
)

if __name__ == "__main__":
    datashop_runner.run_eval_cluster_steps()
