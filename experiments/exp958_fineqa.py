from experiments.datashop.datashop_datasets import datashop_dclm_annotation_subset, datashop_dclm_pretraining_subset
from experiments.datashop.datashop_runner import DatashopRunner, DatashopRunnerConfig

FINEQA_DATA_FILTER_PROMPT = """
Evaluate the following text extract for its potential usefulness for training a language model on question answer format. Use the following 5-point scoring system described below. Points are accumulated based on the satisfaction of
each criterion:

- Add 1 point if the extract contains some content related to a person asking a question and another person answering them.
- Add another point if the extract includes a multi-turn conversation between two people.
- Award a third point if the extract is an educational conversation.
- Grant a fourth point if the extract is a clearly formatted multi-turn educational conversation between two people that would show up
on a user forum, e.g. Quora, StackExchange, or Reddit.
- Give a fifth point if the extract is outstanding in its educational value for teaching and studying. It should include very detailed and easy to follow explanations.
The text extract:
{example}
After examining the extract:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: Final score: <total points>.
"""  # noqa: E501

datashop_runner = DatashopRunner(
    DatashopRunnerConfig(
        experiment_name="fineqa",
        annotator_model_name="Llama-3.3-70B-Instruct",
        pretraining_data_path=datashop_dclm_pretraining_subset,
        annotator_data_path=datashop_dclm_annotation_subset,
        data_filter_prompt=FINEQA_DATA_FILTER_PROMPT,
        dataset_output_processor_config_kwargs={"processor_type": "finalscore0-5"},
    )
)

if __name__ == "__main__":
    datashop_runner.run_eval_cluster_steps()
