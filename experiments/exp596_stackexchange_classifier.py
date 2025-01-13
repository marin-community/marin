from experiments.exp102_classifier_ablations import ExperimentConfig, create_steps
from experiments.exp164_quality_classifiers import dclm_negative_examples_in_dolma_format
from marin.classifiers.utils import DatasetConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.processing.classification.fasttext.train_fasttext import (
    TrainFasttextClassifierConfig,
    train,
)
from operations.transform.stackexchange.filter_stackexchange import FilterStackExchangeConfig, filter_stackexchange

stackexchange_qa_vote_geq_5_rm_duplicate = ExecutorStep(
    name="documents/stackexchange-qa-vote-geq-5-rm-duplicate",
    fn=filter_stackexchange,
    config=FilterStackExchangeConfig(
        # Check out operations/download/stackexchange/README.md for how to download this dataset
        input_path="gs://marin-us-central2/documents/stackexchange/v2024-04-02/md-qa-pair/",
        output_path=this_output_path(),
        min_vote_threshold=8,
        remove_duplicate_questions=True,
    ),
)

stackexchange_qa_vote_geq_5_rm_duplicate_200k_rw_200k = ExecutorStep(
    name="classifiers/stackexchange-qa-vote-geq-5-rm-duplicate-200k-rw-200k",
    fn=train,
    config=TrainFasttextClassifierConfig(
        datasets=[
            DatasetConfig(
                input_doc_path=output_path_of(stackexchange_qa_vote_geq_5_rm_duplicate),
                label="hq",
                sampling_rate=1.0,
                max_sample_size=versioned(200000),
            ),
            DatasetConfig(
                input_doc_path=output_path_of(dclm_negative_examples_in_dolma_format),
                label="lq",
                sampling_rate=1.0,
                max_sample_size=versioned(200000),
            ),
        ],
        output_path=this_output_path(),
        fasttext_args={"lr": versioned(0.1), "thread": 4, "wordNgrams": 2},
        val_frac=versioned(0.0),
        seed=versioned(0),
    ),
    pip_dependency_groups=["fasttext"],
)

stackexchange_experiment_config = ExperimentConfig(
    experiment_name="stackexchange-qa-vote-geq-5-rm-duplicate-200k-rw-200k",
    quality_classifier_model_path=stackexchange_qa_vote_geq_5_rm_duplicate_200k_rw_200k,
)

if __name__ == "__main__":
    steps = create_steps(stackexchange_experiment_config)
    executor_main(steps)
