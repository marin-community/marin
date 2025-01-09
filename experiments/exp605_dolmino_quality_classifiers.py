"""
An experiment to train quality classifiers from high quality data from Dolmino:
https://huggingface.co/datasets/allenai/dolmino-mix-1124

Dolmino is a dataset of high quality data sources including wikipedia, pes2o, stackexchange,
math synthetic data, and many more. In this file, we experiment with training and running quality classifiers
from the Wikipedia and Pes2o splits of the dataset.

We use DCLM's negative examples which are 200k Refined Web examples from
this issue: https://github.com/stanford-crfm/marin/issues/164
"""

from experiments.exp164_quality_classifiers import dclm_negative_examples_in_dolma_format
from experiments.pretraining_datasets import dolmino
from experiments.quality_classifier_experiment_utils import ExperimentConfig, create_steps
from marin.classifiers.utils import DatasetConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.processing.classification.fasttext.train_fasttext import (
    TrainFasttextClassifierConfig,
    train,
)
from operations.transform.dolmino.filter_dolmino import FilterDolminoConfig, filter_dolmino

wiki_longer_than_256_chars = ExecutorStep(
    name="documents/wiki-longer-than-256-chars",
    fn=filter_dolmino,
    config=FilterDolminoConfig(
        input_path=dolmino,
        output_path=this_output_path(),
        split="wiki",
        min_length=256,
    ),
)

wiki_200k_rw_200k = ExecutorStep(
    name="classifiers/wiki-200k-rw-200k",
    fn=train,
    config=TrainFasttextClassifierConfig(
        datasets=[
            DatasetConfig(
                input_doc_path=output_path_of(wiki_longer_than_256_chars),
                label="hq",
                sampling_rate=0.10,  # The dataset is about 3.24M rows, we only need 200k samples
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

wiki_experiment_config = ExperimentConfig(
    experiment_name="wiki-200k-rw-200k",
    quality_classifier_model_path=wiki_200k_rw_200k,
)

wiki_train_steps = create_steps(wiki_experiment_config)

pes2o_dolma_format = ExecutorStep(
    name="documents/pes2o",
    fn=filter_dolmino,
    config=FilterDolminoConfig(
        input_path=dolmino,
        output_path=this_output_path(),
        split="pes2o",
    ),
)

pes2o_200k_rw_200k = ExecutorStep(
    name="classifiers/pes2o-200k-rw-200k",
    fn=train,
    config=TrainFasttextClassifierConfig(
        datasets=[
            DatasetConfig(
                input_doc_path=output_path_of(pes2o_dolma_format),
                label="hq",
                sampling_rate=0.005,  # The dataset is about 95.9M rows, we only need 200k samples
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

pes2o_experiment_config = ExperimentConfig(
    experiment_name="pes2o-200k-rw-200k",
    quality_classifier_model_path=pes2o_200k_rw_200k,
)

pes2o_train_steps = create_steps(pes2o_experiment_config)

if __name__ == "__main__":
    wiki_steps = create_steps(wiki_experiment_config)
    pes2o_steps = create_steps(pes2o_experiment_config)
    steps = wiki_steps + pes2o_steps
    executor_main(steps=steps)
