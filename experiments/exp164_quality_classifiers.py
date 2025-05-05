"""
Train quality classifiers on different subsets similar to DCLM's classifiers.
https://github.com/stanford-crfm/marin/issues/164
TODO: apply these quality classifiers on FineWeb (or DCLM, but that's larger), train models.
"""

import dataclasses

from marin.classifiers.utils import DatasetConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.processing.classification.fasttext.train_fasttext import (
    TrainFasttextClassifierConfig,
    train,
)
from operations.transform.fasttext.transform import TransformFasttextToDolmaConfig
from operations.transform.fasttext.transform import main as fasttext_to_dolma_format

dclm_negative_examples_in_dolma_format = ExecutorStep(
    name="documents/dclm_negative_examples",
    fn=fasttext_to_dolma_format,
    config=TransformFasttextToDolmaConfig(
        input_path=versioned("gs://marin-us-central2/documents/dclm/negative_examples.txt"),
        output_path=this_output_path(),
        source="dclm",
    ),
)

dclm_oh_100k_in_dolma_format = ExecutorStep(
    name="documents/dclm_oh_100k",
    fn=fasttext_to_dolma_format,
    config=TransformFasttextToDolmaConfig(
        input_path=versioned("gs://marin-us-central2/documents/dclm/oh_100k.txt"),
        output_path=this_output_path(),
        source="dclm",
    ),
)

dclm_eli5_200k_rw_200k = ExecutorStep(
    name="classifiers/dclm_eli5_200k_rw_200k",
    fn=train,
    config=TrainFasttextClassifierConfig(
        datasets=[
            DatasetConfig(
                input_doc_path=versioned("gs://marin-us-central2/documents/dclm/explainlikeimfive"),
                label="hq",
                sampling_rate=1.0,
                max_sample_size=versioned(200000),
            ),
            DatasetConfig(
                input_doc_path=output_path_of(dclm_negative_examples_in_dolma_format),
                label="lq",
                sampling_rate=1.0,
                max_sample_size=None,
            ),
        ],
        output_path=this_output_path(),
        fasttext_args={"lr": versioned(0.1), "thread": 4, "wordNgrams": 2},
        val_frac=versioned(0.0),
        seed=versioned(0),
    ),
)

dclm_eli5_100k_oh_100k_rw_200k = ExecutorStep(
    name="classifiers/dclm_eli5_100k_oh_100k_rw_200k",
    fn=train,
    config=TrainFasttextClassifierConfig(
        datasets=[
            DatasetConfig(
                input_doc_path=versioned("gs://marin-us-central2/documents/dclm/explainlikeimfive"),
                label="hq",
                sampling_rate=1.0,
                max_sample_size=versioned(100000),
            ),
            DatasetConfig(
                input_doc_path=output_path_of(dclm_oh_100k_in_dolma_format),
                label="hq",
                sampling_rate=1.0,
                max_sample_size=versioned(100000),
            ),
            DatasetConfig(
                input_doc_path=output_path_of(dclm_negative_examples_in_dolma_format),
                label="lq",
                sampling_rate=1.0,
                max_sample_size=None,
            ),
        ],
        output_path=this_output_path(),
        fasttext_args={"lr": versioned(0.1), "thread": 4, "wordNgrams": 2},
        val_frac=versioned(0.0),
        seed=versioned(0),
    ),
)


def get_classifier_with_different_seed(classifier_step: ExecutorStep, seed: int):
    new_classifier_step = dataclasses.replace(
        classifier_step,
        config=dataclasses.replace(classifier_step.config, seed=versioned(seed)),
    )
    new_classifier_step = dataclasses.replace(new_classifier_step, name=f"{new_classifier_step.name}-seed-{seed}")
    return new_classifier_step


dclm_eli5_100k_oh_100k_rw_200k_seed_1 = get_classifier_with_different_seed(dclm_eli5_100k_oh_100k_rw_200k, 1)
dclm_eli5_100k_oh_100k_rw_200k_seed_2 = get_classifier_with_different_seed(dclm_eli5_100k_oh_100k_rw_200k, 2)

if __name__ == "__main__":
    executor_main(
        steps=[
            dclm_negative_examples_in_dolma_format,
            dclm_oh_100k_in_dolma_format,
            dclm_eli5_200k_rw_200k,
            dclm_eli5_100k_oh_100k_rw_200k,
            dclm_eli5_100k_oh_100k_rw_200k_seed_1,
            dclm_eli5_100k_oh_100k_rw_200k_seed_2,
        ]
    )
