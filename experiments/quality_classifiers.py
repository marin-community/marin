"""
Train quality classifiers on different subsets similar to DCLM's classifiers.
https://github.com/stanford-crfm/marin/issues/164
TODO: apply these quality classifiers on FineWeb (or DCLM, but that's larger), train models.
"""

import dataclasses

from experiments.instruction_datasets import get_directory_friendly_dataset_name, get_instruction_dataset
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.processing.classification.fasttext.train_fasttext import (
    DatasetCurationConfig,
    TrainFasttextClassifierConfig,
    train,
)
from marin.processing.classification.types import DatasetFormat
from operations.transform.conversation.conversation_to_dolma import ConversationToDolmaConfig, process_dataset

openhermes_in_dolma_format = ExecutorStep(
    name=f"documents/{get_directory_friendly_dataset_name('teknium/OpenHermes-2.5')}",
    fn=process_dataset,
    config=ConversationToDolmaConfig(
        input_path=output_path_of(get_instruction_dataset("teknium/OpenHermes-2.5")),
        output_path=this_output_path("text"),
    ),
)

dclm_eli5_200k_rw_200k = ExecutorStep(
    name="classifiers/dclm_eli5_200k_rw_200k",
    fn=train,
    config=TrainFasttextClassifierConfig(
        input_doc_paths=[
            DatasetCurationConfig(
                input_doc_path=versioned("gs://marin-us-central2/documents/dclm/explainlikeimfive"),
                label="hq",
                absolute_sampling_rate=versioned(200000),
                format=DatasetFormat.DOLMA_FORMATTED_JSONL,
            ),
            DatasetCurationConfig(
                input_doc_path=versioned("gs://marin-us-central2/documents/dclm/negative_examples.txt"),
                label="lq",
                relative_sampling_rate=versioned(1.0),
                format=DatasetFormat.FASTTEXT,
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
        input_doc_paths=[
            DatasetCurationConfig(
                input_doc_path=versioned("gs://marin-us-central2/documents/dclm/explainlikeimfive"),
                label="hq",
                absolute_sampling_rate=versioned(100000),
                format=DatasetFormat.DOLMA_FORMATTED_JSONL,
            ),
            DatasetCurationConfig(
                input_doc_path=versioned("gs://marin-us-central2/documents/dclm/oh_100k.txt"),
                label="hq",
                absolute_sampling_rate=versioned(100000),
                format=DatasetFormat.FASTTEXT,
            ),
            DatasetCurationConfig(
                input_doc_path=versioned("gs://marin-us-central2/documents/dclm/negative_examples.txt"),
                label="lq",
                relative_sampling_rate=versioned(1.0),
                format=DatasetFormat.FASTTEXT,
            ),
        ],
        output_path=this_output_path(),
        fasttext_args={"lr": versioned(0.1), "thread": 4, "wordNgrams": 2},
        val_frac=versioned(0.0),
        seed=versioned(0),
    ),
)

teknium_oh_200k_rw_200k = ExecutorStep(
    name="classifiers/dclm_oh_200k_rw_200k",
    fn=train,
    config=TrainFasttextClassifierConfig(
        input_doc_paths=[
            DatasetCurationConfig(
                input_doc_path=output_path_of(openhermes_in_dolma_format, "text"),
                label="hq",
                absolute_sampling_rate=versioned(200000),
                format=DatasetFormat.DOLMA_FORMATTED_JSONL,
            ),
            DatasetCurationConfig(
                input_doc_path=versioned("gs://marin-us-central2/documents/dclm/negative_examples.txt"),
                label="lq",
                relative_sampling_rate=versioned(1.0),
                format=DatasetFormat.FASTTEXT,
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
            openhermes_in_dolma_format,
            dclm_eli5_200k_rw_200k,
            dclm_eli5_100k_oh_100k_rw_200k,
            teknium_oh_200k_rw_200k,
            dclm_eli5_100k_oh_100k_rw_200k_seed_1,
            dclm_eli5_100k_oh_100k_rw_200k_seed_2,
        ]
    )
