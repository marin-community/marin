from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.classification.fasttext.train_fasttext import (
    DatasetCurationConfig,
    TrainFasttextClassifierConfig,
    train,
)
from marin.processing.classification.types import DatasetFormat

train_dclm_eli5_200k_rw_200k_step = ExecutorStep(
    name="classifiers/dclm_eli5_200k_rw_200k",
    fn=train,
    config=TrainFasttextClassifierConfig(
        input_doc_paths=[
            DatasetCurationConfig(
                input_doc_path=versioned("gs://marin-us-central2/documents/dclm/explainlikeimfive"),
                label="__label__hq",
                sampling_rate=versioned(200000),
                format=DatasetFormat.DOLMA_FORMATTED_JSONL,
            ),
            DatasetCurationConfig(
                input_doc_path=versioned("gs://marin-us-central2/documents/dclm/negative_examples.txt"),
                label="__label__lq",
                sampling_rate=versioned(1.0),
                format=DatasetFormat.FASTTEXT,
            ),
        ],
        output_path=this_output_path(),
        fasttext_args={"lr": 0.1},
    ),
)

train_dclm_eli5_100k_oh_100k_rw_200k_step = ExecutorStep(
    name="classifiers/dclm_eli5_100k_oh_100k_rw_200k",
    fn=train,
    config=TrainFasttextClassifierConfig(
        input_doc_paths=[
            DatasetCurationConfig(
                input_doc_path=versioned("gs://marin-us-central2/documents/dclm/explainlikeimfive"),
                label="__label__hq",
                sampling_rate=versioned(100000),
                format=DatasetFormat.DOLMA_FORMATTED_JSONL,
            ),
            DatasetCurationConfig(
                input_doc_path=versioned("gs://marin-us-central2/documents/dclm/oh_100k.txt"),
                label="__label__hq",
                sampling_rate=versioned(100000),
                format=DatasetFormat.FASTTEXT,
            ),
            DatasetCurationConfig(
                input_doc_path=versioned("gs://marin-us-central2/documents/dclm/negative_examples.txt"),
                label="__label__lq",
                sampling_rate=versioned(1.0),
                format=DatasetFormat.FASTTEXT,
            ),
        ],
        output_path=this_output_path(),
        fasttext_args={"lr": 0.1},
    ),
)

train_dclm_oh_200k_rw_200k_step = ExecutorStep(
    name="classifiers/dclm_oh_200k_rw_200k",
    fn=train,
    config=TrainFasttextClassifierConfig(
        input_doc_paths=[
            DatasetCurationConfig(
                input_doc_path=versioned("gs://marin-us-central2/documents/teknium--OpenHermes-2.5/v2024-09-29"),
                label="__label__hq",
                sampling_rate=versioned(200000),
                format=DatasetFormat.DOLMA_FORMATTED_JSONL,
            ),
            DatasetCurationConfig(
                input_doc_path=versioned("gs://marin-us-central2/documents/dclm/negative_examples.txt"),
                label="__label__lq",
                sampling_rate=versioned(1.0),
                format=DatasetFormat.FASTTEXT,
            ),
        ],
        output_path=this_output_path(),
        fasttext_args={"lr": 0.1},
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[
            train_dclm_eli5_200k_rw_200k_step,
            # train_dclm_eli5_100k_oh_100k_rw_200k_step,
            # train_dclm_oh_200k_rw_200k_step,
        ]
    )
