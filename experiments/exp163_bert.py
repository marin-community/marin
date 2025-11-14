# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Experiment 163: Compare BERT vs Fasttext classifiers. In particular, we 1) train both types of classifiers on the
same high-quality versus low-quality data, 2) filter a pool of pretraining documents using the classifiers, and
then 3) train language models on the filtered documents.
For this experiment, we use 100k examples from the MMLU benchmark as the high-quality examples 100k documents from the
DCLM pool as the low-quality examples. We then filter (varying fractions of) FineWeb Common Crawl documents using
the classifiers, and then use our default experiment training pipeline to train language models on the filtered data.

See https://github.com/stanford-crfm/marin/issues/163 for more details.
"""

import os
from dataclasses import dataclass, field

from experiments.defaults import default_tokenize, default_train
from experiments.pretraining_datasets import get_dolmino_step
from experiments.exp274_mmlu_quality_classifier import (
    dclm_negative_examples_in_dolma_format,
    mmlu_eval_aux_in_dolma_format,
)
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from marin.classifiers.utils import DatasetConfig
from marin.core.runtime import TaskConfig
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.processing.tokenize import lm_data_config
from marin.processing.classification.bert.train_bert import (
    TrainBertClassifierConfig,
)
from marin.processing.classification.bert.train_bert import (
    train as train_bert,
)
from marin.processing.classification.config.inference_config import RuntimeConfig
from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig, consolidate
from marin.processing.classification.fasttext.train_fasttext import (
    TrainFasttextClassifierConfig,
)
from marin.processing.classification.fasttext.train_fasttext import (
    train as train_fasttext,
)
from marin.processing.classification.inference import InferenceConfig, run_inference

dolmino_dclm = get_dolmino_step("dclm")

BERT_DEPENDENCIES = [
    "--find-links https://storage.googleapis.com/libtpu-releases/index.html",
    "--find-links https://storage.googleapis.com/libtpu-wheels/index.html",
    "fasttext",
    "datasets",
    "filelock",
    "torch",
    "torch_xla[tpu]",
    "accelerate",
]


@dataclass
class ExperimentConfig:
    """Configuration for comparing BERT vs fastText classifiers.

    This config defines parameters for an experiment that:
    1. Takes input documents from specified data sources
    2. Takes positive and negative documents from specified data sources.
    3. Trains both a BERT and a fastText classifier on the positive/negative documents.
    4. Filters and tokenizes the resulting data.

    Args:
        experiment_name: Identifier for this experiment
        input_data_source_to_path: Mapping of data source names to their GCS paths
        keep_fractions: Fractions of highest-quality documents to keep after filtering
    """

    experiment_name: str
    classifier_training_datasets: list[DatasetConfig]
    input_data_source_to_path: dict[str, str] = field(
        default_factory=lambda: {
            "fineweb_2024_18": (
                # the executor path is:output_path_of(transform_resiliparse_preserve_formatting, "md/CC-MAIN-2024-18"),
                # but we want to use this specific shard without redownloading the full FineWeb so we hardcode instead
                "gs://marin-us-central2/documents/fineweb-small-resiliparse-preserve-formatting-v2-e72837/md/CC-MAIN-2024-18/"
            ),
        }
    )
    keep_fractions: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2])


def create_steps(config: ExperimentConfig) -> list[ExecutorStep]:
    """Create the steps for a single experiment.

    Variation of exp614_quality_filtering.py, but trains both BERT and Fasttext quality classifiers.
    """

    steps = []

    fasttext_classifier_train = ExecutorStep(
        name=f"classifiers/{config.experiment_name}/fasttext",
        fn=train_fasttext,
        config=TrainFasttextClassifierConfig(
            datasets=config.classifier_training_datasets,
            output_path=this_output_path(),
            fasttext_args={"lr": versioned(0.1), "thread": 4, "wordNgrams": 2},
            val_frac=versioned(0.1),
            seed=versioned(0),
        ),
        pip_dependency_groups=["fasttext"],
    )

    bert_classifier_train = ExecutorStep(
        name=f"classifiers/{config.experiment_name}/bert",
        fn=train_bert,
        config=TrainBertClassifierConfig(
            datasets=config.classifier_training_datasets,
            output_path=this_output_path(),
            val_frac=versioned(0.1),
            seed=versioned(0),
        ),
        pip_dependency_groups=BERT_DEPENDENCIES,
    )

    for input_data_source, input_data_path in config.input_data_source_to_path.items():
        # Get the basename of the input directory
        input_basename = os.path.basename(os.path.normpath(input_data_path))

        fasttext_inference = ExecutorStep(
            name=f"attributes/quality_filtering/{config.experiment_name}/fasttext/{input_data_source}",
            fn=run_inference,
            config=InferenceConfig(
                input_path=input_data_path,
                output_path=this_output_path(input_basename),
                model_name=output_path_of(fasttext_classifier_train, "model.bin"),
                model_type="fasttext",
                attribute_name=versioned(f"{config.experiment_name}-fasttext_classifier"),
                runtime=RuntimeConfig(
                    memory_limit_gb=12,
                ),
                task=TaskConfig(max_in_flight=500),
            ),
            pip_dependency_groups=["fasttext", "datasets", "filelock"],
        )

        bert_inference = ExecutorStep(
            name=f"attributes/quality_filtering/{config.experiment_name}/bert/{input_data_source}",
            fn=run_inference,
            config=InferenceConfig(
                input_path=input_data_path,
                output_path=this_output_path(input_basename),
                model_name=output_path_of(bert_classifier_train, "model"),
                model_type="bert",
                attribute_name=versioned(f"{config.experiment_name}-bert_classifier"),
                runtime=RuntimeConfig(
                    memory_limit_gb=12,
                    resources={"TPU": 4, "TPU-v4-8-head": 1},
                ),
                task=TaskConfig(max_in_flight=500),
            ),
            pip_dependency_groups=BERT_DEPENDENCIES,
        )

        fasttext_consolidate_steps, bert_consolidate_steps = [], []
        for keep_fraction in config.keep_fractions:
            fasttext_consolidate_step = ExecutorStep(
                name=f"documents/quality_filtering/{config.experiment_name}/fasttext/{input_data_source}",
                fn=consolidate,
                config=ConsolidateConfig(
                    input_path=input_data_path,
                    output_path=this_output_path(input_basename),
                    filters=[
                        FilterConfig(
                            type=versioned("classify"),
                            attribute_path=output_path_of(fasttext_inference, input_basename),
                            name=versioned(f"{config.experiment_name}-fasttext_classifier"),
                            label="__label__hq",
                            lower_threshold=versioned(None),
                            keep_fraction=versioned(keep_fraction),
                        ),
                    ],
                ),
                pip_dependency_groups=["ddsketch"],
            )

            bert_consolidate_step = ExecutorStep(
                name=f"documents/quality_filtering/{config.experiment_name}/bert/{input_data_source}",
                fn=consolidate,
                config=ConsolidateConfig(
                    input_path=input_data_path,
                    output_path=this_output_path(input_basename),
                    filters=[
                        FilterConfig(
                            type=versioned("classify"),
                            attribute_path=output_path_of(bert_inference, input_basename),
                            name=versioned(f"{config.experiment_name}-bert_classifier"),
                            label="hq",
                            lower_threshold=versioned(None),
                            keep_fraction=versioned(keep_fraction),
                        ),
                    ],
                ),
                pip_dependency_groups=["ddsketch"],
            )

            fasttext_consolidate_steps.append(fasttext_consolidate_step)
            bert_consolidate_steps.append(bert_consolidate_step)

            steps.append(fasttext_consolidate_step)
            steps.append(bert_consolidate_step)

            fasttext_tokenize_step = default_tokenize(
                name=f"tokenized/quality_filtering/{config.experiment_name}/fasttext/{input_data_source}",
                dataset=fasttext_consolidate_step,
                tokenizer=llama3_tokenizer,
            )
            bert_tokenize_step = default_tokenize(
                name=f"tokenized/quality_filtering/{config.experiment_name}/bert/{input_data_source}",
                dataset=bert_consolidate_step,
                tokenizer=llama3_tokenizer,
            )
            steps.append(fasttext_tokenize_step)
            steps.append(bert_tokenize_step)

            fasttext_train_step = default_train(
                name=f"checkpoints/quality_filtering/{config.experiment_name}/fasttext/{input_data_source}/train",
                tokenized=lm_data_config(fasttext_tokenize_step, permutation_type="linear"),
                model_config=llama_1_4b,
                train_config=llama_1_4b_train_config,
            )
            bert_train_step = default_train(
                name=f"checkpoints/quality_filtering/{config.experiment_name}/bert/{input_data_source}/train",
                tokenized=lm_data_config(bert_tokenize_step, permutation_type="linear"),
                model_config=llama_1_4b,
                train_config=llama_1_4b_train_config,
            )
            steps.append(fasttext_train_step)
            steps.append(bert_train_step)

    return steps


def main():
    classifier_training_datasets = [
        DatasetConfig(
            input_doc_path=output_path_of(mmlu_eval_aux_in_dolma_format),
            label="hq",
            sampling_rate=1.0,
            max_sample_size=versioned(100000),
        ),
        DatasetConfig(
            input_doc_path=output_path_of(dclm_negative_examples_in_dolma_format),
            label="lq",
            sampling_rate=1.0,
            max_sample_size=versioned(100000),
        ),
    ]

    experiment_config = ExperimentConfig(
        experiment_name="exp163_compare_bert_fasttext_fineweb",
        classifier_training_datasets=classifier_training_datasets,
    )
    steps = create_steps(experiment_config)
    executor_main(steps=steps)


if __name__ == "__main__":
    main()
