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

from experiments.anneal_config import AnnealConfig
from experiments.defaults import default_anneal, default_tokenize, default_train
from experiments.dolmino.tokenize_dolmino import get_dolmino_step_llama3
from experiments.evals.evals import default_eval
from experiments.exp246_web_extraction_method_training import (
    transform_resiliparse_preserve_formatting,
)
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
from marin.processing.tokenize.data_configs import lm_mixture_data_config

dolmino_dclm = get_dolmino_step_llama3("dclm")


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
            "fineweb_2024_18": output_path_of(transform_resiliparse_preserve_formatting, "md", "CC-MAIN-2024-18"),
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
        pip_dependency_groups=[
            "--find-links https://storage.googleapis.com/libtpu-releases/index.html",
            "--find-links https://storage.googleapis.com/libtpu-wheels/index.html",
            "fasttext",
            "datasets",
            "filelock",
            "torch",
            "torch_xla[tpu]",
            "accelerate",
        ],
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
            pip_dependency_groups=[
                "--find-links https://storage.googleapis.com/libtpu-releases/index.html",
                "--find-links https://storage.googleapis.com/libtpu-wheels/index.html",
                "fasttext",
                "datasets",
                "filelock",
                "torch",
                "torch_xla[tpu]",
                "accelerate",
            ],
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
                            threshold=versioned(None),
                            keep_fraction=versioned(keep_fraction),
                        ),
                    ],
                    ray_memory_limit_gb=12,
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
                            threshold=versioned(None),
                            keep_fraction=versioned(keep_fraction),
                        ),
                    ],
                    ray_memory_limit_gb=12,
                ),
                pip_dependency_groups=["ddsketch"],
            )

            fasttext_consolidate_steps.append(fasttext_consolidate_step)
            bert_consolidate_steps.append(bert_consolidate_step)

        steps.extend(fasttext_consolidate_steps)
        steps.extend(bert_consolidate_steps)

        # dolmino_dclm = get_dolmino_step_llama3("dclm")
        for fasttext_consolidate_step, bert_consolidate_step in zip(
            fasttext_consolidate_steps, bert_consolidate_steps, strict=False
        ):
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
                tokenized=fasttext_tokenize_step,
                model_config=llama_1_4b,
                train_config=llama_1_4b_train_config,
            )
            bert_train_step = default_train(
                name=f"checkpoints/quality_filtering/{config.experiment_name}/bert/{input_data_source}/train",
                tokenized=bert_tokenize_step,
                model_config=llama_1_4b,
                train_config=llama_1_4b_train_config,
            )
            steps.append(fasttext_train_step)
            steps.append(bert_train_step)

            fasttext_anneal_config = AnnealConfig(
                dataset_config=lm_mixture_data_config(
                    components={
                        "fineweb-hq": fasttext_tokenize_step,
                        "dolmino": dolmino_dclm,
                    },
                    weights={"fineweb-hq": 0.3, "dolmino": 0.7},
                ),
            )
            fasttext_anneal_step = default_anneal(
                name=f"checkpoints/quality_filtering/{config.experiment_name}/fasttext/{input_data_source}/anneal",
                anneal_config=fasttext_anneal_config,
            )

            bert_anneal_config = AnnealConfig(
                dataset_config=lm_mixture_data_config(
                    components={
                        "fineweb-hq": bert_tokenize_step,
                        "dolmino": dolmino_dclm,
                    },
                    weights={"fineweb-hq": 0.3, "dolmino": 0.7},
                ),
            )
            bert_anneal_step = default_anneal(
                name=f"checkpoints/quality_filtering/{config.experiment_name}/bert/{input_data_source}/anneal",
                anneal_config=bert_anneal_config,
            )
            steps.append(fasttext_anneal_step)
            steps.append(bert_anneal_step)

            eval_fasttext_train = default_eval(fasttext_train_step)
            eval_bert_train = default_eval(bert_train_step)
            eval_fasttext_anneal = default_eval(fasttext_anneal_step)
            eval_bert_anneal = default_eval(bert_anneal_step)

            steps.append(eval_fasttext_train)
            steps.append(eval_bert_train)
            steps.append(eval_fasttext_anneal)
            steps.append(eval_bert_anneal)

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
