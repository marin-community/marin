import dataclasses
import logging
import os
import sys

import draccus
from levanter.models.gpt2 import Gpt2Config
from levanter.trainer import TrainerConfig

from marin.classifiers.utils import DatasetConfig
from marin.execution.executor import (
    ExecutorMainConfig,
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig, consolidate
from marin.processing.classification.dedupe import DedupeConfig, dedupe
from marin.processing.classification.fasttext.train_fasttext import (
    TrainFasttextClassifierConfig,
    train,
)
from marin.processing.classification.inference import InferenceConfig, run_inference
from marin.processing.tokenize import TokenizeConfig, lm_data_config, tokenize
from marin.schemas.web.convert import HtmlToMarkdownConfig
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from marin.utilities.ray_utils import is_local_ray_cluster
from marin.utils import is_in_ci
from scripts.hello_world_fw.process import FineWebConfig, transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_steps(prefix: str, synth_data: str) -> list[ExecutorStep]:
    # ############################################################
    # Transform HTML to text

    transform_hq_data_step = ExecutorStep(
        name=os.path.join(prefix, "hq-transformed"),
        fn=transform,
        config=FineWebConfig(
            input_path=os.path.join(synth_data, "pos"),
            output_path=this_output_path(),
            extract_method=versioned("readability"),
            config=HtmlToMarkdownConfig.default_config(),
        ),
        pip_dependency_groups=["download_transform"],
    )

    transform_lq_data_step = ExecutorStep(
        name=os.path.join(prefix, "lq-transformed"),
        fn=transform,
        config=FineWebConfig(
            input_path=os.path.join(synth_data, "neg"),
            output_path=this_output_path(),
            extract_method=versioned("readability"),
            config=HtmlToMarkdownConfig.default_config(),
        ),
        pip_dependency_groups=["download_transform"],
    )

    # ############################################################
    # Train quality classifier

    train_quality_step = ExecutorStep(
        name=os.path.join(prefix, "quality-classifier"),
        fn=train,
        config=TrainFasttextClassifierConfig(
            datasets=[
                DatasetConfig(
                    input_doc_path=output_path_of(transform_hq_data_step),
                    label="hq",
                    sampling_rate=1.0,
                ),
                DatasetConfig(
                    input_doc_path=output_path_of(transform_lq_data_step),
                    label="lq",
                    sampling_rate=1.0,
                ),
            ],
            output_path=this_output_path(),
            fasttext_args={
                "lr": 0.001,
                "minCount": 1,
                "epoch": 25,
                "wordNgrams": 2,
                "dim": 50,
                "thread": 1,
            },
        ),
        pip_dependency_groups=["quality_dedup_consolidate"],
    )

    ############################################################
    # Run inference with quality classifier

    inference_hq_step = ExecutorStep(
        name=os.path.join(prefix, "hq-inference"),
        fn=run_inference,
        config=InferenceConfig(
            input_path=output_path_of(transform_hq_data_step),
            output_path=this_output_path(),
            model_name=output_path_of(train_quality_step),
            model_type="fasttext",
            attribute_name="quickstart-fasttext-quality-hq",
        ),
        pip_dependency_groups=["quality_dedup_consolidate"],
    )

    inference_lq_step = ExecutorStep(
        name=os.path.join(prefix, "lq-inference"),
        fn=run_inference,
        config=InferenceConfig(
            input_path=output_path_of(transform_lq_data_step),
            output_path=this_output_path(),
            model_name=output_path_of(train_quality_step),
            model_type="fasttext",
            attribute_name="quickstart-fasttext-quality-lq",
        ),
        pip_dependency_groups=["quality_dedup_consolidate"],
    )

    ############################################################
    # Deduplicate

    dedupe_step = ExecutorStep(
        name=os.path.join(prefix, "dedupe"),
        fn=dedupe,
        config=DedupeConfig(
            input_path=output_path_of(transform_hq_data_step),
            output_path=this_output_path(),
        ),
        pip_dependency_groups=["quality_dedup_consolidate"],
    )

    ############################################################
    # Consolidate

    consolidate_step = ExecutorStep(
        name=os.path.join(prefix, "consolidate"),
        fn=consolidate,
        config=ConsolidateConfig(
            input_path=output_path_of(transform_hq_data_step),
            output_path=this_output_path(),
            filters=[
                FilterConfig(
                    type=versioned("classify"),
                    attribute_path=output_path_of(inference_hq_step),
                    name=versioned("quickstart-fasttext-quality-hq"),
                    label="__label__hq",
                    threshold=versioned(0.1),
                ),
                FilterConfig(
                    type=versioned("remove_spans"),
                    attribute_path=output_path_of(dedupe_step),
                    name=versioned("duplicate_text"),
                ),
            ],
        ),
        pip_dependency_groups=["quality_dedup_consolidate"],
    )

    ############################################################
    # Tokenize
    tokenizer = "gpt2"

    tokenize_step = ExecutorStep(
        name=os.path.join(prefix, "tokenized"),
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=output_path_of(consolidate_step),
            validation_paths=[],
            cache_path=this_output_path(),
            tokenizer=versioned(tokenizer),
        ),
    )

    # ############################################################
    # Training
    if not is_in_ci() and not is_local_ray_cluster():
        tpu_type = "v4-8"
    else:
        tpu_type = None

    train_step = ExecutorStep(
        name=os.path.join(prefix, "train"),
        fn=run_levanter_train_lm,
        config=TrainLmOnPodConfig(
            output_path=this_output_path(),
            data=lm_data_config(tokenize_step),
            env={
                "WANDB_API_KEY": None,
                "WANDB_MODE": "disabled",
                "JAX_PLATFORMS": "cpu",
            },  # Running it locally and turning off wandb
            tpu_type=tpu_type,
            hf_save_steps=1,
            model=Gpt2Config(
                num_layers=2,
                num_heads=2,
                seq_len=64,
                hidden_dim=32,
            ),
            trainer=TrainerConfig(train_batch_size=8, num_train_steps=2, max_eval_batches=1, require_accelerator=False),
        ),
    )

    ##### Evaluate

    # evaluate_step = evaluate_helm_on_step(train_step, ["mmlu"], max_eval_instances=10)

    return [
        transform_hq_data_step,
        transform_lq_data_step,
        train_quality_step,
        inference_hq_step,
        inference_lq_step,
        dedupe_step,
        consolidate_step,
        tokenize_step,
        train_step,
        # evaluate_step,
    ]


@draccus.wrap()
def main(config: ExecutorMainConfig):
    try:
        # Replace this only if it's not in argv
        if "--prefix" in sys.argv:  # Check if prefix is already provided
            bucket_prefix = config.prefix
        else:
            bucket_prefix = "/tmp"  # Default to a temporary directory

        experiment_prefix = "quickstart-tests"
        config = dataclasses.replace(
            config, prefix=bucket_prefix, executor_info_base_path=os.path.join(bucket_prefix, "experiments")
        )

        # path to synthetic test data
        synth_data: str = "./tests/quickstart-data"
        steps = create_steps(experiment_prefix, synth_data)
        config = dataclasses.replace(config, force_run=[step.name for step in steps])  # Always force run all steps
        executor_main(config, steps=steps)
        logger.info(f"Execution completed successfully. All outputs are in {bucket_prefix}/{experiment_prefix}")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise e


if __name__ == "__main__":
    main()
