import logging
import os
from dataclasses import dataclass

import draccus
from levanter.models.gpt2 import Gpt2Config
from levanter.trainer import TrainerConfig

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
    DatasetCurationConfig,
    TrainFasttextClassifierConfig,
    train,
)
from marin.processing.classification.inference import InferenceConfig, run_inference
from marin.processing.tokenize import TokenizeConfig, lm_training_config, tokenize
from marin.schemas.web.convert import HtmlToMarkdownConfig
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from scripts.hello_world_fw.process import FineWebConfig, transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class QuickstartExecutorConfig:
    # all artifacts will be saved in gs://marin-us-central2/{prefix}{commit_hash}
    commit_hash: str = ""
    prefix: str = "quickstart-tests"

    # path to synthetic test data
    synth_data: str = "./tests/quick-start-tests"


def create_steps(config: QuickstartExecutorConfig) -> list[ExecutorStep]:
    # ############################################################
    # Transform HTML to text

    transform_hq_data_step = ExecutorStep(
        name=os.path.join(config.prefix, config.commit_hash, "hq-transformed"),
        fn=transform,
        config=FineWebConfig(
            input_path=os.path.join(config.synth_data, "pos"),
            output_path=this_output_path(),
            extract_method=versioned("readability"),
            config=HtmlToMarkdownConfig.default_config(),
        ),
    )

    transform_lq_data_step = ExecutorStep(
        name=os.path.join(config.prefix, config.commit_hash, "lq-transformed"),
        fn=transform,
        config=FineWebConfig(
            input_path=os.path.join(config.synth_data, "neg"),
            output_path=this_output_path(),
            extract_method=versioned("readability"),
            config=HtmlToMarkdownConfig.default_config(),
        ),
    )

    # ############################################################
    # Train quality classifier

    train_quality_step = ExecutorStep(
        name=os.path.join(config.prefix, config.commit_hash, "quality-classifier"),
        fn=train,
        config=TrainFasttextClassifierConfig(
            input_doc_paths=[
                DatasetCurationConfig(
                    input_doc_path=output_path_of(transform_hq_data_step),
                    label="hq",
                    relative_sampling_rate=1.0,
                    format="dolma_formatted_jsonl",
                ),
                DatasetCurationConfig(
                    input_doc_path=output_path_of(transform_lq_data_step),
                    label="lq",
                    relative_sampling_rate=1.0,
                    format="dolma_formatted_jsonl",
                ),
            ],
            output_path=this_output_path(),
            fasttext_args={
                "lr": 0.001,
                "minCount": 1,
                "epoch": 25,
                "wordNgrams": 2,
                "dim": 50,
            },
        ),
    )

    ############################################################
    # Run inference with quality classifier

    inference_hq_step = ExecutorStep(
        name=os.path.join(config.prefix, config.commit_hash, "hq-inference"),
        fn=run_inference,
        config=InferenceConfig(
            input_path=output_path_of(transform_hq_data_step),
            output_path=this_output_path(),
            model_name=output_path_of(train_quality_step),
            model_type="fasttext",
            attribute_name="quickstart-fasttext-quality-hq",
        ),
    )

    inference_lq_step = ExecutorStep(
        name=os.path.join(config.prefix, config.commit_hash, "lq-inference"),
        fn=run_inference,
        config=InferenceConfig(
            input_path=output_path_of(transform_lq_data_step),
            output_path=this_output_path(),
            model_name=output_path_of(train_quality_step),
            model_type="fasttext",
            attribute_name="quickstart-fasttext-quality-lq",
        ),
    )

    ############################################################
    # Deduplicate

    dedupe_step = ExecutorStep(
        name=os.path.join(config.prefix, config.commit_hash, "dedupe"),
        fn=dedupe,
        config=DedupeConfig(
            input_path=output_path_of(transform_hq_data_step),
            output_path=this_output_path(),
        ),
    )

    ############################################################
    # Consolidate

    consolidate_step = ExecutorStep(
        name=os.path.join(config.prefix, config.commit_hash, "consolidate"),
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
    )

    ############################################################
    # Tokenize

    tokenize_step = ExecutorStep(
        name=os.path.join(config.prefix, config.commit_hash, "tokenized"),
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=output_path_of(consolidate_step),
            validation_paths=[],
            cache_path=this_output_path(),
            tokenizer=versioned("gpt2"),
        ),
    )

    ############################################################
    # Train

    model_name = "quickstart-trained-model"

    train_step = ExecutorStep(
        name=os.path.join(config.prefix, config.commit_hash, "train"),
        fn=run_levanter_train_lm,
        config=TrainLmOnPodConfig(
            output_path=os.path.join(config.prefix, config.commit_hash, "train", model_name),
            data=lm_training_config(tokenize_step),
            env={"WANDB_API_KEY": None, "WANDB_MODE": "disabled"},  # Running it locally and turning off wandb
            tpu_type=None,
            hf_save_steps=1,
            model=Gpt2Config(
                num_layers=2,
                num_heads=2,
                seq_len=64,
                hidden_dim=32,
            ),
            trainer=TrainerConfig(train_batch_size=1, num_train_steps=2, max_eval_batches=1, require_accelerator=False),
        ),
    )

    logger.info(
        f"""Finished training model {model_name} with config: {train_step.config}. Model saved at
                {os.path.join(config.prefix, config.commit_hash, "train", model_name)}"""
    )

    ############################################################
    # Evaluation

    # eval_step = ExecutorStep(
    #     name=os.path.join(config.prefix, config.commit_hash, "eval"),
    #     fn=evaluate,
    #     config=EvaluationConfig(
    #         evaluator="helm",
    #         model_name=f"{model_name}/step-1",
    #         model_path=os.path.join(
    #             "/tmp", config.prefix, config.commit_hash, "train", model_name, "hf", model_name, "step-1"
    #         ),
    #         evaluation_path=this_output_path(),
    #         evals=["mmlu"],
    #         launch_with_ray=False,
    #     ),
    # )

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
        # eval_step,
    ]


@draccus.wrap()
def main(config: QuickstartExecutorConfig):
    try:
        steps = create_steps(config)
        bucket_prefix = "/tmp"
        config_executor = ExecutorMainConfig(
            prefix=bucket_prefix, executor_info_base_path=os.path.join(bucket_prefix, "experiments")
        )
        executor_main(config_executor, steps=steps)
        logger.info(
            f"Execution completed successfully. All outputs are in {bucket_prefix}/{config.prefix}/{config.commit_hash}"
        )
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise e


if __name__ == "__main__":
    main()
