import logging
import os
from dataclasses import dataclass

import draccus
import jax
import ray

from levanter.data.text import LMDatasetConfig
from levanter.distributed import RayConfig
from levanter.main import train_lm
from levanter.tracker.wandb import WandbConfig
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
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.schemas.web.convert import HtmlToMarkdownConfig
from marin.training.training import train_lm_task
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
                    label="__label__hq",
                    relative_sampling_rate=1.0,
                    format="dolma_formatted_jsonl",
                ),
                DatasetCurationConfig(
                    input_doc_path=output_path_of(transform_lq_data_step),
                    label="__label__lq",
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
            input_path=output_path_of(consolidate_step),
            cache_path=this_output_path(),
            dataset_name="quickstart_test",  # Does this have to be unique?
            tokenizer="gpt2",
        ),
    )

    ############################################################
    # Train

    train_step = ExecutorStep(
        name=os.path.join(config.prefix, config.commit_hash, "train"),
        fn=train_lm_task,
        config=train_lm.TrainLmConfig(
            hf_save_steps=1,
            hf_save_path=this_output_path(),
            data=LMDatasetConfig(
                train_urls=[output_path_of(consolidate_step, "*.jsonl.gz")],
                validation_urls=[output_path_of(consolidate_step, "*.jsonl.gz")],
                cache_dir=output_path_of(tokenize_step, "quickstart_test"),
                tokenizer="gpt2",
            ),
            model=train_lm.Gpt2Config(
                num_layers=2,
                num_heads=2,
                seq_len=64,
                hidden_dim=32,
                attn_backend=None,  # use default for platform
            ),
            trainer=train_lm.TrainerConfig(
                num_train_steps=2,
                train_batch_size=len(jax.devices()),
                max_eval_batches=1,
                wandb=WandbConfig(mode="disabled"),
                require_accelerator=False,
                ray=RayConfig(auto_start_cluster=False),
            ),
        ),
    )

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
    ]


@draccus.wrap()
def main(config: QuickstartExecutorConfig):
    try:
        steps = create_steps(config)
        bucket_prefix = "/tmp"
        config_executor = ExecutorMainConfig(prefix=bucket_prefix)
        executor_main(config_executor, steps=steps)
        logger.info(
            f"Execution completed successfully. All outputs are in {bucket_prefix}/{config.prefix}/{config.commit_hash}"
        )
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise e


if __name__ == "__main__":
    ray.init()
    main()
