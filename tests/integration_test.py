# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import logging
import os
import sys

import draccus
from fray.v1.cluster import ResourceConfig, create_cluster, set_current_cluster
import humanfriendly
from levanter.main.train_lm import TrainLmConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.trainer import TrainerConfig
from marin.execution.step_model import StepSpec
from marin.execution.step_runner import StepRunner
from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig, FilterType, consolidate
from marin.processing.classification.dataset_utils import DatasetConfig
from marin.processing.classification.deduplication.dedup_commons import DedupConfig, DedupMode, deduplicate
from marin.processing.classification.fasttext.train_fasttext import train
from marin.processing.classification.inference import InferenceConfig, run_inference
from marin.processing.tokenize import lm_data_config
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from marin.schemas.web.convert import ResiliparseConfig
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from marin.transform.simple_html_to_md.process import html_to_md

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_steps(prefix: str, synth_data: str) -> list[StepSpec]:
    # ############################################################
    # Transform HTML to text

    transform_hq_data_step = StepSpec(
        name=os.path.join(prefix, "hq-transformed"),
        hash_attrs={"extract_method": "resiliparse"},
        fn=lambda output_path: html_to_md(
            input_path=os.path.join(synth_data, "pos"),
            output_path=output_path,
            extract_method="resiliparse",
            config=ResiliparseConfig(),
        ),
    )

    transform_lq_data_step = StepSpec(
        name=os.path.join(prefix, "lq-transformed"),
        hash_attrs={"extract_method": "resiliparse"},
        fn=lambda output_path: html_to_md(
            input_path=os.path.join(synth_data, "neg"),
            output_path=output_path,
            extract_method="resiliparse",
            config=ResiliparseConfig(),
        ),
    )

    # ############################################################
    # Train quality classifier

    train_quality_step = StepSpec(
        name=os.path.join(prefix, "quality-classifier"),
        deps=[transform_hq_data_step, transform_lq_data_step],
        fn=lambda output_path: train(
            datasets=[
                DatasetConfig(
                    input_doc_path=transform_hq_data_step.output_path,
                    label="hq",
                    sampling_rate=1.0,
                ),
                DatasetConfig(
                    input_doc_path=transform_lq_data_step.output_path,
                    label="lq",
                    sampling_rate=1.0,
                ),
            ],
            output_path=output_path,
            fasttext_args={
                "lr": 0.001,
                "minCount": 1,
                "epoch": 25,
                "wordNgrams": 2,
                "dim": 50,
                "thread": 1,
            },
        ),
    )

    ############################################################
    # Run inference with quality classifier

    inference_hq_step = StepSpec(
        name=os.path.join(prefix, "hq-inference"),
        deps=[transform_hq_data_step, train_quality_step],
        fn=lambda output_path: run_inference(
            InferenceConfig(
                input_path=transform_hq_data_step.output_path,
                output_path=output_path,
                model_name=train_quality_step.output_path,
                model_type="fasttext",
                attribute_name="quickstart-fasttext-quality-hq",
            )
        ),
    )

    inference_lq_step = StepSpec(
        name=os.path.join(prefix, "lq-inference"),
        deps=[transform_lq_data_step, train_quality_step],
        fn=lambda output_path: run_inference(
            InferenceConfig(
                input_path=transform_lq_data_step.output_path,
                output_path=output_path,
                model_name=train_quality_step.output_path,
                model_type="fasttext",
                attribute_name="quickstart-fasttext-quality-lq",
            )
        ),
    )

    ############################################################
    # Deduplicate

    dedup_exact_paragraph_step = StepSpec(
        name=os.path.join(prefix, "dedup_exact_paragraph"),
        deps=[transform_hq_data_step],
        fn=lambda output_path: deduplicate(
            DedupConfig(
                input_paths=transform_hq_data_step.output_path,
                output_path=output_path,
                mode=DedupMode.EXACT_PARAGRAPH,
                ray_memory=humanfriendly.parse_size("1GB", binary=True),
                ray_num_cpus=1,
            )
        ),
    )
    dedup_fuzzy_document_step = StepSpec(
        name=os.path.join(prefix, "dedup_fuzzy_document"),
        deps=[transform_hq_data_step],
        fn=lambda output_path: deduplicate(
            DedupConfig(
                input_paths=transform_hq_data_step.output_path,
                output_path=output_path,
                mode=DedupMode.FUZZY_DOCUMENT,
                ray_memory=humanfriendly.parse_size("1GB", binary=True),
                ray_num_cpus=1,
            )
        ),
    )

    ############################################################
    # Consolidate

    consolidate_step = StepSpec(
        name=os.path.join(prefix, "cleaned"),
        deps=[transform_hq_data_step, dedup_exact_paragraph_step, dedup_fuzzy_document_step],
        fn=lambda output_path: consolidate(
            ConsolidateConfig(
                input_path=transform_hq_data_step.output_path,
                output_path=output_path,
                filters=[
                    FilterConfig(
                        type=FilterType.REMOVE_SPANS,
                        attribute_path=os.path.join(dedup_exact_paragraph_step.output_path, "data"),
                        name=str(DedupMode.EXACT_PARAGRAPH),
                    ),
                    FilterConfig(
                        type=FilterType.REMOVE_DOC,
                        attribute_path=os.path.join(dedup_fuzzy_document_step.output_path, "data"),
                        name=str(DedupMode.FUZZY_DOCUMENT),
                    ),
                ],
            )
        ),
    )

    ############################################################
    # Tokenize
    tokenizer = "gpt2"

    tokenize_step = StepSpec(
        name=os.path.join(prefix, "tokenized"),
        deps=[consolidate_step],
        hash_attrs={"tokenizer": tokenizer},
        fn=lambda output_path: tokenize(
            TokenizeConfig(
                train_paths=[consolidate_step.output_path],
                validation_paths=[],
                cache_path=output_path,
                tokenizer=tokenizer,
                zephyr_num_cpus=1,
                zephyr_memory=humanfriendly.parse_size("1MB", binary=True),
            )
        ),
    )

    # ############################################################
    # Training
    train_env_vars = {
        "WANDB_API_KEY": "",
        "WANDB_MODE": "disabled",
        "JAX_TRACEBACK_FILTERING": "off",
    }

    pod_config = ResourceConfig.with_cpu()

    train_step = StepSpec(
        name=os.path.join(prefix, "train"),
        deps=[tokenize_step],
        fn=lambda output_path: run_levanter_train_lm(
            TrainLmOnPodConfig(
                output_path=output_path,
                resources=pod_config,
                env_vars=train_env_vars,
                train_config=TrainLmConfig(
                    data=lm_data_config(tokenize_step),
                    hf_save_steps=1,
                    model=Gpt2Config(
                        num_layers=2,
                        num_heads=2,
                        max_seq_len=64,
                        hidden_dim=32,
                    ),
                    trainer=TrainerConfig(
                        train_batch_size=8, num_train_steps=2, max_eval_batches=1, require_accelerator=False
                    ),
                ),
            )
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
        dedup_exact_paragraph_step,
        dedup_fuzzy_document_step,
        consolidate_step,
        tokenize_step,
        train_step,
        # evaluate_step,
    ]


@dataclass
class IntegrationTestConfig:
    prefix: str | None = None
    """Attached to every output path that's constructed (e.g., the GCS bucket)."""

    dry_run: bool = False
    force_run_failed: bool = True


@draccus.wrap()
def main(config: IntegrationTestConfig):
    try:
        bucket_prefix = config.prefix if config.prefix is not None else "/tmp"

        experiment_prefix = "quickstart-tests"

        # Set MARIN_PREFIX so StepSpec can resolve output paths
        os.environ["MARIN_PREFIX"] = bucket_prefix

        # start Ray explicitly and set it as the current cluster
        # N.B. This script must not be launched via `uv run`, or Ray will prefer to use `uv` for all execution
        # ignoring package dependencies specified in each step.
        if "uv run" in " ".join(sys.argv):
            raise RuntimeError("integration_test.py must not be launched via `uv run`. Please run it directly.")
        import ray

        ray.init(
            resources={"head_node": 1},
            runtime_env={"working_dir": None},
            num_cpus=os.cpu_count(),
            _memory=1024 * 1024 * 1024 * 1024,  # 1TB
        )
        set_current_cluster(create_cluster("ray"))

        # path to synthetic test data
        synth_data: str = "./tests/quickstart-data"
        # delete all previous runs
        if os.path.exists(os.path.join(bucket_prefix, experiment_prefix)):
            os.system(f"rm -rf {os.path.join(bucket_prefix, experiment_prefix)}")
        steps = create_steps(experiment_prefix, synth_data)
        StepRunner().run(steps, dry_run=config.dry_run, force_run_failed=config.force_run_failed)
        logger.info(f"Execution completed successfully. All outputs are in {bucket_prefix}/{experiment_prefix}")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise e


if __name__ == "__main__":
    main()
