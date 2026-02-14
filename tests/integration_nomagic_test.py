# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test expressed using StepSpec and StepRunner directly.

This is equivalent to integration_test.py but avoids the executor "magic".
"""

import dataclasses
import logging
import os
import sys

import draccus
from fray.v1.cluster import create_cluster, set_current_cluster
from fray.v2 import ResourceConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.trainer import TrainerConfig


from marin.execution.artifact import Artifact
from marin.execution.step_model import StepSpec, StepMeta
from marin.execution.step_runner import StepRunner
from marin.processing.classification.consolidate import FilterConfig, FilterType, consolidate_fn
from marin.processing.classification.dataset_utils import DatasetConfig
from marin.processing.classification.deduplication.dedup_commons import DedupMetadata, DedupMode, deduplicate_fn
from marin.processing.classification.fasttext.train_fasttext import train
from marin.processing.tokenize import lm_data_config
from marin.processing.tokenize.tokenize import tokenize_fn, TokenizedMetadata
from marin.schemas.web.convert import ResiliparseConfig
from marin.transform.simple_html_to_md.process import html_to_md
from marin.training.training import run_levanter_train_lm_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_steps(*, prefix: str, synth_data: str) -> list[StepSpec]:
    # ############################################################
    # Transform HTML to text

    hq_step = StepSpec(
        fn=lambda output_path: html_to_md(
            input_path=os.path.join(synth_data, "pos"),
            output_path=output_path,
            extract_method="resiliparse",
            config=ResiliparseConfig(),
        ),
        meta=StepMeta(
            name="hq_html_to_md",
            output_path_prefix=prefix,
            hash_attrs={"input_path": os.path.join(synth_data, "pos"), "extract_method": "resiliparse"},
        ),
    )

    lq_step = StepSpec(
        fn=lambda output_path: html_to_md(
            input_path=os.path.join(synth_data, "neg"),
            output_path=output_path,
            extract_method="resiliparse",
            config=ResiliparseConfig(),
        ),
        meta=StepMeta(
            name="lq_html_to_md",
            output_path_prefix=prefix,
            hash_attrs={"input_path": os.path.join(synth_data, "neg"), "extract_method": "resiliparse"},
        ),
    )

    # ############################################################
    # Train quality classifier

    fasttext_args = {"lr": 0.001, "minCount": 1, "epoch": 25, "wordNgrams": 2, "dim": 50, "thread": 1}

    train_quality_step = StepSpec(
        fn=lambda output_path: train(
            datasets=[
                DatasetConfig(input_doc_path=hq_step.output_path, label="hq", sampling_rate=1.0),
                DatasetConfig(input_doc_path=lq_step.output_path, label="lq", sampling_rate=1.0),
            ],
            output_path=output_path,
            fasttext_args=fasttext_args,
        ),
        meta=StepMeta(
            name="train_quality",
            output_path_prefix=prefix,
            hash_attrs={"fasttext_args": fasttext_args},
            deps=[hq_step, lq_step],
        ),
    )

    # ############################################################
    # Deduplicate

    dedup_exact_step = StepSpec(
        fn=lambda output_path: deduplicate_fn(
            input_paths=hq_step.output_path,
            output_path=output_path,
            mode=DedupMode.EXACT_PARAGRAPH,
            ray_memory=1 * 1024**3,  # 1GB
            ray_num_cpus=1,
        ),
        meta=StepMeta(
            name="dedup_exact_paragraph",
            output_path_prefix=prefix,
            hash_attrs={"mode": DedupMode.EXACT_PARAGRAPH},
            deps=[hq_step],
        ),
    )

    dedup_fuzzy_step = StepSpec(
        fn=lambda output_path: deduplicate_fn(
            input_paths=hq_step.output_path,
            output_path=output_path,
            mode=DedupMode.FUZZY_DOCUMENT,
            ray_memory=1 * 1024**3,  # 1GB
            ray_num_cpus=1,
        ),
        meta=StepMeta(
            name="dedup_fuzzy_document",
            output_path_prefix=prefix,
            hash_attrs={"mode": DedupMode.FUZZY_DOCUMENT},
            deps=[hq_step],
        ),
    )

    # ############################################################
    # Consolidate

    consolidate_step = StepSpec(
        fn=lambda output_path: consolidate_fn(
            input_path=hq_step.output_path,
            output_path=output_path,
            filters=[
                FilterConfig(
                    type=FilterType.REMOVE_SPANS,
                    attribute_path=Artifact.load(dedup_exact_step, DedupMetadata).data_path,
                    name=DedupMode.EXACT_PARAGRAPH,
                ),
                FilterConfig(
                    type=FilterType.REMOVE_DOC,
                    attribute_path=Artifact.load(dedup_fuzzy_step, DedupMetadata).data_path,
                    name=DedupMode.FUZZY_DOCUMENT,
                ),
            ],
        ),
        meta=StepMeta(
            name="consolidate",
            output_path_prefix=prefix,
            deps=[hq_step, dedup_exact_step, dedup_fuzzy_step],
        ),
    )

    # ############################################################
    # Tokenize

    tokenize_step = StepSpec(
        fn=lambda output_path: tokenize_fn(
            train_paths=[consolidate_step.output_path],
            validation_paths=[],
            cache_path=output_path,
            tokenizer="gpt2",
            zephyr_num_cpus=1,
            zephyr_memory=1 * 1024**2,  # 1MB
        ),
        meta=StepMeta(
            name="tokenize",
            output_path_prefix=prefix,
            hash_attrs={"tokenizer": "gpt2"},
            deps=[consolidate_step],
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
        fn=lambda output_path: run_levanter_train_lm_fn(
            output_path=output_path,
            resources=pod_config,
            env_vars=train_env_vars,
            train_config=TrainLmConfig(
                data=lm_data_config(("training_data", Artifact.load(tokenize_step, TokenizedMetadata))),
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
        ),
        meta=StepMeta(
            name="train",
            output_path_prefix=prefix,
            deps=[
                tokenize_step,
            ],
        ),
    )

    return [
        hq_step,
        lq_step,
        train_quality_step,
        dedup_exact_step,
        dedup_fuzzy_step,
        consolidate_step,
        tokenize_step,
        train_step,
    ]


@dataclasses.dataclass(frozen=True)
class NoMagicConfig:
    prefix: str | None = None
    dry_run: bool = False


@draccus.wrap()
def main(config: NoMagicConfig):
    try:
        bucket_prefix = config.prefix or "/tmp"

        if "uv run" in " ".join(sys.argv):
            raise RuntimeError("integration_nomagic_test.py must not be launched via `uv run`. Please run it directly.")

        import ray

        ray.init(
            resources={"head_node": 1},
            runtime_env={"working_dir": None},
            num_cpus=os.cpu_count(),
            _memory=1024 * 1024 * 1024 * 1024,  # 1TB
        )
        set_current_cluster(create_cluster("ray"))

        synth_data = "./tests/quickstart-data"

        experiment_prefix = f"{bucket_prefix}/quickstart-tests-nomagic"
        if os.path.exists(experiment_prefix):
            os.system(f"rm -rf {experiment_prefix}")

        steps = create_steps(prefix=experiment_prefix, synth_data=synth_data)

        StepRunner().run(steps, dry_run=config.dry_run)

        logger.info(f"Execution completed successfully. All outputs are in {experiment_prefix}")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
