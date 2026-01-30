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

import dataclasses
import logging
import os
import sys

import draccus
from fray.v2 import ResourceConfig, set_current_client
from fray.v2.ray.backend import RayClient
import humanfriendly
from levanter.main.train_lm import TrainLmConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.trainer import TrainerConfig
from marin.execution.executor import (
    ExecutorMainConfig,
    ExecutorStep,
    executor_main,
    this_output_path,
    versioned,
)
from marin.processing.classification.consolidate import FilterConfig, FilterType, consolidate, ConsolidateConfig
from marin.processing.classification.dataset_utils import DatasetConfig
from marin.processing.classification.deduplication.dedup_commons import DedupConfig, DedupMode, deduplicate
from marin.processing.classification.fasttext.train_fasttext import (
    TrainFasttextClassifierConfig,
    train,
)
from marin.processing.classification.inference import InferenceConfig, run_inference
from marin.processing.tokenize import lm_data_config
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from marin.schemas.web.convert import ResiliparseConfig
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from marin.transform.simple_html_to_md.process import SimpleHtmlToMdConfig, html_to_md

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_steps(prefix: str, synth_data: str) -> list[ExecutorStep]:
    # ############################################################
    # Transform HTML to text

    transform_hq_data_step = ExecutorStep(
        name=os.path.join(prefix, "hq-transformed"),
        fn=html_to_md,
        config=SimpleHtmlToMdConfig(
            input_path=os.path.join(synth_data, "pos"),
            output_path=this_output_path(),
            extract_method=versioned("resiliparse"),
            config=ResiliparseConfig(),
        ),
    )

    transform_lq_data_step = ExecutorStep(
        name=os.path.join(prefix, "lq-transformed"),
        fn=html_to_md,
        config=SimpleHtmlToMdConfig(
            input_path=os.path.join(synth_data, "neg"),
            output_path=this_output_path(),
            extract_method=versioned("resiliparse"),
            config=ResiliparseConfig(),
        ),
    )

    # ############################################################
    # Train quality classifier

    train_quality_step = ExecutorStep(
        name=os.path.join(prefix, "quality-classifier"),
        fn=train,
        config=TrainFasttextClassifierConfig(
            datasets=[
                DatasetConfig(
                    input_doc_path=transform_hq_data_step,
                    label="hq",
                    sampling_rate=1.0,
                ),
                DatasetConfig(
                    input_doc_path=transform_lq_data_step,
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
    )

    ############################################################
    # Run inference with quality classifier

    inference_hq_step = ExecutorStep(
        name=os.path.join(prefix, "hq-inference"),
        fn=run_inference,
        config=InferenceConfig(
            input_path=transform_hq_data_step,
            output_path=this_output_path(),
            model_name=train_quality_step,
            model_type="fasttext",
            attribute_name="quickstart-fasttext-quality-hq",
        ),
    )

    inference_lq_step = ExecutorStep(
        name=os.path.join(prefix, "lq-inference"),
        fn=run_inference,
        config=InferenceConfig(
            input_path=transform_lq_data_step,
            output_path=this_output_path(),
            model_name=train_quality_step,
            model_type="fasttext",
            attribute_name="quickstart-fasttext-quality-lq",
        ),
    )

    ############################################################
    # Deduplicate

    dedup_exact_paragraph_step = ExecutorStep(
        name=os.path.join(prefix, "dedup_exact_paragraph"),
        fn=deduplicate,
        config=DedupConfig(
            input_paths=transform_hq_data_step,
            output_path=this_output_path(),
            mode=DedupMode.EXACT_PARAGRAPH,
            ray_memory=humanfriendly.parse_size("1GB", binary=True),
            ray_num_cpus=1,
        ),
    )
    dedup_fuzzy_document_step = ExecutorStep(
        name=os.path.join(prefix, "dedup_fuzzy_document"),
        fn=deduplicate,
        config=DedupConfig(
            input_paths=transform_hq_data_step,
            output_path=this_output_path(),
            mode=DedupMode.FUZZY_DOCUMENT,
            ray_memory=humanfriendly.parse_size("1GB", binary=True),
            ray_num_cpus=1,
        ),
    )

    ############################################################
    # Consolidate

    consolidate_step = ExecutorStep(
        name=os.path.join(prefix, "cleaned"),
        fn=consolidate,
        config=ConsolidateConfig(
            input_path=transform_hq_data_step,
            output_path=this_output_path(),
            # TODO (rav): add quality filters
            filters=[
                FilterConfig(
                    type=FilterType.REMOVE_SPANS,
                    attribute_path=dedup_exact_paragraph_step.cd("data"),
                    name=str(DedupMode.EXACT_PARAGRAPH),
                ),
                FilterConfig(
                    type=FilterType.REMOVE_DOC,
                    attribute_path=dedup_fuzzy_document_step.cd("data"),
                    name=str(DedupMode.FUZZY_DOCUMENT),
                ),
            ],
        ),
    )

    ############################################################
    # Tokenize
    tokenizer = "gpt2"

    tokenize_config = TokenizeConfig(
        train_paths=[consolidate_step],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=tokenizer,
        zephyr_num_cpus=1,
        zephyr_memory=humanfriendly.parse_size("1MB", binary=True),
    )

    tokenize_step = ExecutorStep(
        name=os.path.join(prefix, "tokenized"),
        description=f"Tokenize raw text using the {tokenizer} tokenizer.",
        fn=tokenize,
        config=tokenize_config,
    )

    # ############################################################
    # Training
    train_env_vars = {
        "WANDB_API_KEY": "",
        "WANDB_MODE": "disabled",
        "JAX_TRACEBACK_FILTERING": "off",
    }

    pod_config = ResourceConfig.with_cpu()

    train_step = ExecutorStep(
        name=os.path.join(prefix, "train"),
        fn=run_levanter_train_lm,
        config=TrainLmOnPodConfig(
            output_path=this_output_path(),
            resources=pod_config,
            env_vars=train_env_vars,
            train_config=TrainLmConfig(
                data=lm_data_config(tokenize_step, permutation_type="linear"),
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


@draccus.wrap()
def main(config: ExecutorMainConfig):
    try:
        if config.prefix is not None:
            bucket_prefix = config.prefix
        else:
            bucket_prefix = "/tmp"  # Default to a temporary directory

        experiment_prefix = "quickstart-tests"
        config = dataclasses.replace(
            config, prefix=bucket_prefix, executor_info_base_path=os.path.join(bucket_prefix, "experiments")
        )

        # start Ray explicitly and set it as the current cluster
        # N.B. This script must not be launched via `uv run`, or Ray will prefer to use `uv` for all execution
        # ignoring package dependencies specified in each step.
        if "uv run" in " ".join(sys.argv):
            raise RuntimeError("integration_test.py must not be launched via `uv run`. Please run it directly.")
        import ray

        ray.init(resources={"head_node": 1}, runtime_env={"working_dir": None}, num_cpus=os.cpu_count())
        set_current_client(RayClient())

        # path to synthetic test data
        synth_data: str = "./tests/quickstart-data"
        # delete all previous runs
        if os.path.exists(os.path.join(bucket_prefix, experiment_prefix)):
            os.system(f"rm -rf {os.path.join(bucket_prefix, experiment_prefix)}")
        steps = create_steps(experiment_prefix, synth_data)
        config = dataclasses.replace(config)
        executor_main(config, steps=steps)
        logger.info(f"Execution completed successfully. All outputs are in {bucket_prefix}/{experiment_prefix}")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise e


if __name__ == "__main__":
    main()
