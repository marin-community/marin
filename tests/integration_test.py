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
from fray.cluster import create_cluster, set_current_cluster
import humanfriendly
from marin.execution.executor import (
    ExecutorMainConfig,
    ExecutorStep,
    executor_main,
    this_output_path,
    versioned,
)
from marin.execution_v2.executor_step_adapter import deferred_steps_to_executor_steps
from marin.execution_v2.step import Step
from marin.processing.classification.consolidate import FilterConfig, FilterType, consolidate_fn
from marin.processing.classification.dataset_utils import DatasetConfig
from marin.processing.classification.deduplication.dedup_commons import DedupMode, deduplicate_fn
from marin.processing.classification.fasttext.train_fasttext import (
    train,
)
from marin.processing.tokenize.tokenize import tokenize_fn
from marin.schemas.web.convert import ResiliparseConfig
from marin.transform.simple_html_to_md.process import html_to_md

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_steps(*, experiment_prefix: str, bucket_prefix: str, synth_data: str) -> list[ExecutorStep]:
    # ############################################################
    # Transform HTML to text

    hq_ds = Step(html_to_md).defer(
        input_path=os.path.join(synth_data, "pos"),
        output_path=this_output_path(),
        extract_method="resiliparse",
        config=ResiliparseConfig(),
    )
    lq_ds = Step(html_to_md).defer(
        input_path=os.path.join(synth_data, "neg"),
        output_path=this_output_path(),
        extract_method="resiliparse",
        config=ResiliparseConfig(),
    )

    # ############################################################
    # Train quality classifier

    train_quality_step = Step(train).defer(
        datasets=[
            DatasetConfig(
                input_doc_path=hq_ds,
                label="hq",
                sampling_rate=1.0,
            ),
            DatasetConfig(
                input_doc_path=lq_ds,
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
    )

    ############################################################
    # Deduplicate

    deduped_exact_paragraph = Step(deduplicate_fn).defer(
        input_paths=hq_ds,
        output_path=this_output_path(),
        mode=versioned(DedupMode.EXACT_PARAGRAPH),
        ray_memory=humanfriendly.parse_size("1GB", binary=True),
        ray_num_cpus=1,
    )
    deduped_fuzzy_document = Step(deduplicate_fn).defer(
        input_paths=hq_ds,
        output_path=this_output_path(),
        mode=versioned(DedupMode.FUZZY_DOCUMENT),
        ray_memory=humanfriendly.parse_size("1GB", binary=True),
        ray_num_cpus=1,
    )

    ############################################################
    # Consolidate

    consolidated_ds = Step(consolidate_fn).defer(
        input_path=hq_ds,
        output_path=this_output_path(),
        # TODO (rav): add quality filters
        filters=[
            FilterConfig(
                type=FilterType.REMOVE_SPANS,
                attribute_path=deduped_exact_paragraph,
                name=str(DedupMode.EXACT_PARAGRAPH),
            ),
            FilterConfig(
                type=FilterType.REMOVE_DOC,
                attribute_path=deduped_fuzzy_document,
                name=str(DedupMode.FUZZY_DOCUMENT),
            ),
        ],
    )

    ############################################################
    # Tokenize
    tokenizer = "gpt2"

    tokenize_step = Step(tokenize_fn).defer(
        train_paths=[consolidated_ds],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=tokenizer,
        zephyr_num_cpus=1,
        zephyr_memory=humanfriendly.parse_size("1MB", binary=True),
    )

    # ############################################################
    # Training
    # train_env_vars = {
    #     "WANDB_API_KEY": "",
    #     "WANDB_MODE": "disabled",
    #     "JAX_TRACEBACK_FILTERING": "off",
    # }

    # pod_config = ResourceConfig.with_cpu()

    # train_step = ExecutorStep(
    #     name=os.path.join(prefix, "train"),
    #     fn=run_levanter_train_lm,
    #     config=TrainLmOnPodConfig(
    #         output_path=this_output_path(),
    #         resources=pod_config,
    #         env_vars=train_env_vars,
    #         train_config=TrainLmConfig(
    #             data=lm_data_config(tokenize_step, permutation_type="linear"),
    #             hf_save_steps=1,
    #             model=Gpt2Config(
    #                 num_layers=2,
    #                 num_heads=2,
    #                 max_seq_len=64,
    #                 hidden_dim=32,
    #             ),
    #             trainer=TrainerConfig(
    #                 train_batch_size=8, num_train_steps=2, max_eval_batches=1, require_accelerator=False
    #             ),
    #         ),
    #     ),
    # )

    ##### Evaluate

    # evaluate_step = evaluate_helm_on_step(train_step, ["mmlu"], max_eval_instances=10)

    return deferred_steps_to_executor_steps(
        hq_ds,
        lq_ds,
        train_quality_step,
        # inference_hq_step,
        # inference_lq_step,
        deduped_exact_paragraph,
        deduped_fuzzy_document,
        consolidated_ds,
        tokenize_step,
        # train_step,
        # evaluate_step,
    )


@draccus.wrap()
def main(config: ExecutorMainConfig):
    try:
        if config.prefix is not None:
            bucket_prefix = config.prefix
        else:
            bucket_prefix = "/tmp"  # Default to a temporary directory

        experiment_prefix = f"{bucket_prefix}/quickstart-tests"
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
        set_current_cluster(create_cluster("ray"))

        # path to synthetic test data
        synth_data: str = "./tests/quickstart-data"
        # delete all previous runs
        if os.path.exists(os.path.join(bucket_prefix, experiment_prefix)):
            os.system(f"rm -rf {os.path.join(bucket_prefix, experiment_prefix)}")
        steps = create_steps(experiment_prefix=experiment_prefix, bucket_prefix=bucket_prefix, synth_data=synth_data)
        config = dataclasses.replace(config)
        executor_main(config, steps=steps)
        logger.info(f"Execution completed successfully. All outputs are in {experiment_prefix}")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise e


if __name__ == "__main__":
    main()
