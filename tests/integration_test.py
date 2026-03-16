# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os
import sys

import draccus
from fray import ResourceConfig, set_current_client
from fray.v2.ray_backend.backend import RayClient
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

from iris.logging import configure_logging

import threading

configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def _tee_fd(original_fd: int, log_file) -> None:
    """Replace a file descriptor with a pipe that tees to both the original fd and a log file.

    Works at the OS level so child processes (Ray workers) also get captured.
    """
    read_fd, write_fd = os.pipe()
    saved_fd = os.dup(original_fd)
    os.dup2(write_fd, original_fd)
    os.close(write_fd)

    def _pump():
        with os.fdopen(read_fd, "r", errors="replace") as reader:
            for line in reader:
                os.write(saved_fd, line.encode())
                log_file.write(line)
                log_file.flush()

    threading.Thread(target=_pump, daemon=True).start()


def _setup_log_tee(log_path: str) -> None:
    """Tee stdout and stderr to a log file at the given path."""
    log_fh = open(log_path, "w")
    _tee_fd(sys.stdout.fileno(), log_fh)
    _tee_fd(sys.stderr.fileno(), log_fh)


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
            worker_resources=ResourceConfig(cpu=1, ram="1g"),
        ),
    )
    dedup_fuzzy_document_step = ExecutorStep(
        name=os.path.join(prefix, "dedup_fuzzy_document"),
        fn=deduplicate,
        config=DedupConfig(
            input_paths=transform_hq_data_step,
            output_path=this_output_path(),
            mode=DedupMode.FUZZY_DOCUMENT,
            worker_resources=ResourceConfig(cpu=1, ram="1g"),
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
                # TODO: these 2 may collidate on canonical records, removing more data then necessary
                FilterConfig(
                    type=FilterType.REMOVE_SPANS,
                    attribute_path=dedup_exact_paragraph_step.cd("data"),
                    name="dup_spans",
                    attribute_filetype="vortex",
                    keep_if_missing=True,
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

        _setup_log_tee("/tmp/integration_test.log")

        experiment_prefix = "quickstart-tests"
        config = dataclasses.replace(
            config, prefix=bucket_prefix, executor_info_base_path=os.path.join(bucket_prefix, "experiments")
        )

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
        with set_current_client(RayClient()):
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
