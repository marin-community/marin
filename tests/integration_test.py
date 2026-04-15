# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full marin data pipeline integration test on an Iris cluster.

Standalone script (not pytest) so logs stream in real time.

Usage:
    uv run tests/integration_test.py \
        --controller-url http://localhost:10000

When MARIN_CI_S3_PREFIX is set, uploads test fixtures to S3 and submits
the executor as an Iris job so child jobs inherit S3 credentials.
Otherwise runs in-process against local filesystem.
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import fsspec
from fray import ResourceConfig, set_current_client
from fray.v2.iris_backend import FrayIrisClient
from fray.v2.types import Entrypoint, JobRequest, create_environment
from rigging.log_setup import configure_logging
from levanter.main.train_lm import TrainLmConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.trainer import TrainerConfig
from marin.execution.executor import (
    ExecutorMainConfig,
    ExecutorStep,
    executor_main,
    this_output_path,
)
from marin.execution.step_spec import StepSpec
from marin.processing.classification.consolidate import FilterConfig, FilterType, consolidate
from marin.processing.classification.deduplication.exact import dedup_exact_paragraph
from marin.processing.tokenize import lm_data_config
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from marin.schemas.web.convert import ResiliparseConfig
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from marin.transform.simple_html_to_md.process import SimpleHtmlToMdConfig, html_to_md

configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SYNTH_DATA = REPO_ROOT / "tests" / "quickstart-data"

_S3_ENV_KEYS = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL", "FSSPEC_S3"]


def create_steps(prefix: str, synth_data: str) -> list[ExecutorStep]:
    """Build the full marin data pipeline as executor steps."""

    # Transform HTML to markdown
    transform_hq_data_spec = StepSpec(
        name=os.path.join(prefix, "hq-transformed"),
        hash_attrs={"extract_method": "resiliparse"},
        fn=lambda output_path: html_to_md(
            SimpleHtmlToMdConfig(
                input_path=os.path.join(synth_data, "pos"),
                output_path=output_path,
                extract_method="resiliparse",
                config=ResiliparseConfig(),
            )
        ),
    )
    transform_lq_data_spec = StepSpec(
        name=os.path.join(prefix, "lq-transformed"),
        hash_attrs={"extract_method": "resiliparse"},
        fn=lambda output_path: html_to_md(
            SimpleHtmlToMdConfig(
                input_path=os.path.join(synth_data, "neg"),
                output_path=output_path,
                extract_method="resiliparse",
                config=ResiliparseConfig(),
            )
        ),
    )
    transform_hq_data_step = transform_hq_data_spec.as_executor_step()
    transform_lq_data_step = transform_lq_data_spec.as_executor_step()

    # Dedup (exact only — fuzzy dedup has 4 iterative rounds of pod scheduling on K8s)
    dedup_exact_paragraph_spec = StepSpec(
        name=os.path.join(prefix, "dedup_exact_paragraph"),
        hash_attrs={"mode": "exact_paragraph"},
        deps=[transform_hq_data_spec],
        fn=lambda output_path: dedup_exact_paragraph(
            input_paths=transform_hq_data_spec.output_path,
            output_path=output_path,
            max_parallelism=4,
            worker_resources=ResourceConfig(cpu=1, ram="1g"),
        ),
    )
    dedup_exact_paragraph_step = dedup_exact_paragraph_spec.as_executor_step()

    # Consolidate
    consolidate_spec = StepSpec(
        name=os.path.join(prefix, "cleaned"),
        deps=[transform_hq_data_spec, dedup_exact_paragraph_spec],
        fn=lambda output_path: consolidate(
            input_path=transform_hq_data_spec.output_path,
            output_path=output_path,
            filters=[
                FilterConfig(
                    type=FilterType.REMOVE_SPANS,
                    attribute_path=f"{dedup_exact_paragraph_spec.output_path}/data",
                    name="dup_spans",
                    attribute_filetype="parquet",
                    keep_if_missing=True,
                ),
            ],
        ),
    )
    consolidate_step = consolidate_spec.as_executor_step()

    # Tokenize
    tokenize_step = ExecutorStep(
        name=os.path.join(prefix, "tokenized"),
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[consolidate_step],
            validation_paths=[],
            cache_path=this_output_path(),
            tokenizer="gpt2",
        ),
    )

    # Train (tiny model for validation)
    train_step = ExecutorStep(
        name=os.path.join(prefix, "train"),
        fn=run_levanter_train_lm,
        config=TrainLmOnPodConfig(
            output_path=this_output_path(),
            resources=ResourceConfig.with_cpu(),
            env_vars={
                "WANDB_API_KEY": "",
                "WANDB_MODE": "disabled",
                "JAX_TRACEBACK_FILTERING": "off",
            },
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

    return [
        transform_hq_data_step,
        transform_lq_data_step,
        dedup_exact_paragraph_step,
        consolidate_step,
        tokenize_step,
        train_step,
    ]


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def _upload_tree(local_root: Path, s3_dest: str) -> None:
    fs, _ = fsspec.core.url_to_fs(s3_dest)
    for path in local_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_root)
        fs.put(str(path), f"{s3_dest}/{rel}")


def _rm_s3(s3_prefix: str) -> None:
    fs, _ = fsspec.core.url_to_fs(s3_prefix)
    try:
        fs.rm(s3_prefix, recursive=True)
    except FileNotFoundError:
        pass


def _s3_env_vars() -> dict[str, str]:
    return {k: os.environ[k] for k in _S3_ENV_KEYS if k in os.environ}


# ---------------------------------------------------------------------------
# Executor entry point (runs inside the Iris job on remote clusters)
# ---------------------------------------------------------------------------


def _run_executor(prefix: str, synth_data: str) -> None:
    config = ExecutorMainConfig(
        prefix=prefix,
        executor_info_base_path=f"{prefix}/experiments",
    )
    steps = create_steps("quickstart-tests", synth_data)
    executor_main(config, steps=steps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run full marin pipeline on Iris")
    parser.add_argument("--controller-url", required=True)
    args = parser.parse_args()

    s3_base = os.environ.get("MARIN_CI_S3_PREFIX")

    if s3_base:
        run_id = f"marin-itest-{uuid.uuid4().hex[:8]}"
        prefix = f"{s3_base}/{run_id}"
        synth_data = f"{prefix}/quickstart-data"
        logger.info("Uploading test fixtures to %s", synth_data)
        _upload_tree(LOCAL_SYNTH_DATA, synth_data)
        cleanup = lambda: _rm_s3(prefix)  # noqa: E731
    else:
        prefix = tempfile.mkdtemp(prefix="iris-marin-itest-")
        synth_data = str(LOCAL_SYNTH_DATA)
        cleanup = lambda: shutil.rmtree(prefix, ignore_errors=True)  # noqa: E731

    os.environ["MARIN_PREFIX"] = prefix
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_API_KEY"] = ""
    os.environ["JAX_TRACEBACK_FILTERING"] = "off"

    try:
        iris_client = FrayIrisClient(
            controller_address=args.controller_url,
            workspace=REPO_ROOT,
        )

        if s3_base:
            logger.info("Submitting executor as Iris job (S3 mode)")
            env_vars = {
                "MARIN_PREFIX": prefix,
                "WANDB_MODE": "disabled",
                "WANDB_API_KEY": "",
                "JAX_TRACEBACK_FILTERING": "off",
                **_s3_env_vars(),
            }

            with set_current_client(iris_client):
                handle = iris_client.submit(
                    JobRequest(
                        name=f"marin-itest-{uuid.uuid4().hex[:8]}",
                        entrypoint=Entrypoint.from_callable(
                            _run_executor,
                            args=(prefix, synth_data),
                        ),
                        resources=ResourceConfig.with_cpu(),
                        environment=create_environment(env_vars=env_vars),
                    )
                )
                handle.wait(raise_on_failure=True, stream_logs=True)
        else:
            logger.info("Running executor in-process (local mode)")
            config = ExecutorMainConfig(
                prefix=prefix,
                executor_info_base_path=f"{prefix}/experiments",
            )
            steps = create_steps("quickstart-tests", synth_data)
            with set_current_client(iris_client):
                executor_main(config, steps=steps)

        logger.info("Pipeline completed successfully")
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
