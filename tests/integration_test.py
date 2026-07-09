# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full marin data pipeline integration test on an Iris cluster.

Standalone script (not pytest) so logs stream in real time.

Usage:
    uv run tests/integration_test.py \
        --controller-url http://localhost:10000

When MARIN_CI_S3_PREFIX is set, uploads test fixtures to S3 and submits
the pipeline as an Iris job so child jobs inherit S3 credentials.
Otherwise runs in-process against local filesystem.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

from fray import ResourceConfig, set_current_client
from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, create_environment
from levanter.main.train_lm import TrainLmConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.trainer import TrainerConfig
from marin.datakit.normalize import NormalizedData, normalize_step
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.consolidate import FilterConfig, FilterType, consolidate
from marin.processing.classification.deduplication.exact import dedup_exact_paragraph
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from marin.schemas.web.convert import ResiliparseConfig
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from marin.transform.simple_html_to_md.process import SimpleHtmlToMdConfig, html_to_md
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging

configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SYNTH_DATA = REPO_ROOT / "tests" / "quickstart-data"

# Reduced GPT-2 tokenizer already vendored in the repo for offline tests. Reused here so the
# local-mode pipeline never touches HF Hub (frequently unavailable in CI). `load_tokenizer`
# short-circuits a local directory, so pointing the tokenize step at it skips Hub staging.
LOCAL_TOKENIZER_FILE = REPO_ROOT / "lib" / "levanter" / "tests" / "gpt2_tokenizer.json"
LOCAL_TOKENIZER_CONFIG = REPO_ROOT / "lib" / "levanter" / "tests" / "gpt2_tokenizer_config.json"
# Vocab size of the reduced fixture above (matches lib/levanter/tests/conftest.py).
LOCAL_TOKENIZER_VOCAB_SIZE = 5027

_S3_ENV_KEYS = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL", "FSSPEC_S3"]


def _materialize_local_tokenizer(dest_dir: Path) -> str:
    """Lay out the reduced GPT-2 tokenizer fixture as a loadable tokenizer directory.

    ``config.json`` gives ``AutoTokenizer`` the ``model_type`` it needs to resolve the
    GPT-2 class offline; ``tokenizer_config.json`` is a minimal, well-formed config so the
    train step's ``save_pretrained`` round-trip succeeds. Copying the raw ``tokenizer.json``
    into ``tokenizer_config.json`` would load fine but fail to re-serialize on save.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(LOCAL_TOKENIZER_FILE, dest_dir / "tokenizer.json")
    shutil.copy(LOCAL_TOKENIZER_CONFIG, dest_dir / "tokenizer_config.json")
    (dest_dir / "config.json").write_text(json.dumps({"model_type": "gpt2", "vocab_size": LOCAL_TOKENIZER_VOCAB_SIZE}))
    return str(dest_dir)


def create_steps(prefix: str, synth_data: str, tokenizer: str) -> list[StepSpec]:
    """Build the full marin data pipeline as lazy ``StepSpec`` artifacts.

    ``tokenizer`` is an HF model name or a local directory of tokenizer files.
    The local-mode runner passes a vendored fixture directory so the pipeline
    does not depend on HF Hub.
    """

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

    normalize_hq_spec = normalize_step(
        name=os.path.join(prefix, "hq-normalized"),
        download=transform_hq_data_spec,
        file_extensions=(".jsonl.gz",),
    )

    # Dedup (exact only — fuzzy dedup has 4 iterative rounds of pod scheduling on K8s)
    dedup_exact_paragraph_spec = StepSpec(
        name=os.path.join(prefix, "dedup_exact_paragraph"),
        hash_attrs={"mode": "exact_paragraph"},
        deps=[normalize_hq_spec],
        fn=lambda output_path: dedup_exact_paragraph(
            input_paths=[read_artifact(normalize_hq_spec.output_path, NormalizedData).main_output_dir],
            output_path=output_path,
            max_parallelism=4,
            worker_resources=ResourceConfig(cpu=1, ram="1g"),
        ),
    )

    # Consolidate
    consolidate_spec = StepSpec(
        name=os.path.join(prefix, "cleaned"),
        deps=[normalize_hq_spec, dedup_exact_paragraph_spec],
        fn=lambda output_path: consolidate(
            input_path=read_artifact(normalize_hq_spec.output_path, NormalizedData).main_output_dir,
            output_path=output_path,
            # Normalize emits parquet; override the jsonl.gz default.
            filetype="parquet",
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

    # Tokenize
    tokenize_spec = StepSpec(
        name=os.path.join(prefix, "tokenized"),
        hash_attrs={"tokenizer": tokenizer},
        deps=[consolidate_spec],
        fn=lambda output_path: tokenize(
            TokenizeConfig(
                train_paths=[consolidate_spec.output_path],
                validation_paths=[],
                cache_path=output_path,
                tokenizer=tokenizer,
            )
        ),
    )

    # A single-component data mixture over the tokenized cache, for the train step.
    train_data_config = lm_mixture_data_config(
        components={
            "quickstart": TokenizeConfig(
                train_paths=[consolidate_spec.output_path],
                validation_paths=[],
                cache_path=tokenize_spec.output_path,
                tokenizer=tokenizer,
            )
        },
        weights={"quickstart": 1.0},
    )

    # Train (tiny model for validation)
    train_spec = StepSpec(
        name=os.path.join(prefix, "train"),
        deps=[tokenize_spec],
        fn=lambda output_path: run_levanter_train_lm(
            TrainLmOnPodConfig(
                output_path=output_path,
                resources=ResourceConfig.with_cpu(),
                env_vars={
                    "WANDB_API_KEY": "",
                    "WANDB_MODE": "disabled",
                    "JAX_TRACEBACK_FILTERING": "off",
                },
                train_config=TrainLmConfig(
                    data=train_data_config,
                    # hf_save_steps=2 (not 1): at num_train_steps=2, final info.step=1 doesn't match
                    # every=2, so Trainer.train's end-of-train run_hooks(force=True) flushes the HF
                    # save exactly once. hf_save_steps=1 would double-fire on info.step=1.
                    hf_save_steps=2,
                    model=Gpt2Config(
                        num_layers=2,
                        num_heads=2,
                        max_seq_len=64,
                        hidden_dim=32,
                        # Load the converter's tokenizer from the same source as the data so the
                        # train step's HF-checkpoint save never reaches HF Hub in local mode.
                        tokenizer=tokenizer,
                    ),
                    trainer=TrainerConfig(
                        train_batch_size=8, num_train_steps=2, max_eval_batches=1, require_accelerator=False
                    ),
                ),
            )
        ),
    )

    return [
        transform_hq_data_spec,
        normalize_hq_spec,
        dedup_exact_paragraph_spec,
        consolidate_spec,
        tokenize_spec,
        train_spec,
    ]


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def _upload_tree(local_root: Path, s3_dest: str) -> None:
    for path in local_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_root)
        StoragePath(f"{s3_dest}/{rel}").upload_from(str(path))


def _rm_s3(s3_prefix: str) -> None:
    try:
        StoragePath(s3_prefix).rmtree()
    except FileNotFoundError:
        pass


def _s3_env_vars() -> dict[str, str]:
    return {k: os.environ[k] for k in _S3_ENV_KEYS if k in os.environ}


# ---------------------------------------------------------------------------
# Pipeline entry point (runs inside the Iris job on remote clusters)
# ---------------------------------------------------------------------------


def _run_pipeline(prefix: str, synth_data: str, tokenizer: str) -> None:
    # Step output paths resolve against MARIN_PREFIX; pin it for the job process.
    os.environ["MARIN_PREFIX"] = prefix
    steps = create_steps("quickstart-tests", synth_data, tokenizer)
    StepRunner().run(steps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run full marin pipeline on Iris")
    parser.add_argument("--controller-url", required=True)
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Leave the run's output directory in place when the test exits (default: remove it).",
    )
    args = parser.parse_args()

    s3_base = os.environ.get("MARIN_CI_S3_PREFIX")

    if s3_base:
        run_id = f"marin-itest-{uuid.uuid4().hex[:8]}"
        prefix = f"{s3_base}/{run_id}"
        synth_data = f"{prefix}/quickstart-data"
        logger.info("Uploading test fixtures to %s", synth_data)
        _upload_tree(LOCAL_SYNTH_DATA, synth_data)
        cleanup = lambda: _rm_s3(prefix)  # noqa: E731
        # The vendored tokenizer fixture lives on the submitting host's filesystem,
        # which remote Zephyr pods cannot see, so S3 mode stages "gpt2" from HF Hub.
        tokenizer = "gpt2"
    else:
        prefix = tempfile.mkdtemp(prefix="iris-marin-itest-")
        synth_data = str(LOCAL_SYNTH_DATA)
        cleanup = lambda: shutil.rmtree(prefix, ignore_errors=True)  # noqa: E731
        tokenizer = _materialize_local_tokenizer(Path(prefix) / "gpt2-tokenizer")

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
            logger.info("Submitting pipeline as Iris job (S3 mode)")
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
                            _run_pipeline,
                            args=(prefix, synth_data, tokenizer),
                        ),
                        resources=ResourceConfig.with_cpu(),
                        environment=create_environment(env_vars=env_vars),
                    )
                )
                handle.wait(raise_on_failure=True, stream_logs=True)
        else:
            logger.info("Running pipeline in-process (local mode)")
            steps = create_steps("quickstart-tests", synth_data, tokenizer)
            with set_current_client(iris_client):
                StepRunner().run(steps)

        logger.info("Pipeline completed successfully")
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)
    finally:
        if args.no_cleanup:
            logger.info("Skipping cleanup; output directory preserved at: %s", prefix)
        else:
            cleanup()


if __name__ == "__main__":
    main()
