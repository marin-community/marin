# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nightly Grug multislice smoke run.

Config is driven by env vars set in the GitHub Actions workflow env block and
forwarded to the Iris parent job.

    RUN_ID                         unique run identifier
    GRUG_MULTISLICE_BATCH_SIZE     global train batch size
    GRUG_MULTISLICE_CE_IMPL        optional loss implementation override
    GRUG_MULTISLICE_MAX_SEQ_LEN    sequence length
    GRUG_MULTISLICE_OUTPUT_PATH    optional explicit output path
    GRUG_MULTISLICE_REGION         GCP region for TPU slices
    GRUG_MULTISLICE_SLICE_COUNT    TPU slice count
    GRUG_MULTISLICE_STEPS          training steps
    GRUG_MULTISLICE_TPU_TYPE       TPU type per slice
"""

import dataclasses
import logging
import os

import fsspec
from fray.cluster import ResourceConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture
from marin.training.training import LevanterCheckpoint

from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.proofpile import proofpile_dataset
from experiments.datasets.starcoder import starcoder_dataset
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.base.launch import GRUG_130M_MODEL, GrugBaseLaunchConfig, run_grug_base_trial
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

CANARY_STEP_NAME = "grug/multislice-smoke"

DEFAULT_REGION = "us-east5"
DEFAULT_TPU_TYPE = "v6e-8"
DEFAULT_SLICE_COUNT = 2
DEFAULT_STEPS = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_SEQ_LEN = 4096
CHILD_ENV_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_ENDPOINT_URL",
    "AWS_SECRET_ACCESS_KEY",
    "HF_TOKEN",
    "MARIN_PREFIX",
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_MODE",
    "WANDB_PROJECT",
)

# Nemotron CC mixture weights: the corpus's TiB proportions, plus starcoder and
# proof-pile at their published weights. Policy lives here, in the experiment.
_NEMOTRON_WEIGHTS = {
    "hq_actual": 0.91351,
    "hq_synth": 2.72,
    "medium_high": 0.82471,
    "medium": 3.38,
    "medium_low": 1.54,
    "low_actual": 0.70123,
    "low_synth": 0.62771,
}
_STARCODER_WEIGHT = 0.25
_PROOFPILE_WEIGHT = 0.055


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "")
    return int(raw) if raw else default


def _run_id() -> str:
    return os.environ.get("RUN_ID") or os.environ.get("GRUG_RUN_ID") or "grug-multislice-smoke"


def multislice_smoke_resources() -> ResourceConfig:
    region = os.environ.get("GRUG_MULTISLICE_REGION", DEFAULT_REGION)
    tpu_type = os.environ.get("GRUG_MULTISLICE_TPU_TYPE", DEFAULT_TPU_TYPE)
    slice_count = _env_int("GRUG_MULTISLICE_SLICE_COUNT", DEFAULT_SLICE_COUNT)
    return ResourceConfig.with_tpu(tpu_type, slice_count=slice_count, regions=[region])


def build() -> ArtifactStep[LevanterCheckpoint]:
    """The Grug multislice smoke run as a lazy checkpoint, configured from the env.

    The Nemotron mix and the WandB ``replicate_path`` depend on the run context, so
    they are assembled inside ``build_config``; the TPU slice count/region is a
    run-arg, so it never bears on identity. ``GRUG_MULTISLICE_OUTPUT_PATH`` pins the
    output to an explicit location.
    """
    max_seq_len = _env_int("GRUG_MULTISLICE_MAX_SEQ_LEN", DEFAULT_MAX_SEQ_LEN)
    steps = _env_int("GRUG_MULTISLICE_STEPS", DEFAULT_STEPS)
    batch_size = _env_int("GRUG_MULTISLICE_BATCH_SIZE", DEFAULT_BATCH_SIZE)
    loss_implementation = os.environ.get("GRUG_MULTISLICE_CE_IMPL") or None
    run_id = _run_id()
    model = dataclasses.replace(GRUG_130M_MODEL, max_seq_len=max_seq_len)
    override_output_path = _override_output_path()

    nem = nemotron_datasets(tokenizer=llama3_tokenizer)
    train = {nem[split]: weight for split, weight in _NEMOTRON_WEIGHTS.items()}
    train[starcoder_dataset(tokenizer=llama3_tokenizer)] = _STARCODER_WEIGHT
    train[proofpile_dataset(tokenizer=llama3_tokenizer)] = _PROOFPILE_WEIGHT
    validation = [
        *paloma_datasets(tokenizer=llama3_tokenizer).values(),
        *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
    ]

    def build_config(ctx: StepContext) -> GrugBaseLaunchConfig:
        return GrugBaseLaunchConfig(
            model=model,
            data=mixture(ctx, train, validation=validation),
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=steps,
            batch_size=batch_size,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(
                entity=os.environ.get("WANDB_ENTITY") or None,
                project=os.environ.get("WANDB_PROJECT", "marin"),
                tags=["grug", "multislice", "smoke"],
                group="grug-multislice-smoke",
                name=None,
                mode=os.environ.get("WANDB_MODE") or None,
                replicate_path=ctx.output_path,
            ),
            optimizer=AdamConfig(
                learning_rate=3e-3,
                weight_decay=0.1,
                lr_schedule="cosine",
                decay=0.2,
                min_lr_ratio=0.1,
                warmup=1000,
            ),
            z_loss_weight=1e-4,
            ema_beta=None,
            log_every=1,
            eval_batch_size=None,  # disables perplexity eval
            loss_implementation=loss_implementation,
        )

    return ArtifactStep(
        name=CANARY_STEP_NAME,
        version="2026.06.28",
        artifact_type=LevanterCheckpoint,
        run=run_grug_base_trial,
        build_config=build_config,
        deps=(*train, *validation),
        runtime_args={"train_resources": multislice_smoke_resources()},
        override_path=override_output_path,
    )


def _child_env_vars() -> dict[str, str]:
    return {key: value for key in CHILD_ENV_KEYS if (value := os.environ.get(key))}


def _override_output_path() -> str | None:
    """Read ``GRUG_MULTISLICE_OUTPUT_PATH`` and validate it.

    If the env var is unset, returns ``None`` and the run lands at its canonical
    ``{prefix}/{name}/{version}`` path. If the env var is set, it must be non-empty:
    an empty value is a relative pin joined with ``MARIN_PREFIX``, which would point
    the pre-submit wipe at the prefix root (e.g. ``gs://marin-us-east5``) and
    recursively delete it.
    """
    if "GRUG_MULTISLICE_OUTPUT_PATH" not in os.environ:
        return None
    value = os.environ["GRUG_MULTISLICE_OUTPUT_PATH"]
    if not value:
        raise ValueError(
            "GRUG_MULTISLICE_OUTPUT_PATH is set but empty; "
            "unset it to use the default canary path, or provide an absolute output path."
        )
    return value


def wipe_path_if_exists(path: str) -> None:
    """Recursively delete ``path`` via fsspec if it exists; no-op otherwise."""
    fs, _, (plain_path,) = fsspec.get_fs_token_paths(path)
    if not fs.exists(plain_path):
        logger.info("Canary output path %s does not exist; nothing to wipe.", path)
        return
    logger.info("Wiping canary output directory %s before submit.", path)
    fs.rm(plain_path, recursive=True)


def _wipe_canary_output(checkpoint: ArtifactStep[LevanterCheckpoint]) -> None:
    """Delete the canary's output directory if it exists.

    The canary writes to a path that is stable across runs
    (e.g. ``gs://marin-us-east5/grug/multislice-smoke/v1/``). When a previous run
    left a checkpoint behind, Levanter resumes it and the final forced checkpoint
    write trips TensorStore's ``chunk_shape`` lock if the sharding has drifted since
    the prior write. The canary is meant to be self-contained, so wipe the directory
    before submitting the training job — each invocation starts from an empty path.
    """
    wipe_path_if_exists(checkpoint.path())


def main() -> None:
    # The StepRunner runs the grug launcher inline in this process and dispatches
    # the training job from it; that dispatch builds the job environment from this
    # process's env. Re-export the CI-forwarded child env here so every forwarded
    # knob (R2/AWS creds, HF_TOKEN, WANDB credentials, MARIN_PREFIX) reaches the
    # inline launcher and the training job it submits.
    os.environ.update(_child_env_vars())
    checkpoint = build()
    _wipe_canary_output(checkpoint)
    StepRunner().run([checkpoint.lower()])


if __name__ == "__main__":
    main()
