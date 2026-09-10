# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Construct the bounded async N2 mechanism requests without submitting jobs."""

from dataclasses import replace
from urllib.parse import urlsplit

import yaml

from experiments.post_training import async_rl
from experiments.post_training.math_eval.bucket_launcher import BUCKET_ARGUMENT


def _east_object(uri: str) -> str:
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "s3"
        or parsed.netloc != "marin-us-east-02a"
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Qualification objects must use explicit east S3 URIs")
    return uri


def build_qualification(*, version: str, measurement_uri: str, checkpoint_seven: str | None = None):
    """Return a fingerprinted checkpoint step; continuation inputs remain explicit.

    The callback replaces the actual identity-bearing request before lazy
    fingerprinting/materialization. No old request or resolved output is edited.
    Native source, input coverage and checkpoint-byte audits remain separate.
    """
    _east_object(measurement_uri)
    if checkpoint_seven is not None:
        _east_object(checkpoint_seven)
        if checkpoint_seven.rstrip("/").split("/")[-1] != "global_step_7":
            raise ValueError("Continuation must select the original checkpoint seven explicitly")
    export_step, _ = async_rl.build_experiment(
        version=version,
        cluster="cw-us-east-02a",
        runner=async_rl.Runner.ASYNC,
        scale=async_rl.Scale.SCREENING,
        completion="model",
        timeout_seconds=1800,
        seed=17,
        staleness=0,
        weight_sync_interval=1,
        kl_loss=False,
        correction=async_rl.Correction.REGULAR_NO_TIS,
        minibatches=2,
        updates=8,
        eval_updates=8,
        first_token_admission=True,
        generation_workers=64,
        serial_engine_startup=True,
        grad_cosine=True,
        grad_cosine_store="gpu_fp32",
        dataloader_workers=0,
        epoch_seeded_shuffle=True,
        response_tokens=1024,
        eval_response_tokens=1024,
        context_tokens=2048,
        lr=2e-6,
        pool_artifact=BUCKET_ARGUMENT,
    )
    checkpoint_step = export_step.deps[0]
    original = checkpoint_step.build_config

    def build_config(ctx):
        config = original(ctx)
        recipe = yaml.safe_load(config.request.config_yaml)
        trainer = recipe["trainer"]
        trainer.update(
            ckpt_interval=7,
            dump_data_batch=True,
            measurement_guard_uri=measurement_uri,
            measurement_guard_resume_step=7 if checkpoint_seven is not None else None,
        )
        mode = "from_path" if checkpoint_seven is not None else "none"
        overrides = [*config.request.overrides, f"++trainer.resume_mode={mode}"]
        if checkpoint_seven is not None:
            overrides.append(f"++trainer.resume_path={checkpoint_seven}")
        return replace(
            config,
            request=replace(
                config.request, config_yaml=yaml.safe_dump(recipe, sort_keys=False), overrides=tuple(overrides)
            ),
        )

    runtime_args = {
        key: replace(value, max_retries=1, wandb_entity="dogml") for key, value in checkpoint_step.runtime_args.items()
    }
    return replace(checkpoint_step, build_config=build_config, runtime_args=runtime_args)
