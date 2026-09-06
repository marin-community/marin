# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched sync/async Megatron Snowball/GSM8K using the regional SFT export.

Run the two-update gate before the 25-update qualification. Both keep the staged
export's thinking template and use four policy nodes plus one node per inference replica.
Check H100 and host-memory capacity before submitting in cw-us-east-02a.

Use --completion metrics for timing/quality experiments that need no saved model,
checkpoint to retain resumable state, or model (the default) to run the separate
HF export stage as well. W&B and Finelog remain enabled in every mode.

For response-budget calibration, use --context-tokens 8192 and
--eval-response-tokens 4096 for both --response-tokens 2048 and 4096.
Gate the larger budget before running qualification with it.
Use --scale cadence-gate --weight-sync-interval 2 --max-staleness-steps 1
for five updates with initial/final evaluation and a forced final publication.
Use --runner sync for the synchronous control, which publishes every update.
Use --inference-replicas 2 for two independent, node-local DP8/EP8 groups.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import cast

import click
import yaml
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact, is_mutable_version, validate_version
from marin.execution.fingerprint import canonical_json, fingerprint_hash
from marin.execution.lazy import ArtifactStep, StepContext, run
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLCheckpoint,
    SkyRLModel,
    SkyRLRetentionPolicy,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    SkyRLTrainingResult,
    skyrl_checkpoint_step,
    skyrl_metrics_step,
    skyrl_step,
)
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath, prefix_join
from transformers import AutoTokenizer
from zephyr.writers import write_parquet_file

from experiments.post_training.async_rl import (
    DATA_REVISION,
    SEED,
    VALIDATION_ROWS,
    Correction,
    Runner,
    apply_observation_options,
    validate_eval_interval,
    validate_regional_storage,
    validate_validation_window,
)
from experiments.post_training.curriculum_rl.launch import (
    SNOWBALL_MODEL,
    SNOWBALL_POLICY,
    SNOWBALL_SMOKE,
    rl_config_yaml,
)
from experiments.post_training.curriculum_rl.pool import (
    GSM8K_INSTRUCTION,
    SYSTEM_PROMPT,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    _gsm8k_records,
)

TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
CLUSTER = "cw-us-east-02a"
ROLE_PLAN = replace(SNOWBALL_SMOKE.role_plan, micro_train_batch_size_per_gpu=1)


class Scale(StrEnum):
    GATE = "gate"
    CADENCE_GATE = "cadence-gate"
    QUALIFICATION = "qualification"


@dataclass(frozen=True)
class SnowballDataConfig:
    output_path: str
    model_path: str
    dataset_revision: str = DATA_REVISION
    train_rows: int = 1024
    validation_rows: int = 128
    max_prompt_tokens: int = 1024
    system_prompt: str = field(init=False, default=SYSTEM_PROMPT)
    answer_instruction: str = field(init=False, default=GSM8K_INSTRUCTION)


@dataclass(frozen=True)
class SnowballWindowConfig:
    subset: SnowballDataConfig
    validation_offset: int


def write_snowball_window(config: SnowballWindowConfig) -> None:
    validate_validation_window(config.validation_offset, config.subset.validation_rows)
    write_snowball_gsm8k(config.subset, validation_offset=config.validation_offset)


def write_snowball_gsm8k(config: SnowballDataConfig, *, validation_offset: int = 0) -> None:
    """Validate all selected prompts with the actual export tokenizer and template.

    Only tokenizer metadata is downloaded. Keep hashes and prompt-length bounds
    alongside the disjoint row manifest; no weights are copied by this CPU stage.
    """
    hashes = {}
    row_ids = {}
    length_bounds = {}
    with tempfile.TemporaryDirectory() as directory:
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            content = StoragePath(prefix_join(config.model_path, name)).read_bytes()
            Path(directory, name).write_bytes(content)
            hashes[name] = hashlib.sha256(content).hexdigest()
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        for split, count, filename in (
            ("train", config.train_rows, TRAIN_FILENAME),
            ("test", config.validation_rows, VALIDATION_FILENAME),
        ):
            offset = validation_offset if split == "test" else 0
            records = _gsm8k_records(split, offset + count, revision=config.dataset_revision)[offset:]
            if offset and len(records) != count:
                raise ValueError("Locked validation window cannot be truncated")
            lengths = []
            for record in records:
                rendered = tokenizer.apply_chat_template(record["prompt"], tokenize=False, add_generation_prompt=True)
                if not rendered.endswith("<|start_think|>\n"):
                    raise ValueError("Snowball export must preserve its thinking generation prefix")
                tokens = tokenizer.apply_chat_template(
                    record["prompt"], tokenize=True, add_generation_prompt=True, return_dict=True
                )
                lengths.append(len(tokens["input_ids"]))
            if max(lengths) > config.max_prompt_tokens:
                raise ValueError(f"Snowball {split} prompt exceeds {config.max_prompt_tokens} tokens")
            write_parquet_file(records, prefix_join(config.output_path, filename))
            row_ids[split] = [f"{split}/{cast(dict, record['extra_info'])['index']}" for record in records]
            length_bounds[split] = {"min": min(lengths), "max": max(lengths), "count": len(lengths)}
    manifest = {
        "dataset": "openai/gsm8k",
        "revision": config.dataset_revision,
        "rows": row_ids,
        "tokenizer_sha256": hashes,
        "prompt_lengths": length_bounds,
    }
    if validation_offset:
        manifest["validation_window"] = {
            "purpose": "locked_holdout",
            "split": "test",
            "offset": validation_offset,
            "count": config.validation_rows,
            "excluded_development_rows": [0, VALIDATION_ROWS],
        }
    StoragePath(prefix_join(config.output_path, "selection.json")).write_text(json.dumps(manifest, sort_keys=True))


def training_config(
    scale: Scale,
    *,
    runner: Runner = Runner.ASYNC,
    inference_replicas: int = 1,
    response_tokens: int | None = None,
    eval_response_tokens: int | None = None,
    context_tokens: int | None = None,
    weight_sync_interval: int = 1,
    max_staleness_steps: int = 1,
    correction: Correction = Correction.BEHAVIOR_CLIP,
    initial_eval_repeat_count: int = 1,
    weight_change_probe: bool = False,
    epoch_seeded_shuffle: bool = False,
    study_steps: int | None = None,
    eval_interval: int | None = None,
) -> str:
    gate = scale is Scale.GATE
    if not isinstance(epoch_seeded_shuffle, bool):
        raise ValueError("epoch_seeded_shuffle must be a boolean")
    if inference_replicas < 1:
        raise ValueError("Inference replica count must be positive")
    if correction not in Correction:
        raise ValueError(f"Unknown correction mode: {correction}")
    steps = {Scale.GATE: 2, Scale.CADENCE_GATE: 5, Scale.QUALIFICATION: 25}[scale]
    if study_steps is not None:
        if scale is not Scale.QUALIFICATION or type(study_steps) is not int or study_steps <= 0:
            raise ValueError("study_steps must be positive and is only supported by the qualification scale")
        steps = study_steps
    validate_eval_interval(eval_interval, steps, enabled=not gate)
    if weight_sync_interval < 1 or max_staleness_steps < 0 or weight_sync_interval > max_staleness_steps + 1:
        raise ValueError("Weight sync interval must be positive and at most max_staleness_steps + 1")
    if runner is Runner.SYNC and weight_sync_interval != 1:
        raise ValueError("The synchronous runner publishes every update; weight_sync_interval must be 1")
    response_tokens = response_tokens if response_tokens is not None else (512 if gate else 2048)
    context_tokens = context_tokens if context_tokens is not None else (2048 if gate else 4096)
    eval_tokens = eval_response_tokens if eval_response_tokens is not None else response_tokens
    if min(response_tokens, eval_tokens) <= 0:
        raise ValueError("Training and evaluation response budgets must be positive")
    if context_tokens < SnowballDataConfig.max_prompt_tokens + max(response_tokens, eval_tokens):
        raise ValueError("Context budget must fit the validated prompt limit plus either response budget")
    preset = replace(
        SNOWBALL_SMOKE,
        role_plan=replace(ROLE_PLAN, num_inference_engines=inference_replicas),
        max_steps=steps,
        ckpt_interval=steps,
        eval_interval=eval_interval if eval_interval is not None else (-1 if gate else steps),
        request_window_tokens=context_tokens,
        max_new_tokens=response_tokens,
        micro_forward_batch_size_per_gpu=1,
    )
    config = yaml.safe_load(rl_config_yaml(preset))
    config["entrypoint"] = "standard" if runner is Runner.SYNC else "fully_async"
    trainer = config["trainer"]
    trainer.update(
        strategy="megatron",
        flash_attn=False,
        offload_optimizer_during_rollouts=False,
        gradient_checkpointing=True,
        policy_train_spans=True,
        generate_spans=True,
        async_spans=True,
        training_metrics=True,
        logger="wandb",
        tracker_commit_each_step=True,
        project_name="marin-async-non-agentic-rl",
        resume_mode=None,
        eval_before_train=not gate,
        eval_batch_size=128,
    )
    trainer["algorithm"].update(
        use_kl_loss=False, use_kl_in_reward=False, policy_loss_type="behavior_clip", use_tis=False
    )
    if correction == Correction.REGULAR_TIS:
        trainer["algorithm"].update(
            policy_loss_type="regular", use_tis=True, tis_imp_ratio_cap=2.0, require_rollout_logprobs=True
        )
    trainer["fully_async"] = {
        "max_staleness_steps": max_staleness_steps,
        "weight_sync_interval": weight_sync_interval,
        "num_parallel_generation_workers": 64,
        "admission_stall_timeout": 900,
    }
    megatron = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 2,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
    }
    trainer["policy"].pop("fsdp_config")
    trainer["policy"]["optimizer_config"]["lr"] = 1e-6
    trainer["policy"]["megatron_config"] = dict(megatron)
    trainer["ref"] = {"megatron_config": dict(megatron)}
    config["generator"].update(
        inference_engine_data_parallel_size=8,
        inference_engine_expert_parallel_size=8,
        inference_engine_node_local=True,
    )
    config["generator"]["sampling_params"]["logprobs"] = 0
    if eval_response_tokens is not None:
        config["generator"]["eval_sampling_params"] = {"max_generate_length": eval_response_tokens}
    config["generator"]["trajectory_retention"] = {
        "sample_count_per_step": 4,
        "always_retain_failures": False,
        "always_retain_non_terminating": False,
        "always_retain_loops": False,
        "max_bytes_per_step": 262144,
        "max_bytes_per_run": 4194304,
    }
    config["extra_env"] = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # Capture the selected transport at communicator initialization, without
        # enabling per-collective logging or changing NCCL transport selection.
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,NET",
    }
    apply_observation_options(
        config, initial_eval_repeat_count=initial_eval_repeat_count, weight_change_probe=weight_change_probe
    )
    if epoch_seeded_shuffle:
        config.setdefault("data", {})["epoch_seeded_shuffle"] = True
    return yaml.safe_dump(config, sort_keys=False)


def build_experiment(
    *,
    version: str,
    scale: Scale,
    timeout_seconds: int,
    completion: str = "model",
    runner: Runner = Runner.ASYNC,
    inference_replicas: int = 1,
    response_tokens: int | None = None,
    eval_response_tokens: int | None = None,
    context_tokens: int | None = None,
    weight_sync_interval: int = 1,
    max_staleness_steps: int = 1,
    correction: Correction = Correction.BEHAVIOR_CLIP,
    initial_eval_repeat_count: int = 1,
    weight_change_probe: bool = False,
    epoch_seeded_shuffle: bool = False,
    study_steps: int | None = None,
    eval_interval: int | None = None,
    seed: int = SEED,
    validation_offset: int = 0,
    validation_rows: int = VALIDATION_ROWS,
) -> ArtifactStep[SkyRLModel] | ArtifactStep[SkyRLCheckpoint] | ArtifactStep[SkyRLTrainingResult]:
    """Build Snowball training with the requested terminal artifact."""
    validate_version(version)
    if is_mutable_version(version) or timeout_seconds <= 0:
        raise ValueError("Use an immutable version and positive training deadline")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be between 0 and 2**32 - 1")
    validate_validation_window(validation_offset, validation_rows)
    data = ArtifactStep(
        name=user_owned_name(
            "documents/async-rl-snowball-gsm8k"
            + (f"-test{validation_offset}-{validation_rows}" if validation_offset else "")
        ),
        version=version,
        artifact_type=Artifact,
        deps=(SNOWBALL_MODEL,),
        run=remote(
            write_snowball_window if validation_offset else write_snowball_gsm8k,
            resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="32g"),
        ),
        build_config=lambda ctx: (
            SnowballWindowConfig(
                SnowballDataConfig(ctx.output_path, ctx.artifact_path(SNOWBALL_MODEL), validation_rows=validation_rows),
                validation_offset,
            )
            if validation_offset
            else SnowballDataConfig(output_path=ctx.output_path, model_path=ctx.artifact_path(SNOWBALL_MODEL))
        ),
    )
    config = training_config(
        scale,
        runner=runner,
        inference_replicas=inference_replicas,
        response_tokens=response_tokens,
        eval_response_tokens=eval_response_tokens,
        context_tokens=context_tokens,
        weight_sync_interval=weight_sync_interval,
        max_staleness_steps=max_staleness_steps,
        correction=correction,
        initial_eval_repeat_count=initial_eval_repeat_count,
        weight_change_probe=weight_change_probe,
        epoch_seeded_shuffle=epoch_seeded_shuffle,
        study_steps=study_steps,
        eval_interval=eval_interval,
    )
    role_plan = replace(ROLE_PLAN, num_inference_engines=inference_replicas)
    topology = SkyRLTopology(role_plan.policy_num_nodes + inference_replicas, 8, "H100", role_plan)
    identity = fingerprint_hash(
        canonical_json(
            {
                "config": config,
                "model": SNOWBALL_MODEL.fingerprint(),
                "tokenizer_revision": TOKENIZER_REVISION,
                "data": data.fingerprint(),
                "topology": topology,
                "seed": seed,
            }
        )
    )
    spec = SkyRLSpec(
        name=user_owned_name(f"checkpoints/async-rl/snowball-{runner.value}-{scale.value}-{identity}"),
        version=version,
        config_yaml=config,
        runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
        model=ArtifactHfModel(SNOWBALL_MODEL, SNOWBALL_POLICY.tokenizer_uri, TOKENIZER_REVISION, relative_path=""),
        train_data=(ArtifactDataSource(data, relative_path=TRAIN_FILENAME),),
        validation_data=(ArtifactDataSource(data, relative_path=VALIDATION_FILENAME),),
        topology=topology,
        retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
        seed=seed,
        overrides=("++trainer.hf_hub_repo_id=null",),
    )
    execution = IrisSkyRLExecution(
        cluster=CLUSTER,
        cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
        cpu=16,
        # Megatron checkpoint staging can use ~146GB per rank on eight ranks.
        memory="1800GB",
        disk="2TB",
        priority="batch",
        max_retries=0,
        timeout_seconds=timeout_seconds,
    )
    if completion == "metrics":
        return skyrl_metrics_step(spec, execution)
    if completion == "checkpoint":
        return skyrl_checkpoint_step(spec, execution)
    if completion == "model":
        return skyrl_step(spec, execution)
    raise ValueError(f"Unknown completion mode: {completion}")


@click.command(help=__doc__)
@click.option("--version", required=True)
@click.option("--runner", type=click.Choice([r.value for r in Runner]), default="async", show_default=True)
@click.option("--inference-replicas", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--scale", type=click.Choice([s.value for s in Scale]), default="gate", show_default=True)
@click.option("--timeout-seconds", type=click.IntRange(min=1), default=3600, show_default=True)
@click.option("--completion", type=click.Choice(["metrics", "checkpoint", "model"]), default="model", show_default=True)
@click.option("--response-tokens", type=click.IntRange(min=1), help="Training output cap; defaults to the scale preset.")
@click.option("--eval-response-tokens", type=click.IntRange(min=1), help="Evaluation output cap; defaults to training.")
@click.option(
    "--context-tokens", type=click.IntRange(min=1), help="Engine context window; defaults to the scale preset."
)
@click.option(
    "--initial-eval-repeat-count",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Sequential startup evaluation passes; use cadence-gate or qualification to enable evaluation.",
)
@click.option(
    "--weight-change-probe/--no-weight-change-probe",
    default=False,
    show_default=True,
    help="Sample actual wire weights during publication; adds diagnostic overhead.",
)
@click.option("--weight-sync-interval", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--max-staleness-steps", type=click.IntRange(min=0), default=1, show_default=True)
@click.option(
    "--correction", type=click.Choice([c.value for c in Correction]), default="behavior_clip", show_default=True
)
@click.option(
    "--epoch-seeded-shuffle/--no-epoch-seeded-shuffle",
    default=False,
    show_default=True,
    help="Use a shared seed+epoch prompt permutation; off preserves each runner's historical ordering.",
)
@click.option("--study-steps", type=click.IntRange(min=1), help="Qualification-only update count; defaults to 25.")
@click.option("--eval-interval", type=click.IntRange(min=1), help="Evaluation cadence; must divide the update count.")
@click.option("--seed", type=click.IntRange(min=0, max=2**32 - 1), default=SEED, show_default=True)
@click.option("--validation-offset", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--validation-rows", type=click.IntRange(min=1), default=VALIDATION_ROWS, show_default=True)
@click.option("--run/--dry-run", "execute", default=False, show_default=True)
def main(
    version: str,
    runner: str,
    inference_replicas: int,
    scale: str,
    timeout_seconds: int,
    completion: str,
    response_tokens: int | None,
    eval_response_tokens: int | None,
    context_tokens: int | None,
    weight_sync_interval: int,
    max_staleness_steps: int,
    correction: str,
    initial_eval_repeat_count: int,
    weight_change_probe: bool,
    epoch_seeded_shuffle: bool,
    study_steps: int | None,
    eval_interval: int | None,
    seed: int,
    validation_offset: int,
    validation_rows: int,
    execute: bool,
) -> None:
    training = build_experiment(
        version=version,
        runner=Runner(runner),
        inference_replicas=inference_replicas,
        scale=Scale(scale),
        timeout_seconds=timeout_seconds,
        completion=completion,
        response_tokens=response_tokens,
        eval_response_tokens=eval_response_tokens,
        context_tokens=context_tokens,
        weight_sync_interval=weight_sync_interval,
        max_staleness_steps=max_staleness_steps,
        correction=Correction(correction),
        initial_eval_repeat_count=initial_eval_repeat_count,
        weight_change_probe=weight_change_probe,
        epoch_seeded_shuffle=epoch_seeded_shuffle,
        study_steps=study_steps,
        eval_interval=eval_interval,
        seed=seed,
        validation_offset=validation_offset,
        validation_rows=validation_rows,
    )
    prefix = marin_prefix()
    if execute:
        validate_regional_storage(prefix, CLUSTER)
    if completion == "model":
        checkpoint = cast(ArtifactStep[SkyRLCheckpoint], training.deps[0])
        training_context = StepContext.for_fingerprint(checkpoint.runtime_args.keys(), checkpoint.deps)
        export_context = StepContext.for_fingerprint(training.runtime_args.keys(), training.deps)
        preview = {
            "training": asdict(checkpoint.build_config(training_context)),
            "export": asdict(training.build_config(export_context)),
        }
    else:
        context = StepContext.for_fingerprint(training.runtime_args.keys(), training.deps)
        preview = asdict(training.build_config(context))
    click.echo(json.dumps(preview, indent=2))
    if execute:
        run(training, max_concurrent=2)


if __name__ == "__main__":
    main()
