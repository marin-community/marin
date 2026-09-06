# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Async Megatron Snowball/GSM8K qualification using the regional SFT export.

Run the two-update gate before the 25-update qualification. Both keep the staged
export's thinking template and require four policy nodes and one inference node.
Check H100 and host-memory capacity before submitting in cw-us-east-02a.
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
    SkyRLModel,
    SkyRLRetentionPolicy,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath, prefix_join
from transformers import AutoTokenizer
from zephyr.writers import write_parquet_file

from experiments.post_training.async_rl import DATA_REVISION, SEED, validate_regional_storage
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


def write_snowball_gsm8k(config: SnowballDataConfig) -> None:
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
            records = _gsm8k_records(split, count, revision=config.dataset_revision)
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
    StoragePath(prefix_join(config.output_path, "selection.json")).write_text(json.dumps(manifest, sort_keys=True))


def training_config(scale: Scale) -> str:
    gate = scale is Scale.GATE
    steps = 2 if gate else 25
    preset = replace(
        SNOWBALL_SMOKE,
        role_plan=ROLE_PLAN,
        max_steps=steps,
        ckpt_interval=steps,
        eval_interval=-1 if gate else 25,
        request_window_tokens=2048 if gate else 4096,
        max_new_tokens=512 if gate else 2048,
        micro_forward_batch_size_per_gpu=1,
    )
    config = yaml.safe_load(rl_config_yaml(preset))
    config["entrypoint"] = "fully_async"
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
    trainer["fully_async"] = {
        "max_staleness_steps": 1,
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
    config["generator"].update(inference_engine_data_parallel_size=8, inference_engine_expert_parallel_size=8)
    config["generator"]["sampling_params"]["logprobs"] = 0
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
    return yaml.safe_dump(config, sort_keys=False)


def build_experiment(*, version: str, scale: Scale, timeout_seconds: int) -> ArtifactStep[SkyRLModel]:
    """Build the regional data audit, training, and terminal Megatron HF export."""
    validate_version(version)
    if is_mutable_version(version) or timeout_seconds <= 0:
        raise ValueError("Use an immutable version and positive training deadline")
    data = ArtifactStep(
        name=user_owned_name("documents/async-rl-snowball-gsm8k"),
        version=version,
        artifact_type=Artifact,
        deps=(SNOWBALL_MODEL,),
        run=remote(write_snowball_gsm8k, resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="32g")),
        build_config=lambda ctx: SnowballDataConfig(
            output_path=ctx.output_path, model_path=ctx.artifact_path(SNOWBALL_MODEL)
        ),
    )
    config = training_config(scale)
    topology = SkyRLTopology(5, 8, "H100", ROLE_PLAN)
    identity = fingerprint_hash(
        canonical_json(
            {
                "config": config,
                "model": SNOWBALL_MODEL.fingerprint(),
                "tokenizer_revision": TOKENIZER_REVISION,
                "data": data.fingerprint(),
                "topology": topology,
                "seed": SEED,
            }
        )
    )
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(f"checkpoints/async-rl/snowball-{scale.value}-{identity}"),
            version=version,
            config_yaml=config,
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
            model=ArtifactHfModel(SNOWBALL_MODEL, SNOWBALL_POLICY.tokenizer_uri, TOKENIZER_REVISION, relative_path=""),
            train_data=(ArtifactDataSource(data, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(data, relative_path=VALIDATION_FILENAME),),
            topology=topology,
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
            seed=SEED,
            overrides=("++trainer.hf_hub_repo_id=null",),
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=16,
            # Megatron checkpoint staging can use ~146GB per rank on eight ranks.
            memory="1800GB",
            disk="2TB",
            priority="batch",
            max_retries=0,
            timeout_seconds=timeout_seconds,
        ),
    )


@click.command(help=__doc__)
@click.option("--version", required=True)
@click.option("--scale", type=click.Choice([s.value for s in Scale]), default="gate", show_default=True)
@click.option("--timeout-seconds", type=click.IntRange(min=1), default=3600, show_default=True)
@click.option("--run/--dry-run", "execute", default=False, show_default=True)
def main(version: str, scale: str, timeout_seconds: int, execute: bool) -> None:
    training = build_experiment(version=version, scale=Scale(scale), timeout_seconds=timeout_seconds)
    prefix = marin_prefix()
    if execute:
        validate_regional_storage(prefix, CLUSTER)
    context = StepContext.for_run(training.path(prefix), prefix, runtime_args=training.runtime_args, deps=training.deps)
    click.echo(json.dumps(asdict(training.build_config(context)), indent=2))
    if execute:
        run(training, max_concurrent=2)


if __name__ == "__main__":
    main()
