# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched synchronous and asynchronous Megatron GRPO on Qwen3-0.6B/GSM8K.

Print the resolved launch envelope before submitting a coordinator in the selected
H100 cluster (its storage prefix and CPU artifact jobs must be in that region)::

    python -m experiments.post_training.async_rl --version 2026.09.05.1 \
        --cluster cw-us-east-02a --runner sync --scale smoke --dry-run

Use ``--run`` to build the graph. The default evaluation stage includes the model
mirror, deterministic GSM8K fixture, training, terminal HF export, and a 32-row
GSM8K evaluation. ``--stage rl`` stops after export. The training deadline includes
setup; export and evaluation are separate jobs with their own resource accounting.
Source publication/pinning and current capacity checks precede submission.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import cast

import click
import yaml
from fray.types import ResourceConfig
from marin.evaluation.model_config import GenerationConfig, ModelConfig, ResourceHint, ServeConfig
from marin.execution.artifact import Artifact, is_mutable_version, validate_version
from marin.execution.fingerprint import canonical_json, fingerprint_hash
from marin.execution.lazy import ArtifactStep, StepContext, run
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import (
    SKYRL_POLICY_LOCATION,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLEvaluationModel,
    SkyRLModel,
    SkyRLRetentionPolicy,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.cluster_config import StoreType, load_cluster_config, marin_prefix
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.writers import write_parquet_file

from experiments.evaluation.pipeline import EvaluationResult, eval_step
from experiments.post_training.curriculum_rl.launch import SMOKE, HfSnapshotConfig, mirror_hf_model, rl_config_yaml
from experiments.post_training.curriculum_rl.pool import (
    GSM8K_INSTRUCTION,
    MAX_PROMPT_TOKENS,
    QWEN3_MODEL,
    SYSTEM_PROMPT,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    _drop_over_length_records,
    _gsm8k_records,
)

# Full revisions resolved from the curriculum experiment's c1899de/e53f048 pins.
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DATA_REVISION = "e53f048856ff4f594e959d75785d2c2d37b678ee"
SEED = 17
TRAIN_ROWS = 1024
VALIDATION_ROWS = 128
POLICY_GPUS = 8
ROLE_PLAN = replace(SMOKE.role_plan, policy_mini_batch_size=64)
H100_CLUSTERS = ("cw-rno2a", "cw-us-east-02a")


class Runner(StrEnum):
    SYNC = "sync"
    ASYNC = "async"


class Scale(StrEnum):
    SMOKE = "smoke"
    QUALIFICATION = "qualification"


@dataclass(frozen=True)
class Gsm8kSubsetConfig:
    output_path: str
    dataset_revision: str = DATA_REVISION
    tokenizer_revision: str = MODEL_REVISION
    train_rows: int = TRAIN_ROWS
    validation_rows: int = VALIDATION_ROWS
    # These fixed curriculum contracts participate in the artifact fingerprint.
    max_prompt_tokens: int = field(init=False, default=MAX_PROMPT_TOKENS)
    system_prompt: str = field(init=False, default=SYSTEM_PROMPT)
    answer_instruction: str = field(init=False, default=GSM8K_INSTRUCTION)


def write_gsm8k_subset(config: Gsm8kSubsetConfig) -> None:
    """Write deterministic, disjoint train and validation data plus selected row IDs.

    Both Parquet splits retain the curriculum prompt and verifier contracts. The
    selection manifest records their exact source row IDs so a run can audit the
    chosen training and validation examples.
    """
    manifest: dict[str, object] = {"dataset": "openai/gsm8k", "revision": config.dataset_revision, "rows": {}}
    row_ids = {}
    for split, count, filename in (
        ("train", config.train_rows, TRAIN_FILENAME),
        ("test", config.validation_rows, VALIDATION_FILENAME),
    ):
        records = _drop_over_length_records(
            _gsm8k_records(split, count, revision=config.dataset_revision),
            tokenizer_revision=config.tokenizer_revision,
        )
        if len(records) < ROLE_PLAN.train_batch_size:
            raise ValueError(f"GSM8K {split} has fewer than one batch after prompt filtering")
        write_parquet_file(records, prefix_join(config.output_path, filename))
        row_ids[split] = [f"{split}/{cast(dict, record['extra_info'])['index']}" for record in records]
    manifest["rows"] = row_ids
    StoragePath(prefix_join(config.output_path, "selection.json")).write_text(json.dumps(manifest, sort_keys=True))


def training_config(runner: Runner, scale: Scale, *, spans: bool, staleness: int) -> str:
    """Keep optimizer and inference settings identical across scheduler controls."""
    preset = replace(SMOKE, role_plan=ROLE_PLAN, max_steps=4 if scale is Scale.SMOKE else 30)
    config = yaml.safe_load(rl_config_yaml(preset))
    config["entrypoint"] = "standard" if runner is Runner.SYNC else "fully_async"
    trainer = config["trainer"]
    trainer.update(
        strategy="megatron",
        flash_attn=False,
        policy_train_spans=spans,
        generate_spans=spans,
        async_spans=spans,
        training_metrics=True,
        project_name="marin-async-non-agentic-rl",
        resume_mode=None,
        eval_before_train=True,
        eval_interval=2 if scale is Scale.SMOKE else 10,
        eval_batch_size=VALIDATION_ROWS,
    )
    trainer["algorithm"].update(policy_loss_type="behavior_clip", use_tis=False)
    trainer["fully_async"] = {
        "max_staleness_steps": staleness,
        "num_parallel_generation_workers": 64,
        "admission_stall_timeout": 300,
    }
    megatron = {
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
    }
    trainer["policy"].pop("fsdp_config")
    trainer["policy"]["megatron_config"] = dict(megatron)
    trainer["ref"] = {"megatron_config": dict(megatron)}
    config["generator"]["sampling_params"]["logprobs"] = 0
    config["generator"]["trajectory_retention"] = {
        "sample_count_per_step": 2,
        "always_retain_failures": False,
        "always_retain_non_terminating": False,
        "always_retain_loops": False,
        "max_bytes_per_step": 262144,
        "max_bytes_per_run": 4194304,
    }
    return yaml.safe_dump(config, sort_keys=False)


def validate_regional_storage(prefix: str, cluster: str) -> None:
    """Require the artifact prefix to use the configured bucket for the target CoreWeave region."""
    region = cluster.removeprefix("cw-")
    expected = load_cluster_config("coreweave").region_buckets.get(region)
    path = StoragePath(prefix)
    if (
        expected is None
        or expected.store is not StoreType.COREWEAVE
        or path.scheme != "s3"
        or path.bucket != expected.name
    ):
        raise click.ClickException(
            f"Artifact prefix {prefix!r} is not local to {cluster}; " "use its configured CoreWeave regional bucket"
        )


def build_experiment(
    *,
    version: str,
    cluster: str,
    runner: Runner,
    scale: Scale,
    spans: bool = True,
    staleness: int = 1,
    timeout_seconds: int = 1800,
) -> tuple[ArtifactStep[SkyRLModel], ArtifactStep[EvaluationResult]]:
    """Construct versioned dependencies and a bounded, namespaced training attempt."""
    validate_version(version)
    if is_mutable_version(version):
        raise ValueError("Use an immutable artifact version for the matched experiment")
    if cluster not in H100_CLUSTERS:
        raise ValueError("The development preset requires an H100 cluster")
    if timeout_seconds <= 0 or staleness < 0:
        raise ValueError("A positive training deadline and nonnegative staleness are required")
    config = training_config(runner, scale, spans=spans, staleness=staleness)
    cpu = ResourceConfig.with_cpu(cpu=4, ram="16g", disk="32g")
    topology = SkyRLTopology(2, POLICY_GPUS, "H100", ROLE_PLAN)
    model = ArtifactStep(
        name=user_owned_name("models/async-rl-qwen3-0.6b"),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=remote(mirror_hf_model, resources=cpu),
        build_config=lambda ctx: HfSnapshotConfig(output_path=ctx.output_path, revision=MODEL_REVISION),
    )
    data = ArtifactStep(
        name=user_owned_name("documents/async-rl-gsm8k"),
        version=version,
        artifact_type=Artifact,
        run=remote(write_gsm8k_subset, resources=cpu),
        build_config=lambda ctx: Gsm8kSubsetConfig(output_path=ctx.output_path),
    )
    identity = fingerprint_hash(
        canonical_json(
            {
                "config": config,
                "model": model.fingerprint(),
                "data": data.fingerprint(),
                "seed": SEED,
                "topology": topology,
            }
        )
    )
    name = user_owned_name(f"checkpoints/async-rl/{runner.value}-{scale.value}-{identity}")
    training = skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=config,
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
            model=ArtifactHfModel(model, QWEN3_MODEL, MODEL_REVISION, relative_path="hf"),
            train_data=(ArtifactDataSource(data, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(data, relative_path=VALIDATION_FILENAME),),
            topology=topology,
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
            seed=SEED,
            overrides=("++trainer.hf_hub_repo_id=null", "++generator.chat_template_kwargs.enable_thinking=false"),
        ),
        IrisSkyRLExecution(
            cluster=cluster,
            cluster_config=f"lib/iris/config/{cluster}.yaml",
            cpu=16,
            memory="128GB",
            disk="2TB",
            priority="batch",
            max_retries=0,
            wandb_entity="marin-community",
            timeout_seconds=timeout_seconds,
        ),
    )
    evaluation = eval_step(
        SkyRLEvaluationModel(
            step=training,
            model=ModelConfig(
                name=name.replace("/", "-"),
                location=SKYRL_POLICY_LOCATION,
                tokenizer=QWEN3_MODEL,
                apply_chat_template=True,
                resource_hint=ResourceHint(gpu={"H100": 1}),
                serve=ServeConfig(tensor_parallel_size=1, max_model_len=4096, max_num_seqs=32),
                generation=GenerationConfig(max_gen_toks=2048),
            ),
        ),
        "gsm8k-smoke",
        limit=32,
        version=version,
        accelerator="H100x1",
        submission_cluster=cluster,
        federated_cluster=cluster,
    )
    return training, evaluation


@click.command(help=__doc__)
@click.option("--version", required=True)
@click.option("--runner", type=click.Choice([r.value for r in Runner]), default="async", show_default=True)
@click.option("--scale", type=click.Choice([s.value for s in Scale]), default="smoke", show_default=True)
@click.option("--cluster", type=click.Choice(H100_CLUSTERS), default="cw-us-east-02a", show_default=True)
@click.option("--stage", type=click.Choice(["rl", "evaluation"]), default="evaluation", show_default=True)
@click.option("--spans/--no-spans", default=True, show_default=True)
@click.option("--staleness", type=click.IntRange(min=0), default=1, show_default=True)
@click.option("--timeout-seconds", type=click.IntRange(min=1), default=1800, show_default=True)
@click.option("--run/--dry-run", "execute", default=False, show_default=True)
def main(
    version: str,
    runner: str,
    scale: str,
    cluster: str,
    stage: str,
    spans: bool,
    staleness: int,
    timeout_seconds: int,
    execute: bool,
) -> None:
    training, evaluation = build_experiment(
        version=version,
        cluster=cluster,
        runner=Runner(runner),
        scale=Scale(scale),
        spans=spans,
        staleness=staleness,
        timeout_seconds=timeout_seconds,
    )
    prefix = marin_prefix()
    if execute:
        validate_regional_storage(prefix, cluster)
    context = StepContext.for_run(training.path(prefix), prefix, runtime_args=training.runtime_args, deps=training.deps)
    click.echo(json.dumps(asdict(training.build_config(context)), indent=2))
    if execute:
        run(evaluation if stage == "evaluation" else training, max_concurrent=2)


if __name__ == "__main__":
    main()
