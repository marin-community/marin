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
``--completion metrics --stage rl`` retains training metrics and internal evaluation
without checkpointing or export. Screening runs 25 updates over the 1,024-row
fixture (16 updates per epoch), with initial and final evaluation only. Use
``--screening-steps 100`` for a longer confirmation with the same fixture. Use
``--no-kl-loss`` explicitly for the new screening controls; historical defaults use KL.
Source publication/pinning and current capacity checks precede submission.
"""

from __future__ import annotations

import json
import re
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
    SkyRLLaunchRequest,
    SkyRLModel,
    SkyRLRetentionPolicy,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    SkyRLTrainingResult,
    skyrl_metrics_step,
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
from experiments.post_training.math_eval.launcher import pool_inputs

# Full revisions resolved from the curriculum experiment's c1899de/e53f048 pins.
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DATA_REVISION = "e53f048856ff4f594e959d75785d2c2d37b678ee"
SEED = 17
TRAIN_ROWS = 1024
VALIDATION_ROWS = 128
GSM8K_TEST_ROWS = 1319
POLICY_GPUS = 8
ROLE_PLAN = replace(SMOKE.role_plan, policy_mini_batch_size=64)
H100_CLUSTERS = ("cw-rno2a", "cw-us-east-02a")


class Runner(StrEnum):
    SYNC = "sync"
    ASYNC = "async"


class Scale(StrEnum):
    SMOKE = "smoke"
    QUALIFICATION = "qualification"
    COMPARISON = "comparison"
    SCREENING = "screening"


class Correction(StrEnum):
    BEHAVIOR_CLIP = "behavior_clip"
    REGULAR_TIS = "regular_tis"
    REGULAR_NO_TIS = "regular_no_tis"


class OptimizerPrecision(StrEnum):
    NATIVE = "native"
    AWARE_FP32 = "aware_fp32"
    BF16_FIRST = "bf16_first"
    BF16_BOTH = "bf16_both"
    FP32_REMAINDERS = "fp32_remainders"


@dataclass(frozen=True)
class Schedule:
    max_steps: int
    checkpoint_interval: int
    eval_interval: int
    eval_before_train: bool
    train_rows: int


SCHEDULES = {
    Scale.SMOKE: Schedule(4, 2, 2, True, TRAIN_ROWS),
    Scale.QUALIFICATION: Schedule(100, 25, 10, True, TRAIN_ROWS),
    # Keep both controls inside one epoch, with room for async lookahead. Measure
    # publication 0 -> 20; the final evaluation/checkpoint follows that interval.
    Scale.COMPARISON: Schedule(20, 20, 20, False, 1536),
    Scale.SCREENING: Schedule(25, 25, 25, True, TRAIN_ROWS),
}


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


@dataclass(frozen=True)
class Gsm8kWindowConfig:
    subset: Gsm8kSubsetConfig
    validation_offset: int


def validate_validation_window(offset: int, rows: int) -> None:
    """Accept the historical development set or a disjoint pinned test window."""
    if type(offset) is not int or type(rows) is not int or rows <= 0 or offset < 0:
        raise ValueError("Validation offset must be nonnegative and row count positive integers")
    if (offset, rows) == (0, VALIDATION_ROWS):
        return
    if offset < VALIDATION_ROWS or offset + rows > GSM8K_TEST_ROWS:
        raise ValueError("Locked validation must exclude test[0:128] and fit within test[128:1319]")


def validate_eval_interval(interval: int | None, steps: int, *, enabled: bool) -> None:
    if interval is not None and (type(interval) is not int or interval <= 0 or not enabled or steps % interval != 0):
        raise ValueError(
            "Explicit eval_interval must be positive, divide the update count, and use an evaluation schedule"
        )


def write_gsm8k_window(config: Gsm8kWindowConfig) -> None:
    validate_validation_window(config.validation_offset, config.subset.validation_rows)
    write_gsm8k_subset(config.subset, validation_offset=config.validation_offset)


def write_gsm8k_subset(config: Gsm8kSubsetConfig, *, validation_offset: int = 0) -> None:
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
        offset = validation_offset if split == "test" else 0
        selected = _gsm8k_records(split, offset + count, revision=config.dataset_revision)[offset:]
        records = _drop_over_length_records(
            selected,
            tokenizer_revision=config.tokenizer_revision,
        )
        if offset and len(records) != count:
            raise ValueError("Locked validation window cannot be truncated by prompt filtering")
        if (split == "train" or not offset) and len(records) < ROLE_PLAN.train_batch_size:
            raise ValueError(f"GSM8K {split} has fewer than one batch after prompt filtering")
        write_parquet_file(records, prefix_join(config.output_path, filename))
        row_ids[split] = [f"{split}/{cast(dict, record['extra_info'])['index']}" for record in records]
    manifest["rows"] = row_ids
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
    runner: Runner,
    scale: Scale,
    *,
    spans: bool,
    staleness: int,
    weight_sync_interval: int = 1,
    inference_replicas: int = 8,
    kl_loss: bool = True,
    correction: Correction = Correction.BEHAVIOR_CLIP,
    response_tokens: int | None = None,
    eval_response_tokens: int | None = None,
    context_tokens: int | None = None,
    screening_steps: int | None = None,
    minibatches: int = 1,
    updates: int | None = None,
    eval_updates: int | None = None,
    eval_interval: int | None = None,
    initial_eval_repeat_count: int = 1,
    weight_change_probe: bool = False,
    epoch_seeded_shuffle: bool = False,
    dataloader_workers: int | None = None,
    optimizer_precision: OptimizerPrecision = OptimizerPrecision.NATIVE,
    optimizer_state_metrics: bool = False,
) -> str:
    """Keep optimizer and inference settings identical across scheduler controls."""
    schedule = SCHEDULES[scale]
    if not isinstance(epoch_seeded_shuffle, bool):
        raise ValueError("epoch_seeded_shuffle must be a boolean")
    if screening_steps is not None:
        if scale is not Scale.SCREENING or screening_steps <= 0:
            raise ValueError("screening_steps must be positive and is only supported by the screening scale")
        schedule = replace(
            schedule, max_steps=screening_steps, checkpoint_interval=screening_steps, eval_interval=screening_steps
        )
    if type(minibatches) is not int or not 1 <= minibatches <= 16:
        raise ValueError("minibatches must be an integer in [1,16]")
    if minibatches != 1 and (runner is not Runner.SYNC or updates is None):
        raise ValueError("Multiple minibatches require the synchronous runner and explicit total updates")
    if updates is not None:
        if type(updates) is not int or updates <= 0 or updates % minibatches:
            raise ValueError("Total updates must be positive and divisible by minibatches")
        if screening_steps is not None or eval_interval is not None:
            raise ValueError("Use updates/eval_updates without batch-count screening_steps/eval_interval")
        cadence = updates if eval_updates is None else eval_updates
        if type(cadence) is not int or cadence <= 0 or updates % cadence or cadence % minibatches:
            raise ValueError("eval_updates must divide total updates and be divisible by minibatches")
        schedule = replace(
            schedule,
            max_steps=updates // minibatches,
            checkpoint_interval=updates // minibatches,
            eval_interval=cadence // minibatches,
        )
    elif eval_updates is not None:
        raise ValueError("eval_updates requires explicit total updates")
    validate_eval_interval(eval_interval, schedule.max_steps, enabled=schedule.eval_interval > 0)
    if eval_interval is not None:
        schedule = replace(schedule, eval_interval=eval_interval)
    if weight_sync_interval < 1 or staleness < 0 or weight_sync_interval > staleness + 1:
        raise ValueError("Weight sync interval must be positive and at most max_staleness_steps + 1")
    if runner is Runner.SYNC and weight_sync_interval != 1:
        raise ValueError("The synchronous runner publishes every update; weight_sync_interval must be 1")
    if inference_replicas not in (8, 16):
        raise ValueError("Qwen inference replicas must be 8 or 16, using one or two complete H100 nodes")
    if correction not in Correction:
        raise ValueError(f"Unknown correction mode: {correction}")
    response_tokens = SMOKE.max_new_tokens if response_tokens is None else response_tokens
    context_tokens = SMOKE.request_window_tokens if context_tokens is None else context_tokens
    eval_tokens = response_tokens if eval_response_tokens is None else eval_response_tokens
    if min(response_tokens, eval_tokens) <= 0:
        raise ValueError("Training and evaluation response budgets must be positive")
    if context_tokens < MAX_PROMPT_TOKENS + max(response_tokens, eval_tokens):
        raise ValueError("Context budget must fit the validated prompt limit plus either response budget")
    preset = replace(
        SMOKE,
        role_plan=replace(
            ROLE_PLAN,
            num_inference_engines=inference_replicas,
            train_batch_size=ROLE_PLAN.train_batch_size * minibatches,
        ),
        num_nodes=1 + inference_replicas // POLICY_GPUS,
        request_window_tokens=context_tokens,
        max_new_tokens=response_tokens,
        micro_forward_batch_size_per_gpu=ROLE_PLAN.micro_train_batch_size_per_gpu,
        max_steps=schedule.max_steps,
        ckpt_interval=schedule.checkpoint_interval,
    )
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
        logger="wandb",
        tracker_commit_each_step=True,
        project_name="marin-async-non-agentic-rl",
        resume_mode=None,
        eval_before_train=schedule.eval_before_train,
        eval_interval=schedule.eval_interval,
        eval_batch_size=VALIDATION_ROWS,
    )
    trainer["algorithm"].update(use_kl_loss=kl_loss, policy_loss_type="behavior_clip", use_tis=False)
    if not kl_loss:
        trainer["algorithm"]["use_kl_in_reward"] = False
    if correction == Correction.REGULAR_NO_TIS:
        trainer["algorithm"].update(policy_loss_type="regular", require_rollout_logprobs=True)
    if correction == Correction.REGULAR_TIS:
        trainer["algorithm"].update(
            policy_loss_type="regular", use_tis=True, tis_imp_ratio_cap=2.0, require_rollout_logprobs=True
        )
    trainer["fully_async"] = {
        "max_staleness_steps": staleness,
        "num_parallel_generation_workers": 64,
        "admission_stall_timeout": 300,
    }
    # Omit the default to preserve historical configuration fingerprints.
    if weight_sync_interval != 1:
        trainer["fully_async"]["weight_sync_interval"] = weight_sync_interval
    megatron = {
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
    }
    trainer["policy"].pop("fsdp_config")
    trainer["policy"]["megatron_config"] = dict(megatron)
    trainer["ref"] = {"megatron_config": dict(megatron)}
    apply_optimizer_precision(trainer, scale=scale, precision=optimizer_precision, state_metrics=optimizer_state_metrics)
    config["generator"]["sampling_params"]["logprobs"] = 0
    if eval_response_tokens is not None:
        config["generator"]["eval_sampling_params"] = {"max_generate_length": eval_response_tokens}
    config["generator"]["trajectory_retention"] = {
        "sample_count_per_step": 2,
        "always_retain_failures": False,
        "always_retain_non_terminating": False,
        "always_retain_loops": False,
        "max_bytes_per_step": 262144,
        "max_bytes_per_run": 4194304,
    }
    apply_observation_options(
        config, initial_eval_repeat_count=initial_eval_repeat_count, weight_change_probe=weight_change_probe
    )
    if dataloader_workers is not None:
        if type(dataloader_workers) is not int or dataloader_workers < 0:
            raise ValueError("dataloader_workers must be a nonnegative integer")
        config.setdefault("data", {})["num_workers"] = dataloader_workers
    if epoch_seeded_shuffle:
        config.setdefault("data", {})["epoch_seeded_shuffle"] = True
    return yaml.safe_dump(config, sort_keys=False)


def apply_optimizer_precision(
    trainer: dict, *, scale: Scale, precision: OptimizerPrecision, state_metrics: bool
) -> None:
    """Keep native identities stable and isolate the declared optimizer storage change."""
    if precision not in OptimizerPrecision or type(state_metrics) is not bool:
        raise ValueError("Declare a supported optimizer precision and boolean state-metrics setting")
    if precision == OptimizerPrecision.NATIVE and not state_metrics:
        return
    if scale != Scale.SCREENING:
        raise ValueError("Optimizer precision experiments require the Qwen screening scale")
    if not trainer["policy_train_spans"]:
        raise ValueError("Optimizer state metrics require policy training spans for matching memory peaks")
    trainer["optimizer_state_metrics"] = True
    if precision == OptimizerPrecision.NATIVE:
        return
    config = trainer["policy"]["megatron_config"]
    config["ddp_config"] = {"grad_reduce_in_fp32": True}
    config["optimizer_config_kwargs"] = {
        "use_precision_aware_optimizer": True,
        "optimizer_cuda_graph": False,
        "store_param_remainders": precision == OptimizerPrecision.FP32_REMAINDERS,
        "optimizer_cpu_offload": False,
        "main_params_dtype": "float32",
        "main_grads_dtype": "float32",
        "exp_avg_dtype": (
            "bfloat16" if precision in (OptimizerPrecision.BF16_FIRST, OptimizerPrecision.BF16_BOTH) else "float32"
        ),
        "exp_avg_sq_dtype": "bfloat16" if precision == OptimizerPrecision.BF16_BOTH else "float32",
    }


def apply_observation_options(config: dict, *, initial_eval_repeat_count: int, weight_change_probe: bool) -> None:
    """Validate opt-in diagnostics and omit defaults to preserve recipe identities."""
    if (
        isinstance(initial_eval_repeat_count, bool)
        or not isinstance(initial_eval_repeat_count, int)
        or initial_eval_repeat_count < 1
    ):
        raise ValueError("initial_eval_repeat_count must be a positive integer")
    if not isinstance(weight_change_probe, bool):
        raise ValueError("weight_change_probe must be a boolean")
    trainer, generator = config["trainer"], config["generator"]
    if initial_eval_repeat_count > 1:
        if (
            not trainer["eval_before_train"]
            or trainer["eval_interval"] <= 0
            or not trainer.get("dump_eval_results", True)
        ):
            raise ValueError(
                "Initial evaluation repeats require a schedule with eval_before_train=true, "
                "eval_interval>0, and evaluation dumps"
            )
        trainer["initial_eval_repeat_count"] = initial_eval_repeat_count
    if weight_change_probe:
        if (
            trainer["strategy"] != "megatron"
            or trainer["placement"]["colocate_all"]
            or generator.get("fuse_weights", False)
            or not generator["run_engines_locally"]
        ):
            raise ValueError(
                "Weight change probe requires Megatron with noncolocated, unfused, locally managed inference engines"
            )
        trainer["weight_change_probe"] = True


def validate_regional_storage(prefix: str, cluster: str, *, allow_cross_region_io: bool = False) -> None:
    """Require local storage, with the explicit Qwen screening RNO/east exception."""
    region = cluster.removeprefix("cw-")
    expected = load_cluster_config("coreweave").region_buckets.get(region)
    path = StoragePath(prefix)
    if allow_cross_region_io and cluster == "cw-rno2a" and path.scheme == "s3" and path.bucket == "marin-us-east-02a":
        return
    if (
        expected is None
        or expected.store is not StoreType.COREWEAVE
        or path.scheme != "s3"
        or path.bucket != expected.name
    ):
        raise click.ClickException(
            f"Artifact prefix {prefix!r} is not local to {cluster}; " "use its configured CoreWeave regional bucket"
        )


def validate_qwen_cross_region_request(request: SkyRLLaunchRequest) -> None:
    """Check all resolved artifact roots before the RNO exception can submit."""
    model = request.model
    expected_name = user_owned_name("models/async-rl-qwen3-0.6b")
    if (
        model.tokenizer_uri != "Qwen/Qwen3-0.6B"
        or model.tokenizer_revision != "c1899de289a04d12100db370d81485cdf75e47ca"
        or not model.identity.startswith(expected_name + "@")
        or not model.identity.endswith(":8a30d2b5")
    ):
        raise ValueError("Cross-region I/O requires the pinned Qwen3-0.6B model artifact")
    version = model.identity.removeprefix(expected_name + "@").removesuffix(":8a30d2b5")
    validate_version(version)
    if model.uri != f"s3://marin-us-east-02a/marin/{expected_name}/{version}/hf":
        raise ValueError("Cross-region I/O requires the pinned Qwen3-0.6B model path")
    trainer_config = yaml.safe_load(request.config_yaml).get("trainer", {})
    hub_repo = trainer_config.get("hf_hub_repo_id")
    save_interval = trainer_config.get("hf_save_interval", -1)
    for override in request.overrides:
        key, separator, value = override.lstrip("+").partition("=")
        if separator and key == "trainer.hf_hub_repo_id":
            hub_repo = yaml.safe_load(value)
        if separator and key == "trainer.hf_save_interval":
            save_interval = yaml.safe_load(value)
    if request.completion_mode != "metrics" or hub_repo is not None or save_interval is None or save_interval > 0:
        raise ValueError("Cross-region I/O requires metrics completion with HF export disabled")
    paths = [model.uri, *(item.uri for item in request.train_data), *(item.uri for item in request.validation_data)]
    paths.extend(asdict(request.output).values())
    for path in paths:
        storage = StoragePath(path)
        if storage.scheme != "s3" or storage.bucket != "marin-us-east-02a":
            raise ValueError(f"Cross-region I/O requires east S3 artifact paths: {path}")
    # Config and overrides can add OOD inputs or dump destinations outside the
    # typed locator fields. Check every explicit remote URI in those surfaces.
    for uri in re.findall(
        r"(?:s3|gs|https?|hf)://[^\s'\"\],}]+", request.config_yaml + "\n" + "\n".join(request.overrides)
    ):
        storage = StoragePath(uri)
        if storage.scheme != "s3" or storage.bucket != "marin-us-east-02a":
            raise ValueError(f"Cross-region I/O requires east S3 config paths: {uri}")


def build_experiment(
    *,
    version: str,
    cluster: str,
    runner: Runner,
    scale: Scale,
    spans: bool = True,
    staleness: int = 1,
    timeout_seconds: int = 1800,
    completion: str = "model",
    weight_sync_interval: int = 1,
    inference_replicas: int = 8,
    seed: int = SEED,
    kl_loss: bool = True,
    correction: Correction = Correction.BEHAVIOR_CLIP,
    response_tokens: int | None = None,
    eval_response_tokens: int | None = None,
    context_tokens: int | None = None,
    screening_steps: int | None = None,
    minibatches: int = 1,
    updates: int | None = None,
    eval_updates: int | None = None,
    eval_interval: int | None = None,
    validation_offset: int = 0,
    validation_rows: int = VALIDATION_ROWS,
    initial_eval_repeat_count: int = 1,
    weight_change_probe: bool = False,
    epoch_seeded_shuffle: bool = False,
    dataloader_workers: int | None = None,
    optimizer_precision: OptimizerPrecision = OptimizerPrecision.NATIVE,
    optimizer_state_metrics: bool = False,
    pool_artifact: str | None = None,
    allow_cross_region_io: bool = False,
) -> tuple[ArtifactStep[SkyRLModel] | ArtifactStep[SkyRLTrainingResult], ArtifactStep[EvaluationResult] | None]:
    """Construct versioned dependencies and a bounded, namespaced training attempt."""
    validate_version(version)
    if is_mutable_version(version):
        raise ValueError("Use an immutable artifact version for the matched experiment")
    if cluster not in H100_CLUSTERS:
        raise ValueError("The development preset requires an H100 cluster")
    if timeout_seconds <= 0 or staleness < 0:
        raise ValueError("A positive training deadline and nonnegative staleness are required")
    if completion not in ("model", "metrics"):
        raise ValueError(f"Unknown completion mode: {completion}")
    if allow_cross_region_io and (cluster != "cw-rno2a" or scale is not Scale.SCREENING or completion != "metrics"):
        raise ValueError("Cross-region I/O is restricted to Qwen screening metrics jobs on cw-rno2a")
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be between 0 and 2**32 - 1")
    validate_validation_window(validation_offset, validation_rows)
    config = training_config(
        runner,
        scale,
        spans=spans,
        staleness=staleness,
        weight_sync_interval=weight_sync_interval,
        inference_replicas=inference_replicas,
        kl_loss=kl_loss,
        correction=correction,
        response_tokens=response_tokens,
        eval_response_tokens=eval_response_tokens,
        context_tokens=context_tokens,
        screening_steps=screening_steps,
        minibatches=minibatches,
        updates=updates,
        eval_updates=eval_updates,
        eval_interval=eval_interval,
        initial_eval_repeat_count=initial_eval_repeat_count,
        weight_change_probe=weight_change_probe,
        epoch_seeded_shuffle=epoch_seeded_shuffle,
        dataloader_workers=dataloader_workers,
        optimizer_precision=optimizer_precision,
        optimizer_state_metrics=optimizer_state_metrics,
    )
    cpu = ResourceConfig.with_cpu(cpu=4, ram="16g", disk="32g")
    topology = SkyRLTopology(
        1 + inference_replicas // POLICY_GPUS,
        POLICY_GPUS,
        "H100",
        replace(
            ROLE_PLAN,
            num_inference_engines=inference_replicas,
            train_batch_size=ROLE_PLAN.train_batch_size * minibatches,
        ),
    )
    model = ArtifactStep(
        name=user_owned_name("models/async-rl-qwen3-0.6b"),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=remote(mirror_hf_model, resources=cpu),
        build_config=lambda ctx: HfSnapshotConfig(output_path=ctx.output_path, revision=MODEL_REVISION),
    )
    if pool_artifact is None:
        data = ArtifactStep(
            name=user_owned_name(
                "documents/async-rl-gsm8k" + (f"-test{validation_offset}-{validation_rows}" if validation_offset else "")
            ),
            version=version,
            artifact_type=Artifact,
            run=remote(write_gsm8k_window if validation_offset else write_gsm8k_subset, resources=cpu),
            build_config=lambda ctx: (
                Gsm8kWindowConfig(
                    Gsm8kSubsetConfig(
                        ctx.output_path, train_rows=SCHEDULES[scale].train_rows, validation_rows=validation_rows
                    ),
                    validation_offset,
                )
                if validation_offset
                else Gsm8kSubsetConfig(ctx.output_path, train_rows=SCHEDULES[scale].train_rows)
            ),
        )
        train_source = ArtifactDataSource(data, relative_path=TRAIN_FILENAME)
        validation_source = ArtifactDataSource(data, relative_path=VALIDATION_FILENAME)
    else:
        if validation_offset != 0 or validation_rows != VALIDATION_ROWS:
            raise ValueError("A frozen pool selects its own development rows")
        pool_config = yaml.safe_load(config)
        budget = pool_config["context_budget"]
        generation = max(budget["max_new_tokens_per_turn"], eval_response_tokens or 0)
        if budget["request_window_tokens"] - generation < 1024:
            raise ValueError("The frozen pool requires a 1024-token prompt budget")
        train_source, validation_source = pool_inputs(pool_artifact)
        data = train_source.step
    identity = fingerprint_hash(
        canonical_json(
            {
                "config": config,
                "model": model.fingerprint(),
                "data": data.fingerprint(),
                "seed": seed,
                "topology": topology,
            }
        )
    )
    name = user_owned_name(f"checkpoints/async-rl/{runner.value}-{scale.value}-{identity}")
    spec = SkyRLSpec(
        name=name,
        version=version,
        config_yaml=config,
        runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
        model=ArtifactHfModel(model, QWEN3_MODEL, MODEL_REVISION, relative_path="hf"),
        train_data=(train_source,),
        validation_data=(validation_source,),
        topology=topology,
        retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
        seed=seed,
        overrides=("++trainer.hf_hub_repo_id=null", "++generator.chat_template_kwargs.enable_thinking=false"),
    )
    execution = IrisSkyRLExecution(
        cluster=cluster,
        cluster_config=f"lib/iris/config/{cluster}.yaml",
        cpu=16,
        memory="128GB",
        disk="2TB",
        priority="batch",
        max_retries=0,
        timeout_seconds=timeout_seconds,
    )
    if completion == "metrics":
        step = skyrl_metrics_step(spec, execution)
        if allow_cross_region_io:
            original_build_config = step.build_config

            def checked_build_config(ctx):
                config = original_build_config(ctx)
                if not ctx.is_fingerprint:
                    validate_qwen_cross_region_request(config.request)
                return config

            step = replace(step, build_config=checked_build_config)
        return step, None
    training = skyrl_step(spec, execution)
    evaluation = eval_step(
        SkyRLEvaluationModel(
            step=training,
            model=ModelConfig(
                name=name.replace("/", "-"),
                location=SKYRL_POLICY_LOCATION,
                tokenizer=QWEN3_MODEL,
                apply_chat_template=True,
                resource_hint=ResourceHint(gpu={"H100": 1}),
                serve=ServeConfig(
                    tensor_parallel_size=1,
                    max_model_len=4096,
                    max_num_seqs=32,
                    vllm_extra_args=("--default-chat-template-kwargs", '{"enable_thinking":false}'),
                ),
                generation=GenerationConfig(max_gen_toks=512),
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
@click.option(
    "--allow-cross-region-io",
    is_flag=True,
    help="Allow Qwen screening on RNO to read pool/model artifacts and write results in the east bucket.",
)
@click.option("--stage", type=click.Choice(["rl", "evaluation"]), default="evaluation", show_default=True)
@click.option("--completion", type=click.Choice(["metrics", "model"]), default="model", show_default=True)
@click.option("--spans/--no-spans", default=True, show_default=True)
@click.option("--staleness", "--max-staleness-steps", type=click.IntRange(min=0), default=1, show_default=True)
@click.option(
    "--initial-eval-repeat-count",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Sequential startup evaluation passes; the selected schedule must enable evaluation.",
)
@click.option(
    "--weight-change-probe/--no-weight-change-probe",
    default=False,
    show_default=True,
    help="Sample actual wire weights during publication; adds diagnostic overhead.",
)
@click.option("--weight-sync-interval", type=click.IntRange(min=1), default=1, show_default=True)
@click.option(
    "--epoch-seeded-shuffle/--no-epoch-seeded-shuffle",
    default=False,
    show_default=True,
    help="Use a shared seed+epoch prompt permutation; off preserves each runner's historical ordering.",
)
@click.option(
    "--dataloader-workers", type=click.IntRange(min=0), help="Override loader workers; zero avoids spawn stalls."
)
@click.option("--inference-replicas", type=click.Choice(["8", "16"]), default="8", show_default=True)
@click.option("--seed", type=click.IntRange(min=0, max=2**32 - 1), default=SEED, show_default=True)
@click.option("--kl-loss/--no-kl-loss", default=True, show_default=True)
@click.option(
    "--optimizer-precision",
    type=click.Choice([precision.value for precision in OptimizerPrecision]),
    default="native",
    show_default=True,
    help="Qwen screening optimizer-state preset; nonnative settings require runtime qualification.",
)
@click.option(
    "--optimizer-state-metrics/--no-optimizer-state-metrics",
    default=False,
    show_default=True,
    help="Inventory actual optimizer storage after its first update; automatic for nonnative presets.",
)
@click.option(
    "--correction", type=click.Choice([c.value for c in Correction]), default="behavior_clip", show_default=True
)
@click.option("--response-tokens", type=click.IntRange(min=1), help="Training output cap; defaults to 1024.")
@click.option(
    "--eval-response-tokens", type=click.IntRange(min=1), help="Internal evaluation cap; defaults to training."
)
@click.option("--context-tokens", type=click.IntRange(min=1), help="Engine context window; defaults to 2048.")
@click.option("--minibatches", type=click.IntRange(min=1, max=16), default=1, show_default=True)
@click.option(
    "--updates", type=click.IntRange(min=1), help="Total optimizer updates; explicit with multiple minibatches."
)
@click.option("--eval-updates", type=click.IntRange(min=1), help="Evaluation cadence in optimizer updates.")
@click.option("--screening-steps", type=click.IntRange(min=1), help="Screening-only update count; defaults to 25.")
@click.option("--eval-interval", type=click.IntRange(min=1), help="Evaluation cadence; must divide the update count.")
@click.option("--validation-offset", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--validation-rows", type=click.IntRange(min=1), default=VALIDATION_ROWS, show_default=True)
@click.option("--timeout-seconds", type=click.IntRange(min=1), default=1800, show_default=True)
@click.option("--run/--dry-run", "execute", default=False, show_default=True)
@click.option("--pool-artifact", default=None, help="Use a frozen audited math pool by name@version.")
def main(
    version: str,
    runner: str,
    scale: str,
    cluster: str,
    allow_cross_region_io: bool,
    stage: str,
    completion: str,
    spans: bool,
    staleness: int,
    weight_sync_interval: int,
    inference_replicas: str,
    seed: int,
    kl_loss: bool,
    optimizer_precision: str,
    optimizer_state_metrics: bool,
    correction: str,
    response_tokens: int | None,
    eval_response_tokens: int | None,
    context_tokens: int | None,
    screening_steps: int | None,
    minibatches: int,
    updates: int | None,
    eval_updates: int | None,
    eval_interval: int | None,
    validation_offset: int,
    validation_rows: int,
    initial_eval_repeat_count: int,
    weight_change_probe: bool,
    epoch_seeded_shuffle: bool,
    dataloader_workers: int | None,
    timeout_seconds: int,
    execute: bool,
    pool_artifact: str | None,
) -> None:
    if allow_cross_region_io and (
        cluster != "cw-rno2a" or scale != Scale.SCREENING.value or stage != "rl" or completion != "metrics"
    ):
        raise click.UsageError("Cross-region I/O is restricted to Qwen screening RL jobs on cw-rno2a")
    if completion == "metrics" and stage == "evaluation":
        raise click.UsageError("Metrics completion requires --stage rl; external evaluation requires a model export")
    training, evaluation = build_experiment(
        pool_artifact=pool_artifact,
        allow_cross_region_io=allow_cross_region_io,
        version=version,
        cluster=cluster,
        runner=Runner(runner),
        scale=Scale(scale),
        spans=spans,
        staleness=staleness,
        timeout_seconds=timeout_seconds,
        completion=completion,
        weight_sync_interval=weight_sync_interval,
        inference_replicas=int(inference_replicas),
        seed=seed,
        kl_loss=kl_loss,
        optimizer_precision=OptimizerPrecision(optimizer_precision),
        optimizer_state_metrics=optimizer_state_metrics,
        correction=Correction(correction),
        response_tokens=response_tokens,
        eval_response_tokens=eval_response_tokens,
        context_tokens=context_tokens,
        screening_steps=screening_steps,
        minibatches=minibatches,
        updates=updates,
        eval_updates=eval_updates,
        eval_interval=eval_interval,
        validation_offset=validation_offset,
        validation_rows=validation_rows,
        initial_eval_repeat_count=initial_eval_repeat_count,
        weight_change_probe=weight_change_probe,
        epoch_seeded_shuffle=epoch_seeded_shuffle,
        dataloader_workers=dataloader_workers,
    )
    prefix = marin_prefix()
    if allow_cross_region_io:
        validate_regional_storage(prefix, cluster, allow_cross_region_io=True)
    elif execute:
        validate_regional_storage(prefix, cluster)
    if completion == "model":
        checkpoint = training.deps[0]
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
        target = evaluation if stage == "evaluation" else training
        assert target is not None
        run(target, max_concurrent=2)


if __name__ == "__main__":
    main()
