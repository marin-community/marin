# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball experiment presets, evaluators and input identities."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Protocol

import click
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from marin.execution.artifact import RECORD_FILENAME, Artifact, ArtifactRecord, is_mutable_version
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import (
    MISSING_CONFIG_VALUE,
    ArtifactDataSource,
    MegatronGeometry,
    SkyRLRolePlan,
    SkyRLRun,
    declared_config_value,
    role_plan_config_values,
    set_config_value,
)
from marin.training.training import LevanterCheckpoint
from rigging.config_discovery import find_project_root
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.provenance import Provenance
from taskcompendium.models import TaskSpec, grades_with_judge

from experiments.post_training.task_curation.driver import CANONICAL_ARTIFACT_ROOT
from experiments.post_training.task_curation.inventory import (
    FULL_VALIDATION_FILENAME,
    SELECTION_PREFIX,
    TASK_MANIFEST_FILENAME,
    TASK_SPEC_COLUMN,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
)

REPO_ROOT = find_project_root(Path(__file__))
PRESET_REVISION = "2026.10.27"
SEED = 20261020
MODEL_ID = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
MODEL_REVISION = "fe666478454e79fe6027a0bab0388d9abd68f51a"
MODEL_URI = "s3://marin-us-east-02a/models/open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
SELECTION_ROOT = str(StoragePath(CANONICAL_ARTIFACT_ROOT) / SELECTION_PREFIX)
DEFAULT_SELECTION_VERSION = "2026.10.26"
JUDGE_KEY = "TOGETHER_API_KEY"


class Focus(StrEnum):
    SPEED = "speed"
    LEARNING = "learning"
    DATA = "data"
    TRAIN = "train"


class Duration(StrEnum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class WindowStatistic(StrEnum):
    MEAN = "mean"
    MEDIAN = "median"


class Gap(StrEnum):
    REMOTE_JUDGE = "remote-judge"
    LAYOUT_P40_ONLY = "layout-p40-only"
    NO_ASYNC_SPEED_GATE = "no-async-speed-gate"
    AAII_COVERAGE = "aaii-coverage"
    REUSED_DEV_PANEL = "reused-dev-panel"
    NO_AGENTIC_TASKS = "no-agentic-tasks"
    NO_IMAGE_SANDBOX = "no-image-sandbox"
    NO_GRADING_TIMER = "no-grading-timer"
    MISMATCH_IS_AGREEMENT = "mismatch-is-agreement"
    EVAL_USES_TRAINING_SAMPLING = "eval-uses-training-sampling"


class EvalHarness(StrEnum):
    PANEL = "panel"


@dataclass(frozen=True)
class Evaluation:
    name: str
    harness: EvalHarness
    source: str
    file: str
    sample: int | None = None
    answers: int = 1
    every: int | None = None
    at_start: bool = False
    local_only: bool = False
    post_run: bool = False


@dataclass(frozen=True)
class Layout:
    name: str
    role_plan: SkyRLRolePlan
    megatron: MegatronGeometry
    cluster: str
    host_memory: str
    sockets: bool


@dataclass(frozen=True)
class Recipe:
    layout: Layout
    request_window_tokens: int
    max_new_tokens: int
    fixed_reply_tokens: int | None
    staleness: int
    estimator: Mapping[str, object]
    optimizer: Mapping[str, object]
    max_turns: int = 1
    engine_init_kwargs: Mapping[str, object] = field(default_factory=dict)
    policy_megatron_overrides: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Metric:
    key: str
    higher_is_better: bool
    window: WindowStatistic = WindowStatistic.MEAN


@dataclass(frozen=True)
class Preset:
    focus: Focus
    duration: Duration
    recipe: Recipe
    updates: int
    measured: range
    evaluations: tuple[Evaluation, ...] = ()
    checkpoint_every: int | None = None
    guards: tuple[Metric, ...] = ()
    reported: tuple[Metric, ...] = ()
    gaps: tuple[Gap, ...] = ()
    budget: str = "hours to days"

    @property
    def name(self) -> str:
        return f"{self.focus}-{self.duration}"


@dataclass(frozen=True)
class Setting:
    key: str
    value: object
    allows_new_key: bool


def parse_setting(text: str) -> Setting:
    key, separator, value = text.partition("=")
    if not separator or not key:
        raise click.BadParameter(f"expected dotted.key=value, got {text!r}")
    return Setting(key.lstrip("+"), yaml.safe_load(value), key.startswith("+"))


def setting_leaves(setting: Setting) -> tuple[Setting, ...]:
    if isinstance(setting.value, Mapping):
        return tuple(
            leaf
            for key, value in setting.value.items()
            for leaf in setting_leaves(Setting(f"{setting.key}.{key}", value, setting.allows_new_key))
        )
    return (setting,)


def task_manifest(root: StoragePath) -> dict[str, dict]:
    path = root / TASK_MANIFEST_FILENAME
    fs, key = filesystem_for(str(path))
    if not fs.exists(key):
        return {}
    with fs.open(key, "rb") as stream:
        return {row["task_id"]: row for row in pq.read_table(stream).to_pylist()}


def read_text(path: str | StoragePath) -> str:
    fs, key = filesystem_for(str(path))
    with fs.open(key, "rt") as stream:
        return stream.read()


def paths(pattern: StoragePath) -> list[StoragePath]:
    fs, key = filesystem_for(str(pattern))
    return [StoragePath(fs.unstrip_protocol(path)) for path in sorted(fs.glob(key))]


def exists(path: str | StoragePath) -> bool:
    fs, key = filesystem_for(str(path))
    return fs.exists(key)


class PanelEvaluator:
    required_env: tuple[str, ...] = ()

    def callbacks(self, evaluation: Evaluation, *, updates: int) -> list[dict]:
        return [
            {
                "type": "evaluation",
                "eval_steps": evaluation.every or updates + 1,
                "eval_before_train": evaluation.at_start,
                "eval_on_train_end": not evaluation.every or updates % evaluation.every != 0,
            }
        ]

    def configure(
        self,
        evaluation: Evaluation,
        config: dict,
        *,
        selection: ArtifactStep,
        default_selection: ArtifactStep,
        seed: int,
    ) -> tuple[ArtifactDataSource, ...]:
        source = default_selection if evaluation.source == "default" else selection
        if evaluation.sample is not None:
            source = sampled_tasks_step(
                source, evaluation.file, evaluation.sample, seed, local_only=evaluation.local_only
            )
        config["trainer"].update(eval_num_prompts=None, dump_eval_results=True)
        config["generator"].update(
            eval_n_samples_per_prompt=evaluation.answers,
            eval_sampling_params={
                "temperature": 0.0 if evaluation.answers == 1 else 1.0,
                "top_p": 1.0,
                "logprobs": None if config["trainer"]["algorithm"]["off_policy_correction"] == "tis" else 0,
                "ignore_eos": config["generator"]["sampling_params"]["ignore_eos"],
            },
        )
        return (ArtifactDataSource(source, relative_path=f"{evaluation.file}.parquet"),)


class Evaluator(Protocol):
    required_env: tuple[str, ...]

    def callbacks(self, evaluation: Evaluation, *, updates: int) -> list[dict]: ...
    def configure(
        self,
        evaluation: Evaluation,
        config: dict,
        *,
        selection: ArtifactStep,
        default_selection: ArtifactStep,
        seed: int,
    ) -> tuple[ArtifactDataSource, ...]: ...


EVALUATORS: Mapping[EvalHarness, Evaluator] = MappingProxyType({EvalHarness.PANEL: PanelEvaluator()})
EVALUATIONS = MappingProxyType(
    {
        evaluation.name: evaluation
        for evaluation in (
            Evaluation(
                name="panel-100",
                harness=EvalHarness.PANEL,
                source="default",
                file=Path(VALIDATION_FILENAME).stem,
            ),
            Evaluation(
                name="panel-910",
                harness=EvalHarness.PANEL,
                source="default",
                file=Path(FULL_VALIDATION_FILENAME).stem,
            ),
            Evaluation(
                name="speed-512x4",
                harness=EvalHarness.PANEL,
                source="default",
                file=Path(TRAIN_FILENAME).stem,
                sample=512,
                answers=4,
                local_only=True,
            ),
            Evaluation(
                name="screen-256x4",
                harness=EvalHarness.PANEL,
                source="tasks",
                file=Path(TRAIN_FILENAME).stem,
                sample=256,
                answers=4,
            ),
        )
    }
)
_P40_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=4,
    policy_num_gpus_per_node=8,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=8,
    inference_engine_expert_parallel_size=8,
    train_batch_size=128,
    policy_mini_batch_size=128,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=4,
)
P40 = Layout(
    name="p40",
    role_plan=_P40_PLAN,
    megatron=MegatronGeometry(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
    ),
    cluster="cw-us-east-02a",
    host_memory="1800GB",
    sockets=True,
)
E8 = replace(P40, name="e8", role_plan=replace(_P40_PLAN, generate_only=True))
L112 = Layout(
    name="l112",
    role_plan=SkyRLRolePlan(
        colocate_all=False,
        policy_num_nodes=8,
        policy_num_gpus_per_node=8,
        num_inference_engines=12,
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=4,
        inference_engine_expert_parallel_size=4,
        train_batch_size=512,
        policy_mini_batch_size=512,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=16,
    ),
    megatron=MegatronGeometry(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=16,
        expert_tensor_parallel_size=1,
    ),
    cluster="cw-rno2a",
    host_memory="1800GB",
    sockets=False,
)
LAYOUTS = MappingProxyType({layout.name: layout for layout in (P40, E8, L112)})
_ESTIMATOR = MappingProxyType(
    {
        "advantage_estimator": "rloo_n",
        "policy_loss_type": "regular",
        "loss_reduction": "token_mean",
        "use_kl_loss": False,
        "use_kl_in_reward": False,
        "kl_estimator_type": "k3_unbiased_gradient",
        "off_policy_correction": "none",
        "eps_clip_low": 0.2,
        "eps_clip_high": 0.2,
        "group_advantage_min_size": 2,
        "advantage_batch_normalize": False,
        "rloo_n_filter_zero_reward_groups": True,
        "use_entropy_loss": False,
    }
)
_OPTIMIZER = MappingProxyType(
    {
        "optimizer": "AdamW",
        "lr": 1e-6,
        "adam_betas": [0.9, 0.999],
        "weight_decay": 0.0,
        "max_grad_norm": 0.5,
        "scheduler": "constant_with_warmup",
        "num_warmup_steps": 0,
    }
)
_LEARNING = Recipe(
    layout=P40,
    request_window_tokens=24576,
    max_new_tokens=8192,
    fixed_reply_tokens=None,
    staleness=0,
    estimator=_ESTIMATOR,
    optimizer=_OPTIMIZER,
    engine_init_kwargs={"moe_backend": "triton"},
)
_GAPS = (
    Gap.AAII_COVERAGE,
    Gap.REUSED_DEV_PANEL,
    Gap.NO_AGENTIC_TASKS,
    Gap.NO_IMAGE_SANDBOX,
    Gap.REMOTE_JUDGE,
    Gap.NO_GRADING_TIMER,
)
_PANEL = EVALUATIONS["panel-100"]
_MISMATCH_REPORTS = tuple(
    Metric(f"policy/mismatch/staleness0/{name}", False)
    for name in (
        "log_ratio_abs_mean",
        "log_ratio_abs_p99",
        "log_ratio_abs_p999",
        "frac_outside_0_5_2",
        "pos_first256/log_ratio_abs_mean",
        "pos_last256/log_ratio_abs_mean",
        "pos_middle/log_ratio_abs_mean",
    )
)
_TRAINING_REPORTS = (
    *_MISMATCH_REPORTS,
    *(
        Metric(key, higher)
        for key, higher in (
            ("loss/avg_final_rewards", True),
            ("policy/policy_entropy", True),
            ("policy/raw_grad_norm", False),
            ("policy/ppo_clip_ratio", False),
            ("policy/ppo_clip_ratio_low", False),
            ("policy/ppo_clip_ratio_high", False),
            ("reward/zero_std_group_fraction", False),
            ("generate/avg_num_tokens", False),
            ("generate/max_num_tokens", False),
            ("generate/failed_trajectory_fraction", False),
        )
    ),
)
_SPEED_REPORTS = (
    *_MISMATCH_REPORTS,
    *(
        Metric(f"timing/step_wall/{phase}", False)
        for phase in (
            "group_admission",
            "batch_assembly",
            "training_preparation",
            "advantages",
            "policy_training",
            "group_bookkeeping",
            "weight_sync",
            "step_end_bookkeeping",
            "checkpoint_work",
            "evaluation",
            "unaccounted",
        )
    ),
    Metric("vllm/median_generation_throughput", True),
    Metric("vllm/latency_ttft_mean", False),
    Metric("startup/init_weight_sync_state", False),
    Metric("startup/load_checkpoints", False),
)
_LEARNING_MEDIUM = Preset(
    focus=Focus.LEARNING,
    duration=Duration.MEDIUM,
    recipe=_LEARNING,
    updates=25,
    measured=range(21, 26),
    evaluations=(EVALUATIONS["panel-910"],),
    guards=(Metric("panel-910/family_mean", True),),
    reported=(*_TRAINING_REPORTS, Metric("panel-910/mean", True)),
    gaps=(*_GAPS, Gap.EVAL_USES_TRAINING_SAMPLING),
    budget="105-120 min; 70-80 H100-h",
)
PRESETS = MappingProxyType(
    {
        preset.name: preset
        for preset in (
            Preset(
                focus=Focus.SPEED,
                duration=Duration.SHORT,
                recipe=replace(_LEARNING, layout=E8, fixed_reply_tokens=4096),
                updates=0,
                measured=range(1),
                evaluations=(EVALUATIONS["speed-512x4"],),
                guards=(Metric("generate/tokens_per_second_per_gpu", True),),
                reported=(
                    Metric("generate/wall_seconds", False),
                    Metric("generate/response_tokens", True),
                    Metric("generate/vllm/latency_ttft_mean", False),
                    Metric("generate/vllm/peak_gpu_cache_usage_perc", False),
                ),
                gaps=_GAPS,
                budget="9-12 min; 1.5 H100-h",
            ),
            Preset(
                focus=Focus.DATA,
                duration=Duration.SHORT,
                recipe=replace(_LEARNING, layout=E8),
                updates=0,
                measured=range(1),
                evaluations=(EVALUATIONS["screen-256x4"],),
                guards=(Metric("screen-256x4/mixed_fraction", True),),
                reported=(
                    Metric("screen-256x4/mean", True),
                    Metric("screen-256x4/all_right_fraction", False),
                    Metric("screen-256x4/all_wrong_fraction", False),
                    Metric("screen-256x4/scored_fraction", True),
                    Metric("generate/prompts_dropped_too_long", False),
                ),
                gaps=_GAPS,
                budget="10-15 min; 2 H100-h",
            ),
            Preset(
                focus=Focus.SPEED,
                duration=Duration.MEDIUM,
                recipe=replace(
                    _LEARNING,
                    fixed_reply_tokens=4096,
                    estimator={**_ESTIMATOR, "rloo_n_filter_zero_reward_groups": False},
                ),
                updates=10,
                measured=range(2, 11),
                guards=(
                    Metric("speed/update_seconds", False, WindowStatistic.MEDIAN),
                    Metric("policy/mismatch/staleness0/kl_k3", False),
                ),
                reported=_SPEED_REPORTS,
                gaps=(*_GAPS, Gap.MISMATCH_IS_AGREEMENT, Gap.LAYOUT_P40_ONLY, Gap.NO_ASYNC_SPEED_GATE),
                budget="35-40 min; 25 H100-h",
            ),
            _LEARNING_MEDIUM,
            replace(_LEARNING_MEDIUM, focus=Focus.DATA),
            Preset(
                focus=Focus.TRAIN,
                duration=Duration.LONG,
                recipe=replace(
                    _LEARNING,
                    layout=L112,
                    request_window_tokens=65536,
                    max_new_tokens=32768,
                    staleness=2,
                    estimator={**_ESTIMATOR, "group_advantage_min_size": 4, "off_policy_correction": "tis"},
                    optimizer={**_OPTIMIZER, "lr": 4e-06},
                    max_turns=999999,
                    engine_init_kwargs={
                        "moe_backend": "triton",
                        "enable_return_routed_experts": True,
                        "enable_auto_tools": True,
                        "tool_parser": "hermes",
                        "logprobs_mode": "processed_logprobs",
                        "generation_config": "vllm",
                        "validate_rollout_logprob_sampling": True,
                    },
                    policy_megatron_overrides={
                        "logprob_chunk_size": 256,
                        "ddp_config": {
                            "overlap_grad_reduce": False,
                            "overlap_param_gather": False,
                            "grad_reduce_in_fp32": False,
                        },
                        "transformer_config_kwargs": {
                            "recompute_granularity": "full",
                            "recompute_method": "uniform",
                            "recompute_modules": ["core_attn"],
                            "recompute_num_layers": 1,
                        },
                        "optimizer_checkpoint_sharding_type": "dp_reshardable",
                    },
                ),
                updates=100,
                measured=range(1, 101),
                evaluations=(replace(_PANEL, every=10, at_start=True), replace(EVALUATIONS["panel-910"], post_run=True)),
                checkpoint_every=10,
                gaps=(*_GAPS, Gap.EVAL_USES_TRAINING_SAMPLING),
            ),
        )
    }
)
TRAIN_LONG_RECIPES = MappingProxyType(
    {
        "p40": replace(_LEARNING, staleness=2, estimator={**_ESTIMATOR, "off_policy_correction": "tis"}),
        "l112": PRESETS["train-long"].recipe,
    }
)


def run_name(preset: Preset, provenance: Provenance, inputs_hash: str, replicate: int, *, control: bool) -> str:
    if control:
        return f"checkpoints/snowball-rl/control/{preset.name}-{provenance.base_commit}-r{replicate}"
    name = f"checkpoints/snowball-rl/{preset.name}-{provenance.tree_hash}-in-{inputs_hash}"
    if preset.focus != Focus.TRAIN:
        name += f"-r{replicate}"
    return user_owned_name(name)


def run_tree_hash(name: str) -> str:
    return name.partition("train-long-")[2].partition("-in-")[0]


def selection_summary(selection: ArtifactStep) -> dict:
    return json.loads(read_text(StoragePath(selection.path()) / "summary.json"))


def evaluation_answer_count(evaluation: Evaluation, selection: ArtifactStep, default: ArtifactStep) -> int:
    source = default if evaluation.source == "default" else selection
    count = (
        evaluation.sample
        or selection_summary(source)["training_rows" if evaluation.file == "train" else f"{evaluation.file}_rows"]
    )
    return count * evaluation.answers


def _write_sample(selection: str, filename: str, count: int, seed: int, local_only: bool, output: str) -> None:
    generator = random.Random(seed)
    rows = []
    seen = 0
    fs, key = filesystem_for(prefix_join(selection, f"{filename}.parquet"))
    with fs.open(key, "rb") as stream:
        source = pq.ParquetFile(stream)
        schema = source.schema_arrow
        for batch in source.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                if local_only and grades_with_judge(TaskSpec.model_validate_json(row[TASK_SPEC_COLUMN]).grader):
                    continue
                seen += 1
                if len(rows) < count:
                    rows.append(row)
                else:
                    index = generator.randrange(seen)
                    if index < count:
                        rows[index] = row
    if len(rows) != count:
        raise ValueError(f"selection has only {len(rows)} eligible tasks; {count} are required")
    fs, key = filesystem_for(prefix_join(output, f"{filename}.parquet"))
    fs.makedirs(str(PurePosixPath(key).parent), exist_ok=True)
    with fs.open(key, "wb") as stream:
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), stream)
    for name in (TASK_MANIFEST_FILENAME, "summary.json"):
        source_fs, source_key = filesystem_for(prefix_join(selection, name))
        if source_fs.exists(source_key):
            fs, key = filesystem_for(prefix_join(output, name))
            with source_fs.open(source_key, "rb") as source, fs.open(key, "wb") as target:
                target.write(source.read())


def sampled_tasks_step(
    selection: ArtifactStep, filename: str, count: int, seed: int, *, local_only: bool
) -> ArtifactStep:
    identity = fingerprint_hash(json.dumps((selection.name, selection.version, filename, count, seed, local_only)))
    return apply(
        f"datasets/snowball-rl/sample-{identity}",
        _write_sample,
        version=PRESET_REVISION,
        selection=selection,
        filename=filename,
        count=count,
        seed=seed,
        local_only=local_only,
        output=OUT,
    )


BATCH_SETTINGS = {
    "trainer.train_batch_size": "train_batch_size",
    "trainer.policy_mini_batch_size": "policy_mini_batch_size",
    "generator.n_samples_per_prompt": "n_samples_per_prompt",
    "trainer.micro_train_batch_size_per_gpu": "micro_train_batch_size_per_gpu",
}
_COMPARISON_KEYS = {
    "trainer.max_steps",
    "trainer.seed",
    "trainer.logger",
    "trainer.project_name",
}
_DERIVED_KEYS = {
    "generator.max_input_length",
    "trainer.max_prompt_length",
    "generator.max_turns",
    "generator.engine_init_kwargs.max_model_len",
    "generator.sampling_params.max_generate_length",
    "generator.trajectory_reward_shaping.overlong.l_max",
    "generator.trajectory_reward_shaping.overlong.l_cache",
}
_SETTING_OWNERS = (
    (("entrypoint",), "choose the preset's role plan"),
    (("trainer.resume_mode",), "use --resume-from for checkpoint continuity"),
    (("extra_env.NCCL_",), "choose network transport in Layout"),
    (
        ("trainer.placement", "generator.inference_engine_", "generator.num_inference_engines"),
        "choose role geometry with --layout or Layout.role_plan",
    ),
    (("generator.trajectory_retention", "trainer.max_ckpts_to_keep"), "set checkpoint retention in the preset"),
    (("generator.run_engines_locally",), "choose the preset's Layout"),
    (
        ("trainer.callbacks", "trainer.eval_", "generator.eval_"),
        "choose evaluations with --eval on train-long or the preset's evaluations",
    ),
    (("trainer.ckpt_interval", "trainer.hf_save_interval"), "set checkpoint frequency in the preset"),
)


def apply_setting(config: dict, setting: Setting, preset: Preset) -> None:
    key = setting.key
    fix = next((message for prefixes, message in _SETTING_OWNERS if key.startswith(prefixes)), None)
    if fix:
        raise click.BadParameter(f"{key} is owned by the preset: {fix}")
    if key in role_plan_config_values(preset.recipe.layout.role_plan) and key not in BATCH_SETTINGS:
        fix = "choose role geometry with --layout or Layout.role_plan"
    elif key in _DERIVED_KEYS:
        fix = "set the recipe's context budget instead of a derived key"
    elif key.startswith("data.") and not (
        preset.focus in (Focus.LEARNING, Focus.TRAIN) and (key == "data.shuffle" or key.startswith("data.sampling."))
    ):
        fix = "select curated data with --tasks on a data or train-long preset"
    elif preset.focus != Focus.TRAIN and (
        key in _COMPARISON_KEYS
        or key.startswith("context_budget.")
        or any(part in key.split(".") for part in ("max_generate_length", "ignore_eos", "min_tokens"))
    ):
        fix = "change the preset and recalibrate its control to keep runs comparable"
    elif preset.focus == Focus.SPEED and key in (
        "generator.sampling_params.logprobs",
        "trainer.training_metrics",
        "generator.apply_overlong_filtering",
    ):
        fix = "keep mismatch metrics and fixed-length training masks enabled in speed presets"
    elif key in BATCH_SETTINGS and preset.focus not in (Focus.LEARNING, Focus.TRAIN):
        fix = "change batch shape only in learning-medium or train-long"
    elif preset.recipe.layout.role_plan.generate_only and (
        not key.startswith("generator.") or key.startswith("generator.sampling_params.")
    ):
        fix = "use generator performance keys; generate-only sampling belongs to the evaluation"
    if fix:
        raise click.BadParameter(f"{key} is owned by the preset: {fix}")
    if (
        declared_config_value(config, key) is MISSING_CONFIG_VALUE
        and key not in BATCH_SETTINGS
        and not setting.allows_new_key
    ):
        raise click.BadParameter(f"unknown setting {key}; prefix with + to add it")
    try:
        set_config_value(config, key, setting.value)
    except ValueError as error:
        raise click.BadParameter(str(error)) from error


def default_selection() -> ArtifactStep[Artifact]:
    return ArtifactStep.adopt(
        f"{SELECTION_PREFIX}/snowball-100k",
        DEFAULT_SELECTION_VERSION,
        str(StoragePath(SELECTION_ROOT) / "snowball-100k" / DEFAULT_SELECTION_VERSION),
    )


def artifact_prefix(cluster: str) -> str:
    config = yaml.safe_load((REPO_ROOT / f"lib/iris/config/{cluster}.yaml").read_text())
    return config["defaults"]["task_env"]["MARIN_PREFIX"]


def artifact_record(path: str) -> ArtifactRecord:
    return ArtifactRecord.model_validate_json(read_text(StoragePath(path) / RECORD_FILENAME))


def selection_artifact(value: str) -> ArtifactStep:
    name, separator, version = value.rpartition("@")
    if not separator or is_mutable_version(version):
        raise click.BadParameter("use NAME@VERSION with an immutable task selection version")
    return ArtifactStep.adopt(f"{SELECTION_PREFIX}/{name}", version, str(StoragePath(SELECTION_ROOT) / name / version))


def published_artifact(value: str) -> ArtifactStep:
    name, separator, version = value.rpartition("@")
    if not separator:
        name, version = value, PRESET_REVISION
    if is_mutable_version(version):
        raise click.BadParameter("use an immutable artifact version instead of dev")
    for layout in (P40, L112):
        root = str(StoragePath(artifact_prefix(layout.cluster)) / name / version)
        if exists(StoragePath(root) / "resolved-launch.yaml"):
            return ArtifactStep.adopt(name, version, root, kind=SkyRLRun)
        path = StoragePath(root) / RECORD_FILENAME
        if exists(path):
            record = ArtifactRecord.model_validate_json(read_text(path))
            kind = SkyRLRun if record.result_type.endswith(".SkyRLRun") else LevanterCheckpoint
            return ArtifactStep.adopt(name, version, record.source or root, kind=kind, config=record.config)
    raise click.BadParameter(f"no published artifact record for {name}@{version}")
