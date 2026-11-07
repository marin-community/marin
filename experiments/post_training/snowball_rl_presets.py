# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball experiment presets, evaluators and input identities."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlencode

import click
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from marin.execution.artifact import RECORD_FILENAME, Artifact, ArtifactRecord, is_mutable_version
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    MISSING_CONFIG_VALUE,
    RESOLVED_LAUNCH_FILENAME,
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
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.provenance import Provenance
from rolloutengine.models import TASK_TIMING_KEYS

from experiments.post_training.task_curation.driver import CANONICAL_ARTIFACT_ROOT
from experiments.post_training.task_curation.inventory import (
    FULL_VALIDATION_FILENAME,
    SELECTION_PREFIX,
    SUMMARY_FILENAME,
    TASK_MANIFEST_FILENAME,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
)

REPO_ROOT = find_project_root(Path(__file__))
PRESET_REVISION = "2026.11.06.1"
RUN_PREFIX = "checkpoints/snowball-rl"
GENERATE_WALL_SECONDS_METRIC = "generate/wall_seconds"
TRAIN_LONG_RUN = f"{RUN_PREFIX}/train-long-"
# Fraction of a native panel that must be scored before its scores count.
MINIMUM_PANEL_COVERAGE = 0.99
SEED = 20261020
MODEL_ID = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
MODEL_REVISION = "fe666478454e79fe6027a0bab0388d9abd68f51a"
MODEL_URI = "s3://marin-us-east-02a/models/open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
SELECTION_ROOT = str(StoragePath(CANONICAL_ARTIFACT_ROOT) / SELECTION_PREFIX)
DEFAULT_SELECTION_VERSION = "2026.11.04.1"
DEFAULT_TEST_PANEL_NAME = "snowball-test-panel"
DEFAULT_TEST_PANEL_VERSION = "2026.11.06.1"
EVAL_REPLY_TOKENS = 8192
GRAFANA_URL = "https://grafana.oa.dev/d/marin-rl-runs/rl-post-training"
GRAFANA_SPEED_PANELS = (
    ("response_tokens", 3),
    ("rollout_wait", 102),
    ("step_composition", 121),
    ("vllm_throughput", 109),
)
GRAFANA_LEARNING_PANELS = (("reward", 22), ("evaluation", 23), ("loss_and_kl", 24))
GRAFANA_MISMATCH_PANEL = ("logprob_mismatch", 54)


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
    SELECTED_TOKEN_WEIGHTED_MEAN = "selected-token-weighted-mean"


class Gap(StrEnum):
    LAYOUT_P40_ONLY = "layout-p40-only"
    AAII_COVERAGE = "aaii-coverage"
    NO_AGENTIC_TASKS = "no-agentic-tasks"
    NO_IMAGE_SANDBOX = "no-image-sandbox"
    MISMATCH_IS_AGREEMENT = "mismatch-is-agreement"


@dataclass(frozen=True)
class Evaluation:
    name: str
    file: str
    source: str = "default"
    sample: int | None = None
    answers: int = 1
    every: int | None = None
    at_start: bool = False

    def tasks(self, selection: ArtifactStep, default: ArtifactStep) -> ArtifactStep:
        if self.source == "default":
            return default
        if self.source == "tasks":
            return selection
        return selection_artifact(self.source)


@dataclass(frozen=True)
class Layout:
    name: str
    role_plan: SkyRLRolePlan
    megatron: MegatronGeometry
    cluster: str
    host_memory: str


@dataclass(frozen=True)
class Recipe:
    layout: Layout
    request_window_tokens: int
    max_new_tokens: int
    fixed_reply_tokens: int | None
    staleness: int
    estimator: Mapping[str, object]
    optimizer: Mapping[str, object]


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
    objectives: tuple[Metric, ...] = ()
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


def read_text(path: str | StoragePath) -> str:
    return StoragePath(str(path)).read_text()


def exists(path: str | StoragePath) -> bool:
    return StoragePath(str(path)).exists()


class PanelEvaluator:
    def callbacks(self, evaluation: Evaluation, *, updates: int) -> list[dict]:
        return [
            {
                "type": "evaluation",
                "eval_steps": evaluation.every or updates + 1,
                "eval_before_train": evaluation.at_start,
                "eval_on_train_end": not evaluation.every or updates % evaluation.every != 0,
            }
        ]

    def configure(self, evaluation: Evaluation, config: dict, tasks: ArtifactStep) -> tuple[ArtifactDataSource, ...]:
        if evaluation.sample is not None:
            tasks = sampled_tasks_step(tasks, evaluation.file, evaluation.sample, config["trainer"]["seed"])
        config["trainer"].update(eval_num_prompts=None, dump_eval_results=True)
        sampling = (
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "logprobs": None,
                "ignore_eos": False,
                "max_generate_length": EVAL_REPLY_TOKENS,
            }
            if evaluation.answers == 1
            else {
                "temperature": 1.0,
                "top_p": 1.0,
                "logprobs": None,
                "ignore_eos": config["generator"]["sampling_params"]["ignore_eos"],
            }
        )
        config["generator"].update(eval_n_samples_per_prompt=evaluation.answers, eval_sampling_params=sampling)
        return (ArtifactDataSource(tasks, relative_path=f"{evaluation.file}.parquet"),)


_TEST_PANEL = f"{DEFAULT_TEST_PANEL_NAME}@{DEFAULT_TEST_PANEL_VERSION}"

EVALUATIONS = MappingProxyType(
    {
        evaluation.name: evaluation
        for evaluation in (
            Evaluation(name="test_panel", file=Path(FULL_VALIDATION_FILENAME).stem, source=_TEST_PANEL),
            Evaluation(name="test_panel_100", file=Path(VALIDATION_FILENAME).stem, source=_TEST_PANEL),
            Evaluation(name="speed-512x4", file=Path(TRAIN_FILENAME).stem, sample=512, answers=4),
            Evaluation(name="screen-256x4", file=Path(TRAIN_FILENAME).stem, source="tasks", sample=256, answers=4),
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
)
E8 = replace(P40, name="e8", role_plan=replace(_P40_PLAN, generate_only=True))
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
)
_ASYNC_P40 = replace(_LEARNING, staleness=2, estimator={**_ESTIMATOR, "off_policy_correction": "tis"})
_GAPS = (
    Gap.AAII_COVERAGE,
    Gap.NO_AGENTIC_TASKS,
    Gap.NO_IMAGE_SANDBOX,
)
_PANEL = EVALUATIONS["test_panel_100"]
_TIMER_REPORTS = tuple(
    Metric(f"{phase}/{statistic}", False)
    for phase in sorted(TASK_TIMING_KEYS)
    for statistic in ("mean", "p50", "p95", "max")
)
_MISMATCH_REPORTS = tuple(
    Metric(f"policy/mismatch/staleness0/{name}", False, WindowStatistic.SELECTED_TOKEN_WEIGHTED_MEAN)
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
    *_TIMER_REPORTS,
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
    *_TIMER_REPORTS,
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
_ASYNC_SPEED_REPORTS = (
    *_SPEED_REPORTS,
    *(Metric(f"async/staleness_{statistic}", False) for statistic in ("mean", "max", "ratio")),
    *(
        Metric(f"policy/mismatch/pooled/{statistic}", False)
        for statistic in ("kl_k3", "log_ratio_abs_mean", "log_ratio_abs_p99", "log_ratio_abs_p999")
    ),
    *(
        Metric(f"policy/mismatch/{bucket}/selected_tokens", bucket == "staleness0")
        for bucket in ("staleness0", "staleness1", "staleness2", "staleness3", "staleness4-7", "staleness8+")
    ),
)
_LEARNING_MEDIUM = Preset(
    focus=Focus.LEARNING,
    duration=Duration.MEDIUM,
    recipe=_LEARNING,
    updates=25,
    measured=range(21, 26),
    evaluations=(replace(EVALUATIONS["test_panel"], at_start=True),),
    guards=(Metric("test_panel/family_mean", True),),
    objectives=(Metric("test_panel/family_mean", True), Metric("test_panel/family_mean/gain_per_h100_hour", True)),
    reported=(*_TRAINING_REPORTS, Metric("test_panel/mean", True)),
    gaps=_GAPS,
    budget="105-120 min; 70-80 H100-h",
)
_LEARNING_LONG = replace(
    _LEARNING_MEDIUM,
    duration=Duration.LONG,
    objectives=(Metric("test_panel/family_mean", True), Metric("test_panel_100/family_mean/gain_per_h100_hour", True)),
    recipe=_ASYNC_P40,
    updates=100,
    measured=range(91, 101),
    evaluations=(replace(_PANEL, every=25, at_start=True), EVALUATIONS["test_panel"]),
    reported=(*_LEARNING_MEDIUM.reported, Metric("test_panel_100/mean", True)),
    budget="about 4 h; about 160 H100-h (estimate)",
)
PRESETS = MappingProxyType(
    {
        preset.name: preset
        for preset in (
            Preset(
                focus=Focus.SPEED,
                duration=Duration.SHORT,
                objectives=(Metric("generate/tokens_per_second_per_gpu", True),),
                recipe=replace(_LEARNING, layout=E8, fixed_reply_tokens=4096),
                updates=0,
                measured=range(1),
                evaluations=(EVALUATIONS["speed-512x4"], EVALUATIONS["test_panel"]),
                guards=(Metric("generate/tokens_per_second_per_gpu", True), Metric("test_panel/family_mean", True)),
                reported=(
                    *_TIMER_REPORTS,
                    Metric(GENERATE_WALL_SECONDS_METRIC, False),
                    Metric("generate/response_tokens", True),
                    Metric("generate/vllm/latency_ttft_mean", False),
                    Metric("generate/vllm/peak_gpu_cache_usage_perc", False),
                ),
                gaps=_GAPS,
                budget="14-17 min; 2 H100-h",
            ),
            Preset(
                focus=Focus.DATA,
                duration=Duration.SHORT,
                objectives=(Metric("screen-256x4/mixed_fraction", True),),
                recipe=replace(_LEARNING, layout=E8),
                updates=0,
                measured=range(1),
                evaluations=(EVALUATIONS["screen-256x4"],),
                guards=(Metric("screen-256x4/mixed_fraction", True),),
                reported=(
                    *_TIMER_REPORTS,
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
                objectives=(Metric("speed/update_seconds", False, WindowStatistic.MEDIAN),),
                recipe=replace(
                    _LEARNING,
                    fixed_reply_tokens=4096,
                    estimator={**_ESTIMATOR, "rloo_n_filter_zero_reward_groups": False},
                ),
                updates=10,
                measured=range(2, 11),
                guards=(
                    Metric("speed/update_seconds", False, WindowStatistic.MEDIAN),
                    Metric("policy/mismatch/staleness0/kl_k3", False, WindowStatistic.SELECTED_TOKEN_WEIGHTED_MEAN),
                ),
                reported=_SPEED_REPORTS,
                gaps=(*_GAPS, Gap.MISMATCH_IS_AGREEMENT, Gap.LAYOUT_P40_ONLY),
                budget="35-40 min; 25 H100-h",
            ),
            Preset(
                focus=Focus.SPEED,
                duration=Duration.LONG,
                objectives=(
                    Metric("speed/update_seconds", False, WindowStatistic.MEDIAN),
                    Metric("async/performance/loss_tokens_per_configured_policy_gpu_second", True),
                    Metric("async/performance/response_tokens_per_configured_inference_gpu_second", True),
                ),
                recipe=_ASYNC_P40,
                updates=20,
                measured=range(3, 21),
                guards=(
                    Metric("speed/update_seconds", False, WindowStatistic.MEDIAN),
                    Metric("async/performance/loss_tokens_per_configured_policy_gpu_second", True),
                    Metric("async/performance/response_tokens_per_configured_inference_gpu_second", True),
                    Metric("policy/mismatch/staleness0/kl_k3", False, WindowStatistic.SELECTED_TOKEN_WEIGHTED_MEAN),
                ),
                reported=_ASYNC_SPEED_REPORTS,
                gaps=(*_GAPS, Gap.LAYOUT_P40_ONLY),
                budget="45-75 min; 30-50 H100-h",
            ),
            _LEARNING_MEDIUM,
            replace(_LEARNING_MEDIUM, focus=Focus.DATA),
            _LEARNING_LONG,
            replace(_LEARNING_LONG, focus=Focus.DATA),
            Preset(
                focus=Focus.TRAIN,
                duration=Duration.LONG,
                recipe=replace(_LEARNING, staleness=2, estimator={**_ESTIMATOR, "off_policy_correction": "tis"}),
                updates=100,
                measured=range(1, 101),
                evaluations=(replace(_PANEL, every=10, at_start=True),),
                checkpoint_every=10,
                reported=_TIMER_REPORTS,
                gaps=_GAPS,
            ),
        )
    }
)


def run_name(preset: Preset, provenance: Provenance, inputs_hash: str, replicate: int, *, control: bool) -> str:
    if control:
        return control_name(preset.name, provenance.base_commit, replicate)
    name = f"{RUN_PREFIX}/{preset.name}-{provenance.tree_hash}-in-{inputs_hash}"
    if preset.focus != Focus.TRAIN:
        name += f"-r{replicate}"
    return user_owned_name(name)


def gated(preset: Preset) -> bool:
    return preset.name in PRESETS and preset.focus != Focus.TRAIN


def run_tree_hash(name: str) -> str:
    return name.partition(TRAIN_LONG_RUN)[2].partition("-in-")[0]


def selection_summary(selection: ArtifactStep) -> dict:
    return json.loads(read_text(StoragePath(selection.path()) / SUMMARY_FILENAME))


def evaluation_answer_count(evaluation: Evaluation, tasks: ArtifactStep) -> int:
    key = "training_rows" if evaluation.file == Path(TRAIN_FILENAME).stem else f"{evaluation.file}_rows"
    return (evaluation.sample or selection_summary(tasks)[key]) * evaluation.answers


def _write_sample(selection: str, filename: str, count: int, seed: int, output: str) -> None:
    generator = random.Random(seed)
    rows = []
    seen = 0
    fs, key = filesystem_for(prefix_join(selection, f"{filename}.parquet"))
    with fs.open(key, "rb") as stream:
        source = pq.ParquetFile(stream)
        schema = source.schema_arrow
        for batch in source.iter_batches(batch_size=256):
            for row in batch.to_pylist():
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
    for name in (TASK_MANIFEST_FILENAME, SUMMARY_FILENAME):
        source_fs, source_key = filesystem_for(prefix_join(selection, name))
        if source_fs.exists(source_key):
            fs, key = filesystem_for(prefix_join(output, name))
            with source_fs.open(source_key, "rb") as source, fs.open(key, "wb") as target:
                target.write(source.read())


def sampled_tasks_step(selection: ArtifactStep, filename: str, count: int, seed: int) -> ArtifactStep:
    identity = fingerprint_hash(json.dumps((selection.name, selection.version, filename, count, seed)))
    return apply(
        f"datasets/snowball-rl/sample-{identity}",
        _write_sample,
        version=PRESET_REVISION,
        selection=selection,
        filename=filename,
        count=count,
        seed=seed,
        output=OUT,
    )


BATCH_SETTINGS = {
    "trainer.train_batch_size": "train_batch_size",
    "trainer.policy_mini_batch_size": "policy_mini_batch_size",
    "generator.n_samples_per_prompt": "n_samples_per_prompt",
    "trainer.micro_train_batch_size_per_gpu": "micro_train_batch_size_per_gpu",
}
_COMPARISON_KEYS = {"trainer.max_steps", "trainer.seed", "trainer.logger", "trainer.project_name"}
_DERIVED_KEYS = {
    "generator.max_input_length",
    "trainer.max_prompt_length",
    "generator.max_turns",
    "generator.engine_init_kwargs.max_model_len",
    "generator.sampling_params.max_generate_length",
    "generator.trajectory_reward_shaping.overlong.l_max",
    "generator.trajectory_reward_shaping.overlong.l_cache",
}


def apply_setting(config: dict, setting: Setting, preset: Preset) -> None:
    key = setting.key
    owners = {
        "choose the preset's role plan": key.startswith("entrypoint"),
        "use --resume-from for checkpoint continuity": key.startswith("trainer.resume_mode"),
        "choose network transport in Layout": key.startswith("extra_env.NCCL_"),
        "choose role geometry with Layout.role_plan": (
            key.startswith(("trainer.placement", "generator.inference_engine_", "generator.num_inference_engines"))
            or (key in role_plan_config_values(preset.recipe.layout.role_plan) and key not in BATCH_SETTINGS)
        ),
        "set checkpoint retention in the preset": key.startswith(
            ("generator.trajectory_retention", "trainer.max_ckpts_to_keep")
        ),
        "choose the preset's Layout": key.startswith("generator.run_engines_locally"),
        "choose evaluations with --eval on train-long or the preset's evaluations": key.startswith(
            ("trainer.callbacks", "trainer.eval_", "generator.eval_")
        ),
        "set checkpoint frequency in the preset": key.startswith(("trainer.ckpt_interval", "trainer.hf_save_interval")),
        "set the recipe's context budget instead of a derived key": key in _DERIVED_KEYS,
        "select curated data with --tasks on a data or train-long preset": (
            key.startswith("data.")
            and (
                not (
                    preset.focus in (Focus.LEARNING, Focus.TRAIN)
                    and (key == "data.shuffle" or key.startswith("data.sampling."))
                )
            )
        ),
        "change the preset and recalibrate its control to keep runs comparable": (
            preset.focus != Focus.TRAIN
            and (
                key in _COMPARISON_KEYS
                or key.startswith("context_budget.")
                or any(part in key.split(".") for part in ("max_generate_length", "ignore_eos", "min_tokens"))
            )
        ),
        "keep mismatch metrics and fixed-length training masks enabled in speed presets": (
            preset.focus == Focus.SPEED
            and key
            in ("generator.sampling_params.logprobs", "trainer.training_metrics", "generator.apply_overlong_filtering")
        ),
        "change batch shape only in a learning or train-long preset": (
            key in BATCH_SETTINGS and preset.focus not in (Focus.LEARNING, Focus.TRAIN)
        ),
        "use generator performance keys; generate-only sampling belongs to the evaluation": (
            preset.recipe.layout.role_plan.generate_only
            and (not key.startswith("generator.") or key.startswith("generator.sampling_params."))
        ),
    }
    fix = next((message for message, owned in owners.items() if owned), None)
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
    return selection_artifact(f"snowball-100k@{DEFAULT_SELECTION_VERSION}")


def cluster_config_path(cluster: str) -> str:
    return f"lib/iris/config/{cluster}.yaml"


def grafana_links(
    preset: Preset,
    *,
    run_id: str,
    cluster: str,
    training_type: Literal["sync", "async"],
    started_ms: int,
    finished_ms: int | None,
) -> dict[str, str]:
    """Build dashboard and preset-panel links for a native run; an unfinished run links through now."""
    query = urlencode(
        {
            "orgId": 1,
            "timezone": "utc",
            "from": started_ms,
            "to": finished_ms if finished_ms is not None else "now",
            "var-training_type": training_type,
            "var-cluster": cluster,
            "var-run": run_id,
            "var-job": "$__all",
            "var-execution": "$__all",
        },
        safe="$",
    )
    url = f"{GRAFANA_URL}?{query}"
    panels = (
        GRAFANA_SPEED_PANELS
        if preset.focus == Focus.SPEED
        else GRAFANA_LEARNING_PANELS if preset.focus in (Focus.LEARNING, Focus.DATA) else ()
    )
    if any(metric.key.startswith("policy/mismatch/") for metric in preset.guards):
        panels = (*panels, GRAFANA_MISMATCH_PANEL)
    return {"run": url, **{name: f"{url}&viewPanel={panel}" for name, panel in panels}}


def artifact_prefix(cluster: str) -> str:
    config = yaml.safe_load((REPO_ROOT / cluster_config_path(cluster)).read_text())
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
    root = str(StoragePath(artifact_prefix(P40.cluster)) / name / version)
    if exists(StoragePath(root) / RESOLVED_LAUNCH_FILENAME):
        return ArtifactStep.adopt(name, version, root, kind=SkyRLRun)
    path = StoragePath(root) / RECORD_FILENAME
    if exists(path):
        record = artifact_record(root)
        kind = SkyRLRun if record.result_type.endswith(".SkyRLRun") else LevanterCheckpoint
        return ArtifactStep.adopt(name, version, record.source or root, kind=kind, config=record.config)
    raise click.BadParameter(f"no published artifact record for {name}@{version}")


def control_name(preset: str, commit: str, replicate: int) -> str:
    return f"{RUN_PREFIX}/control/{preset}-{commit}-r{replicate}"


def guards_evaluation(evaluation: Evaluation, metric: Metric) -> bool:
    return metric.key.startswith(evaluation.name + "/")


def run_path(cluster: str, name: str, version: str) -> str:
    return str(StoragePath(artifact_prefix(cluster)) / name / version)


def panel_evaluation(preset: Preset) -> Evaluation | None:
    return next(
        (
            evaluation
            for evaluation in preset.evaluations
            if any(guards_evaluation(evaluation, metric) for metric in preset.guards)
        ),
        None,
    )


def utc_millis(value: str) -> int:
    """Parse a native receipt's UTC timestamp for duration and dashboard windows."""
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


EXIT_CODES = {"pass": 0, "fail": 1, "invalid": 2, None: 3}
REFUSAL_EXIT_CODE = 4


class SnowballCommand(click.Command):
    def make_context(self, *args, **kwargs):
        try:
            return super().make_context(*args, **kwargs)
        except click.UsageError as error:
            error.exit_code = REFUSAL_EXIT_CODE
            raise

    def invoke(self, context):
        configure_coreweave_s3()
        try:
            return super().invoke(context)
        except click.UsageError as error:
            error.exit_code = REFUSAL_EXIT_CODE
            raise


def run_command(*, help_text: str | None = None):
    def decorate(function):
        options = (
            click.option(
                "--preset",
                "preset_name",
                required=True,
                type=click.Choice(sorted(PRESETS)),
                help="Experiment to run; the table above says which preset fits a change.",
            ),
            click.option(
                "--set",
                "settings",
                multiple=True,
                metavar="KEY=VALUE",
                help="Change one key of the composed SkyRL config: a dotted key and a YAML value, for example "
                "trainer.algorithm.advantage_estimator=grpo. Use +KEY=VALUE for a key the base config lacks. "
                "Repeatable. Keys the preset owns (placement, data, evaluation, measurement) are refused with the "
                "reason; data presets accept none. Code changes need no flag: the run name includes the source "
                "tree hash.",
            ),
            click.option(
                "--require",
                "requirements",
                multiple=True,
                metavar="KEY<=|>=BOUND",
                help="Add a condition the gate checks on a metric key; the examples above show two. With a "
                "number, every candidate replicate must meet the bound. With control, the candidate mean must not "
                "be worse than the controls' beyond tolerance. Repeatable; gated presets only, not with --control. "
                "A training metric the controls never recorded is refused before launch.",
            ),
            click.option(
                "--baseline",
                "baselines",
                multiple=True,
                metavar="NAME@VERSION",
                help="Compare with earlier finished runs; the verdict still uses pinned controls. Repeatable.",
            ),
            click.option(
                "--tasks",
                "tasks_name",
                metavar="NAME@VERSION",
                help="Curated task selection at an immutable version. Required by data presets, whose selection "
                "must hold out the same panel as the default; train-long defaults "
                "to the pinned snowball-100k.",
            ),
            click.option(
                "--init",
                "init_name",
                metavar="NAME@VERSION",
                help="Initial weights: a published model artifact, or a finished train-long run (its HF export). "
                "learning presets and train-long only. The initial weights are evaluated too, so the report shows "
                "the gain.",
            ),
            click.option(
                "--keep-weights",
                is_flag=True,
                help="Keep a learning or data candidate's final HF weights. "
                "Learning controls and train-long keep them automatically.",
            ),
            click.option(
                "--resume-from",
                "resume_name",
                metavar="NAME[@VERSION]",
                help="Continue an earlier train-long run's checkpoints and optimizer state under a new run name. "
                "train-long only.",
            ),
            click.option(
                "--eval",
                "evaluations",
                multiple=True,
                type=click.Choice(sorted(EVALUATIONS)),
                help="Train-long evaluation requests; repeatable. "
                "The default 100-task panel runs before training and every 10 updates. "
                "A second native panel is refused.",
            ),
            click.option(
                "--replicates",
                type=click.IntRange(1, 5),
                default=1,
                show_default=True,
                help="Identical copies of a gated preset, sharing one seed. The gate compares their "
                "mean with the controls; more copies narrow the tolerance. Not for train-long.",
            ),
            click.option(
                "--control",
                is_flag=True,
                help="Run the preset's control replicates from a clean checkout with only its pinned default "
                "inputs. Only for pinning new controls; refuses --set, --tasks, --init, --eval, --require, "
                "--keep-weights and data-medium/data-long, which use the corresponding learning preset's controls.",
            ),
        )
        command = rl_build_options(function)
        for option in reversed(options):
            command = option(command)
        return click.command(cls=SnowballCommand, help=help_text)(command)

    return decorate
