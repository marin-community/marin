# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Snowball RL experiments through versioned artifact presets."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Protocol

import click
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from iris.rpc import job_pb2
from marin.execution.artifact import RECORD_FILENAME, Artifact, ArtifactRecord, is_mutable_version
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply
from marin.execution.step_status import get_status_path
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    MegatronGeometry,
    SkyRLModelSource,
    SkyRLPolicyModel,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    artifact_identity,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance
from taskcompendium.models import TaskSpec, grades_with_judge

PRESET_REVISION = "2026.10.26"
SEED = 20261020
MODEL_ID = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
MODEL_REVISION = "fe666478454e79fe6027a0bab0388d9abd68f51a"
MODEL_URI = "s3://marin-us-east-02a/models/open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
SELECTION_PREFIX = "rl-data/task-selection"
SELECTION_ROOT = "s3://marin-us-east-02a/marin/rl-data/task-selection"
DEFAULT_SELECTION_VERSION = "2026.10.26"
TASK_COLUMN = "task_spec"
MANIFEST_FILENAME = "task_manifest.parquet"
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


class Loop(StrEnum):
    GENERATE_ONLY = "generate-only"
    SYNC = "sync"
    ASYNC = "async"


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
    HARBOR_ON_DAYTONA = "harbor-on-daytona"


class EvalHarness(StrEnum):
    PANEL = "panel"
    HARBOR = "harbor"


@dataclass(frozen=True)
class Evaluation:
    name: str
    harness: EvalHarness
    tasks: str
    sample: int | None
    answers: int
    every: int | None
    at_start: bool
    gates: bool


@dataclass(frozen=True)
class Layout:
    name: str
    role_plan: SkyRLRolePlan
    megatron: MegatronGeometry | None
    cluster: str
    host_memory: str
    sockets: bool


@dataclass(frozen=True)
class Recipe:
    loop: Loop
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
    evaluations: tuple[Evaluation, ...]
    checkpoint_every: int | None
    guards: tuple[Metric, ...]
    reported: tuple[Metric, ...]
    gaps: tuple[Gap, ...]

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


EVAL_SCORE_SCHEMA = pa.schema(
    [
        ("evaluation", pa.string()),
        ("harness", pa.string()),
        ("step", pa.int32()),
        ("item", pa.string()),
        ("attempt", pa.int32()),
        ("family", pa.string()),
        ("aaii_category", pa.string()),
        ("score", pa.float64()),
        ("scored", pa.bool_()),
    ]
)


def _manifest(root: StoragePath) -> dict[str, dict]:
    path = root / MANIFEST_FILENAME
    fs, key = filesystem_for(str(path))
    if not fs.exists(key):
        return {}
    with fs.open(key, "rb") as stream:
        return {row["task_id"]: row for row in pq.read_table(stream).to_pylist()}


def _read_text(path: str | StoragePath) -> str:
    fs, key = filesystem_for(str(path))
    with fs.open(key, "rt") as stream:
        return stream.read()


def _paths(pattern: StoragePath) -> list[StoragePath]:
    fs, key = filesystem_for(str(pattern))
    return [StoragePath(fs.unstrip_protocol(path)) for path in sorted(fs.glob(key))]


class PanelEvaluator:
    required_env: tuple[str, ...] = ()

    def callbacks(self, evaluation: Evaluation) -> list[dict]:
        return [
            {
                "type": "evaluation",
                "eval_steps": evaluation.every,
                "eval_before_train": evaluation.at_start,
                "eval_on_train_end": True,
            }
        ]

    def configure(self, evaluation: Evaluation, config: dict, inputs: RunInputs) -> None:
        source, filename = evaluation.tasks.split(":")
        selection = inputs.default_selection if source == "default" else inputs.selection
        if evaluation.sample is not None:
            selection = sampled_tasks_step(
                selection, filename, evaluation.sample, inputs.seed, local_only=evaluation.name == "speed-512x4"
            )
        inputs.validation_data = (ArtifactDataSource(selection, relative_path=f"{filename}.parquet"),)
        config["trainer"].update(eval_num_prompts=None, dump_eval_results=True)
        config["generator"].update(
            eval_n_samples_per_prompt=evaluation.answers,
            eval_sampling_params={
                "temperature": 0.0 if evaluation.answers == 1 else 1.0,
                "top_p": 1.0,
                "logprobs": 0,
                "ignore_eos": config["generator"]["sampling_params"]["ignore_eos"],
            },
        )

    def scores(self, evaluation: Evaluation, run_output: str) -> pa.Table:
        root = StoragePath(run_output)
        launch = yaml.safe_load(_read_text(root / "resolved-launch.yaml"))
        manifest = {}
        for source in launch["inputs"]["validation_data"]:
            manifest.update(_manifest(StoragePath(source["uri"])))
        attempts = Counter()
        rows = []
        for session in _paths(root / "exports/dumped_evals/*"):
            name = PurePosixPath(str(session)).name
            step = -1 if name == "eval_only" else int(name.removeprefix("global_step_").removesuffix("_evals"))
            for path in _paths(session / "*.jsonl"):
                if PurePosixPath(str(path)).name == "aggregated_results.jsonl":
                    continue
                for line in _read_text(path).splitlines():
                    row = json.loads(line)
                    item = row.get("task_id")
                    if not item:
                        raise ValueError(f"evaluation answer lacks a TaskSpec id: {path}")
                    metadata = manifest.get(item, {})
                    if evaluation.tasks.split(":")[1] != "train" and not metadata:
                        raise ValueError(f"panel task {item} is absent from its task manifest")
                    value = row["score"]
                    score = float(sum(value) if isinstance(value, list) else value)
                    if not math.isfinite(score):
                        raise ValueError(f"nonfinite evaluation score for {item}")
                    rows.append(
                        {
                            "evaluation": evaluation.name,
                            "harness": evaluation.harness.value,
                            "step": step,
                            "item": item,
                            "attempt": attempts[step, item],
                            "family": metadata.get("family"),
                            "aaii_category": metadata.get("aaii_category"),
                            "score": score,
                            "scored": not row.get("exception_type") and not row.get("error_treatment"),
                        }
                    )
                    attempts[step, item] += 1
        return pa.Table.from_pylist(rows, schema=EVAL_SCORE_SCHEMA)


def evaluation_statistics(scores: pa.Table) -> dict[str, float]:
    """Aggregate answer scores separately for each evaluation and weight step."""
    sessions = defaultdict(list)
    for row in scores.to_pylist():
        sessions[row["evaluation"], row["step"]].append(row)
    result = {}
    for (evaluation, _step), rows in sessions.items():
        if len({row["step"] for row in scores.to_pylist() if row["evaluation"] == evaluation}) != 1:
            raise ValueError(f"select one weight step before aggregating {evaluation}")
        prefix = evaluation
        scored = [row for row in rows if row["scored"]]
        result[f"{prefix}/scored_fraction"] = len(scored) / len(rows)
        if not scored:
            continue
        result[f"{prefix}/mean"] = math.fsum(row["score"] for row in scored) / len(scored)
        tasks = defaultdict(list)
        for row in scored:
            tasks[row["item"]].append(row)
        result[f"{prefix}/mixed_fraction"] = sum(
            any(row["score"] == 1 for row in group) and any(row["score"] != 1 for row in group)
            for group in tasks.values()
        ) / len(tasks)
        result[f"{prefix}/all_right_fraction"] = sum(
            all(row["score"] == 1 for row in group) for group in tasks.values()
        ) / len(tasks)
        result[f"{prefix}/all_wrong_fraction"] = sum(
            all(row["score"] != 1 for row in group) for group in tasks.values()
        ) / len(tasks)
        for column in ("family", "aaii_category"):
            groups = defaultdict(list)
            for group in tasks.values():
                category = group[0][column]
                if category is not None:
                    groups[category].append(math.fsum(row["score"] for row in group) / len(group))
            means = {category: math.fsum(values) / len(values) for category, values in groups.items()}
            result.update({f"{prefix}/{column}/{category}/mean": mean for category, mean in means.items()})
            if column == "family" and means:
                result[f"{prefix}/family_mean"] = math.fsum(means.values()) / len(means)
    return result


@dataclass
class RunInputs:
    selection: ArtifactStep[Artifact]
    default_selection: ArtifactStep[Artifact]
    seed: int
    validation_data: tuple[ArtifactDataSource, ...] = ()


class Evaluator(Protocol):
    required_env: tuple[str, ...]

    def callbacks(self, evaluation: Evaluation) -> list[dict]: ...
    def configure(self, evaluation: Evaluation, config: dict, inputs: RunInputs) -> None: ...
    def scores(self, evaluation: Evaluation, run_output: str) -> pa.Table: ...


EVALUATORS: Mapping[EvalHarness, Evaluator] = MappingProxyType({EvalHarness.PANEL: PanelEvaluator()})
EVALUATIONS = MappingProxyType(
    {
        "panel-100": Evaluation("panel-100", EvalHarness.PANEL, "default:validation", None, 1, None, False, True),
        "panel-910": Evaluation("panel-910", EvalHarness.PANEL, "default:full_validation", None, 1, None, False, True),
        "speed-512x4": Evaluation("speed-512x4", EvalHarness.PANEL, "default:train", 512, 4, None, False, False),
        "screen-256x4": Evaluation("screen-256x4", EvalHarness.PANEL, "tasks:train", 256, 4, None, False, True),
    }
)
_P40_PLAN = SkyRLRolePlan(False, 4, 8, 1, 1, 1, 8, 8, 128, 128, 1, 4)
P40 = Layout("p40", _P40_PLAN, MegatronGeometry(1, 2, 1, 8, 1), "cw-us-east-02a", "1800GB", True)
E8 = replace(P40, name="e8", role_plan=replace(_P40_PLAN, generate_only=True), megatron=None)
L112 = Layout(
    "l112",
    SkyRLRolePlan(False, 8, 8, 12, 1, 1, 4, 4, 512, 512, 1, 16),
    MegatronGeometry(1, 2, 1, 16, 1),
    "cw-rno2a",
    "1800GB",
    False,
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
_LEARNING = Recipe(Loop.SYNC, P40, 24576, 8192, None, 0, _ESTIMATOR, _OPTIMIZER)
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
PRESETS = MappingProxyType(
    {
        preset.name: preset
        for preset in (
            Preset(
                Focus.SPEED,
                Duration.SHORT,
                replace(_LEARNING, loop=Loop.GENERATE_ONLY, layout=E8, max_new_tokens=4096, fixed_reply_tokens=4096),
                0,
                range(1),
                (EVALUATIONS["speed-512x4"],),
                None,
                (Metric("generate/tokens_per_second_per_gpu", True),),
                (
                    Metric("generate/wall_seconds", False),
                    Metric("generate/response_tokens", True),
                    Metric("generate/vllm/latency_ttft_mean", False),
                    Metric("generate/vllm/peak_gpu_cache_usage_perc", False),
                ),
                _GAPS,
            ),
            Preset(
                Focus.DATA,
                Duration.SHORT,
                replace(_LEARNING, loop=Loop.GENERATE_ONLY, layout=E8),
                0,
                range(1),
                (EVALUATIONS["screen-256x4"],),
                None,
                (Metric("screen-256x4/mixed_fraction", True),),
                (
                    Metric("screen-256x4/mean", True),
                    Metric("screen-256x4/all_right_fraction", False),
                    Metric("screen-256x4/all_wrong_fraction", False),
                    Metric("screen-256x4/scored_fraction", True),
                    Metric("generate/prompts_dropped_too_long", False),
                ),
                _GAPS,
            ),
            Preset(
                Focus.SPEED,
                Duration.MEDIUM,
                replace(
                    _LEARNING,
                    max_new_tokens=4096,
                    fixed_reply_tokens=4096,
                    estimator={**_ESTIMATOR, "rloo_n_filter_zero_reward_groups": False},
                ),
                10,
                range(2, 11),
                (),
                None,
                (
                    Metric("speed/update_seconds", False, WindowStatistic.MEDIAN),
                    Metric("policy/mismatch/staleness0/kl_k3", False),
                ),
                _SPEED_REPORTS,
                (*_GAPS, Gap.MISMATCH_IS_AGREEMENT, Gap.LAYOUT_P40_ONLY, Gap.NO_ASYNC_SPEED_GATE),
            ),
            Preset(
                Focus.LEARNING,
                Duration.MEDIUM,
                _LEARNING,
                25,
                range(21, 26),
                (EVALUATIONS["panel-910"],),
                None,
                (Metric("panel-910/family_mean", True),),
                (*_TRAINING_REPORTS, Metric("panel-910/mean", True)),
                _GAPS,
            ),
            Preset(
                Focus.DATA,
                Duration.MEDIUM,
                _LEARNING,
                25,
                range(21, 26),
                (EVALUATIONS["panel-910"],),
                None,
                (Metric("panel-910/family_mean", True),),
                (*_TRAINING_REPORTS, Metric("panel-910/mean", True)),
                _GAPS,
            ),
            Preset(
                Focus.TRAIN,
                Duration.LONG,
                replace(
                    _LEARNING,
                    loop=Loop.ASYNC,
                    layout=L112,
                    request_window_tokens=65536,
                    max_new_tokens=32768,
                    staleness=2,
                    estimator={**_ESTIMATOR, "group_advantage_min_size": 4, "off_policy_correction": "tis"},
                    optimizer={**_OPTIMIZER, "lr": 4e-6},
                ),
                100,
                range(1, 101),
                (replace(_PANEL, every=10, at_start=True), EVALUATIONS["panel-910"]),
                10,
                (),
                (),
                _GAPS,
            ),
        )
    }
)


def _summary(selection: ArtifactStep) -> dict:
    return json.loads(_read_text(StoragePath(selection.path()) / "summary.json"))


def _write_sample(selection: str, filename: str, count: int, seed: int, local_only: bool, output: str) -> None:
    generator = random.Random(seed)
    rows = []
    seen = 0
    fs, key = filesystem_for(f"{selection}/{filename}.parquet")
    with fs.open(key, "rb") as stream:
        source = pq.ParquetFile(stream)
        schema = source.schema_arrow
        for batch in source.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                if local_only and grades_with_judge(TaskSpec.model_validate_json(row[TASK_COLUMN]).grader):
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
    fs, key = filesystem_for(f"{output}/{filename}.parquet")
    fs.makedirs(str(PurePosixPath(key).parent), exist_ok=True)
    with fs.open(key, "wb") as stream:
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), stream)
    for name in (MANIFEST_FILENAME, "summary.json"):
        source_fs, source_key = filesystem_for(f"{selection}/{name}")
        if source_fs.exists(source_key):
            fs, key = filesystem_for(f"{output}/{name}")
            with source_fs.open(source_key, "rb") as source, fs.open(key, "wb") as target:
                target.write(source.read())


def sampled_tasks_step(
    selection: ArtifactStep, filename: str, count: int, seed: int, *, local_only: bool
) -> ArtifactStep:
    identity = fingerprint_hash(json.dumps((selection.name, selection.version, filename, count, seed, local_only)))
    return apply(
        user_owned_name(f"datasets/snowball-rl/sample-{identity}"),
        _write_sample,
        version=PRESET_REVISION,
        selection=selection,
        filename=filename,
        count=count,
        seed=seed,
        local_only=local_only,
        output=OUT,
    )


_BATCH_SETTINGS = {
    "trainer.train_batch_size": "train_batch_size",
    "trainer.policy_mini_batch_size": "policy_mini_batch_size",
    "generator.n_samples_per_prompt": "n_samples_per_prompt",
}
_COMPARISON_KEYS = {
    "trainer.max_steps",
    "trainer.eval_interval",
    "trainer.eval_before_train",
    "trainer.seed",
    "trainer.ckpt_interval",
    "trainer.logger",
    "trainer.project_name",
}
_DERIVED_KEYS = {
    "generator.max_input_length",
    "trainer.max_prompt_length",
    "generator.max_turns",
    "generator.engine_init_kwargs.max_model_len",
    "trainer.hf_save_interval",
    "trainer.eval_interval",
    "trainer.eval_before_train",
    "trainer.ckpt_interval",
    "generator.sampling_params.max_generate_length",
    "generator.trajectory_reward_shaping.overlong.l_max",
    "generator.trajectory_reward_shaping.overlong.l_cache",
}


def _apply_setting(config: dict, setting: Setting, preset: Preset) -> None:
    key = setting.key
    refused = (
        key in _DERIVED_KEYS
        or key == "entrypoint"
        or key == "trainer.resume_mode"
        or key.startswith(
            (
                "trainer.placement",
                "trainer.callbacks",
                "trainer.eval_",
                "generator.eval_",
                "generator.retention",
                "generator.run_engines_locally",
                "trainer.max_ckpts_to_keep",
                "generator.inference_engine_",
                "generator.num_inference_engines",
            )
        )
    )
    refused |= key in ("trainer", "generator", "data", "context_budget")
    if key.startswith("data."):
        refused |= not (
            preset.focus in (Focus.LEARNING, Focus.TRAIN) and (key == "data.shuffle" or key.startswith("data.sampling."))
        )
    if preset.focus != Focus.TRAIN:
        refused |= (
            key in _COMPARISON_KEYS
            or key.startswith(("context_budget.", "generator.eval_"))
            or any(part in key.split(".") for part in ("max_generate_length", "ignore_eos", "min_tokens"))
        )
    if preset.focus == Focus.SPEED:
        refused |= key in ("generator.sampling_params", "generator.sampling_params.logprobs", "trainer.training_metrics")
    refused |= key in _BATCH_SETTINGS and preset.focus not in (Focus.LEARNING, Focus.TRAIN)
    if preset.recipe.loop == Loop.GENERATE_ONLY:
        refused |= not key.startswith("generator.") or key.startswith(("generator.sampling_params.", "generator.eval_"))
    refused |= preset.recipe.loop == Loop.GENERATE_ONLY and key == "generator.sampling_params"
    if refused:
        raise click.BadParameter(f"{key} is owned by the preset, context budget, layout or evaluation")
    base = yaml.safe_load(
        (Path(__file__).parents[2] / "vendor/skyrl/skyrl-train/skyrl_train/config/ppo_base_config.yaml").read_text()
    )
    current = base
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            if not setting.allows_new_key:
                raise click.BadParameter(f"unknown setting {key}; prefix with + to add it")
            break
        current = current[part]
    node = config
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise click.BadParameter(f"{key} descends through a scalar")
    node[parts[-1]] = setting.value


def training_config(
    preset: Preset, settings: tuple[Setting, ...] = (), evaluations: tuple[Evaluation, ...] = ()
) -> dict:
    recipe = preset.recipe
    plan = recipe.layout.role_plan
    evaluations = evaluations or tuple(
        evaluation for evaluation in preset.evaluations if preset.focus != Focus.TRAIN or evaluation.name != "panel-910"
    )
    config = {
        "entrypoint": "generate" if plan.generate_only else "taskcompendium",
        "context_budget": {
            "request_window_tokens": recipe.request_window_tokens,
            "max_new_tokens_per_turn": recipe.max_new_tokens,
            "max_turns": 999999 if recipe.layout.name == "l112" else 1,
        },
        "environment": {
            "task_sessions": {
                "local": {},
                "remote_judge": {"base_url": "https://api.together.xyz/v1", "api_key_env": JUDGE_KEY},
            }
        },
        "data": {"kind": "parquet", "shuffle": True},
        "trainer": {
            "strategy": "megatron",
            "flash_attn": False,
            "use_sample_packing": False,
            "gradient_checkpointing": True,
            "offload_optimizer_during_rollouts": False,
            "epochs": 50,
            "max_steps": preset.updates or 1,
            "update_epochs_per_batch": 1,
            "micro_forward_batch_size_per_gpu": plan.micro_train_batch_size_per_gpu,
            "eval_batch_size": 256,
            "eval_interval": preset.updates if evaluations else -1,
            "eval_before_train": False,
            "resume_mode": "latest" if preset.focus == Focus.TRAIN else "none",
            "max_ckpts_to_keep": 5 if preset.checkpoint_every else 2,
            "seed": SEED,
            "logger": ["console", "files", "wandb"],
            "project_name": "snowball-rl",
            "tracker_commit_each_step": True,
            "training_metrics": True,
            "rollout_spans": True,
            "policy_train_spans": True,
            "algorithm": dict(recipe.estimator),
            "policy": {"optimizer_config": dict(recipe.optimizer)},
            "rollout_buffer": {
                "max_staleness_steps": recipe.staleness,
                "batch_policy": "full_batch",
                "max_in_flight": plan.train_batch_size,
                "object_store_root": None,
            },
            "ckpt_interval": 0,
            "hf_save_interval": -1,
            "callbacks": [],
        },
        "generator": {
            "backend": "vllm",
            "model_dtype": "bfloat16",
            "vllm_attention_backend": "FLASH_ATTN",
            "run_engines_locally": True,
            "weight_sync_backend": "nccl",
            "weight_sync_pause": {"mode": "keep", "clear_cache": True},
            "weight_sync_transport": "expert_block",
            "expert_block_sync": {"timeout_seconds": 600, "verify": False},
            "gpu_memory_utilization": 0.75,
            "max_num_seqs": 1024,
            "max_num_batched_tokens": 8192,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "enforce_eager": False,
            "enable_http_endpoint": False,
            "chat_template": None,
            "apply_overlong_filtering": False,
            "engine_init_kwargs": {"moe_backend": "triton"},
            "sampling_params": {
                "temperature": 1.0,
                "top_p": 1.0,
                "logprobs": 0,
                "ignore_eos": recipe.fixed_reply_tokens is not None,
            },
        },
        "trajectory_runner": {"max_concurrent_tasks": 1024},
        "extra_env": {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "VERIFYIT_JUDGE_MODEL": "openai/gpt-oss-120b",
        },
    }
    geometry = recipe.layout.megatron or P40.megatron
    config["trainer"]["policy"]["megatron_config"] = asdict(geometry)
    config["trainer"]["ref"] = {"megatron_config": asdict(geometry)}
    if recipe.layout.name == "l112":
        config["generator"]["engine_init_kwargs"].update(
            enable_return_routed_experts=True,
            enable_auto_tools=True,
            tool_parser="hermes",
            logprobs_mode="processed_logprobs",
            generation_config="vllm",
            validate_rollout_logprob_sampling=True,
        )
        config["trainer"]["policy"]["megatron_config"].update(
            logprob_chunk_size=256,
            ddp_config={"overlap_grad_reduce": False, "overlap_param_gather": False, "grad_reduce_in_fp32": False},
            transformer_config_kwargs={
                "recompute_granularity": "full",
                "recompute_method": "uniform",
                "recompute_modules": ["core_attn"],
                "recompute_num_layers": 1,
            },
            optimizer_checkpoint_sharding_type="dp_reshardable",
        )
    if recipe.layout.sockets:
        config["extra_env"].update(NCCL_NET="Socket", NCCL_IB_DISABLE="1", NCCL_SOCKET_FAMILY="AF_INET")
    for setting in settings:
        _apply_setting(config, setting, preset)
    return config


@dataclass(frozen=True)
class SnowballRun:
    runs: tuple[ArtifactStep[SkyRLRun], ...]
    post_run_evaluations: tuple[ArtifactStep[SkyRLRun], ...] = ()
    gate: ArtifactStep[Artifact] | None = None


def _default_selection() -> ArtifactStep[Artifact]:
    return ArtifactStep.adopt(
        f"{SELECTION_PREFIX}/snowball-100k",
        DEFAULT_SELECTION_VERSION,
        f"{SELECTION_ROOT}/snowball-100k/{DEFAULT_SELECTION_VERSION}",
    )


def _artifact_prefix(cluster: str) -> str:
    config = yaml.safe_load((Path(__file__).parents[2] / f"lib/iris/config/{cluster}.yaml").read_text())
    return config["defaults"]["task_env"]["MARIN_PREFIX"]


def _record(step: ArtifactStep) -> ArtifactRecord:
    return ArtifactRecord.model_validate_json(_read_text(StoragePath(step.path()) / RECORD_FILENAME))


def _named_artifact(value: str, *, selection: bool = False, resume: bool = False) -> ArtifactStep:
    name, separator, version = value.rpartition("@")
    if not separator:
        name, version = value, PRESET_REVISION
    if is_mutable_version(version):
        raise click.BadParameter("use an immutable artifact version instead of dev")
    if selection:
        return ArtifactStep.adopt(f"{SELECTION_PREFIX}/{name}", version, f"{SELECTION_ROOT}/{name}/{version}")
    for layout in (P40, L112):
        root = f"{_artifact_prefix(layout.cluster)}/{name}/{version}"
        if resume:
            fs, key = filesystem_for(f"{root}/terminal.json")
            if fs.exists(key):
                return ArtifactStep.adopt(name, version, root, kind=SkyRLRun)
        fs, key = filesystem_for(f"{root}/{RECORD_FILENAME}")
        if fs.exists(key):
            record = ArtifactRecord.model_validate_json(_read_text(f"{root}/{RECORD_FILENAME}"))
            kind = SkyRLRun if record.result_type.endswith(".SkyRLRun") else LevanterCheckpoint
            return ArtifactStep.adopt(name, version, record.source or root, kind=kind, config=record.config)
    raise click.BadParameter(f"no published artifact record for {name}@{version}")


def _model_input(init: ArtifactStep | None) -> SkyRLModelSource:
    if init is None:
        model = ArtifactStep.adopt(
            f"models/snowball-datakit-sft-{MODEL_REVISION}", "2026.09.21", MODEL_URI, kind=LevanterCheckpoint
        )
        return ArtifactHfModel(model, MODEL_ID, MODEL_REVISION, relative_path="")
    if init.artifact_type is SkyRLRun:
        result = _record(init).result or {}
        if not result.get("hf_model_uri"):
            raise click.BadParameter("the initializer run has no HF export; gated runs cannot initialize another run")
        return SkyRLPolicyModel(init)
    return ArtifactHfModel(init, MODEL_ID, MODEL_REVISION)


def _provenance(*, control: bool) -> Provenance:
    root = Path(__file__).parents[2]
    provenance = Provenance.capture(root)
    if (root / ".git").exists():
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "experiments", "lib", "vendor"], cwd=root, text=True
        ).splitlines()
        if untracked:
            raise click.BadParameter(f"stage source files before naming a run: {', '.join(untracked)}")
    if not provenance.tree_hash or (control and provenance.dirty):
        raise click.BadParameter("controls require a clean checkout; runs require git or Iris source provenance")
    return provenance


def _callbacks(preset: Preset, evaluations: tuple[Evaluation, ...], expected_tasks: int) -> list[dict]:
    callbacks = []
    if preset.checkpoint_every:
        callbacks.append({"type": "checkpoint", "save_steps": preset.checkpoint_every})
    for evaluation in evaluations:
        scheduled = replace(evaluation, every=evaluation.every or preset.updates + 1)
        entries = EVALUATORS[evaluation.harness].callbacks(scheduled)
        if evaluation.every and preset.updates % evaluation.every == 0:
            for entry in entries:
                if entry["type"] == "evaluation":
                    entry["eval_on_train_end"] = False
        callbacks.extend(entries)
    if preset.focus == Focus.TRAIN:
        callbacks.extend(
            [
                {
                    "type": "best_checkpoint",
                    "initial_model_uri": "<resolved-model>",
                    "expected_tasks": expected_tasks,
                    "minimum_scored_tasks": math.ceil(0.99 * expected_tasks),
                },
                {
                    "type": "hf_model_save",
                    "save_steps": ((preset.updates // preset.checkpoint_every) + 1) * preset.checkpoint_every,
                    "save_on_train_end": True,
                },
            ]
        )
    return callbacks


def _resume_step(step: ArtifactStep[SkyRLRun], earlier: ArtifactStep[SkyRLRun]) -> ArtifactStep[SkyRLRun]:
    build = step.build_config

    def configure(ctx: StepContext):
        config = build(ctx)
        checkpoint_root = (
            f"<resume:{earlier.name}@{earlier.version}>" if ctx.is_fingerprint else ctx.resolved(earlier).checkpoint_root
        )
        launch = yaml.safe_load(config.launch_config_yaml)
        launch["artifacts"]["checkpoint_root"] = checkpoint_root
        return replace(
            config,
            launch_config_yaml=yaml.safe_dump(launch, sort_keys=False),
            output=replace(config.output, checkpoint_root=checkpoint_root),
        )

    return replace(step, build_config=configure, deps=(*step.deps, earlier))


def _resume_metadata(
    source: str,
    source_name: str,
    source_version: str,
    model_identity: str,
    tokenizer: str,
    tokenizer_revision: str,
    output: str,
) -> SkyRLRun:
    terminal = json.loads(_read_text(f"{source}/terminal.json"))
    config = terminal["config"]
    result = terminal["result"]
    if (
        "/checkpoints/snowball-rl/train-long-" not in f"/{source_name}"
        or not config["run"]["export_hf"]
        or config["run"]["id"] != f"{source_name}-{source_version}"
    ):
        raise click.BadParameter("--resume-from requires a train-long run")
    model = config["inputs"]["model"]
    if (model["identity"], model["tokenizer_uri"], model["tokenizer_revision"]) != (
        model_identity,
        tokenizer,
        tokenizer_revision,
    ):
        raise click.BadParameter("resume source uses another model or tokenizer; use its matching --init artifact")
    root = config["artifacts"]["checkpoint_root"]
    marker = f"{root}/latest_ckpt_global_step.txt"
    fs, key = filesystem_for(marker)
    missing = f"no checkpoint remains at {root}; temporary checkpoints expire after 14 days"
    if not fs.exists(key):
        raise click.BadParameter(missing)
    step = int(_read_text(marker).strip())
    checkpoint_fs, checkpoint_key = filesystem_for(f"{root}/global_step_{step}")
    if not checkpoint_fs.exists(checkpoint_key):
        raise click.BadParameter(missing)
    export = result.get("model")
    return SkyRLRun(
        path=output,
        hf_model_uri=export["policy_export_uri"] if export else None,
        global_step=step,
        tokenizer_uri=tokenizer,
        tokenizer_revision=tokenizer_revision,
        checkpoint_root=root,
        draft_checkpoint_root=None,
        terminal_manifest_uri=f"{source}/terminal.json",
        iris_job_id=result["iris_job_id"],
    )


def _resume_input(source: ArtifactStep[SkyRLRun], model: SkyRLModelSource) -> ArtifactStep[SkyRLRun]:
    if isinstance(model, ArtifactHfModel):
        tokenizer, revision = model.tokenizer_uri, model.tokenizer_revision
    else:
        result = _record(model.step).result
        tokenizer, revision = result["tokenizer_uri"], result["tokenizer_revision"]
    inputs = dict(
        source=source.path(),
        source_name=source.name,
        source_version=source.version,
        model_identity=artifact_identity(model.step),
        tokenizer=tokenizer,
        tokenizer_revision=revision,
    )
    _resume_metadata(**inputs, output="<resume-input>")
    identity = fingerprint_hash(json.dumps((source.name, source.version)))
    return apply(
        user_owned_name(f"inputs/snowball-rl/resume-{identity}"),
        _resume_metadata,
        version=source.version,
        artifact_type=SkyRLRun,
        output=OUT,
        **inputs,
    )


def _run_step(
    name: str, preset: Preset, config: dict, model: SkyRLModelSource, inputs: RunInputs, required: set[str]
) -> ArtifactStep[SkyRLRun]:
    layout = preset.recipe.layout
    plan = layout.role_plan
    step = skyrl_step(
        SkyRLSpec(
            name=name,
            version=PRESET_REVISION,
            config_yaml=yaml.safe_dump(config, sort_keys=False),
            model=model,
            train_data=(
                () if plan.generate_only else (ArtifactDataSource(inputs.selection, relative_path="train.parquet"),)
            ),
            validation_data=inputs.validation_data,
            topology=SkyRLTopology(plan.num_nodes, 8, "H100", plan),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=5 if preset.checkpoint_every else 2),
            seed=config["trainer"]["seed"],
        ),
        IrisSkyRLExecution(
            cluster=layout.cluster,
            cluster_config=f"lib/iris/config/{layout.cluster}.yaml",
            cpu=32,
            memory=layout.host_memory,
            disk="2TB",
            priority="interactive",
            max_retries=0,
            target_cluster=layout.cluster,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=72,
            job_timeout_seconds=72 * 3600,
            wandb_entity="dogml",
            required_env_vars=tuple(sorted(required | {"WANDB_API_KEY"})),
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
        ),
        export_hf=preset.focus == Focus.TRAIN,
    )
    build = step.build_config

    def configure(ctx: StepContext):
        resolved = build(ctx)
        launch = yaml.safe_load(resolved.launch_config_yaml)
        for callback in launch["skyrl"]["trainer"].get("callbacks", []):
            if callback["type"] == "best_checkpoint":
                callback["initial_model_uri"] = launch["inputs"]["model"]["uri"]
        return replace(resolved, launch_config_yaml=yaml.safe_dump(launch, sort_keys=False))

    return replace(step, build_config=configure)


def _post_evaluation(
    run: ArtifactStep[SkyRLRun], evaluation: Evaluation, selection: ArtifactStep, parent: Layout
) -> ArtifactStep[SkyRLRun]:
    layout = replace(E8, cluster=parent.cluster, sockets=parent.sockets)
    preset = replace(
        PRESETS["data-short"],
        recipe=replace(_LEARNING, loop=Loop.GENERATE_ONLY, layout=layout),
        evaluations=(evaluation,),
    )
    config = training_config(preset)
    inputs = RunInputs(selection, selection, SEED)
    evaluator = EVALUATORS[evaluation.harness]
    evaluator.configure(evaluation, config, inputs)
    return _run_step(
        f"{run.name}/eval/{evaluation.name}", preset, config, SkyRLPolicyModel(run), inputs, set(evaluator.required_env)
    )


def build_run(
    preset: Preset,
    *,
    tasks: ArtifactStep | None = None,
    init: ArtifactStep | None = None,
    layout: Layout | None = None,
    resume_from: ArtifactStep[SkyRLRun] | None = None,
    settings: tuple[Setting, ...] = (),
    evaluations: tuple[str, ...] = (),
    replicates: int = 1,
    control: bool = False,
) -> SnowballRun:
    """Construct the complete Snowball experiment and its terminal evaluation dependencies."""
    if not 1 <= replicates <= 5:
        raise click.BadParameter("replicates must be between 1 and 5")
    if preset.focus == Focus.DATA:
        if settings or (tasks is None and not control):
            raise click.BadParameter("data presets require --tasks and accept no --set")
    elif preset.focus != Focus.TRAIN and tasks is not None:
        raise click.BadParameter("use a data or train-long preset with --tasks")
    if preset.focus not in (Focus.LEARNING, Focus.TRAIN) and init is not None:
        raise click.BadParameter("--init requires learning-medium or train-long")
    if preset.focus != Focus.TRAIN and (layout is not None or resume_from is not None or evaluations):
        raise click.BadParameter("--layout, --resume-from and --eval require train-long")
    if preset.focus == Focus.TRAIN and (replicates != 1 or control):
        raise click.BadParameter("train-long accepts neither --replicates nor --control")
    if control and (settings or tasks is not None or init is not None or evaluations or preset.name == "data-medium"):
        raise click.BadParameter("controls use only the preset's pinned default inputs")
    provenance = _provenance(control=control)
    if layout is not None:
        if layout.name == "p40":
            recipe = replace(
                _LEARNING, loop=Loop.ASYNC, staleness=2, estimator={**_ESTIMATOR, "off_policy_correction": "tis"}
            )
        else:
            recipe = replace(preset.recipe, layout=layout)
        preset = replace(preset, recipe=recipe)
    batch_changes = {field: setting.value for setting in settings if (field := _BATCH_SETTINGS.get(setting.key))}
    if batch_changes:
        plan = replace(preset.recipe.layout.role_plan, **batch_changes)
        preset = replace(preset, recipe=replace(preset.recipe, layout=replace(preset.recipe.layout, role_plan=plan)))
    default = _default_selection()
    selection = tasks or default
    if is_mutable_version(selection.version):
        raise click.BadParameter("curated task selections must have an immutable version")
    summary = _summary(selection)
    if (
        tasks is not None
        and preset.focus == Focus.DATA
        and summary["recipe"]["heldout"] != _summary(default)["recipe"]["heldout"]
    ):
        raise click.BadParameter("candidate tasks use a different heldout input; curate against the default panel")
    selected = {evaluation.name: evaluation for evaluation in preset.evaluations}
    for name in evaluations:
        if name not in EVALUATIONS:
            raise click.BadParameter(f"unknown evaluation {name}; choose from {', '.join(EVALUATIONS)}")
        selected[name] = EVALUATIONS[name]
    if init is not None:
        selected = {name: replace(evaluation, at_start=True) for name, evaluation in selected.items()}
    post = tuple(
        evaluation for evaluation in selected.values() if preset.focus == Focus.TRAIN and evaluation.name == "panel-910"
    )
    in_run = tuple(evaluation for evaluation in selected.values() if evaluation not in post)
    if sum(evaluation.harness == EvalHarness.PANEL for evaluation in in_run) > 1:
        raise click.BadParameter("one in-run TaskSpec panel is supported; panel-910 runs on the final export")
    required = {key for evaluation in selected.values() for key in EVALUATORS[evaluation.harness].required_env}
    counts = summary["grader_kind_counts"]["train"]
    if counts.get("remote_judge", 0) and preset.name != "speed-short":
        required.add(JUDGE_KEY)
    missing = sorted(key for key in required if not os.environ.get(key))
    if missing:
        raise click.BadParameter(f"missing required credentials: {', '.join(missing)}")
    config = training_config(preset, settings, in_run)
    preset = replace(preset, updates=config["trainer"]["max_steps"])
    model = _model_input(init)
    if resume_from is not None:
        resume_from = _resume_input(resume_from, model)
    inputs = RunInputs(selection, default, config["trainer"]["seed"])
    for evaluation in in_run:
        EVALUATORS[evaluation.harness].configure(evaluation, config, inputs)
    if not preset.recipe.layout.role_plan.generate_only:
        expected_tasks = _summary(default)["validation_rows"]
        config["trainer"]["callbacks"] = _callbacks(preset, in_run, expected_tasks)
    changes = {
        "tasks": (selection.name, selection.version),
        "init": None if init is None else (init.name, init.version),
        "layout": preset.recipe.layout.name,
        "resume": None if resume_from is None else (resume_from.name, resume_from.version),
        "settings": sorted({setting.key: setting.value for setting in settings}.items()),
        "evaluations": sorted(selected),
    }
    inputs_hash = fingerprint_hash(json.dumps(changes, sort_keys=True))
    runs = []
    post_runs = []
    for replicate in range(replicates):
        name = f"checkpoints/snowball-rl/{preset.name}-{provenance.tree_hash}-in-{inputs_hash}"
        if preset.focus != Focus.TRAIN:
            name += f"-r{replicate}"
        if control:
            name = f"checkpoints/snowball-rl/control/{preset.name}-{provenance.base_commit}-r{replicate}"
        else:
            name = user_owned_name(name)
        run = _run_step(name, preset, config, model, inputs, required)
        if resume_from is not None:
            run = _resume_step(run, resume_from)
        runs.append(run)
        for evaluation in post:
            post_runs.append(_post_evaluation(run, evaluation, default, preset.recipe.layout))
    return SnowballRun(tuple(runs), tuple(post_runs))


def _print_status(run: SnowballRun, preset: Preset) -> None:
    rows = []
    planned = []
    for step in (*run.runs, *run.post_run_evaluations):
        execution = next(value for value in step.runtime_args.values() if isinstance(value, IrisSkyRLExecution))
        path = f"{_artifact_prefix(execution.cluster)}/{step.name}/{step.version}"
        config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
        planned.append(yaml.safe_load(config.launch_config_yaml))
        fs, key = filesystem_for(f"{path}/terminal.json")
        result = json.loads(_read_text(f"{path}/terminal.json"))["result"] if fs.exists(key) else {}
        status_fs, status_key = filesystem_for(get_status_path(path))
        state = result.get("state")
        if state is None:
            state = _read_text(get_status_path(path)).strip() if status_fs.exists(status_key) else "not built"
        rows.append(
            {
                "name": step.name,
                "version": step.version,
                "state": state,
                "iris_job_id": result.get("iris_job_id"),
                "cluster": execution.cluster,
                "allocation": {
                    **planned[-1]["iris"]["allocation"],
                    "cpu": execution.cpu,
                    "memory": execution.memory,
                    "disk": execution.disk,
                },
                "output": path,
            }
        )
    click.echo(
        yaml.safe_dump(
            {
                "preset": preset.name,
                "code": asdict(Provenance.capture(Path(__file__).parents[2])),
                "inputs": {
                    "model": planned[0]["inputs"]["model"]["identity"],
                    **{
                        name: [source["identity"] for source in planned[0]["inputs"][name]]
                        for name in ("train_data", "validation_data")
                    },
                },
                "context_budget": planned[0]["skyrl"]["context_budget"],
                "updates": (
                    0 if preset.recipe.loop == Loop.GENERATE_ONLY else planned[0]["skyrl"]["trainer"]["max_steps"]
                ),
                "measured_updates": list(preset.measured),
                "budget_estimate": {
                    "speed-short": "9-12 min; 1.5 H100-h",
                    "data-short": "10-15 min; 2 H100-h",
                    "speed-medium": "35-40 min; 25 H100-h",
                    "learning-medium": "105-120 min; 70-80 H100-h",
                    "data-medium": "105-120 min; 70-80 H100-h",
                    "train-long": "hours to days",
                }[preset.name],
                "question": [metric.key for metric in preset.guards],
                "reported_metrics": [{**asdict(metric), "window": metric.window.value} for metric in preset.reported],
                "gaps": [gap.value for gap in preset.gaps],
                "runs": rows,
            },
            sort_keys=False,
        )
    )


@click.command(help=__doc__)
@click.option("--preset", "preset_name", required=True, type=click.Choice(sorted(PRESETS)))
@click.option("--set", "settings", multiple=True)
@click.option("--tasks", "tasks_name")
@click.option("--init", "init_name")
@click.option("--resume-from", "resume_name")
@click.option("--layout", "layout_name", type=click.Choice(["p40", "l112"]))
@click.option("--eval", "evaluations", multiple=True)
@click.option("--replicates", type=click.IntRange(1, 5), default=1)
@click.option("--control", is_flag=True)
@rl_build_options
def main(
    preset_name: str,
    settings: tuple[str, ...],
    tasks_name: str | None,
    init_name: str | None,
    resume_name: str | None,
    layout_name: str | None,
    evaluations: tuple[str, ...],
    replicates: int,
    control: bool,
) -> tuple[ArtifactStep, ...]:
    preset = PRESETS[preset_name]
    layout = LAYOUTS[layout_name] if layout_name else None
    run = build_run(
        preset,
        settings=tuple(parse_setting(setting) for setting in settings),
        tasks=_named_artifact(tasks_name, selection=True) if tasks_name else None,
        init=_named_artifact(init_name) if init_name else None,
        resume_from=_named_artifact(resume_name, resume=True) if resume_name else None,
        layout=layout,
        evaluations=evaluations,
        replicates=replicates,
        control=control,
    )
    if resume_name:
        click.echo(
            "Resume uses the earlier checkpoint root and optimizer state. Temporary checkpoints expire after 14 days. "
            "Verify checkpoint compatibility for changes to vendor/skyrl or lib/marin/src/marin/rl."
        )
    effective = replace(preset, recipe=replace(preset.recipe, layout=layout)) if layout else preset
    _print_status(run, effective)
    return (*run.runs, *run.post_run_evaluations, *((run.gate,) if run.gate else ()))


if __name__ == "__main__":
    main()
