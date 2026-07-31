# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch repeated two-phase StarCoder anchors for heteroskedastic SNR.

The historical two-phase StarCoder swarm is dense in two coordinates,
``(phase_0_starcoder, phase_1_starcoder)``, but it does not contain exact
same-mixture replicates. This launcher adds an extensible repeated-anchor panel
so we can estimate within-mixture variance as a function of location in that
2D landscape.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from fray.cluster import ResourceConfig
from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main
from marin.execution.remote import RemoteCallable, remote
from marin.rl.placement import marin_prefix_for_region
from marin.training.training import TrainLmOnPodConfig

from experiments.domain_phase_mix.analysis import create_analysis_step
from experiments.domain_phase_mix.config import PhaseSchedule, WeightConfig
from experiments.domain_phase_mix.domains import NEMOTRON_FULL_DOMAIN, STARCODER_DOMAIN
from experiments.domain_phase_mix.experiment import MixtureExperiment
from experiments.domain_phase_mix.proxy_sweep import regmix_60m_proxy
from experiments.domain_phase_mix.qsplit240_replay import (
    EVAL_DATASETS_CACHE_DEP_ENV_VAR,
    SKIP_EVAL_HARNESS_ENV_VAR,
    add_eval_cache_dependency_to_training_step,
    skip_eval_harness_for_training_step,
)
from experiments.domain_phase_mix.starcoder_metadata import (
    DEFAULT_STARCODER_OBJECTIVE,
    DOMAIN_TOKEN_COUNTS,
    NEMOTRON_TOKENS,
    STARCODER_TOKENS,
    TARGET_BUDGET,
)
from experiments.evals.task_configs import CODE_TASKS, CORE_TASKS, convert_to_task_metrics

logger = logging.getLogger(__name__)

BASE_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/t2s_heteroskedastic_snr_20260523"
COHORT = "two_phase_starcoder_heteroskedastic_snr"
SOURCE_PANEL_CSV = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "paper_plots"
    / "data"
    / "two_phase_starcoder_combined_143_from_wandb.csv"
)
SOURCE_PANEL_CSV_SHA256 = "sha256:3a953d2836dd0b5fcdd0ac335eff3e91b4360d34bb37a2e062372f1ded82ba04"
DEFAULT_LOCAL_ARTIFACT_DIR = (
    Path(__file__).resolve().parent / "exploratory" / "reference_outputs" / "starcoder_heteroskedastic_snr_20260523"
)
LOCAL_ANCHOR_MANIFEST_CSV = "anchor_manifest.csv"
LOCAL_TRAINING_MANIFEST_CSV = "training_manifest.csv"
LOCAL_RUN_SPECS_JSON = "run_specs.json"
LOCAL_SUMMARY_JSON = "summary.json"

DEFAULT_REPEATS = 5
DEFAULT_RUN_ID_BASE = 810_000
DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_REGION = "us-central1"
DEFAULT_TPU_ZONE = "us-central1-a"
DEFAULT_MAX_CONCURRENT = 8
EXPERIMENT_BUDGET = 1_000_000_000
BATCH_SIZE = 128
SEQ_LEN = 2048
PHASE_NAMES = ("phase_0", "phase_1")
DOMAIN_NAMES = ("nemotron_full", "starcoder")
PHASE_BOUNDARIES = (0.5,)
EVAL_DATASETS_CACHE_RELATIVE_PATH = "raw/eval-datasets/code-tasks"
TOKENIZER_CACHE_RELATIVE_PATH = "raw/tokenizers"
EXCLUDED_INLINE_EVAL_TASKS = ("wsc273",)
STARCODER_CORE_TASKS = tuple(task for task in CORE_TASKS if task.name not in EXCLUDED_INLINE_EVAL_TASKS)
EVAL_TASKS = STARCODER_CORE_TASKS + CODE_TASKS
TRAINING_EXTRA_DEPENDENCY_GROUPS = ("eval",)
ANALYSIS_METRICS = [
    "eval/loss",
    "eval/paloma/dolma_100_programing_languages/bpb",
    "eval/uncheatable_eval/bpb",
    *convert_to_task_metrics(STARCODER_CORE_TASKS, "acc"),
    *convert_to_task_metrics(STARCODER_CORE_TASKS, "acc_norm"),
    *convert_to_task_metrics(STARCODER_CORE_TASKS, "bpb"),
    *convert_to_task_metrics(STARCODER_CORE_TASKS, "choice_logprob"),
    "lm_eval/code2text_go_0shot/smoothed_bleu_4",
    "lm_eval/code2text_java_0shot/smoothed_bleu_4",
    "lm_eval/code2text_javascript_0shot/smoothed_bleu_4",
    "lm_eval/code2text_php_0shot/smoothed_bleu_4",
    "lm_eval/code2text_python_0shot/smoothed_bleu_4",
    "lm_eval/code2text_ruby_0shot/smoothed_bleu_4",
    "lm_eval/jsonschema_bench_easy_2shot/json_validity",
    "lm_eval/jsonschema_bench_easy_2shot/schema_compliance",
    "lm_eval/jsonschema_bench_medium_2shot/json_validity",
    "lm_eval/jsonschema_bench_medium_2shot/schema_compliance",
    "lm_eval/jsonschema_bench_hard_2shot/json_validity",
    "lm_eval/jsonschema_bench_hard_2shot/schema_compliance",
    "lm_eval/humaneval_0shot/pass@1,create_test",
    "lm_eval/averages/macro_avg_acc",
    "lm_eval/averages/macro_avg_acc_norm",
    "lm_eval/averages/macro_avg_bpb",
    "lm_eval/averages/macro_avg_smoothed_bleu_4",
]


@dataclass(frozen=True)
class AnchorSpec:
    """One location in the two-phase StarCoder landscape."""

    anchor_index: int
    anchor_id: str
    phase_0_starcoder: float
    phase_1_starcoder: float
    source: str
    description: str
    reference_run_id: int | None = None
    reference_bpb: float | None = None


@dataclass(frozen=True)
class StarcoderRepeatRunSpec:
    """Manifest row for one repeated StarCoder anchor training run."""

    run_id: int
    run_name: str
    cohort: str
    anchor_index: int
    anchor_id: str
    repeat_index: int
    repeats_per_anchor: int
    trainer_seed: int | None
    data_seed: int
    simulated_epoch_subset_seed: int | None
    source: str
    description: str
    reference_run_id: int | None
    reference_bpb: float | None
    phase_0_starcoder: float
    phase_1_starcoder: float
    phase_0_starcoder_epochs: float
    phase_1_starcoder_epochs: float
    total_starcoder_epochs: float
    experiment_budget: int
    target_budget: int
    num_train_steps: int
    phase_weights: dict[str, dict[str, float]]


DEFAULT_OBSERVED_GLOBAL_BEST = AnchorSpec(
    anchor_index=-1,
    anchor_id="observed_global_best",
    phase_0_starcoder=0.228,
    phase_1_starcoder=0.253,
    source="observed_143_panel",
    description="Best observed target BPB in the 143-row two-phase StarCoder panel.",
    reference_run_id=90040,
    reference_bpb=0.9057711362838744,
)
DEFAULT_OBSERVED_P0_ZERO_SLICE_BEST = AnchorSpec(
    anchor_index=-1,
    anchor_id="observed_p0_zero_slice_best",
    phase_0_starcoder=0.0,
    phase_1_starcoder=0.28140808480217894,
    source="observed_143_panel",
    description="Best observed target BPB among p0 StarCoder == 0 slice.",
    reference_run_id=10,
    reference_bpb=0.9086067080497742,
)


@dataclass(frozen=True)
class LaunchArtifacts:
    """Resolved StarCoder repeated-anchor launch graph."""

    name_prefix: str
    source_panel_csv: Path
    anchors: list[AnchorSpec]
    run_specs: list[StarcoderRepeatRunSpec]
    weight_configs_step: ExecutorStep
    cache_eval_datasets_step: ExecutorStep
    training_steps: list[ExecutorStep]
    analysis_step: ExecutorStep

    @property
    def steps(self) -> list[ExecutorStep]:
        return [
            self.cache_eval_datasets_step,
            self.weight_configs_step,
            *self.training_steps,
            self.analysis_step,
        ]


def _natural_starcoder_proportion() -> float:
    return STARCODER_TOKENS / (NEMOTRON_TOKENS + STARCODER_TOKENS)


def _validate_probability(value: float, *, name: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1], got {value}")


def _phase_weights(phase_0_starcoder: float, phase_1_starcoder: float) -> dict[str, dict[str, float]]:
    _validate_probability(phase_0_starcoder, name="phase_0_starcoder")
    _validate_probability(phase_1_starcoder, name="phase_1_starcoder")
    return {
        "phase_0": {"nemotron_full": 1.0 - phase_0_starcoder, "starcoder": phase_0_starcoder},
        "phase_1": {"nemotron_full": 1.0 - phase_1_starcoder, "starcoder": phase_1_starcoder},
    }


def _target_metric(row: dict[str, str]) -> float | None:
    value = row.get(DEFAULT_STARCODER_OBJECTIVE)
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _read_completed_source_rows(source_panel_csv: Path) -> list[dict[str, str]]:
    with source_panel_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    completed = [
        row
        for row in rows
        if row.get("status") == "completed"
        and _target_metric(row) is not None
        and row.get("phase_0_starcoder")
        and row.get("phase_1_starcoder")
    ]
    if not completed:
        raise ValueError(f"No completed StarCoder rows with {DEFAULT_STARCODER_OBJECTIVE} in {source_panel_csv}")
    return completed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _source_panel_csv_sha256(path: Path) -> str:
    if path.exists():
        return _sha256_file(path)
    if path == SOURCE_PANEL_CSV:
        return SOURCE_PANEL_CSV_SHA256
    raise FileNotFoundError(path)


def _observed_anchor(
    rows: list[dict[str, str]],
    *,
    anchor_id: str,
    description: str,
    predicate: Any,
) -> AnchorSpec:
    candidates = [row for row in rows if predicate(row)]
    if not candidates:
        raise ValueError(f"No source rows available for anchor {anchor_id}")
    best = min(candidates, key=lambda row: float(_target_metric(row)))
    return AnchorSpec(
        anchor_index=-1,
        anchor_id=anchor_id,
        phase_0_starcoder=float(best["phase_0_starcoder"]),
        phase_1_starcoder=float(best["phase_1_starcoder"]),
        source="observed_143_panel",
        description=description,
        reference_run_id=int(float(best["run_id"])),
        reference_bpb=float(_target_metric(best)),
    )


def _default_anchor_specs(source_panel_csv: Path) -> list[AnchorSpec]:
    natural = _natural_starcoder_proportion()
    if source_panel_csv.exists():
        rows = _read_completed_source_rows(source_panel_csv)
        p0_zero_best = _observed_anchor(
            rows,
            anchor_id="observed_p0_zero_slice_best",
            description="Best observed target BPB among p0 StarCoder == 0 slice.",
            predicate=lambda row: math.isclose(float(row["phase_0_starcoder"]), 0.0, rel_tol=0.0, abs_tol=1e-12),
        )
        global_best = _observed_anchor(
            rows,
            anchor_id="observed_global_best",
            description="Best observed target BPB in the 143-row two-phase StarCoder panel.",
            predicate=lambda row: True,
        )
    elif source_panel_csv == SOURCE_PANEL_CSV:
        logger.warning(
            "Source panel CSV %s is not present; using frozen observed anchors from %s.",
            source_panel_csv,
            SOURCE_PANEL_CSV_SHA256,
        )
        global_best = DEFAULT_OBSERVED_GLOBAL_BEST
        p0_zero_best = DEFAULT_OBSERVED_P0_ZERO_SLICE_BEST
    else:
        raise FileNotFoundError(source_panel_csv)
    static = [
        AnchorSpec(
            -1,
            "proportional",
            natural,
            natural,
            "analytic_baseline",
            "Token-proportional StarCoder share in both phases.",
        ),
        AnchorSpec(-1, "nemotron_only", 0.0, 0.0, "analytic_baseline", "No StarCoder in either phase."),
        AnchorSpec(-1, "balanced", 0.5, 0.5, "analytic_baseline", "Equal Nemotron/StarCoder in both phases."),
        AnchorSpec(-1, "starcoder_only", 1.0, 1.0, "analytic_baseline", "All StarCoder in both phases."),
        AnchorSpec(
            -1,
            "late_code_moderate",
            0.0,
            0.25,
            "designed_anchor",
            "No early StarCoder and moderate late StarCoder, near the observed U-shape optimum.",
        ),
        AnchorSpec(
            -1,
            "two_stage_default",
            0.01,
            0.8,
            "historical_baseline",
            "Original two-stage-inspired high late-code baseline.",
        ),
        AnchorSpec(
            -1,
            "early_code_high_late_low",
            0.8,
            0.1,
            "designed_anchor",
            "High early StarCoder with low late StarCoder.",
        ),
        AnchorSpec(
            -1,
            "high_both_moderate_late",
            0.5,
            0.8,
            "designed_anchor",
            "High StarCoder in both phases, with heavier phase 1.",
        ),
    ]
    return [*static[:4], global_best, p0_zero_best, *static[4:]]


def _read_extra_anchor_specs(path: Path) -> list[AnchorSpec]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"anchor_id", "phase_0_starcoder", "phase_1_starcoder"}
    if not rows:
        raise ValueError(f"No rows in anchor CSV {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Anchor CSV {path} is missing columns: {sorted(missing)}")
    anchors = []
    for row in rows:
        anchors.append(
            AnchorSpec(
                anchor_index=-1,
                anchor_id=row["anchor_id"],
                phase_0_starcoder=float(row["phase_0_starcoder"]),
                phase_1_starcoder=float(row["phase_1_starcoder"]),
                source=row.get("source") or "extra_anchor_csv",
                description=row.get("description") or "Extra StarCoder repeated-anchor point.",
                reference_run_id=int(row["reference_run_id"]) if row.get("reference_run_id") else None,
                reference_bpb=float(row["reference_bpb"]) if row.get("reference_bpb") else None,
            )
        )
    return anchors


def build_anchor_specs(
    *,
    source_panel_csv: Path = SOURCE_PANEL_CSV,
    extra_anchor_csv: Path | None = None,
    include_default_anchors: bool = True,
) -> list[AnchorSpec]:
    """Build anchor specs, appending optional user-provided anchors."""
    anchors = _default_anchor_specs(source_panel_csv) if include_default_anchors else []
    if extra_anchor_csv is not None:
        anchors.extend(_read_extra_anchor_specs(extra_anchor_csv))
    ids = [anchor.anchor_id for anchor in anchors]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate anchor IDs: {[anchor_id for anchor_id in ids if ids.count(anchor_id) > 1]}")
    resolved = [replace(anchor, anchor_index=index) for index, anchor in enumerate(anchors)]
    validate_anchors(resolved)
    return resolved


def validate_anchors(anchors: list[AnchorSpec]) -> None:
    """Validate anchor-level invariants."""
    if not anchors:
        raise ValueError("At least one anchor is required")
    for index, anchor in enumerate(anchors):
        if anchor.anchor_index != index:
            raise ValueError(f"{anchor.anchor_id} has anchor_index={anchor.anchor_index}, expected {index}")
        if not anchor.anchor_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid anchor_id {anchor.anchor_id!r}")
        _validate_probability(anchor.phase_0_starcoder, name=f"{anchor.anchor_id}.phase_0_starcoder")
        _validate_probability(anchor.phase_1_starcoder, name=f"{anchor.anchor_id}.phase_1_starcoder")


def _starcoder_epochs(phase_starcoder: float) -> float:
    return 0.5 * TARGET_BUDGET * phase_starcoder / DOMAIN_TOKEN_COUNTS["starcoder"]


def _region_local_eval_cache_path(tpu_region: str) -> str:
    return f"{marin_prefix_for_region(tpu_region).rstrip('/')}/{EVAL_DATASETS_CACHE_RELATIVE_PATH}"


def _region_local_tokenizer_cache_base(tpu_region: str) -> str:
    return f"{marin_prefix_for_region(tpu_region).rstrip('/')}/{TOKENIZER_CACHE_RELATIVE_PATH}"


def build_run_specs(
    *,
    anchors: list[AnchorSpec],
    repeats: int = DEFAULT_REPEATS,
    run_id_base: int = DEFAULT_RUN_ID_BASE,
) -> list[StarcoderRepeatRunSpec]:
    """Expand anchors into repeated training specs."""
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    num_train_steps = EXPERIMENT_BUDGET // (BATCH_SIZE * SEQ_LEN)
    run_specs = []
    for anchor in anchors:
        for repeat_index in range(repeats):
            offset = anchor.anchor_index * repeats + repeat_index
            run_id = run_id_base + offset
            phase_weights = _phase_weights(anchor.phase_0_starcoder, anchor.phase_1_starcoder)
            run_specs.append(
                StarcoderRepeatRunSpec(
                    run_id=run_id,
                    run_name=f"t2s_snr_a{anchor.anchor_index:02d}_r{repeat_index:02d}",
                    cohort=COHORT,
                    anchor_index=anchor.anchor_index,
                    anchor_id=anchor.anchor_id,
                    repeat_index=repeat_index,
                    repeats_per_anchor=repeats,
                    trainer_seed=None,
                    data_seed=run_id,
                    simulated_epoch_subset_seed=None,
                    source=anchor.source,
                    description=anchor.description,
                    reference_run_id=anchor.reference_run_id,
                    reference_bpb=anchor.reference_bpb,
                    phase_0_starcoder=anchor.phase_0_starcoder,
                    phase_1_starcoder=anchor.phase_1_starcoder,
                    phase_0_starcoder_epochs=_starcoder_epochs(anchor.phase_0_starcoder),
                    phase_1_starcoder_epochs=_starcoder_epochs(anchor.phase_1_starcoder),
                    total_starcoder_epochs=_starcoder_epochs(anchor.phase_0_starcoder)
                    + _starcoder_epochs(anchor.phase_1_starcoder),
                    experiment_budget=EXPERIMENT_BUDGET,
                    target_budget=TARGET_BUDGET,
                    num_train_steps=num_train_steps,
                    phase_weights=phase_weights,
                )
            )
    validate_run_specs(run_specs, anchors=anchors, repeats=repeats, run_id_base=run_id_base)
    return run_specs


def validate_run_specs(
    run_specs: list[StarcoderRepeatRunSpec],
    *,
    anchors: list[AnchorSpec],
    repeats: int,
    run_id_base: int,
) -> None:
    """Validate repeated-anchor launch invariants."""
    expected_count = len(anchors) * repeats
    if len(run_specs) != expected_count:
        raise ValueError(f"Expected {expected_count} run specs, got {len(run_specs)}")
    run_ids = [spec.run_id for spec in run_specs]
    if run_ids != list(range(run_id_base, run_id_base + expected_count)):
        raise ValueError("Run IDs are not contiguous")
    run_names = [spec.run_name for spec in run_specs]
    if len(set(run_names)) != len(run_names):
        raise ValueError("Duplicate run names")
    for spec in run_specs:
        if spec.repeats_per_anchor != repeats:
            raise ValueError(f"{spec.run_name} has wrong repeats_per_anchor")
        for phase_name, weights in spec.phase_weights.items():
            if set(weights) != set(DOMAIN_NAMES):
                raise ValueError(f"{spec.run_name}/{phase_name} has wrong domains")
            total = sum(weights.values())
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{spec.run_name}/{phase_name} sums to {total}")
            if any(weight < 0.0 for weight in weights.values()):
                raise ValueError(f"{spec.run_name}/{phase_name} has negative weights")
        if spec.trainer_seed is not None:
            raise ValueError(f"{spec.run_name} changed trainer_seed away from historical convention")
        if spec.data_seed != spec.run_id:
            raise ValueError(f"{spec.run_name} data_seed must equal run_id")
        if spec.simulated_epoch_subset_seed is not None:
            raise ValueError(f"{spec.run_name} changed simulated_epoch_subset_seed away from historical convention")


def create_starcoder_repeat_experiment(
    *,
    name_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    eval_datasets_cache_path: str,
) -> MixtureExperiment:
    """Create the current-code equivalent of the old two-phase StarCoder proxy."""
    return MixtureExperiment(
        name=name_prefix,
        domains=[NEMOTRON_FULL_DOMAIN, STARCODER_DOMAIN],
        phase_schedule=PhaseSchedule.from_boundaries(boundaries=list(PHASE_BOUNDARIES), names=list(PHASE_NAMES)),
        model_config=regmix_60m_proxy,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        experiment_budget=EXPERIMENT_BUDGET,
        target_budget=TARGET_BUDGET,
        resources=ResourceConfig.with_tpu(tpu_type, regions=[tpu_region], zone=tpu_zone),
        eval_harness_tasks=EVAL_TASKS,
        eval_datasets_cache_path=eval_datasets_cache_path,
    )


def _configure_training_step(training_step: ExecutorStep, *, tpu_region: str) -> ExecutorStep:
    config = training_step.config
    if not isinstance(config, TrainLmOnPodConfig):
        raise TypeError(f"Expected TrainLmOnPodConfig for {training_step.name!r}, got {type(config)!r}")
    env_vars = dict(config.env_vars or {})
    env_vars["MARIN_PREFIX"] = marin_prefix_for_region(tpu_region)
    env_vars["MARIN_TOKENIZER_CACHE_PATH"] = _region_local_tokenizer_cache_base(tpu_region)
    env_vars["HF_ALLOW_CODE_EVAL"] = "1"
    configured = replace(training_step, config=replace(config, env_vars=env_vars))
    return _add_training_dependency_groups(configured, groups=TRAINING_EXTRA_DEPENDENCY_GROUPS)


def _merge_unique_strings(left: list[str], right: tuple[str, ...]) -> list[str]:
    merged = list(left)
    for item in right:
        if item not in merged:
            merged.append(item)
    return merged


def _add_training_dependency_groups(training_step: ExecutorStep, *, groups: tuple[str, ...]) -> ExecutorStep:
    """Install extra dependency groups in child training jobs.

    Old-style TPU training steps infer only the ``tpu`` extra from resources.
    The inline lm-eval harness imports ``lm_eval`` inside each child training
    job, so this wrapper carries the ``eval`` extra through Fray submission
    without changing the step's pinned output path.
    """
    if isinstance(training_step.fn, RemoteCallable):
        return replace(
            training_step,
            fn=replace(
                training_step.fn,
                pip_dependency_groups=_merge_unique_strings(training_step.fn.pip_dependency_groups, groups),
            ),
        )
    return replace(
        training_step,
        fn=remote(
            training_step.fn,
            resources=training_step.resources,
            pip_dependency_groups=list(groups),
        ),
    )


def build_launch_artifacts(
    *,
    name_prefix: str = BASE_NAME_PREFIX,
    repeats: int = DEFAULT_REPEATS,
    run_id_base: int = DEFAULT_RUN_ID_BASE,
    source_panel_csv: Path = SOURCE_PANEL_CSV,
    extra_anchor_csv: Path | None = None,
    include_default_anchors: bool = True,
    tpu_type: str = DEFAULT_TPU_TYPE,
    tpu_region: str = DEFAULT_TPU_REGION,
    tpu_zone: str = DEFAULT_TPU_ZONE,
    eval_datasets_cache_path: str | None = None,
    skip_inline_eval_harness: bool = False,
) -> LaunchArtifacts:
    """Resolve the launch graph without submitting it."""
    eval_cache_path = eval_datasets_cache_path or _region_local_eval_cache_path(tpu_region)
    expected_prefix = marin_prefix_for_region(tpu_region)
    if not eval_cache_path.startswith(expected_prefix):
        raise ValueError(f"Eval cache path must be local to {tpu_region} ({expected_prefix}), got {eval_cache_path}")
    os.environ["MARIN_PREFIX"] = marin_prefix_for_region(tpu_region)
    os.environ["MARIN_TOKENIZER_CACHE_PATH"] = _region_local_tokenizer_cache_base(tpu_region)
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    anchors = build_anchor_specs(
        source_panel_csv=source_panel_csv,
        extra_anchor_csv=extra_anchor_csv,
        include_default_anchors=include_default_anchors,
    )
    run_specs = build_run_specs(anchors=anchors, repeats=repeats, run_id_base=run_id_base)
    experiment = create_starcoder_repeat_experiment(
        name_prefix=name_prefix,
        tpu_type=tpu_type,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
        eval_datasets_cache_path=eval_cache_path,
    )
    weight_configs = [WeightConfig(run_id=spec.run_id, phase_weights=spec.phase_weights) for spec in run_specs]
    weight_configs_step = experiment.create_weight_configs_step(
        configs=weight_configs,
        summary={
            "cohort": COHORT,
            "repeats": repeats,
            "anchor_count": len(anchors),
            "source_panel_csv": str(source_panel_csv),
        },
        seed=0,
        name_prefix=name_prefix,
    )
    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=EVAL_TASKS,
        gcs_path=eval_cache_path,
        name_prefix=name_prefix,
    )
    training_steps = []
    for spec in run_specs:
        training_step = experiment.create_training_step(
            WeightConfig(run_id=spec.run_id, phase_weights=spec.phase_weights),
            name_prefix=name_prefix,
            run_name=spec.run_name,
            data_seed=spec.data_seed,
            simulated_epoch_subset_seed=spec.simulated_epoch_subset_seed,
        )
        training_step = add_eval_cache_dependency_to_training_step(training_step, cache_eval_datasets_step)
        training_step = _configure_training_step(training_step, tpu_region=tpu_region)
        if skip_inline_eval_harness:
            training_step = skip_eval_harness_for_training_step(training_step)
        training_steps.append(training_step)
    analysis_step = create_analysis_step(
        weight_configs_step=weight_configs_step,
        name_prefix=name_prefix,
        metrics=ANALYSIS_METRICS,
        depends_on=training_steps,
    )
    artifacts = LaunchArtifacts(
        name_prefix=name_prefix,
        source_panel_csv=source_panel_csv,
        anchors=anchors,
        run_specs=run_specs,
        weight_configs_step=weight_configs_step,
        cache_eval_datasets_step=cache_eval_datasets_step,
        training_steps=training_steps,
        analysis_step=analysis_step,
    )
    validate_launch_artifacts(
        artifacts,
        repeats=repeats,
        tpu_region=tpu_region,
        skip_inline_eval_harness=skip_inline_eval_harness,
    )
    return artifacts


def validate_launch_artifacts(
    artifacts: LaunchArtifacts,
    *,
    repeats: int,
    tpu_region: str,
    skip_inline_eval_harness: bool = False,
) -> None:
    """Validate graph-level invariants before launch."""
    validate_anchors(artifacts.anchors)
    validate_run_specs(
        artifacts.run_specs,
        anchors=artifacts.anchors,
        repeats=repeats,
        run_id_base=artifacts.run_specs[0].run_id,
    )
    if len(artifacts.training_steps) != len(artifacts.run_specs):
        raise ValueError("Training step count does not match run spec count")
    output_roots = [str(step.override_output_path or step.name) for step in artifacts.training_steps]
    if len(set(output_roots)) != len(output_roots):
        raise ValueError("Duplicate training output paths")
    expected_steps = EXPERIMENT_BUDGET // (BATCH_SIZE * SEQ_LEN)
    for step in artifacts.training_steps:
        config = step.config
        if not isinstance(config, TrainLmOnPodConfig):
            raise TypeError(f"Expected TrainLmOnPodConfig for {step.name!r}, got {type(config)!r}")
        env_vars = dict(config.env_vars or {})
        if env_vars.get("MARIN_PREFIX") != marin_prefix_for_region(tpu_region):
            raise ValueError(f"{step.name} has invalid MARIN_PREFIX={env_vars.get('MARIN_PREFIX')!r}")
        if env_vars.get("HF_ALLOW_CODE_EVAL") != "1":
            raise ValueError(f"{step.name} is missing HF_ALLOW_CODE_EVAL=1")
        if EVAL_DATASETS_CACHE_DEP_ENV_VAR not in env_vars:
            raise ValueError(f"{step.name} is missing eval cache dependency")
        has_skip = env_vars.get(SKIP_EVAL_HARNESS_ENV_VAR) == "1"
        if skip_inline_eval_harness and not has_skip:
            raise ValueError(f"{step.name} is missing {SKIP_EVAL_HARNESS_ENV_VAR}=1")
        if not skip_inline_eval_harness and has_skip:
            raise ValueError(f"{step.name} unexpectedly skips inline eval harness")
        if int(config.train_config.trainer.num_train_steps) != expected_steps:
            raise ValueError(f"{step.name} has wrong train steps")
        if config.train_config.hf_save_steps is None or int(config.train_config.hf_save_steps) != expected_steps:
            raise ValueError(f"{step.name} does not export HF at the final step")


def _flat_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    phase_weights = row.pop("phase_weights")
    for phase_name, weights in phase_weights.items():
        for domain_name, value in weights.items():
            row[f"{phase_name}_{domain_name}"] = value
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_local_manifests(artifacts: LaunchArtifacts, output_dir: Path) -> None:
    """Write local audit artifacts for dry-run and live launch review."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / LOCAL_ANCHOR_MANIFEST_CSV, [asdict(anchor) for anchor in artifacts.anchors])
    _write_csv(
        output_dir / LOCAL_TRAINING_MANIFEST_CSV,
        [_flat_manifest_row(asdict(spec)) for spec in artifacts.run_specs],
    )
    (output_dir / LOCAL_RUN_SPECS_JSON).write_text(
        json.dumps(
            {
                "name_prefix": artifacts.name_prefix,
                "cohort": COHORT,
                "anchors": [asdict(anchor) for anchor in artifacts.anchors],
                "run_specs": [asdict(spec) for spec in artifacts.run_specs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    summary = {
        "name_prefix": artifacts.name_prefix,
        "cohort": COHORT,
        "source_panel_csv": str(artifacts.source_panel_csv),
        "source_panel_csv_sha256": _source_panel_csv_sha256(artifacts.source_panel_csv),
        "anchor_count": len(artifacts.anchors),
        "repeats": artifacts.run_specs[0].repeats_per_anchor,
        "training_run_count": len(artifacts.run_specs),
        "target_metric": DEFAULT_STARCODER_OBJECTIVE,
        "excluded_inline_eval_tasks": list(EXCLUDED_INLINE_EVAL_TASKS),
        "skip_inline_eval_harness": any(
            isinstance(step.config, TrainLmOnPodConfig)
            and (step.config.env_vars or {}).get(SKIP_EVAL_HARNESS_ENV_VAR) == "1"
            for step in artifacts.training_steps
        ),
        "experiment_budget": EXPERIMENT_BUDGET,
        "target_budget": TARGET_BUDGET,
        "num_train_steps": artifacts.run_specs[0].num_train_steps,
        "natural_starcoder_proportion": _natural_starcoder_proportion(),
        "outputs": {
            "anchor_manifest_csv": LOCAL_ANCHOR_MANIFEST_CSV,
            "training_manifest_csv": LOCAL_TRAINING_MANIFEST_CSV,
            "run_specs_json": LOCAL_RUN_SPECS_JSON,
        },
    }
    (output_dir / LOCAL_SUMMARY_JSON).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _has_iris_context() -> bool:
    try:
        from iris.client.client import get_iris_ctx
    except ImportError:
        return False
    return get_iris_ctx() is not None


def _executor_prefix(executor_prefix: str | None) -> str | None:
    if executor_prefix is None:
        return None
    raise ValueError(
        "--executor-prefix is disabled for this launcher. The StarCoder panel relies on shared raw/tokenized/eval "
        "caches, and a custom executor prefix risks rematerializing shared data."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-name-prefix", default=BASE_NAME_PREFIX)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--run-id-base", type=int, default=DEFAULT_RUN_ID_BASE)
    parser.add_argument("--source-panel-csv", type=Path, default=SOURCE_PANEL_CSV)
    parser.add_argument("--extra-anchor-csv", type=Path)
    parser.add_argument("--no-default-anchors", action="store_true")
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--executor-prefix")
    parser.add_argument("--eval-datasets-cache-path")
    parser.add_argument(
        "--skip-inline-eval-harness",
        action="store_true",
        help=(
            "Set LEVANTER_SKIP_EVAL_HARNESS=1 on child training jobs. This keeps normal training "
            "validation/perplexity and checkpoint export, but avoids inline lm-eval/generation failures; "
            "run follow-up eval jobs from the preserved checkpoints instead."
        ),
    )
    parser.add_argument("--local-artifact-dir", type=Path, default=DEFAULT_LOCAL_ARTIFACT_DIR)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    sys.argv = [sys.argv[0]]
    _executor_prefix(args.executor_prefix)
    os.environ["MARIN_PREFIX"] = marin_prefix_for_region(args.tpu_region)
    os.environ["MARIN_TOKENIZER_CACHE_PATH"] = _region_local_tokenizer_cache_base(args.tpu_region)
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    if not args.dry_run and not args.allow_local and os.getenv("CI") is None and not _has_iris_context():
        raise ValueError("Non-dry-run launches must run inside Iris, e.g. via 'uv run iris --cluster=marin job run'.")

    artifacts = build_launch_artifacts(
        name_prefix=args.base_name_prefix,
        repeats=args.repeats,
        run_id_base=args.run_id_base,
        source_panel_csv=args.source_panel_csv,
        extra_anchor_csv=args.extra_anchor_csv,
        include_default_anchors=not args.no_default_anchors,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        eval_datasets_cache_path=args.eval_datasets_cache_path,
        skip_inline_eval_harness=args.skip_inline_eval_harness,
    )
    write_local_manifests(artifacts, args.local_artifact_dir)
    logger.info("Wrote local manifests to %s", args.local_artifact_dir)
    logger.info(
        "Prepared %d anchors x %d repeats = %d training runs.",
        len(artifacts.anchors),
        args.repeats,
        len(artifacts.training_steps),
    )
    logger.info(
        "Launch config: tpu=%s region=%s zone=%s max_concurrent=%d name_prefix=%s",
        args.tpu_type,
        args.tpu_region,
        args.tpu_zone,
        args.max_concurrent,
        args.base_name_prefix,
    )
    if args.dry_run or os.getenv("CI") is not None:
        return

    executor_main(
        ExecutorMainConfig(prefix=None, max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=(
            f"{args.base_name_prefix}: repeated two-phase StarCoder anchors for heteroskedastic SNR. "
            f"Runs {len(artifacts.anchors)} anchors x {args.repeats} repeats using the region-local "
            "StarCoder/Nemotron proxy setup."
        ),
    )


if __name__ == "__main__":
    main()
