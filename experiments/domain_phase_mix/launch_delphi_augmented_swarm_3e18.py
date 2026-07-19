# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the exact 280-row augmented two-phase fit panel at Delphi 3e18.

The source panel is the 300M modeling panel: 240 qsplit designs, the
stratified baseline, and 39 proportional domain deletions. This launcher keeps
the panel identities and weights fixed while replacing the model/training
configuration with the Delphi 3e18 configuration used by validation runs.

Every run includes the standard smooth validation datasets during training and
is followed by the Marin-native OLMoBaseEval Table-9 evaluator. Stable names and
seeds make the full graph safe to resubmit after partial failures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import fsspec
import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent
from levanter.main import train_lm
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, InputName, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.processing.tokenize.data_configs import (
    TokenizedMixtureGroup,
    TokenizerConfigLike,
    lm_varying_mixture_data_config,
)
from marin.rl.placement import marin_prefix_for_region
from marin.scaling_laws import ScalingFit, predict_optimal_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.domain_phase_mix.config import PhaseSchedule, WeightConfig
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (
    TARGET_BUDGET_DOLMA3_COMMON_CRAWL,
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
    TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
)
from experiments.domain_phase_mix.qsplit240_replay import SKIP_EVAL_HARNESS_ENV_VAR
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    DEFAULT_RUNTIME_CACHE_REGION,
    DOMAIN_NAMES,
    PHASE_BOUNDARIES,
    PHASE_NAMES,
    build_top_level_domains,
)
from experiments.llama import llama3_tokenizer
from experiments.paloma import paloma_tokenized
from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUT_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many" / "reference_outputs"
DEFAULT_SOURCE_PANEL = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_augmented_swarm_3e18_20260714/"
    "source/fit_panel_table9_macro-4f283bacb4ef269c.csv"
)
SOURCE_PANEL_SHA256 = "4f283bacb4ef269c396277cbd518ef74212a51741c909a1e1e9ace040751d507"
LOCAL_ARTIFACT_DIR = REFERENCE_OUTPUT_DIR / "delphi_augmented_swarm_3e18_20260714"

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_augmented_swarm_3e18_20260714"
DEFAULT_ANALYSIS_OUTPUT_PATH = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/" "delphi_baseline_mixtures_issue6607_20260623/analysis-af9355"
)
LABEL = "adamh_scaling_v6"
SEQ_LEN_DELPHI = 4096
MIXTURE_BLOCK_SIZE = 2048
SIMULATED_EPOCH_TARGET_BUDGET = TARGET_BUDGET_DOLMA3_COMMON_CRAWL
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
TARGET_FLOPS = 3e18
TARGET_TPU_TYPE = "v5p-8"
TARGET_BATCH_SIZE = 128
EXPECTED_RUNS = 280
EXPECTED_PANEL_COUNTS = {"qsplit_signal": 241, "domain_deletion": 39}
EXPECTED_QSPLIT_EXPERIMENT_COUNTS = {
    "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b": 240,
    "pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_300m_6b": 1,
}
RUN_ID_BASE = 7_141_000
DEFAULT_MAX_CONCURRENT = 56
HF_HUB_DISABLE_XET_ENV_VAR = "HF_HUB_DISABLE_XET"
TABLE9_REQUEST_SET_DIR = InputName.hardcoded("raw/eval-datasets/olmo_base_eval_table9/v2")
TABLE9_EVAL_RESOURCES = ResourceConfig.with_tpu("v6e-8", regions=["us-east5"], zone="us-east5-b", disk="80g")
PHASE_SCHEDULE = PhaseSchedule.from_boundaries(boundaries=PHASE_BOUNDARIES, names=list(PHASE_NAMES))
PHASE_FRACTIONS = {phase.name: phase.end_fraction - phase.start_fraction for phase in PHASE_SCHEDULE.phases}
RUN_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True)
class DelphiSwarmRunSpec:
    """Resolved identity, provenance, and weights for one swarm row."""

    run_order: int
    run_id: int
    run_name: str
    source_run_name: str
    source_experiment: str
    panel_source: str
    target_flops: float
    tpu_type: str
    tpu_region: str
    tpu_zone: str
    batch_size: int
    train_steps: int
    realized_train_tokens: int
    expected_checkpoint_step: int
    model_hidden_dim: int
    model_layers: int
    non_embedding_params: int
    total_trainable_params: int
    tensor_parallel_size: int
    data_seed: int
    trainer_seed: int
    phase_boundary: float
    phase_0_fraction: float
    phase_1_fraction: float
    simulated_epoch_target_budget: int
    available_top_level_tokens: int
    max_simulated_epoch: float
    q95_simulated_epoch: float
    mean_phase_tv_to_proportional: float
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class DelphiSwarmTrainingConfig:
    """Runtime configuration for one Delphi swarm training row."""

    analysis_output_path: str
    output_path: str
    run_spec: DelphiSwarmRunSpec
    validation_configs: dict[str, DatasetComponent] | None
    wandb_tags: tuple[str, ...]


@dataclass(frozen=True)
class SaveDelphiSwarmManifestConfig:
    """Configuration for writing the resolved swarm manifest."""

    output_path: str
    source_panel: str
    source_panel_sha256: str
    analysis_output_path: str
    run_specs_json: str


@dataclass(frozen=True)
class LaunchArtifacts:
    """Resolved manifest, training, and Table-9 evaluation graph."""

    manifest_step: ExecutorStep
    training_steps: list[ExecutorStep]
    eval_steps: list[ExecutorStep]

    @property
    def steps(self) -> list[ExecutorStep]:
        return [self.manifest_step, *self.training_steps, *self.eval_steps]


def _run_name(run_order: int, source_run_name: str) -> str:
    slug = RUN_NAME_PATTERN.sub("_", source_run_name).strip("_")
    if not slug:
        raise ValueError(f"Could not derive run name from {source_run_name!r}")
    return f"fit_{run_order:03d}_{slug}"


def _default_validation_sets(tokenizer: str, base_path: str = "tokenized/"):
    validation_sets = dict(paloma_tokenized(base_path=base_path, tokenizer=tokenizer))
    validation_sets.update(
        {
            os.path.join("uncheatable_eval", subset): step
            for subset, step in uncheatable_datasets(tokenizer=tokenizer).items()
        }
    )
    return validation_sets


def _read_scaling_fits(analysis_output_path: str) -> dict[str, ScalingFit]:
    result_path = os.path.join(analysis_output_path, "isoflop_analysis_result.json")
    fs, _, _ = fsspec.get_fs_token_paths(result_path)
    with fs.open(result_path, "r") as handle:
        analysis_result = json.load(handle)
    return {key: ScalingFit(float(value[0]), float(value[1])) for key, value in analysis_result["scaling_fits"].items()}


def _candidate_for_budget(*, scaling_fits: dict[str, ScalingFit]):
    candidate = predict_optimal_config(
        scaling_fits=scaling_fits,
        target_flops=TARGET_FLOPS,
        label=LABEL,
        heuristic=completed_adamh_heuristic,
        seq_len=SEQ_LEN_DELPHI,
    )
    if candidate is None:
        raise RuntimeError(f"Could not find optimal config for target_flops={TARGET_FLOPS:.2e}")
    train_steps = round(candidate.tokens / (TARGET_BATCH_SIZE * SEQ_LEN_DELPHI))
    optimizer_config = completed_adamh_heuristic.build_optimizer_config(TARGET_BATCH_SIZE, candidate.tokens)
    return replace(
        candidate,
        batch_size=TARGET_BATCH_SIZE,
        train_steps=train_steps,
        optimizer_config=optimizer_config,
    )


def _tensor_parallel_size(hidden_dim: int, tpu_type: str) -> int:
    chips = int(tpu_type.split("-")[1]) // 2
    tensor_parallel_size = 1
    while hidden_dim % (chips // tensor_parallel_size) != 0:
        tensor_parallel_size *= 2
        if tensor_parallel_size > chips:
            raise ValueError(f"Could not find tensor parallel size for hidden_dim={hidden_dim}, {tpu_type=}")
    return tensor_parallel_size


def _add_validation_components(data, validation_configs: dict[str, DatasetComponent] | None):
    if not validation_configs:
        return data
    new_components = {
        **data.components,
        **{key: value for key, value in validation_configs.items() if key not in data.components},
    }
    if isinstance(data.train_weights, dict):
        new_weights = {
            **data.train_weights,
            **{name: 0.0 for name in validation_configs if name not in data.train_weights},
        }
    else:
        new_weights = [
            (step_idx, {**weights, **{name: 0.0 for name in validation_configs if name not in weights}})
            for step_idx, weights in data.train_weights
        ]
    return replace(data, components=new_components, train_weights=new_weights)


def _quantile_95(values: list[float]) -> float:
    values = sorted(values)
    index = round(0.95 * (len(values) - 1))
    return values[index]


def _proportional_weights() -> dict[str, float]:
    total_tokens = float(TOP_LEVEL_TOTAL_AVAILABLE_TOKENS)
    return {domain: TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain] / total_tokens for domain in DOMAIN_NAMES}


def _phase_weights_from_row(row: dict[str, str], *, source_run_name: str) -> dict[str, dict[str, float]]:
    phase_weights: dict[str, dict[str, float]] = {phase_name: {} for phase_name in PHASE_NAMES}
    for phase_name in PHASE_NAMES:
        for domain in DOMAIN_NAMES:
            column = f"{phase_name}_{domain}"
            try:
                weight = float(row[column])
            except KeyError as error:
                raise ValueError(f"Source panel is missing required column {column!r}") from error
            if weight < 0:
                raise ValueError(f"{source_run_name}/{phase_name}/{domain} has negative weight {weight}")
            phase_weights[phase_name][domain] = weight

        total = sum(phase_weights[phase_name].values())
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"{source_run_name}/{phase_name} weights sum to {total}, expected 1.0")
    return phase_weights


def _weight_diagnostics(phase_weights: dict[str, dict[str, float]]) -> tuple[float, float, float]:
    proportional = _proportional_weights()
    aggregate_weights = {
        domain: sum(PHASE_FRACTIONS[phase] * phase_weights[phase][domain] for phase in PHASE_NAMES)
        for domain in DOMAIN_NAMES
    }
    simulated_epochs = [
        SIMULATED_EPOCH_TARGET_BUDGET * aggregate_weights[domain] / TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain]
        for domain in DOMAIN_NAMES
    ]
    phase_tvs = [
        0.5 * sum(abs(phase_weights[phase][domain] - proportional[domain]) for domain in DOMAIN_NAMES)
        for phase in PHASE_NAMES
    ]
    return max(simulated_epochs), _quantile_95(simulated_epochs), sum(phase_tvs) / len(phase_tvs)


def load_source_panel(
    *,
    source_panel: str,
    analysis_output_path: str,
    tpu_region: str,
    tpu_zone: str,
) -> list[DelphiSwarmRunSpec]:
    """Load and strictly validate the canonical 280-row panel."""
    with fsspec.open(source_panel, "rb") as handle:
        source_bytes = handle.read()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != SOURCE_PANEL_SHA256:
        raise ValueError(f"Source panel SHA-256 changed: {source_sha256} != {SOURCE_PANEL_SHA256}")
    rows = list(csv.DictReader(io.StringIO(source_bytes.decode("utf-8"))))

    if len(rows) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} source rows, found {len(rows)} in {source_panel}")
    source_names = [row["run_name"] for row in rows]
    if len(set(source_names)) != len(source_names):
        duplicates = [name for name, count in Counter(source_names).items() if count > 1]
        raise ValueError(f"Source panel has duplicate run names: {duplicates}")
    panel_counts = Counter(row["panel_source"] for row in rows)
    if dict(panel_counts) != EXPECTED_PANEL_COUNTS:
        raise ValueError(f"Panel source counts changed: {dict(panel_counts)} != {EXPECTED_PANEL_COUNTS}")
    qsplit_experiment_counts = Counter(
        row["source_experiment"] for row in rows if row["panel_source"] == "qsplit_signal"
    )
    if dict(qsplit_experiment_counts) != EXPECTED_QSPLIT_EXPERIMENT_COUNTS:
        raise ValueError(
            "Qsplit source experiment counts changed: "
            f"{dict(qsplit_experiment_counts)} != {EXPECTED_QSPLIT_EXPERIMENT_COUNTS}"
        )

    tpu_type, batch_size = TARGET_TPU_TYPE, TARGET_BATCH_SIZE
    scaling_fits = _read_scaling_fits(analysis_output_path)
    candidate = _candidate_for_budget(scaling_fits=scaling_fits)
    realized_train_tokens = candidate.train_steps * batch_size * SEQ_LEN_DELPHI
    if realized_train_tokens > SIMULATED_EPOCH_TARGET_BUDGET:
        raise ValueError(
            f"Delphi train tokens {realized_train_tokens} exceed simulated-epoch target "
            f"{SIMULATED_EPOCH_TARGET_BUDGET}"
        )
    non_embedding_params = int(candidate.model_config.total_trainable_params(0))
    total_params = int(candidate.model_config.total_trainable_params(completed_adamh_heuristic.vocab_size))
    tensor_parallel_size = _tensor_parallel_size(candidate.model_config.hidden_dim, tpu_type)

    run_specs: list[DelphiSwarmRunSpec] = []
    for run_order, row in enumerate(rows):
        source_run_name = row["run_name"]
        phase_weights = _phase_weights_from_row(row, source_run_name=source_run_name)
        max_epoch, q95_epoch, phase_tv = _weight_diagnostics(phase_weights)
        run_specs.append(
            DelphiSwarmRunSpec(
                run_order=run_order,
                run_id=RUN_ID_BASE + run_order,
                run_name=_run_name(run_order, source_run_name),
                source_run_name=source_run_name,
                source_experiment=row["source_experiment"],
                panel_source=row["panel_source"],
                target_flops=TARGET_FLOPS,
                tpu_type=tpu_type,
                tpu_region=tpu_region,
                tpu_zone=tpu_zone,
                batch_size=batch_size,
                train_steps=candidate.train_steps,
                realized_train_tokens=realized_train_tokens,
                expected_checkpoint_step=candidate.train_steps - 1,
                model_hidden_dim=int(candidate.model_config.hidden_dim),
                model_layers=int(candidate.model_config.num_layers),
                non_embedding_params=non_embedding_params,
                total_trainable_params=total_params,
                tensor_parallel_size=tensor_parallel_size,
                data_seed=RUN_ID_BASE + run_order,
                trainer_seed=0,
                phase_boundary=PHASE_BOUNDARIES[0],
                phase_0_fraction=PHASE_FRACTIONS["phase_0"],
                phase_1_fraction=PHASE_FRACTIONS["phase_1"],
                simulated_epoch_target_budget=SIMULATED_EPOCH_TARGET_BUDGET,
                available_top_level_tokens=TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
                max_simulated_epoch=max_epoch,
                q95_simulated_epoch=q95_epoch,
                mean_phase_tv_to_proportional=phase_tv,
                phase_weights=phase_weights,
            )
        )
    return run_specs


def _build_mixture_data(run_spec: DelphiSwarmRunSpec):
    domains = build_top_level_domains(runtime_cache_region=DEFAULT_RUNTIME_CACHE_REGION)
    runtime_components: dict[str, TokenizerConfigLike | TokenizedMixtureGroup] = {}
    for domain in domains:
        if len(domain.components) == 1:
            runtime_components[domain.name] = domain.components[0].get_step()
            continue
        runtime_components[domain.name] = TokenizedMixtureGroup(
            components={component.name: component.get_step() for component in domain.components},
            weights=domain.get_component_weights(),
            token_counts={component.name: int(component.weight) for component in domain.components},
        )

    weight_config = WeightConfig(run_id=run_spec.run_id, phase_weights=run_spec.phase_weights)
    weights_list = []
    for phase in PHASE_SCHEDULE.phases:
        start_step = phase.get_start_step_aligned(
            run_spec.train_steps,
            run_spec.batch_size,
            MIXTURE_BLOCK_SIZE,
        )
        weights_list.append((start_step, weight_config.get_weights_for_phase(phase.name)))
    data = lm_varying_mixture_data_config(
        components=runtime_components,
        weights_list=weights_list,
        shuffle=True,
        mixture_block_size=MIXTURE_BLOCK_SIZE,
    )
    return replace(
        data,
        target_budget=SIMULATED_EPOCH_TARGET_BUDGET,
        experiment_budget=run_spec.realized_train_tokens,
        simulated_epoch_subset_seed=None,
    )


def run_delphi_swarm_training(config: DelphiSwarmTrainingConfig) -> None:
    """Train one row using the exact Delphi 3e18 configuration."""
    run_spec = config.run_spec
    scaling_fits = _read_scaling_fits(config.analysis_output_path)
    candidate = _candidate_for_budget(scaling_fits=scaling_fits)
    if candidate.train_steps != run_spec.train_steps:
        raise ValueError(f"Resolved train steps changed: {candidate.train_steps} != {run_spec.train_steps}")
    params = candidate.model_config.total_trainable_params(completed_adamh_heuristic.vocab_size)
    tensor_parallel_size = _tensor_parallel_size(candidate.model_config.hidden_dim, run_spec.tpu_type)
    if int(params) != run_spec.total_trainable_params:
        raise ValueError(f"Resolved parameter count changed: {int(params)} != {run_spec.total_trainable_params}")

    data = _build_mixture_data(run_spec)
    data = _add_validation_components(data, config.validation_configs)
    inner_config = train_lm.TrainLmConfig(
        data=data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                entity="marin-community",
                project="marin",
                tags=[
                    "issue-6611",
                    *config.wandb_tags,
                    "completed-adamh",
                    f"panel_source={run_spec.panel_source}",
                    f"source_run={run_spec.source_run_name}",
                    f"FLOPs={run_spec.target_flops:.1e}",
                    f"D={run_spec.realized_train_tokens:.1e}",
                    f"D/N={run_spec.realized_train_tokens / params:.3f}",
                    f"label={LABEL}",
                    f"N={params:.1e}",
                    f"data_seed={run_spec.data_seed}",
                    f"trainer_seed={run_spec.trainer_seed}",
                ],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=run_spec.batch_size,
            per_device_parallelism=-1,
            num_train_steps=run_spec.train_steps,
            steps_per_eval=1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=5000)],
            ),
            mesh=MeshConfig(
                axes={"data": -1, "replica": 1, "model": tensor_parallel_size},
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                },
            ),
            seed=run_spec.trainer_seed,
            allow_nondivisible_batch_size=True,
        ),
        train_seq_len=SEQ_LEN_DELPHI,
        model=candidate.model_config,
        optimizer=candidate.optimizer_config,
        data_seed=run_spec.data_seed,
    )
    resources = ResourceConfig.with_tpu(
        run_spec.tpu_type,
        regions=[run_spec.tpu_region],
        zone=run_spec.tpu_zone,
    )
    run_levanter_train_lm(
        TrainLmOnPodConfig(
            train_config=inner_config,
            resources=resources,
            output_path=config.output_path,
            env_vars={
                "MARIN_PREFIX": marin_prefix_for_region(run_spec.tpu_region),
                SKIP_EVAL_HARNESS_ENV_VAR: "1",
            },
        )
    )


def save_delphi_swarm_manifest(config: SaveDelphiSwarmManifestConfig) -> None:
    """Persist immutable source identities, resolved configs, and phase weights."""
    run_specs = [DelphiSwarmRunSpec(**item) for item in json.loads(config.run_specs_json)]
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    with fs.open(os.path.join(config.output_path, "run_specs.json"), "w") as handle:
        json.dump([asdict(spec) for spec in run_specs], handle, indent=2, sort_keys=True)

    manifest_fields = [field for field in asdict(run_specs[0]) if field != "phase_weights"]
    manifest_buffer = io.StringIO(newline="")
    manifest_writer = csv.DictWriter(manifest_buffer, fieldnames=manifest_fields)
    manifest_writer.writeheader()
    for spec in run_specs:
        row = asdict(spec)
        row.pop("phase_weights")
        manifest_writer.writerow(row)
    with fs.open(os.path.join(config.output_path, "training_manifest.csv"), "w") as handle:
        handle.write(manifest_buffer.getvalue())

    weights_buffer = io.StringIO(newline="")
    weights_writer = csv.DictWriter(
        weights_buffer,
        fieldnames=["run_name", "source_run_name", "panel_source", "phase", "domain", "weight"],
    )
    weights_writer.writeheader()
    for spec in run_specs:
        for phase_name in PHASE_NAMES:
            for domain in DOMAIN_NAMES:
                weights_writer.writerow(
                    {
                        "run_name": spec.run_name,
                        "source_run_name": spec.source_run_name,
                        "panel_source": spec.panel_source,
                        "phase": phase_name,
                        "domain": domain,
                        "weight": spec.phase_weights[phase_name][domain],
                    }
                )
    with fs.open(os.path.join(config.output_path, "phase_weights.csv"), "w") as handle:
        handle.write(weights_buffer.getvalue())

    summary: dict[str, Any] = {
        "experiment_name": EXPERIMENT_NAME,
        "source_panel": config.source_panel,
        "source_panel_sha256": config.source_panel_sha256,
        "analysis_output_path": config.analysis_output_path,
        "n_runs": len(run_specs),
        "panel_counts": dict(Counter(spec.panel_source for spec in run_specs)),
        "source_experiment_counts": dict(Counter(spec.source_experiment for spec in run_specs)),
        "target_flops": TARGET_FLOPS,
        "phase_boundary": PHASE_BOUNDARIES[0],
        "phase_fractions": PHASE_FRACTIONS,
        "simulated_epoch_target_budget": SIMULATED_EPOCH_TARGET_BUDGET,
        "native_table9_scheduled": True,
        "heldout_policy": "These 280 rows are fit data; prior and future validation mixtures remain heldout.",
    }
    with fs.open(os.path.join(config.output_path, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def build_launch_artifacts(
    *,
    run_specs: list[DelphiSwarmRunSpec],
    analysis_output_path: str,
    source_panel: str,
    validation_configs: dict[str, DatasetComponent],
) -> LaunchArtifacts:
    """Build the full 280-train plus 280-native-eval graph."""
    training_steps: list[ExecutorStep] = []
    eval_steps: list[ExecutorStep] = []
    for run_spec in run_specs:
        training_resources = ResourceConfig.with_tpu(
            run_spec.tpu_type,
            regions=[run_spec.tpu_region],
            zone=run_spec.tpu_zone,
        )
        training_step = ExecutorStep(
            name=f"{EXPERIMENT_NAME}/{run_spec.run_name}",
            fn=remote(
                run_delphi_swarm_training,
                resources=training_resources,
                env_vars={HF_HUB_DISABLE_XET_ENV_VAR: "1"},
            ),
            resources=training_resources,
            config=DelphiSwarmTrainingConfig(
                analysis_output_path=analysis_output_path,
                output_path=this_output_path(),
                run_spec=run_spec,
                validation_configs=validation_configs,
                wandb_tags=("delphi-3e18-augmented-swarm", "fit-panel", "two-phase"),
            ),
        )
        training_steps.append(training_step)
        eval_steps.append(
            olmo_base_eval_step(
                name=f"t9_{run_spec.run_name}",
                checkpoint=training_step / f"hf/step-{run_spec.expected_checkpoint_step}",
                request_set_dir=TABLE9_REQUEST_SET_DIR,
                resource_config=TABLE9_EVAL_RESOURCES,
                wandb_group="olmo_base_eval_table9_delphi_3e18_augmented_swarm",
                provenance={
                    "evaluator": "marin-native-table9-bpb",
                    "panel": "delphi_3e18_augmented_fit_swarm",
                    "scale": "3e18",
                    "source_run_name": run_spec.source_run_name,
                    "swarm_run_name": run_spec.run_name,
                    "panel_source": run_spec.panel_source,
                },
            )
        )

    manifest_step = ExecutorStep(
        name=f"{EXPERIMENT_NAME}/manifest",
        fn=save_delphi_swarm_manifest,
        config=SaveDelphiSwarmManifestConfig(
            output_path=this_output_path(),
            source_panel=source_panel,
            source_panel_sha256=SOURCE_PANEL_SHA256,
            analysis_output_path=analysis_output_path,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
        ),
    )
    return LaunchArtifacts(
        manifest_step=manifest_step,
        training_steps=training_steps,
        eval_steps=eval_steps,
    )


def _write_local_dry_run(
    *,
    source_panel: str,
    analysis_output_path: str,
    run_specs: list[DelphiSwarmRunSpec],
) -> None:
    LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_delphi_swarm_manifest(
        SaveDelphiSwarmManifestConfig(
            output_path=str(LOCAL_ARTIFACT_DIR),
            source_panel=str(source_panel),
            source_panel_sha256=SOURCE_PANEL_SHA256,
            analysis_output_path=analysis_output_path,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
        )
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", default=DEFAULT_SOURCE_PANEL)
    parser.add_argument("--analysis-output-path", default=DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if args.tpu_region != DEFAULT_TPU_REGION or args.tpu_zone != DEFAULT_TPU_ZONE:
        raise ValueError(f"This launcher is pinned to {DEFAULT_TPU_REGION}/{DEFAULT_TPU_ZONE}")
    if args.max_concurrent < 1 or args.max_concurrent > DEFAULT_MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {DEFAULT_MAX_CONCURRENT}]")
    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required prefix {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    run_specs = load_source_panel(
        source_panel=args.source_panel,
        analysis_output_path=args.analysis_output_path,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    if args.dry_run:
        _write_local_dry_run(
            source_panel=args.source_panel,
            analysis_output_path=args.analysis_output_path,
            run_specs=run_specs,
        )
        logger.info("Wrote %d dry-run run specs under %s", len(run_specs), LOCAL_ARTIFACT_DIR)
        return

    validation_steps = _default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    with executor_context():
        artifacts = build_launch_artifacts(
            run_specs=run_specs,
            analysis_output_path=args.analysis_output_path,
            source_panel=str(args.source_panel),
            validation_configs=validation_configs,
        )
    if os.getenv("CI") is not None:
        logger.info(
            "Built Delphi augmented-swarm graph with %d training and %d Table-9 steps; skipping launch in CI",
            len(artifacts.training_steps),
            len(artifacts.eval_steps),
        )
        return

    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=(
            f"{EXPERIMENT_NAME}: exact 280-row augmented fit panel at Delphi 3e18 " "with native Table-9 evaluation"
        ),
    )


if __name__ == "__main__":
    main()
