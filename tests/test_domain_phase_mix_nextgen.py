# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from dataclasses import asdict

import numpy as np
import pandas as pd
from fray.cluster import ResourceConfig
from levanter.data.text import DatasetComponent as LmDatasetComponent
from levanter.data.text import HierarchicalMixtureDatasetComponent
from marin.datakit.download.huggingface import DownloadConfig
from marin.processing.tokenize import TokenizeConfig
from marin.transform.stack_edu.hydrate import StackEduHydrationConfig

import experiments.domain_phase_mix.nextgen_experiment as nextgen_experiment
from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.general_scaling_models import DatasetSpec
from experiments.domain_phase_mix.nextgen.contracts import (
    Candidate,
    LoopConfig,
    LoopState,
    RunRecord,
    ValidationRecord,
)
from experiments.domain_phase_mix.nextgen.design import plan_new_runs
from experiments.domain_phase_mix.nextgen.fit_propose import (
    CANDIDATE_ASSIGNMENTS_JSON,
    CANDIDATES_JSON,
)
from experiments.domain_phase_mix.nextgen.import_sources import (
    CsvDomainPhaseImportSource,
    THREE_PHASE_EXPERIMENT,
    THREE_PHASE_STARCODER_EXPERIMENT,
    TWO_PHASE_STARCODER_EXPERIMENT,
    NamedWandbRunImportSource,
    default_legacy_sources,
    source_from_dict,
)
from experiments.domain_phase_mix.nextgen.merge_export import (
    MERGED_RUNS_JSON,
    MERGED_TRAJ_PARQUET,
    RESULTS_CSV,
    RUNS_PARQUET,
    TRAJ_CSV,
    ExportDatasetConfig,
    MergeDatasetConfig,
    export_dataset,
    merge_dataset,
)
from experiments.domain_phase_mix.nextgen.collect import (
    CollectNewRunDataConfig,
    IMPORTED_RUNS_FILE,
    IMPORTED_TRAJ_FILE,
    NEW_RUNS_FILE,
    NEW_TRAJ_FILE,
    collect_new_run_data,
    source_to_dict,
)
from experiments.domain_phase_mix.nextgen.model_registry import _propose_top1_candidate
from experiments.domain_phase_mix.nextgen.state_store import write_loop_state
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import REMOVED_DOLMA3_CC_TOPICS
from experiments.pretraining_datasets.dolma3_dolmino_pool import (
    download_dolmino_pool,
    tokenize_dolmino_pool_subset,
)
from experiments.pretraining_datasets.dolma3_pool import (
    download_dolma3_pool,
    hydrate_stack_edu_subset,
    tokenize_dolma3_pool_subset,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    DEFAULT_RUNTIME_CACHE_REGION,
    MERGED_CC_DOMAIN_NAMES,
    MIN_RECOMMENDED_SAMPLED_RUNS,
    MIN_RECOMMENDED_SWARM_RUNS,
    PHASE_BOUNDARIES,
    build_top_level_domains,
    build_top_level_domain_steps,
    create_two_phase_dolma3_dolmino_top_level_experiment,
    resolve_two_phase_wsd_boundary_schedule,
)
from experiments.domain_phase_mix.proxy_sweep import olmo3_30m_proxy, regmix_300m_muonh_base, regmix_300m_proxy
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear import (
    OLMIX_LOGLINEAR_PREDICTED_BPB,
    OLMIX_LOGLINEAR_RUN_NAME,
    create_olmix_loglinear_import_source,
)
from experiments.evals.task_configs import MMLU_5_SHOT, MMLU_PRO_5_SHOT, MMLU_SL_VERB_5_SHOT
from experiments.domain_phase_mix.nextgen.validation import (
    CollectValidationConfig,
    PENDING_CANDIDATES_JSON,
    PlanValidationConfig,
    SLOT_ASSIGNMENTS_JSON,
    VALIDATION_RESULTS_JSON,
    VALIDATION_PLAN_JSON,
    collect_validation_results,
    plan_validation,
    run_validation_slot,
    ValidationSlotConfig,
)


class _FakeSampler:
    def __init__(self, configs):
        self._configs = configs

    def sample_n_configs(self, n, deduplicate=True, existing_configs=None):
        assert n == len(self._configs)
        return self._configs


class _FakeExperiment:
    def __init__(self, configs):
        self._sampler = _FakeSampler(configs)

    def create_weight_sampler(self, seed=42):
        return self._sampler


def _write_json(path: str | os.PathLike[str], payload) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def test_append_runs_planning_uses_next_local_run_id():
    existing_state = LoopState(
        loop_name="loop",
        objective_metric="eval/loss",
        next_local_run_id=8,
        runs=[
            RunRecord(
                wandb_run_id="abc",
                source_experiment="loop",
                local_run_id=7,
                run_name="run_00007",
                phase_weights={"phase_0": {"a": 0.7, "b": 0.3}},
                status="completed",
            )
        ],
    )

    new_configs = [
        WeightConfig(run_id=0, phase_weights={"phase_0": {"a": 0.2, "b": 0.8}}),
        WeightConfig(run_id=1, phase_weights={"phase_0": {"a": 0.1, "b": 0.9}}),
    ]
    experiment = _FakeExperiment(new_configs)

    loop = LoopConfig(name="loop", objective_metric="eval/loss", model_names=("Linear",), n_new_runs=2)
    planned = plan_new_runs(loop, experiment, existing_state)

    assert [item.local_run_id for item in planned] == [8, 9]
    assert [item.run_name for item in planned] == ["run_00008", "run_00009"]


def test_default_legacy_sources_cover_core_experiments(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://unit-test-prefix")
    sources = default_legacy_sources()
    source_names = {source.source_experiment for source in sources}
    assert TWO_PHASE_STARCODER_EXPERIMENT in source_names
    assert THREE_PHASE_EXPERIMENT in source_names
    assert THREE_PHASE_STARCODER_EXPERIMENT in source_names


def test_named_wandb_import_source_roundtrip_and_collect_runs(monkeypatch):
    source = create_olmix_loglinear_import_source(local_run_id=240)
    payload = source_to_dict(source)
    restored = source_from_dict(payload)

    assert isinstance(restored, NamedWandbRunImportSource)
    assert restored.local_run_id == 240
    assert restored.run_name == OLMIX_LOGLINEAR_RUN_NAME

    monkeypatch.setattr(
        "experiments.domain_phase_mix.nextgen.import_sources.query_wandb_runs",
        lambda **_: [
            {
                "wandb_run_id": "abc123",
                "wandb_run_name": "pinlin_calvin_xu/data_mixture/~hash/baseline_olmix_loglinear",
                "status": "finished",
                "lm_eval/mmlu_5shot/bpb": 2.11,
            }
        ],
    )

    records = restored.collect_runs()
    assert len(records) == 1
    assert records[0].wandb_run_id == "abc123"
    assert records[0].status == "completed"
    assert records[0].local_run_id == 240
    assert records[0].run_name == OLMIX_LOGLINEAR_RUN_NAME
    assert records[0].metrics["lm_eval/mmlu_5shot/bpb"] == 2.11


def test_nextgen_experiment_loads_import_source_json(tmp_path):
    source = create_olmix_loglinear_import_source(local_run_id=240)
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source_to_dict(source)), encoding="utf-8")

    restored = nextgen_experiment._load_import_source_json(str(source_path))
    assert isinstance(restored, NamedWandbRunImportSource)
    assert restored.local_run_id == 240
    assert restored.run_name == OLMIX_LOGLINEAR_RUN_NAME


def test_export_csv_contains_phase_columns_and_objective(tmp_path):
    merged_dir = tmp_path / "merged"
    export_dir = tmp_path / "export"
    merged_dir.mkdir()

    runs = [
        RunRecord(
            wandb_run_id="run-a",
            source_experiment="legacy",
            local_run_id=42,
            run_name="run_00042",
            phase_weights={
                "phase_0": {"nemotron_full": 0.9, "starcoder": 0.1},
                "phase_1": {"nemotron_full": 0.5, "starcoder": 0.5},
            },
            status="completed",
            metrics={"eval/paloma/dolma_100_programing_languages/bpb": 0.8123},
        )
    ]

    _write_json(str(merged_dir / MERGED_RUNS_JSON), [asdict(r) for r in runs])

    traj = pd.DataFrame(
        [
            {
                "wandb_run_id": "run-a",
                "source_experiment": "legacy",
                "local_run_id": 42,
                "run_name": "run_00042",
                "step": 1000,
                "total_tokens": 1.0e8,
                "metric_key": "eval/paloma/dolma_100_programing_languages/bpb",
                "metric_value": 0.95,
            },
            {
                "wandb_run_id": "run-a",
                "source_experiment": "legacy",
                "local_run_id": 42,
                "run_name": "run_00042",
                "step": 2000,
                "total_tokens": 2.0e8,
                "metric_key": "eval/paloma/dolma_100_programing_languages/bpb",
                "metric_value": 0.81,
            },
        ]
    )
    traj.to_parquet(merged_dir / MERGED_TRAJ_PARQUET, index=False)

    export_dataset(
        ExportDatasetConfig(
            output_path=str(export_dir),
            merged_output_path=str(merged_dir),
        )
    )

    result_df = pd.read_csv(export_dir / RESULTS_CSV)
    assert "phase_0_nemotron_full" in result_df.columns
    assert "phase_1_starcoder" in result_df.columns
    assert "eval/paloma/dolma_100_programing_languages/bpb" in result_df.columns

    traj_df = pd.read_csv(export_dir / TRAJ_CSV)
    assert len(traj_df) == 2

    # Parquet exports must exist for downstream model fitting.
    assert (export_dir / RUNS_PARQUET).exists()


def test_collect_new_backfills_objective_from_trajectory(tmp_path, monkeypatch):
    output_dir = tmp_path / "collect_new"
    output_dir.mkdir()

    monkeypatch.setattr(
        "experiments.domain_phase_mix.nextgen.collect.query_wandb_runs",
        lambda **_: [
            {
                "wandb_run_id": "run-a",
                "wandb_run_name": "loop/baseline_proportional",
                "status": "finished",
                "eval/loss": 1.23,
            }
        ],
    )
    monkeypatch.setattr(
        "experiments.domain_phase_mix.nextgen.collect._scan_trajectory",
        lambda **_: pd.DataFrame(
            [
                {
                    "step": 4576,
                    "total_tokens": 1.199833088e9,
                    "metric_key": "lm_eval/mmlu_5shot/bpb",
                    "metric_value": 0.42,
                }
            ]
        ),
    )

    collect_new_run_data(
        CollectNewRunDataConfig(
            output_path=str(output_dir),
            loop_name="loop",
            objective_metric="lm_eval/mmlu_5shot/bpb",
            wandb_entity="marin-community",
            wandb_project="marin",
            planned_runs_json=json.dumps(
                [
                    {
                        "local_run_id": 0,
                        "run_name": "baseline_proportional",
                        "phase_weights": {"phase_0": {"a": 1.0}},
                    }
                ]
            ),
        )
    )

    new_runs = json.load(open(output_dir / NEW_RUNS_FILE))
    assert new_runs[0]["status"] == "completed"
    assert new_runs[0]["metrics"]["eval/loss"] == 1.23
    assert new_runs[0]["metrics"]["lm_eval/mmlu_5shot/bpb"] == 0.42

    traj_df = pd.read_parquet(output_dir / NEW_TRAJ_FILE)
    assert len(traj_df) == 1
    assert traj_df.iloc[0]["run_name"] == "baseline_proportional"


def test_top_level_experiment_uses_hierarchical_runtime_domains():
    domains = build_top_level_domains()
    domain_names = {domain.name for domain in domains}

    assert len(domains) == 39
    assert sum(len(domain.components) for domain in domains) > 39
    assert "dolma3_cc/art_and_design_high" in domain_names
    assert "dolma3_cc/art_and_design_low" in domain_names
    assert "dolma3_cc/science_math_and_technology_high" in domain_names
    assert "dolma3_cc/science_math_and_technology_low" in domain_names
    assert all(
        not any(name.startswith(f"dolma3_cc/{topic}") for name in domain_names) for topic in REMOVED_DOLMA3_CC_TOPICS
    )

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment()
    baseline = experiment.initial_fixed_weight_configs[0].weight_config
    mixture = experiment.create_mixture_config(baseline)

    assert len(mixture.components) == 39
    assert set(mixture.components) == domain_names
    assert isinstance(mixture.components["dolma3_arxiv"], LmDatasetComponent)
    assert isinstance(mixture.components["dolma3_finemath_3plus"], LmDatasetComponent)
    assert isinstance(mixture.components["dolma3_wikipedia"], LmDatasetComponent)
    assert isinstance(mixture.components["dolmino_synth_code"], LmDatasetComponent)
    assert isinstance(mixture.components["dolma3_cc/art_and_design_high"], LmDatasetComponent)
    assert isinstance(mixture.components["dolma3_stack_edu"], LmDatasetComponent)
    assert isinstance(mixture.components["dolmino_stem_heavy_crawl"], LmDatasetComponent)
    assert isinstance(mixture.components["dolmino_common_crawl_hq"], HierarchicalMixtureDatasetComponent)
    assert isinstance(mixture.components["dolmino_stack_edu_fim"], HierarchicalMixtureDatasetComponent)


def test_build_top_level_domain_steps_only_materializes_cc_split_domains_in_east5():
    steps = build_top_level_domain_steps(runtime_cache_region=DEFAULT_RUNTIME_CACHE_REGION)
    assert set(steps) == set(MERGED_CC_DOMAIN_NAMES)
    assert len(steps) == 26
    assert "dolma3_stack_edu" not in steps
    assert "dolmino_common_crawl_hq" not in steps


def test_build_top_level_domain_steps_uses_prebuilt_stackedu_and_stemheavy_for_central1():
    steps = build_top_level_domain_steps(runtime_cache_region="us-central1")
    assert set(steps) == set(MERGED_CC_DOMAIN_NAMES)
    assert len(steps) == 26
    assert "dolma3_stack_edu" not in steps
    assert "dolmino_stem_heavy_crawl" not in steps


def test_build_top_level_domain_steps_region_agnostic_uses_prebuilt_mirror_caches():
    steps = build_top_level_domain_steps(runtime_cache_region=("us-east5", "us-central1"))
    assert set(steps) == set(MERGED_CC_DOMAIN_NAMES)
    assert len(steps) == 26
    assert "dolma3_stack_edu" not in steps
    assert "dolmino_stem_heavy_crawl" not in steps


def test_top_level_experiment_uses_linear_80_20_wsd():
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment()
    schedule = resolve_two_phase_wsd_boundary_schedule(phase_schedule=experiment.phase_schedule)

    assert PHASE_BOUNDARIES == [0.8]
    assert MIN_RECOMMENDED_SWARM_RUNS == 234
    assert MIN_RECOMMENDED_SAMPLED_RUNS == 231
    assert schedule.total_steps == 4577
    assert schedule.boundary_step == 3648
    assert schedule.warmup_steps == 45
    assert schedule.decay_steps == 929
    assert experiment.optimizer_config.warmup == 45
    assert experiment.optimizer_config.decay == 929
    assert experiment.optimizer_config.lr_schedule == "linear"
    assert len(experiment.initial_fixed_weight_configs) == 3
    assert experiment.initial_fixed_weight_configs[2].run_name == OLMIX_LOGLINEAR_RUN_NAME
    assert OLMIX_LOGLINEAR_PREDICTED_BPB < 2.2


def test_top_level_experiment_includes_sl_verb_mmlu_in_default_eval_tasks():
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment()

    assert experiment.eval_harness_tasks == (MMLU_5_SHOT, MMLU_SL_VERB_5_SHOT, MMLU_PRO_5_SHOT)


def test_top_level_experiment_region_agnostic_uses_mirror_backed_prebuilt_domains():
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        resources=ResourceConfig.with_tpu("v5p-32", regions=["us-east5", "us-central1"], zone=None)
    )
    stack_edu_domain = next(domain for domain in experiment.domains if domain.name == "dolma3_stack_edu")
    stack_edu_cache = stack_edu_domain.components[0].get_step()

    assert "mirror://" in stack_edu_domain.description
    assert stack_edu_cache.cache_path == "mirror://tokenized/merged/dolma3_dolmino_top_level/dolma3_stack_edu-a7297b"


def test_top_level_experiment_region_agnostic_pins_data_prep_workers_to_tpu_regions():
    expected_regions = ["us-east5", "us-central1"]
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        resources=ResourceConfig.with_tpu("v5p-32", regions=expected_regions, zone=None)
    )

    tokenized_steps = []
    for domain in experiment.domains:
        for component in domain.components:
            step = component.get_step()
            if (
                hasattr(step, "config")
                and isinstance(step.config, TokenizeConfig)
                and step.name.startswith(("tokenized/dolma3_pool/", "tokenized/dolma3_dolmino_pool/"))
            ):
                tokenized_steps.append(step)

    assert tokenized_steps
    assert all(list(step.config.worker_resources.regions or []) == expected_regions for step in tokenized_steps)


def test_dataset_data_prep_steps_accept_worker_region_pinning():
    worker_resources = ResourceConfig(regions=["us-east5", "us-central1"])

    dolma3_download = download_dolma3_pool(worker_resources=worker_resources)
    assert isinstance(dolma3_download.config, DownloadConfig)
    assert list(dolma3_download.config.worker_resources.regions or []) == ["us-east5", "us-central1"]

    dolmino_download = download_dolmino_pool(worker_resources=worker_resources)
    assert isinstance(dolmino_download.config, DownloadConfig)
    assert list(dolmino_download.config.worker_resources.regions or []) == ["us-east5", "us-central1"]

    hydrate_step = hydrate_stack_edu_subset("stack_edu/Python", worker_resources=worker_resources)
    assert isinstance(hydrate_step.config, StackEduHydrationConfig)
    assert list(hydrate_step.config.worker_resources.regions or []) == ["us-east5", "us-central1"]

    dolma3_tokenize = tokenize_dolma3_pool_subset(
        "common_crawl/adult_content/0007",
        worker_resources=worker_resources,
    )
    assert isinstance(dolma3_tokenize.config, TokenizeConfig)
    assert list(dolma3_tokenize.config.worker_resources.regions or []) == ["us-east5", "us-central1"]

    dolmino_tokenize = tokenize_dolmino_pool_subset(
        "common_crawl_hq/19_adult_content",
        worker_resources=worker_resources,
    )
    assert isinstance(dolmino_tokenize.config, TokenizeConfig)
    assert list(dolmino_tokenize.config.worker_resources.regions or []) == ["us-east5", "us-central1"]


def test_top_level_experiment_allows_budget_and_model_override():
    experiment_budget = 3_000_000_000
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name="unit-test-30m-3b",
        experiment_budget=experiment_budget,
        model_config=olmo3_30m_proxy,
    )
    schedule = resolve_two_phase_wsd_boundary_schedule(
        experiment_budget=experiment_budget,
        phase_schedule=experiment.phase_schedule,
    )

    assert experiment.model_config == olmo3_30m_proxy
    assert experiment.num_train_steps == 11_444
    assert experiment.experiment_budget == 11_444 * 128 * 2048
    assert experiment.target_budget is not None
    assert experiment.phase_schedule.phases[1].start_fraction == 0.8
    assert experiment.optimizer_config.lr_schedule == "linear"
    assert experiment.optimizer_config.warmup == schedule.warmup_steps
    assert experiment.optimizer_config.decay == schedule.decay_steps


def test_top_level_experiment_allows_optimizer_override():
    experiment_budget = 6_000_000_000
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name="unit-test-300m-6b",
        experiment_budget=experiment_budget,
        model_config=regmix_300m_proxy,
        optimizer_config=regmix_300m_muonh_base,
    )
    schedule = resolve_two_phase_wsd_boundary_schedule(
        experiment_budget=experiment_budget,
        phase_schedule=experiment.phase_schedule,
    )

    assert experiment.model_config == regmix_300m_proxy
    assert experiment.optimizer_config.learning_rate == regmix_300m_muonh_base.learning_rate
    assert experiment.optimizer_config.adam_lr == regmix_300m_muonh_base.adam_lr
    assert experiment.optimizer_config.momentum == regmix_300m_muonh_base.momentum
    assert experiment.optimizer_config.lr_schedule == "linear"
    assert experiment.optimizer_config.warmup == schedule.warmup_steps
    assert experiment.optimizer_config.decay == schedule.decay_steps


def test_top_level_baseline_steps_keep_distinct_checkpoint_names_under_wandb_truncation():
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment()
    name_prefix = "pinlin_calvin_xu/data_mixture/ngd3dm2_hier_canary_fix_with_extra_length"

    proportional_step = experiment.create_training_step(
        experiment.initial_fixed_weight_configs[0].weight_config,
        name_prefix=name_prefix,
        run_name="baseline_proportional",
    )
    unimax_step = experiment.create_training_step(
        experiment.initial_fixed_weight_configs[1].weight_config,
        name_prefix=name_prefix,
        run_name="baseline_unimax",
    )
    olmix_step = experiment.create_training_step(
        experiment.initial_fixed_weight_configs[2].weight_config,
        name_prefix=name_prefix,
        run_name=OLMIX_LOGLINEAR_RUN_NAME,
    )

    assert proportional_step.name.endswith("/baseline_proportional")
    assert unimax_step.name.endswith("/baseline_unimax")
    assert olmix_step.name.endswith(f"/{OLMIX_LOGLINEAR_RUN_NAME}")
    assert proportional_step.name != unimax_step.name
    assert proportional_step.name != olmix_step.name
    assert unimax_step.name != olmix_step.name

    proportional_wandb_name = proportional_step.config.train_config.trainer.tracker.name
    unimax_wandb_name = unimax_step.config.train_config.trainer.tracker.name
    olmix_wandb_name = olmix_step.config.train_config.trainer.tracker.name
    proportional_wandb_tags = proportional_step.config.train_config.trainer.tracker.tags
    unimax_wandb_tags = unimax_step.config.train_config.trainer.tracker.tags
    olmix_wandb_tags = olmix_step.config.train_config.trainer.tracker.tags
    assert proportional_wandb_name is not None
    assert unimax_wandb_name is not None
    assert olmix_wandb_name is not None
    assert len(proportional_wandb_name) <= 64
    assert len(unimax_wandb_name) <= 64
    assert len(olmix_wandb_name) <= 64
    assert proportional_wandb_name != unimax_wandb_name
    assert proportional_wandb_name != olmix_wandb_name
    assert unimax_wandb_name != olmix_wandb_name
    assert proportional_wandb_name.endswith("/baseline_proportional")
    assert unimax_wandb_name.endswith("/baseline_unimax")
    assert olmix_wandb_name.endswith(f"/{OLMIX_LOGLINEAR_RUN_NAME}")
    assert all(len(tag) <= 64 for tag in proportional_wandb_tags)
    assert all(len(tag) <= 64 for tag in unimax_wandb_tags)
    assert all(len(tag) <= 64 for tag in olmix_wandb_tags)


def test_model_resubmit_validation_plan_reuses_prior_candidate(tmp_path):
    state_dir = tmp_path / "state"
    fit_dir = tmp_path / "fit"
    output_dir = tmp_path / "plan"
    state_dir.mkdir()
    fit_dir.mkdir()

    state = LoopState(
        loop_name="loop",
        objective_metric="eval/loss",
        validated_candidates={
            "cand-1": ValidationRecord(
                candidate_id="cand-1",
                model_name="Linear",
                status="planned",
                metric_value=0.81,
            )
        },
    )
    write_loop_state(os.path.join(state_dir, "loop_state.json"), state)

    _write_json(
        os.path.join(fit_dir, CANDIDATES_JSON),
        [
            asdict(
                Candidate(
                    candidate_id="cand-1",
                    model_name="Linear",
                    kind="schedule",
                    phase_weights={"phase_0": {"a": 0.8, "b": 0.2}},
                    policy_ref=None,
                    predicted_objective=0.79,
                )
            )
        ],
    )
    _write_json(os.path.join(fit_dir, CANDIDATE_ASSIGNMENTS_JSON), {"Linear": "cand-1"})

    plan_validation(
        PlanValidationConfig(
            output_path=str(output_dir),
            fit_output_path=str(fit_dir),
            state_output_path=str(state_dir),
            model_names_json=json.dumps(["Linear"]),
        )
    )

    pending = json.load(open(output_dir / PENDING_CANDIDATES_JSON))
    assert pending == []

    slot_assignments = json.load(open(output_dir / SLOT_ASSIGNMENTS_JSON))
    assert slot_assignments["Linear"] is None

    validation_plan = json.load(open(output_dir / VALIDATION_PLAN_JSON))
    assert validation_plan[0]["status"] == "reused"


def test_legacy_merge_dedup_by_wandb_preserves_local_run_id(tmp_path):
    state_dir = tmp_path / "state"
    import_dir = tmp_path / "imported"
    new_dir = tmp_path / "new"
    merge_dir = tmp_path / "merge"
    state_dir.mkdir()
    import_dir.mkdir()
    new_dir.mkdir()
    merge_dir.mkdir()

    state = LoopState(
        loop_name="loop",
        objective_metric="eval/loss",
        runs=[
            RunRecord(
                wandb_run_id="run-1",
                source_experiment="legacy",
                local_run_id=11,
                run_name="run_00011",
                phase_weights={"phase_0": {"a": 0.6, "b": 0.4}},
                status="completed",
                metrics={"eval/loss": 1.2},
            )
        ],
    )
    write_loop_state(os.path.join(state_dir, "loop_state.json"), state)

    imported_runs = [
        asdict(
            RunRecord(
                wandb_run_id="run-1",
                source_experiment="legacy",
                local_run_id=11,
                run_name="run_00011",
                phase_weights={"phase_0": {"a": 0.7, "b": 0.3}},
                status="completed",
                metrics={"eval/loss": 0.9},
            )
        )
    ]
    _write_json(import_dir / IMPORTED_RUNS_FILE, imported_runs)

    new_runs = [
        asdict(
            RunRecord(
                wandb_run_id="run-1",
                source_experiment="loop",
                local_run_id=None,
                run_name=None,
                phase_weights={},
                status="completed",
                metrics={"eval/loss": 0.8, "eval/aux": 0.3},
            )
        )
    ]
    _write_json(new_dir / NEW_RUNS_FILE, new_runs)

    pd.DataFrame(
        [
            {
                "wandb_run_id": "run-1",
                "source_experiment": "legacy",
                "local_run_id": 11,
                "run_name": "run_00011",
                "step": 1000,
                "total_tokens": 1.0e8,
                "metric_key": "eval/loss",
                "metric_value": 0.9,
            }
        ]
    ).to_parquet(import_dir / IMPORTED_TRAJ_FILE, index=False)
    pd.DataFrame(
        [
            {
                "wandb_run_id": "run-1",
                "source_experiment": "loop",
                "local_run_id": None,
                "run_name": None,
                "step": 1000,
                "total_tokens": 1.0e8,
                "metric_key": "eval/loss",
                "metric_value": 0.8,
            }
        ]
    ).to_parquet(new_dir / NEW_TRAJ_FILE, index=False)

    merge_dataset(
        MergeDatasetConfig(
            output_path=str(merge_dir),
            loop_name="loop",
            objective_metric="eval/loss",
            state_output_path=str(state_dir),
            imported_output_path=str(import_dir),
            new_output_path=str(new_dir),
        )
    )

    merged_runs = json.load(open(merge_dir / MERGED_RUNS_JSON))
    assert len(merged_runs) == 1
    merged = merged_runs[0]
    assert merged["wandb_run_id"] == "run-1"
    assert merged["local_run_id"] == 11
    assert merged["metrics"]["eval/loss"] == 0.8
    assert merged["metrics"]["eval/aux"] == 0.3


def test_policy_and_schedule_candidates_share_validation_flow(tmp_path):
    state_dir = tmp_path / "state"
    fit_dir = tmp_path / "fit"
    plan_dir = tmp_path / "plan"
    slot_schedule_dir = tmp_path / "slot_schedule"
    slot_policy_dir = tmp_path / "slot_policy"
    collect_dir = tmp_path / "collect"
    state_dir.mkdir()
    fit_dir.mkdir()

    write_loop_state(
        os.path.join(state_dir, "loop_state.json"),
        LoopState(loop_name="loop", objective_metric="eval/loss"),
    )

    _write_json(
        fit_dir / CANDIDATES_JSON,
        [
            asdict(
                Candidate(
                    candidate_id="cand-schedule",
                    model_name="Linear",
                    kind="schedule",
                    phase_weights={"phase_0": {"a": 0.9, "b": 0.1}},
                    policy_ref=None,
                    predicted_objective=0.8,
                )
            ),
            {
                "candidate_id": "cand-policy",
                "model_name": "OfflineRL",
                "kind": "policy",
                "phase_weights": None,
                "policy_ref": {"uri": "gs://policy/artifact.json", "format": "json"},
                "predicted_objective": 0.79,
            },
        ],
    )
    _write_json(
        fit_dir / CANDIDATE_ASSIGNMENTS_JSON,
        {"Linear": "cand-schedule", "OfflineRL": "cand-policy"},
    )

    plan_validation(
        PlanValidationConfig(
            output_path=str(plan_dir),
            fit_output_path=str(fit_dir),
            state_output_path=str(state_dir),
            model_names_json=json.dumps(["Linear", "OfflineRL"]),
        )
    )

    pending = json.load(open(plan_dir / PENDING_CANDIDATES_JSON))
    assert {row["candidate_id"] for row in pending} == {"cand-schedule", "cand-policy"}

    run_validation_slot(
        ValidationSlotConfig(
            output_path=str(slot_schedule_dir),
            plan_output_path=str(plan_dir),
            fit_output_path=str(fit_dir),
            model_name="Linear",
            execute_slot=False,
        )
    )
    run_validation_slot(
        ValidationSlotConfig(
            output_path=str(slot_policy_dir),
            plan_output_path=str(plan_dir),
            fit_output_path=str(fit_dir),
            model_name="OfflineRL",
            execute_slot=False,
        )
    )

    collect_validation_results(
        CollectValidationConfig(
            output_path=str(collect_dir),
            slot_output_paths=(str(slot_schedule_dir), str(slot_policy_dir)),
            slot_model_names_json=json.dumps(["Linear", "OfflineRL"]),
        )
    )

    results = json.load(open(collect_dir / VALIDATION_RESULTS_JSON))
    assert len(results) == 2
    by_model = {row["model_name"]: row for row in results}
    assert by_model["Linear"]["candidate_id"] == "cand-schedule"
    assert by_model["OfflineRL"]["candidate_id"] == "cand-policy"
    assert by_model["Linear"]["status"] == "planned"
    assert by_model["OfflineRL"]["status"] == "planned"


def test_csv_import_source_parses_phase_weights_and_metrics(tmp_path):
    csv_path = tmp_path / "runs.csv"
    df = pd.DataFrame(
        [
            {
                "run_id": 1,
                "wandb_run_id": "run-a",
                "run_name": "run_00001",
                "status": "completed",
                "phase_0_nemotron_full": 0.8,
                "phase_0_starcoder": 0.2,
                "phase_1_nemotron_full": 0.6,
                "phase_1_starcoder": 0.4,
                "eval/paloma/dolma_100_programing_languages/bpb": 0.81,
                "lm_eval/foo/acc": 0.5,
            },
            {
                "run_id": 2,
                "wandb_run_id": "run-b",
                "run_name": "run_00002",
                "status": "failed",
                "phase_0_nemotron_full": 0.7,
                "phase_0_starcoder": 0.3,
                "phase_1_nemotron_full": 0.5,
                "phase_1_starcoder": 0.5,
                "eval/paloma/dolma_100_programing_languages/bpb": 0.9,
            },
        ]
    )
    df.to_csv(csv_path, index=False)

    source = CsvDomainPhaseImportSource(source_experiment="loop", csv_path=str(csv_path))
    runs = source.collect_runs()

    assert len(runs) == 1
    run = runs[0]
    assert run.local_run_id == 1
    assert run.wandb_run_id == "run-a"
    assert run.phase_weights["phase_0"]["starcoder"] == 0.2
    assert run.phase_weights["phase_1"]["nemotron_full"] == 0.6
    assert run.metrics["eval/paloma/dolma_100_programing_languages/bpb"] == 0.81
    assert run.metrics["lm_eval/foo/acc"] == 0.5

    trajectories = source.collect_trajectories("eval/paloma/dolma_100_programing_languages/bpb")
    assert trajectories.empty
    assert "metric_value" in trajectories.columns


def test_csv_import_source_serialization_roundtrip(tmp_path):
    csv_path = tmp_path / "runs.csv"
    pd.DataFrame([{"run_id": 1, "status": "completed"}]).to_csv(csv_path, index=False)

    source = CsvDomainPhaseImportSource(
        source_experiment="loop",
        csv_path=str(csv_path),
        status_filter="completed",
    )
    payload = source_to_dict(source)
    restored = source_from_dict(payload)

    assert isinstance(restored, CsvDomainPhaseImportSource)
    assert restored.source_experiment == "loop"
    assert restored.csv_path == str(csv_path)
    assert restored.status_filter == "completed"


def test_csv_import_fixture_has_expected_three_phase_row_count():
    csv_path = "experiments/domain_phase_mix/exploratory/three_phase_starcoder.csv"
    source = CsvDomainPhaseImportSource(
        source_experiment="three_phase_starcoder",
        csv_path=csv_path,
        status_filter="completed",
    )
    runs = source.collect_runs()
    assert len(runs) == 160


def test_scipy_minimize_candidate_beats_sampling_on_convex_objective():
    rng = np.random.default_rng(0)
    samples = rng.uniform(0.0, 1.0, size=(32, 3))
    weights = np.zeros((32, 3, 2), dtype=float)
    weights[:, :, 1] = samples
    weights[:, :, 0] = 1.0 - samples
    spec = DatasetSpec(
        weights=weights,
        y=np.zeros(32, dtype=float),
        epoch_multipliers=np.ones((3, 2), dtype=float),
        domain_names=["nemotron_full", "starcoder"],
        phase_names=["phase_0", "phase_1", "phase_2"],
        small_domains=[1],
        name="synthetic",
    )

    target = np.array([0.23, 0.41, 0.17], dtype=float)

    def predict_fn(points: np.ndarray) -> np.ndarray:
        return np.sum((points[:, :, 1] - target[None, :]) ** 4, axis=1)

    sample_loop = LoopConfig(
        name="loop",
        objective_metric="eval/loss",
        model_names=("DS-RE-CEQ",),
        candidate_search_points=512,
        candidate_search_seed=7,
        candidate_opt_method="sample",
    )
    scipy_loop = LoopConfig(
        name="loop",
        objective_metric="eval/loss",
        model_names=("DS-RE-CEQ",),
        candidate_search_points=512,
        candidate_search_seed=7,
        candidate_opt_method="scipy_minimize",
        candidate_opt_restarts=16,
        candidate_opt_maxiter=200,
    )

    sample_candidate = _propose_top1_candidate(
        loop=sample_loop,
        model_name="DS-RE-CEQ",
        predict_fn=predict_fn,
        spec=spec,
        training_setup={},
    )
    scipy_candidate = _propose_top1_candidate(
        loop=scipy_loop,
        model_name="DS-RE-CEQ",
        predict_fn=predict_fn,
        spec=spec,
        training_setup={},
    )

    assert scipy_candidate.predicted_objective <= sample_candidate.predicted_objective
    assert sample_candidate.predicted_objective - scipy_candidate.predicted_objective > 1e-6


def test_nextgen_experiment_selects_three_phase(monkeypatch):
    calls = {"three_phase": 0, "two_phase": 0}

    def _fake_two_phase(*, name):
        calls["two_phase"] += 1
        return ("two", name)

    def _fake_three_phase(*, name):
        calls["three_phase"] += 1
        return ("three", name)

    monkeypatch.setattr(nextgen_experiment, "create_two_phase_experiment", _fake_two_phase)
    monkeypatch.setattr(nextgen_experiment, "create_three_phase_experiment", _fake_three_phase)

    args = nextgen_experiment.argparse.Namespace(experiment="three_phase_starcoder", name="loop")
    result = nextgen_experiment._build_experiment(args)

    assert calls["three_phase"] == 1
    assert calls["two_phase"] == 0
    assert result == ("three", "loop")
