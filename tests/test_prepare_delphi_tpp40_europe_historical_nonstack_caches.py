# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from fray.cluster import ResourceConfig
from marin.execution.context import executor_context
from marin.execution.executor import collect_dependencies_and_version, compute_output_path, instantiate_config
from marin.execution.types import versioned
from marin.processing.tokenize import HistoricalFullDocumentTokenizeConfig

from experiments.domain_phase_mix.delphi_tpp40_europe_runtime_caches import (
    EUROPE_HISTORICAL_FLAN_EAST5_SUBSET_PATH,
    EUROPE_HISTORICAL_NONSTACK_REPAIR_PATHS,
    EUROPE_RUNTIME_CACHE_PREFIX,
)
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import TOP_LEVEL_DOMAIN_PARTITIONS
from experiments.domain_phase_mix.prepare_delphi_tpp40_europe_historical_nonstack_caches import (
    CANARY_PARTITION,
    DEFAULT_WORKER_RAM,
    EUROPE_REGION,
    FINEMATH_DATASET_ID,
    FINEMATH_PARTITION,
    FINEMATH_RAW_OUTPUT_PATH,
    FINEMATH_REVISION,
    FINEMATH_WORKER_RAM,
    FLAN_EXCLUDED_SOURCE_SHARD_INDICES,
    FLAN_OUTPUT_NAME,
    FLAN_PARTITION,
    FLAN_RAW_PREFIX,
    FLAN_SOURCE_SHARD_COUNT,
    HISTORICAL_REPAIR_COMPONENTS,
    HISTORICAL_SINGLETON_PARTITIONS,
    PARTITION_RAM_OVERRIDES,
    STEM_DOMAIN,
    STEM_HISTORICAL_PREPROCESSOR_METADATA,
    STRESS_PARTITION,
    FrozenSourceHistoricalFullDocumentTokenizeConfig,
    finemath_raw_step,
    historical_dolmino_partition_step,
    historical_flan_step,
    historical_nonstack_steps,
    historical_stem_cache_step,
)
from experiments.llama import llama3_tokenizer


def test_historical_nonstack_canary_reuses_full_stem_graph_step() -> None:
    with executor_context():
        [canary_step] = historical_nonstack_steps("canary")
        stem_step = historical_stem_cache_step()

    assert canary_step is historical_dolmino_partition_step(CANARY_PARTITION)
    assert canary_step is stem_step.config.input_configs[CANARY_PARTITION].cache_dir.step
    assert isinstance(canary_step.config, HistoricalFullDocumentTokenizeConfig)
    assert not canary_step.config.split_long_documents
    assert canary_step.config.worker_resources == ResourceConfig(ram=DEFAULT_WORKER_RAM, regions=[EUROPE_REGION])


def test_historical_nonstack_stress_scope_reuses_singleton_step() -> None:
    with executor_context():
        [stress_step] = historical_nonstack_steps("stress")

    assert stress_step is historical_dolmino_partition_step(STRESS_PARTITION)
    assert STRESS_PARTITION in HISTORICAL_SINGLETON_PARTITIONS
    assert isinstance(stress_step.config, HistoricalFullDocumentTokenizeConfig)
    assert not stress_step.config.split_long_documents
    assert stress_step.config.worker_resources == ResourceConfig(ram=DEFAULT_WORKER_RAM, regions=[EUROPE_REGION])


def test_historical_stem_scope_freezes_merge_metadata_and_identity() -> None:
    with executor_context():
        [stem_step] = historical_nonstack_steps("stem")
        independently_built_stem_step = historical_stem_cache_step()

    assert stem_step.config == independently_built_stem_step.config
    assert stem_step.config.preprocessor_metadata.value == STEM_HISTORICAL_PREPROCESSOR_METADATA

    output_path = compute_output_path(
        stem_step.name,
        stem_step.config,
        override_output_path=stem_step.override_output_path,
        prefix=EUROPE_RUNTIME_CACHE_PREFIX,
    )
    assert output_path == EUROPE_HISTORICAL_NONSTACK_REPAIR_PATHS[STEM_DOMAIN]

    dependencies = collect_dependencies_and_version(stem_step.config).dependencies
    materialized = instantiate_config(
        stem_step.config,
        output_path=output_path,
        output_paths={
            dependency: f"{EUROPE_RUNTIME_CACHE_PREFIX}/materialized-input-{index}"
            for index, dependency in enumerate(dependencies)
        },
        prefix=EUROPE_RUNTIME_CACHE_PREFIX,
    )
    assert materialized.preprocessor_metadata == STEM_HISTORICAL_PREPROCESSOR_METADATA

    changed_metadata = {**STEM_HISTORICAL_PREPROCESSOR_METADATA, "max_length": None}
    changed_config = replace(stem_step.config, preprocessor_metadata=versioned(changed_metadata))
    changed_output_path = compute_output_path(
        stem_step.name,
        changed_config,
        override_output_path=stem_step.override_output_path,
        prefix=EUROPE_RUNTIME_CACHE_PREFIX,
    )

    assert changed_output_path != output_path


def test_historical_flan_scope_freezes_the_east5_source_subset() -> None:
    with executor_context():
        [flan_step] = historical_nonstack_steps("flan")

    assert flan_step.name == FLAN_OUTPUT_NAME
    assert isinstance(flan_step.config, FrozenSourceHistoricalFullDocumentTokenizeConfig)
    assert not flan_step.config.split_long_documents
    assert flan_step.config.worker_resources == ResourceConfig(ram=DEFAULT_WORKER_RAM, regions=[EUROPE_REGION])
    train_paths = flan_step.config.train_paths
    assert len(train_paths) == FLAN_SOURCE_SHARD_COUNT - len(FLAN_EXCLUDED_SOURCE_SHARD_INDICES)
    assert train_paths == sorted(train_paths)
    assert all(path.startswith(FLAN_RAW_PREFIX + "/") for path in train_paths)
    for index in FLAN_EXCLUDED_SOURCE_SHARD_INDICES:
        assert f"{FLAN_RAW_PREFIX}/tulu_flan-{index:04d}.jsonl.zst" not in train_paths
    assert flan_step.config.source_paths_identity.value == tuple(train_paths)
    assert historical_dolmino_partition_step(FLAN_PARTITION).config == historical_flan_step().config


def test_historical_flan_output_path_binds_the_versioned_source_subset() -> None:
    with executor_context():
        flan_step = historical_flan_step()

    output_path = compute_output_path(
        flan_step.name,
        flan_step.config,
        override_output_path=flan_step.override_output_path,
        prefix=EUROPE_RUNTIME_CACHE_PREFIX,
    )

    assert output_path == EUROPE_HISTORICAL_FLAN_EAST5_SUBSET_PATH

    changed_paths = [*flan_step.config.train_paths, f"{FLAN_RAW_PREFIX}/tulu_flan-0064.jsonl.zst"]
    changed_paths.sort()
    changed_config = replace(
        flan_step.config,
        train_paths=changed_paths,
        source_paths_identity=versioned(tuple(changed_paths)),
    )
    changed_output_path = compute_output_path(
        flan_step.name,
        changed_config,
        override_output_path=flan_step.override_output_path,
        prefix=EUROPE_RUNTIME_CACHE_PREFIX,
    )

    assert changed_output_path != output_path


def test_historical_nonstack_full_graph_is_complete_and_region_local() -> None:
    with executor_context():
        steps = historical_nonstack_steps("all")
        raw_step = finemath_raw_step()
        singleton_steps = {
            partition_name: historical_dolmino_partition_step(partition_name)
            for partition_name in HISTORICAL_SINGLETON_PARTITIONS
        }

    assert HISTORICAL_REPAIR_COMPONENTS == {
        "finemath_3plus",
        "dolmino_stem_heavy_crawl",
        "synth_instruction/dolmino_flan",
        "synth_math/dolmino_math",
        "synth_qa/wiki_to_rcqa",
        "synth_thinking/code_meta_reasoning",
        "synth_thinking/math_meta_reasoning",
        "synth_thinking/program_verifiable",
    }
    assert len(steps) == len(HISTORICAL_REPAIR_COMPONENTS)
    finemath_step = next(step for step in steps if step.name.endswith("finemath_3_plus_historical_full_document_v1"))
    stem_step = next(step for step in steps if step.name.endswith("dolmino_stem_heavy_crawl"))

    assert isinstance(finemath_step.config, HistoricalFullDocumentTokenizeConfig)
    [finemath_input] = finemath_step.config.train_paths
    assert finemath_input.name == FINEMATH_PARTITION
    assert finemath_input.step.name == raw_step.name
    assert finemath_input.step.override_output_path == FINEMATH_RAW_OUTPUT_PATH
    assert finemath_input.step.config.hf_dataset_id == FINEMATH_DATASET_ID
    assert finemath_input.step.config.revision == FINEMATH_REVISION
    assert finemath_step.config.tokenizer.value == llama3_tokenizer
    assert finemath_step.config.worker_resources == ResourceConfig(
        ram=FINEMATH_WORKER_RAM,
        regions=[EUROPE_REGION],
    )
    assert FINEMATH_WORKER_RAM == "14g"
    assert PARTITION_RAM_OVERRIDES == {"synth_math/dolmino_math": "14g"}

    assert set(stem_step.config.input_configs) == set(TOP_LEVEL_DOMAIN_PARTITIONS[STEM_DOMAIN])
    assert stem_step.config.preprocessor_metadata.value == STEM_HISTORICAL_PREPROCESSOR_METADATA
    for source in stem_step.config.input_configs.values():
        tokenize_step = source.cache_dir.step
        assert isinstance(tokenize_step.config, HistoricalFullDocumentTokenizeConfig)
        assert tokenize_step.config.worker_resources.regions == [EUROPE_REGION]
        assert "dolma3_dolmino_pool_historical_full_document_v1" in tokenize_step.name

    expected_singleton_names = {step.name for step in singleton_steps.values()}
    actual_singleton_names = {step.name for step in steps if step not in {finemath_step, stem_step}}
    assert actual_singleton_names == expected_singleton_names
    for partition_name, step in singleton_steps.items():
        expected_ram = PARTITION_RAM_OVERRIDES.get(partition_name, DEFAULT_WORKER_RAM)
        assert isinstance(step.config, HistoricalFullDocumentTokenizeConfig)
        assert step.config.worker_resources == ResourceConfig(ram=expected_ram, regions=[EUROPE_REGION])
