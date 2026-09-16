# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fray.cluster import ResourceConfig
from marin.execution.context import executor_context
from marin.execution.executor import collect_dependencies_and_version, instantiate_config
from marin.processing.tokenize import HistoricalFullDocumentTokenizeConfig

from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import TOP_LEVEL_DOMAIN_PARTITIONS
from experiments.domain_phase_mix.prepare_delphi_tpp40_europe_historical_stack_caches import (
    EUROPE_REGION,
    MERGED_OUTPUT_NAME,
    SQL_PARTITION,
    SQL_WORKER_RAM,
    STACK_DOMAIN,
    WORKER_RAM,
    historical_stack_cache_step,
    historical_stack_partition_step,
)
from experiments.domain_phase_mix.prepare_delphi_tpp40_europe_historical_stack_canary import CANARY_PARTITION


def test_historical_stack_graph_is_complete_isolated_and_region_local() -> None:
    with executor_context():
        merge_step = historical_stack_cache_step()

    assert merge_step.name == f"tokenized/merged/{MERGED_OUTPUT_NAME}"
    assert set(merge_step.config.input_configs) == set(TOP_LEVEL_DOMAIN_PARTITIONS[STACK_DOMAIN])

    for partition_name, source in merge_step.config.input_configs.items():
        tokenize_step = source.cache_dir.step
        assert isinstance(tokenize_step.config, HistoricalFullDocumentTokenizeConfig)
        assert tokenize_step.config.historical_full_document_tokenization
        expected_ram = SQL_WORKER_RAM if partition_name == SQL_PARTITION else WORKER_RAM
        assert tokenize_step.config.worker_resources == ResourceConfig(
            ram=expected_ram,
            regions=[EUROPE_REGION],
        )
        assert "dolma3_pool_historical_full_document_v1" in tokenize_step.name

        [hydrated_path] = tokenize_step.config.train_paths
        hydration_step = hydrated_path.step
        assert hydration_step.config.worker_resources.regions == [EUROPE_REGION]
        assert hydration_step.config.input_path.step.config.worker_resources.regions == [EUROPE_REGION]


def test_canary_reuses_the_full_graph_ruby_step() -> None:
    with executor_context():
        canary_step = historical_stack_partition_step(CANARY_PARTITION)
        full_graph = historical_stack_cache_step()

    full_graph_ruby_step = full_graph.config.input_configs[CANARY_PARTITION].cache_dir.step
    assert canary_step is full_graph_ruby_step

    dependencies = collect_dependencies_and_version(canary_step.config).dependencies
    materialized = instantiate_config(
        canary_step.config,
        output_path="gs://marin-eu-west4/materialized-output",
        output_paths={
            dependency: f"gs://marin-eu-west4/materialized-input-{index}"
            for index, dependency in enumerate(dependencies)
        },
        prefix="gs://marin-eu-west4",
    )
    assert isinstance(materialized, HistoricalFullDocumentTokenizeConfig)
    assert materialized.historical_full_document_tokenization
