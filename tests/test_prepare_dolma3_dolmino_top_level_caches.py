# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from fray.cluster import ResourceConfig
from marin.execution.context import executor_context
from marin.processing.tokenize import HistoricalFullDocumentTokenizeConfig, TokenizeConfig
from marin.processing.tokenize.data_configs import ExistingTokenizedCacheConfig

from experiments.domain_phase_mix import two_phase_dolma3_dolmino_top_level as top_level
from experiments.domain_phase_mix.delphi_tpp40_europe_runtime_caches import (
    EUROPE_HISTORICAL_NONSTACK_REPAIR_PATHS,
    EUROPE_SOURCE_RUNTIME_CACHE_PATHS,
)
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import TOP_LEVEL_DOMAIN_PARTITIONS
from experiments.domain_phase_mix.prepare_dolma3_dolmino_top_level_caches import (
    STACK_EDU_DOMAIN,
    _prep_steps,
    _selected_domain_names,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import DOMAIN_NAMES, _partition_step_fn
from experiments.pretraining_datasets.dolma3_dolmino_pool import (
    DOLMINO_PARTITIONS_WITH_TEST_NAMED_TRAIN_SHARDS,
    tokenize_dolmino_pool_subset,
)


def test_stack_edu_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="--allow-stack-edu"):
        _selected_domain_names("all", allow_stack_edu=False)

    non_stack_domains = tuple(name for name in DOMAIN_NAMES if name != STACK_EDU_DOMAIN)
    selected = _selected_domain_names(",".join(non_stack_domains), allow_stack_edu=False)

    assert selected == non_stack_domains


@pytest.mark.parametrize("partition_name", sorted(DOLMINO_PARTITIONS_WITH_TEST_NAMED_TRAIN_SHARDS))
def test_known_dolmino_training_partitions_allow_test_named_shards(partition_name: str) -> None:
    step = tokenize_dolmino_pool_subset(partition_name)

    assert isinstance(step.config, TokenizeConfig)
    assert step.config.allow_test_in_train


def test_other_dolmino_training_partitions_keep_test_path_guard() -> None:
    step = tokenize_dolmino_pool_subset("synth_math/cranemath")

    assert isinstance(step.config, TokenizeConfig)
    assert not step.config.allow_test_in_train


def test_dolmino_historical_tokenization_uses_isolated_cache_namespace() -> None:
    step = tokenize_dolmino_pool_subset(
        "stem_heavy_crawl/adult_content",
        split_long_documents=False,
    )

    assert step.name == ("tokenized/dolma3_dolmino_pool_historical_full_document_v1/stem_heavy_crawl_adult_content")
    assert isinstance(step.config, HistoricalFullDocumentTokenizeConfig)
    assert not step.config.split_long_documents


def test_large_partitions_use_targeted_tokenization_memory_overrides() -> None:
    regional_resources = ResourceConfig(regions=["europe-west4"])

    sql_step = _partition_step_fn("stack_edu/SQL", worker_resources=regional_resources)()
    math_step = _partition_step_fn("synth_math/dolmino_math", worker_resources=regional_resources)()
    other_step = _partition_step_fn("synth_math/cranemath", worker_resources=regional_resources)()

    assert isinstance(sql_step.config, TokenizeConfig)
    assert sql_step.config.worker_resources == ResourceConfig(ram="20g", regions=["europe-west4"])
    assert isinstance(math_step.config, TokenizeConfig)
    assert math_step.config.worker_resources == ResourceConfig(ram="20g", regions=["europe-west4"])
    assert isinstance(other_step.config, TokenizeConfig)
    assert other_step.config.worker_resources == regional_resources


def test_europe_stack_edu_graph_is_complete_and_region_local(monkeypatch: pytest.MonkeyPatch) -> None:
    target_path = top_level.PREBUILT_MERGED_RUNTIME_CACHE_PATHS_BY_REGION["europe-west4"][STACK_EDU_DOMAIN]
    monkeypatch.setattr(
        top_level,
        "_runtime_cache_is_complete",
        lambda path: path != target_path,
    )
    with executor_context():
        [merge_step] = _prep_steps((STACK_EDU_DOMAIN,), "europe-west4")

    assert merge_step.name == "tokenized/merged/dolma3_dolmino_top_level/dolma3_stack_edu"
    assert len(merge_step.config.input_configs) == 15
    for source in merge_step.config.input_configs.values():
        tokenize_step = source.cache_dir.step
        assert tokenize_step.config.worker_resources.regions == ["europe-west4"]
        [hydrated_path] = tokenize_step.config.train_paths
        hydration_step = hydrated_path.step
        assert hydration_step.name.startswith("documents/stack_edu/")
        assert hydration_step.config.worker_resources.regions == ["europe-west4"]
        assert hydration_step.config.input_path.step.name == "raw/stack_edu"
        assert hydration_step.config.input_path.step.config.worker_resources.regions == ["europe-west4"]


def test_europe_training_domains_use_only_pinned_finished_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(top_level, "_resolve_finished_gcs_cache_path", lambda path: path)

    domains = top_level.build_top_level_domains(runtime_cache_region="europe-west4")

    assert len(domains) == 39
    assert set(EUROPE_SOURCE_RUNTIME_CACHE_PATHS) == {
        partition
        for domain_name, partitions in TOP_LEVEL_DOMAIN_PARTITIONS.items()
        if domain_name not in top_level.PREFERRED_MERGED_RUNTIME_DOMAIN_NAMES
        for partition in partitions
    }
    configs = [component.step_fn() for domain in domains for component in domain.components]
    assert configs
    assert all(isinstance(config, ExistingTokenizedCacheConfig) for config in configs)
    assert all(config.cache_path.startswith("gs://marin-eu-west4/") for config in configs)


def test_europe_training_bindings_use_verified_repair_paths() -> None:
    source_repair_paths = {
        component: path
        for component, path in EUROPE_HISTORICAL_NONSTACK_REPAIR_PATHS.items()
        if component != "dolmino_stem_heavy_crawl"
    }

    assert {component: EUROPE_SOURCE_RUNTIME_CACHE_PATHS[component] for component in source_repair_paths} == (
        source_repair_paths
    )
    assert (
        top_level.PREBUILT_MERGED_RUNTIME_CACHE_PATHS_BY_REGION["europe-west4"]["dolmino_stem_heavy_crawl"]
        == EUROPE_HISTORICAL_NONSTACK_REPAIR_PATHS["dolmino_stem_heavy_crawl"]
    )


def test_strict_training_domains_reject_multiregion_mirror_fallback() -> None:
    with pytest.raises(ValueError, match="exactly one region"):
        top_level.build_top_level_domains(runtime_cache_region=("us-east5", "europe-west4"))


def test_runtime_cache_completion_requires_finished_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(top_level, "_read_executor_status", lambda path: "SUCCESS")
    monkeypatch.setattr(
        top_level,
        "_read_runtime_cache_json",
        lambda cache_path, relative_path: {"is_finished": False},
    )
    monkeypatch.setattr(top_level, "_runtime_cache_path_exists", lambda cache_path, relative_path: True)
    top_level._runtime_cache_is_complete.cache_clear()

    assert not top_level._runtime_cache_is_complete("gs://bucket/cache")


def test_runtime_cache_completion_accepts_finished_ledger_and_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(top_level, "_read_executor_status", lambda path: "FAILED")
    monkeypatch.setattr(
        top_level,
        "_read_runtime_cache_json",
        lambda cache_path, relative_path: {"is_finished": True},
    )
    monkeypatch.setattr(top_level, "_runtime_cache_path_exists", lambda cache_path, relative_path: True)
    top_level._runtime_cache_is_complete.cache_clear()

    assert top_level._runtime_cache_is_complete("gs://bucket/cache")


def test_wildcard_runtime_cache_resolution_rejects_ambiguous_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        top_level,
        "_finished_cache_paths_under",
        lambda parent: ("gs://bucket/cache-a", "gs://bucket/cache-b"),
    )
    top_level._resolve_finished_gcs_cache_path.cache_clear()

    with pytest.raises(ValueError, match="Multiple complete"):
        top_level._resolve_finished_gcs_cache_path("gs://bucket/cache-*")
