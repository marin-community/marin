# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild Europe TPP40 exception caches with historical tokenization semantics."""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

from fray.cluster import ResourceConfig
from marin.datakit.download.huggingface import DownloadConfig, download_hf
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorStep, executor_main
from marin.execution.types import VersionedValue, this_output_path, versioned
from marin.processing.tokenize import HistoricalFullDocumentTokenizeConfig, tokenize
from marin.processing.tokenize.merge_tokenized_caches import merge_tokenized_caches

from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import TOP_LEVEL_DOMAIN_PARTITIONS
from experiments.llama import llama3_tokenizer
from experiments.marin_tokenizer import marin_tokenizer
from experiments.pretraining_datasets.dolma3_dolmino_pool import tokenize_dolmino_pool_subset

EUROPE_REGION = "europe-west4"
EUROPE_PREFIX = "gs://marin-eu-west4"
FINEMATH_DATASET_ID = "HuggingFaceTB/finemath"
FINEMATH_REVISION = "8f233cf"
FINEMATH_RAW_OUTPUT_PATH = f"{EUROPE_PREFIX}/raw/finemath-7090a5"
FINEMATH_PARTITION = "finemath-3plus"
FINEMATH_OUTPUT_NAME = "tokenized/finemath_3_plus_historical_full_document_v1"
STEM_DOMAIN = "dolmino_stem_heavy_crawl"
STEM_MERGED_OUTPUT_NAME = "dolma3_dolmino_top_level_historical_full_document_v1/dolmino_stem_heavy_crawl"
STEM_HISTORICAL_PREPROCESSOR_METADATA = {
    "tokenizer": "marin-community/marin-tokenizer",
    "vocab_size": 128256,
    "return_attention_mask": False,
    "padding": False,
    "max_length": 131072,
    "append_bos": False,
    "append_eos": True,
}
CANARY_PARTITION = "stem_heavy_crawl/adult_content"
STRESS_PARTITION = "synth_thinking/program_verifiable"
HISTORICAL_SINGLETON_PARTITIONS = (
    "synth_instruction/dolmino_flan",
    "synth_math/dolmino_math",
    "synth_qa/wiki_to_rcqa",
    "synth_thinking/code_meta_reasoning",
    "synth_thinking/math_meta_reasoning",
    "synth_thinking/program_verifiable",
)
FLAN_PARTITION = "synth_instruction/dolmino_flan"
FLAN_RAW_PREFIX = f"{EUROPE_PREFIX}/raw/dolma3_dolmino_pool-72089d/data/dolmino_1-flan"
FLAN_SOURCE_SHARD_COUNT = 209
# These three raw objects were present in both regional source buckets but were
# not represented in the frozen East5 runtime cache's 206-shard ledger.
FLAN_EXCLUDED_SOURCE_SHARD_INDICES = (64, 122, 163)
FLAN_OUTPUT_NAME = (
    "tokenized/dolma3_dolmino_pool_historical_full_document_east5_subset_v1/" "synth_instruction_dolmino_flan"
)
# Frozen from the zero-exclusion v4 diagnostic inventory. The exact o4mini
# canary already matches across regions and therefore is not rebuilt.
HISTORICAL_REPAIR_COMPONENTS = frozenset(
    {
        "finemath_3plus",
        STEM_DOMAIN,
        "synth_instruction/dolmino_flan",
        "synth_math/dolmino_math",
        "synth_qa/wiki_to_rcqa",
        "synth_thinking/code_meta_reasoning",
        "synth_thinking/math_meta_reasoning",
        "synth_thinking/program_verifiable",
    }
)
PARTITION_RAM_OVERRIDES = {
    # Europe CPU workers have 16 GiB physical RAM. Leave headroom for the
    # worker runtime while retaining a larger envelope than the 10g default.
    "synth_math/dolmino_math": "14g",
}
DEFAULT_WORKER_RAM = "10g"
FINEMATH_WORKER_RAM = "14g"


@dataclasses.dataclass(frozen=True, kw_only=True)
class FrozenSourceHistoricalFullDocumentTokenizeConfig(HistoricalFullDocumentTokenizeConfig):
    """Historical tokenizer config whose explicit source manifest affects identity."""

    source_paths_identity: VersionedValue[tuple[str, ...]]


def historical_flan_step() -> ExecutorStep:
    """Reproduce the exact 206-source-shard FLAN cache consumed in East5."""
    excluded_indices = set(FLAN_EXCLUDED_SOURCE_SHARD_INDICES)
    train_paths = [
        f"{FLAN_RAW_PREFIX}/tulu_flan-{index:04d}.jsonl.zst"
        for index in range(FLAN_SOURCE_SHARD_COUNT)
        if index not in excluded_indices
    ]
    return ExecutorStep(
        name=FLAN_OUTPUT_NAME,
        fn=tokenize,
        config=FrozenSourceHistoricalFullDocumentTokenizeConfig(
            train_paths=train_paths,
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(marin_tokenizer),
            worker_resources=ResourceConfig(ram=DEFAULT_WORKER_RAM, regions=[EUROPE_REGION]),
            source_paths_identity=versioned(tuple(train_paths)),
        ),
    )


def historical_dolmino_partition_step(partition_name: str) -> ExecutorStep:
    """Return one Europe-local Dolmino partition with unsplit BPE boundaries."""
    historical_partitions = set(TOP_LEVEL_DOMAIN_PARTITIONS[STEM_DOMAIN]) | set(HISTORICAL_SINGLETON_PARTITIONS)
    if partition_name not in historical_partitions:
        raise ValueError(f"Partition is outside the frozen historical-repair set: {partition_name}")
    if partition_name == FLAN_PARTITION:
        return historical_flan_step()
    worker_ram = PARTITION_RAM_OVERRIDES.get(partition_name, DEFAULT_WORKER_RAM)
    return tokenize_dolmino_pool_subset(
        partition_name,
        tokenizer=marin_tokenizer,
        worker_resources=ResourceConfig(ram=worker_ram, regions=[EUROPE_REGION]),
        split_long_documents=False,
    )


def finemath_raw_step() -> ExecutorStep:
    """Reference the already-complete Europe Finemath download as a dependency."""
    return ExecutorStep(
        name="raw/finemath",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id=FINEMATH_DATASET_ID,
            revision=FINEMATH_REVISION,
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
        ),
    ).with_output_path(FINEMATH_RAW_OUTPUT_PATH)


def historical_finemath_step() -> ExecutorStep:
    """Return the Europe-local Finemath cache with unsplit BPE boundaries."""
    return ExecutorStep(
        name=FINEMATH_OUTPUT_NAME,
        fn=tokenize,
        config=HistoricalFullDocumentTokenizeConfig(
            train_paths=[finemath_raw_step().cd(FINEMATH_PARTITION)],
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            # Match the frozen East5 producer exactly. TPP40 adopts this
            # singleton cache through ExistingTokenizedCacheConfig with the
            # canonical Marin tokenizer identity.
            tokenizer=versioned(llama3_tokenizer),
            worker_resources=ResourceConfig(ram=FINEMATH_WORKER_RAM, regions=[EUROPE_REGION]),
        ),
    )


def historical_stem_cache_step() -> ExecutorStep:
    """Return the merged Europe STEM cache over historical source partitions."""
    input_steps = {
        partition_name: historical_dolmino_partition_step(partition_name)
        for partition_name in TOP_LEVEL_DOMAIN_PARTITIONS[STEM_DOMAIN]
    }
    return merge_tokenized_caches(
        output_cache_path_name=STEM_MERGED_OUTPUT_NAME,
        input_steps=input_steps,
        tokenizer=marin_tokenizer,
        tags=[STEM_DOMAIN, "historical_full_document_tokenization"],
        preprocessor_metadata=STEM_HISTORICAL_PREPROCESSOR_METADATA,
    )


def historical_nonstack_steps(scope: str) -> list[ExecutorStep]:
    """Materialize the canary or the complete non-Stack repair graph."""
    if scope == "canary":
        return [historical_dolmino_partition_step(CANARY_PARTITION)]
    if scope == "stress":
        return [historical_dolmino_partition_step(STRESS_PARTITION)]
    if scope == "flan":
        return [historical_flan_step()]
    if scope == "stem":
        return [historical_stem_cache_step()]
    if scope != "all":
        raise ValueError(f"Unknown scope: {scope}")
    configured_components = frozenset({"finemath_3plus", STEM_DOMAIN, *HISTORICAL_SINGLETON_PARTITIONS})
    if configured_components != HISTORICAL_REPAIR_COMPONENTS:
        raise ValueError("Historical repair graph does not match the frozen v4 diagnostic inventory")
    return [
        historical_finemath_step(),
        historical_stem_cache_step(),
        *(historical_dolmino_partition_step(partition) for partition in HISTORICAL_SINGLETON_PARTITIONS),
    ]


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("canary", "stress", "flan", "stem", "all"), required=True)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if os.environ.get("MARIN_PREFIX") != EUROPE_PREFIX:
        raise ValueError(f"MARIN_PREFIX must be {EUROPE_PREFIX!r} for this Europe-only repair")
    with executor_context():
        steps = historical_nonstack_steps(args.scope)
    executor_main(
        steps=steps,
        description=f"Prepare Europe TPP40 historical non-Stack caches ({args.scope})",
    )


if __name__ == "__main__":
    main()
