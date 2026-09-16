# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build Europe-local Stack caches with the historical East5 tokenizer semantics."""

from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main
from marin.processing.tokenize.merge_tokenized_caches import merge_tokenized_caches

from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import TOP_LEVEL_DOMAIN_PARTITIONS
from experiments.marin_tokenizer import marin_tokenizer
from experiments.pretraining_datasets.dolma3_pool import tokenize_dolma3_pool_subset

EUROPE_REGION = "europe-west4"
STACK_DOMAIN = "dolma3_stack_edu"
MERGED_OUTPUT_NAME = "dolma3_dolmino_top_level_historical_full_document_v1/dolma3_stack_edu"
SQL_PARTITION = "stack_edu/SQL"
WORKER_RAM = "10g"
SQL_WORKER_RAM = "20g"


def historical_stack_partition_step(partition_name: str) -> ExecutorStep:
    """Return one Europe-local Stack partition with historical tokenization."""
    if partition_name not in TOP_LEVEL_DOMAIN_PARTITIONS[STACK_DOMAIN]:
        raise ValueError(f"Unknown Stack partition: {partition_name}")
    worker_ram = SQL_WORKER_RAM if partition_name == SQL_PARTITION else WORKER_RAM
    return tokenize_dolma3_pool_subset(
        partition_name,
        tokenizer=marin_tokenizer,
        worker_resources=ResourceConfig(ram=worker_ram, regions=[EUROPE_REGION]),
        split_long_documents=False,
    )


def historical_stack_cache_step() -> ExecutorStep:
    """Return the merged Stack cache rooted in Europe-local raw data."""
    input_steps = {
        partition_name: historical_stack_partition_step(partition_name)
        for partition_name in TOP_LEVEL_DOMAIN_PARTITIONS[STACK_DOMAIN]
    }

    return merge_tokenized_caches(
        output_cache_path_name=MERGED_OUTPUT_NAME,
        input_steps=input_steps,
        tokenizer=marin_tokenizer,
        tags=[STACK_DOMAIN, "historical_full_document_tokenization"],
    )


def main() -> None:
    executor_main(
        steps=[historical_stack_cache_step()],
        description="Prepare Europe Stack cache with historical East5 tokenizer semantics",
    )


if __name__ == "__main__":
    main()
