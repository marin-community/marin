# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dolma 3 data mixture swarm experiment.

This experiment prepares data from:
- Dolma 3 Pool (https://huggingface.co/datasets/allenai/dolma3_pool) - 9.31T tokens (333 partitions)
- Dolma 3 Dolmino Pool (https://huggingface.co/datasets/allenai/dolma3_dolmino_pool) - 2.19T tokens (133 partitions)

Phase 1: Download and tokenize the datasets
Phase 2: Run mixture experiments (TODO)

Usage:
    # Prepare Dolmino Pool data (download and tokenize)
    python -m experiments.domain_phase_mix.swarm_run --prepare_data --dataset dolmino_pool

    # Prepare Dolma 3 Pool data (download and tokenize)
    python -m experiments.domain_phase_mix.swarm_run --prepare_data --dataset dolma3_pool

    # Prepare specific category from either dataset
    python -m experiments.domain_phase_mix.swarm_run --prepare_data --dataset dolmino_pool --category stem_heavy_crawl
    python -m experiments.domain_phase_mix.swarm_run --prepare_data --dataset dolma3_pool --category stack_edu

    # Run training (after data is prepared)
    python -m experiments.domain_phase_mix.swarm_run [--n_runs N] [--seed SEED]
"""

import logging
import os

from marin.execution.executor import executor_main

from experiments.pretraining_datasets.dolma3_dolmino_pool import (
    download_dolmino_pool,
    tokenize_dolmino_pool,
    get_common_crawl_hq_partitions as get_dolmino_common_crawl_hq_partitions,
    get_olmocr_pdfs_hq_partitions as get_dolmino_olmocr_pdfs_hq_partitions,
    get_stack_edu_fim_partitions as get_dolmino_stack_edu_fim_partitions,
    get_stem_heavy_crawl_partitions as get_dolmino_stem_heavy_crawl_partitions,
    get_synthetic_partitions as get_dolmino_synthetic_partitions,
    DOLMINO_POOL_PARTITIONS,
)
from experiments.pretraining_datasets.dolma3_pool import (
    download_all_dolma3_pool_sources,
    tokenize_dolma3_pool,
    get_common_crawl_partitions as get_dolma3_common_crawl_partitions,
    get_olmocr_pdfs_partitions as get_dolma3_olmocr_pdfs_partitions,
    get_stack_edu_partitions as get_dolma3_stack_edu_partitions,
    get_finemath_partitions as get_dolma3_finemath_partitions,
    get_arxiv_partitions as get_dolma3_arxiv_partitions,
    get_wikipedia_partitions as get_dolma3_wikipedia_partitions,
    DOLMA3_POOL_PARTITIONS,
)

logger = logging.getLogger("ray")


# ============================================================================
# DATA PREPARATION
# ============================================================================


def prepare_dolmino_pool_data(
    partitions: list[str] | None = None,
    tokenizer: str | None = None,
):
    """Prepare Dolmino Pool data by downloading and tokenizing.

    Args:
        partitions: List of partition names to prepare. If None, prepares all.
        tokenizer: Tokenizer to use. Defaults to marin_tokenizer.
    """
    if os.getenv("CI", None) is not None:
        logger.info("Skipping data preparation on CI environment.")
        return

    # Get download step
    download_step = download_dolmino_pool()

    # Get tokenization steps
    tokenize_steps = tokenize_dolmino_pool(
        tokenizer=tokenizer,
        partitions=partitions,
    )

    # Combine all steps
    all_steps = [download_step, *tokenize_steps.values()]

    logger.info(f"Preparing {len(tokenize_steps)} partitions from Dolmino Pool")
    logger.info(f"Download step: {download_step.name}")
    logger.info(f"Tokenization steps: {len(tokenize_steps)}")

    executor_main(
        steps=all_steps,
        description=f"Dolmino Pool data preparation: {len(tokenize_steps)} partitions",
    )


def prepare_dolmino_pool_by_category(
    category: str,
    tokenizer: str | None = None,
):
    """Prepare a specific category of Dolmino Pool data.

    Args:
        category: One of "common_crawl_hq", "olmocr_pdfs_hq", "stack_edu_fim",
                  "stem_heavy_crawl", "synthetic", or "all"
        tokenizer: Tokenizer to use. Defaults to marin_tokenizer.
    """
    if category == "common_crawl_hq":
        partitions = get_dolmino_common_crawl_hq_partitions()
    elif category == "olmocr_pdfs_hq":
        partitions = get_dolmino_olmocr_pdfs_hq_partitions()
    elif category == "stack_edu_fim":
        partitions = get_dolmino_stack_edu_fim_partitions()
    elif category == "stem_heavy_crawl":
        partitions = get_dolmino_stem_heavy_crawl_partitions()
    elif category == "synthetic":
        partitions = get_dolmino_synthetic_partitions()
    elif category == "all":
        partitions = None  # All partitions
    else:
        raise ValueError(
            f"Unknown category: {category}. "
            "Must be one of: common_crawl_hq, olmocr_pdfs_hq, stack_edu_fim, stem_heavy_crawl, synthetic, all"
        )

    logger.info(f"Preparing Dolmino Pool category: {category}")
    if partitions:
        logger.info(f"Partitions: {len(partitions)}")

    prepare_dolmino_pool_data(partitions=partitions, tokenizer=tokenizer)


# ============================================================================
# DOLMA 3 POOL DATA PREPARATION
# ============================================================================


def prepare_dolma3_pool_data(
    partitions: list[str] | None = None,
    tokenizer: str | None = None,
):
    """Prepare Dolma 3 Pool data by downloading and tokenizing.

    Args:
        partitions: List of partition names to prepare. If None, prepares all.
        tokenizer: Tokenizer to use. Defaults to marin_tokenizer.
    """
    if os.getenv("CI", None) is not None:
        logger.info("Skipping data preparation on CI environment.")
        return

    # Get download steps (multiple sources)
    download_steps = download_all_dolma3_pool_sources()

    # Get tokenization steps
    tokenize_steps = tokenize_dolma3_pool(
        tokenizer=tokenizer,
        partitions=partitions,
    )

    # Combine all steps
    all_steps = [*download_steps, *tokenize_steps.values()]

    logger.info(f"Preparing {len(tokenize_steps)} partitions from Dolma 3 Pool")
    logger.info(f"Download steps: {len(download_steps)}")
    logger.info(f"Tokenization steps: {len(tokenize_steps)}")

    executor_main(
        steps=all_steps,
        description=f"Dolma 3 Pool data preparation: {len(tokenize_steps)} partitions",
    )


def prepare_dolma3_pool_by_category(
    category: str,
    tokenizer: str | None = None,
):
    """Prepare a specific category of Dolma 3 Pool data.

    Args:
        category: One of "common_crawl", "olmocr_pdfs", "stack_edu",
                  "finemath", "arxiv", "wikipedia", or "all"
        tokenizer: Tokenizer to use. Defaults to marin_tokenizer.
    """
    if category == "common_crawl":
        partitions = get_dolma3_common_crawl_partitions()
    elif category == "olmocr_pdfs":
        partitions = get_dolma3_olmocr_pdfs_partitions()
    elif category == "stack_edu":
        partitions = get_dolma3_stack_edu_partitions()
    elif category == "finemath":
        partitions = get_dolma3_finemath_partitions()
    elif category == "arxiv":
        partitions = get_dolma3_arxiv_partitions()
    elif category == "wikipedia":
        partitions = get_dolma3_wikipedia_partitions()
    elif category == "all":
        partitions = None  # All partitions
    else:
        raise ValueError(
            f"Unknown category: {category}. "
            "Must be one of: common_crawl, olmocr_pdfs, stack_edu, finemath, arxiv, wikipedia, all"
        )

    logger.info(f"Preparing Dolma 3 Pool category: {category}")
    if partitions:
        logger.info(f"Partitions: {len(partitions)}")

    prepare_dolma3_pool_data(partitions=partitions, tokenizer=tokenizer)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

# Valid categories for each dataset
DOLMINO_POOL_CATEGORIES = [
    "common_crawl_hq",
    "olmocr_pdfs_hq",
    "stack_edu_fim",
    "stem_heavy_crawl",
    "synthetic",
    "all",
]

DOLMA3_POOL_CATEGORIES = [
    "common_crawl",
    "olmocr_pdfs",
    "stack_edu",
    "finemath",
    "arxiv",
    "wikipedia",
    "all",
]


def main(
    prepare_data: bool = False,
    dataset: str = "dolmino_pool",
    category: str = "all",
    n_runs: int = 100,
    seed: int = 42,
    name_prefix: str = "pinlin_calvin_xu/data_mixture/dolma3_swarm",
    batch_size: int | None = None,
):
    """Main entry point for the Dolma 3 swarm experiment.

    Args:
        prepare_data: If True, download and tokenize data instead of running training.
        dataset: Which dataset to prepare ("dolmino_pool" or "dolma3_pool").
        category: Data category to prepare (only used with --prepare_data).
        n_runs: Number of training runs (only used without --prepare_data).
        seed: Random seed for weight sampling.
        name_prefix: Prefix for run names.
        batch_size: If set, run training steps in batches to limit parallelism.
    """
    if prepare_data:
        if dataset == "dolmino_pool":
            prepare_dolmino_pool_by_category(category=category)
        elif dataset == "dolma3_pool":
            prepare_dolma3_pool_by_category(category=category)
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Must be 'dolmino_pool' or 'dolma3_pool'")
        return

    # TODO: Implement training runs using MixtureExperiment
    # This will be similar to three_phase_experiment.py but with Dolma 3 domains
    logger.info("Training runs not yet implemented. Use --prepare_data to prepare data first.")

    logger.info("")
    logger.info("Dolmino Pool (midtraining):")
    logger.info(f"  Total partitions: {len(DOLMINO_POOL_PARTITIONS)}")
    logger.info(f"  - Common Crawl HQ: {len(get_dolmino_common_crawl_hq_partitions())} partitions")
    logger.info(f"  - olmOCR PDFs HQ: {len(get_dolmino_olmocr_pdfs_hq_partitions())} partitions")
    logger.info(f"  - Stack-Edu FIM: {len(get_dolmino_stack_edu_fim_partitions())} partitions")
    logger.info(f"  - STEM Heavy Crawl: {len(get_dolmino_stem_heavy_crawl_partitions())} partitions")
    logger.info(f"  - Synthetic: {len(get_dolmino_synthetic_partitions())} partitions")

    logger.info("")
    logger.info("Dolma 3 Pool (pretraining):")
    logger.info(f"  Total partitions: {len(DOLMA3_POOL_PARTITIONS)}")
    logger.info(f"  - Common Crawl: {len(get_dolma3_common_crawl_partitions())} partitions")
    logger.info(f"  - olmOCR PDFs: {len(get_dolma3_olmocr_pdfs_partitions())} partitions")
    logger.info(f"  - Stack-Edu: {len(get_dolma3_stack_edu_partitions())} partitions")
    logger.info(f"  - FineMath: {len(get_dolma3_finemath_partitions())} partitions")
    logger.info(f"  - arXiv: {len(get_dolma3_arxiv_partitions())} partitions")
    logger.info(f"  - Wikipedia: {len(get_dolma3_wikipedia_partitions())} partitions")


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Dolma 3 data mixture swarm experiment.")
    parser.add_argument(
        "--prepare_data",
        action="store_true",
        help="Download and tokenize data instead of running training.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dolmino_pool",
        choices=["dolmino_pool", "dolma3_pool"],
        help="Which dataset to prepare (only used with --prepare_data).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        help=(
            "Data category to prepare (only used with --prepare_data). "
            "For dolmino_pool: common_crawl_hq, olmocr_pdfs_hq, stack_edu_fim, stem_heavy_crawl, synthetic, all. "
            "For dolma3_pool: common_crawl, olmocr_pdfs, stack_edu, finemath, arxiv, wikipedia, all."
        ),
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=100,
        help="Number of training runs (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for weight sampling (default: 42).",
    )
    parser.add_argument(
        "--name_prefix",
        type=str,
        default="pinlin_calvin_xu/data_mixture/dolma3_swarm",
        help="Prefix for run names.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Run training steps in batches of this size to limit parallelism.",
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    import sys

    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    main(
        prepare_data=args.prepare_data,
        dataset=args.dataset,
        category=args.category,
        n_runs=args.n_runs,
        seed=args.seed,
        name_prefix=args.name_prefix,
        batch_size=args.batch_size,
    )
