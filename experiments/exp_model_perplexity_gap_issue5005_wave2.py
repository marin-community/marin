# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the issue #5005 wave-2 gap eval on the currently materialized slices.

See https://github.com/marin-community/marin/issues/5005.

This runner is intentionally limited to families that already have concrete
raw-text materializers in-tree:

- #5052 synthetic reasoning
- #5056 raw web / markup (SVG XML)
- #5058 bio / chem notation
- #5059 first-wave tabular text
- #5060 formal methods / RTL
- #5062 game / music symbolic notation

Registry-only families that still need concrete corpora (#5057 security
artifacts, #5061 package metadata) are left out of this launch.
"""

from fray.v2.types import ResourceConfig

from experiments.bio_chem_notation import bio_chem_raw_validation_sets
from experiments.evals.exp5060_formal_methods_evals import exp5060_raw_validation_sets
from experiments.evals.long_tail_ppl import LongTailPplFamily
from experiments.evals.long_tail_ppl_runnable import runnable_long_tail_ppl_slices
from experiments.evals.synthetic_reasoning_ppl import synthetic_reasoning_raw_validation_sets
from experiments.exp5052_synthetic_reasoning_ppl import RAW_SYNTHETIC_REASONING_PPL
from experiments.exp5056_raw_web_markup_ppl import raw_web_markup_raw_validation_sets
from experiments.marin_models import marin_tokenizer
from marin.evaluation.perplexity_gap import (
    GapFinderModelConfig,
    RawTextEvaluationDataset,
    default_model_perplexity_gap,
)
from marin.execution.executor import executor_main

RESOURCE_CONFIG = ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])
MAX_DOCS_PER_DATASET = 128
MAX_DOC_BYTES = 32_768
MAX_EVAL_LENGTH = 4096
PER_DEVICE_BATCH_SIZE = 1
SKIPPED_DATASETS = frozenset(
    {
        "formal_methods/tptp",
        "formal_methods/dimacs_cnf",
        "hardware_rtl/rtl_repo",
        "hardware_rtl/rtl_coder",
        "bio_chem/rcsb/rcsb_mmcif",
        "bio_chem/refseq/refseq_viral_fasta",
        "bio_chem/refseq/refseq_viral_gff",
        "long_tail_ppl_runnable/game_music/lichess_pgn_2013_06",
    }
)


def _game_music_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    return {
        slice_.registry_key: slice_.to_raw_text_dataset()
        for slice_ in runnable_long_tail_ppl_slices(family=LongTailPplFamily.GAME_MUSIC)
        if slice_.registry_key not in SKIPPED_DATASETS
    }


def issue5005_wave2_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    datasets: dict[str, RawTextEvaluationDataset] = {}
    datasets.update(synthetic_reasoning_raw_validation_sets(synthetic_reasoning_raw=RAW_SYNTHETIC_REASONING_PPL))
    datasets.update(raw_web_markup_raw_validation_sets())
    datasets.update(_game_music_raw_validation_sets())
    datasets.update(
        {name: dataset for name, dataset in exp5060_raw_validation_sets().items() if name not in SKIPPED_DATASETS}
    )
    datasets.update(
        {name: dataset for name, dataset in bio_chem_raw_validation_sets().items() if name not in SKIPPED_DATASETS}
    )
    return datasets


DATASETS = issue5005_wave2_raw_validation_sets()

MARIN_MODEL = GapFinderModelConfig(
    checkpoint_path="marin-community/marin-32b-base",
    checkpoint_is_hf=True,
    tokenizer=marin_tokenizer,
)

COMMON_TAGS = [
    "eval=perplexity-gap",
    "issue=5005",
    "rerun=wave2-materialized-slices",
    "model_a=marin-community/marin-32b-base",
    "model_b=Qwen/Qwen3-32B",
    "region=us-central1",
    "dataset_bundle=issue5005_wave2_materialized",
    "families=wave2_core",
    f"max_docs_per_dataset={MAX_DOCS_PER_DATASET}",
]

MARIN32_VS_QWEN32 = default_model_perplexity_gap(
    name="issue5005-wave2-marin-32b-base-vs-qwen3-32b-doccap128",
    model_a=MARIN_MODEL,
    model_b=GapFinderModelConfig(
        checkpoint_path="Qwen/Qwen3-32B",
        checkpoint_is_hf=True,
        tokenizer="Qwen/Qwen3-32B",
    ),
    datasets=DATASETS,
    resource_config=RESOURCE_CONFIG,
    per_device_batch_size=PER_DEVICE_BATCH_SIZE,
    max_eval_length=MAX_EVAL_LENGTH,
    max_docs_per_dataset=MAX_DOCS_PER_DATASET,
    max_doc_bytes=MAX_DOC_BYTES,
    wandb_tags=COMMON_TAGS,
)


if __name__ == "__main__":
    executor_main(
        [MARIN32_VS_QWEN32],
        description="Issue #5005 wave-2 gap eval on currently materialized long-tail slices.",
    )
