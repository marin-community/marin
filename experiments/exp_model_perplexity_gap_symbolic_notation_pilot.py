# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pilot perplexity-gap report for the game / music symbolic-notation slices.

Scoped to issue #5062 under epic #5005. Answers the DoD question:

    Do gaps concentrate in metadata headers, symbolic sequences, comments,
    or numeric annotations?

The answer comes from the gap-report's per-slice byte-bucket rollup
(whitespace / punctuation / numbers / words), not from any post-hoc
analysis here. This file only wires up the models and the slice subset.

Unlike ``exp_model_perplexity_gap_long_tail_runnable`` which sweeps every
runnable slice, this pilot intentionally narrows to the
``GAME_MUSIC`` family so the report surfaces symbolic-notation behaviour
without being dominated by larger SVG / Verilog slices.

PGN and ABC docs are typically an order of magnitude shorter than an
average Paloma document, so we raise ``max_docs_per_dataset`` well above
the long-tail-runnable default of 256 to keep the compressed-byte budget
comparable to a Paloma slice (per dlwh, #5062).
"""

from fray.v2.types import ResourceConfig

from experiments.evals.long_tail_ppl import LongTailPplFamily
from experiments.evals.long_tail_ppl_runnable import runnable_long_tail_ppl_slices
from marin.evaluation.perplexity_gap import (
    GapFinderModelConfig,
    RawTextEvaluationDataset,
    default_model_perplexity_gap,
)
from marin.execution.executor import executor_main

RESOURCE_CONFIG = ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])

# PGN / ABC docs are much shorter than an average Paloma document, so a higher
# doc cap keeps the compressed-byte volume per slice roughly Paloma-sized while
# still being deterministic (HF datasets return rows in a fixed order).
MAX_DOCS_PER_DATASET = 2048
MAX_DOC_BYTES = 32_768


def _game_music_datasets() -> dict[str, RawTextEvaluationDataset]:
    return {
        slice_.registry_key: slice_.to_raw_text_dataset()
        for slice_ in runnable_long_tail_ppl_slices(family=LongTailPplFamily.GAME_MUSIC)
    }


DATASETS = _game_music_datasets()

MARIN_MODEL = GapFinderModelConfig(
    checkpoint_path="marin-community/marin-8b-base",
    checkpoint_is_hf=True,
    tokenizer="meta-llama/Llama-3.1-8B",
)

_COMMON_TAGS = [
    "eval=perplexity-gap",
    "rerun=symbolic-notation-pilot",
    "issue=5062",
    "epic=5005",
    "dataset_bundle=runnable_long_tail_hf_backed",
    "family=game_music",
    "source_split=hf_dataset",
    "region=us-central1",
    f"max_docs_per_dataset={MAX_DOCS_PER_DATASET}",
]

MARIN_VS_LLAMA = default_model_perplexity_gap(
    name="symbolic-notation-pilot-marin-8b-base-vs-llama-3.1-8b-base",
    model_a=MARIN_MODEL,
    model_b=GapFinderModelConfig(
        checkpoint_path="meta-llama/Llama-3.1-8B",
        checkpoint_is_hf=True,
        tokenizer="meta-llama/Llama-3.1-8B",
    ),
    datasets=DATASETS,
    resource_config=RESOURCE_CONFIG,
    per_device_batch_size=4,
    max_eval_length=4096,
    max_docs_per_dataset=MAX_DOCS_PER_DATASET,
    max_doc_bytes=MAX_DOC_BYTES,
    wandb_tags=[
        *_COMMON_TAGS,
        "model_a=marin-community/marin-8b-base",
        "model_b=meta-llama/Llama-3.1-8B",
    ],
)

MARIN_VS_QWEN3 = default_model_perplexity_gap(
    name="symbolic-notation-pilot-marin-8b-base-vs-qwen3-8b-base",
    model_a=MARIN_MODEL,
    model_b=GapFinderModelConfig(
        checkpoint_path="Qwen/Qwen3-8B-Base",
        checkpoint_is_hf=True,
        tokenizer="Qwen/Qwen3-8B",
    ),
    datasets=DATASETS,
    resource_config=RESOURCE_CONFIG,
    per_device_batch_size=4,
    max_eval_length=4096,
    max_docs_per_dataset=MAX_DOCS_PER_DATASET,
    max_doc_bytes=MAX_DOC_BYTES,
    wandb_tags=[
        *_COMMON_TAGS,
        "model_a=marin-community/marin-8b-base",
        "model_b=Qwen/Qwen3-8B-Base",
    ],
)

# Gemma uses a distinct tokenizer (SentencePiece, 256k vocab) with very different
# whitespace handling from Llama-3 / Qwen3. Useful for seeing whether apparent
# gaps on whitespace-sensitive slices (kern, ABC) track with tokenizer choice.
MARIN_VS_GEMMA2 = default_model_perplexity_gap(
    name="symbolic-notation-pilot-marin-8b-base-vs-gemma-2-9b",
    model_a=MARIN_MODEL,
    model_b=GapFinderModelConfig(
        checkpoint_path="google/gemma-2-9b",
        checkpoint_is_hf=True,
        tokenizer="google/gemma-2-9b",
    ),
    datasets=DATASETS,
    resource_config=RESOURCE_CONFIG,
    per_device_batch_size=4,
    max_eval_length=4096,
    max_docs_per_dataset=MAX_DOCS_PER_DATASET,
    max_doc_bytes=MAX_DOC_BYTES,
    wandb_tags=[
        *_COMMON_TAGS,
        "model_a=marin-community/marin-8b-base",
        "model_b=google/gemma-2-9b",
    ],
)


if __name__ == "__main__":
    executor_main(
        [MARIN_VS_LLAMA, MARIN_VS_QWEN3, MARIN_VS_GEMMA2],
        description="Game / music symbolic-notation pilot perplexity-gap report (issue #5062).",
    )
