# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HuggingFaceFW/finetranslations download + parallel-text transform + normalize.

FineTranslations is a parallel corpus: each row pairs ``og_full_text`` (the
original web text in its native language) with ``translated_text`` (a machine
English translation). For pretraining we fold both sides into a single document
so the model sees the same content in both languages.

The transform concatenates the two with a blank-line separator in a per-document
random order — in expectation half the documents read original→English and half
read English→original. The order is chosen deterministically from a content hash
so the step is reproducible across re-runs.
"""

from collections.abc import Iterator

import dupekit
from fray.types import ResourceConfig
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

FINETRANSLATIONS_HF_ID = "HuggingFaceFW/finetranslations"
FINETRANSLATIONS_REVISION = "af3f4ca"

SEPARATOR = "\n\n"

# Only the columns the fold needs; projected at read time so the upstream's
# heavy chunk/score/warc columns never leave parquet.
COLUMNS = ["id", "og_full_text", "translated_text"]


def fold_parallel_text(record: dict) -> Iterator[dict]:
    """Concatenate the English translation and the original text into one document.

    Yields a single ``{"text": ...}`` record. When both sides are present the
    English/original order is a deterministic per-document coin flip on a content
    hash, so in expectation half the corpus reads original→English and half
    English→original while staying reproducible. A row with only one side present
    is kept with just that side. Records with neither side present are dropped.
    """
    original = record.get("og_full_text") or ""
    english = record.get("translated_text") or ""
    if not original and not english:
        counters.pipeline.update_counter("finetranslations/empty", 1)
        return

    # Coin flip from the row id (falls back to the original text) so the chosen
    # order is stable across re-runs rather than depending on shard order. Empty
    # sides are dropped from the join, so single-sided rows pass through as-is.
    coin = record.get("id") or original
    if dupekit.hash_xxh3_128(coin.encode("utf-8")) & 1:
        counters.pipeline.update_counter("finetranslations/english_first", 1)
        parts = (english, original)
    else:
        counters.pipeline.update_counter("finetranslations/original_first", 1)
        parts = (original, english)

    counters.pipeline.update_counter("finetranslations/kept", 1)
    yield {"text": SEPARATOR.join(p for p in parts if p)}


def transform(input_path: str, output_path: str) -> None:
    """Fold each parallel row into a single-text-column parquet document."""
    pipeline = (
        Dataset.from_files(f"{input_path}/data/**/*.parquet")
        .load_parquet(columns=COLUMNS)
        .flat_map(fold_parallel_text)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="finetranslations-parallel", resources=ResourceConfig(cpu=2, ram="16g"))
    ctx.execute(pipeline)


def finetranslations_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the ``(download, fold, normalize)`` chain for finetranslations."""
    download = download_hf_step(
        "raw/finetranslations",
        hf_dataset_id=FINETRANSLATIONS_HF_ID,
        revision=FINETRANSLATIONS_REVISION,
        hf_urls_glob=["data/**/*.parquet"],
    )
    processed = StepSpec(
        name="processed/finetranslations",
        deps=[download],
        fn=lambda output_path: transform(input_path=download.output_path, output_path=output_path),
        hash_attrs={"version": "v1", "separator": SEPARATOR},
    )
    normalize = normalize_step(
        name="normalized/finetranslations",
        download=processed,
    )
    return (download, processed, normalize)
