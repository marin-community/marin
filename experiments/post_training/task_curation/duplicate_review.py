# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measured advisory source-pair reviews; task identifiers never select partners."""

from collections.abc import Iterator

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.source_fingerprints import DuplicateReview, PairSimilarity

from experiments.post_training.task_curation.task_neighbors import SAMPLE_SIZE, PopulationIndex, compare_populations


def review_pairs(
    pairs: list[PairSimilarity],
    populations: dict[str, PopulationIndex],
    acknowledgements: dict[str, str],
    response_receipts: StoragePath,
    sample_size: int = SAMPLE_SIZE,
) -> Iterator[DuplicateReview]:
    """Match against both complete populations and checkpoint the measured evidence."""
    for pair in pairs:
        result = compare_populations(pair, populations[pair.left], populations[pair.right], sample_size)
        receipt = response_receipts / f"{result.prompt_sha256}.json"
        payload = result.model_dump_json().encode()
        if receipt.exists():
            if receipt.read_bytes() != payload:
                raise ValueError(f"Measured receipt already differs: {receipt}")
        else:
            with receipt.open("wb", auto_mkdir=True) as stream:
                stream.write(payload)
        yield result.model_copy(update={"acknowledgement": acknowledgements.get(result.prompt_sha256)})
        populations[pair.left].buckets = None
        populations[pair.right].buckets = None
