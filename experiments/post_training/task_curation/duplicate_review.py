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
        assert result.measurement is not None
        receipt = response_receipts / f"{result.prompt_sha256}.json"
        documents = [(receipt, result.model_dump_json())]
        directions = {}
        for side, direction in (
            ("left_to_right", result.measurement.left_to_right),
            ("right_to_left", result.measurement.right_to_left),
        ):
            path = response_receipts / f"{result.prompt_sha256}-{side}.json"
            documents.append((path, direction.model_dump_json()))
            directions[side] = direction.model_copy(
                update={"neighbors": direction.neighbors[:20], "neighbor_receipts_uri": str(path)}
            )
        for path, encoded in documents:
            payload = encoded.encode()
            if path.exists():
                if path.read_bytes() != payload:
                    raise ValueError(f"Measured receipt already differs: {path}")
            else:
                with path.open("wb", auto_mkdir=True) as stream:
                    stream.write(payload)
        yield result.model_copy(
            update={
                "acknowledgement": acknowledgements.get(result.prompt_sha256),
                "measurement": result.measurement.model_copy(update=directions),
            }
        )
