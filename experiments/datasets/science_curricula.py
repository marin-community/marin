# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenized text handles for the licensed science-curriculum experiment."""

from fray.types import ResourceConfig

from marin.datakit.science_source_candidates import science_source_candidates
from marin.datakit.sources import all_sources
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache
from rigging.filesystem.storage_path import prefix_join

TOKENIZER = "gs://marin-us-central2/grug_sft/tokenizer/2026.09.12"
VERSION = "2026.09.20"
_TOKENIZE_RESOURCES = ResourceConfig(cpu=2, ram="32g", disk="5g")

BIOCOLLECTION_SOURCES = (
    "biocollection/free_text_stream",
    "biocollection/instruction_stream",
)


def science_curriculum_datasets() -> dict[str, ArtifactStep[TokenizedCache]]:
    """Return the new math and science text sources under stable training keys."""
    candidates = science_source_candidates()
    active = all_sources()
    selected = {name: active[name] for name in BIOCOLLECTION_SOURCES}
    selected.update(candidates)
    return {
        name: tokenized(
            f"grug_sft/science_text/{name}",
            tokenizer=TOKENIZER,
            version=VERSION,
            paths=[prefix_join(source.normalized.output_path, "outputs/main/*.parquet")],
            tags=["science-curriculum", f"source:{name}"],
            resources=_TOKENIZE_RESOURCES,
        )
        for name, source in selected.items()
    }


if __name__ == "__main__":
    dataset_main(science_curriculum_datasets())
