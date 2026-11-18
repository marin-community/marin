# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source pins recorded by the Exp4 curriculum."""

from experiments.post_training.task_curation.datasets.long_loop_rl import (
    AGENTIC_MANIFEST_SHA256,
    AGENTIC_RELEASE_URL,
    NONAGENTIC_MANIFEST_SHA256,
    NONAGENTIC_RELEASE_URL,
)
from experiments.post_training.task_curation.exp4_curriculum import (
    NEMOTRON_IF_LABEL,
    exp4_rl_curriculum,
)
from experiments.post_training.task_curation.sources import all_sources


def test_the_entries_pin_the_consumed_releases_and_the_new_source():
    curriculum = exp4_rl_curriculum()
    catalog = all_sources()
    for entry in curriculum.entries:
        if entry.label == NEMOTRON_IF_LABEL:
            source = catalog["MarinSkyRL:nemotron_if"]
            assert entry.source == NEMOTRON_IF_LABEL
            assert entry.pin.name == "nvidia/Llama-Nemotron-Post-Training-Dataset"
            assert entry.pin.revision == source.dataset.revision
        else:
            release_pin = AGENTIC_RELEASE_URL if entry.source == "long-loop-rl-agentic-train" else NONAGENTIC_RELEASE_URL
            release_revision = (
                AGENTIC_MANIFEST_SHA256 if entry.source == "long-loop-rl-agentic-train" else NONAGENTIC_MANIFEST_SHA256
            )
            assert entry.pin.url == release_pin
            assert entry.pin.revision == release_revision
