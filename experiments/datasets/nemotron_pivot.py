# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA's Nemotron agentic pivot releases, laid out as one candidate per expert turn.

Each pinned release is downloaded as-is, then :mod:`marin.rl.nemotron_pivot` keeps every
trajectory's longest row, emits a candidate for each expert turn in it, drops NVIDIA's pass-rate
fields, and holds out whole SWE instances or Terminal tasks for validation by hash bucket. Each
processed artifact holds ``candidates.parquet`` and ``validation.parquet``.
"""

from dataclasses import dataclass

from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.experiment.data import hf_download
from marin.rl.nemotron_pivot import Holdout, prepare_swe_candidates, prepare_terminal_candidates


@dataclass(frozen=True)
class NemotronPivotRelease:
    name: str
    hf_id: str
    revision: str
    filename: str


SWE_PIVOT_RELEASE = NemotronPivotRelease(
    name="raw/nvidia-nemotron-rl-agentic-swe-pivot-v1",
    hf_id="nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1",
    revision="4947a3c8ea803413a65f9eca14a96ef521b2ddf5",
    filename="train.jsonl",
)

TERMINAL_PIVOT_RELEASE = NemotronPivotRelease(
    name="raw/nvidia-nemotron-rl-agentic-terminal-pivot-v1",
    hf_id="nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1",
    revision="eaef26944643644c8a3dbbf361ce6128142f5976",
    filename="atcb_terminal_pivot_release_final_v2.jsonl",
)


def nemotron_pivot_release(release: NemotronPivotRelease) -> ArtifactStep[Artifact]:
    """The pinned release file, downloaded unchanged."""
    return hf_download(
        release.name,
        hf_id=release.hf_id,
        revision=release.revision,
        version="2026.10.09",
        urls_glob=(release.filename,),
    )


def nemotron_pivot_datasets() -> dict[str, ArtifactStep[Artifact]]:
    """Long-format candidates keyed by release: ``swe`` (73k training rows) and ``terminal`` (36k)."""
    # Validation: 100 SWE instances (4%, 508 rows) and 30 Terminal tasks (5%, 503 rows), four
    # evenly spaced turns of each held-out trajectory.
    releases = {
        "swe": (SWE_PIVOT_RELEASE, prepare_swe_candidates, Holdout(buckets=400, turns_per_trajectory=4)),
        "terminal": (TERMINAL_PIVOT_RELEASE, prepare_terminal_candidates, Holdout(buckets=500, turns_per_trajectory=4)),
    }
    return {
        key: apply(
            f"processed/nemotron-pivot-{key}-long",
            remote(prepare, resources=ResourceConfig.with_cpu(cpu=4, ram="32g")),
            version="2026.10.09",
            release_path=nemotron_pivot_release(release),
            release_filename=release.filename,
            output_path=OUT,
            holdout=holdout,
        )
        for key, (release, prepare, holdout) in releases.items()
    }
