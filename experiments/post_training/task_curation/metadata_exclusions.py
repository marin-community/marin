# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declaration-only metadata rows for known-but-rejected upstreams.

These sources have no catalog conversion and never will from their current
pins: the user has rejected them. They are published as Excluded rows with
typed claims and evidence so the Atlas shows the rejection instead of silence.
A metadata row never downloads, converts, hosts or mirrors upstream data; it
carries only identity metadata, its exclusion claims and the upstream pin the
ruling was made against.
"""

from dataclasses import dataclass

from finestore.rl.status_taxonomy import ExclusionClaim, ExclusionEvidence, ExclusionEvidenceKind, ExclusionReason


@dataclass(frozen=True)
class MetadataExclusion:
    """A source excluded by policy without any catalog population."""

    id: str
    name: str
    display_name: str
    origin: str
    upstream_dataset: str
    upstream_url: str
    dataset_id: str
    dataset_revision: str
    notes: str
    claims: tuple[ExclusionClaim, ...]


def _claim(reason: ExclusionReason, message: str, uri: str | None = None) -> ExclusionClaim:
    return ExclusionClaim(
        reason=reason,
        evidence=ExclusionEvidence(
            kind=ExclusionEvidenceKind.RECEIPT if uri is not None else ExclusionEvidenceKind.NOTE,
            message=message,
            uri=uri,
        ),
    )


CYBENCH_PIN = "ca483c6443f17a20ba7d6bac3d274859a1721e51"
HAL_HARNESS_PIN = "16bb03ebc11577fb5ea6dc8bb6c968387085e6aa"
KERNELBOT_PIN = "4159cf6b2c6bab208be6dda885d6d87631cc16df"
MAI_GYM_PIN = "dc279a3735f0d59f9b9b9806f1fc8cd99c28e6f9"
MIROVERSE_PIN = "11fbf325f6eda37b34f519d1ba7268f40f469ac6"

METADATA_EXCLUSIONS: tuple[MetadataExclusion, ...] = (
    MetadataExclusion(
        id="GPUMODE:kernelbot-data",
        name="kernelbot-data",
        display_name="GPUMODE · kernelbot-data",
        origin="GPUMODE",
        upstream_dataset="GPUMODE/kernelbot-data",
        upstream_url="https://huggingface.co/datasets/GPUMODE/kernelbot-data",
        dataset_id="GPUMODE/kernelbot-data",
        dataset_revision=KERNELBOT_PIN,
        notes=(
            "Known-but-rejected upstream; declaration metadata only. The dataset is never "
            "downloaded, converted, hosted or mirrored."
        ),
        claims=(
            _claim(
                ExclusionReason.LICENSE_BAN,
                "Ban registry 2026.11.16.1 at pin " + KERNELBOT_PIN + ": user rejected the required "
                "model-license reciprocity condition for plain OpenMDW-1.1 releases; checker rule E1 "
                'ESCALATE, LICENSE quote "June 9 Researcher Reciprocity License"; issue 11497.',
                uri="https://huggingface.co/datasets/GPUMODE/kernelbot-data/blob/" + KERNELBOT_PIN + "/LICENSE",
            ),
        ),
    ),
    MetadataExclusion(
        id="CyBench:andyzorigin-cybench",
        name="andyzorigin-cybench",
        display_name="CyBench · andyzorigin/cybench",
        origin="CyBench",
        upstream_dataset="andyzorigin/cybench",
        upstream_url="https://github.com/andyzorigin/cybench",
        dataset_id="andyzorigin/cybench",
        dataset_revision=CYBENCH_PIN,
        notes=(
            "Known-but-rejected upstream; declaration metadata only. The dataset is never "
            "downloaded, converted, hosted or mirrored."
        ),
        claims=(
            _claim(
                ExclusionReason.LICENSE_BAN,
                "Checker receipt F1 FAIL at pin " + CYBENCH_PIN + ": the harness is Apache-2.0 but the "
                "benchmark/project-sekai-ctf/sekaictf-2022 task content is CC BY-NC-SA 4.0 (quoted from "
                "its LICENSE.content.md); issue 11497.",
                uri="https://github.com/andyzorigin/cybench/blob/"
                + CYBENCH_PIN
                + "/benchmark/project-sekai-ctf/sekaictf-2022/LICENSE.content.md",
            ),
        ),
    ),
    MetadataExclusion(
        id="HAL:princeton-pli-hal-harness",
        name="princeton-pli-hal-harness",
        display_name="HAL harness · princeton-pli/hal-harness",
        origin="HAL",
        upstream_dataset="princeton-pli/hal-harness",
        upstream_url="https://github.com/princeton-pli/hal-harness",
        dataset_id="princeton-pli/hal-harness",
        dataset_revision=HAL_HARNESS_PIN,
        notes=(
            "Known-but-rejected upstream; declaration metadata only. The dataset is never "
            "downloaded, converted, hosted or mirrored."
        ),
        claims=(
            _claim(
                ExclusionReason.LICENSE_MISSING,
                "Checker receipt F2 FAIL at pin " + HAL_HARNESS_PIN + ": no LICENSE or COPYING file in "
                "the pinned repository tree; issue 11497.",
                uri="https://api.github.com/repos/princeton-pli/hal-harness/git/trees/"
                + HAL_HARNESS_PIN
                + "?recursive=1",
            ),
        ),
    ),
    MetadataExclusion(
        id="MiroVerse:miromind-ai-MiroVerse-v0.1",
        name="miromind-ai-MiroVerse-v0.1",
        display_name="MiroVerse · miromind-ai/MiroVerse-v0.1",
        origin="MiroVerse",
        upstream_dataset="miromind-ai/MiroVerse-v0.1",
        upstream_url="https://huggingface.co/datasets/miromind-ai/MiroVerse-v0.1",
        dataset_id="miromind-ai/MiroVerse-v0.1",
        dataset_revision=MIROVERSE_PIN,
        notes=(
            "Known-but-rejected upstream; declaration metadata only. The dataset is never "
            "downloaded, converted, hosted or mirrored."
        ),
        claims=(
            _claim(
                ExclusionReason.LICENSE_BAN,
                "Checker receipt F1 FAIL at pin " + MIROVERSE_PIN + ": dataset card declares "
                "license cc-by-nc-4.0; issue 11497.",
                uri="https://huggingface.co/datasets/miromind-ai/MiroVerse-v0.1/resolve/" + MIROVERSE_PIN + "/README.md",
            ),
        ),
    ),
    MetadataExclusion(
        id="MAI-Gym:tasks",
        name="MAI-Gym-tasks",
        display_name="MAI-Gym · tasks",
        origin="MAI-Gym",
        upstream_dataset="Tongyi-MAI/MAI-Gym",
        upstream_url="https://github.com/Tongyi-MAI/MAI-Gym",
        dataset_id="Tongyi-MAI/MAI-Gym",
        dataset_revision=MAI_GYM_PIN,
        notes=(
            "Known-but-rejected upstream; declaration metadata only. The dataset is never "
            "downloaded, converted, hosted or mirrored."
        ),
        claims=(
            _claim(
                ExclusionReason.LICENSE_BAN,
                "Ban registry 2026.11.16.1 at pin " + MAI_GYM_PIN + " (classification missing): no "
                "license grant for the apps or task data at the pinned repository; issue 11497 and "
                "issue 10231 discussion.",
                uri="https://github.com/Tongyi-MAI/MAI-Gym/tree/" + MAI_GYM_PIN,
            ),
        ),
    ),
    MetadataExclusion(
        id="OpenGCUA:trajectories",
        name="OpenGCUA-trajectories",
        display_name="OpenGCUA · trajectories",
        origin="OpenGCUA",
        upstream_dataset="https://tongyi-mai.github.io/OpenGCUA/trajectories.html",
        upstream_url="https://tongyi-mai.github.io/OpenGCUA/trajectories.html",
        dataset_id="https://tongyi-mai.github.io/OpenGCUA/trajectories.html",
        dataset_revision="no published trajectory pin",
        notes=(
            "Known-but-rejected upstream; declaration metadata only. The dataset is never "
            "downloaded, converted, hosted or mirrored."
        ),
        claims=(
            _claim(
                ExclusionReason.NOT_RL_TASK_DATA,
                "Upstream has no RL task data; harness/trajectories only. CANDIDATES STATE.md "
                "(new-rl-data-ingest): no RL task data to ingest. The code is an Apache-2.0 GUI "
                "agent harness, not a task set. Its trajectories are unlicensed (ROADMAP: "
                "licensing 'will be documented') and are mostly closed-model outputs, so they "
                "carry provider output terms. Don't train on them.",
                uri="https://github.com/Tongyi-MAI/OpenGCUA",
            ),
            _claim(
                ExclusionReason.LICENSE_BAN,
                "Ban registry 2026.11.16.1: no trajectory-data license; Apache-2.0 applies to the "
                "harness code; issue 11497.",
                uri="https://github.com/Tongyi-MAI/OpenGCUA/tree/55c33aea792b48b9c1330a514bfcc8f6d3ecc704",
            ),
        ),
    ),
)


def metadata_exclusions() -> dict[str, MetadataExclusion]:
    """Index the metadata-only exclusion rows by canonical source ID."""
    return {row.id: row for row in METADATA_EXCLUSIONS}
