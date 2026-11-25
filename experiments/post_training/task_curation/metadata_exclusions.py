# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Policy-excluded upstreams without a registered catalog population.

Rows carry identity metadata, typed claims and the upstream pin covered by the
ruling. Offline conversion evidence may remain archived for an excluded source;
these declarations register no conversion or selectable tasks.
"""

from dataclasses import dataclass

from finestore.rl.status_taxonomy import ExclusionClaim, ExclusionEvidence, ExclusionEvidenceKind, ExclusionReason


@dataclass(frozen=True)
class MetadataExclusion:
    """A source excluded by policy without any catalog population."""

    id: str
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
        id="Nemotron:Nemotron-RL-knowledge-web_search-mcqa",
        origin="Nemotron",
        upstream_dataset="nvidia/Nemotron-RL-knowledge-web_search-mcqa",
        upstream_url="https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-web_search-mcqa",
        dataset_id="nvidia/Nemotron-RL-knowledge-web_search-mcqa",
        dataset_revision="41028dbc9e06eb388ba99f9fb7a0078123b564c3",
        notes=(
            "Excluded under the 2026-11-23 third-party credential ruling; design issue "
            "https://github.com/marin-community/marin/issues/12082. The native Google Search server "
            "requires google_api_key and google_cx. No credentials or substitute search tool are provisioned. "
            "Offline conversion retained all 2,929 rows; the unchanged native verifier scored all correct "
            "boxed answers 1 and wrong boxed answers 0. Blank replies produced native parsed_option "
            "validation errors, preserved as errors. Offline evidence admits no tasks: "
            "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/nemo-gym-remaining/2026.11.24.1/"
            "native-web-search-conversion-v1/result.json "
            "(SHA-256 1c8770707d536e1728eb4c37a17416807c08ee43bd178b41db763b1e9476922e)."
        ),
        claims=(
            ExclusionClaim(
                reason=ExclusionReason.THIRD_PARTY_CREDENTIALS_REQUIRED,
                evidence=ExclusionEvidence(
                    kind=ExclusionEvidenceKind.RECEIPT,
                    uri="https://github.com/NVIDIA-NeMo/Gym/blob/"
                    "39d2bde61e426f93e4bc00c0255fc1d839f859e6/"
                    "resources_servers/google_search/configs/google_search.yaml",
                    sha256="bde849d0b28a4ba24bbb7fdb2c3c951f14c9bfc66727940acba322dbd7e8f848",
                    message=(
                        "The pinned config binds this dataset to the google_search resource server. "
                        "Its app.py requires google_api_key and google_cx and calls Google Custom Search JSON API; "
                        "app SHA-256 76f8ccd53e0346fdd791c283f6b5be2585f4efc61e9d57b8058ee19ddf919ceb. "
                        "No live credential provisioning is approved; design and clearing criteria are in #12082."
                    ),
                ),
            ),
        ),
    ),
    MetadataExclusion(
        id="GPUMODE:kernelbot-data",
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
    MetadataExclusion(
        id="TRL:kto_mix/argilla/distilabel-capybara-dpo-7k-binarized",
        origin="TRL",
        upstream_dataset="kto_mix",
        upstream_url="https://huggingface.co/datasets/trl-lib/kto-mix-14k",
        dataset_id="trl-lib/kto-mix-14k",
        dataset_revision="4470f033f33364e7d064c9f920c3df54d0cce767",
        notes=(
            "Main-sync landing (2026-11-25): origin/main declares this KTO component but the Atlas "
            "r39 cut does not carry it. The pinned upstream trl-lib/kto-mix-14k card declares no "
            "license anywhere (frontmatter, body or LICENSE file), so the doubt-means-ban ruling "
            "bans it; the component upstreams' grants (argilla cards) are lineage evidence, not a "
            "declared grant at this pin. Held out of the served declarations and out of intake; the "
            "ban lifts only with a recorded permissive grant at the pin (issue 11497)."
        ),
        claims=(
            _claim(
                ExclusionReason.LICENSE_MISSING,
                "Ban registry 2026.11.25.a-main-sync.1: no license declaration in the pinned "
                "upstream primary evidence; the standing doubt rule bans TRAIN/EVAL; issue 11497.",
                uri="https://huggingface.co/datasets/trl-lib/kto-mix-14k/resolve/4470f033f33364e7d064c9f920c3df54d0cce767/README.md",
            ),
        ),
    ),
    MetadataExclusion(
        id="TRL:kto_mix/argilla/distilabel-intel-orca-dpo-pairs",
        origin="TRL",
        upstream_dataset="kto_mix",
        upstream_url="https://huggingface.co/datasets/trl-lib/kto-mix-14k",
        dataset_id="trl-lib/kto-mix-14k",
        dataset_revision="4470f033f33364e7d064c9f920c3df54d0cce767",
        notes=(
            "Main-sync landing (2026-11-25): origin/main declares this KTO component but the Atlas "
            "r39 cut does not carry it. The pinned upstream trl-lib/kto-mix-14k card declares no "
            "license anywhere (frontmatter, body or LICENSE file), so the doubt-means-ban ruling "
            "bans it; the component upstreams' grants (argilla cards) are lineage evidence, not a "
            "declared grant at this pin. Held out of the served declarations and out of intake; the "
            "ban lifts only with a recorded permissive grant at the pin (issue 11497)."
        ),
        claims=(
            _claim(
                ExclusionReason.LICENSE_MISSING,
                "Ban registry 2026.11.25.a-main-sync.1: no license declaration in the pinned "
                "upstream primary evidence; the standing doubt rule bans TRAIN/EVAL; issue 11497.",
                uri="https://huggingface.co/datasets/trl-lib/kto-mix-14k/resolve/4470f033f33364e7d064c9f920c3df54d0cce767/README.md",
            ),
        ),
    ),
    MetadataExclusion(
        id="TRL:kto_mix/argilla/ultrafeedback-binarized-preferences-cleaned",
        origin="TRL",
        upstream_dataset="kto_mix",
        upstream_url="https://huggingface.co/datasets/trl-lib/kto-mix-14k",
        dataset_id="trl-lib/kto-mix-14k",
        dataset_revision="4470f033f33364e7d064c9f920c3df54d0cce767",
        notes=(
            "Main-sync landing (2026-11-25): origin/main declares this KTO component but the Atlas "
            "r39 cut does not carry it. The pinned upstream trl-lib/kto-mix-14k card declares no "
            "license anywhere (frontmatter, body or LICENSE file), so the doubt-means-ban ruling "
            "bans it; the component upstreams' grants (argilla cards) are lineage evidence, not a "
            "declared grant at this pin. Held out of the served declarations and out of intake; the "
            "ban lifts only with a recorded permissive grant at the pin (issue 11497)."
        ),
        claims=(
            _claim(
                ExclusionReason.LICENSE_MISSING,
                "Ban registry 2026.11.25.a-main-sync.1: no license declaration in the pinned "
                "upstream primary evidence; the standing doubt rule bans TRAIN/EVAL; issue 11497.",
                uri="https://huggingface.co/datasets/trl-lib/kto-mix-14k/resolve/4470f033f33364e7d064c9f920c3df54d0cce767/README.md",
            ),
        ),
    ),
)


def metadata_exclusions() -> dict[str, MetadataExclusion]:
    """Index the metadata-only exclusion rows by canonical source ID."""
    return {row.id: row for row in METADATA_EXCLUSIONS}
