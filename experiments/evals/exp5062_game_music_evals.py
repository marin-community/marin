# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue #5062 manifest-backed game/music long-tail eval slices."""

from __future__ import annotations

import posixpath

from marin.datakit.download.game_music_evals import (
    HfJsonTextStagingConfig,
    LichessPgnStagingConfig,
    stage_hf_json_text_source,
    stage_lichess_pgn_sample,
)
from marin.datakit.ingestion_manifest import (
    IdentityTreatment,
    IngestionPolicy,
    IngestionSourceManifest,
    SampleCapConfig,
    SecretRedaction,
    StagingMetadata,
    UsagePolicy,
)
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

ISSUE_5062 = 5062
EPIC_5005 = 5005
DEFAULT_LICHESS_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"
IRISHMAN_DATASET_ID = "sander-wood/irishman"
IRISHMAN_REVISION = "30902e69ca45266207f8466e0d04e4bc742c5604"
GAME_MUSIC_LICHESS_KEY = "game_music/lichess_pgn_2013_01"
GAME_MUSIC_IRISHMAN_KEY = "game_music/irishman_abc"


def _eval_only_policy(provenance_notes: str) -> IngestionPolicy:
    return IngestionPolicy(
        usage_policy=UsagePolicy.EVAL_ONLY,
        use_policy="Eval-only long-tail PPL probe. Do not mix into training.",
        requires_sanitization=False,
        identity_treatment=IdentityTreatment.PRESERVE,
        secret_redaction=SecretRedaction.NONE,
        contamination_risk="high: direct contamination if the held-out probe slice is copied into training data",
        provenance_notes=provenance_notes,
    )


LICHESS_PGN_2013_01_MANIFEST = IngestionSourceManifest(
    dataset_key="lichess/public",
    slice_key=GAME_MUSIC_LICHESS_KEY,
    source_label="lichess_pgn_2013_01",
    source_urls=("https://database.lichess.org/", DEFAULT_LICHESS_URL),
    source_license="CC0 1.0",
    source_format="pgn_zst",
    surface_form="pgn",
    policy=_eval_only_policy("Official Lichess public database sample, bounded to a small deterministic month slice."),
    staging=StagingMetadata(
        transform_name="stage_lichess_pgn_sample",
        metadata={
            "output_filename": "data.jsonl.gz",
            "provenance_fields": ["index", "source_url"],
        },
    ),
    epic_issue=EPIC_5005,
    issue_numbers=(ISSUE_5062,),
    sample_caps=SampleCapConfig(max_records=2048),
    compressed_size_bytes=17_761_302,
    source_metadata={"month": "2013-01"},
)

IRISHMAN_ABC_MANIFEST = IngestionSourceManifest(
    dataset_key=IRISHMAN_DATASET_ID,
    slice_key=GAME_MUSIC_IRISHMAN_KEY,
    source_label="irishman_abc",
    source_urls=(f"https://huggingface.co/datasets/{IRISHMAN_DATASET_ID}",),
    source_license="MIT / public-domain compositions",
    source_format="hf_json",
    surface_form="abc_notation",
    policy=_eval_only_policy(
        "IrishMAN validation split mirrored on Hugging Face; staged from a pinned dataset revision."
    ),
    staging=StagingMetadata(
        transform_name="stage_hf_json_text_source",
        split="validation",
        metadata={
            "output_filename": "data.jsonl.gz",
            "provenance_fields": ["dataset_id", "revision", "split_filename", "index"],
        },
    ),
    epic_issue=EPIC_5005,
    issue_numbers=(ISSUE_5062,),
    sample_caps=SampleCapConfig(max_examples=2048),
    compressed_size_bytes=796_938,
    source_metadata={"hf_revision": IRISHMAN_REVISION, "split_filename": "validation.json", "text_key": "abc notation"},
)

LICHESS_PGN_RAW = ExecutorStep(
    name="evaluation/game_music/lichess_pgn_2013_01",
    fn=stage_lichess_pgn_sample,
    config=LichessPgnStagingConfig(
        source_url=DEFAULT_LICHESS_URL,
        output_path=this_output_path(),
        source_label=LICHESS_PGN_2013_01_MANIFEST.source_label,
        max_records=LICHESS_PGN_2013_01_MANIFEST.sample_caps.max_records or 2048,
        source_manifest=LICHESS_PGN_2013_01_MANIFEST,
        content_fingerprint=LICHESS_PGN_2013_01_MANIFEST.fingerprint(),
    ),
)

IRISHMAN_ABC_RAW = ExecutorStep(
    name="evaluation/game_music/irishman_abc",
    fn=stage_hf_json_text_source,
    config=HfJsonTextStagingConfig(
        dataset_id=IRISHMAN_DATASET_ID,
        revision=IRISHMAN_REVISION,
        split_filename="validation.json",
        text_key="abc notation",
        output_path=this_output_path(),
        source_label=IRISHMAN_ABC_MANIFEST.source_label,
        max_examples=IRISHMAN_ABC_MANIFEST.sample_caps.max_examples or 2048,
        source_manifest=IRISHMAN_ABC_MANIFEST,
        content_fingerprint=IRISHMAN_ABC_MANIFEST.fingerprint(),
    ),
)


def game_music_raw_validation_sets(
    *,
    raw_root: str | None = None,
    lichess_raw: ExecutorStep | None = None,
    irishman_raw: ExecutorStep | None = None,
) -> dict[str, RawTextEvaluationDataset]:
    """Return first-pass manifest-backed game/music raw validation slices."""

    if raw_root is None and lichess_raw is None:
        lichess_raw = LICHESS_PGN_RAW
    if raw_root is None and irishman_raw is None:
        irishman_raw = IRISHMAN_ABC_RAW

    if raw_root is not None:
        lichess_source: str | ExecutorStep = posixpath.join(raw_root, "lichess/2013-01/data.jsonl.gz")
        irishman_source: str | ExecutorStep = posixpath.join(raw_root, "music/abc/irishman/data.jsonl.gz")
    else:
        assert lichess_raw is not None
        assert irishman_raw is not None
        lichess_source = lichess_raw.cd("data.jsonl.gz")
        irishman_source = irishman_raw.cd("data.jsonl.gz")

    return {
        GAME_MUSIC_LICHESS_KEY: raw_text_dataset(
            lichess_source,
            tags=(
                "game_music",
                "long_tail_ppl",
                f"epic:{EPIC_5005}",
                f"issue:{ISSUE_5062}",
                "source:lichess",
                "surface:pgn",
            ),
        ),
        GAME_MUSIC_IRISHMAN_KEY: raw_text_dataset(
            irishman_source,
            tags=(
                "game_music",
                "long_tail_ppl",
                f"epic:{EPIC_5005}",
                f"issue:{ISSUE_5062}",
                "source:irishman",
                "surface:abc_notation",
            ),
        ),
    }


if __name__ == "__main__":
    executor_main(steps=[LICHESS_PGN_RAW, IRISHMAN_ABC_RAW])
