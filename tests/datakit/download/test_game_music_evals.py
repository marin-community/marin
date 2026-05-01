# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
from pathlib import Path

import marin.datakit.download.game_music_evals as game_music_evals
import pytest
import zstandard
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


class _FakeResponse:
    def __init__(self, *, raw_bytes: bytes | None = None, json_payload: object | None = None):
        self.raw = io.BytesIO(raw_bytes or b"")
        self.raw.decode_content = False
        self._json_payload = json_payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._json_payload

    def iter_lines(self, *, decode_unicode: bool = False):
        for line in self.raw.getvalue().splitlines():
            yield line.decode("utf-8") if decode_unicode else line

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeSession:
    def __init__(self, responses: dict[str, _FakeResponse]):
        self._responses = responses

    def get(self, url: str, *, timeout: int, stream: bool = False):
        return self._responses[url]

    def close(self) -> None:
        return None


def _source_manifest(
    *,
    dataset_key: str,
    slice_key: str,
    source_label: str,
    source_url: str,
    source_format: str,
    surface_form: str,
    max_records: int,
) -> IngestionSourceManifest:
    return IngestionSourceManifest(
        dataset_key=dataset_key,
        slice_key=slice_key,
        source_label=source_label,
        source_urls=(source_url,),
        source_license="eval-only",
        source_format=source_format,
        surface_form=surface_form,
        policy=IngestionPolicy(
            usage_policy=UsagePolicy.EVAL_ONLY,
            use_policy="Eval-only long-tail slice.",
            requires_sanitization=False,
            identity_treatment=IdentityTreatment.PRESERVE,
            secret_redaction=SecretRedaction.NONE,
            contamination_risk="high: held-out eval slice",
            provenance_notes="Test fixture",
        ),
        staging=StagingMetadata(transform_name="test_fixture", metadata={"provenance_fields": ["index"]}),
        epic_issue=5005,
        issue_numbers=(5062,),
        sample_caps=SampleCapConfig(max_records=max_records),
    )


def test_stage_lichess_pgn_sample_preserves_symbolic_text_and_writes_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, read_jsonl_gz
) -> None:
    source_url = "https://example.test/lichess_db_standard_rated_2013-01.pgn.zst"
    game_one = """[Event "Rated Blitz game"]
[Site "https://lichess.org/abcd1234"]
[Date "2013.01.01"]
[White "alice"]
[Black "bob"]
[Result "1-0"]

1. e4 {comment} e5 2. Nf3 Nc6 3. Bb5 a6 (3... Nf6) 4. Ba4 Nf6 1-0
"""
    game_two = """[Event "Rated Rapid game"]
[Site "https://lichess.org/wxyz9876"]
[Date "2013.01.02"]
[White "carol"]
[Black "dave"]
[Result "0-1"]

1. d4 d5 2. c4 e6 3. Nc3 Be7 $1 0-1
"""
    compressed = zstandard.ZstdCompressor().compress(f"{game_one}\n{game_two}\n".encode())
    monkeypatch.setattr(
        game_music_evals,
        "_build_session",
        lambda: _FakeSession({source_url: _FakeResponse(raw_bytes=compressed)}),
    )

    manifest = _source_manifest(
        dataset_key="lichess/public",
        slice_key="game_music/lichess_pgn_2013_01",
        source_label="lichess_pgn_2013_01",
        source_url=source_url,
        source_format="pgn_zst",
        surface_form="pgn",
        max_records=1,
    )
    output_dir = tmp_path / "output"

    result = stage_lichess_pgn_sample(
        LichessPgnStagingConfig(
            source_url=source_url,
            output_path=str(output_dir),
            source_label=manifest.source_label,
            max_records=1,
            source_manifest=manifest,
            manifest_fingerprint=manifest.fingerprint(),
        )
    )

    records = read_jsonl_gz(output_dir / "data.jsonl.gz")
    assert len(records) == 1
    assert records[0]["id"] == "abcd1234"
    assert records[0]["source"] == "lichess_pgn_2013_01"
    assert records[0]["text"] == game_one.rstrip("\n")
    assert "{comment}" in records[0]["text"]
    assert "(3... Nf6)" in records[0]["text"]
    assert records[0]["provenance"]["index"] == 0

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["manifest_fingerprint"] == manifest.provenance_fingerprint()
    assert metadata["content_fingerprint"] == manifest.fingerprint()
    assert metadata["materialized_output"]["record_count"] == 1
    assert metadata["materialized_output"]["metadata"]["source_url"] == source_url
    assert result["output_file"].endswith("data.jsonl.gz")


def test_stage_hf_json_text_source_preserves_abc_notation_and_caps_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, read_jsonl_gz
) -> None:
    source_url = "https://example.test/validation.json"
    records = [
        {
            "abc notation": "X:1\nT:First Tune\nM:6/8\nK:Bb\n|: B2d c2f :|",
            "control code": "S:2\nB:9\n",
        },
        {
            "abc notation": "X:2\nT:Second Tune\nM:4/4\nK:G\nGABc d2d2 ||",
            "control code": "S:3\nB:5\n",
        },
    ]
    monkeypatch.setattr(
        game_music_evals,
        "_build_session",
        lambda: _FakeSession({source_url: _FakeResponse(json_payload=records)}),
    )

    manifest = _source_manifest(
        dataset_key="irishman/public",
        slice_key="game_music/irishman_abc",
        source_label="irishman_abc",
        source_url=source_url,
        source_format="hf_json",
        surface_form="abc_notation",
        max_records=1,
    )
    output_dir = tmp_path / "output"

    result = stage_hf_json_text_source(
        HfJsonTextStagingConfig(
            dataset_id="sander-wood/irishman",
            revision="test-sha",
            split_filename="validation.json",
            text_key="abc notation",
            output_path=str(output_dir),
            source_label=manifest.source_label,
            max_examples=1,
            source_manifest=manifest,
            manifest_fingerprint=manifest.fingerprint(),
            source_file_url_override=source_url,
        )
    )

    staged = read_jsonl_gz(output_dir / "data.jsonl.gz")
    assert len(staged) == 1
    assert staged[0]["id"] == "irishman_abc:00000000"
    assert staged[0]["text"] == records[0]["abc notation"]
    assert staged[0]["provenance"]["dataset_id"] == "sander-wood/irishman"
    assert staged[0]["provenance"]["split_filename"] == "validation.json"
    assert "T:First Tune" in staged[0]["text"]
    assert "Second Tune" not in staged[0]["text"]

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["materialized_output"]["record_count"] == 1
    assert metadata["materialized_output"]["metadata"]["text_key"] == "abc notation"
    assert result["bytes_written"] == len(records[0]["abc notation"].encode("utf-8"))
