# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import zstandard

from marin.datakit.ingestion_manifest import (
    IdentityTreatment,
    IngestionPolicy,
    IngestionSourceManifest,
    SampleCapConfig,
    SecretRedaction,
    StagingMetadata,
    UsagePolicy,
)
from marin.datakit.download.game_music_evals import (
    HfJsonTextStagingConfig,
    LichessPgnStagingConfig,
    stage_hf_json_text_source,
    stage_lichess_pgn_sample,
)


@pytest.fixture()
def local_http_server(tmp_path: Path):
    server_root = tmp_path / "server"
    server_root.mkdir()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(server_root), **kwargs)

        def log_message(self, format, *args):  # noqa: A002  # stdlib signature
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = httpd.server_address
        yield f"http://{host}:{port}", server_root
    finally:
        httpd.shutdown()
        thread.join()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_zstd(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed = zstandard.ZstdCompressor().compress(content.encode("utf-8"))
    path.write_bytes(compressed)


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
        staging=StagingMetadata(
            transform_name="test_fixture",
            output_filename="data.jsonl.gz",
            record_provenance_fields=("index",),
        ),
        epic_issue=5005,
        issue_numbers=(5062,),
        sample_caps=SampleCapConfig(max_records=max_records),
    )


def test_stage_lichess_pgn_sample_preserves_symbolic_text_and_writes_metadata(
    tmp_path: Path,
    local_http_server,
    read_jsonl_gz,
) -> None:
    base_url, server_root = local_http_server
    source_url = f"{base_url}/lichess_db_standard_rated_2013-01.pgn.zst"
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
    _write_zstd(server_root / "lichess_db_standard_rated_2013-01.pgn.zst", f"{game_one}\n{game_two}\n")

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
    assert metadata["manifest_fingerprint"] == manifest.fingerprint()
    assert metadata["materialized_output"]["record_count"] == 1
    assert metadata["materialized_output"]["metadata"]["source_url"] == source_url
    assert result["output_file"].endswith("data.jsonl.gz")


def test_stage_hf_json_text_source_preserves_abc_notation_and_caps_examples(
    tmp_path: Path,
    local_http_server,
    read_jsonl_gz,
) -> None:
    base_url, server_root = local_http_server
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
    _write_text(server_root / "validation.json", json.dumps(records))
    manifest = _source_manifest(
        dataset_key="irishman/public",
        slice_key="game_music/irishman_abc",
        source_label="irishman_abc",
        source_url=f"{base_url}/validation.json",
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
            source_file_url_override=f"{base_url}/validation.json",
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
