# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``marin.datakit.download.formal_methods_evals``.

These tests drive the downloader with synthetic in-repo archives served via ``file://`` URLs,
so no network access is required. The focus is on externally-observable behaviour:

* Raw file bytes are preserved verbatim (comments, long symbols, Unicode).
* The byte budget truncates the compressed JSONL output as requested.
* Include / exclude globs filter archive members correctly.
* ``jsonl_text_column`` extracts the configured text column.
* ``zip`` and ``tar.gz`` archive formats both round-trip.
"""

from __future__ import annotations

import gzip
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest
from marin.datakit.download.formal_methods_evals import (
    JSONL_TEXT_COLUMN_CONTENT_MODE,
    RAW_FILE_CONTENT_MODE,
    ArchiveSourceConfig,
    DownloadArchiveSliceConfig,
    archive_slice_step,
    download_archive_slice,
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

from experiments.evals.formal_methods_ppl import (
    FORMAL_METHODS_SOURCES,
    HARDWARE_RTL_SOURCES,
)

# A mix of SMT-LIB-style content exercising comments, long symbols, and Unicode.
SMT_FILE_WITH_COMMENTS = """; comment at top
(set-logic QF_LIA)
(declare-fun a_very_long_symbol_with_underscores_12345 () Int)
(assert (= a_very_long_symbol_with_underscores_12345 42))
(check-sat)
; status sat
"""

VERILOG_MODULE = """// line comment
module my_module #(parameter WIDTH = 8) (
    input  logic clk,
    input  logic rst_n,
    output logic [WIDTH-1:0] data_out
);
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) data_out <= '0;
    else       data_out <= data_out + 1;
  end
endmodule
"""

DIMACS_CNF = """c this is a DIMACS comment
p cnf 3 2
1 -2 0
2 3 0
"""

UNICODE_FILE = "π ⊕ ∀x. P(x) ≠ Q(x)\n"


@pytest.fixture
def make_zip_archive(tmp_path: Path):
    def _make(name: str, files: dict[str, str]) -> Path:
        archive_path = tmp_path / name
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for member_name, content in files.items():
                zf.writestr(member_name, content)
        return archive_path

    return _make


@pytest.fixture
def make_tar_gz_archive(tmp_path: Path):
    def _make(name: str, files: dict[str, str]) -> Path:
        archive_path = tmp_path / name
        with tarfile.open(archive_path, "w:gz") as tf:
            for member_name, content in files.items():
                data = content.encode("utf-8")
                info = tarfile.TarInfo(name=member_name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        return archive_path

    return _make


def _read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _test_source(
    *,
    slice_key: str,
    archive_url: str,
    archive_format: str,
    include_globs: tuple[str, ...],
    exclude_globs: tuple[str, ...] = (),
    content_mode: str = RAW_FILE_CONTENT_MODE,
    jsonl_text_column: str | None = None,
    max_compressed_bytes: int = 40 * 1024 * 1024,
    max_files: int | None = None,
) -> ArchiveSourceConfig:
    return ArchiveSourceConfig(
        manifest=IngestionSourceManifest(
            dataset_key=slice_key.replace("/", "_"),
            slice_key=slice_key,
            source_label=slice_key,
            source_urls=(archive_url,),
            source_license="test",
            source_format=f"{archive_format}_archive",
            surface_form="raw_file_text" if content_mode == RAW_FILE_CONTENT_MODE else "jsonl_text_column",
            policy=IngestionPolicy(
                usage_policy=UsagePolicy.EVAL_ONLY,
                use_policy="test",
                requires_sanitization=False,
                identity_treatment=IdentityTreatment.PRESERVE,
                secret_redaction=SecretRedaction.NONE,
            ),
            staging=StagingMetadata(
                transform_name="download_archive_slice",
                serializer_name="archive_member_jsonl",
                metadata={
                    "archive_format": archive_format,
                    "include_globs": list(include_globs),
                    "exclude_globs": list(exclude_globs),
                    "content_mode": content_mode,
                    "jsonl_text_column": jsonl_text_column,
                    "output_filename": "data.jsonl.gz",
                    "provenance_fields": ["id", "source", "filename"],
                },
            ),
            sample_caps=SampleCapConfig(
                max_bytes_per_source=max_compressed_bytes,
                max_files=max_files,
            ),
        )
    )


def test_zip_raw_file_mode_preserves_bytes(make_zip_archive, tmp_path: Path) -> None:
    archive = make_zip_archive(
        "smt.zip",
        {
            "z3/examples/a.smt2": SMT_FILE_WITH_COMMENTS,
            "z3/examples/unicode.smt2": UNICODE_FILE,
            "z3/README.md": "not an smt file",
        },
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    cfg = DownloadArchiveSliceConfig(
        source=_test_source(
            slice_key="formal_methods/smt_lib_test",
            archive_url=f"file://{archive}",
            archive_format="zip",
            include_globs=("*.smt2",),
        ),
        output_path=str(output_dir),
    )

    result = download_archive_slice(cfg)

    assert result["records"] == 2
    assert result["metadata_path"] == str(output_dir / "metadata.json")
    records = _read_jsonl_gz(output_dir / "data.jsonl.gz")
    texts_by_filename = {r["filename"]: r["text"] for r in records}
    # README excluded by glob.
    assert "z3/README.md" not in texts_by_filename
    # File bytes preserved verbatim: comments, long symbols, status markers, Unicode.
    assert texts_by_filename["z3/examples/a.smt2"] == SMT_FILE_WITH_COMMENTS
    assert texts_by_filename["z3/examples/unicode.smt2"] == UNICODE_FILE
    # Slice key is embedded in source label and id prefix.
    assert all(r["source"] == "formal_methods/smt_lib_test" for r in records)
    assert all(r["id"].startswith("formal_methods/smt_lib_test#") for r in records)
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_manifest"]["slice_key"] == "formal_methods/smt_lib_test"
    assert metadata["materialized_output"]["record_count"] == 2


def test_tar_gz_raw_file_mode_preserves_bytes(make_tar_gz_archive, tmp_path: Path) -> None:
    archive = make_tar_gz_archive(
        "tptp.tar.gz",
        {
            "TPTP-v8.2.0/Problems/GRP/GRP001-1.p": "% TPTP problem\nfof(ax1, axiom, (p(X) | q(X))).\n",
            "TPTP-v8.2.0/Problems/LAT/LAT042+1.p": "% another problem\n",
            "TPTP-v8.2.0/README": "not a problem file",
        },
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    cfg = DownloadArchiveSliceConfig(
        source=_test_source(
            slice_key="formal_methods/tptp_test",
            archive_url=f"file://{archive}",
            archive_format="tar.gz",
            include_globs=("*.p",),
        ),
        output_path=str(output_dir),
    )

    result = download_archive_slice(cfg)

    assert result["records"] == 2
    records = _read_jsonl_gz(output_dir / "data.jsonl.gz")
    filenames = {r["filename"] for r in records}
    assert filenames == {"TPTP-v8.2.0/Problems/GRP/GRP001-1.p", "TPTP-v8.2.0/Problems/LAT/LAT042+1.p"}


def test_exclude_globs_drop_members(make_zip_archive, tmp_path: Path) -> None:
    archive = make_zip_archive(
        "coqgym.zip",
        {
            "CoqGym-master/proofs/core/lemma.v": "Lemma foo : True. Proof. trivial. Qed.",
            "CoqGym-master/node_modules/vendored.v": "should be excluded",
            "CoqGym-master/.git/index": "definitely not a proof",
        },
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    cfg = DownloadArchiveSliceConfig(
        source=_test_source(
            slice_key="formal_methods/coqgym_test",
            archive_url=f"file://{archive}",
            archive_format="zip",
            include_globs=("*.v",),
            exclude_globs=("*/node_modules/*", "*/.git/*"),
        ),
        output_path=str(output_dir),
    )

    download_archive_slice(cfg)
    records = _read_jsonl_gz(output_dir / "data.jsonl.gz")
    filenames = {r["filename"] for r in records}
    assert filenames == {"CoqGym-master/proofs/core/lemma.v"}


def test_byte_budget_truncates(make_zip_archive, tmp_path: Path) -> None:
    # ~500 files of 2KB each => ~1MB raw before gzip; budget of 4KB compressed forces truncation.
    large_file_count = 500
    files = {f"bench/file_{i:04d}.cnf": DIMACS_CNF * 200 for i in range(large_file_count)}
    archive = make_zip_archive("dimacs.zip", files)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    cfg = DownloadArchiveSliceConfig(
        source=_test_source(
            slice_key="formal_methods/dimacs_test",
            archive_url=f"file://{archive}",
            archive_format="zip",
            include_globs=("*.cnf",),
            max_compressed_bytes=4_096,
        ),
        output_path=str(output_dir),
    )

    result = download_archive_slice(cfg)

    # Budget was reached, so we wrote fewer records than archive members.
    assert result["records"] < large_file_count
    # The downloader stops before appending a record that would cross the compressed-byte budget.
    assert result["compressed_bytes"] <= 4_096


def test_max_files_cap(make_zip_archive, tmp_path: Path) -> None:
    files = {f"mod_{i}.v": VERILOG_MODULE for i in range(10)}
    archive = make_zip_archive("verilog.zip", files)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    cfg = DownloadArchiveSliceConfig(
        source=_test_source(
            slice_key="hardware_rtl/verilog_eval_test",
            archive_url=f"file://{archive}",
            archive_format="zip",
            include_globs=("*.v",),
            max_files=3,
        ),
        output_path=str(output_dir),
    )

    result = download_archive_slice(cfg)
    assert result["records"] == 3


def test_jsonl_text_column_mode(make_zip_archive, tmp_path: Path) -> None:
    jsonl_content = (
        json.dumps({"code": "module a(); endmodule", "id": 1})
        + "\n"
        + json.dumps({"code": "module b(); endmodule", "id": 2})
        + "\n"
        + json.dumps({"id": 3})  # missing text column; skipped
        + "\n"
    )
    archive = make_zip_archive(
        "rtl_coder.zip",
        {"RTL-Coder-main/data/train.jsonl": jsonl_content},
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    cfg = DownloadArchiveSliceConfig(
        source=_test_source(
            slice_key="hardware_rtl/rtl_coder_test",
            archive_url=f"file://{archive}",
            archive_format="zip",
            include_globs=("*.jsonl",),
            content_mode=JSONL_TEXT_COLUMN_CONTENT_MODE,
            jsonl_text_column="code",
        ),
        output_path=str(output_dir),
    )

    download_archive_slice(cfg)
    records = _read_jsonl_gz(output_dir / "data.jsonl.gz")
    assert [r["text"] for r in records] == [
        "module a(); endmodule",
        "module b(); endmodule",
    ]


def test_json_array_and_list_text_column_mode(make_zip_archive, tmp_path: Path) -> None:
    json_content = json.dumps(
        [
            {"Response": ["module a(); endmodule"]},
            {"Response": ["module b(); endmodule", "module c(); endmodule"]},
            {"Response": []},
        ]
    )
    archive = make_zip_archive(
        "rtl_coder.zip",
        {"RTL-Coder-main/dataset/Resyn27k.json": json_content},
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    cfg = DownloadArchiveSliceConfig(
        source=_test_source(
            slice_key="hardware_rtl/rtl_coder_test",
            archive_url=f"file://{archive}",
            archive_format="zip",
            include_globs=("*.json",),
            content_mode=JSONL_TEXT_COLUMN_CONTENT_MODE,
            jsonl_text_column="Response",
        ),
        output_path=str(output_dir),
    )

    result = download_archive_slice(cfg)

    assert result["records"] == 3
    records = _read_jsonl_gz(output_dir / "data.jsonl.gz")
    assert [r["text"] for r in records] == [
        "module a(); endmodule",
        "module b(); endmodule",
        "module c(); endmodule",
    ]
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["materialized_output"]["metadata"]["content_mode"] == JSONL_TEXT_COLUMN_CONTENT_MODE
    assert metadata["materialized_output"]["metadata"]["jsonl_text_column"] == "Response"


def test_jsonl_text_column_mode_rejects_malformed_json(make_zip_archive, tmp_path: Path) -> None:
    archive = make_zip_archive(
        "broken.zip",
        {"broken.jsonl": '{"code": "module a(); endmodule"}\n{"code": [not valid json]\n'},
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    cfg = DownloadArchiveSliceConfig(
        source=_test_source(
            slice_key="hardware_rtl/broken_test",
            archive_url=f"file://{archive}",
            archive_format="zip",
            include_globs=("*.jsonl",),
            content_mode=JSONL_TEXT_COLUMN_CONTENT_MODE,
            jsonl_text_column="code",
        ),
        output_path=str(output_dir),
    )

    with pytest.raises(ValueError, match="malformed JSONL line 2"):
        download_archive_slice(cfg)


def test_validate_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="unsupported archive_format"):
        _test_source(
            slice_key="x/y",
            archive_url="file:///nowhere",
            archive_format="rar",
            include_globs=("*",),
        ).validate()


def test_validate_requires_jsonl_text_column() -> None:
    with pytest.raises(ValueError, match="jsonl_text_column"):
        _test_source(
            slice_key="x/y",
            archive_url="file:///nowhere",
            archive_format="zip",
            include_globs=("*.jsonl",),
            content_mode=JSONL_TEXT_COLUMN_CONTENT_MODE,
        ).validate()


def test_validate_requires_non_empty_include_globs() -> None:
    with pytest.raises(ValueError, match="include_globs"):
        _test_source(
            slice_key="x/y",
            archive_url="file:///nowhere",
            archive_format="zip",
            include_globs=(),
        ).validate()


def test_archive_slice_step_has_deterministic_name() -> None:
    source = _test_source(
        slice_key="formal_methods/smt_lib",
        archive_url="file:///tmp/unused",
        archive_format="zip",
        include_globs=("*.smt2",),
    )
    step = archive_slice_step(source)
    assert step.name == "raw/formal_methods/smt_lib"


def test_exp5060_sources_match_expected_formats() -> None:
    source_by_key = {source.slice_key: source for source in (*FORMAL_METHODS_SOURCES, *HARDWARE_RTL_SOURCES)}

    assert source_by_key["formal_methods/tptp"].archive_url == "https://tptp.org/TPTP/Archive/TPTP-v8.2.0.tgz"
    assert source_by_key["formal_methods/dimacs_cnf"].archive_format == "tar.gz"
    assert source_by_key["formal_methods/dimacs_cnf"].include_globs == ("*.cnf",)
    assert "refs/heads" not in source_by_key["formal_methods/smt_lib"].archive_url
    assert "refs/heads" not in source_by_key["formal_methods/coqgym"].archive_url
    assert source_by_key["formal_methods/smt_lib"].manifest.staging.transform_name == "download_archive_slice"
    assert source_by_key["formal_methods/smt_lib"].manifest.policy.eval_only is True

    rtl_repo = source_by_key["hardware_rtl/rtl_repo"]
    assert rtl_repo.content_mode == "jsonl_text_column"
    assert rtl_repo.jsonl_text_column == "label"
    assert rtl_repo.include_globs == ("predictions/*.jsonl",)
    assert "refs/heads" not in rtl_repo.archive_url

    rtl_coder = source_by_key["hardware_rtl/rtl_coder"]
    assert rtl_coder.content_mode == "jsonl_text_column"
    assert rtl_coder.jsonl_text_column == "Response"
    assert rtl_coder.include_globs == ("dataset/*.json", "data_generation/data_sample.json")
    assert "refs/heads" not in rtl_coder.archive_url
