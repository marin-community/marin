# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic security-artifact text renderers."""

from __future__ import annotations

import pytest
from marin.transform.security_artifacts.renderers import (
    DEFAULT_ZEEK_EMPTY_FIELD,
    DEFAULT_ZEEK_UNSET_FIELD,
    render_hex_dump,
    render_zeek_tsv_log,
    render_zeek_tsv_record,
    render_zeek_tsv_value,
)


def test_render_hex_dump_matches_xxd_style_for_short_input():
    data = b"Hello World\n"

    rendered = render_hex_dump(data)

    # Hex column is padded to bytes_per_line*3 (48 chars for default 16/line);
    # then a single space and the ASCII gutter. ASCII gutter is always 16 chars.
    expected = "00000000  48 65 6c 6c 6f 20 57 6f  72 6c 64 0a             |Hello World.    |\n"
    assert rendered == expected


def test_render_hex_dump_handles_multiple_lines_and_partial_trailing_line():
    data = bytes(range(0, 40))  # 40 bytes → 2 full lines + trailing 8-byte line

    rendered = render_hex_dump(data)

    lines = rendered.splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("00000000  00 01 02 03 04 05 06 07  08 09 0a 0b 0c 0d 0e 0f")
    assert lines[1].startswith("00000010  10 11 12 13 14 15 16 17  18 19 1a 1b 1c 1d 1e 1f")
    # Trailing line has 8 bytes, ASCII gutter is still full-width padded.
    assert lines[2].startswith("00000020  20 21 22 23 24 25 26 27")
    assert lines[2].endswith("| !\"#$%&'        |")


def test_render_hex_dump_empty_returns_empty_string():
    assert render_hex_dump(b"") == ""


def test_render_hex_dump_respects_offset_start_and_uppercase():
    rendered = render_hex_dump(b"\xde\xad\xbe\xef", offset_start=0x100, uppercase=True)

    assert rendered.startswith("00000100  DE AD BE EF")
    assert "|...." in rendered  # non-printable → `.`


def test_render_hex_dump_drops_ascii_gutter_when_disabled():
    rendered = render_hex_dump(b"abcd", include_ascii=False)

    assert "|" not in rendered
    assert rendered == "00000000  61 62 63 64\n"


@pytest.mark.parametrize("bad", [0, -4, 3, 5])
def test_render_hex_dump_rejects_invalid_line_width(bad: int):
    with pytest.raises(ValueError):
        render_hex_dump(b"abc", bytes_per_line=bad)


def test_render_hex_dump_is_deterministic():
    data = b"\x00\x01\x02payload-42\x7f\xff"
    assert render_hex_dump(data) == render_hex_dump(data)


def test_render_zeek_tsv_value_handles_unset_and_empty():
    assert render_zeek_tsv_value(None) == DEFAULT_ZEEK_UNSET_FIELD
    assert render_zeek_tsv_value("") == DEFAULT_ZEEK_EMPTY_FIELD
    assert render_zeek_tsv_value([]) == DEFAULT_ZEEK_EMPTY_FIELD


def test_render_zeek_tsv_value_serializes_scalars_and_bools():
    assert render_zeek_tsv_value(True) == "T"
    assert render_zeek_tsv_value(False) == "F"
    assert render_zeek_tsv_value(42) == "42"
    assert render_zeek_tsv_value(1.5) == "1.5"
    assert render_zeek_tsv_value("192.168.0.1") == "192.168.0.1"


def test_render_zeek_tsv_value_joins_sets_with_separator():
    # tuples/lists preserve order.
    assert render_zeek_tsv_value(["dns", "http"]) == "dns,http"
    assert render_zeek_tsv_value(("a", "b", "c"), set_separator=";") == "a;b;c"
    # sets are sorted by rendered value so their output is stable across hash seeds.
    assert render_zeek_tsv_value({"ssh", "dns", "http"}) == "dns,http,ssh"


def test_render_zeek_tsv_record_writes_missing_fields_as_unset():
    record = {"ts": "1700000000.000000", "uid": "Cabc", "id.orig_h": "10.0.0.1"}
    fields = ("ts", "uid", "id.orig_h", "id.orig_p")

    rendered = render_zeek_tsv_record(record, fields)

    assert rendered == "1700000000.000000\tCabc\t10.0.0.1\t-"


def test_render_zeek_tsv_record_rejects_empty_fields():
    with pytest.raises(ValueError):
        render_zeek_tsv_record({}, ())


def test_render_zeek_tsv_log_emits_canonical_headers_and_trailer():
    records = [
        {"ts": "1700000000.000000", "uid": "C1", "id.orig_h": "10.0.0.1", "proto": "tcp"},
        {"ts": "1700000001.000000", "uid": "C2", "id.orig_h": "10.0.0.2", "proto": "udp"},
    ]
    fields = ("ts", "uid", "id.orig_h", "proto")
    types = ("time", "string", "addr", "enum")

    rendered = render_zeek_tsv_log(
        records,
        fields=fields,
        zeek_path="conn",
        types=types,
        open_time="2024-01-01-00-00-00",
        close_time="2024-01-01-01-00-00",
    )

    lines = rendered.splitlines()
    assert lines[0] == "#separator \\x09"
    assert lines[1] == "#set_separator\t,"
    assert lines[2] == "#empty_field\t(empty)"
    assert lines[3] == "#unset_field\t-"
    assert lines[4] == "#path\tconn"
    assert lines[5] == "#open\t2024-01-01-00-00-00"
    assert lines[6] == "#fields\tts\tuid\tid.orig_h\tproto"
    assert lines[7] == "#types\ttime\tstring\taddr\tenum"
    assert lines[8] == "1700000000.000000\tC1\t10.0.0.1\ttcp"
    assert lines[9] == "1700000001.000000\tC2\t10.0.0.2\tudp"
    assert lines[10] == "#close\t2024-01-01-01-00-00"
    assert rendered.endswith("\n")


def test_render_zeek_tsv_log_rejects_mismatched_types_length():
    with pytest.raises(ValueError):
        render_zeek_tsv_log(
            [],
            fields=("a", "b"),
            zeek_path="conn",
            types=("string",),
        )
