# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import random
from pathlib import Path

import pytest
from marin.datakit import junk_filter
from marin.datakit.junk_filter import JunkCategory, junk_verdict

DOCUMENT_CHARS = 40_000


def _words(rng: random.Random, count: int) -> list[str]:
    return ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(2, 10))) for _ in range(count)]


def _prose(seed: int = 0, chars: int = DOCUMENT_CHARS) -> str:
    rng = random.Random(seed)
    vocabulary = _words(rng, 3000)
    paragraphs = []
    while sum(map(len, paragraphs)) < chars:
        sentences = [
            " ".join(rng.choice(vocabulary) for _ in range(rng.randint(6, 24))).capitalize() + "."
            for _ in range(rng.randint(3, 8))
        ]
        paragraphs.append(" ".join(sentences))
    return "\n\n".join(paragraphs)


def _base64(seed: int = 0, chars: int = DOCUMENT_CHARS) -> str:
    return base64.b64encode(random.Random(seed).randbytes(chars)).decode()[:chars]


def _zero_matrix() -> str:
    header = "," + ",".join(str(i) for i in range(60))
    return "\n".join([header, *(f"{row}," + ",".join(["0.0"] * 60) for row in range(160))])


def _escaped_bytes() -> str:
    rng = random.Random(1)
    payload = "".join(f"\\x{rng.randrange(256):02x}" for _ in range(DOCUMENT_CHARS // 4))
    return f"cnumpy.core.multiarray\n_reconstruct\np0\n(cnumpy\nndarray\np1\nS'{payload}'\np2\ntp3\n."


def _minified_javascript() -> str:
    rng = random.Random(2)
    names = _words(rng, 400)
    modules = [
        f'{index}:[function(e,t,n){{"use strict";var r=e("./{rng.choice(names)}"),i=function(){{function e(t){{'
        f"this._{rng.choice(names)}=t}}return e.prototype.{rng.choice(names)}=function(){{return "
        f"this._{rng.choice(names)}.{rng.choice(names)}(r.{rng.choice(names)})}},e}}();n.default=i}},{{}}]"
        for index in range(220)
    ]
    return "!function e(t,n,r){function i(o,s){return n[o].exports}return i}({" + ",".join(modules) + "},{},[1]);"


def _lock_file() -> str:
    rng = random.Random(3)
    names = _words(rng, 300)
    packages = {
        f"node_modules/{name}": {
            "version": f"{rng.randint(0, 9)}.{rng.randint(0, 30)}.{rng.randint(0, 9)}",
            "resolved": f"https://registry.npmjs.org/{name}/-/{name}-1.0.0.tgz",
            "integrity": "sha512-" + base64.b64encode(rng.randbytes(64)).decode(),
        }
        for name in names
    }
    return json.dumps({"lockfileVersion": 3, "packages": packages}, indent=2)


def _survey_csv() -> str:
    rng = random.Random(4)
    answers = ["Strongly Agree", "Partly Agree", "Quite Disagree", "Don't Know", "Higher", "Low", "Mid", "Very high"]
    rows = [",".join(f'"{rng.choice(answers)}"' for _ in range(140)) for _ in range(40)]
    return "\n".join(rows)


def _json_records() -> str:
    rng = random.Random(5)
    names = _words(rng, 500)
    records = [
        {"id": index, "name": rng.choice(names), "score": rng.random(), "tags": [rng.choice(names) for _ in range(3)]}
        for index in range(450)
    ]
    return json.dumps(records, indent=1)


def _word_list() -> str:
    rng = random.Random(6)
    vocabulary = _words(rng, 2000)
    return "\n".join(sorted(" ".join(rng.choice(vocabulary) for _ in range(rng.randint(1, 3))) for _ in range(2500)))


def _interleave(first: str, second: str, second_share: float) -> list[str]:
    """Windows of one chunk each, ``second_share`` of them drawn from ``second``."""
    size = junk_filter.CHUNK_CHARS
    total = DOCUMENT_CHARS // size
    from_second = round(total * second_share)
    return [second[i * size : (i + 1) * size] for i in range(from_second)] + [
        first[i * size : (i + 1) * size] for i in range(total - from_second)
    ]


def _markdown_table() -> str:
    rng = random.Random(7)
    names = _words(rng, 300)
    rows = [f"| {rng.choice(names):<28} | {rng.choice(names):<20} | {rng.randint(1, 9000):>8} |" for _ in range(700)]
    return "\n".join(rows)


@pytest.mark.parametrize(
    ("document", "category"),
    [
        (_base64(), JunkCategory.ENCODED_BLOB),
        (_escaped_bytes(), JunkCategory.ENCODED_BLOB),
        (_zero_matrix(), JunkCategory.REPETITION),
        ("Amazon Transcribe " * 2500, JunkCategory.REPETITION),
        (_minified_javascript(), JunkCategory.MINIFIED_CODE),
        (_lock_file(), JunkCategory.LOCK_FILE),
        (_survey_csv(), JunkCategory.STRUCTURED_DATA),
        (_json_records(), JunkCategory.STRUCTURED_DATA),
        (_word_list(), JunkCategory.ITEM_LIST),
    ],
    ids=["base64", "escaped-bytes", "zero-matrix", "repeated-phrase", "minified-js", "lock-file", "csv", "json", "list"],
)
def test_junk_verdict_machine_output_gets_its_category(document, category):
    assert junk_verdict([document]).category == category


def test_junk_verdict_prose_is_kept():
    assert junk_verdict([_prose()]).category is None


def test_junk_verdict_formatted_source_code_is_kept():
    source = Path(junk_filter.__file__).read_text()

    assert junk_verdict([source]).category is None


def test_junk_verdict_numbered_verses_are_kept():
    # A scripture translation stores one verse per line as "<verse id>\t<text>"; the tab does not make it a table.
    rng = random.Random(8)
    vocabulary = _words(rng, 3000)
    verses = "\n".join(
        f"{40001001 + n}\t" + " ".join(rng.choice(vocabulary) for _ in range(rng.randint(15, 40))).capitalize() + "."
        for n in range(400)
    )

    assert junk_verdict([verses]).category is None


def test_junk_verdict_line_wrapped_base64_is_an_encoded_blob():
    payload = _base64()
    wrapped = "\n".join(payload[start : start + 76] for start in range(0, len(payload), 76))

    assert junk_verdict([wrapped]).category == JunkCategory.ENCODED_BLOB


def test_junk_verdict_report_with_some_tables_is_kept():
    windows = _interleave(_prose(), _markdown_table(), second_share=0.4)

    assert junk_verdict(windows).category is None


def test_junk_verdict_document_one_third_base64_is_machine_junk():
    # Below the share that makes a document junk in general, but encoded payloads count at a lower one.
    windows = _interleave(_prose(), _base64(), second_share=0.34)

    assert junk_verdict(windows).category == JunkCategory.ENCODED_BLOB


def test_junk_verdict_judges_sampled_windows_like_the_whole_document():
    document = _json_records()
    size = junk_filter.CHUNK_CHARS
    windows = [document[start : start + size] for start in (0, len(document) // 2, len(document) - size)]

    assert junk_verdict(windows).category == junk_verdict([document]).category == JunkCategory.STRUCTURED_DATA
