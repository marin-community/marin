# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import csv
import hashlib
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from marin.datakit.download.planetmath import (
    HF_DATASET_ID,
    HF_REVISION,
    PLANETMATH_CSV_NAME,
    clean_planetmath_html,
    planetmath_normalize_steps,
    row_to_doc,
    transform,
)


def _long_sentence() -> str:
    return (
        "This PlanetMath article explains a mathematical construction with definitions, "
        "examples, relationships to neighboring concepts, and enough expository detail "
        "to represent the source as useful pretraining text."
    )


def _article_html(body: str | None = None) -> str:
    text = body or " ".join([_long_sentence()] * 3)
    return f"""
    <html>
      <head>
        <style>.hidden {{ display: none; }}</style>
        <script>window.noisy = true;</script>
      </head>
      <body>
        <article class="ltx_document">
          <table>
            <tr><th>Title</th><td>Ring</td></tr>
            <tr><th>Canonical name</th><td>Ring</td></tr>
            <tr><th>Entry type</th><td>Definition</td></tr>
          </table>
          <p>
            Let <math alttext="x_1^2 + y_2"><mi>x</mi></math> be the motivating polynomial.
            <sup style="display: none;">hidden concept expansion</sup>
          </p>
          <div class="ltx_equation">
            <math alttext="\\sum_{{i=1}}^n i = \\frac{{n(n+1)}}{{2}}">
              <mi>x</mi>
            </math>
          </div>
          <p>{text}</p>
          <p><a href="https://planetmath.org/ring">visible reference</a></p>
          <img src="data:image/png;base64,abc" alt="encoded image">
        </article>
        <footer>download boilerplate footer</footer>
      </body>
    </html>
    """


def _row(**overrides) -> dict:
    row = {
        "Unnamed: 0": "0",
        "name": "Ring",
        "url": "https://planetmath.org/ring",
        "content": _article_html(),
    }
    row.update(overrides)
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Unnamed: 0", "name", "url", "content"])
        writer.writeheader()
        writer.writerows(rows)


def test_clean_planetmath_html_preserves_math_and_removes_boilerplate():
    text = clean_planetmath_html(_article_html())

    assert "$x_1^2 + y_2$" in text
    assert "$$\\sum_{i=1}^n i = \\frac{n(n+1)}{2}$$" in text
    assert "This PlanetMath article explains" in text
    assert "visible reference" in text
    assert "https://planetmath.org/ring" not in text
    assert "Canonical name" not in text
    assert "Entry type" not in text
    assert "hidden concept expansion" not in text
    assert "encoded image" not in text
    assert "download boilerplate footer" not in text
    assert "window.noisy" not in text


def test_row_to_doc_preserves_stable_provenance_fields():
    [doc] = row_to_doc(_row())

    assert doc["id"] == hashlib.sha256(b"https://planetmath.org/ring").hexdigest()
    assert doc["source"] == HF_DATASET_ID
    assert doc["planetmath_name"] == "Ring"
    assert doc["source_url"] == "https://planetmath.org/ring"
    assert "$x_1^2 + y_2$" in doc["text"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": ""},
        {"name": "   "},
        {"name": None},
        {"url": ""},
        {"url": "   "},
        {"url": None},
        {"content": ""},
        {"content": "   "},
        {"content": None},
        {"content": "<article>short after cleaning</article>"},
    ],
)
def test_row_to_doc_drops_missing_or_too_short_rows(overrides):
    assert row_to_doc(_row(**overrides)) == []


def test_transform_requires_exactly_one_raw_csv(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    with pytest.raises(ValueError, match="Expected exactly one"):
        transform(str(raw_dir), str(tmp_path / "processed-empty"))

    first = raw_dir / PLANETMATH_CSV_NAME
    first.write_text("name,url,content\n", encoding="utf-8")
    nested = raw_dir / "nested"
    nested.mkdir()
    second = nested / PLANETMATH_CSV_NAME
    second.write_text("name,url,content\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected exactly one"):
        transform(str(raw_dir), str(tmp_path / "processed-duplicate"))


def test_transform_reads_raw_csv_and_writes_clean_parquet(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "planetmath"
    raw_dir.mkdir(parents=True)
    _write_csv(
        raw_dir / PLANETMATH_CSV_NAME,
        [
            _row(),
            _row(name="Too Short", url="https://planetmath.org/too-short", content="<article>short</article>"),
        ],
    )

    output_dir = tmp_path / "processed"
    result = transform(str(tmp_path / "raw"), str(output_dir))

    assert result["count"] == 1
    rows = [row for path in output_dir.rglob("*.parquet") for row in pq.read_table(path).to_pylist()]
    assert len(rows) == 1
    assert rows[0]["source"] == HF_DATASET_ID
    assert rows[0]["planetmath_name"] == "Ring"
    assert rows[0]["source_url"] == "https://planetmath.org/ring"
    assert "Canonical name" not in rows[0]["text"]
    assert "$x_1^2 + y_2$" in rows[0]["text"]


def test_transform_rejects_csv_missing_required_columns(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / PLANETMATH_CSV_NAME).write_text("name,url\nRing,https://planetmath.org/ring\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns: content"):
        transform(str(raw_dir), str(tmp_path / "processed"))


def test_planetmath_normalize_steps_use_pinned_csv_download_and_stable_names():
    processed, normalized = planetmath_normalize_steps()
    download = processed.deps[0]

    assert download.name == "raw/planetmath"
    assert download.hash_attrs["hf_dataset_id"] == HF_DATASET_ID
    assert download.hash_attrs["revision"] == HF_REVISION
    assert download.hash_attrs["hf_urls_glob"] == [PLANETMATH_CSV_NAME]
    assert processed.name == "processed/planetmath"
    assert processed.deps == [download]
    assert normalized.name == "normalized/planetmath"
    assert normalized.deps == [processed]
