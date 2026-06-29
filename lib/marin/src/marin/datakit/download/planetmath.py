# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PlanetMath pretraining source.

PlanetMath is a compact CC-BY-SA-4.0 encyclopedia of math articles exported on
Hugging Face as one CSV file with HTML article bodies. The transform keeps the
raw download intact, converts the article HTML to Markdown, preserves MathML
``alttext`` as inline/display LaTeX, and writes one Parquet shard for Datakit
normalization.
"""

import csv
import hashlib
import os
import re
import sys
from collections.abc import Iterable
from typing import Any

import fsspec
import pyarrow as pa
from bs4 import BeautifulSoup, NavigableString, Tag
from rigging.filesystem import url_to_fs
from zephyr import counters
from zephyr.writers import write_parquet_file

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec
from marin.markdown import to_markdown
from marin.schemas.web.convert import HtmlToMarkdownConfig
from marin.utils import fsspec_url

HF_DATASET_ID = "aarjaneiro/planetmath"
HF_REVISION = "e4006c4172da7d737a7bb0b24dfd47cc87054728"
PLANETMATH_CSV_NAME = "planet_math.csv"
PLANETMATH_ROUGH_TOKENS_B = 0.008

MIN_CLEAN_CHARS = 200
OUTPUT_FILE_NAME = "data-00000-of-00001.parquet"

_HTML_NOISE_TAGS = frozenset({"base", "footer", "head", "script", "style", "template"})
_MATH_PLACEHOLDER_PREFIX = "MARINPLANETMATHMATH"
_MARKDOWN_CONFIG = HtmlToMarkdownConfig(include_images=False, include_links=False)
_METADATA_LABELS = frozenset(
    {
        "canonical name",
        "classification",
        "date of creation",
        "entry type",
        "owner",
        "title",
    }
)
_REQUIRED_RAW_COLUMNS = frozenset({"name", "url", "content"})
_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("source", pa.string()),
        pa.field("planetmath_name", pa.string()),
        pa.field("source_url", pa.string()),
    ]
)


def _is_display_math(math_element: Tag) -> bool:
    display = str(math_element.get("display", "")).lower()
    if display == "block":
        return True

    parent = math_element.parent
    while isinstance(parent, Tag):
        class_attr = parent.get("class")
        if isinstance(class_attr, str):
            classes = (class_attr,)
        elif isinstance(class_attr, list):
            classes = tuple(str(class_name) for class_name in class_attr)
        else:
            classes = ()
        if any(class_name.startswith("ltx_equation") for class_name in classes):
            return True
        parent = parent.parent

    return False


def _replace_math_alttext(root: BeautifulSoup | Tag) -> dict[str, str]:
    """Replace MathML nodes with placeholders and return exact Markdown math replacements."""
    replacements: dict[str, str] = {}
    for math_element in root.find_all("math"):
        alttext = math_element.get("alttext")
        if not isinstance(alttext, str) or not alttext.strip():
            continue

        latex = alttext.strip()
        if _is_display_math(math_element):
            replacement = f"\n\n$${latex}$$\n\n"
        else:
            replacement = f"${latex}$"
        placeholder = f"{_MATH_PLACEHOLDER_PREFIX}{len(replacements):08d}"
        replacements[placeholder] = replacement
        math_element.replace_with(NavigableString(placeholder))

    return replacements


def _remove_noise(root: BeautifulSoup | Tag) -> None:
    for tag_name in _HTML_NOISE_TAGS:
        for tag in root.find_all(tag_name):
            tag.decompose()

    for image in root.find_all("img"):
        src = image.get("src")
        if isinstance(src, str) and src.lower().startswith("data:"):
            image.decompose()

    for element in root.select("[style]"):
        style = str(element.get("style", "")).lower().replace(" ", "")
        if "display:none" in style or "visibility:hidden" in style:
            element.decompose()


def _metadata_labels(table: Tag) -> set[str]:
    labels: set[str] = set()
    for cell in table.find_all(["th", "td"]):
        text = re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip().lower()
        if text:
            labels.add(text)
    return labels


def _is_metadata_table(table: Tag) -> bool:
    labels = _metadata_labels(table)
    return len(labels & _METADATA_LABELS) >= 3 and bool({"canonical name", "entry type"} & labels)


def _remove_metadata_tables(root: BeautifulSoup | Tag) -> None:
    for table in root.find_all("table"):
        if _is_metadata_table(table):
            table.decompose()


def _content_root(soup: BeautifulSoup) -> BeautifulSoup | Tag:
    article = soup.select_one("article.ltx_document")
    if article is not None:
        return article

    fallback_article = soup.find("article")
    if isinstance(fallback_article, Tag):
        return fallback_article

    if soup.body is not None:
        return soup.body

    return soup


def _normalize_markdown(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_planetmath_html(html: str) -> str:
    """Convert one PlanetMath HTML article to Markdown text."""
    soup = BeautifulSoup(html, "html.parser")
    _remove_noise(soup)
    math_replacements = _replace_math_alttext(soup)

    root = _content_root(soup)
    _remove_metadata_tables(root)
    text = to_markdown(root, _MARKDOWN_CONFIG)
    for placeholder, replacement in math_replacements.items():
        text = text.replace(placeholder, replacement)
    return _normalize_markdown(text)


def _required_text(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    return text


def row_to_doc(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one raw PlanetMath CSV row into zero or one Datakit document."""
    counters.increment("planetmath/rows_total")

    name = _required_text(row, "name")
    source_url = _required_text(row, "url")
    content = _required_text(row, "content")
    if name is None or source_url is None or content is None:
        counters.increment("planetmath/dropped_missing_field")
        return []

    text = clean_planetmath_html(content)
    if not text:
        counters.increment("planetmath/dropped_empty_after_clean")
        return []

    if len(text) < MIN_CLEAN_CHARS:
        counters.increment("planetmath/dropped_short_after_clean")
        return []

    counters.increment("planetmath/rows_kept")
    return [
        {
            "id": hashlib.sha256(source_url.encode("utf-8")).hexdigest(),
            "text": text,
            "source": HF_DATASET_ID,
            "planetmath_name": name,
            "source_url": source_url,
        }
    ]


def _find_planetmath_csv(input_path: str) -> str:
    fs, root = url_to_fs(input_path)
    matches: list[str] = []

    for walk_root, _dirs, files in fs.walk(root):
        for filename in files:
            if filename == PLANETMATH_CSV_NAME:
                matches.append(fsspec_url(fs, os.path.join(walk_root, filename)))

    matches.sort()
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {PLANETMATH_CSV_NAME!r} under {input_path}, found {len(matches)}")

    return matches[0]


def _records_from_csv(csv_path: str) -> Iterable[dict[str, Any]]:
    csv.field_size_limit(sys.maxsize)

    with fsspec.open(csv_path, "rt", encoding="utf-8", newline="") as text_file:
        reader = csv.DictReader(text_file)
        missing = _REQUIRED_RAW_COLUMNS - set(reader.fieldnames or [])
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"PlanetMath CSV {csv_path} is missing required columns: {missing_columns}")

        for row in reader:
            yield from row_to_doc(row)


def transform(input_path: str, output_path: str) -> dict:
    """Clean the downloaded PlanetMath CSV and write one Parquet file."""
    csv_path = _find_planetmath_csv(input_path)
    output_file = os.path.join(output_path, OUTPUT_FILE_NAME)
    return write_parquet_file(_records_from_csv(csv_path), output_file, schema=_OUTPUT_SCHEMA)


def download_planetmath_step() -> StepSpec:
    """Create the raw-download plus CSV-cleaning StepSpec for PlanetMath."""
    raw = download_hf_step(
        "raw/planetmath",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[PLANETMATH_CSV_NAME],
    )

    return StepSpec(
        name="processed/planetmath",
        deps=[raw],
        fn=lambda output_path: transform(raw.output_path, output_path),
        hash_attrs={"version": "v1"},
    )


def planetmath_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for PlanetMath."""
    processed = download_planetmath_step()
    return (
        processed,
        normalize_step(name="normalized/planetmath", download=processed, file_extensions=(".parquet",)),
    )
