# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
ar5iv/transform_ar5iv.py

Performs HTML->Text/MD conversion using the specified tools over a ar5iv dump save in DOLMA format.

Example Usage:
uv run zephyr --backend=ray --max-parallelism=200 --memory=2GB --cluster=us-central2 \
    lib/marin/src/marin/transform/ar5iv/transform_ar5iv.py \
    --input_path gs://path/to/input --output_path gs://path/to/output ...
"""

import logging
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup
from marin.schemas.web.convert import ExtractionConfig
from marin.transform.ar5iv.transform import (
    clean_li,
    deconstruct_eqn,
    linelisting_to_newline,
    remove_ar5iv_footer,
    remove_authors,
    remove_before_section,
    remove_biblinks,
    remove_biblio,
    remove_figure_captions,
    remove_footnotes,
    remove_references,
    remove_title_page,
    transform_abstract,
    unwrap_eqn,
)
from marin.utils import fsspec_glob
from marin.web.convert import convert_page
from zephyr import Dataset, ZephyrContext, load_jsonl

logger = logging.getLogger("ray")


@dataclass
class Ar5ivExtractionConfig:
    input_path: str
    output_path: str
    revision: str
    remove_reference_section: bool
    extract_method: str
    extract_config: ExtractionConfig


def clean_html(html: str, remove_reference_section: bool = True) -> str:
    """
    Clean the HTML content by removing unnecessary elements and formatting.

    The cleaning is mainly to remove non-essential elements like metadata (title page, authors, footer), academic paper
    artifacts (bibliography, footnotes, figure captions), and formatting that could easily be parsed by the resiliparse
    (equation tables, duplicate list numbering).


    Most of the steps are standard boilerplate cleanups based on intuition from experiments with wikipedia.
    For example, we remove the references section by default based on the performance of the model on wikipedia.
    Transformations are applied in the order cleanup the content for better extraction.


    Args:
        html (str): The HTML content to clean.
        remove_reference_section (bool): Whether to remove the reference section.

    Returns:
        str: The cleaned HTML content.
    """

    html = BeautifulSoup(html, "html.parser")

    # Transform the abstract section into an h2 heading to ensure proper structure
    # This makes the abstract a section in the markdownified output
    transform_abstract(html)

    # Remove author information to reduce noise and remove PII from appearing
    remove_authors(html)

    # Remove the title page elements which typically contain redundant information
    # that will be prepended elsewhere
    remove_title_page(html)

    # Clean list items to avoid duplicate numbering patterns like (1. 1.)
    # which can occur when LaTeX numbering is combined with HTML list markers
    clean_li(html)

    # Remove bibliography sections to remove references
    remove_biblio(html)

    # Remove footnotes
    remove_footnotes(html)

    # Remove biblinks since we're removing the references section
    remove_biblinks(html)

    # Convert code listing lines to proper newlines to preserve code formatting
    linelisting_to_newline(html)

    # Transform equation tables into inline elements for better markdown conversion
    deconstruct_eqn(html)

    # Extract mathematical notation from alt text attributes and convert to LaTeX format
    html = unwrap_eqn(html)

    # Remove the ar5iv footer which contains boilerplate text about the conversion process
    remove_ar5iv_footer(html)

    # Remove content before the first main section (typically metadata and preamble)
    remove_before_section(html)

    # Remove figure captions
    remove_figure_captions(html)

    if remove_reference_section:
        remove_references(html)

    return str(html)


def process_record(
    row: dict,
    extract_method: str,
    extract_config: ExtractionConfig,
    remove_reference_section: bool = True,
) -> dict[str, str]:
    """Process a single ar5iv record and return transformed record.

    Args:
        row: Record from JSONL file
        extract_method: Method to use for HTML extraction
        extract_config: Configuration for the extraction method
        remove_reference_section: Whether to remove reference sections

    Returns:
        Transformed record in Dolma format
    """
    try:
        filtered_html = clean_html(row["content"], remove_reference_section)
        result = convert_page(filtered_html, extract_method=extract_method, config=extract_config)
        if remove_reference_section:
            result["content"] = re.sub(r"\s?\\\[(?:\d+(?:,\s*\d+)*)\\\]", "", result["content"])

        out_dict = {
            "id": row["filename"],
            "source": "ar5iv",
            "format": "text",
            "text": result["content"],
        }

        return out_dict
    except Exception as e:
        logger.exception(f"Error processing line: {e}")
        raise


def process_ar5iv_dump(cfg: Ar5ivExtractionConfig) -> None:
    files = fsspec_glob(f"{cfg.input_path}/*.jsonl.gz")

    pipeline = (
        Dataset.from_list(files)
        .flat_map(load_jsonl)
        .map(
            lambda row: process_record(
                row,
                cfg.extract_method,
                cfg.extract_config,
                cfg.remove_reference_section,
            )
        )
        .write_jsonl(f"{cfg.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    with ZephyrContext(name="transform-ar5iv-v2") as ctx:
        ctx.execute(pipeline)
