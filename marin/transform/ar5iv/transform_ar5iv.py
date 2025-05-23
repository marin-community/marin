"""
ar5iv/transform_ar5iv.py

Performs HTML->Text/MD conversion using the specified tools over a ar5iv dump save in DOLMA format.
"""

import json
import logging
import os
import re
from dataclasses import dataclass

import draccus
import fsspec
import ray
from bs4 import BeautifulSoup
from tqdm_loggable.auto import tqdm

from marin.core.runtime import cached_or_construct_output
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


@ray.remote(memory=2 * 1024 * 1024 * 1024)
@cached_or_construct_output(success_suffix="SUCCESS")
def process_file(
    input_file_path: str,
    output_file_path: str,
    extract_method: str,
    extract_config: ExtractionConfig,
    remove_reference_section: bool = True,
) -> None:
    logger.info(f"Starting processing of file {input_file_path}")
    logger.info(f"Source: {input_file_path}")
    logger.info(f"Destination: {output_file_path}")
    try:
        with (
            fsspec.open(input_file_path, compression="gzip") as source,
            fsspec.open(output_file_path, "wt", compression="gzip") as output,
        ):
            for line in tqdm(source, desc="Processing lines"):
                row = json.loads(line)

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

                    print(json.dumps(out_dict), file=output)
                except Exception as e:
                    logger.exception(f"Error processing line: {e}")
                    raise

        logger.info("\nProcessing completed successfully!")
        logger.info(f"File available at: {output_file_path}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


@draccus.wrap()
def process_ar5iv_dump(cfg: Ar5ivExtractionConfig) -> None:
    files = fsspec_glob(f"{cfg.input_path}/*.jsonl.gz")

    result_refs = []
    MAX_CONCURRENT_WORKERS = 200

    for file in files:
        if len(result_refs) > MAX_CONCURRENT_WORKERS:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                logger.exception(f"Error processing the group: {e}")
                continue

        output_file_path = os.path.join(cfg.output_path, file.split("/")[-1])
        result_refs.append(
            process_file.remote(
                file,
                output_file_path,
                cfg.extract_method,
                cfg.extract_config,
                cfg.remove_reference_section,
            )
        )
    try:
        ray.get(result_refs)
    except Exception as e:
        logger.exception(f"Error processing the group: {e}")
