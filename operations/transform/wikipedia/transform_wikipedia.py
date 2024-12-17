"""
wikipedia/transform_wikipedia.py

Performs HTML->Text/MD conversion using the specified tools over a wiki dump save in DOLMA format.
"""

import json
import logging
import os
from dataclasses import dataclass

import draccus
import fsspec
import ray
from bs4 import BeautifulSoup
from tqdm_loggable.auto import tqdm

from marin.schemas.web.convert import ExtractionConfig
from marin.utils import fsspec_glob
from marin.web.convert import convert_page

logger = logging.getLogger("ray")


@dataclass
class WikiExtractionConfig:
    input_path: str
    output_path: str
    revision: str
    extract_method: str
    extract_config: ExtractionConfig
    remove_reference_section: bool


def remove_and_append_infobox(html: str) -> str:
    """
    Wraps the infobox in a new section with heading 'Notes' and appends it to the end of the article.
    """
    soup = BeautifulSoup(html, "html.parser")

    infobox = soup.find("table", {"class": "infobox"})
    if infobox:
        # Remove the infobox from its current position
        infobox.extract()

        # Create new section with heading
        notes_section = soup.new_tag("div")
        heading = soup.new_tag("h2")
        heading.string = "Notes"
        notes_section.append(heading)
        notes_section.append(infobox)

        # Find the body tag and append the new section
        body = soup.find('body')
        if body:
            body.append(notes_section)
        else:
            soup.append(notes_section)

    return str(soup)


def remove_references_from_html(html: str) -> str:
    """
    Removes the references list and heading from the article.
    """
    soup = BeautifulSoup(html, "html.parser")

    reflist = soup.find("div", {"class": "reflist"})
    if reflist:
        reflist.extract()

    ref_heading = soup.find("span", {"class": "mw-heading", "id": "References"})
    if ref_heading:
        ref_heading.extract()

    return str(soup)


def clean_html(html: str, remove_reference_section: bool = True) -> str:
    """
    Cleans the HTML by removing unwanted elements.
    """
    html = BeautifulSoup(html, "html.parser")

    remove_and_append_infobox(html)

    if remove_reference_section:
        remove_references_from_html(html)

    return str(html)


@ray.remote(memory=2 * 1024 * 1024 * 1024)
def process_file(input_file_path: str, output_path: str, extract_method: str, extract_config: ExtractionConfig, remove_reference_section: bool = True) -> None:
    output_file_path = os.path.join(output_path, input_file_path.split("/")[-1].replace(".ndjson", ".jsonl.gz"))

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
                    filtered_html = clean_html(row["article_body"]["html"], remove_reference_section)
                    result = convert_page(
                        filtered_html, extract_method=extract_method, config=extract_config
                    )
                    out_dict = {
                        "id": row["identifier"],
                        "url": row["url"],
                        "title": row["name"],
                        "abstract": row.get("abstract", ""),
                        "date_created": row["date_created"] if "date_created" in row else row.get("date_modified", ""),
                        "text": result["content"],
                    }

                    print(json.dumps(out_dict), file=output)  # Without this line, the JSON file will be corrupted
                except Exception as e:
                    logger.info(f"Keys in row: {row.keys()}")
                    logger.info(f"Article body keys: {row['article_body'].keys()}")

                    logger.exception(f"Error processing line: {e}")
                    continue

        logger.info("\nProcessing completed successfully!")
        logger.info(f"File available at: {output_path}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


@draccus.wrap()
def process_wiki_dump(cfg: WikiExtractionConfig) -> None:
    logger.info(f"Starting processing of Wikipedia dump in {cfg.input_path}")

    files = fsspec_glob(f"{cfg.input_path}/*.ndjson")
    logger.info(f"Found {len(files)} files to process")

    result_refs = []
    MAX_CONCURRENT_WORKERS = 15

    for file in files:
        if len(result_refs) > MAX_CONCURRENT_WORKERS:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                logger.exception(f"Error processing the group: {e}")
                continue

        output_path = os.path.join(cfg.output_path, cfg.revision)
        result_refs.append(process_file.remote(file, output_path, cfg.extract_method, cfg.extract_config, cfg.remove_reference_section))
    try:
        ray.get(result_refs)
    except Exception as e:
        logger.exception(f"Error processing the group: {e}")
