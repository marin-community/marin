"""
wikipedia/transform_wikipedia.py

Performs HTML->Text/MD conversion using the specified tools over a wiki dump save in DOLMA format.
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
from marin.utils import fsspec_glob
from marin.web.convert import convert_page

logger = logging.getLogger("ray")


@dataclass
class WikiExtractionConfig:
    """
    input_path: The path to the Wikipedia dump file or directory containing the dump files in JSONL format
    output_path: The path where the processed text/markdown files will be saved
    revision: The revision identifier of the Wikipedia dump (e.g., "20241201") for versioning and tracking
    extract_method: The method to use for HTML extraction (e.g., "readability", "resiliparse", "trafilatura")
    extract_config: Configuration object for the extraction method (e.g., ResiliparseConfig, HtmlToMarkdownConfig)
    remove_reference_section: If True, removes reference sections from articles to reduce noise in the extracted text
    max_files: Optional limit on the number of files to process, useful for testing or partial processing
    digit_threshold: Percentage threshold for filtering out pages with excessive digits
                     (e.g., 50 means pages with >50% digits are filtered out)
    word_threshold: Percentage threshold for filtering out pages with insufficient words
                    (e.g., 70 means pages with <70% words are filtered out)
    special_char_threshold: Percentage threshold for filtering out pages with excessive special characters
                            (e.g., 50 means pages with >50% special characters are filtered out)
    """

    input_path: str
    output_path: str
    revision: str
    extract_method: str
    extract_config: ExtractionConfig
    remove_reference_section: bool
    max_files: int | None = None
    digit_threshold: int = 50
    word_threshold: int = 70
    special_char_threshold: int = 50


def remove_and_append_infobox(html: str) -> str:
    """
    Wraps the infobox in a new section with heading 'InfoBox' and appends it to the end of the article.
    """
    soup = BeautifulSoup(html, "html.parser")

    infobox = soup.find("table", {"class": "infobox"})
    if infobox:
        # Remove the infobox from its current position
        infobox.extract()

        # Create new section with heading
        br = soup.new_tag("br")
        notes_section = soup.new_tag("div")
        notes_section.append(br)
        heading = soup.new_tag("h2")
        heading.string = "InfoBox"
        notes_section.append(heading)
        notes_section.append(infobox)
        notes_section.append(br)

        # Find the body tag and append the new section
        body = soup.find("body")
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

    ref_heading = soup.find("h2", {"id": "References"})
    if ref_heading:
        ref_heading.extract()

    return str(soup)


def unwrap_eqn(html: str):
    """Extract equations from math elements and convert to LaTeX inline/block quotes,
    wrapping display math in <p> tags."""
    html = BeautifulSoup(html, "html.parser")
    # Find all annotations containing equations
    annotations = html.findAll("annotation", {"encoding": "application/x-tex"})

    for annotation in annotations:
        # Extract the LaTeX content and remove \displaystyle wrapper
        latex = annotation.get_text()
        latex = latex.replace(r"{\displaystyle ", "").rstrip("}")

        # Fix common LaTeX formatting issues
        latex = latex.replace(r"\_{", "_{")  # Remove unnecessary backslash before subscript
        latex = re.sub(r"_\{([^}]*?)_\{", r"_{", latex)  # Fix nested subscripts
        latex = re.sub(r"\\([a-zA-Z]+)_\{", r"\\\1_{", latex)  # Fix function subscripts
        latex = latex.strip("{}")  # Remove wrapping curly braces
        latex = latex.replace(r"\[ ", "").replace(r" \]", "")  # Remove \[ \] display math delimiters
        latex = re.sub(r"\\!", "", latex)  # Remove \! spacing commands

        # Balance remaining curly braces
        open_count = latex.count("{")
        close_count = latex.count("}")
        if open_count > close_count:
            latex += "}" * (open_count - close_count)

        # Get the containing span element for the equation
        span_element = annotation.find_parent("span", {"class": "mwe-math-element"})
        if not span_element:
            continue

        # Check if this is display math by looking for <dd> inside <dl>
        dd_parent = span_element.find_parent("dd")
        if dd_parent:
            dl_parent = dd_parent.find_parent("dl")
            if dl_parent:
                # Check dd contents for other non-math text
                other_content = False
                for child in dd_parent.children:
                    if child.name != "span" or "mwe-math-element" not in child.get("class", []):
                        if str(child).strip():
                            other_content = True
                            break
                is_display = not other_content
            else:
                is_display = False
        else:
            is_display = False

        # Format equations
        if is_display:
            # Create a new <p> tag and insert the br tags plus the display math
            p_tag = html.new_tag("p")
            p_tag.append(html.new_tag("br"))
            p_tag.append(html.new_tag("br"))
            p_tag.append(BeautifulSoup(f"$${latex}$$", "html.parser"))
            p_tag.append(html.new_tag("br"))
            p_tag.append(html.new_tag("br"))
            span_element.replace_with(p_tag)
        else:
            # Inline math: handle spacing
            prev_sibling = span_element.previous_sibling
            needs_left_space = prev_sibling and not str(prev_sibling).endswith(" ")

            left_space = " " if needs_left_space else ""
            formatted_latex = f"{left_space}${latex}$"
            span_element.replace_with(formatted_latex)

    return str(html)


def postprocess_content(content: str, digit_threshold: int, word_threshold: int, special_char_threshold: float) -> str:
    """
    Postprocesses the content by deleting it if its is mainly digits, words, and special characters.
    """

    if not content or len(content) < 10:
        return None

    digit_percentage = int(sum(c.isdigit() for c in content) / len(content) * 100)
    word_percentage = len(content.split())
    special_char_percentage = int(sum(not c.isalnum() for c in content) / len(content) * 100)

    if (
        digit_percentage > digit_threshold
        or word_percentage < word_threshold
        or special_char_percentage > special_char_threshold
    ):
        return None

    return content


def clean_wiki_html(html: str, remove_reference_section: bool = True) -> str:
    """
    Cleans the HTML by removing unwanted elements.
    """
    html = unwrap_eqn(html)
    html = remove_and_append_infobox(html)

    if remove_reference_section:
        html = remove_references_from_html(html)

    return html


@ray.remote(memory=2 * 1024 * 1024 * 1024)
@cached_or_construct_output(success_suffix="SUCCESS")
def process_file(
    input_file_path: str,
    output_file_path: str,
    extract_method: str,
    extract_config: ExtractionConfig,
    remove_reference_section: bool = True,
    digit_threshold: int = 50,
    word_threshold: int = 70,
    special_char_threshold: float = 0.2,
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
                    content = None
                    if "html" not in row["article_body"].keys() and "wikitext" in row["article_body"].keys():
                        continue
                    elif "html" in row["article_body"]:
                        html_string = row["article_body"]["html"]

                        filtered_html = clean_wiki_html(html_string, remove_reference_section)
                        content = convert_page(filtered_html, extract_method=extract_method, config=extract_config)[
                            "content"
                        ]
                    else:
                        logger.error(f"No content found in the row: {row}")
                        continue

                    content = postprocess_content(content, digit_threshold, word_threshold, special_char_threshold)
                    if content is None:
                        continue

                    out_dict = {
                        "id": row["identifier"],
                        "url": row["url"],
                        "title": row["name"],
                        "abstract": row.get("abstract", ""),
                        "date_created": row["date_created"] if "date_created" in row else row.get("date_modified", ""),
                        "text": content,
                    }

                    print(json.dumps(out_dict), file=output)  # Without this line, the JSON file will be corrupted
                except Exception as e:
                    logger.info(f"Keys in row: {row.keys()}")
                    logger.info(f"Article body keys: {row['article_body'].keys()}")

                    logger.exception(f"Error processing line: {e}")
                    continue

        logger.info("\nProcessing completed successfully!")
        logger.info(f"File available at: {output_file_path}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


@draccus.wrap()
def process_wiki_dump(cfg: WikiExtractionConfig) -> None:
    logger.info(f"Starting processing of Wikipedia dump in {cfg.input_path}")

    files = fsspec_glob(f"{cfg.input_path}/*.ndjson")
    logger.info(f"Found {len(files)} files to process")

    result_refs = []
    MAX_CONCURRENT_WORKERS = 400

    if cfg.max_files:
        files = files[: cfg.max_files]

    for file in files:
        if len(result_refs) > MAX_CONCURRENT_WORKERS:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                logger.exception(f"Error processing the group: {e}")
                continue

        output_path = os.path.join(cfg.output_path, cfg.revision)
        output_file_path = os.path.join(output_path, file.split("/")[-1].replace(".ndjson", ".jsonl.gz"))
        result_refs.append(
            process_file.remote(
                file,
                output_file_path,
                cfg.extract_method,
                cfg.extract_config,
                cfg.remove_reference_section,
                cfg.digit_threshold,
                cfg.word_threshold,
                cfg.special_char_threshold,
            )
        )
    try:
        ray.get(result_refs)
    except Exception as e:
        logger.exception(f"Error processing the group: {e}")
