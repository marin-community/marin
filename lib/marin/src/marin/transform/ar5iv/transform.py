# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Transform ar5iv HTML to markdown in two stages: clean_html and markdownify.

Example Usage:
uv run zephyr --backend=ray --max-parallelism=600 --memory=512MB \
    lib/marin/src/marin/transform/ar5iv/transform.py \
    --input_path gs://bucket/ar5iv/ \
    --output_path gs://bucket/ar5iv-processed/ \
    --file_size 256
"""

import datetime
from dataclasses import dataclass

import draccus
from bs4 import BeautifulSoup
from marin import markdown
from zephyr import Backend, Dataset, load_jsonl


def transform_abstract(html: BeautifulSoup):
    # Transform the abstract from h6 to h2
    abstract = html.findAll("h6", {"class": "ltx_title_abstract"})
    for ab in abstract:
        ab.name = "h2"
    return html


def remove_authors(html: BeautifulSoup):
    # Remove authors since we only care about information after first section
    authors = html.findAll("div", {"class": "ltx_authors"})
    for author in authors:
        author.decompose()
        section = author.previous_sibling
        while section:
            new_section = section.previous_sibling
            section.decompose()
            section = new_section
    return html


def remove_title_page(html: BeautifulSoup):
    # Remove title page since we only care about information after first section
    title_page = html.findAll("div", {"class": "ltx_titlepage"})
    for tp in title_page:
        tp.decompose()


def clean_li(html: BeautifulSoup):
    # Remove the li tags since they repeat the same information (eg 1. 1.)
    tags = html.findAll("span", {"class": "ltx_tag_item"})
    for tag in tags:
        tag.decompose()
    tags = html.findAll("span", {"class": "ltx_tag_listingline"})
    for tag in tags:
        tag.decompose()


def remove_biblio(html: BeautifulSoup):
    # Remove the biblio since there is a lot of noise
    biblio = html.findAll("section", {"id": "bib"})
    for bib in biblio:
        bib.decompose()


def remove_footnotes(html: BeautifulSoup):
    # Remove footnotes since they are plopped in the middle of the text
    footnotes = html.findAll("div", {"class": "ltx_role_footnote"})
    for fn in footnotes:
        fn.decompose()


def remove_biblinks(html: BeautifulSoup):
    # Remove the biblinks since we are removing the biblio
    biblinks = html.findAll("a", {"class": "ltx_ref"})
    for biblink in biblinks:
        # Removes reference links
        # biblink.decompose()
        # Removes linking but keeps text
        biblink.unwrap()


def remove_references(html: BeautifulSoup):
    # Remove the reference section
    references = html.findAll("section", {"id": "ltx_bibliography"})
    for ref in references:
        ref.decompose()

    # Remove the references section
    references = html.findAll("ul", {"class": "ltx_biblist"})
    for ref in references:
        ref.decompose()


def linelisting_to_newline(html: BeautifulSoup):
    # Turn new line listings into new lines
    linelisting = html.findAll("div", {"class": "ltx_listingline"})
    for fn in linelisting:
        fn.append(BeautifulSoup("<br>", "html.parser"))


def unwrap_eqn(page: BeautifulSoup) -> BeautifulSoup:
    """
    Extract alttext from math element and convert to LaTeX format.
    Returns BeautifulSoup object with the formatted equation.
    """
    from html import escape, unescape

    math_elements = page.find_all("math")

    for math_elem in math_elements:
        if not math_elem or "alttext" not in math_elem.attrs:
            continue

        equation = math_elem["alttext"]
        equation = unescape(equation)
        equation = equation.replace("\\", "\\\\")
        equation = equation.replace("<", r"\<")
        equation = equation.replace(">", r"\>")

        # HTML-escape the equation to prevent < and > from being interpreted as tags
        # This is critical for equations like $T_{0}\<T\<6$ where \< would otherwise
        # be parsed as an opening tag by HTML parsers like lxml/resiliparse
        equation = escape(equation)

        is_display = math_elem.get("display") == "block"

        if is_display:
            formatted_eq = BeautifulSoup(f"<p><br><br>$${equation}$$<br><br></p>", "html.parser")
        else:
            formatted_eq = BeautifulSoup(f"${equation}$", "html.parser")

        math_elem.replace_with(formatted_eq)

    return page


def deconstruct_eqn(html: BeautifulSoup):
    # Unwrap equation tables to ensure math mode is not in a table
    eqntables = html.findAll("table", {"class": "ltx_eqn_table"})
    for eqn in eqntables:
        eqn.append(BeautifulSoup("<br>", "html.parser"))
        eqn.unwrap()
    eqnrows = html.findAll("tr", {"class": "ltx_eqn_row"})
    for eqn in eqnrows:
        eqn.append(BeautifulSoup("<br>", "html.parser"))
        eqn.unwrap()

    eqncell = html.findAll("td", {"class": "ltx_eqn_cell"})
    for eqn in eqncell:
        eqn.unwrap()


def remove_ar5iv_footer(html: BeautifulSoup):
    # This is the ar5iv footer generated on xyz date
    footer = html.findAll("footer")
    for fn in footer:
        fn.decompose()


def remove_before_section(html: BeautifulSoup):
    # We only care about information after the first section
    section = html.find("section")
    if section:
        section = section.previous_sibling
        while section:
            new_section = section.previous_sibling
            section.extract()
            section = new_section


def remove_title(html: BeautifulSoup):
    # Title is added by markdown parser
    title = html.find("title")
    if title:
        title.decompose()


def remove_figure_captions(html: BeautifulSoup):
    # Remove the figure captions since they are not needed
    captions = html.findAll("figcaption", {"class": "ltx_caption"})
    for caption in captions:
        caption.decompose()


def clean_html(html: BeautifulSoup | str) -> str:
    if isinstance(html, str):
        html = BeautifulSoup(html, "html.parser")
    remove_authors(html)
    remove_title_page(html)
    clean_li(html)
    remove_biblio(html)
    remove_footnotes(html)
    remove_biblinks(html)
    linelisting_to_newline(html)
    deconstruct_eqn(html)
    remove_ar5iv_footer(html)
    remove_before_section(html)
    remove_title(html)
    return str(html)


def clean_ar5iv_record(html_blob: dict) -> dict:
    """Clean HTML in a single ar5iv record.

    Args:
        html_blob: Record with 'id' and 'text' (HTML content)

    Returns:
        Record with cleaned HTML text
    """
    content = clean_html(html_blob["text"])
    return {
        "id": html_blob["id"],
        "text": content,
        "source": "ar5iv",
        "added": datetime.datetime.now().isoformat(),
    }


def markdownify_ar5iv_record(html_blob: dict) -> dict:
    """Convert cleaned HTML to markdown for a single ar5iv record.

    Args:
        html_blob: Record with 'id' and 'text' (cleaned HTML content)

    Returns:
        Record with markdown text
    """
    content = BeautifulSoup(html_blob["text"], "html.parser")
    try:
        content = markdown.MyMarkdownConverter().convert_soup(content)
    except Exception as e:
        print(f"Error converting to markdown: {e}")
        print("content: ", content)
        raise e
    # cleanup: replace nbsp as space
    # this isn't quite right if we preserve html in places, but we currently are not doing that
    content = content.replace("\xa0", " ").strip()
    return {
        "id": html_blob["id"],
        "text": content,
        "source": "ar5iv",
        "added": datetime.datetime.now().isoformat(),
    }


@dataclass
class Config:
    """Configuration for ar5iv transformation."""

    input_path: str
    """Path to the ar5iv jsonl.gz folder"""
    output_path: str
    """Path to the ar5iv output folder"""
    file_size: int = 256
    """Number of ar5iv documents per file (unused, kept for compatibility)"""


@draccus.wrap()
def main(cfg: Config) -> None:
    """Convert ar5iv HTML to markdown in two stages."""
    # Stage 1: Clean HTML
    print("Stage 1: Cleaning HTML...")
    clean_pipeline = (
        Dataset.from_files(f"{cfg.input_path}/**/*.jsonl.gz")
        .flat_map(load_jsonl)
        .map(clean_ar5iv_record)
        .write_jsonl(f"{cfg.output_path}/html_clean/{{shard:05d}}.jsonl.gz", skip_existing=True)
    )
    Backend.execute(clean_pipeline)

    # Stage 2: Convert to Markdown
    print("Stage 2: Converting to markdown...")
    markdown_pipeline = (
        Dataset.from_files(f"{cfg.output_path}/html_clean/**/*.jsonl.gz")
        .flat_map(load_jsonl)
        .map(markdownify_ar5iv_record)
        .write_jsonl(f"{cfg.output_path}/md/{{shard:05d}}.jsonl.gz", skip_existing=True)
    )
    Backend.execute(markdown_pipeline)

    print("Transformation complete!")
