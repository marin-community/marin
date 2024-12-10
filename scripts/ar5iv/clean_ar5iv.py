"""
clean_ar5iv.py

Tries to read from jsonl.gz file, clean html, and write the contents to a jsonl.gz file.

Run with:
  - [Local] python tests/test_ray_cluster.py
  - [Ray] ray job submit --no-wait --address=http://127.0.0.1:8265 --working-dir . -- python scripts/ar5iv/clean_ar5iv.py
        => Assumes that `ray dashboard infra/marin-cluster.yaml` running in a separate terminal (port forwarding)!
"""

"""Convert fineweb to markdown"""
import argparse
import json
import os
import time
import traceback

import fsspec
import zipfile
import ray
import requests
import datetime

from marin.utils import get_gcs_path

# from scripts.ar5iv.utils import get_ar5iv_success_path
from marin import markdown
import re
from bs4 import BeautifulSoup
import markdownify

n = 256


def remove_authors(html):
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


def remove_title_page(html):
    # Remove title page since we only care about information after first section
    title_page = html.findAll("div", {"class": "ltx_titlepage"})
    for tp in title_page:
        tp.decompose()


def clean_li(html):
    # Remove the li tags since they repeat the same information (eg 1. 1.)
    tags = html.findAll("span", {"class": "ltx_tag_item"})
    for tag in tags:
        tag.decompose()
    tags = html.findAll("span", {"class": "ltx_tag_listingline"})
    for tag in tags:
        tag.decompose()


def remove_biblio(html):
    # Remove the biblio since there is a lot of noise
    biblio = html.findAll("section", {"id": "bib"})
    for bib in biblio:
        bib.decompose()


def remove_footnotes(html):
    # Remove footnotes since they are plopped in the middle of the text
    footnotes = html.findAll("div", {"class": "ltx_role_footnote"})
    for fn in footnotes:
        fn.decompose()


def remove_biblinks(html):
    # Remove the biblinks since we are removing the biblio
    biblinks = html.findAll("a", {"class": "ltx_ref"})
    for biblink in biblinks:
        # Removes reference links
        # biblink.decompose()
        # Removes linking but keeps text
        biblink.unwrap()


def linelisting_to_newline(html):
    # Turn new line listings into new lines
    linelisting = html.findAll("div", {"class": "ltx_listingline"})
    for fn in linelisting:
        fn.append(BeautifulSoup("<br>", "html.parser"))


def unwrap_eqn(html):
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


def remove_ar5iv_footer(html):
    # This is the ar5iv footer generated on xyz date
    footer = html.findAll("footer")
    for fn in footer:
        fn.decompose()


def remove_before_section(html):
    # We only care about information after the first section
    section = html.find("section")
    if section:
        section = section.previous_sibling
        while section:
            new_section = section.previous_sibling
            section.extract()
            section = new_section


def remove_title(html):
    # Title is added by markdown parser
    title = html.find("title")
    if title:
        title.decompose()


def clean_html(html):
    if isinstance(html, str):
        html = BeautifulSoup(html, "html.parser")
    remove_authors(html)
    remove_title_page(html)
    clean_li(html)
    remove_biblio(html)
    remove_footnotes(html)
    remove_biblinks(html)
    linelisting_to_newline(html)
    unwrap_eqn(html)
    remove_ar5iv_footer(html)
    remove_before_section(html)
    remove_title(html)
    return str(html)


@ray.remote(memory=512 * 1024 * 1024)  # 512 MB
def clean_ar5iv_html(file):
    """
    Takes in the input file and processes it to get the html content.
    Args:
    input_file_path (str): The input file to process
    zip_path (str): The path to the zip file
    """

    try:
        outs = ""
        with fsspec.open(get_gcs_path(file), "rb", compression="gzip") as outputf:
            # print(input_file_paths)
            for _ in range(n):
                line = outputf.readline()
                if not line:
                    break
                html_blob = json.loads(line)
                content = clean_html(html_blob["text"])
                outs += (
                    json.dumps(
                        {
                            "id": html_blob["id"],  # MANDATORY: source-specific identifier
                            "text": content,  # MANDATORY: textual content of the document
                            "source": "ar5iv",  # MANDATORY: source of the data, such as peS2o, common-crawl, etc.
                            "added": datetime.datetime.now().isoformat(),  # OPTIONAL: timestamp ai2 acquired this data
                        }
                    )
                    + "\n"
                )
        out_file = file.replace("html", "html_clean").replace("ar5iv", "ar5iv_clean")
        with fsspec.open(get_gcs_path(out_file), "wb", compression="gzip") as outputf:
            outputf.write(outs.encode("utf-8"))
        print(f"Wrote to file {out_file}")
    except FileNotFoundError as e:
        print(f"Error reading the zip file: {e}")
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ar5iv to markdown.")
    parser.add_argument("--input_path", type=str, help="Path to the ar5iv html folder", required=True)

    args = parser.parse_args()
    gfs = fsspec.filesystem("gcs")
    html_folder = get_gcs_path(args.input_path)
    files = gfs.ls(html_folder)

    MAX_NUM_PENDING_TASKS = 600  # Max number of html files we want to process in pending state
    result_refs = []

    for html in files:
        if len(result_refs) > MAX_NUM_PENDING_TASKS:
            # update result_refs to only
            # track the remaining tasks.
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                print(f"Error processing the group: {e}")
                continue
        print(f"Starting Processing for the ar5iv file: {html}")
        result_refs.append(clean_ar5iv_html.remote(html))

    # Wait for all the tasks to finish
    try:
        ray.get(result_refs)
    except Exception as e:
        print(f"Error processing the group: {e}")
