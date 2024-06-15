"""
load_ar5iv.py

Tries to read from zip file and write the contents to a jsonl file.

Run with:
  - [Local] python tests/test_ray_cluster.py
  - [Ray] ray job submit --no-wait --address=http://127.0.0.1:8265 --working-dir . -- python scripts/ar5iv/load_ar5iv.py
        => Assumes that `ray dashboard infra/marin-cluster.yaml` running in a separate terminal (port forwarding)!
"""

'''Convert fineweb to markdown'''
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


def to_markdown(html):
    if isinstance(html, str):
        html = BeautifulSoup(html, "html.parser")
    authors = html.findAll('div', {'class': 'ltx_authors'})
    for author in authors:
        author.decompose()
    tags = html.findAll('span', {'class': 'ltx_tag_item'})
    for author in tags:
        author.decompose()
    tags = html.findAll('span', {'class': 'ltx_tag_listingline'})
    for author in tags:
        author.decompose()
    title_page = html.findAll('div', {'class': 'ltx_titlepage'})
    for tp in title_page:
        tp.decompose()
    biblio = html.findAll('section', {'id': 'bib'})
    for bib in biblio:
        bib.decompose()
    footnotes = html.findAll('div', {'class': 'ltx_role_footnote'})
    for fn in footnotes:
        fn.decompose()
    linelisting = html.findAll('div', {'class': 'ltx_listingline'})
    for fn in linelisting:
        fn.append(BeautifulSoup("<br>", "html.parser"))
    
    eqntables = html.findAll('table', {'class': 'ltx_eqn_table'})
    for eqn in eqntables:
        eqn.append(BeautifulSoup("<br>", "html.parser"))
        eqn.unwrap()
    
    eqnrows = html.findAll('tr', {'class': 'ltx_eqn_row'})
    for eqn in eqnrows:
        eqn.append(BeautifulSoup("<br>", "html.parser"))
        eqn.unwrap()
    eqncell = html.findAll('td', {'class': 'ltx_eqn_cell'})
    for eqn in eqncell:
        eqn.unwrap()
    footer = html.findAll('footer')
    for fn in footer:
        fn.decompose()
    # data = html.findAll('div', {'class': 'ltx_listing_data'})
    # for fn in data:
    #     fn.decompose()
    title = html.find('title')
    if title:
        title.decompose()
    text = markdown.MyMarkdownConverter().convert_soup(html)
    # cleanup: replace nbsp as space
    # this isn't quite right if we preserve html in places, but we currently are not doing that
    text = text.replace("\xa0", " ").strip()
    return text

@ray.remote(memory=512 * 1024 * 1024)  # 512 MB
def load_ar5iv_html(input_file_paths, zip_path, counts):
    """
    Takes in the input file and processes it to get the html content.
    Args:
    input_file_path (str): The input file to process
    zip_path (str): The path to the zip file
    """

    try:
        outs = ""
        for input_file_path in input_file_paths:
            with fsspec.open(zip_path, "rb") as f:
                with zipfile.ZipFile(f) as z:
                    with z.open(input_file_path) as f:
                        content = f.read().decode('utf-8', 'ignore')
                        outs += json.dumps({
                            "id": f"ar5iv/{input_file_path}",             # MANDATORY: source-specific identifier
                            "text": content,           # MANDATORY: textual content of the document
                            "source": "ar5iv",         # MANDATORY: source of the data, such as peS2o, common-crawl, etc.
                            "added": datetime.datetime.now().isoformat(),          # OPTIONAL: timestamp ai2 acquired this data
                            "created": datetime.datetime(2024, 4, 1).isoformat()         # OPTIONAL: timestamp when orig document was created (best-guess if not available)
                        }) + "\n"
        out_file = "marin-data/processed_test/ar5iv/"+"/".join(input_file_path.split("/")[:2]) + f"_{counts}_html.jsonl.gz"
        print(out_file)
        with fsspec.open(get_gcs_path(out_file), 'wb', compression='gzip') as outputf:
            outputf.write(outs.encode('utf-8'))
    except FileNotFoundError as e:
        print(f"Error reading the zip file: {e}")
        return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert ar5iv to markdown.")
    parser.add_argument('--input_dir', type=str, help='Path to the ar5iv zip file', required=True)

    args = parser.parse_args()
    gfs = fsspec.filesystem("gcs")
    zip_file = get_gcs_path(args.input_dir)
    with gfs.open(zip_file, "rb") as f:
        with zipfile.ZipFile(f) as z:
            files = z.namelist()
            files = list(filter(lambda x:x.endswith(("html")), files))
    n = 256  # Number of files to process in parallel
    files.sort()
    all_files = []
    counts = []
    dictionary = {}
    for i, file in enumerate(files):
        out_file = "/".join(file.split("/")[:2])
        if len(all_files) and len(all_files[-1]) < n and out_file == "/".join(files[i-1].split("/")[:2]):
            all_files[-1].append(file)
        else:
            all_files.append([file])
            if out_file not in dictionary:
                dictionary[out_file] = 0
            else:
                dictionary[out_file] += 1
            counts.append(dictionary[out_file])
    

    MAX_NUM_PENDING_TASKS = 300  # Max number of html files we want to process in pending state
    ray.init()
    result_refs = []

    for idx, file in enumerate(all_files):
        if len(result_refs) > MAX_NUM_PENDING_TASKS:
            # update result_refs to only
            # track the remaining tasks.
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                print(f"Error processing the group: {e}")
                continue
        print(f"Starting Processing for the ar5iv file: {file}")
        result_refs.append(load_ar5iv_html.remote(file, zip_file, counts[idx]))

    # Wait for all the tasks to finish
    try:
        ray.get(result_refs)
    except Exception as e:
        print(f"Error processing the group: {e}")