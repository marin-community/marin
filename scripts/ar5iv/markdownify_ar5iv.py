"""
markdownify_ar5iv.py

Tries to read from jsonl.gz file, convert to md, and write the contents to a jsonl.gz file.

Run with:
  - [Local] python tests/test_ray_cluster.py
  - [Ray] ray job submit --no-wait --address=http://127.0.0.1:8265 --working-dir . -- python scripts/ar5iv/markdownify_ar5iv.py
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


@ray.remote(memory=1024 * 1024 * 1024)  # 1 GB
def markdownify_ar5iv_html(file):
    """
    Takes in the input file and processes it to get the html content.
    Args:
    input_file_path (str): The input file to process
    zip_path (str): The path to the zip file
    """

    try:
        outs = ""
        print(f"Starting Processing for the ar5iv file: {html}")
        with fsspec.open(get_gcs_path(file), "rb", compression="gzip") as outputf:
            # print(input_file_paths)
            for _ in range(n):
                line = outputf.readline()
                if not line:
                    break
                html_blob = json.loads(line)
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
        out_file = file.replace("html_clean", "md").replace("ar5iv_clean", "ar5iv_md")
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
        result_refs.append(markdownify_ar5iv_html.remote(html))

    # Wait for all the tasks to finish
    try:
        ray.get(result_refs)
    except Exception as e:
        print(f"Error processing the group: {e}")
