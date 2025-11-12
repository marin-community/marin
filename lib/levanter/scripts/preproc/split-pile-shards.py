# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
from pathlib import Path

import fsspec
import tqdm


OUT_PATH = "gs://levanter-data/pile-domains"

categories_to_out_names = {
    "ArXiv": "arxiv",
    "BookCorpus2": "books2",
    "Books3": "books3",
    "DM Mathematics": "dm_math",
    "Enron Emails": "enron",
    "EuroParl": "europarl",
    "FreeLaw": "freelaw",
    "Github": "github",
    "Gutenberg (PG-19)": "pg_19",
    "HackerNews": "hackernews",
    "NIH ExPorter": "nih",
    "OpenSubtitles": "opensubtitles",
    "OpenWebText2": "owt2",
    "PhilPapers": "philpapers",
    "Pile-CC": "pile_cc",
    "PubMed Abstracts": "pubmed_abs",
    "PubMed Central": "pubmed_central",
    "StackExchange": "stack_exchange",
    "USPTO Backgrounds": "uspto",
    "Ubuntu IRC": "ubuntu_irc",
    "Wikipedia (en)": "wiki_en",
    "YoutubeSubtitles": "youtube_subtitles",
}


def format_category(category):
    return categories_to_out_names[category]


def process_file(input_file_path):
    base_file = Path(input_file_path).stem
    compressors = {}

    with fsspec.open(input_file_path, "r", compression="infer") as text_stream:
        for line in tqdm.tqdm(text_stream):
            if not line.strip():
                continue  # Skip empty lines

            # Decode line to string and load as JSON
            data = json.loads(line)
            category = data["meta"]["pile_set_name"]
            category = format_category(category)
            output_file_path = os.path.join(OUT_PATH, category, f"{base_file}.zst")

            # Check if compressor exists for this category, if not create it
            if category not in compressors:
                # output_file = open(output_file_path, 'wb')
                output_file = fsspec.open(str(output_file_path), "wb", compression="infer").open()
                print("opened", output_file_path)
                compressors[category] = output_file

            # Write to the compressor
            compressors[category].write(line.encode("utf-8"))
            compressors[category].flush()

    # Close all open compressors
    for compressor in compressors.values():
        compressor.close()


if __name__ == "__main__":
    for path in sys.argv[1:]:
        process_file(path)
