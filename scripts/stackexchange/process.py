"""
stackexchange/process.py

Script for converting raw StackExchange dumps (.7z files from `https://archive.org/download/stackexchange`) to sequences
of (question, answer) pairs formatted in Markdown. We only keep the accepted answer for simplicity, rather than the
highest voted answer (or all answers).

StackExchange dumps typically are encoded as a single 7z-compressed file of XML entries, with top-level entries
for fields such as "Posts", "Comments", or "Users". We only process the Posts field!

Note that StackOverflow is in a slightly different format than the other StackExchange dumps, consisting of "expanded"
files that flatten out the top-level XML fields (e.g., `stackoverflow.com-Badges.7z`,`stackoverflow.com-Comments.7z`);
again, we only use the "Posts" data.

Run with:
    - [Local] python scripts/stackexchange/process.py
    - [Ray] ray job submit --address=http://127.0.0.1:8265 --working-dir . --no-wait -- \
            python scripts/stackexchange/process.py
"""

import json
import logging
import os.path
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import draccus
import fsspec
import py7zr
import ray

from marin.core.runtime import cached_or_construct_output
from marin.domains.stackexchange.utils import (
    StackExchangeMarkdownFormat,
    extract_stackexchange_threads,
    markdownify_thread,
)
from marin.utils import fsspec_glob

# Initialize Logger
logger = logging.getLogger(__name__)


# === OVERRIDES (Subdomains w/ Higher Resource Requirements) ===
RAY_MEMORY_OVERRIDES = {
    "askubuntu": 16 * 1024 * 1024 * 1024,
    "serverfault": 16 * 1024 * 1024 * 1024,
    "superuser": 16 * 1024 * 1024 * 1024,
    "es.stackoverflow": 16 * 1024 * 1024 * 1024,
    "pt.stackoverflow": 16 * 1024 * 1024 * 1024,
    "ru.stackoverflow": 16 * 1024 * 1024 * 1024,
    "math": 64 * 1024 * 1024 * 1024,
    "stackoverflow": 64 * 1024 * 1024 * 1024,
}


@ray.remote(memory=4 * 1024 * 1024 * 1024, num_cpus=1)  # 4 GB of RAM, 1 CPU by Default
@cached_or_construct_output(success_suffix="SUCCESS")  # Make idempotent / setup ledger for easy resumption
def post_to_md(
    input_file_path: str,
    output_file_path: str,
    subdomain: str,
    markdown_format: StackExchangeMarkdownFormat = StackExchangeMarkdownFormat.ATOMIC,
    min_vote_threshold: int = -1024,
    max_answer_threshold: int = 512,
) -> bool:
    with (
        fsspec.open(input_file_path, "rb") as f,
        fsspec.open(output_file_path, "wt", compression="gzip") as output_jsonl_gz,
    ):
        # Extract `Posts.xml` content from .7z archive
        with py7zr.SevenZipFile(f, mode="r") as archive:
            posts_content = archive.read(targets=["Posts.xml"])["Posts.xml"]

        # Extract Questions/Answers from each Post =>> Convert to Markdown =>> Write to `jsonl.gz`
        for thread_data in extract_stackexchange_threads(
            subdomain, posts_content, min_vote_threshold=min_vote_threshold, max_answer_threshold=max_answer_threshold
        ):
            for doc_id, markdown in markdownify_thread(thread_data, markdown_format=markdown_format):
                doc = dict(
                    id=doc_id,
                    text=markdown,
                    source="stackexchange",
                    added=datetime.now(timezone.utc).isoformat(),
                    created=thread_data["creation_time_utc"],
                    metadata=thread_data,
                )

                # Write Document as JSON
                output_jsonl_gz.write(f"{json.dumps(doc)}\n")

    return True


# === Main ===


@dataclass
class ProcessStackExchangeConfig:
    # fmt: off
    input_dir: str = (                                      # GCS Path with StackExchange dumps per subdomain (.7z)
        "gs://marin-data/raw/stackexchange/archive.org/download/stackexchange"
    )
    output_dir: str = (                                     # GCS Path to write Dolma-formatted markdown files
        "gs://marin-data/processed/stackexchange/v2024-04-02"
    )

    # StackExchange Parameters
    markdown_format: StackExchangeMarkdownFormat = (        # Format specifier for "linearizing" StackExchange threads
        StackExchangeMarkdownFormat.COMPLETE
    )

    min_vote_threshold: int = -1024                         # Minimum number of votes for keeping questions/answers
    max_answer_threshold: int = 512                         # Maximum number of high-voted answers to keep per thread
    # fmt: on


@draccus.wrap()
def main(config: ProcessStackExchangeConfig) -> None:
    logger.info(f"Processing StackExchange XML (.7z) Dumps to Markdown (Format = `{config.markdown_format.value}`)")

    # Initialize Connection to Cluster
    ray.init()

    # GCS Output Path should reflect Markdown format
    output_dir = os.path.join(config.output_dir, f"md-{config.markdown_format.value}")

    # === This is basically a rewrite of `map_files_in_directory` so we can have finer-grained control ===

    # Handle StackOverflow's unique format -- purge the -Badges/-Comments/-<Whatever> files!
    files = [f for f in fsspec_glob(os.path.join(config.input_dir, "*.7z")) if "stackoverflow.com-" not in f]
    files.append(os.path.join(config.input_dir, "stackoverflow.com-Posts.7z"))

    # Invoke Ray Functions --> track `refs`
    responses = []
    for input_file in files:
        subdomain = re.match(r"(.+?)\.(stackexchange|net|com)", os.path.basename(input_file)).group(1)
        output_file = os.path.join(output_dir, f"{subdomain}.jsonl.gz")

        # Handle RAM Overrides
        if subdomain in RAY_MEMORY_OVERRIDES:
            responses.append(
                post_to_md.options(memory=RAY_MEMORY_OVERRIDES[subdomain]).remote(
                    input_file, output_file, subdomain, markdown_format=config.markdown_format
                )
            )
        else:
            responses.append(
                post_to_md.remote(input_file, output_file, subdomain, markdown_format=config.markdown_format)
            )

    # Wait on Success
    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error Processing: {e}")


if __name__ == "__main__":
    main()
