"""
posts_to_markdown.py

Converts raw StackExchange dumps (.7z files from `https://archive.org/download/stackexchange`) to sequences of
(question, answer) pairs formatted in Markdown. We only keep the accepted answer for simplicity, rather than the
highest voted answer (or all answers).

StackExchange dumps typically are encoded as a single 7z-compressed file of XML entries, with top-level entries
for fields such as "Posts", "Comments", or "Users". We only process the Posts field!

Note that StackOverflow is in a slightly different format than the other StackExchange dumps, consisting of "expanded"
files that flatten out the top-level XML fields (e.g., `stackoverflow.com-Badges.7z`,`stackoverflow.com-Comments.7z`);
again, we only use the "Posts" data.

Run with:
    - [Local] python scripts/stackexchange/posts_to_markdown.py
    - [Ray] ray job submit --no-wait --address=http://127.0.0.1:8265 --working-dir . -- \
            python scripts/stackexchange/posts_to_markdown
        => Assumes that `ray dashboard infra/marin-cluster.yaml` running in a separate terminal (port forwarding)!
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import draccus
import fsspec
import py7zr
import ray
from tqdm import tqdm

from marin.domains.stackexchange.extract import (
    StackExchangeMarkdownFormat,
    extract_stackexchange_threads,
    markdownify_thread,
)
from marin.overwatch import initialize_overwatch
from marin.schema.document import Document

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class PostsToMarkdownConfig:
    # fmt: off
    stackexchange_urls_tsv: Path = Path(                    # Path to TSV file indexing StackExchange subdomains
        "scripts/stackexchange/stackexchange-urls.tsv"
    )

    # GCS Input / Output Paths
    raw_data_gcs_path: str | os.PathLike = (                # GCS Path with raw StackExchange dumps per subdomain (.7z)
        "gs://marin-data/raw/stackexchange/archive.org/download/stackexchange"
    )
    output_gcs_path: str | os.PathLike = (                  # GCS Path to write markdownified `.jsonl.gz` files
        "gs://marin-data/processed/stackexchange/2024-04-02"
    )

    # StackExchange Parameters
    markdown_format: str = "complete"                       # Markdown format in < atomic | qa-pair | complete >

    min_vote_threshold: int = -1024                         # Minimum number of votes for keeping questions/answers
    max_answer_threshold: int = 512                         # Maximum number of highest-voted answers to keep per thread

    # Debugging
    local_mode: bool = False                                # Whether to run things locally (disable `ray`)

    def __post_init__(self) -> None:
        # See `marin.domains.stackexchange.StackExchangeMarkdownFormat` for various formatting options
        self.markdown_format = StackExchangeMarkdownFormat(self.markdown_format)

    # fmt: on


@ray.remote
def run_post_markdownification(
    gcs_raw_input: str | os.PathLike,
    gcs_markdown_output: str | os.PathLike,
    subdomain: str,
    markdown_format: StackExchangeMarkdownFormat,
    min_vote_threshold: int,
    max_answer_threshold: int,
) -> bool:

    # Initialize GCS File System =>> `fsspec`
    fs = fsspec.filesystem("gcs")

    # [Short-Circuit] If .LEDGER.COMPLETE exists, we're done; otherwise, get set of processed thread IDs
    gcs_ledger_prefix = f"{gcs_markdown_output}.LEDGER"
    if fs.exists(f"{gcs_ledger_prefix}-COMPLETE"):
        return True

    # Otherwise, load "processed_ids" from appropriate Ledger version (as newline-separate .txt)
    #   =>> *IMPORTANT* :: Files on GCS are *immutable* so "appending" to a file isn't a thing... "versioning" instead!
    processed_ids, version = set(), 0
    while fs.exists(f"{gcs_ledger_prefix}-v{version}"):
        with fs.open(f"{gcs_ledger_prefix}-v{version}", "r") as f:
            processed_ids.update(set(f.read().splitlines()))

        # Bump Version
        version += 1

    # Version `gcs_markdown_output` and `gcs_ledger_path`
    vgcs_ledger_path = f"{gcs_ledger_prefix}-v{version}"
    vgcs_markdown_output = os.path.join(
        os.path.dirname(gcs_markdown_output), f"v{version}-{os.path.basename(gcs_markdown_output)}"
    )

    # Read "Posts" content from 7z-compressed XML
    with fs.open(gcs_raw_input, "rb") as f:
        with py7zr.SevenZipFile(f, mode="r") as archive:
            post_xml_content = archive.read(targets=["Posts.xml"])

    # Extract Questions/Answers from each Post =>> Convert to Markdown =>> Write `Document` to GCS
    with fs.open(vgcs_markdown_output, "w", compression="gzip") as out, fs.open(vgcs_ledger_path, "w") as ledger:
        for thread_metadata in tqdm(
            extract_stackexchange_threads(
                subdomain,
                post_xml_content["Posts.xml"],
                processed_ids=processed_ids,
                min_vote_threshold=min_vote_threshold,
                max_answer_threshold=max_answer_threshold,
            ),
            desc=f"[*] Processing StackExchange :: {subdomain} in {markdown_format.name} Format",
        ):
            for doc_id, markdown in markdownify_thread(thread_metadata, markdown_format):
                doc = Document(
                    id=doc_id,
                    text=markdown,
                    source="stackexchange",
                    added=datetime.now(timezone.utc),
                    created=thread_metadata.creation_time_utc,
                    metadata=thread_metadata,
                )

                # Write Document as JSON =>> Update Ledger w/ `thread_metadata.id`
                out.write(f"{doc.model_dump_json()}\n")
                ledger.write(f"{thread_metadata.id}\n")

    # Write `LEDGER.COMPLETE` and return True
    fs.touch(f"{gcs_ledger_prefix}-COMPLETE")

    return True


@draccus.wrap()
def posts_to_markdown(cfg: PostsToMarkdownConfig) -> None:
    overwatch.info("Converting StackExchange Posts to Markdown")

    # Initialize Connection to Cluster
    ray.init()

    # Load StackExchange Index from TSV and Parse Subdomains (i.e. "chemistry.stackexchange.com.7z" --> "chemistry")
    with open(cfg.stackexchange_urls_tsv, "r") as f:
        dumps_7z = [os.path.basename(p) for p in f.read().splitlines() if p.endswith(".7z")]
        subdomains = [re.match(r"(.+?)\.(stackexchange|net|com)", url).group(1) for url in dumps_7z]

    # Set GCS Output Paths =>> One `.jsonl.gz` per Subdomain
    gcs_output_paths = [f"{subdomain}.jsonl.gz" for subdomain in subdomains]

    # Invoke / Dispatch Tasks =>> one per StackExchange Subdomain
    success_refs = []
    for idx in range(len(subdomains)):
        success_refs.append(
            run_post_markdownification.remote(
                os.path.join(cfg.raw_data_gcs_path, dumps_7z[idx]),
                os.path.join(cfg.output_gcs_path, gcs_output_paths[idx]),
                subdomains[idx],
                cfg.markdown_format,
                cfg.min_vote_threshold,
                cfg.max_answer_threshold,
            )
        )

    # Resolve / Verify Task Successes
    successes = ray.get(success_refs)

    overwatch.info(f"Job Complete with {successes.count(True)} Successes")


if __name__ == "__main__":
    posts_to_markdown()
