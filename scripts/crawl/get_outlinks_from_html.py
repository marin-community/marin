#!/usr/bin/env python3
"""
Get the out-bound links from Dolma-format examples containing HTML.

Running on OpenWebMath:

```
python scripts/crawl/get_outlinks_from_html.py \
    --html_input_path gs://marin-us-central2/documents/open-web-math-fde8ef8/html/ \
    --outlinks_output_path gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8/
```

```
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- \
    python scripts/crawl/get_outlinks_from_html.py \
    --html_input_path gs://marin-us-central2/documents/open-web-math-fde8ef8/html/ \
    --outlinks_output_path gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8/
```

Running on FineWeb-Edu:

```
for fineweb_edu_dump_html_path in $(gcloud storage ls gs://marin-us-central2/documents/fineweb-edu/html); do
    dump_name=$(basename -- ${fineweb_edu_dump_html_path})
    echo "ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- \
    python scripts/crawl/get_outlinks_from_html.py \
    --html_input_path ${fineweb_edu_dump_html_path} \
    --outlinks_output_path gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu/${dump_name}"
done
```
"""

import json
import logging
import os
import pathlib
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import draccus
import fsspec
import ray
from bs4 import BeautifulSoup

from marin.core.runtime import cached_or_construct_output
from marin.utils import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class OutlinksExtractionConfig:
    html_input_path: str
    outlinks_output_path: str


def is_internal_link(base_url, target_url):
    # Parse the base URL
    base_parsed = urlparse(base_url)
    base_host = base_parsed.netloc.lstrip("www.")

    # Parse the target URL
    target_parsed = urlparse(target_url)

    # Check if the target URL is relative
    if not target_parsed.netloc:
        return True

    target_host = target_parsed.netloc.lstrip("www.")

    # Compare the hosts
    return base_host == target_host


def is_parseable(link):
    try:
        result = urlparse(link)
        # Check if the URL has a valid scheme and netloc
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


@ray.remote(
    memory=4 * 1024 * 1024 * 1024,
    runtime_env={
        "pip": [
            "resiliparse_dom @ git+https://github.com/nelson-liu/chatnoir-resiliparse@58247de82b4d881223435113f1a07a86ad66494c#egg=resiliparse_dom&subdirectory=resiliparse_dom",
            "courlan",
            "w3lib",
        ]
    },
)  # 4 GB  # 4 GB
@cached_or_construct_output(
    success_suffix="SUCCESS", verbose=False
)  # We use this decorator to make this function idempotent
def process_one_batch(html_paths_batch: list[str], output_path: str):
    """
    Takes in a batch of input files, extracts the outlinks, and writes them to output_path.

    Args:
    html_paths_batch (list[str]): Paths of HTML files to extract outlinks from.
    output_path (str): Path to write JSONL file with outlinks.
    """
    import w3lib.url
    from courlan import check_url
    from resiliparse_dom.extract.html2text import extract_main_dom_tree

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    with fsspec.open(output_path, "w", compression="gzip") as fout:
        num_failed_to_parse = 0
        num_total = 0
        for html_path in html_paths_batch:
            with fsspec.open(html_path, "rt", compression="gzip") as fin:
                for line in fin:
                    record = json.loads(line)
                    html_str = record.get("html", "")
                    url = record.get("metadata", {}).get("url", "")

                    if not html_str or not url:
                        continue

                    num_total += 1
                    # Get all the outbound links in the HTML
                    try:
                        parsed_html = BeautifulSoup(html_str, "html.parser")
                    except Exception:
                        # Skip documents that don't parse
                        num_failed_to_parse += 1
                        continue
                    unfiltered_outbound_links = set()
                    for link in parsed_html.find_all("a", href=True):
                        href = link.get("href")
                        if href and is_parseable(href):
                            absolute_link_target = urljoin(url, href)
                            canonical_link = w3lib.url.canonicalize_url(absolute_link_target)
                            unfiltered_outbound_links.add(canonical_link)

                    # Heuristically filter the outbound_links
                    outbound_links = set()
                    for link in unfiltered_outbound_links:
                        # check_url removes invalid links (e.g., http://666.0.0.1/)
                        # and those that don't point to HTML (e.g., .mp3 extension, etc.)
                        if (
                            check_url(
                                link,
                                strict=False,
                                with_redirects=False,
                                language=None,
                                with_nav=True,
                                trailing_slash=True,
                            )
                            is not None
                        ):
                            outbound_links.add(link)

                    # Now, get all of the outbound links in the main text
                    # Need to convert to string here, since bs4 can't directly
                    # use the resiliparse.parse.html.HTMLTree
                    main_text_dom_str = str(
                        extract_main_dom_tree(
                            html_str,
                            main_content=True,
                        )
                    ).strip()
                    if not main_text_dom_str:
                        continue

                    # Get all the outbound links in the HTML
                    try:
                        main_text_with_links = BeautifulSoup(main_text_dom_str, "html.parser")
                    except Exception:
                        # Skip documents that don't parse
                        num_failed_to_parse += 1
                        continue
                    main_text_outbound_links = set()
                    for link in main_text_with_links.find_all("a", href=True):
                        href = link.get("href")
                        if href and is_parseable(href):
                            absolute_link_target = urljoin(url, href)
                            canonical_link = w3lib.url.canonicalize_url(absolute_link_target)
                            main_text_outbound_links.add(canonical_link)

                    # Filter outbound_links to allowed protocols
                    allowed_protocols = {"http", "https"}
                    outbound_links = [link for link in outbound_links if urlparse(link).scheme in allowed_protocols]

                    # Prepare the list of outbound link records
                    for link in outbound_links:
                        is_internal = is_internal_link(url, link)
                        in_main_content = link in main_text_outbound_links
                        fout.write(
                            json.dumps(
                                {
                                    "page_url": url,
                                    "link_target": link,
                                    "is_internal_link": is_internal,
                                    "in_main_content": in_main_content,
                                }
                            )
                            + "\n"
                        )
        logger.info(f"Failed to parse {num_failed_to_parse} out of {num_total} records")


def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


@ray.remote(memory=32 * 1024 * 1024 * 1024)
def get_shards_indices_to_process(shard_path: str):
    # Get the HTML files (of form <int index>.jsonl.gz) and sort by the integer index.
    # We sort to ensure that the sharding is reproducible.
    html_path_indices: list[int] = [
        int(pathlib.Path(path).name.removesuffix(".jsonl.gz"))
        for path in fsspec_glob(os.path.join(shard_path, "*.jsonl.gz"))
    ]
    html_path_indices: list[int] = sorted(html_path_indices)
    return html_path_indices


@draccus.wrap()
def get_outlinks_from_html(cfg: OutlinksExtractionConfig):
    get_shards_ref = get_shards_indices_to_process.remote(cfg.html_input_path)
    shard_indices = ray.get(get_shards_ref)

    # Group into chunks of 1000 WARCs each
    # open-web-math has ~3M WARCs in total, which yields 3000 resharded chunks
    refs = []
    for i, html_shard_indices_batch in enumerate(batched(shard_indices, 1000)):
        output_path = os.path.join(cfg.outlinks_output_path, f"{i}_links.jsonl.gz")
        html_path_batch = [
            os.path.join(cfg.html_input_path, f"{shard_index}.jsonl.gz") for shard_index in html_shard_indices_batch
        ]
        refs.append(process_one_batch.remote(html_path_batch, output_path))
    logger.info(f"Submitted {len(refs)} tasks")

    # Wait for the tasks to finish
    _ = ray.get(refs)


if __name__ == "__main__":
    get_outlinks_from_html()
