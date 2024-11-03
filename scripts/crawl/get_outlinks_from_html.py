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
    ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- \
    python scripts/crawl/get_outlinks_from_html.py \
    --html_input_path ${fineweb_edu_dump_html_path} \
    --outlinks_output_path gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu/${dump_name} \
done
```
"""

import json
import logging
import os
from urllib.parse import urljoin, urlparse
import pathlib
from dataclasses import dataclass
from bs4 import BeautifulSoup

import draccus
import fsspec
import ray
import w3lib.url

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


@ray.remote(
    memory=4 * 1024 * 1024 * 1024,
    runtime_env={"pip": ["https://github.com/krypticmouse/chatnoir-resiliparse/tree/develop/resiliparse", "courlan"]},
)  # 4 GB
@cached_or_construct_output(success_suffix="SUCCESS")  # We use this decorator to make this function idempotent
def process_one_batch(html_paths_batch: list[str], output_path: str):
    """
    Takes in a batch of input files, extracts the outlinks, and writes them to output_path.

    Args:
    html_paths_batch (list[str]): Paths of HTML files to extract outlinks from.
    output_path (str): Path to write gzipped JSONL with outlinks.
    """
    from resiliparse.extract.html2text import extract_main_dom_tree as resiliparse_extract_main_dom_tree
    from courlan import check_url, validate_url

    with fsspec.open(output_path, "w", compression="gzip") as fout:
        for html_path in html_paths_batch:
            with fsspec.open(html_path, "r", compression="gzip") as fin:
                for line in fin:
                    record = json.loads(line)
                    html_str = record["html"]
                    url = record["metadata"]["url"]

                    if not html_str:
                        continue

                    # Get all the outbound links in the HTML
                    parsed_html = BeautifulSoup(html_str, "html.parser")
                    unfiltered_outbound_links = set()
                    for link in parsed_html.find_all("a", href=True):
                        if link["href"]:
                            # Make sure the link is absolute
                            absolute_link_target = urljoin(url, link["href"])
                            unfiltered_outbound_links.add(w3lib.url.canonicalize_url(absolute_link_target))

                    # Heuristically filter the outbound_links
                    outbound_links = set()
                    for link in unfiltered_outbound_links:
                        if check_url(link) is not None and validate_url(link)[0]:
                            outbound_links.add(link)

                    # Now, get all of the outbound links in the main text
                    main_text_with_links = BeautifulSoup(
                        resiliparse_extract_main_dom_tree(
                            html_str,
                            main_content=True,
                        ),
                        "html.parser",
                    )
                    main_text_outbound_links = set()
                    for link in main_text_with_links.find_all("a", href=True):
                        if link["href"]:
                            # Make sure the link is absolute
                            absolute_link_target = urljoin(url, link["href"])
                            main_text_outbound_links.add(w3lib.url.canonicalize_url(absolute_link_target))
                    # Write out the links to file
                    allowed_protocols = set(["http", "https"])
                    outbound_links = [link for link in outbound_links if urlparse(link).scheme in allowed_protocols]
                    fout.write(
                        json.dumps(
                            {
                                "page_url": url,
                                "outbound_links": [
                                    {
                                        "link_target": link,
                                        "is_internal_link": is_internal_link(url, link),
                                        "in_main_content": link in main_text_outbound_links,
                                    }
                                    for link in outbound_links
                                ],
                            }
                        )
                    )


def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


@draccus.wrap()
def get_outlinks_from_html(cfg: OutlinksExtractionConfig):
    # Get the HTML files (of form <int index>.jsonl.gz) and sort by the integer index
    html_paths = fsspec_glob(os.path.join(cfg.html_input_path, "*.jsonl.gz")).sort(
        key=lambda x: int(pathlib.Path(x).name.removesuffix(".jsonl.gz"))
    )

    # Group into chunks of 1000 WARCs each
    # open-web-math has ~3M WARCs in total, which yields 3000 resharded chunks
    refs = []
    for i, html_path_batch in enumerate(batched(html_paths, 1000)):
        output_path = os.path.join(cfg.outlinks_output_path, f"{i}_links.parquet")
        refs.append(process_one_batch.remote(html_path_batch, output_path))
    logger.info(f"Submitted {len(refs)} tasks")

    # Wait for the tasks to finish
    _ = ray.get(refs)


if __name__ == "__main__":
    get_outlinks_from_html()
