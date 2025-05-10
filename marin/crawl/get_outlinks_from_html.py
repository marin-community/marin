#!/usr/bin/env python3
"""
Get the out-bound links from Dolma-format examples containing HTML.

Running on OpenWebMath:

```
python marin/run/ray_run.py \
    --pip_deps 'resiliparse_dom @ git+https://github.com/nelson-liu/chatnoir-resiliparse@58247de82b4d881223435113f1a07a86ad66494c#egg=resiliparse_dom&subdirectory=resiliparse_dom,courlan,w3lib,cchardet,beautifulsoup4,lxml' \
    --no_wait -- \
    python marin/crawl/get_outlinks_from_html.py \
    --html_input_path gs://marin-us-central2/documents/open-web-math-fde8ef8/html/ \
    --prefix openwebmath \
    --outlinks_output_path gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8/
```

Running on FineWeb-Edu:

```
for fineweb_edu_dump_html_path in $(gcloud storage ls gs://marin-us-central2/documents/fineweb-edu/html); do
    dump_name=$(basename -- ${fineweb_edu_dump_html_path})

    python marin/run/ray_run.py \
        --pip_deps 'resiliparse_dom @ git+https://github.com/nelson-liu/chatnoir-resiliparse@58247de82b4d881223435113f1a07a86ad66494c#egg=resiliparse_dom&subdirectory=resiliparse_dom,courlan,w3lib,cchardet,beautifulsoup4,lxml' \
        --no_wait -- \
        python marin/crawl/get_outlinks_from_html.py \
        --html_input_path ${fineweb_edu_dump_html_path} \
        --prefix fineweb_edu \
        --outlinks_output_path gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu/${dump_name}
done
```
"""  # noqa: E501

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
    prefix: str
    outlinks_output_path: str


def is_internal_link(base_url: str, target_url: str):
    """
    Given an absolute base URL (e.g., the URL of a page) and an href target
    (absolute or relative), check if the linked page is an internal link of
    the base url. For example, if the base url is
    https://en.wikipedia.org/wiki/Marin_County,_California and the target URL is
    https://en.wikipedia.org/wiki/San_Francisco_Bay_Area, then it is an internal link.
    If the target URL href is "San_Francisco_Bay_Area", then it's also an internal link.
    If the target URL href is https://www.presidioyachtclub.org/ , then it isn't an internal
    link to the base url.

    Args:
    base_url (str): base URL to use in determining if a target is internal
    target_url (str): target link href to check.

    Returns:
    True if target_url is internal to base_url, else False
    """
    # Parse the base URL
    base_parsed = urlparse(base_url)
    base_host = base_parsed.netloc.removeprefix("www.")

    # Parse the target URL
    target_parsed = urlparse(target_url)

    # Check if the target URL is relative
    if not target_parsed.netloc:
        return True

    target_host = target_parsed.netloc.removeprefix("www.")

    # Compare the hosts
    return base_host == target_host


def is_absolute_link_parseable(link: str) -> bool:
    """
    Takes in an absolute link (i.e., not a relative href target) and returns whether
    the link is parseable or not.

    Args:
    link (str): Absolute link to check.

    Returns:
    True if link is parseable, else False.
    """
    try:
        result = urlparse(link)
        # Check if the URL has a valid scheme and netloc
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


@ray.remote(
    memory=4 * 1024 * 1024 * 1024,
)
@cached_or_construct_output(
    success_suffix="SUCCESS", verbose=False
)  # We use this decorator to make this function idempotent
def process_one_batch(input_path: str, output_path: str):
    """
    Takes in an input file, extracts the outlinks, and writes them to output_path.

    Args:
    input_path (str): Path of HTML file (Dolma-format JSONL) to extract outlinks from.
    output_path (str): Path to write JSONL file with outlinks.
    """
    import w3lib.url
    from courlan import check_url
    from resiliparse_dom.extract.html2text import extract_main_dom_tree

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    with fsspec.open(output_path, "w", compression="gzip") as fout:
        num_docs_failed_to_parse = 0
        num_links_failed_to_parse = 0
        num_total_links = 0
        num_total = 0
        with fsspec.open(input_path, "rt", compression="gzip") as fin:
            for line in fin:
                record = json.loads(line)
                html_str = record.get("html", "")
                url = record.get("metadata", {}).get("url", "")

                if not html_str or not url:
                    continue

                num_total += 1
                # Get all the outbound links in the HTML
                try:
                    parsed_html = BeautifulSoup(html_str, "lxml")
                except Exception:
                    # Skip documents that don't parse
                    num_docs_failed_to_parse += 1
                    continue

                unjoinable_links: set[tuple[str, str]] = set()
                unfiltered_outbound_links = set()
                for link in parsed_html.find_all("a", href=True):
                    num_total_links += 1
                    href = link.get("href")
                    if href:
                        try:
                            absolute_link_target = urljoin(url, href)
                        except Exception:
                            unjoinable_links.add((url, href))
                            num_links_failed_to_parse += 1
                            continue
                        if is_absolute_link_parseable(absolute_link_target):
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
                    main_text_with_links = BeautifulSoup(main_text_dom_str, "lxml")
                except Exception:
                    # Skip documents that don't parse
                    num_docs_failed_to_parse += 1
                    continue
                main_text_outbound_links = set()
                for link in main_text_with_links.find_all("a", href=True):
                    href = link.get("href")
                    if href:
                        # If we've already seen this link and know it's unjoinable,
                        # just skip it
                        if (url, href) in unjoinable_links:
                            continue

                        # Check again, since there seem to be some cases where
                        # main text extraction mutates the link target
                        try:
                            absolute_link_target = urljoin(url, href)
                        except Exception:
                            continue
                        if is_absolute_link_parseable(absolute_link_target):
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
        logger.info(
            f"Failed to parse {num_docs_failed_to_parse} out of {num_total} records\n"
            f"Failed to parse {num_links_failed_to_parse} links out of {num_total_links} links"
        )


@ray.remote(memory=32 * 1024 * 1024 * 1024)
def get_shards_indices_to_process(shard_path: str, prefix: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    shard_indices: list[int] = [
        int(pathlib.Path(path).name.removesuffix(".jsonl.gz").removeprefix(f"{prefix}_"))
        for path in fsspec_glob(os.path.join(shard_path, f"{prefix}_*.jsonl.gz"))
    ]
    shard_indices = sorted(shard_indices)
    logger.info(f"Found {len(shard_indices)} shards to process")
    return shard_indices


@draccus.wrap()
def get_outlinks_from_html(cfg: OutlinksExtractionConfig):
    logger.info(f"Getting outlinks from HTML for {cfg.prefix} at {cfg.html_input_path}")
    shard_indices = ray.get(get_shards_indices_to_process.remote(cfg.html_input_path, cfg.prefix))

    logger.info(f"Processing {len(shard_indices)} shards")

    refs = []
    for i, html_shard_index in enumerate(shard_indices):
        input_path = os.path.join(cfg.html_input_path, f"{cfg.prefix}_{html_shard_index}.jsonl.gz")
        output_path = os.path.join(cfg.outlinks_output_path, f"{i}_links.jsonl.gz")
        refs.append(process_one_batch.remote(input_path, output_path))
    logger.info(f"Submitted {len(refs)} tasks")

    # Wait for the tasks to finish
    ray.get(refs)


if __name__ == "__main__":
    get_outlinks_from_html()
