import json
import logging

import html2text
import fsspec
import pyarrow.parquet as pq
import trafilatura
import readabilipy

from lookup_cc import fetch_page_from_cc, search_cc_index

# Setup logging
logger = logging.getLogger(__name__)

h = html2text.HTML2Text()
h.ignore_links = False  # Optionally ignore links
h.body_width = 0  # Optionally disable line wrapping
# TODO: html2text uses [code]...[/code] for code blocks. would prefer github markdown style
h.mark_code = True  # Optionally convert code blocks to markdown
h.include_sup_sub = True  # Optionally include <sup> and <sub> tags
h.unicode_snob = True  # Optionally protect links from being included in the output
h.pad_tables = False  # We disable b/c it seems like a silly thing for the LLM to do

def process_single_doc(url, snapshot_id):
    # Implement or integrate your methods for fetching the WARC location and HTML content
    warc_location = search_cc_index(url, snapshot_id)
    if not warc_location:
        logger.warning(f"WARC location not found for URL: {url}")
        return None

    html_content = fetch_page_from_cc(warc_location)
    if not html_content:
        logger.warning(f"Failed to fetch content for URL: {url}")
        return None

    cleaned_article = readabilipy.simple_json_from_html_string(html_content, use_readability=True)
    if cleaned_article:
        cleaned_html = cleaned_article["content"]
        cleaned_md = h.handle(cleaned_html)
    else:
        logger.warning(f"Failed to extract content from URL: {url}")
        return None

    # plain_text = '\n\n'.join(p['text'] for p in cleaned_article["plain_text"])

    return {
        "content": cleaned_md,
        "title": cleaned_article["title"],
        "byline": cleaned_article["byline"],
        "date": cleaned_article["date"],
        "plain_content": cleaned_article["plain_content"],
    }

_URL_BASE = "https://data.together.xyz/redpajama-data-v2/v1.0.0"
_NUM_SHARDS = 5000
_LANGUAGES = ("en", "de", "fr", "es", "it")

def process_files(snapshot, n, lang, part):
    base_tag = f"{snapshot}/{n:04d}/{lang}_{part}"
    qs_file = f"{_URL_BASE}/quality_signals/{base_tag}.signals.json.gz"
    dupe_file = f"{_URL_BASE}/duplicates/{base_tag}.duplicates.parquet"

    # Load duplicates
    try:
        with fsspec.open(dupe_file, "rb", compression="infer") as df:
            duplicates = set(pq.read_table(df, columns=["doc_id"], use_pandas_metadata=False)["doc_id"].to_pylist())
    except Exception as e:
        logger.exception(f"No duplicate ids found for {base_tag}: {e}")
        duplicates = set()

    # Process quality signals
    try:
        with fsspec.open(qs_file, "r", compression="infer", encoding="utf-8") as qf:
            for row, line in enumerate(qf):
                qs = json.loads(line)
                doc_id = f"{base_tag}.json.gz/{row}"
                url = qs["metadata"]["url"]
                snapshot_id = qs["metadata"]["snapshot_id"]
                is_duplicate = doc_id in duplicates

                cleaned_article = process_single_doc(url, snapshot_id)
                if cleaned_article:
                    # Yield the cleaned text along with other metadata and the quality signals
                    yield {
                        "id": qs["id"],
                        "id_int": qs["id_int"],
                        "metadata": qs["metadata"],
                        "content": cleaned_article["content"],
                        "title": cleaned_article["title"],
                        "byline": cleaned_article["byline"],
                        "date": cleaned_article["date"],
                        "plain_content": cleaned_article["plain_content"],
                        # "plain_text": cleaned_article["plain_text"],
                        "is_duplicate": is_duplicate,
                        "quality_signals": qs["quality_signals"]  # Include the entire quality signals data
                    }
                else:
                    logger.warning(f"Content could not be processed for URL: {url}")

    except Exception as e:
        logger.exception(f"Failed to process files for {base_tag}: {e}")


if __name__ == "__main__":
    lang = "en"
    shard = 0
    snapshot = "2023-06"
    part = "head"

    for n in range(_NUM_SHARDS):
        for doc in process_files(snapshot, n, lang, part):
            print(doc)