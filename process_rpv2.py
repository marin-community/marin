import gzip
import json
import logging
import os

import html2text
import trafilatura
import readabilipy

from lookup_cc import fetch_page_from_cc, search_cc_index
from rpv2 import NUM_SHARDS, iterate_rpv2_file

# Setup logging
logger = logging.getLogger(__name__)

h = html2text.HTML2Text()
h.ignore_links = False  # Optionally ignore links
h.body_width = 0  # Optionally disable line wrapping
# TODO: html2text uses [code]...[/code] for code blocks. would prefer github markdown style
h.mark_code = True  # Optionally convert code blocks to markdown
h.include_sup_sub = True  # Optionally include <sup> and <sub> tags
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

    trafilatura_result = trafilatura.extract(html_content, include_formatting=True, output_format="txt", include_images=True, include_links=True)
    trafilatura_xml = trafilatura.extract(html_content, output_format="xml")

    cleaned_article = readabilipy.simple_json_from_html_string(html_content, use_readability=True)
    if cleaned_article:
        cleaned_html = cleaned_article["content"]
        cleaned_md = h.handle(cleaned_html)
    else:
        logger.warning(f"Failed to extract content from URL: {url}")
        return None

    plain_text = '\n\n'.join(p['text'] for p in cleaned_article["plain_text"][-1:])

    return {
        "content": cleaned_md,
        "title": cleaned_article["title"],
        "byline": cleaned_article["byline"],
        "date": cleaned_article["date"],
        "plain_content": cleaned_article["plain_content"],
        "original_html": html_content,
        "readable_html": cleaned_html,
        "readable_text": plain_text,
        "patched_traf_content": trafilatura_result,
        "traf_xml": trafilatura_xml,
    }


def process_files(snapshot, n, lang, part):
   for doc_id, qs in iterate_rpv2_file(snapshot, n, lang, part):
        url = qs["metadata"]["url"]
        snapshot_id = qs["metadata"]["snapshot_id"]
        try:
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
                    "is_duplicate": qs["is_duplicate"],
                    "quality_signals": qs["quality_signals"],  # Include the entire quality signals data
                    "original_html": cleaned_article["original_html"],
                    "readable_html": cleaned_article["readable_html"],
                    "readable_text": cleaned_article["readable_text"],
                    # "traf_content": cleaned_article["traf_content"],
                    "traf_xml": cleaned_article["traf_xml"],
                    "patched_traf_content": cleaned_article["patched_traf_content"],
                }
            else:
                logger.exception(f"Content could not be processed for {doc_id} URL: {url}")
        except Exception as e:
            logger.exception(f"Error processing {doc_id}: {e}")

if __name__ == "__main__":
    lang = "en"
    snapshot = "2023-06"
    part = "middle"
    i = 0
    for n in range(NUM_SHARDS):
        with gzip.open(f"{lang}_{n:04d}.jsonl.gz", "wt") as f:
            for doc in process_files(snapshot, n, lang, part):
                i += 1
                f.write(json.dumps(doc) + "\n")
                id = doc["id_int"]
                os.makedirs(f"{lang}_{n:04d}_{id}", exist_ok=True)
                for key in ["content", "readable_html", "readable_text", "original_html", "patched_traf_content", "traf_xml"]:
                    suf = "md"
                    if "html" in key:
                        suf = "html"
                    elif "text" in key:
                        suf = "txt"
                    elif "xml" in key:
                        suf = "xml"
                    with open(f"{lang}_{n:04d}_{id}/{key}.{suf}", "w") as out_f:
                        out_f.write(str(doc[key]))

                qs = doc["quality_signals"]
                with open(f"{lang}_{n:04d}_{id}/quality_signals.json", "w") as out_f:
                    json.dump(qs, out_f, indent=2)

                if i % 100 == 0:
                    raise ValueError("Stop here")

