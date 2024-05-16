import gzip
import json
import logging
import os

import html2text

import readabilipy

from marin.markdown import to_markdown
from marin.web.convert import convert_page
from marin.web.lookup_cc import fetch_page_from_cc, search_cc_index
from marin.web.rpv2 import NUM_SHARDS, iterate_rpv2_file

# Setup logging
logger = logging.getLogger(__name__)

h = html2text.HTML2Text()
h.ignore_links = False  # Optionally ignore links
h.body_width = 0  # Optionally disable line wrapping
# TODO: html2text uses [code]...[/code] for code blocks. would prefer github markdown style
# Could also use some kind of PL lang-id to highlight code blocks, but probably not super necessary
h.mark_code = True  # Optionally convert code blocks to markdown
h.include_sup_sub = True  # Optionally include <sup> and <sub> tags
h.pad_tables = False  # We disable b/c it seems like a silly thing for the LLM to do


def process_single_doc(url, snapshot_id, debug=False):
    # Implement or integrate your methods for fetching the WARC location and HTML content
    warc_location = search_cc_index(url, snapshot_id)
    if not warc_location:
        logger.warning(f"WARC location not found for URL: {url}")
        return None

    html_content = fetch_page_from_cc(warc_location[0:1])
    if not html_content:
        logger.warning(f"Failed to fetch content for URL: {url}")
        return None

    return extract_from_html(url, html_content, debug=debug)


def extract_from_html(url, html_content, debug=False):
    # trafilatura_result = trafilatura.extract(html_content, include_formatting=True, output_format="txt", include_images=True, include_links=True)
    # trafilatura_xml = trafilatura.extract(html_content, output_format="xml")

    cleaned = convert_page(html_content, url)

    out = {
        "content": cleaned["content"],
        "title": cleaned["title"],
        "byline": cleaned["byline"],
        "date": cleaned["date"],
        # "original_html": html_content,
        # "readable_html": cleaned_html,
        # "patched_traf_content": trafilatura_result,
        # "traf_xml": trafilatura_xml,
    }


    if debug:
        cleaned_html = cleaned["html"]
        md = h.handle(cleaned_html)
        out["text2html_md"] = md
        out["original_html"] = html_content
        out["readable_html"] = cleaned_html

    return out


def process_files(snapshot, n, lang, part, debug=False):
   for doc_id, qs in iterate_rpv2_file(snapshot, n, lang, part):
        url = qs["metadata"]["url"]
        snapshot_id = qs["metadata"]["snapshot_id"]
        try:
            cleaned_article = process_single_doc(url, snapshot_id, debug=debug)
            if cleaned_article:
                # Yield the cleaned text along with other metadata and the quality signals
                yield {
                    "id": qs["id"],
                    "id_int": qs["id_int"],
                    "metadata": qs["metadata"],
                    # "plain_text": cleaned_article["plain_text"],
                    "is_duplicate": qs["is_duplicate"],
                    "quality_signals": qs["quality_signals"],  # Include the entire quality signals data
                    **cleaned_article,
                }
            else:
                logger.error(f"Content could not be processed for {doc_id} URL: {url}")
        except Exception as e:
            logger.exception(f"Error processing {doc_id}: {e}")

if __name__ == "__main__":
    lang = "en"
    snapshot = "2023-06"
    part = "middle"
    i = 0
    base_dir = f"output/{snapshot}/{lang}"
    os.makedirs(base_dir, exist_ok=True)
    for n in range(NUM_SHARDS):
        with gzip.open(f"{lang}_{n:04d}.jsonl.gz", "wt") as f:
            for doc in process_files(snapshot, n, lang, part, debug=True):
                i += 1
                f.write(json.dumps(doc) + "\n")
                id = doc["id_int"]
                os.makedirs(f"{base_dir}/{lang}_{n:04d}_{id}", exist_ok=True)
                for key in ["content", "readable_html", "original_html", "markdownify_md"]:
                    if not key in doc:
                        continue
                    suf = "md"
                    if "html" in key:
                        suf = "html"
                    elif "text" in key:
                        suf = "txt"
                    elif "xml" in key:
                        suf = "xml"
                    with open(f"{base_dir}/{lang}_{n:04d}_{id}/{key}.{suf}", "w") as out_f:
                        out_f.write(str(doc[key]))

                qs = doc["quality_signals"]
                with open(f"{base_dir}/{lang}_{n:04d}_{id}/quality_signals.json", "w") as out_f:
                    json.dump(qs, out_f, indent=2)

                if i % 100 == 0:
                    raise ValueError("Stop here")

