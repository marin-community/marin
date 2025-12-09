# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re
from dataclasses import asdict
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from marin.markdown import to_markdown
from marin.schemas.web.convert import (
    ExtractionConfig,
    HtmlToMarkdownConfig,
    ResiliparseConfig,
    TrafilaturaConfig,
)

logger = logging.getLogger("ray")
HTML_CORRECTION_PREFIX = "<!DOCTYPE html>"


def extract_content_from_dom(
    html: str,
    kwargs: dict,
    markdownify_config: HtmlToMarkdownConfig,
) -> str:
    """
    This function extracts the main content DOM from the HTML content. We have a custom fork of Resiliparse at
    https://github.com/krypticmouse/chatnoir-resiliparse/tree/develop/resiliparse which modifies the `extract_plain_text`
    method to return the main content DOM instead of plain text.

    This method then converts the main content DOM to markdown using the `to_markdown` method via markdownify.

    Parameters:
        html (str): HTML content to extract.
        kwargs (dict): Keyword arguments to pass to the `extract_plain_text` method.
        markdownify_config (HtmlToMarkdownConfig): Configuration for markdownify.

    Returns:
        str: Markdown content of the main content DOM.

    NOTE: This method is using as custom fork of Resiliparse that is not meant to be merged into the main repository of
    Resiliparse. This is a custom modification for the purpose of this experiment. So, this method will not work with
    the main Resiliparse package. No plans to merge this into the main Resiliparse package yet.
    """

    from resiliparse_dom.extract.html2text import extract_simplified_dom

    tree = extract_simplified_dom(html, **kwargs)
    tree = BeautifulSoup(str(tree), "html.parser")

    # convert to markdown
    markdown = to_markdown(tree, markdownify_config)
    return markdown.replace("\x00", "").strip()


def convert_page_with_trafilatura(
    html: str,
    url: str | None = None,
    config: ExtractionConfig = TrafilaturaConfig.default_config(),
) -> dict[str, str]:
    """
    Convert HTML to text[non-markdown] using Trafilatura.

    Parameters:
        html (str): HTML content to convert.
        url (str | None): URL of the page.
        config (TrafilaturaConfig): Configuration for Trafilatura.

    Returns:
        dict[str, str]: Dictionary containing the title, content, and HTML of the page.
    """
    from trafilatura import extract, extract_metadata

    title = None
    try:
        metadata = extract_metadata(html)

        if metadata:
            title = metadata.title
    except Exception as e:
        logger.warning(f"Error extracting metadata: {e}")
        title = None

    content = extract(html, **asdict(config))
    if not content:
        content = extract(HTML_CORRECTION_PREFIX + html, **asdict(config))

    if title:
        content = f"{title}\n\n{content}"

    out = {"title": title, "content": content, "html": html}

    if url:
        out["url"] = url

    return out


def convert_page_with_resiliparse(
    html: str,
    url: str | None = None,
    config: ResiliparseConfig = ResiliparseConfig.default_config(),
) -> dict[str, str]:
    """
    Convert HTML to text[non-markdown] using Resiliparse.

    Note: This method does not convert the content to markdown. Resiliparse does not have a markdown conversion method.
    You can use the markdown conversion method from the `marin.markdown` module over HTMLTree
    from `resiliparse.parse.html`.

    But, then this method will be identical to the `convert_page_with_readability` method then.

    Parameters:
        html (str): HTML content to convert.
        url (str | None): URL of the page.
        config (ResiliparseConfig): Configuration for Resiliparse.

    Returns:
        dict[str, str]: Dictionary containing the title, content, and HTML of the page.
    """
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.html import HTMLTree

    tree = HTMLTree.parse(html)
    title = tree.title or None

    content = None
    if not config.use_custom_variant:
        content = extract_plain_text(html, **config.resiliparse_kwargs)

        if title and config.prepend_title:
            # remove html tags from title
            title = re.sub(r"<[^>]*>", "", title).strip()

            content = f"{title}\n\n{content}"

    else:
        # We override the existing resiliparse package with our custom fork in the worker
        # environment. So we call the remote function with `pip` argument in decorator to
        # install the custom package.
        content = extract_content_from_dom(html, config.resiliparse_kwargs, config.markdownify_config)

        if title and config.prepend_title:
            # remove html tags from title
            title = re.sub(r"<[^>]*>", "", title).strip()

            content = f"# {title}\n\n{content}"

    out = {"title": title, "content": content, "html": html}

    if url:
        out["url"] = url

    return out


def convert_page_with_readability(
    html: str,
    url: str | None = None,
    config: HtmlToMarkdownConfig = HtmlToMarkdownConfig.default_config(),
) -> dict[str, str]:
    """
    Convert HTML to text[markdown] using Readability and markdownify.

    Parameters:
        html (str): HTML content to convert.
        url (str | None): URL of the page.
        config (HtmlToMarkdownConfig): Configuration for markdownify.

    Returns:
        dict[str, str]: Dictionary containing the title, content, and HTML of the page.
    """
    import htmlmin
    from bs4 import BeautifulSoup
    from readability import Document

    # remove null character and control characters
    html = re.sub("[^\u0020-\ud7ff\u0009\u000a\u000d\ue000-\ufffd\U00010000-\U0010ffff]+", "", html)

    doc = Document(html)
    title = doc.title()

    tree = doc.summary()

    tree = htmlmin.minify(tree, remove_empty_space=True, keep_pre=True)
    tree = BeautifulSoup(tree, "html.parser")
    if url:
        tree = make_links_absolute(tree, url)

    # reconvert tree to str with absolute URLs
    content = str(tree)

    # convert to markdown
    markdown = to_markdown(tree, config)

    # readability-lxml uses "[no-title]" for pages without a title
    if title == "[no-title]":
        title = None

    # add title to markdown
    if title:
        markdown = f"# {title}\n\n{markdown}"

    out = {
        "title": title,
        "content": markdown,
        "html": content,
    }
    if url:
        out["url"] = url

    return out


def convert_page(
    html: str,
    url: str | None = None,
    extract_method: str = "readability",
    config: ExtractionConfig = HtmlToMarkdownConfig.default_config(),
) -> dict[str, str]:
    """
    Convert HTML to text using the specified method.

    Parameters:
        html (str): HTML content to convert.
        url (str | None): URL of the page.
        extract_method (str): Method to use for extraction. Defaults to "readability".
        config (ExtractionConfig): Configuration for the extraction method.

    Returns:
        dict[str, str]: Dictionary containing the title, content, and HTML of the page.
    """

    match extract_method:
        case "trafilatura":
            return convert_page_with_trafilatura(html, url, config)
        case "readability":
            return convert_page_with_readability(html, url, config)
        case "resiliparse":
            return convert_page_with_resiliparse(html, url, config)
        case _:
            raise Exception(f"Invalid extract_method: {extract_method}")


def make_links_absolute(soup, base_url):
    """Converts relative image/anchor URLs to absolute URLs."""
    for tag in soup.select("a, img"):
        # handle images and anchors
        if tag.has_attr("src"):
            try:
                tag["src"] = urljoin(base_url, tag["src"])
            except Exception as e:
                # Leave it unchanged
                print(f"Error in src {e} {tag['src']}")

        if tag.has_attr("href"):
            try:
                tag["href"] = urljoin(base_url, tag["href"])
            except Exception as e:
                # Leave it unchanged
                print(f"Error in href {e} {tag['href']}")

    return soup
