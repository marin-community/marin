# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import re

from bs4 import BeautifulSoup

from marin.markdown import to_markdown
from marin.schemas.web.convert import (
    ExtractionConfig,
    HtmlToMarkdownConfig,
    ResiliparseConfig,
)

logger = logging.getLogger("ray")


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

    from resiliparse.extract.html2text import extract_simplified_dom

    tree = extract_simplified_dom(html, preserve_formatting=True, main_content=True, **kwargs)
    tree = BeautifulSoup(str(tree), "html.parser")
    markdown = to_markdown(tree, markdownify_config)
    return markdown.replace("\x00", "").strip()


def convert_page_with_resiliparse(
    html: str,
    url: str | None = None,
    config: ResiliparseConfig = ResiliparseConfig(),
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
    from resiliparse.parse.html import HTMLTree

    tree = HTMLTree.parse(html)
    title = tree.title or None

    content = extract_content_from_dom(html, config.resiliparse_kwargs, config.markdownify_config)

    if title and config.prepend_title:
        # remove html tags from title
        title = re.sub(r"<[^>]*>", "", title).strip()

        content = f"# {title}\n\n{content}"

    out = {"title": title, "content": content, "html": html}

    if url:
        out["url"] = url

    return out


def convert_page(
    html: str,
    url: str | None = None,
    extract_method: str = "resiliparse",
    config: ExtractionConfig = ResiliparseConfig(),
) -> dict[str, str]:
    """
    Convert HTML to text/markdown using Resiliparse.

    Parameters:
        html (str): HTML content to convert.
        url (str | None): URL of the page.
        extract_method (str): Method to use for extraction. Only "resiliparse" is supported.
        config (ExtractionConfig): Configuration for the extraction method.

    Returns:
        dict[str, str]: Dictionary containing the title, content, and HTML of the page.
    """
    if extract_method != "resiliparse":
        raise ValueError(f"Only 'resiliparse' extraction method is supported, got: {extract_method}")

    return convert_page_with_resiliparse(html, url, config)
