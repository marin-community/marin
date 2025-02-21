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
from marin.web.utils import extract_content_from_dom

logger = logging.getLogger("ray")
HTML_CORRECTION_PREFIX = "<!DOCTYPE html>"


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
    config: ResiliparseConfig | None = ResiliparseConfig.default_config(),
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
            title = re.sub(r'<[^>]*>', '', title).strip()

            content = f"{title}\n\n{content}"

    else:
        # We override the existing resiliparse package with our custom fork in the worker
        # environment. So we call the remote function with `pip` argument in decorator to
        # install the custom package.
        content = extract_content_from_dom(html, config.resiliparse_kwargs, config.markdownify_config)

        if title and config.prepend_title:
            # remove html tags from title
            title = re.sub(r'<[^>]*>', '', title).strip()
            
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


def convert_page_legacy(html: str, url: str | None = None) -> dict[str, str]:
    print("This is Legacy method, use convert_page_python instead")
    import htmlmin
    from readabilipy import simple_json_from_html_string

    reabilitied = simple_json_from_html_string(html, use_readability=True)
    tree = reabilitied["content"]
    tree = htmlmin.minify(tree, remove_empty_space=True, keep_pre=True)
    tree = BeautifulSoup(tree, "html.parser")
    if url:
        tree = make_links_absolute(tree, url)
    # reconvert tree to str with absolute URLs
    reabilitied["content"] = str(tree)
    markdown = to_markdown(tree)

    # add title to markdown
    if reabilitied["title"]:
        markdown = f"# {reabilitied['title']}\n\n{markdown}"

    out = {
        "title": reabilitied["title"],
        "content": markdown,
        "date": reabilitied["date"],
        "byline": reabilitied["byline"],
        "html": reabilitied["content"],
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
        case "legacy":
            return convert_page_legacy(html, url)
        case _:
            raise Exception(f"Invalid extract_method: {extract_method}")


def make_links_absolute(soup: BeautifulSoup, base_url):
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
