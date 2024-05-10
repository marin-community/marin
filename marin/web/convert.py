from urllib.parse import urljoin

import htmlmin
from bs4 import BeautifulSoup

from marin.markdown import to_markdown


def convert_page(html: str, url: str | None = None) -> dict[str, str]:
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


def make_links_absolute(soup: BeautifulSoup, base_url):
    """Converts relative image/anchor URLs to absolute URLs."""
    for tag in soup.select("a, img"):
        # handle images and anchors
        if tag.has_attr("src"):
            tag["src"] = urljoin(base_url, tag["src"])
        if tag.has_attr("href"):
            tag["href"] = urljoin(base_url, tag["href"])
    return soup
