from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .markdown import to_markdown

def convert_page(html: str, url: str | None = None) -> dict[str, str]:
    from readabilipy import simple_json_from_html_string

    reabilitied = simple_json_from_html_string(html, use_readability=True)
    tree = BeautifulSoup(reabilitied["content"], "html.parser")
    if url:
        tree = make_links_absolute(tree, url)
    markdown = to_markdown(tree)

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