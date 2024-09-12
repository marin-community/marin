import re
import htmlmin

from bs4 import BeautifulSoup
from urllib.parse import urljoin

from marin.markdown import to_markdown


def convert_page_with_trafilatura(html: str, url: str | None = None) -> dict[str, str]:
    from trafilatura import extract, extract_metadata

    title = extract_metadata(html).title
    content = extract(
        html,
        favor_recall=True,
        include_links=True,
        output_format="markdown",
    )

    if title == "[no-title]":
        title = None

    if title:
        content = f"# {title}\n\n{content}"
    
    out = {
        "title": title,
        "content": content,
        "html": html
    }

    if url:
        out["url"] = url

    return out


def convert_page_with_readability(html: str, url: str | None = None) -> dict[str, str]:
    from readability import Document
    # remove null character and control characters
    html = re.sub(u'[^\u0020-\uD7FF\u0009\u000A\u000D\uE000-\uFFFD\U00010000-\U0010FFFF]+', '', html)

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
    markdown = to_markdown(tree)

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
    print(f"This is Legacy method, use convert_page_python instead")
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


def convert_page(html: str, url: str | None = None, extract_method: str = "readability") -> dict[str, str]:
    match extract_method:
        case "trafilatura":
            return convert_page_with_trafilatura(html, url)
        case "readability":
            return convert_page_with_readability(html, url)
        case "legacy":
            return convert_page_legacy(html, url)
        case _:
            print(f"Invalid extract_method: {extract_method}. Switching to readability for extraction.")
            return convert_page_with_readability(html, url)


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
