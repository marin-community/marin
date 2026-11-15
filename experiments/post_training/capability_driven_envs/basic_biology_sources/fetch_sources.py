"""Archive pinned-by-hash government biology pages before any task generation.

Download only official handbook/fact-sheet HTML, not external links or media.
The archive is *not* a training corpus: inspect page-level credits and extract
eligible text separately before passing any content to a model.
"""

import argparse
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

MEDLINE_BASE = "https://medlineplus.gov/genetics/understanding/"
NHGRI_BASE = "https://www.genome.gov/about-genomics/fact-sheets"
POLICIES = {
    "medlineplus": "https://medlineplus.gov/about/using/usingcontent/",
    "nhgri": "https://www.genome.gov/about-nhgri/Policies-Guidance/Copyright",
}
USER_AGENT = "MarinFold source-inventory/1.0 (public government educational pages)"
MAX_HTML_BYTES = 2_000_000


class Links(HTMLParser):
    """Extract only link targets; never fetch images or script references."""

    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a" and (href := dict(attrs).get("href")):
            self.urls.append(href)


def scoped_links(data: bytes, page_url: str, prefix: str) -> list[str]:
    """Keep only same-scheme/host, query-free descendants of a verified index."""
    parsed_prefix = urlsplit(prefix)
    links = Links()
    links.feed(data.decode("utf-8"))
    selected: set[str] = set()
    for href in links.urls:
        parsed = urlsplit(urljoin(page_url, href))
        if (
            parsed.scheme == parsed_prefix.scheme
            and parsed.netloc == parsed_prefix.netloc
            and not parsed.query
            and parsed.path.startswith(parsed_prefix.path)
        ):
            selected.add(urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", "")))
    return sorted(selected)


def fetch(url: str, *, allowed_prefix: str) -> tuple[str, bytes]:
    """Read a bounded HTML page and refuse redirects outside its source scope."""
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=40) as response:
        final_url = response.geturl()
        parsed = urlsplit(final_url)
        allowed = urlsplit(allowed_prefix)
        if (
            parsed.scheme != allowed.scheme
            or parsed.netloc != allowed.netloc
            or not parsed.path.startswith(allowed.path)
        ):
            raise ValueError(f"Page redirected outside approved scope: {url} -> {final_url}")
        if response.headers.get_content_type() != "text/html":
            raise ValueError(f"Not HTML: {final_url}")
        data = response.read(MAX_HTML_BYTES + 1)
        if not data or len(data) > MAX_HTML_BYTES:
            raise ValueError(f"Empty or oversized HTML: {final_url}")
        return final_url, data


def archive(url: str, prefix: str, root: Path, records: dict[str, dict[str, object]]) -> bytes:
    """Save one uniquely identified official HTML page, including its source URL."""
    final_url, data = fetch(url, allowed_prefix=prefix)
    parsed = urlsplit(final_url)
    relative = parsed.path[len(urlsplit(prefix).path) :].strip("/")
    if not relative or final_url.rstrip("/") == prefix.rstrip("/"):
        relative = "index"
    if parsed.path.endswith("/") and relative != "index":
        relative += "/index"
    source = "medlineplus" if prefix == MEDLINE_BASE else "nhgri"
    name = f"pages/{source}/{relative}.html"
    destination = root / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != data:
        raise ValueError(f"Existing page differs from fresh official response: {destination}")
    destination.write_bytes(data)
    if name in records and records[name]["url"] != final_url:
        raise ValueError(f"Two URLs mapped to {name}")
    records[name] = {"url": final_url, "path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return data


def download(root: Path) -> dict[str, object]:
    """Fetch the handbook index and sections, fact-sheet index, and rights pages."""
    if (root / "manifest.json").exists():
        raise ValueError("Snapshot already frozen; choose a new output directory")
    root.mkdir(parents=True, exist_ok=True)
    pages: dict[str, dict[str, object]] = {}
    policies: dict[str, dict[str, object]] = {}
    for name, url in POLICIES.items():
        final_url, data = fetch(url, allowed_prefix=url)
        path = f"policies/{name}.html"
        (root / "policies").mkdir(exist_ok=True)
        (root / path).write_bytes(data)
        policies[name] = {"url": final_url, "path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    medline_index = archive(MEDLINE_BASE, MEDLINE_BASE, root, pages)
    sections = [
        url
        for url in scoped_links(medline_index, MEDLINE_BASE, MEDLINE_BASE)
        if url.rstrip("/") != MEDLINE_BASE.rstrip("/")
        and len(urlsplit(url).path[len(urlsplit(MEDLINE_BASE).path) :].strip("/").split("/")) == 1
    ]
    if len(sections) < 10:
        raise ValueError(f"Handbook section inventory unexpectedly small: {len(sections)}")
    for section in sections:
        content = archive(section, MEDLINE_BASE, root, pages)
        for article in scoped_links(content, section, section.rstrip("/") + "/"):
            if article.rstrip("/") != section.rstrip("/"):
                archive(article, MEDLINE_BASE, root, pages)

    nhgri_index = archive(NHGRI_BASE, NHGRI_BASE, root, pages)
    sheets = [url for url in scoped_links(nhgri_index, NHGRI_BASE, NHGRI_BASE + "/")]
    if len(sheets) < 20:
        raise ValueError(f"NHGRI fact-sheet inventory unexpectedly small: {len(sheets)}")
    for sheet in sheets:
        archive(sheet, NHGRI_BASE, root, pages)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "raw_official_html_not_cleared_for_model_input",
        "scope": "MedlinePlus Help Me Understand Genetics and NHGRI fact sheets; HTML only, no media or external sites",
        "rights_policy_snapshots": policies,
        "pages": [pages[key] for key in sorted(pages)],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Private local snapshot directory")
    args = parser.parse_args()
    manifest = download(args.output)
    print(f"Saved {len(manifest['pages'])} HTML pages and two rights policies to {args.output}")


if __name__ == "__main__":
    main()
