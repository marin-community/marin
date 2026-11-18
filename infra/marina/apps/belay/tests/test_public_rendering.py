# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The public agent log renders agent text inert: no element common.js did not build, and links only to http(s)."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"
CHECK = Path(__file__).with_name("sanitize_check.mjs")
# The tags markdown() and link() build; anything else would have come from the text itself.
BUILT_TAGS = {
    "div",
    "p",
    "ul",
    "ol",
    "li",
    "h4",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
    "code",
    "strong",
    "a",
    "span",
}
HOSTILE = """**X0001 · plan** <script>alert(1)</script>

<img src=x onerror=alert(1)> and [click](javascript:alert(1)) and [data](data:text/html,<b>x</b>)

| Run | Note |
|---|---|
| a | <iframe src="https://evil.example"></iframe> [ok](https://wandb.ai/x) |

- [file](file:///etc/passwd) `<b>code</b>`
"""


def nodes(tree: dict) -> list[dict]:
    found = [tree]
    for child in tree.get("children", []):
        found += nodes(child)
    return found


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_script_img_onerror_and_javascript_links_render_inert() -> None:
    out = subprocess.run(
        ["node", str(CHECK), str(WEB / "common.js")], input=HOSTILE, capture_output=True, text=True, check=True
    ).stdout
    body, script_link, protocol_relative, own_page, https = json.loads(out)
    elements = [n for n in nodes(body) if "tag" in n]
    assert {n["tag"] for n in elements} <= BUILT_TAGS
    assert all(not key.startswith("on") for n in elements for key in n["attrs"])
    anchors = [n["attrs"]["href"] for n in elements if n["tag"] == "a"]
    assert anchors == ["https://wandb.ai/x"]
    text = "".join(n["text"] for n in nodes(body) if "text" in n)
    assert "<script>alert(1)</script>" in text and "<img src=x onerror=alert(1)>" in text  # shown as text
    assert script_link["tag"] == "span" and protocol_relative["tag"] == "span"
    assert own_page["tag"] == "a" and own_page["attrs"]["href"] == "/belay/#status/X0001"
    assert https["attrs"]["href"] == "https://example.com/ok"


# Marina's canonical applet URL (/a/<id>/) serves only the root document; its redirect to the current revision keeps
# a fragment and drops a path suffix or query. So every view and every in-page link must be a fragment route.
OWN_LINK = re.compile(r'href="(?!https?://)([^"]*)"')


def test_every_own_link_is_a_fragment_route() -> None:
    hrefs = OWN_LINK.findall((WEB / "index.html").read_text())
    assert hrefs
    paths = [h for h in hrefs if not (h.startswith("#") or h == "./" or "?v=__BELAY_VERSION__" in h)]
    assert paths == []
