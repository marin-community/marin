# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from scripts.pm.goals_doc import parse_status, render_document, status_requests

STATUS = """As of Oct 20, 2026 🚀:

- **Hero: on track.** See [#8435](https://example.com/8435).
  - **Nested.** Box 7 ([#10164](https://example.com/10164)) has no owner.
- Last item.
"""


def _paragraph(start: int, text: str, style: str = "NORMAL_TEXT") -> dict:
    return {
        "startIndex": start,
        "endIndex": start + len(text),
        "paragraph": {
            "elements": [{"textRun": {"content": text, "textStyle": {}}}],
            "paragraphStyle": {"namedStyleType": style},
        },
    }


def _styled_text(requests: list[dict], key: str) -> list[str]:
    """Return the inserted text that each `updateTextStyle` request setting `key` covers."""
    insert = next(request["insertText"] for request in requests if "insertText" in request)
    units = insert["text"].encode("utf-16-le")
    start = insert["location"]["index"]
    spans = []
    for request in requests:
        update = request.get("updateTextStyle")
        if update is None or key not in update["textStyle"]:
            continue
        span = update["range"]
        spans.append(units[2 * (span["startIndex"] - start) : 2 * (span["endIndex"] - start)].decode("utf-16-le"))
    return spans


def test_status_requests_style_the_markdown_spans():
    body = {"content": [_paragraph(1, "Status\n", "HEADING_2"), _paragraph(8, "Old status.\n")]}

    requests = status_requests(body, parse_status(STATUS))

    assert requests[0] == {"deleteContentRange": {"range": {"startIndex": 8, "endIndex": 19}}}
    assert requests[1]["insertText"]["text"] == (
        "As of Oct 20, 2026 🚀:\nHero: on track. See #8435.\n\tNested. Box 7 (#10164) has no owner.\nLast item."
    )
    assert _styled_text(requests, "bold") == ["Hero: on track.", "Nested."]
    assert _styled_text(requests, "link") == ["#8435", "#10164"]
    bullets = [request["createParagraphBullets"]["range"] for request in requests if "createParagraphBullets" in request]
    # The emoji is two UTF-16 code units, so the list starts after 23 units of "As of ...:\n".
    assert bullets == [{"startIndex": 8 + 23, "endIndex": 8 + 98}]


def test_status_requests_replace_every_section_below_status():
    body = {
        "content": [
            _paragraph(1, "Goals\n", "HEADING_2"),
            _paragraph(7, "Human text.\n"),
            _paragraph(19, "Status\n", "HEADING_2"),
            _paragraph(26, "Old status.\n"),
            _paragraph(38, "Notes\n", "HEADING_2"),
            _paragraph(44, "Old notes.\n"),
        ]
    }

    requests = status_requests(body, parse_status("New status.\n\n### Board changes\n\n- Added #1.\n"))

    assert requests[:2] == [
        {"deleteContentRange": {"range": {"startIndex": 26, "endIndex": 54}}},
        {"insertText": {"location": {"index": 26}, "text": "New status.\nBoard changes\nAdded #1."}},
    ]
    headings = [
        (request["updateParagraphStyle"]["range"]["startIndex"], request["updateParagraphStyle"]["paragraphStyle"])
        for request in requests
        if request.get("updateParagraphStyle", {}).get("fields") == "namedStyleType"
    ]
    assert headings == [(26 + 12, {"namedStyleType": "HEADING_3"})]


def test_rendered_status_section_parses_back_to_the_same_blocks():
    blocks = parse_status(STATUS)
    content = []
    for block in blocks:
        elements = []
        for run in block.runs:
            style: dict = {"bold": True} if run.bold else {}
            if run.url is not None:
                style["link"] = {"url": run.url}
            elements.append({"textRun": {"content": run.text, "textStyle": style}})
        elements.append({"textRun": {"content": "\n", "textStyle": {}}})
        extra = {} if block.level is None else {"bullet": {"listId": "list", "nestingLevel": block.level}}
        content.append(
            {"paragraph": {"elements": elements, "paragraphStyle": {"namedStyleType": "NORMAL_TEXT"}, **extra}}
        )

    assert parse_status(render_document({"content": content})) == blocks
