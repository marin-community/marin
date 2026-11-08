# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import struct

from infra.loom.gdoc.docs_format import (
    PAGE_WIDTH_PT,
    Block,
    BlockKind,
    Tab,
    WriteBlocks,
    append_markdown,
    find_tab,
    group_requests,
    image_size_pt,
    inline,
    inserted_table,
    new_units,
    old_units,
    parse,
    section_bounds,
    section_edits,
    section_top,
    tab_length,
    tab_text,
    table_fill_requests,
    top_insert_point,
    utf16_length,
)

MARKDOWN = """# Review

Verdict: **OK** to merge, see [PR](https://github.com/x/y/pull/1).

- first point with `code`
  - nested point
- second point

1. step one
2. step two

> a quote

```
uv run pytest
```

| Item | State |
|---|---|
| gate | *done* |
"""


def test_parse_finds_each_block_kind():
    kinds = [block.kind for block in parse(MARKDOWN)]
    assert kinds == [
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
        BlockKind.BULLET,
        BlockKind.BULLET,
        BlockKind.BULLET,
        BlockKind.NUMBERED,
        BlockKind.NUMBERED,
        BlockKind.QUOTE,
        BlockKind.CODE,
        BlockKind.TABLE,
    ]


def test_nested_list_items_start_with_one_tab_per_level():
    blocks = parse(MARKDOWN)
    assert blocks[3].text == "\tnested point" and blocks[3].level == 1


def test_inline_markup_becomes_plain_text_with_styles():
    text, spans = inline("Verdict: **OK**, see [PR](https://x.y) and `uv sync`, not snake_case_name.")
    assert text == "Verdict: OK, see PR and uv sync, not snake_case_name."
    assert [(text[s.start : s.end], s.fields) for s in spans] == [
        ("OK", "bold"),
        ("PR", "link"),
        ("uv sync", "weightedFontFamily"),
    ]


def test_inline_offsets_count_utf16_units():
    _, spans = inline("café 🙂 **bold**")
    # "café 🙂 " is 7 code points but 8 UTF-16 units, because the emoji needs two.
    assert (spans[0].start, spans[0].end) == (8, 12)


def test_group_requests_insert_once_and_apply_bullets_last_from_the_end():
    blocks = [block for block in parse(MARKDOWN) if block.kind not in (BlockKind.TABLE, BlockKind.DIAGRAM)]
    requests = group_requests(blocks, at=100, tab_id="t.0")

    assert requests[0]["insertText"]["location"] == {"index": 100, "tabId": "t.0"}
    assert requests[0]["insertText"]["text"].endswith("\n")
    kinds = [next(iter(request)) for request in requests]
    first_bullet = kinds.index("createParagraphBullets")
    assert set(kinds[first_bullet:]) == {"createParagraphBullets"}
    bullet_starts = [r["createParagraphBullets"]["range"]["startIndex"] for r in requests[first_bullet:]]
    assert bullet_starts == sorted(bullet_starts, reverse=True)
    assert [r["createParagraphBullets"]["bulletPreset"] for r in requests[first_bullet:]] == [
        "NUMBERED_DECIMAL_ALPHA_ROMAN",
        "BULLET_DISC_CIRCLE_SQUARE",
    ]


def test_group_requests_name_the_tab_and_stay_inside_the_inserted_text():
    requests = group_requests(parse("# Title\n\nBody"), at=10, tab_id="t.progress")
    end = 10 + len(requests[0]["insertText"]["text"])
    ranges = [r[k]["range"] for r in requests for k in r if "range" in r[k]]
    assert all(rng["tabId"] == "t.progress" and 10 <= rng["startIndex"] < rng["endIndex"] <= end for rng in ranges)


def paragraph_element(start: int, text: str, style: str = "NORMAL_TEXT") -> dict:
    return {
        "startIndex": start,
        "endIndex": start + len(text),
        "paragraph": {"elements": [{"textRun": {"content": text}}], "paragraphStyle": {"namedStyleType": style}},
    }


def body(*paragraphs: tuple[str, str]) -> list[dict]:
    elements, at = [], 1
    for text, style in paragraphs:
        elements.append(paragraph_element(at, text, style))
        at += len(text)
    return elements


def test_section_bounds_stop_at_the_next_heading_of_the_same_level():
    tab_body = body(
        ("Status: \n", "NORMAL_TEXT"),
        ("Design\n", "HEADING_3"),  # 10..17
        ("Old text\n", "NORMAL_TEXT"),  # 17..26
        ("Detail\n", "HEADING_4"),  # 26..33
        ("More\n", "NORMAL_TEXT"),  # 33..38
        ("Gates\n", "HEADING_3"),  # 38..44
        ("Gate 1\n", "NORMAL_TEXT"),  # 44..51
    )
    assert section_bounds(tab_body, "design") == (17, 38, False)
    assert section_bounds(tab_body, "Gates") == (44, 50, True)


def test_tabs_are_found_by_title_and_long_tabs_count_pages():
    words = "word " * 1000
    document = {
        "tabs": [
            {
                "tabProperties": {"tabId": "t.0", "title": "Main"},
                "documentTab": {"body": {"content": body((words + "\n", "NORMAL_TEXT"))}},
            },
            {
                "tabProperties": {"tabId": "t.1", "title": "Progress"},
                "documentTab": {"body": {"content": body(("x\n", "NORMAL_TEXT"))}},
            },
        ]
    }
    assert find_tab(document, "progress").id == "t.1"
    assert find_tab(document, None).title == "Main"
    assert tab_length(find_tab(document, "Main"), None) == (1000, 0, 2.0)


def test_table_fill_writes_the_last_cell_first_and_bolds_the_header():
    table = {
        "table": {
            "tableRows": [
                {"tableCells": [{"content": [{"startIndex": 5}]}, {"content": [{"startIndex": 7}]}]},
                {"tableCells": [{"content": [{"startIndex": 10}]}, {"content": [{"startIndex": 12}]}]},
            ]
        }
    }
    requests = table_fill_requests(table, (("Item", "State"), ("gate", "*done*")), "t.0")

    inserts = [(r["insertText"]["location"]["index"], r["insertText"]["text"]) for r in requests if "insertText" in r]
    assert inserts == [(12, "done"), (10, "gate"), (7, "State"), (5, "Item")]
    bold = [
        r["updateTextStyle"]["range"]["startIndex"]
        for r in requests
        if r.get("updateTextStyle", {}).get("fields") == "bold"
    ]
    assert bold == [7, 5]


def test_mermaid_and_dot_blocks_become_diagrams_and_other_fences_stay_code():
    blocks = parse(
        "```mermaid\nflowchart LR\n  a --> b\n```\n\n```dot\ndigraph { a -> b }\n```\n\n```python\nprint(1)\n```\n"
    )
    assert [(b.kind, b.language, b.text) for b in blocks] == [
        (BlockKind.DIAGRAM, "mermaid", "flowchart LR\n  a --> b"),
        (BlockKind.DIAGRAM, "dot", "digraph { a -> b }"),
        (BlockKind.CODE, "python", "print(1)"),
    ]


def png_header(width_px: int, height_px: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", width_px, height_px)


def test_image_size_keeps_small_diagrams_and_scales_wide_ones_to_the_page():
    assert image_size_pt(png_header(288, 144)) == (144.0, 72.0)
    width, height = image_size_pt(png_header(2 * 936, 2 * 468))
    assert (width, height) == (PAGE_WIDTH_PT, PAGE_WIDTH_PT / 2)


def test_top_insert_point_is_below_the_title_heading():
    tab_body = body(("Progress\n", "HEADING_1"), ("old update\n", "NORMAL_TEXT"))
    assert top_insert_point(tab_body) == (10, False)


def test_top_insert_point_is_the_start_without_a_heading_and_needs_a_paragraph_after_a_lone_heading():
    assert top_insert_point(body(("old update\n", "NORMAL_TEXT"))) == (1, False)
    assert top_insert_point(body(("Progress\n", "HEADING_1"))) == (10, True)


def test_section_top_is_below_the_heading_and_needs_a_paragraph_after_a_last_heading():
    with_entries = body(("Gates\n", "HEADING_3"), ("Progress\n", "HEADING_3"), ("old update\n", "NORMAL_TEXT"))
    assert section_top(with_entries, "Progress") == (16, False)
    heading_last = body(("Gates\n", "HEADING_3"), ("Progress\n", "HEADING_3"))
    assert section_top(heading_last, "Progress") == (16, True)


def test_tab_length_can_leave_out_the_progress_section():
    words = "word " * 500
    tab_body = body((words + "\n", "NORMAL_TEXT"), ("Progress\n", "HEADING_3"), (words + words + "\n", "NORMAL_TEXT"))
    document = {
        "tabs": [{"tabProperties": {"tabId": "t.0", "title": "Main"}, "documentTab": {"body": {"content": tab_body}}}]
    }
    main = find_tab(document, "Main")
    assert tab_length(main, None)[0] == 1501
    assert tab_length(main, "Progress") == (500, 0, 1.0)


def test_tab_length_counts_the_whole_tab_when_it_has_no_section_to_leave_out():
    # An older doc keeps Progress in a tab of its own, so Main has no Progress heading.
    main = Tab("t.0", "Main", body(("word " * 500 + "\n", "NORMAL_TEXT"), ("Gates\n", "HEADING_3")))
    assert tab_length(main, "Progress") == tab_length(main, None)


LISTS = {"L1": {"listProperties": {"nestingLevels": [{"glyphType": "DECIMAL"}]}}}
SECTION_START = 10


def text_element(start: int, text: str, *, numbered: bool = False, bold: bool = False) -> dict:
    """A paragraph as the Docs API returns it: its one run ends with the paragraph's newline."""
    paragraph = {
        "elements": [{"textRun": {"content": text + "\n", "textStyle": {"bold": True} if bold else {}}}],
        "paragraphStyle": {"namedStyleType": "NORMAL_TEXT"},
        **({"bullet": {"listId": "L1"}} if numbered else {}),
    }
    return {"startIndex": start, "endIndex": start + utf16_length(text) + 1, "paragraph": paragraph}


def section(*paragraphs: tuple[str, bool]) -> tuple[list[dict], int]:
    """(elements, end) of a section of (text, numbered) paragraphs."""
    elements, at = [], SECTION_START
    for text, numbered in paragraphs:
        elements.append(text_element(at, text, numbered=numbered))
        at = elements[-1]["endIndex"]
    return elements, at


def edits_for(elements: list[dict], end: int, markdown: str) -> list:
    return section_edits(old_units(elements, LISTS), new_units(parse(markdown)), end, "t.0")


def requests_of(edits: list, kind: str) -> list[dict]:
    return [request[kind] for edit in edits for request in edit if kind in request]


GATES = (("⬜ Gate A", True), ("⬜ Gate B", True), ("A note.", False))
GATES_MARKDOWN = "1. ⬜ Gate A\n2. ⬜ Gate B\n\nA note.\n"


def test_an_unchanged_section_needs_no_edit():
    assert edits_for(*section(*GATES), GATES_MARKDOWN) == []


def test_a_changed_gate_changes_only_its_emoji():
    elements, end = section(*GATES)

    edits = edits_for(elements, end, GATES_MARKDOWN.replace("⬜ Gate A", "✅ Gate A"))

    deletes = [(d["range"]["startIndex"], d["range"]["endIndex"]) for d in requests_of(edits, "deleteContentRange")]
    inserts = [(i["location"]["index"], i["text"]) for i in requests_of(edits, "insertText")]
    assert deletes == [(SECTION_START, SECTION_START + 1)] and inserts == [(SECTION_START, "✅")]
    gate_a_end = elements[0]["endIndex"]
    assert all(r["range"]["endIndex"] <= gate_a_end for r in requests_of(edits, "updateTextStyle"))


def test_a_new_gate_joins_the_numbered_list_after_the_last_gate():
    elements, end = section(*GATES)

    edits = edits_for(elements, end, GATES_MARKDOWN.replace("\n\nA note.", "\n3. ⬜ Gate C\n\nA note."))

    gate_b = elements[1]
    assert requests_of(edits, "insertText") == [
        {"location": {"index": gate_b["endIndex"] - 1, "tabId": "t.0"}, "text": "\n⬜ Gate C"}
    ]
    assert requests_of(edits, "createParagraphBullets") == [] and requests_of(edits, "deleteContentRange") == []


def test_a_new_first_paragraph_is_written_before_the_kept_ones_and_a_dropped_gate_is_deleted_alone():
    elements, end = section(*GATES)

    edits = edits_for(elements, end, "Intro.\n\n1. ⬜ Gate A\n\nA note.\n")

    gate_b = elements[1]
    assert edits[0] == [
        {
            "deleteContentRange": {
                "range": {"startIndex": gate_b["startIndex"], "endIndex": gate_b["endIndex"], "tabId": "t.0"}
            }
        }
    ]
    assert edits[1] == WriteBlocks((Block(BlockKind.PARAGRAPH, "Intro."),), SECTION_START)
    assert len(edits) == 2


def test_an_unchanged_table_stays_when_the_paragraph_after_it_changes():
    def cell(text: str, bold: bool) -> dict:
        return {"content": [text_element(0, text, bold=bold)]}

    header = {"tableCells": [cell("Item", True), cell("State", True)]}
    table = {
        "startIndex": SECTION_START + 1,
        "endIndex": 40,
        "table": {"tableRows": [header, {"tableCells": [cell("gate", False), cell("done", False)]}]},
    }
    elements = [text_element(SECTION_START, ""), table, text_element(40, "After.")]

    edits = edits_for(elements, 47, "| Item | State |\n|---|---|\n| gate | done |\n\nAfter, changed.\n")

    assert all(r["range"]["startIndex"] >= 40 for r in requests_of(edits, "deleteContentRange"))
    assert all(r["location"]["index"] >= 40 for r in requests_of(edits, "insertText"))


def chip_paragraph(start: int) -> dict:
    """A paragraph with a person chip, as documents.get returns it: the chip takes one index."""
    person = {"personId": "kix.p1", "personProperties": {"name": "Ada Lovelace", "email": "ada@example.com"}}
    elements = [
        {"startIndex": start, "endIndex": start + 4, "textRun": {"content": "Ask ", "textStyle": {}}},
        {"startIndex": start + 4, "endIndex": start + 5, "person": person},
        {"startIndex": start + 5, "endIndex": start + 15, "textRun": {"content": " about it\n", "textStyle": {}}},
    ]
    return {
        "startIndex": start,
        "endIndex": start + 15,
        "paragraph": {"elements": elements, "paragraphStyle": {"namedStyleType": "NORMAL_TEXT"}},
    }


def test_a_paragraph_with_a_person_chip_stays_when_the_new_section_keeps_its_text():
    # Regression: such a paragraph never matched, so replace_section retyped it as plain text and lost the chip.
    gates, end = section(*GATES)
    elements = [*gates, chip_paragraph(end)]
    # The doc's markdown export, which the agent reads, writes the chip as a mailto link.
    markdown = GATES_MARKDOWN + "\nAsk [Ada Lovelace](mailto:ada@example.com) about it\n"

    assert edits_for(elements, end + 15, markdown) == []
    assert tab_text(Tab("t.0", "Main", [chip_paragraph(1)])) == "Ask Ada Lovelace about it"


def test_a_new_tables_cells_are_where_docs_puts_them():
    # The cell paragraph indices that Docs returned after an insertTable of 2 rows and 3 columns at index 65.
    cells = [
        [cell["content"][0]["startIndex"] for cell in row["tableCells"]]
        for row in inserted_table(65, 2, 3)["table"]["tableRows"]
    ]
    assert cells == [[69, 71, 73], [76, 78, 80]]


class RecordingDocsApi:
    """Serves one document, and counts its reads and records the batchUpdate bodies posted to it."""

    def __init__(self, document: dict):
        self.document = document
        self.reads = 0
        self.batches: list[dict] = []

    def get(self, url: str, params: dict | None = None) -> dict:
        self.reads += 1
        return self.document

    def post(self, url: str, body: dict, params: dict | None = None) -> dict:
        self.batches.append(body)
        return {}


def test_a_write_is_one_batch_that_docs_transforms_over_edits_made_since_its_read():
    # Regression: index-based batches without write control misplaced text that a person typed meanwhile.
    content = [{"endIndex": 1, "sectionBreak": {}}, *body(("Progress\n", "HEADING_3"), ("Old entry.\n", "NORMAL_TEXT"))]
    tab = {"tabProperties": {"tabId": "t.0", "title": "Main"}, "documentTab": {"body": {"content": content}}}
    api = RecordingDocsApi({"documentId": "doc", "revisionId": "rev-7", "tabs": [tab]})

    append_markdown(api, "doc", "New entry.\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n", None, None)

    assert (api.reads, len(api.batches)) == (1, 1)
    assert api.batches[0]["writeControl"] == {"targetRevisionId": "rev-7"}
    assert any("insertTable" in request for request in api.batches[0]["requests"])


def test_a_numbered_list_reads_with_its_numbers_and_parses_back_as_numbered():
    # Regression: tab_text wrote every list item with `-`, so text read back through replace-section lost numbering.
    elements, _ = section(*GATES)
    text = tab_text(Tab("t.0", "Main", elements, lists=LISTS))

    assert text.splitlines() == ["1. ⬜ Gate A", "2. ⬜ Gate B", "A note."]
    assert [block.kind for block in parse(text)] == [BlockKind.NUMBERED, BlockKind.NUMBERED, BlockKind.PARAGRAPH]
