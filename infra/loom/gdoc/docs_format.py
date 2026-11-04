# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Write markdown into a Google Doc as native Docs formatting.

It handles the markdown that agents write: headings, bullet and numbered lists
(nested by indentation), paragraphs, fenced code blocks, block quotes, tables,
and inline bold, italics, code and links. A ```mermaid, ```dot or ```graphviz
block becomes a diagram: mermaid-cli or Graphviz draws it, the PNG goes to a GCS
prefix that expires after a day, and Docs copies it in from a short-lived signed URL. Text goes at the end of a tab
(append_markdown), at the top of a section (prepend_markdown), or becomes the
content under one heading (replace_section).

replace_section changes only what differs, because a comment loses its place in the
doc when the text it is on is deleted: paragraphs that stay the same are not touched,
and a paragraph that keeps its kind (heading, list item, quote or normal) is edited
only where its characters differ. New list items go into the list next to them, so a
numbered list keeps counting.

Docs API indices count UTF-16 code units, and every request names its tab. All
blocks go in at one index, the start of a paragraph, from the last block to the
first: each insert pushes the blocks written before it down, so no index moves
under a later request. Consecutive text blocks go in one batchUpdate; a table
or diagram needs its own, because a table's cell indices exist only after Docs
inserts it. Bullets are applied last in a batch and from the end backwards,
because createParagraphBullets removes the tabs that set nesting levels.
"""

import difflib
import hashlib
import json
import re
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from infra.loom.gdoc.google_api import DOCS_API, GoogleApi, gcloud

CODE_FONT = "Roboto Mono"
QUOTE_INDENT_PT = 36
BULLET_PRESET = "BULLET_DISC_CIRCLE_SQUARE"
NUMBERED_PRESET = "NUMBERED_DECIMAL_ALPHA_ROMAN"
MAX_LIST_LEVEL = 8
# The text styles a new block must not inherit from the paragraph before it.
RESET_TEXT_FIELDS = "bold,italic,underline,strikethrough,link,weightedFontFamily"
MERMAID_LANGUAGE = "mermaid"
# Pinned: npx installs it on first use into the npm cache, which Loom sessions share.
MERMAID_CLI = "@mermaid-js/mermaid-cli@12.0.0"
DIAGRAM_LANGUAGES = (MERMAID_LANGUAGE, "dot", "graphviz")
DIAGRAM_DPI = 144
# mermaid-cli takes a whole scale factor. At 2, image_size_pt shows its diagrams at 4/3 of their CSS size.
MERMAID_SCALE = 2
# Chrome's sandbox needs user namespaces, which a session's container does not allow.
MERMAID_PUPPETEER_CONFIG = {"args": ["--no-sandbox"]}
# Letter page width minus 1-inch margins.
PAGE_WIDTH_PT = 468
# Docs copies an image when it inserts it, so the URL and the object only need to outlive the request.
DIAGRAM_URL_LIFETIME = "15m"
DIAGRAM_PREFIX = "gs://marin-us-central1/tmp/ttl=1d/loom-gdoc/diagrams"
# For a rough page estimate of a tab.
WORDS_PER_PAGE = 500
PAGES_PER_DIAGRAM = 1 / 3

HEADING = re.compile(r"(#{1,6})\s+(.*)")
BULLET = re.compile(r"(\s*)[-*+]\s+(.*)")
NUMBERED = re.compile(r"(\s*)\d+[.)]\s+(.*)")
QUOTE = re.compile(r">\s?(.*)")
RULE = re.compile(r"(-{3,}|\*{3,}|_{3,})\s*")
TABLE_SEPARATOR = re.compile(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*")
INLINE = re.compile(
    r"`(?P<code>[^`]+)`"
    r"|\*\*(?P<bold>.+?)\*\*"
    r"|(?<!\w)__(?P<bold2>.+?)__(?!\w)"
    r"|(?<![\w*])\*(?P<italic>[^*\s][^*]*?)\*(?![\w*])"
    r"|(?<!\w)_(?P<italic2>[^_\s][^_]*?)_(?!\w)"
    r"|\[(?P<link_text>[^\]]+)\]\((?P<url>[^)\s]+)\)"
)


def u16(text: str) -> int:
    """Length in UTF-16 code units, the unit of Docs API indices."""
    return len(text.encode("utf-16-le")) // 2


class BlockKind(StrEnum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    BULLET = "bullet"
    NUMBERED = "numbered"
    CODE = "code"
    QUOTE = "quote"
    TABLE = "table"
    DIAGRAM = "diagram"


@dataclass(frozen=True)
class Span:
    """A text style over [start, end) of a block's text, in UTF-16 units."""

    start: int
    end: int
    style: dict
    fields: str


@dataclass(frozen=True)
class Block:
    kind: BlockKind
    text: str = ""
    level: int = 0
    spans: tuple[Span, ...] = ()
    rows: tuple[tuple[str, ...], ...] = field(default=())
    # A fenced block's language, such as "python" or "mermaid".
    language: str = ""


def inline(markdown: str) -> tuple[str, tuple[Span, ...]]:
    """Plain text and its styles from inline markdown."""
    plain: list[str] = []
    spans: list[Span] = []
    at, last = 0, 0
    for match in INLINE.finditer(markdown):
        before = markdown[last : match.start()]
        plain.append(before)
        at += u16(before)
        groups = match.groupdict()
        if groups["code"] is not None:
            text, style, fields = (
                groups["code"],
                {"weightedFontFamily": {"fontFamily": CODE_FONT}},
                "weightedFontFamily",
            )
        elif groups["bold"] is not None or groups["bold2"] is not None:
            text, style, fields = groups["bold"] or groups["bold2"], {"bold": True}, "bold"
        elif groups["italic"] is not None or groups["italic2"] is not None:
            text, style, fields = groups["italic"] or groups["italic2"], {"italic": True}, "italic"
        else:
            text, style, fields = groups["link_text"], {"link": {"url": groups["url"]}}, "link"
        plain.append(text)
        spans.append(Span(at, at + u16(text), style, fields))
        at += u16(text)
        last = match.end()
    plain.append(markdown[last:])
    return "".join(plain), tuple(spans)


def _table_row(line: str) -> tuple[str, ...]:
    return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))


def _list_block(kind: BlockKind, indent: str, markdown: str) -> Block:
    level = min(len(indent.replace("\t", "    ")) // 2, MAX_LIST_LEVEL)
    text, spans = inline(markdown)
    tabs = "\t" * level
    # Leading tabs set the nesting level; createParagraphBullets removes them.
    return Block(kind, tabs + text, level, tuple(Span(s.start + level, s.end + level, s.style, s.fields) for s in spans))


def parse(markdown: str) -> list[Block]:
    blocks: list[Block] = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or RULE.fullmatch(stripped):
            i += 1
        elif stripped.startswith("```"):
            language = stripped[3:].strip().lower()
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            kind = BlockKind.DIAGRAM if language in DIAGRAM_LANGUAGES else BlockKind.CODE
            blocks.append(Block(kind, "\n".join(code), language=language))
        elif stripped.startswith("|") and i + 1 < len(lines) and TABLE_SEPARATOR.fullmatch(lines[i + 1].strip()):
            rows = [_table_row(line)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_table_row(lines[i]))
                i += 1
            width = max(len(row) for row in rows)
            blocks.append(Block(BlockKind.TABLE, rows=tuple(row + ("",) * (width - len(row)) for row in rows)))
        elif match := HEADING.fullmatch(stripped):
            text, spans = inline(match.group(2))
            blocks.append(Block(BlockKind.HEADING, text, len(match.group(1)), spans))
            i += 1
        elif match := BULLET.fullmatch(line):
            blocks.append(_list_block(BlockKind.BULLET, match.group(1), match.group(2)))
            i += 1
        elif match := NUMBERED.fullmatch(line):
            blocks.append(_list_block(BlockKind.NUMBERED, match.group(1), match.group(2)))
            i += 1
        elif match := QUOTE.fullmatch(stripped):
            text, spans = inline(match.group(1))
            blocks.append(Block(BlockKind.QUOTE, text, spans=spans))
            i += 1
        else:
            # A paragraph runs to the next blank line or block; its line breaks are soft.
            parts = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip() and not _starts_block(lines, i):
                parts.append(lines[i].strip())
                i += 1
            text, spans = inline(" ".join(parts))
            blocks.append(Block(BlockKind.PARAGRAPH, text, spans=spans))
    return blocks


def _starts_block(lines: list[str], i: int) -> bool:
    stripped = lines[i].strip()
    return bool(
        stripped.startswith(("```", "|", ">"))
        or HEADING.fullmatch(stripped)
        or BULLET.fullmatch(lines[i])
        or NUMBERED.fullmatch(lines[i])
        or RULE.fullmatch(stripped)
    )


@dataclass(frozen=True)
class Tab:
    id: str
    title: str
    body: list[dict]
    # The tab's lists by id, which say whether a list is numbered.
    lists: dict = field(default_factory=dict)
    # The tab's dropdown definitions by id, with each dropdown chip's options.
    dropdown_definitions: dict = field(default_factory=dict)


def doc_tabs(document: dict) -> list[Tab]:
    """The document's tabs, child tabs after their parent. Needs a document read with includeTabsContent."""
    found: list[Tab] = []

    def walk(tabs: list[dict]) -> None:
        for tab in tabs:
            properties = tab["tabProperties"]
            document_tab = tab["documentTab"]
            found.append(
                Tab(
                    properties["tabId"],
                    properties["title"],
                    document_tab["body"]["content"],
                    document_tab.get("lists", {}),
                    document_tab.get("dropdownDefinitions", {}),
                )
            )
            walk(tab.get("childTabs", []))

    walk(document["tabs"])
    return found


def find_tab(document: dict, title: str | None) -> Tab:
    """The tab with this title (any case), or the first tab for None."""
    tabs = doc_tabs(document)
    if title is None:
        return tabs[0]
    matches = [tab for tab in tabs if tab.title.strip().lower() == title.strip().lower()]
    if not matches:
        raise ValueError(f"no tab named {title!r}; the tabs are {[tab.title for tab in tabs]}")
    return matches[0]


def paragraph_text(element: dict) -> str:
    """A paragraph's text, with dropdown chips as their value and images as [image]."""
    parts = []
    for part in element["paragraph"]["elements"]:
        if "textRun" in part:
            parts.append(part["textRun"]["content"])
        elif "dropdown" in part:
            parts.append(part["dropdown"]["dropdownProperties"].get("displayValue", ""))
        elif "inlineObjectElement" in part:
            parts.append("[image]")
    return "".join(parts)


def heading_level(element: dict) -> int | None:
    """1 to 6 for a heading paragraph, 0 for a title, None for any other element."""
    named = element.get("paragraph", {}).get("paragraphStyle", {}).get("namedStyleType", "")
    if named == "TITLE":
        return 0
    return int(named.removeprefix("HEADING_")) if named.startswith("HEADING_") else None


def _range(start: int, end: int, tab_id: str) -> dict:
    return {"startIndex": start, "endIndex": end, "tabId": tab_id}


def _text_style(start: int, end: int, style: dict, fields: str, tab_id: str) -> dict:
    return {"updateTextStyle": {"range": _range(start, end, tab_id), "textStyle": style, "fields": fields}}


def _paragraph_style(start: int, end: int, style: dict, tab_id: str) -> dict:
    return {
        "updateParagraphStyle": {
            "range": _range(start, end, tab_id),
            "paragraphStyle": style,
            "fields": ",".join(style),
        }
    }


def group_requests(blocks: list[Block], at: int, tab_id: str) -> list[dict]:
    """Requests that insert text blocks (no tables or diagrams) as new paragraphs at `at`, a paragraph start."""
    text = "".join(block.text + "\n" for block in blocks)
    end = at + u16(text)
    requests: list[dict] = [
        {"insertText": {"location": {"index": at, "tabId": tab_id}, "text": text}},
        {"deleteParagraphBullets": {"range": _range(at, end, tab_id)}},
        _text_style(at, end, {}, RESET_TEXT_FIELDS, tab_id),
    ]
    bullets: list[tuple[int, int, BlockKind]] = []
    block_at = at
    for block in blocks:
        block_end = block_at + u16(block.text)
        named = f"HEADING_{block.level}" if block.kind is BlockKind.HEADING else "NORMAL_TEXT"
        indent = QUOTE_INDENT_PT if block.kind is BlockKind.QUOTE else 0
        style = {"namedStyleType": named, "indentStart": {"magnitude": indent, "unit": "PT"}, "alignment": "START"}
        requests.append(_paragraph_style(block_at, block_end + 1, style, tab_id))
        if block.kind is BlockKind.CODE and block_end > block_at:
            code_font = {"weightedFontFamily": {"fontFamily": CODE_FONT}}
            requests.append(_text_style(block_at, block_end, code_font, "weightedFontFamily", tab_id))
        if block.kind is BlockKind.QUOTE and block_end > block_at:
            requests.append(_text_style(block_at, block_end, {"italic": True}, "italic", tab_id))
        requests += [_text_style(block_at + s.start, block_at + s.end, s.style, s.fields, tab_id) for s in block.spans]
        if block.kind in (BlockKind.BULLET, BlockKind.NUMBERED):
            if bullets and bullets[-1][2] is block.kind and bullets[-1][1] == block_at:
                bullets[-1] = (bullets[-1][0], block_end + 1, block.kind)
            else:
                bullets.append((block_at, block_end + 1, block.kind))
        block_at = block_end + 1
    for list_start, list_end, kind in reversed(bullets):
        preset = BULLET_PRESET if kind is BlockKind.BULLET else NUMBERED_PRESET
        requests.append(
            {"createParagraphBullets": {"range": _range(list_start, list_end, tab_id), "bulletPreset": preset}}
        )
    return requests


def table_fill_requests(table: dict, rows: tuple[tuple[str, ...], ...], tab_id: str) -> list[dict]:
    """Requests that write `rows` into a new, empty table element, the last cell first, with a bold header row."""
    requests: list[dict] = []
    cells = [
        (r, c, cell["content"][0]["startIndex"])
        for r, row in enumerate(table["table"]["tableRows"])
        for c, cell in enumerate(row["tableCells"])
    ]
    for r, c, cell_at in reversed(cells):
        text, spans = inline(rows[r][c])
        if not text:
            continue
        requests.append({"insertText": {"location": {"index": cell_at, "tabId": tab_id}, "text": text}})
        # A new table's cells take the text style of the paragraph before it, e.g. a code block's font.
        requests.append(_text_style(cell_at, cell_at + u16(text), {}, RESET_TEXT_FIELDS, tab_id))
        if r == 0:
            requests.append(_text_style(cell_at, cell_at + u16(text), {"bold": True}, "bold", tab_id))
        requests += [_text_style(cell_at + s.start, cell_at + s.end, s.style, s.fields, tab_id) for s in spans]
    return requests


def render_dot(source: str) -> bytes:
    result = subprocess.run(
        ["dot", "-Tpng", f"-Gdpi={DIAGRAM_DPI}"], input=source.encode(), capture_output=True, check=False
    )
    if result.returncode != 0:
        raise ValueError(f"Graphviz could not draw the diagram: {result.stderr.decode().strip()}")
    return result.stdout


def render_mermaid(source: str) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        source_file, png_file, config_file = (Path(directory) / name for name in ("in.mmd", "out.png", "config.json"))
        source_file.write_text(source)
        config_file.write_text(json.dumps(MERMAID_PUPPETEER_CONFIG))
        mmdc = ["npx", "--yes", f"--package={MERMAID_CLI}", "mmdc"]
        command = [*mmdc, "--quiet", "-p", str(config_file), "-i", str(source_file), "-o", str(png_file)]
        result = subprocess.run(
            [*command, "-s", str(MERMAID_SCALE), "-b", "white"], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise ValueError(
                f"mermaid-cli could not draw the diagram: {result.stderr.strip()}. Draw it in a ```dot block instead."
            )
        return png_file.read_bytes()


def render_diagram(block: Block) -> bytes:
    return render_mermaid(block.text) if block.language == MERMAID_LANGUAGE else render_dot(block.text)


def image_size_pt(png: bytes) -> tuple[float, float]:
    """The PNG's size in points at DIAGRAM_DPI, scaled down to fit the page width."""
    width_px, height_px = struct.unpack(">II", png[16:24])
    width, height = width_px * 72 / DIAGRAM_DPI, height_px * 72 / DIAGRAM_DPI
    scale = min(1.0, PAGE_WIDTH_PT / width)
    return width * scale, height * scale


@dataclass(frozen=True)
class DiagramStore:
    """Diagram images under DIAGRAM_PREFIX, served to Docs by URLs that `signer`, a service account, signs."""

    signer: str

    def publish(self, doc_id: str, png: bytes) -> str:
        url = f"{DIAGRAM_PREFIX}/{doc_id}/{hashlib.sha256(png).hexdigest()[:16]}.png"
        gcloud("storage", "cp", "--content-type=image/png", "-", url, stdin=png)
        return gcloud(
            "storage",
            "sign-url",
            f"--impersonate-service-account={self.signer}",
            f"--duration={DIAGRAM_URL_LIFETIME}",
            "--format=value(signed_url)",
            url,
        )


def diagram_requests(at: int, uri: str, size_pt: tuple[float, float], tab_id: str) -> list[dict]:
    """Requests that insert a centred image paragraph at `at`, a paragraph start."""
    width, height = size_pt
    return [
        {"insertText": {"location": {"index": at, "tabId": tab_id}, "text": "\n"}},
        {"deleteParagraphBullets": {"range": _range(at, at + 1, tab_id)}},
        _paragraph_style(at, at + 1, {"namedStyleType": "NORMAL_TEXT", "alignment": "CENTER"}, tab_id),
        {
            "insertInlineImage": {
                "uri": uri,
                "location": {"index": at, "tabId": tab_id},
                "objectSize": {
                    "width": {"magnitude": width, "unit": "PT"},
                    "height": {"magnitude": height, "unit": "PT"},
                },
            }
        },
    ]


def read_document(api: GoogleApi, doc_id: str) -> dict:
    return api.get(f"{DOCS_API}/documents/{doc_id}", params={"includeTabsContent": "true"})


def _batch_update(api: GoogleApi, doc_id: str, requests: list[dict]) -> None:
    api.post(f"{DOCS_API}/documents/{doc_id}:batchUpdate", {"requests": requests})


def _groups(blocks: list[Block]) -> list[list[Block]]:
    """Runs of text blocks, with each table and diagram on its own."""
    groups: list[list[Block]] = []
    for block in blocks:
        alone = block.kind in (BlockKind.TABLE, BlockKind.DIAGRAM)
        if alone or not groups or groups[-1][0].kind in (BlockKind.TABLE, BlockKind.DIAGRAM):
            groups.append([block])
        else:
            groups[-1].append(block)
    return groups


def _write_blocks(
    api: GoogleApi,
    doc_id: str,
    tab_title: str | None,
    blocks: list[Block],
    at: int,
    diagrams: DiagramStore | None,
) -> None:
    """Write blocks at `at`, a paragraph start in the tab, the last group first."""
    tab_id = find_tab(read_document(api, doc_id), tab_title).id
    for group in reversed(_groups(blocks)):
        block = group[0]
        if block.kind is BlockKind.DIAGRAM:
            png = render_diagram(block)
            uri = diagrams.publish(doc_id, png)
            _batch_update(api, doc_id, diagram_requests(at, uri, image_size_pt(png), tab_id))
        elif block.kind is BlockKind.TABLE:
            rows = block.rows
            table_request = {"rows": len(rows), "columns": len(rows[0]), "location": {"index": at, "tabId": tab_id}}
            _batch_update(api, doc_id, [{"insertTable": table_request}])
            # Docs puts a newline at `at` and the table after it.
            body = find_tab(read_document(api, doc_id), tab_title).body
            table = next(e for e in body if "table" in e and e["startIndex"] == at + 1)
            normal = _paragraph_style(at, at + 1, {"namedStyleType": "NORMAL_TEXT"}, tab_id)
            _batch_update(api, doc_id, [normal, *table_fill_requests(table, rows, tab_id)])
        else:
            _batch_update(api, doc_id, group_requests(group, at, tab_id))


def _add_last_paragraph(api: GoogleApi, doc_id: str, tab_id: str, end: int) -> None:
    """Add an empty normal paragraph after the tab's last paragraph, which ends at `end`.

    The new paragraph would take the last paragraph's style, e.g. a heading or a bullet, so it is reset.
    """
    new = _range(end, end + 1, tab_id)
    requests = [
        {"insertText": {"location": {"index": end - 1, "tabId": tab_id}, "text": "\n"}},
        {"deleteParagraphBullets": {"range": new}},
        _paragraph_style(end, end + 1, {"namedStyleType": "NORMAL_TEXT"}, tab_id),
    ]
    _batch_update(api, doc_id, requests)


def _parse_checked(markdown: str, diagrams: DiagramStore | None) -> list[Block]:
    blocks = parse(markdown)
    if diagrams is None and any(block.kind is BlockKind.DIAGRAM for block in blocks):
        raise ValueError("diagrams need a service account to sign their URLs; a personal gcloud login cannot")
    return blocks


def append_markdown(api: GoogleApi, doc_id: str, markdown: str, diagrams: DiagramStore | None, tab: str | None) -> None:
    """Add markdown at the end of a tab (the first tab for None)."""
    blocks = _parse_checked(markdown, diagrams)
    found = find_tab(read_document(api, doc_id), tab)
    last = found.body[-1]
    if "paragraph" in last and paragraph_text(last) == "\n":
        at = last["startIndex"]
    else:
        # Give the tab an empty last paragraph, so the blocks become new paragraphs before it.
        at = last["endIndex"]
        _add_last_paragraph(api, doc_id, found.id, at)
    _write_blocks(api, doc_id, tab, blocks, at, diagrams)


def top_insert_point(body: list[dict]) -> tuple[int, bool]:
    """(index, needs_paragraph) for new text at the top of a tab: below its first paragraph when that is a
    heading (the tab's title), else at the start. needs_paragraph is True when that heading is the tab's last
    paragraph, so an empty paragraph must be added after it first."""
    paragraphs = [element for element in body if "paragraph" in element]
    first = paragraphs[0]
    if heading_level(first) is None:
        return first["startIndex"], False
    return first["endIndex"], first is paragraphs[-1]


class MissingHeading(ValueError):
    """A tab has no heading with the given text."""


def _is_heading(element: dict, heading: str) -> bool:
    return heading_level(element) is not None and paragraph_text(element).strip().lower() == heading.strip().lower()


def _naming_tabs(error: MissingHeading, document: dict, tab: Tab) -> MissingHeading:
    """The error with the doc's tabs, since an older doc keeps its Progress section in a tab of its own."""
    tabs = [found.title for found in doc_tabs(document)]
    return MissingHeading(f"{error}, in tab {tab.title!r}; the doc's tabs are {tabs}, and --tab picks one")


def section_top(body: list[dict], heading: str) -> tuple[int, bool]:
    """(index, needs_paragraph) for new text at the top of a section, right below its heading. needs_paragraph
    is True when the heading is the tab's last paragraph, so an empty paragraph must be added after it first."""
    start, end, _ = section_bounds(body, heading)
    return start, end < start


def prepend_markdown(
    api: GoogleApi,
    doc_id: str,
    markdown: str,
    diagrams: DiagramStore | None,
    tab: str | None,
    section: str | None,
) -> None:
    """Add markdown at the top of a section (right below its heading), or of a tab (below its title heading)
    for no section, so the newest entry comes first."""
    blocks = _parse_checked(markdown, diagrams)
    document = read_document(api, doc_id)
    found = find_tab(document, tab)
    try:
        at, needs_paragraph = section_top(found.body, section) if section else top_insert_point(found.body)
    except MissingHeading as error:
        raise _naming_tabs(error, document, found) from None
    if needs_paragraph:
        _add_last_paragraph(api, doc_id, found.id, at)
    _write_blocks(api, doc_id, tab, blocks, at, diagrams)


def section_bounds(body: list[dict], heading: str) -> tuple[int, int, bool]:
    """(start, end, last) of the content under a heading: from the heading's end to the next heading of the
    same or a higher level, or to the tab's last newline when the section is last in the tab."""
    for i, element in enumerate(body):
        if not _is_heading(element, heading):
            continue
        level = heading_level(element)
        for later in body[i + 1 :]:
            later_level = heading_level(later)
            if later_level is not None and later_level <= level:
                return element["endIndex"], later["startIndex"], False
        return element["endIndex"], body[-1]["endIndex"] - 1, True
    headings = [paragraph_text(e).strip() for e in body if heading_level(e) is not None]
    raise MissingHeading(f"no heading {heading!r}; the headings are {headings}")


# The kinds of paragraph that replace_section edits in place: normal, heading, list item and quote.
EDITABLE_KINDS = ("P", "H", "L", "Q")
EMPTY_PARAGRAPH = ("P", 0, False, ())
UNNUMBERED_GLYPHS = (None, "GLYPH_TYPE_UNSPECIFIED", "NONE")


@dataclass(frozen=True)
class OldUnit:
    """A paragraph of a section, or an empty paragraph and the table after it, as replace_section compares it."""

    start: int
    end: int
    # (kind, level, numbered, runs) for a paragraph, ("T", cells) for a table, and a unique value for anything
    # replace_section cannot compare, such as an image or a chip.
    signature: tuple
    # A paragraph's text without its newline.
    text: str = ""


@dataclass(frozen=True)
class NewUnit:
    """A paragraph, table or diagram of new content, as replace_section compares it."""

    signature: tuple
    block: Block
    text: str = ""
    # Styles relative to text, with the ones over the whole paragraph, such as a code line's font.
    spans: tuple[Span, ...] = ()


@dataclass(frozen=True)
class WriteBlocks:
    """An edit that writes blocks as new paragraphs at a paragraph start (see _write_blocks)."""

    blocks: tuple[Block, ...]
    at: int


def _shape(unit: OldUnit | NewUnit) -> tuple | None:
    """(kind, level, numbered) of a paragraph that can be edited in place, else None."""
    return unit.signature[:3] if unit.signature[0] in EDITABLE_KINDS else None


def _runs(pieces: list[tuple[str, tuple]]) -> tuple:
    """(text, bold, italic, code, link) runs, with neighbours of the same style joined."""
    runs: list[tuple] = []
    for text, style in pieces:
        if not text:
            continue
        if runs and runs[-1][1:] == style:
            runs[-1] = (runs[-1][0] + text, *style)
        else:
            runs.append((text, *style))
    return tuple(runs)


def _span_runs(text: str, spans: tuple[Span, ...]) -> tuple:
    """The runs that text gets when it is written with spans."""
    styles = [[False, False, False, None] for _ in text]
    offset = 0
    for char, style in zip(text, styles, strict=True):
        for span in spans:
            if span.start <= offset < span.end:
                if span.fields == "link":
                    style[3] = span.style["link"]["url"]
                else:
                    style[("bold", "italic", "weightedFontFamily").index(span.fields)] = True
        offset += u16(char)
    return _runs([(char, tuple(style)) for char, style in zip(text, styles, strict=True)])


def _text_run_style(text_style: dict) -> tuple:
    return (
        text_style.get("bold", False),
        text_style.get("italic", False),
        text_style.get("weightedFontFamily", {}).get("fontFamily") == CODE_FONT,
        text_style.get("link", {}).get("url"),
    )


def _numbered(lists: dict, bullet: dict) -> bool:
    levels = lists.get(bullet["listId"], {}).get("listProperties", {}).get("nestingLevels", [])
    nesting = bullet.get("nestingLevel", 0)
    return nesting < len(levels) and levels[nesting].get("glyphType") not in UNNUMBERED_GLYPHS


def _old_paragraph(element: dict, lists: dict) -> OldUnit:
    start, end = element["startIndex"], element["endIndex"]
    paragraph = element["paragraph"]
    if not all("textRun" in part for part in paragraph["elements"]):
        return OldUnit(start, end, ("X", start))
    pieces = [
        (part["textRun"]["content"], _text_run_style(part["textRun"].get("textStyle", {})))
        for part in paragraph["elements"]
    ]
    # Leave out the paragraph's newline.
    pieces[-1] = (pieces[-1][0].removesuffix("\n"), pieces[-1][1])
    level, bullet = heading_level(element), paragraph.get("bullet")
    if level is not None:
        shape = ("H", level, False)
    elif bullet is not None:
        shape = ("L", bullet.get("nestingLevel", 0), _numbered(lists, bullet))
    elif paragraph.get("paragraphStyle", {}).get("indentStart", {}).get("magnitude", 0) == QUOTE_INDENT_PT:
        shape = ("Q", 0, False)
    else:
        shape = ("P", 0, False)
    return OldUnit(start, end, (*shape, _runs(pieces)), "".join(text for text, _ in pieces))


def _old_table_cells(element: dict) -> tuple | None:
    """The runs of each cell, or None for a table with a cell that is not one plain paragraph."""
    rows = []
    for row in element["table"]["tableRows"]:
        cells = []
        for cell in row["tableCells"]:
            content = cell["content"]
            unit = _old_paragraph(content[0], {}) if len(content) == 1 and "paragraph" in content[0] else None
            if unit is None or unit.signature[0] != "P":
                return None
            cells.append(unit.signature[3])
        rows.append(tuple(cells))
    return tuple(rows)


def old_units(elements: list[dict], lists: dict) -> list[OldUnit]:
    """A section's elements as units. A table joins the empty paragraph before it, which Docs adds with it."""
    units: list[OldUnit] = []
    for element in elements:
        start, end = element["startIndex"], element["endIndex"]
        if "paragraph" in element:
            units.append(_old_paragraph(element, lists))
        elif (
            "table" in element
            and (cells := _old_table_cells(element)) is not None
            and units
            and units[-1].signature == EMPTY_PARAGRAPH
        ):
            units[-1] = OldUnit(units[-1].start, end, ("T", cells))
        else:
            units.append(OldUnit(start, end, ("X", start)))
    return units


def _new_paragraph(block: Block) -> NewUnit:
    tabs = block.level if block.kind in (BlockKind.BULLET, BlockKind.NUMBERED) else 0
    text = block.text[tabs:]
    spans = tuple(Span(s.start - tabs, s.end - tabs, s.style, s.fields) for s in block.spans)
    if text and block.kind is BlockKind.CODE:
        spans = (Span(0, u16(text), {"weightedFontFamily": {"fontFamily": CODE_FONT}}, "weightedFontFamily"), *spans)
    elif text and block.kind is BlockKind.QUOTE:
        spans = (Span(0, u16(text), {"italic": True}, "italic"), *spans)
    shape = {
        BlockKind.HEADING: ("H", block.level, False),
        BlockKind.BULLET: ("L", block.level, False),
        BlockKind.NUMBERED: ("L", block.level, True),
        BlockKind.QUOTE: ("Q", 0, False),
    }.get(block.kind, ("P", 0, False))
    return NewUnit((*shape, _span_runs(text, spans)), block, text, spans)


def _new_table_cells(rows: tuple[tuple[str, ...], ...]) -> tuple:
    """The runs of each cell as table_fill_requests writes them, with a bold header row."""
    cells = []
    for r, row in enumerate(rows):
        row_cells = []
        for cell in row:
            text, spans = inline(cell)
            if r == 0 and text:
                spans = (Span(0, u16(text), {"bold": True}, "bold"), *spans)
            row_cells.append(_span_runs(text, spans))
        cells.append(tuple(row_cells))
    return tuple(cells)


def new_units(blocks: list[Block]) -> list[NewUnit]:
    """Blocks as units. Each line of a code block is a paragraph of its own; a diagram never equals anything."""
    units = []
    for block in blocks:
        if block.kind is BlockKind.CODE:
            units += [
                _new_paragraph(Block(BlockKind.CODE, line, language=block.language)) for line in block.text.split("\n")
            ]
        elif block.kind is BlockKind.TABLE:
            units.append(NewUnit(("T", _new_table_cells(block.rows)), block))
        elif block.kind is BlockKind.DIAGRAM:
            units.append(NewUnit(("D", object()), block))
        else:
            units.append(_new_paragraph(block))
    return units


def _styled(units: list[NewUnit], at: int, separator_first: bool, tab_id: str) -> tuple[str, list[dict]]:
    """The text of units as paragraphs inserted at `at`, and the requests that style it after the insert.

    With separator_first, each paragraph starts with its newline (inserted before an existing newline), else ends
    with it (inserted at a paragraph start)."""
    text = "".join("\n" + u.text if separator_first else u.text + "\n" for u in units)
    requests = [_text_style(at, at + u16(text), {}, RESET_TEXT_FIELDS, tab_id)]
    offset = at
    for unit in units:
        offset += 1 if separator_first else 0
        requests += [_text_style(offset + s.start, offset + s.end, s.style, s.fields, tab_id) for s in unit.spans]
        offset += u16(unit.text) + (0 if separator_first else 1)
    return text, requests


def _insert(
    units: list[NewUnit], at: int, previous: OldUnit | None, following: OldUnit | None, tab_id: str
) -> list[dict] | WriteBlocks:
    """Requests that insert units at `at`, a paragraph start between previous and following.

    List items of the same kind and level as a neighbouring list item go into that item's list, so a numbered
    list keeps counting: before the following item, which they split off, or else after the previous one.
    """
    shapes = {_shape(unit) for unit in units}
    shape = shapes.pop() if len(shapes) == 1 else None
    if shape is None or shape[0] != "L":
        return WriteBlocks(tuple(unit.block for unit in units), at)
    if following is not None and _shape(following) == shape:
        position, separator_first = at, False
    elif previous is not None and _shape(previous) == shape:
        # Just before the previous item's newline, so the new items follow it.
        position, separator_first = previous.end - 1, True
    else:
        return WriteBlocks(tuple(unit.block for unit in units), at)
    text, styles = _styled(units, position, separator_first, tab_id)
    return [{"insertText": {"location": {"index": position, "tabId": tab_id}, "text": text}}, *styles]


def _edit_paragraph(old: OldUnit, new: NewUnit, tab_id: str) -> list[dict]:
    """Requests that change only the characters of a paragraph that differ, then set its styles."""
    requests = []
    matcher = difflib.SequenceMatcher(None, old.text, new.text, autojunk=False)
    for tag, i1, i2, j1, j2 in reversed(matcher.get_opcodes()):
        if tag == "equal":
            continue
        at = old.start + u16(old.text[:i1])
        if i2 > i1:
            requests.append({"deleteContentRange": {"range": _range(at, old.start + u16(old.text[:i2]), tab_id)}})
        if j2 > j1:
            requests.append({"insertText": {"location": {"index": at, "tabId": tab_id}, "text": new.text[j1:j2]}})
    if new.text:
        end = old.start + u16(new.text)
        requests.append(_text_style(old.start, end, {}, RESET_TEXT_FIELDS, tab_id))
        requests += [_text_style(old.start + s.start, old.start + s.end, s.style, s.fields, tab_id) for s in new.spans]
    return requests


def section_edits(old: list[OldUnit], new: list[NewUnit], end: int, tab_id: str) -> list[list[dict] | WriteBlocks]:
    """The edits that turn a section's units into new ones, the last in the tab first, so each edit's indices
    stay valid. end is where the section ends, a paragraph start."""
    edits: list[list[dict] | WriteBlocks] = []
    matcher = difflib.SequenceMatcher(None, [u.signature for u in old], [u.signature for u in new], autojunk=False)
    for tag, i1, i2, j1, j2 in reversed(matcher.get_opcodes()):
        if tag == "equal":
            continue
        # Pair old and new paragraphs of the same shape to edit in place; else replace the old units.
        pairs = min(i2 - i1, j2 - j1)
        if not all(_shape(old[i1 + k]) is not None and _shape(old[i1 + k]) == _shape(new[j1 + k]) for k in range(pairs)):
            pairs = 0
        if i2 - i1 > pairs:
            deleted = old[i1 + pairs : i2]
            edits.append([{"deleteContentRange": {"range": _range(deleted[0].start, deleted[-1].end, tab_id)}}])
        if j2 - j1 > pairs:
            # After that delete, the unit after the deleted ones starts where the first deleted one did.
            at = old[i1 + pairs].start if i1 + pairs < len(old) else end
            previous = old[i1 + pairs - 1] if i1 + pairs > 0 else None
            following = old[i2] if i2 < len(old) else None
            edits.append(_insert(list(new[j1 + pairs : j2]), at, previous, following, tab_id))
        edits += [_edit_paragraph(old[i1 + k], new[j1 + k], tab_id) for k in reversed(range(pairs))]
    return edits


def replace_section(
    api: GoogleApi, doc_id: str, tab: str | None, heading: str, markdown: str, diagrams: DiagramStore | None
) -> None:
    """Make the content under a heading in a tab equal to markdown, changing only what differs (see the module
    docstring). The heading itself stays."""
    blocks = _parse_checked(markdown, diagrams)
    document = read_document(api, doc_id)
    found = find_tab(document, tab)
    try:
        start, end, last = section_bounds(found.body, heading)
    except MissingHeading as error:
        raise _naming_tabs(error, document, found) from None
    if last and paragraph_text(found.body[-1]) != "\n":
        # A last section ends at an empty paragraph, which no edit deletes, since a tab must end with a newline.
        _add_last_paragraph(api, doc_id, found.id, found.body[-1]["endIndex"])
        found = find_tab(read_document(api, doc_id), tab)
        start, end, last = section_bounds(found.body, heading)
    if last:
        end = found.body[-1]["startIndex"]
    # The tab's first element, a section break, has no startIndex.
    content = [element for element in found.body if start <= element.get("startIndex", 0) < end]
    units = old_units(content, found.lists)
    requests: list[dict] = []
    for edit in section_edits(units, new_units(blocks), end, found.id):
        if isinstance(edit, WriteBlocks):
            if requests:
                _batch_update(api, doc_id, requests)
                requests = []
            _write_blocks(api, doc_id, tab, list(edit.blocks), edit.at, diagrams)
        else:
            requests += edit
    if requests:
        _batch_update(api, doc_id, requests)


def tab_text(tab: Tab) -> str:
    """The tab as plain markdown-like text, for an agent to read: headings, bullets and table rows."""
    lines = []
    for element in tab.body:
        if "table" in element:
            for row in element["table"]["tableRows"]:
                cells = [
                    "".join(paragraph_text(c) for c in cell["content"] if "paragraph" in c).strip()
                    for cell in row["tableCells"]
                ]
                lines.append("| " + " | ".join(cells) + " |")
            continue
        if "paragraph" not in element:
            continue
        text = paragraph_text(element).rstrip("\n")
        level = heading_level(element)
        bullet = element["paragraph"].get("bullet")
        if level:
            text = "#" * level + " " + text
        elif bullet is not None:
            text = "  " * bullet.get("nestingLevel", 0) + "- " + text
        lines.append(text)
    return "\n".join(lines)


def tab_length(tab: Tab, exclude_section: str | None) -> tuple[int, int, float]:
    """(words, images, estimated pages) of a tab, without one section (its heading and content) when given
    and the tab has it. Docs has no page count, so pages are an estimate."""
    if exclude_section and any(_is_heading(element, exclude_section) for element in tab.body):
        start, end, _ = section_bounds(tab.body, exclude_section)
        kept = [e for e in tab.body if e["endIndex"] != start and not start <= e.get("startIndex", 0) < end]
        tab = Tab(tab.id, tab.title, kept)
    # Plain text, without tab_text's markdown markers, which would count as words.
    texts = []
    for element in tab.body:
        if "paragraph" in element:
            texts.append(paragraph_text(element))
        elif "table" in element:
            texts += [
                paragraph_text(c)
                for row in element["table"]["tableRows"]
                for cell in row["tableCells"]
                for c in cell["content"]
                if "paragraph" in c
            ]
    text = " ".join(texts)
    images = text.count("[image]")
    words = len(text.replace("[image]", "").split())
    return words, images, words / WORDS_PER_PAGE + images * PAGES_PER_DIAGRAM
