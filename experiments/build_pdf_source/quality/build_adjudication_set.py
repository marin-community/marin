# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package documents for blind adjudication of which extraction route read the page correctly.

The agreement numbers in :mod:`~experiments.build_pdf_source.quality.route_agreement` say how far
apart the two routes are. They cannot say which one is right, and on this corpus that distinction
is not academic: sorting by disagreement surfaces chart-heavy pages where Docling transcribed axis
labels the VLM was told to ignore, and RTL documents where Docling's Unicode is visibly better than
the VLM's. Treating either as "Docling failed" would train the router to send good documents to the
expensive route, and the reverse mistake is worse.

So the label comes from looking at the page. This module writes, per document, a directory holding
the rendered pages and both extractions of those same pages, with the routes **blinded and their
order randomized per document** -- a judge sees "Extraction A" and "Extraction B" and cannot tell
which produced them, so a systematic preference for one style cannot masquerade as a verdict. The
key mapping is written separately, next to the work rather than inside it.

Pages are chosen rather than taken from the front: the first page of a report is a title page that
both routes get right and that says nothing about the document. :func:`informative_pages` picks the
pages where the two extractions differ most, which is where the verdict is actually decided, and
always includes one page where they agree so a judge can calibrate on the document's own typography.

Rendering is at :data:`RENDER_DPI`, well above the ~146 DPI median the VLM itself saw, because the
judge has to be able to read what the model was working from and adjudicate ligatures and equation
layout at the same time.
"""

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from experiments.build_pdf_source.quality import route_agreement

logger = logging.getLogger(__name__)

RENDER_DPI = 160
# Pages shown per document. Three is what fits a judge's attention while still showing a
# disagreement, a control, and one more of whatever the document is mostly made of.
PAGES_PER_DOCUMENT = 3
# Characters of each extraction shown per page. Long enough to see structure and reading order,
# short enough that three pages of both routes stay readable.
EXCERPT_CHARS = 4000

ROUTES = ("docling", "vlm")


@dataclass(frozen=True)
class PageChoice:
    """A page selected for adjudication: the PDF page, and each route's text for it.

    ``docling_index`` is ``None`` when Docling has no page matching this one, which is a real and
    common outcome -- Docling drops pages it reads nothing from.
    """

    page_index: int
    docling_index: int | None
    reason: str
    bigram_recall: float


def informative_pages(docling_pages: list[str], ocr_pages: list[str], count: int) -> list[PageChoice]:
    """Pick the pages worth adjudicating: the worst disagreements, plus one agreeing control.

    Pages are matched through :func:`route_agreement.align_pages` rather than by index. Pairing by
    index here is not a cosmetic error: on a document where Docling dropped a page, every later
    page is shown beside its neighbour's text, and judges reading those packets reported one route
    "fabricating" content and the other "losing" it on documents where neither had done anything
    wrong. The PDF page number is taken from the VLM side, whose page list is the PDF's own.

    A page both routes left empty is skipped -- a blank page adjudicates nothing, and on a scanned
    document it would be most of what gets shown.
    """
    scored: list[PageChoice] = []
    for docling_index, ocr_index in route_agreement.align_pages(docling_pages, ocr_pages):
        if ocr_index is None:
            # A page only Docling produced has no PDF page number on the VLM's side to render.
            continue
        docling_page = docling_pages[docling_index] if docling_index is not None else ""
        agreement = route_agreement.page_agreement(docling_page, ocr_pages[ocr_index])
        if agreement.ocr_tokens == 0 and agreement.docling_tokens == 0:
            continue
        scored.append(
            PageChoice(
                page_index=ocr_index,
                docling_index=docling_index,
                reason="disagreement",
                bigram_recall=agreement.bigram_recall,
            )
        )
    if not scored:
        return [PageChoice(page_index=0, docling_index=0, reason="only page", bigram_recall=0.0)]

    scored.sort(key=lambda choice: choice.bigram_recall)
    chosen = scored[: max(count - 1, 1)]
    # One page the routes agree on, as a control: it tells a judge what this document's typography
    # looks like when nothing has gone wrong, so a verdict rests on the difference rather than on
    # unfamiliarity with the layout.
    control = scored[-1]
    if control.page_index not in {choice.page_index for choice in chosen}:
        chosen.append(
            PageChoice(
                page_index=control.page_index,
                docling_index=control.docling_index,
                reason="control",
                bigram_recall=control.bigram_recall,
            )
        )
    return sorted(chosen, key=lambda choice: choice.page_index)


def render_page(doc: pymupdf.Document, index: int, destination: Path) -> None:
    pixmap = doc.load_page(index).get_pixmap(dpi=RENDER_DPI)
    pixmap.save(destination)


def excerpt(text: str) -> str:
    """A page's extraction, whitespace-tidied and clipped to a readable length."""
    collapsed = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(collapsed) <= EXCERPT_CHARS:
        return collapsed or "(this route produced no text for this page)"
    return f"{collapsed[:EXCERPT_CHARS]}\n... [clipped at {EXCERPT_CHARS} characters]"


def write_document(row: dict, destination: Path, rng: random.Random) -> dict:
    """Write one document's adjudication packet and return its key entry.

    Returns the blinding key -- which label each route was given -- which the caller keeps out of
    the packet directory.
    """
    docling_pages = route_agreement.split_pages(row["docling_text"], row["docling_page_offsets"])
    ocr_pages = route_agreement.split_pages(row["text"], row["page_offsets"])
    choices = informative_pages(docling_pages, ocr_pages, PAGES_PER_DOCUMENT)

    labels = dict(zip(ROUTES, rng.sample(["A", "B"], 2), strict=True))
    destination.mkdir(parents=True, exist_ok=True)

    with pymupdf.open(stream=row["pdf"], filetype="pdf") as doc:
        rendered = []
        for choice in choices:
            if choice.page_index >= len(doc):
                continue
            name = f"page_{choice.page_index + 1:03d}.png"
            render_page(doc, choice.page_index, destination / name)
            rendered.append((choice, name))

    sections = [
        f"# Document {row['source_id']}",
        f"source url: {row['url']}",
        f"pages in PDF: {row['num_pages']}",
        "",
        "Each section below is one page of this PDF. The rendered page image is the ground truth; "
        "the two extractions are what two different systems read off it.",
        "",
    ]
    for choice, name in rendered:
        page_texts = {
            "docling": docling_pages[choice.docling_index] if choice.docling_index is not None else "",
            "vlm": ocr_pages[choice.page_index],
        }
        sections.append(f"## Page {choice.page_index + 1} (image: {name})")
        for route in ROUTES:
            sections.append(f"### Extraction {labels[route]}")
            sections.append(excerpt(page_texts[route]))
            sections.append("")

    (destination / "document.md").write_text("\n".join(sections))
    return {
        "source_id": row["source_id"],
        "directory": destination.name,
        "url": row["url"],
        "labels": {label: route for route, label in labels.items()},
        "pages": [
            {
                "page_index": choice.page_index,
                "docling_index": choice.docling_index,
                "reason": choice.reason,
                "image": name,
            }
            for choice, name in rendered
        ],
    }


def build(rows: list[dict], destination: Path, seed: int) -> Path:
    """Write an adjudication set for *rows* and return the path to its blinding key."""
    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    key = []
    for position, row in enumerate(rows):
        try:
            key.append(write_document(row, destination / f"doc_{position:04d}", rng))
        except Exception as error:
            logger.warning("Could not package %s: %s", row["source_id"], error)
    key_path = destination.parent / f"{destination.name}_key.json"
    key_path.write_text(json.dumps(key, indent=2))
    logger.info("Packaged %d/%d documents -> %s (key: %s)", len(key), len(rows), destination, key_path)
    return key_path
