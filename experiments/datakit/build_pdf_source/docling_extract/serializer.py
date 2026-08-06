# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serialise the assembled document to text for a language model, not Markdown for a reader.

Docling's Markdown serialiser is built for display: it escapes underscores and HTML, marks headings
with hashes, emphasis with asterisks, and links with bracket syntax. None of that is content, and
all of it is tokens, so it is stripped here. What replaces it is a small set of explicit tags that
let a downstream filter find the structured regions and decide what to do with them:

``<docling_table>``
    A ruled table, rendered as a GitHub-style grid.
``<docling_picture_annotation>`` / ``<docling_picture_annotation_non_text>``
    Text found inside a figure. The two tags separate captions and labelled diagrams, which are
    prose, from axis ticks and legends, which are not: a region whose alphabetic share falls below
    :data:`DEFAULT_ALPHA_RATIO` gets the ``_non_text`` tag. Nothing is dropped at this stage --
    the tag records the judgement and a later pass acts on it.
``<docling_formula>``
    A formula region, which is LaTeX at best and noise at worst.
``<docling_image>``
    Where an image sat, so that a paragraph split by a figure is not silently joined.

Pages are separated by :data:`PAGE_BREAK`, which is how the extractor recovers per-page offsets
without serialising each page separately.
"""

from pathlib import Path
from typing import Any

from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    BasePictureSerializer,
    BaseTableSerializer,
    BaseTextSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
    MarkdownPictureSerializer,
    MarkdownTextSerializer,
    OrigListItemMarkerMode,
)
from docling_core.types.doc.document import (
    DoclingDocument,
    FormulaItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)
from pydantic import AnyUrl
from tabulate import tabulate

PAGE_BREAK = "<--- page break --->"
IMAGE_PLACEHOLDER = "<docling_image></docling_image>"
# Below this alphabetic share, text inside a figure is chrome rather than prose.
DEFAULT_ALPHA_RATIO = 0.4


def alphabetic_ratio(text: str) -> float:
    """Share of a string that is alphabetic. Empty text counts as entirely non-alphabetic."""
    if not text:
        return 0.0
    return sum(character.isalpha() for character in text) / len(text)


class PlainTextSerializer(MarkdownTextSerializer):
    """Text items without heading markers, with formulas tagged."""

    def _format_heading(self, text: str, item: TitleItem | SectionHeaderItem) -> str:
        """Headings are content, not structure: drop the hashes and keep the words."""
        return text

    def serialize(self, *, item: TextItem, **kwargs: Any) -> SerializationResult:
        result = super().serialize(item=item, **kwargs)
        if isinstance(item, FormulaItem):
            return create_ser_result(text=f"<docling_formula>{result.text}</docling_formula>", span_source=item)
        return result


class TaggedTableSerializer(BaseTableSerializer):
    """Tables as a GitHub-style grid inside ``<docling_table>``, captions outside it."""

    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        parts: list[SerializationResult] = []

        caption = doc_serializer.serialize_captions(item=item, **kwargs)
        if caption.text:
            parts.append(caption)

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            # A newline inside a cell would break the row apart, so cells are flattened first.
            rows = [[cell.text.replace("\n", " ") for cell in row] for row in item.data.grid]
            table_text = ""
            if len(rows) > 1 and rows[0]:
                try:
                    table_text = tabulate(rows[1:], headers=rows[0], tablefmt="github")
                except ValueError:
                    # tabulate parses numbers to align them, and throws on ragged numeric columns.
                    table_text = tabulate(rows[1:], headers=rows[0], tablefmt="github", disable_numparse=True)
            parts.append(create_ser_result(text=f"<docling_table>{table_text}</docling_table>", span_source=item))

        return create_ser_result(text="\n\n".join(part.text for part in parts), span_source=parts)


class AnnotatedPictureSerializer(MarkdownPictureSerializer):
    """Figures as their placeholder plus whatever text sits inside them, tagged by prose-likeness."""

    def __init__(self, alpha_ratio: float = DEFAULT_ALPHA_RATIO, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_ratio = alpha_ratio

    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        separator: str | None = None,
        **kwargs: Any,
    ) -> SerializationResult:
        parent = super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)
        parts = [parent.text]

        # Captions are serialised by the parent; anything else inside the figure is annotation.
        parts.extend(
            child.text
            for child, _ in doc.iterate_items(root=item, traverse_pictures=True)
            if isinstance(child, TextItem) and child.get_ref() not in item.captions
        )

        text = (separator or "\n").join(parts)
        tag = (
            "docling_picture_annotation"
            if alphabetic_ratio(text) >= self.alpha_ratio
            else "docling_picture_annotation_non_text"
        )
        return create_ser_result(text=f"<{tag}>{text}</{tag}>", span_source=item)


class PlainTextDocSerializer(MarkdownDocSerializer):
    """The document serialiser this extractor uses: docling's, with the display syntax removed."""

    text_serializer: BaseTextSerializer = PlainTextSerializer()
    table_serializer: BaseTableSerializer = TaggedTableSerializer()
    picture_serializer: BasePictureSerializer = AnnotatedPictureSerializer()

    def serialize_bold(self, text: str, **kwargs: Any) -> str:
        return text

    def serialize_italic(self, text: str, **kwargs: Any) -> str:
        return text

    def serialize_strikethrough(self, text: str, **kwargs: Any) -> str:
        return text

    def serialize_hyperlink(self, text: str, hyperlink: AnyUrl | Path, **kwargs: Any) -> str:
        """Keep the anchor text, drop the URL: a link target is not something a reader reads."""
        return text


def text_params(alpha_ratio: float = DEFAULT_ALPHA_RATIO) -> MarkdownParams:
    """Serialisation parameters for plain text output."""
    return MarkdownParams(
        page_break_placeholder=PAGE_BREAK,
        image_placeholder=IMAGE_PLACEHOLDER,
        escape_underscores=False,
        escape_html=False,
        # Emit each list item's own marker, which the normaliser has already mapped onto "-", "*"
        # or "[x]", and suppress docling's Markdown marker so items do not get a second one.
        ensure_valid_list_item_marker=False,
        orig_list_item_marker_mode=OrigListItemMarkerMode.ALWAYS,
    )


def serialize_document(doc: DoclingDocument, alpha_ratio: float = DEFAULT_ALPHA_RATIO) -> str:
    """Serialise a postprocessed document to text, page breaks included."""
    serializer = PlainTextDocSerializer(
        doc=doc,
        params=text_params(alpha_ratio),
        picture_serializer=AnnotatedPictureSerializer(alpha_ratio=alpha_ratio),
    )
    return serializer.serialize().text
