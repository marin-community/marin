# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FinePDFs OCR router's PyMuPDF feature extraction."""

import random
import struct
import zlib

import pytest

from experiments.build_pdf_source.ocr_feature_names import DOC_FEATURE_NAMES, FEATURE_NAMES, FEATURE_PAGES

# PyMuPDF only ships in marin-core's ``datakit`` extra, which the workspace root does not install,
# and ``ocr_features`` imports it at module scope -- so the skip has to precede that import. The
# feature contract above is pure data and needs no such guard.
pymupdf = pytest.importorskip("pymupdf")

from experiments.build_pdf_source.ocr_features import (  # noqa: E402
    CorruptPdf,
    _PagePass,
    document_features,
    junk_image_xrefs,
    merge_image_strips,
    sample_page_indices,
)

_SEED = 4242


def _solid_png(color: tuple[int, int, int], width: int = 48, height: int = 48) -> bytes:
    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(b"\x00" + bytes(color) * width for _ in range(height))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def _page_scan(variant: int) -> bytes:
    """A full-page bitmap, distinct per page like a real scan's."""
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 300, 400))
    pixmap.set_rect(pixmap.irect, (250, 248, 244))
    for y in range(10, 400, 17 + variant % 3):
        pixmap.set_rect(pymupdf.IRect(20, y, 280, y + 5), (20, 20, 20))
    return pixmap.tobytes("jpeg", jpg_quality=60)


def _document(pages: int, build) -> pymupdf.Document:
    doc = pymupdf.open()
    for index in range(pages):
        build(doc.new_page(width=612, height=792), index)
    return pymupdf.open(stream=doc.tobytes(deflate=True), filetype="pdf")


def _text_page(page, index: int) -> None:
    page.insert_text((72, 100), f"Born-digital body text on page {index}. " * 10, fontsize=11)


def _scanned_page(page, index: int) -> None:
    page.insert_image(pymupdf.Rect(0, 0, 612, 792), stream=_page_scan(index))


def test_feature_names_cover_the_boosters_layout():
    """The booster's 124 inputs are four document features then each page feature across 8 slots."""
    assert len(FEATURE_NAMES) == 124
    assert FEATURE_NAMES[: len(DOC_FEATURE_NAMES)] == DOC_FEATURE_NAMES
    page_names = FEATURE_NAMES[len(DOC_FEATURE_NAMES) :]
    assert len(page_names) % FEATURE_PAGES == 0
    # Page slots of one feature are contiguous, so the first block runs page1..page8.
    assert [name.rpartition("_page")[2] for name in page_names[:FEATURE_PAGES]] == [
        str(slot) for slot in range(1, FEATURE_PAGES + 1)
    ]


def test_scanned_and_text_documents_produce_opposite_evidence():
    """The features the router keys on must separate a page of pixels from a page of glyphs."""
    with _document(FEATURE_PAGES, _scanned_page) as doc:
        scanned = document_features(doc, seed=_SEED)
    with _document(FEATURE_PAGES, _text_page) as doc:
        text = document_features(doc, seed=_SEED)

    assert all(page.char_count == 0 for page in scanned.pages)
    assert all(page.bitmap_proportion > 0.9 for page in scanned.pages)
    assert all(page.image_count == 1 for page in scanned.pages)

    assert all(page.char_count > 0 for page in text.pages)
    assert all(page.bitmap_proportion == 0.0 for page in text.pages)
    assert all(page.image_count == 0 for page in text.pages)


def test_vector_has_one_float_per_feature_name():
    with _document(FEATURE_PAGES, _text_page) as doc:
        vector = document_features(doc, seed=_SEED).vector()

    assert vector.shape == (len(FEATURE_NAMES),)
    assert vector.dtype.name == "float32"


def test_short_documents_fill_every_page_slot_by_repeating_sampled_pages():
    """The booster always wants 8 page slots, so a 3-page document repeats what it has."""
    with _document(3, _text_page) as doc:
        features = document_features(doc, seed=_SEED)

    assert features.num_pages == 3
    assert features.num_pages_successfully_sampled == 3
    assert len(features.pages) == FEATURE_PAGES
    # Every slot, padding included, describes one of the three real pages.
    assert set(features.pages) <= set(features.pages[:3])


def test_features_are_reproducible_for_a_given_seed():
    """A re-run of a shard has to reproduce its predictions, which page sampling threatens."""
    with _document(40, _text_page) as doc:
        first = document_features(doc, seed=_SEED).vector()
        again = document_features(doc, seed=_SEED).vector()
        other = document_features(doc, seed=_SEED + 1)

    assert (first == again).all()
    assert other.num_pages_successfully_sampled == FEATURE_PAGES


def test_only_pages_are_sampled_not_the_whole_document():
    """Reading every page of a long document was the upstream cost this port removes."""
    with _document(60, _text_page) as doc:
        features = document_features(doc, seed=_SEED)

    assert features.num_pages == 60
    assert features.num_pages_successfully_sampled == FEATURE_PAGES


def test_an_image_on_every_page_is_junk_but_a_per_page_figure_is_not():
    def letterhead(page, index: int) -> None:
        page.insert_image(pymupdf.Rect(20, 20, 120, 70), stream=_solid_png((10, 40, 200)))
        page.insert_image(pymupdf.Rect(200, 300, 400, 500), stream=_solid_png((index * 30 % 256, 90, 30)))

    with _document(FEATURE_PAGES, letterhead) as doc:
        features = document_features(doc, seed=_SEED)

    # Two images drawn per page; the reused logo is excluded, the changing figure is kept.
    assert all(page.image_count == 2 for page in features.pages)
    assert all(page.non_junk_image_count == 1 for page in features.pages)


def test_a_page_whose_images_cannot_be_read_still_counts_as_sampled(monkeypatch):
    """Only ``load_page`` decides whether a page counts, matching the original's granularity.

    Dropping the page instead would change ``num_pages_successfully_sampled`` -- a document-level
    feature -- and slide every later page slot, on any PDF with one unreadable resource list.
    """
    with _document(FEATURE_PAGES, _scanned_page) as doc:
        monkeypatch.setattr(
            pymupdf.Page, "get_images", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("broken"))
        )
        features = document_features(doc, seed=_SEED)

    assert features.num_pages_successfully_sampled == FEATURE_PAGES
    assert len(features.pages) == FEATURE_PAGES
    # The images are gone, but the text measurements survived.
    assert all(page.image_count == 0 for page in features.pages)
    assert all(page.bitmap_proportion == 0.0 for page in features.pages)


def test_an_unreadable_sampled_page_leaves_nothing_marked_junk():
    """The junk threshold counts pages *drawn*, not pages read -- as the original does.

    An xref's appearance count cannot exceed the number of pages that yielded resources, so one
    unreadable page puts the threshold out of reach. Counting only readable pages would find junk
    where the original finds none and silently lower ``non_junk_image_count``.
    """
    passes = [
        _PagePass(
            page_index=index,
            text_length=0,
            garbled_count=0,
            unique_font_count=0,
            char_count=0,
            text_box_count=0,
            text_area=0.0,
            hidden_char_count=0,
            hidden_text_box_count=0,
            hidden_text_area=0.0,
            page_width=612.0,
            page_height=792.0,
            resource_xrefs=frozenset({7}),
            resource_shapes=frozenset({(10, 10, 8)}),
            placements=((0.0, 0.0, 10.0, 10.0),),
            drawing_stroke_count=0,
            vector_graphics_object_count=0,
        )
        for index in range(4)
    ]

    # All four drawn pages named xref 7, so it is page furniture.
    assert junk_image_xrefs(passes, num_sampled=4) == frozenset({7})
    # A fifth page was drawn but could not be read, so 7 appears on 4 of 5 and is not junk.
    assert junk_image_xrefs(passes, num_sampled=5) == frozenset()


def test_no_image_is_junk_when_too_few_pages_were_sampled():
    """Below three sampled pages, "on every page" carries no signal."""

    def logo_only(page, _index: int) -> None:
        page.insert_image(pymupdf.Rect(20, 20, 120, 70), stream=_solid_png((10, 40, 200)))

    with _document(2, logo_only) as doc:
        features = document_features(doc, seed=_SEED)

    assert all(page.non_junk_image_count == 1 for page in features.pages)


def test_inline_images_are_not_counted_as_page_images():
    """The router was trained on resource-list images only, so ``BI``/``ID``/``EI`` art is ignored.

    Counting inline images would inflate ``image_count`` relative to the training distribution, and
    they cannot be excluded by decoding because they are not objects at all.
    """
    plain = pymupdf.open()
    for index in range(FEATURE_PAGES):
        plain.new_page(width=612, height=792).insert_text((72, 400), f"body {index}", fontsize=11)
    plain_bytes = plain.tobytes()
    plain.close()

    doc = pymupdf.open(stream=plain_bytes, filetype="pdf")
    pixels = "".join(f"{value:02X}" for value in (255, 0, 0) * 4)
    inline = b"\nq 200 0 0 150 300 500 cm\nBI /W 2 /H 2 /CS /RGB /BPC 8 /F /AHx ID " + pixels.encode() + b"> EI\nQ\n"
    for page_number in range(len(doc)):
        contents = doc.load_page(page_number).get_contents()[0]
        doc.update_stream(contents, doc.xref_stream(contents) + inline)
    with_inline = doc.tobytes()
    doc.close()

    with pymupdf.open(stream=with_inline, filetype="pdf") as reopened:
        features = document_features(reopened, seed=_SEED)

    assert all(page.image_count == 0 for page in features.pages)
    assert all(page.bitmap_proportion == 0.0 for page in features.pages)


def _with_garbled_font(document_bytes: bytes, text: str) -> bytes:
    """Rewrite every font's ToUnicode CMap to map its glyphs to the replacement character.

    This is what an unextractable text layer looks like in the wild -- a font whose character map
    is missing or broken, so extraction yields U+FFFD -- and it is the condition that routes a
    document to OCR regardless of the model's own confidence.
    """
    cmap_entries = b"".join(b"<%02X> <FFFD>\n" % code for code in sorted({ord(c) for c in text if c != " "}))
    cmap = (
        (
            b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
            b"/CMapName /Garbled def /CMapType 2 def\n"
            b"1 begincodespacerange <00> <FF> endcodespacerange\n"
            b"%d beginbfchar\n" % cmap_entries.count(b"\n")
        )
        + cmap_entries
        + b"endbfchar\nendcmap CMapName currentdict /CMap defineresource pop end end"
    )

    doc = pymupdf.open(stream=document_bytes, filetype="pdf")
    for xref in range(1, doc.xref_length()):
        if doc.xref_get_key(xref, "Type")[1] == "/Font":
            cmap_xref = doc.get_new_xref()
            doc.update_object(cmap_xref, "<<>>")
            doc.update_stream(cmap_xref, cmap)
            doc.xref_set_key(xref, "ToUnicode", f"{cmap_xref} 0 R")
    patched = doc.tobytes()
    doc.close()
    return patched


def test_an_unmappable_text_layer_is_measured_as_garbled():
    text = "unextractable body text"
    clean = pymupdf.open()
    for _ in range(FEATURE_PAGES):
        clean.new_page(width=612, height=792).insert_text((72, 100), text, fontsize=11)
    clean_bytes = clean.tobytes(deflate=True)
    clean.close()

    with pymupdf.open(stream=clean_bytes, filetype="pdf") as doc:
        assert document_features(doc, seed=_SEED).garbled_text_ratio == 0.0

    with pymupdf.open(stream=_with_garbled_font(clean_bytes, text), filetype="pdf") as doc:
        garbled = document_features(doc, seed=_SEED)

    assert garbled.garbled_text_ratio > 0.5


def test_scanner_metadata_is_detected_from_creator_or_producer():
    doc = pymupdf.open()
    doc.new_page()
    doc.set_metadata({"producer": "HP ScanJet Pro 2000 s2"})
    with pymupdf.open(stream=doc.tobytes(), filetype="pdf") as reopened:
        assert document_features(reopened, seed=_SEED).creator_or_producer_is_known_scanner
    doc.close()


def test_an_empty_document_is_rejected():
    doc = pymupdf.open()
    with pytest.raises(CorruptPdf, match="no pages"):
        document_features(doc, seed=_SEED)
    doc.close()


def test_an_encrypted_document_is_rejected():
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "secret")
    encrypted = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    doc.close()

    with pymupdf.open(stream=encrypted, filetype="pdf") as reopened, pytest.raises(CorruptPdf, match="encrypted"):
        document_features(reopened, seed=_SEED)


@pytest.mark.parametrize("num_pages", [1, 5, 8, 9, 500])
def test_sample_page_indices_is_sorted_distinct_and_capped(num_pages):
    sampled = sample_page_indices(num_pages, random.Random(_SEED))

    assert sampled == sorted(sampled)
    assert len(set(sampled)) == len(sampled) == min(FEATURE_PAGES, num_pages)
    assert all(0 <= index < num_pages for index in sampled)


def test_merge_image_strips_recombines_full_width_bands():
    """Scanners slice a page into bands; the bitmap proportion is meaningless until they rejoin."""
    bands = [(0.0, top, 600.0, top + 100.0) for top in (0.0, 100.0, 200.0, 300.0)]

    assert merge_image_strips(bands, page_width=600.0, page_height=400.0) == [(0.0, 0.0, 600.0, 400.0)]


def test_merge_image_strips_leaves_a_gap_wider_than_the_tolerance_alone():
    bands = [(0.0, 0.0, 600.0, 100.0), (0.0, 140.0, 600.0, 240.0)]

    assert merge_image_strips(bands, page_width=600.0, page_height=400.0) == bands


def test_merge_image_strips_does_not_merge_boxes_that_do_not_span_the_page():
    boxes = [(0.0, 0.0, 100.0, 100.0), (0.0, 100.0, 100.0, 200.0)]

    assert merge_image_strips(boxes, page_width=600.0, page_height=400.0) == boxes


def test_merge_image_strips_deduplicates_repeated_boxes():
    box = (10.0, 10.0, 50.0, 50.0)

    assert merge_image_strips([box, box, box], page_width=600.0, page_height=400.0) == [box]
