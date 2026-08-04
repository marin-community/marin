# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for detecting and cutting out a vision model's repetition loops.

The fixtures are built to hold the distinction the detector exists for: a page that is *genuinely*
repetitive (a results table whose cells happen to repeat, a register whose blocks differ only in
their reference numbers) must survive untouched, while a page where the model emitted one invariant
unit until the token cap stopped it must be cut back to the transcription in front of it.
"""

from concurrent.futures import Future

import pytest

from experiments.build_pdf_source.boilerplate import BoilerplateOptions
from experiments.build_pdf_source.extract_ocr import OcrStatus, _Document
from experiments.build_pdf_source.loop_repair import LoopOptions, find_loop, repair_page
from experiments.build_pdf_source.ocr_extract.client import PageOcr

_OPTIONS = LoopOptions()
_BOILERPLATE = BoilerplateOptions()

_PROSE = (
    "# Groundwater nitrate pollution in the Vistrenque aquifer\n\n"
    "The Vistrenque aquifer supplies drinking water to roughly forty thousand people in the lower "
    "Rhone valley, and its nitrate concentration has risen steadily since the nineteen seventies. "
    "This section reports the sampling campaign carried out between March and October, covering "
    "sixty-one wells distributed across the recharge zone and the confined section further south. "
    "Concentrations are reported in milligrams per litre and compared against the regulatory "
    "threshold of fifty milligrams per litre.\n\n"
)


def _looping_page(unit: str, prefix: str = _PROSE, repeats: int = 400) -> str:
    """A page whose transcription degenerates into ``unit`` repeated to the end of the output."""
    return prefix + unit * repeats


def _table_page(rows: int = 260) -> str:
    """A faithfully transcribed results table: repetitive in form, varying in content."""
    lines = ["| dataset | method | resolution | error |", "| --- | --- | --- | --- |"]
    for index in range(rows):
        # Vary the cells the way a real benchmark table does. Most resolutions really are 1.000,
        # which is what makes this page look degenerate to a redundancy-based detector.
        lines.append(f"| 1000M{index % 4 + 1} | PASTA-{index % 7} | 1.000 | 0.{index % 97:03d} |")
    return "# Table S3: resolution by dataset\n\n" + "\n".join(lines) + "\n"


def _register_page(entries: int = 150) -> str:
    """A parliamentary-style register: identical block structure, unrelated reference numbers.

    Folding digits makes this exactly periodic, so only the counter guard keeps it out of the
    detector's jaws. The numbers must be unrelated -- neither constant nor an arithmetic run.
    """
    references = [(index * 7919 + 104729) % 899999 + 100000 for index in range(entries)]
    blocks = []
    for index, reference in enumerate(references):
        blocks.append(
            f"Expedient {reference}\nPresentat el {reference % 28 + 1}/{index % 12 + 1}/2026\n"
            f"Grup Parlamentari numero {reference % 9 + 1}\nEstat: admes a tramit\n"
        )
    return "# Registre de documents\n\n" + "\n".join(blocks)


def _repair(text: str, truncated: bool = True):
    return repair_page(text, truncated, _OPTIONS)


def test_a_repeated_unit_running_to_the_cap_is_cut_back_to_the_transcription():
    page = _looping_page("|  |  |  |\n")
    repair = _repair(page)

    assert repair.looped
    assert repair.text.startswith("# Groundwater nitrate pollution")
    assert "threshold of fifty milligrams per litre." in repair.text
    # Essentially all of the degeneracy is gone; what survives is the transcription in front.
    assert repair.dropped_chars > 0.95 * (len(page) - len(_PROSE))
    assert repair.dropped_chars == len(page) - len(repair.text)


def test_the_cut_can_leave_up_to_one_period_of_the_loop_behind():
    """A known, measured limitation, pinned so a change to the period search has to face it.

    The period comes from an ``rfind`` that requires the match to clear the probe, so it is the true
    unit rounded up to a multiple of itself; the backward walk strides in that multiple and can stop
    a repetition or two short. Shortening the stride was tried and is worse -- it walks back through
    real content -- so the residue is accepted and bounded here.
    """
    unit = "|  |  |  |\n"
    repair = _repair(_looping_page(unit))
    loop = find_loop(_looping_page(unit), _OPTIONS)

    assert repair.text.count(unit) <= loop.period // len(unit)


def test_a_faithfully_transcribed_repetitive_table_is_left_alone():
    page = _table_page()
    repair = _repair(page)

    assert not repair.looped
    assert repair.text == page
    assert repair.dropped_chars == 0


def test_a_register_whose_blocks_differ_only_in_reference_numbers_is_left_alone():
    page = _register_page()
    # The block structure really is periodic once digits are folded; the guard is what saves it.
    assert not find_loop(page, _OPTIONS).exact
    assert not _repair(page).looped


def test_an_incrementing_counter_is_detected_as_a_loop():
    page = _PROSE + "".join(f"[Fig. {number}] See caption above for details.\n" for number in range(400))
    repair = _repair(page)

    assert repair.looped
    assert "[Fig. 300]" not in repair.text


def test_a_page_the_model_finished_is_never_examined():
    """The cap gate: a runaway loop is stopped by ``max_tokens``, so an untruncated page is trusted."""
    page = _looping_page("|  |  |  |\n")

    assert not _repair(page, truncated=False).looped
    assert _repair(page, truncated=False).text == page


def test_a_short_page_is_not_repaired():
    page = _looping_page("...\n", prefix="Contents\n\n", repeats=40)

    assert len(page) < _OPTIONS.min_page_chars
    assert not _repair(page).looped


def test_a_repetitive_block_the_model_transcribed_past_is_not_a_loop():
    """A cycle it exited is a table it read; only degeneracy that runs to the end counts."""
    page = _looping_page("| --- | --- |\n", repeats=200) + _PROSE * 4

    assert not _repair(page).looped


def test_a_page_that_looped_from_the_start_is_emptied():
    page = _looping_page("..... ..... .....\n", prefix="| |\n")
    repair = _repair(page)

    assert repair.looped
    assert repair.text == ""
    assert repair.dropped_chars == len(page)


def test_salvage_cuts_at_a_line_boundary():
    page = _looping_page("___ ___ ___\n")
    repair = _repair(page)

    assert repair.text
    assert not repair.text.endswith("_")
    assert page.startswith(repair.text)


def test_salvage_drops_what_the_model_emitted_after_the_loop():
    """Text on the far side of a several-thousand-character cycle has no known place on the page."""
    page = _looping_page("|  |\n", repeats=900) + "Concluding remarks on the sampling campaign.\n"
    repair = _repair(page)

    assert repair.looped
    assert "Concluding remarks" not in repair.text


def test_thresholds_are_honoured_from_the_options():
    page = _looping_page("|  |  |  |\n")
    permissive = LoopOptions(min_salvage_prefix=1)
    strict = LoopOptions(min_loop_fraction=0.99)

    assert repair_page(page, True, permissive).looped
    assert not repair_page(page, True, strict).looped


@pytest.mark.parametrize("unit", ["|  |  |  |\n", "..... ..... .....\n", "\\quad \\quad \\quad\n", "```markdown\n"])
def test_known_loop_shapes_are_detected(unit):
    """The unit classes hand-labeling found: empty cells, leader dots, spacing macros, fences."""
    assert _repair(_looping_page(unit)).looped


def _document(pages: list[PageOcr], declared: int | None = None) -> _Document:
    """Feed pages through the real absorb path, as the sender does."""
    row = {
        "warc_filename": "crawl/warc/a.warc.gz",
        "warc_record_offset": 42,
        "content_digest": "sha1:AAAA",
        "url": "https://example.org/report.pdf",
    }
    document = _Document(row=row, declared_pages=declared if declared is not None else len(pages))
    for page in pages:
        future: Future[PageOcr] = Future()
        future.set_result(page)
        document.submitted += 1
        document.dpis.append(150.0)
        document.absorb(future, _OPTIONS)
    document.closed = True
    return document


def _record(pages: list[PageOcr]) -> dict:
    """The output row for ``pages``, which the caller expects the document to produce."""
    record = _document(pages).record(_BOILERPLATE, floor_dpi=72.0)
    assert record is not None
    return record


def test_a_looped_page_makes_the_document_partial_and_is_reported_by_page_number():
    clean = PageOcr(text=_PROSE * 3, completion_tokens=900)
    looped = PageOcr(text=_looping_page("|  |  |  |\n"), completion_tokens=4096, truncated=True)
    record = _record([clean, looped, clean])

    assert record["extraction_status"] == str(OcrStatus.PARTIAL)
    assert record["looped_pages"] == [2]
    assert record["loop_chars_dropped"] > 0.95 * (len(looped.text) - len(_PROSE))
    assert "repeated themselves" in record["extraction_error"]


def test_a_clean_document_reports_no_loops_and_stays_successful():
    pages = [PageOcr(text=_PROSE * 3, completion_tokens=900) for _ in range(3)]
    record = _record(pages)

    assert record["extraction_status"] == str(OcrStatus.SUCCESS)
    assert record["looped_pages"] == []
    assert record["loop_chars_dropped"] == 0


def test_a_faithful_table_document_survives_the_pipeline_unrepaired():
    """The regression that matters: a truncated table page keeps its text and its SUCCESS-shaped record."""
    pages = [PageOcr(text=_table_page(), completion_tokens=4096, truncated=True)]
    record = _record(pages)

    assert record["looped_pages"] == []
    assert record["loop_chars_dropped"] == 0
    assert "1000M1" in record["text"]


def test_page_offsets_follow_the_repaired_text():
    clean = PageOcr(text=_PROSE * 3, completion_tokens=900)
    looped = PageOcr(text=_looping_page("|  |  |  |\n"), completion_tokens=4096, truncated=True)
    record = _record([clean, looped])

    assert record["page_offsets"][-1] == len(record["text"])
    assert record["page_offsets"][0] < record["page_offsets"][1]


def test_a_document_whose_only_page_was_emptied_is_dropped():
    looped = PageOcr(text=_looping_page("..... .....\n", prefix="| |\n"), completion_tokens=4096, truncated=True)

    assert _document([looped]).record(_BOILERPLATE, floor_dpi=72.0) is None
