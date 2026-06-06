# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.evals.long_context_ppl import (
    booksum_supervised_record,
    qasper_supervised_record,
    render_booksum_summary_prompt,
    render_qasper_answer_prompt,
)
from experiments.evals.perplexity_gap_registry import (
    LONG_CONTEXT_32K_EVAL_LENGTH,
    LONG_CONTEXT_64K_EVAL_LENGTH,
    LONG_CONTEXT_MAX_DOC_BYTES,
    LONG_CONTEXT_MAX_DOCS_PER_DATASET,
    long_context_64k_bundle,
    long_context_bundle,
)


def test_qasper_renderer_moves_question_next_to_answer() -> None:
    rendered = render_qasper_answer_prompt(
        "Q: what is the source of the news sentences?\nText: "
        "Introduction\nNamed entity recognition is important.\nConclusion sentence."
    )

    assert rendered == (
        "Text:\nIntroduction\nNamed entity recognition is important.\nConclusion sentence.\n\n"
        "Question:\nwhat is the source of the news sentences?\n\n"
        "Answer:\n"
    )


def test_qasper_renderer_rejects_unparseable_source_rows() -> None:
    with pytest.raises(ValueError, match="must start with 'Q: '"):
        render_qasper_answer_prompt("Question: what is this?\nText: paper body")

    with pytest.raises(ValueError, match="contain a '\\\\nText: ' marker"):
        render_qasper_answer_prompt("Q: what is this?\nPassage: paper body")


def test_qasper_staged_record_scores_answer_after_answer_prompt() -> None:
    record = qasper_supervised_record(
        {
            "id": "paper-id",
            "pid": "paper-id_0",
            "input": "Q: what is the source of the news sentences?\nText: Paper body final sentence.",
            "output": "ilur.am",
        },
        row_index=7,
    )

    assert record is not None
    text = record["input"] + record["target"]
    assert text.endswith("Question:\nwhat is the source of the news sentences?\n\nAnswer:\nilur.am")
    assert "Paper body final sentence.ilur.am" not in text
    assert record["provenance"]["render"] == "text_question_answer_target_only"


def test_qasper_staged_record_skips_empty_targets() -> None:
    record = qasper_supervised_record(
        {
            "input": "Q: what is the source?\nText: Paper body final sentence.",
            "output": "   ",
        },
        row_index=4,
    )

    assert record is None


def test_booksum_renderer_scores_summary_after_summary_prompt() -> None:
    row = {
        "book_id": "Bleak House.chapters 1-4",
        "summary_id": "chapters 1-4",
        "chapter": "London fog fills the street.",
        "summary_text": "The chapter opens in London during foggy weather.",
        "source": "gradesaver",
    }

    rendered = render_booksum_summary_prompt(row, row_index=3)
    record = booksum_supervised_record(row, row_index=3, split="test")

    assert rendered == ("Chapter:\nLondon fog fills the street.\n\n" "Summary:\n")
    assert not rendered.startswith("Book:")
    assert "Section: chapters 1-4" not in rendered
    assert record is not None
    combined = record["input"] + record["target"]
    assert combined.endswith("Summary:\nThe chapter opens in London during foggy weather.")
    assert "London fog fills the street.The chapter opens" not in combined
    assert record["provenance"]["split"] == "test"
    assert record["provenance"]["render"] == "chapter_summary_target_only"


def test_booksum_staged_record_skips_empty_targets_and_rejects_bad_splits() -> None:
    row = {
        "chapter": "London fog fills the street.",
        "summary_text": "   ",
    }

    assert booksum_supervised_record(row, row_index=8, split="validation") is None

    with pytest.raises(ValueError, match="Unsupported BookSum split"):
        booksum_supervised_record(row, row_index=8, split="train")


def test_long_context_bundles_use_large_doc_budget_and_limited_doc_count() -> None:
    bundle = long_context_bundle()
    opt_in_bundle = long_context_64k_bundle()

    assert bundle.max_eval_length == LONG_CONTEXT_32K_EVAL_LENGTH == 32_768
    assert bundle.max_docs_per_dataset == LONG_CONTEXT_MAX_DOCS_PER_DATASET
    assert bundle.max_doc_bytes == LONG_CONTEXT_MAX_DOC_BYTES

    assert opt_in_bundle.max_eval_length == LONG_CONTEXT_64K_EVAL_LENGTH == 65_536
    assert opt_in_bundle.max_docs_per_dataset == LONG_CONTEXT_MAX_DOCS_PER_DATASET
    assert opt_in_bundle.max_doc_bytes == LONG_CONTEXT_MAX_DOC_BYTES
