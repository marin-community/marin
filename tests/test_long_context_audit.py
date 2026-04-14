# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.long_context_datasets.audit import (
    AuditSource,
    ExactSourceStats,
    ExtractedDocument,
    compute_text_quality,
    extract_document_fields,
    summarize_source,
)

LONGMINO_SOURCE = AuditSource(
    key="longmino/8k-16k",
    raw_path="dolma3_longmino_pool/data/*-2e13/*.jsonl.zst",
    format="jsonl",
    text_paths=(("text",),),
    id_paths=(("id",),),
    language_paths=(("metadata", "language"), ("language",)),
)

FINEPDFS_SOURCE = AuditSource(
    key="finepdfs-edu/eng_Latn",
    raw_path="finepdfs_edu_eng_Latn/data/eng_Latn/train/*.parquet",
    format="parquet",
    text_paths=(("text",),),
    id_paths=(("doc_id",), ("id",)),
    language_paths=(("language",),),
    token_count_paths=(("token_count",),),
)


def test_extract_document_fields_from_longmino_record():
    record = {
        "id": "doc-17",
        "text": "This is a long document.",
        "metadata": {"language": "en"},
    }

    document = extract_document_fields(record, LONGMINO_SOURCE)

    assert document == ExtractedDocument(
        doc_id="doc-17",
        text="This is a long document.",
        token_count=None,
        language="en",
    )


def test_extract_document_fields_prefers_explicit_token_count():
    record = {
        "doc_id": "pdf-3",
        "text": "Educational PDF text.",
        "token_count": 1536,
        "language": "eng_Latn",
    }

    document = extract_document_fields(record, FINEPDFS_SOURCE)

    assert document == ExtractedDocument(
        doc_id="pdf-3",
        text="Educational PDF text.",
        token_count=1536,
        language="eng_Latn",
    )


def test_compute_text_quality_flags_repetition_and_ocr_noise():
    noisy_text = "\n".join(
        [
            "COURSE READER 2025",
            "COURSE READER 2025",
            "COURSE READER 2025",
            "1",
            "intr0ducti0n t0 mach1ne learn1ng ~~",
            "th1s l1ne has extract10n n01se ||",
            "broken line with no ending",
            "another broken line with no ending",
        ]
    )

    quality = compute_text_quality(noisy_text)

    assert quality.repeat_line_ratio > 0.2
    assert quality.boilerplate_line_ratio > 0.2
    assert quality.broken_line_ratio > 0.2
    assert quality.non_alnum_ratio > 0
    assert "repetition" in quality.flags
    assert "ocr_noise" in quality.flags


def test_summarize_source_uses_exact_stats_when_available():
    documents = [
        ExtractedDocument(doc_id="a", text="alpha beta", token_count=2, language="en"),
        ExtractedDocument(doc_id="b", text="alpha beta gamma delta", token_count=4, language="en"),
        ExtractedDocument(doc_id="c", text="one two three four five six", token_count=6, language="fr"),
    ]

    summary = summarize_source(
        source=FINEPDFS_SOURCE,
        documents=documents,
        exact_stats=ExactSourceStats(total_documents=100, total_tokens=10_000, stats_source="tokenized_cache"),
    )

    assert summary.source_key == "finepdfs-edu/eng_Latn"
    assert summary.sample_documents == 3
    assert summary.total_documents == 100
    assert summary.total_tokens == 10_000
    assert summary.total_documents_source == "tokenized_cache"
    assert summary.total_tokens_source == "tokenized_cache"
    assert summary.token_count_p50 == 4
    assert summary.language_counts == {"en": 2, "fr": 1}
