# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.datakit.scripts.dedup_ab_summarize import (
    _fraction_bin,
    _length_bin,
    summarize_comparisons,
    summarize_scores,
)


def _score(variant: str, role: str, doc_id: str, **overrides) -> dict:
    return {
        "variant": variant,
        "role": role,
        "evidence_class": "canonical" if role == "canonical" else "ambiguous",
        "source_main_dir": f"s3://normalized/{variant}",
        "canonical_source_main_dir": f"s3://normalized/{variant}",
        "cluster_id": f"{variant}-cluster",
        "id": doc_id,
        "cross_source": False,
        "raw_chars": 100,
        "canonical_raw_chars": 200,
        "member_is_longer": False,
        "member_text_truncated_for_minhash": False,
        "canonical_text_truncated_for_minhash": False,
        "exact_raw_text": False,
        "exact_clean_text": False,
        "char_5gram_jaccard": 0.5,
        "word_5gram_jaccard": 0.75,
        "word_5gram_canonical_containment": 0.8,
        "word_5gram_member_containment": 0.9,
        "length_ratio": 0.5,
        "baseline_shared_buckets": 1,
        "treatment_shared_buckets": 2,
        **overrides,
    }


def test_fraction_and_length_bins_have_stable_boundaries() -> None:
    assert _fraction_bin(0.0) == "0.00-0.05"
    assert _fraction_bin(0.05) == "0.05-0.10"
    assert _fraction_bin(1.0) == "0.95-1.00"
    assert _length_bin(0) == "0"
    assert _length_bin(1) == "2^0-2^1"
    assert _length_bin(1024) == "2^10-2^11"


def test_score_summary_counts_all_rows_clusters_and_drop_properties() -> None:
    records = [
        _score("baseline", "canonical", "b-canonical"),
        _score("baseline", "drop", "b-drop", cross_source=True, member_text_truncated_for_minhash=True),
        _score("treatment", "canonical", "t-canonical"),
        _score("treatment", "drop", "t-drop", exact_raw_text=True, evidence_class="strong_duplicate"),
    ]

    result = summarize_scores(records)

    assert result["score_rows"] == 4
    assert result["variants"]["baseline"]["roles"] == {"canonical": 1, "drop": 1}
    assert result["variants"]["baseline"]["cross_source"] == {"True": 1}
    assert result["variants"]["baseline"]["either_text_truncated_for_minhash"] == {"True": 1}
    assert result["variants"]["baseline"]["clusters"] == 1
    assert result["variants"]["baseline"]["cluster_size_histogram"] == {"2": 1}
    assert result["variants"]["treatment"]["evidence"] == {"strong_duplicate": 1}
    assert result["variants"]["treatment"]["word_5gram_jaccard_bins"] == {"0.75-0.80": 1}


def test_comparison_summary_counts_every_category_and_distance() -> None:
    records = [
        {
            "category": "baseline_drop_treatment_keep",
            "baseline_only_attribution": "word_ngram",
            "baseline_graph_distance": 1,
            "baseline_evidence_class": "ambiguous",
            "treatment_evidence_class": "missing",
        },
        {
            "category": "both_drop",
            "baseline_only_attribution": "not_applicable",
            "baseline_graph_distance": 2,
            "baseline_evidence_class": "strong_duplicate",
            "treatment_evidence_class": "strong_duplicate",
        },
    ]

    result = summarize_comparisons(records)

    assert result["comparison_rows"] == 2
    assert result["categories"] == {"baseline_drop_treatment_keep": 1, "both_drop": 1}
    assert result["baseline_graph_distances"] == {"1": 1, "2": 1}
