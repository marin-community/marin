# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.evals.synthetic_identifier_encoding_ppl import (
    IDENTIFIER_ENCODING_HF_DATASET_ID,
    IDENTIFIER_ENCODING_PPL_SLICES,
    generate_tiny_identifier_encoding_sample,
    identifier_encoding_raw_validation_sets,
)


def test_identifier_encoding_registry_uses_supervised_target_only_format() -> None:
    datasets = identifier_encoding_raw_validation_sets()

    assert set(datasets) == {
        "synthetic_identifier_encoding_ppl/identifier_grammars/package_names_versions",
        "synthetic_identifier_encoding_ppl/identifier_grammars/commit_hashes",
        "synthetic_identifier_encoding_ppl/identifier_grammars/uuid_build_ids",
        "synthetic_identifier_encoding_ppl/identifier_grammars/bio_accessions",
        "synthetic_identifier_encoding_ppl/identifier_grammars/mixed_case_symbolic_identifiers",
        "synthetic_identifier_encoding_ppl/encoded_text/base64_continuation",
        "synthetic_identifier_encoding_ppl/encoded_text/data_image_base64_prefixes",
        "synthetic_identifier_encoding_ppl/encoded_text/hex_dump_continuation",
        "synthetic_identifier_encoding_ppl/escaped_text/url_query_escaping",
        "synthetic_identifier_encoding_ppl/escaped_text/json_unicode_byte_escapes",
    }

    for key, dataset in datasets.items():
        assert key.startswith("synthetic_identifier_encoding_ppl/")
        assert dataset.hf_dataset_id == IDENTIFIER_ENCODING_HF_DATASET_ID
        assert dataset.input_key == "input"
        assert dataset.target_key == "target"
        assert dataset.split == "validation"
        assert "loss:target_only" in dataset.tags


def test_identifier_encoding_hf_config_names_match_subsets() -> None:
    assert [slice_.hf_config_name for slice_ in IDENTIFIER_ENCODING_PPL_SLICES] == [
        "package_names_versions",
        "commit_hashes",
        "uuid_build_ids",
        "bio_accessions",
        "mixed_case_symbolic_identifiers",
        "base64_continuation",
        "data_image_base64_prefixes",
        "hex_dump_continuation",
        "url_query_escaping",
        "json_unicode_byte_escapes",
    ]


def test_tiny_identifier_encoding_sample_has_expected_schema_and_base_model_prompts() -> None:
    rows = generate_tiny_identifier_encoding_sample()
    slice_by_task = {slice_.task_name: slice_ for slice_ in IDENTIFIER_ENCODING_PPL_SLICES}

    assert len(rows) == len(IDENTIFIER_ENCODING_PPL_SLICES)
    for row in rows:
        assert set(row) == {"input", "target", "id", "subset", "task", "seed", "metadata"}
        assert row["input"]
        assert row["target"].endswith("\n")
        assert row["subset"] == slice_by_task[row["task"]].hf_config_name
        assert row["metadata"]["family"] in {"identifier_grammars", "encoded_text", "escaped_text"}
        assert "User:" not in row["input"]
        assert "Assistant:" not in row["input"]
