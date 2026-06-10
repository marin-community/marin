# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.evals.synthetic_identifier_encoding_ppl import (
    IDENTIFIER_ENCODING_HF_DATASET_ID,
    generate_tiny_identifier_encoding_sample,
    identifier_encoding_raw_validation_sets,
)

EXPECTED_IDENTIFIER_KEYS = {
    "synthetic_identifier_encoding_ppl/encoded_text/base64_continuation",
    "synthetic_identifier_encoding_ppl/encoded_text/data_image_base64_prefixes",
    "synthetic_identifier_encoding_ppl/encoded_text/hex_dump_continuation",
    "synthetic_identifier_encoding_ppl/escaped_text/json_unicode_byte_escapes",
    "synthetic_identifier_encoding_ppl/escaped_text/url_query_escaping",
    "synthetic_identifier_encoding_ppl/identifier_grammars/bio_accessions",
    "synthetic_identifier_encoding_ppl/identifier_grammars/commit_hashes",
    "synthetic_identifier_encoding_ppl/identifier_grammars/mixed_case_symbolic_identifiers",
    "synthetic_identifier_encoding_ppl/identifier_grammars/package_names_versions",
    "synthetic_identifier_encoding_ppl/identifier_grammars/uuid_build_ids",
}

EXPECTED_SUBSET_BY_TASK = {
    "base64_continuation": "base64_continuation",
    "bio_accessions": "bio_accessions",
    "commit_hashes": "commit_hashes",
    "data_image_base64_prefixes": "data_image_base64_prefixes",
    "hex_dump_continuation": "hex_dump_continuation",
    "json_unicode_byte_escapes": "json_unicode_byte_escapes",
    "mixed_case_symbolic_identifiers": "mixed_case_symbolic_identifiers",
    "package_names_versions": "package_names_versions",
    "url_query_escaping": "url_query_escaping",
    "uuid_build_ids": "uuid_build_ids",
}


def test_identifier_encoding_registry_uses_supervised_target_only_format() -> None:
    datasets = identifier_encoding_raw_validation_sets()

    assert set(datasets) == EXPECTED_IDENTIFIER_KEYS

    for key, dataset in datasets.items():
        assert key.startswith("synthetic_identifier_encoding_ppl/")
        assert dataset.hf_dataset_id == IDENTIFIER_ENCODING_HF_DATASET_ID
        assert dataset.input_key == "input"
        assert dataset.target_key == "target"
        assert dataset.split == "validation"
        assert "loss:target_only" in dataset.tags


def test_tiny_identifier_encoding_sample_has_expected_schema_and_base_model_prompts() -> None:
    rows = generate_tiny_identifier_encoding_sample()
    instruction_fragments = ("Complete ", "Continue ", "Task:", "User:", "Assistant:")

    assert len(rows) == 10
    for row in rows:
        assert set(row) == {"input", "target", "id", "subset", "task", "seed", "metadata"}
        assert row["input"]
        assert row["target"].endswith("\n")
        assert row["subset"] == EXPECTED_SUBSET_BY_TASK[row["task"]]
        assert row["metadata"]["family"] in {"identifier_grammars", "encoded_text", "escaped_text"}
        assert not any(fragment in row["input"] for fragment in instruction_fragments)


def test_identifier_examples_are_raw_field_continuations():
    rows_by_task = {row["task"]: row for row in generate_tiny_identifier_encoding_sample()}

    package = rows_by_task["package_names_versions"]
    assert str(package["input"]).startswith('"node_modules/@')
    assert "registry.npmjs.org" in str(package["input"])
    assert ".tgz" in str(package["target"])

    base64_row = rows_by_task["base64_continuation"]
    assert str(base64_row["input"]).startswith('payload_b64="')
    assert str(base64_row["target"]).endswith('"\n')

    hexdump = rows_by_task["hex_dump_continuation"]
    assert str(hexdump["input"])[8:10] == "  "
    assert "|" in str(hexdump["target"])

    url_row = rows_by_task["url_query_escaping"]
    assert str(url_row["input"]).startswith("https://example.invalid/search?q=")
    assert str(url_row["target"]).endswith("&src=eval\n")
