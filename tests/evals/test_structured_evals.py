# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.structured_evals import structured_evals_raw_validation_sets


def test_structured_evals_raw_validation_sets_cover_higher_value_sources() -> None:
    datasets = structured_evals_raw_validation_sets()

    assert "structured_text/totto" in datasets
    assert "structured_text/wikitablequestions" in datasets
    assert "structured_text/gittables" in datasets
    assert "structured_text/web_data_commons_sample10" in datasets
    assert "structured_text/web_data_commons_sample1k" in datasets
    assert datasets["structured_text/gittables"].text_key == "text"
    assert "issue:5059" in datasets["structured_text/gittables"].tags
