# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.evals.fineweb2_multilingual import fineweb2_multilingual_raw_validation_sets


def test_fineweb2_multilingual_raw_sets_use_eval_split_in_path_only():
    datasets = fineweb2_multilingual_raw_validation_sets(configs=("deu_Latn",))

    dataset = datasets["fineweb2_multilingual/deu_Latn"]

    assert isinstance(dataset.input_path, str)
    assert "/deu_Latn/test/*.parquet" in dataset.input_path
    assert dataset.split == "validation"
    assert dataset.tags == (
        "fineweb2_multilingual",
        "fineweb2_multilingual/script/Latn",
        "fineweb2_multilingual/language/deu",
        "fineweb2_multilingual/top_50_by_rows",
    )
