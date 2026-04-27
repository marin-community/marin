# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.evals.paired_robustness_ppl import (
    DEFAULT_SAMPLE_CAP,
    PairedRobustnessFamily,
    PairedTextView,
    linearized_text_views_for_example,
    paired_robustness_raw_steps,
    paired_robustness_raw_validation_sets,
    paired_robustness_slices,
)
from marin.execution.executor import InputName


def test_paired_robustness_slices_are_held_out_and_capped():
    paraphrase_slices = paired_robustness_slices(family=PairedRobustnessFamily.PARAPHRASE)
    translation_slices = paired_robustness_slices(family=PairedRobustnessFamily.TRANSLATION)

    assert {slice_.split for slice_ in paraphrase_slices} == {"validation", "test"}
    assert {slice_.split for slice_ in translation_slices} == {"dev", "devtest"}
    assert all(slice_.max_pairs <= DEFAULT_SAMPLE_CAP for slice_ in (*paraphrase_slices, *translation_slices))


def test_paws_linearization_uses_stable_labels_and_filters_negative_pairs():
    paws_validation = next(
        slice_
        for slice_ in paired_robustness_slices(family=PairedRobustnessFamily.PARAPHRASE)
        if slice_.split == "validation"
    )

    positive_example = {
        "id": 7,
        "label": 1,
        "sentence1": "The cat sat on the mat.",
        "sentence2": "The cat was sitting on the mat.",
    }
    negative_example = {
        "id": 8,
        "label": 0,
        "sentence1": "A short sentence.",
        "sentence2": "A different meaning.",
    }

    views = linearized_text_views_for_example(paws_validation, positive_example)

    assert views is not None
    assert views[PairedTextView.SOURCE] == "sentence_1: The cat sat on the mat."
    assert views[PairedTextView.TARGET] == "sentence_2: The cat was sitting on the mat."
    assert views[PairedTextView.TARGET_GIVEN_SOURCE] == (
        "sentence_1: The cat sat on the mat.\nsentence_2: The cat was sitting on the mat."
    )
    assert linearized_text_views_for_example(paws_validation, negative_example) is None


def test_paired_raw_validation_sets_register_conditional_view_paths_and_tags():
    slices = tuple(
        slice_
        for slice_ in paired_robustness_slices()
        if (slice_.family == PairedRobustnessFamily.PARAPHRASE and slice_.split == "validation")
        or (slice_.family == PairedRobustnessFamily.TRANSLATION and slice_.split == "devtest")
    )
    raw_steps = paired_robustness_raw_steps(slices=slices)
    datasets = paired_robustness_raw_validation_sets(slices=slices, raw_steps=raw_steps)

    paws_key = "paired_robustness_ppl/paraphrase/paws_labeled_final/validation/target_given_source"
    flores_key = "paired_robustness_ppl/translation/flores_eng_deu/devtest/source"

    assert isinstance(datasets[paws_key].input_path, InputName)
    assert datasets[paws_key].input_path.name == "target_given_source/shard-*.jsonl.gz"
    assert "split:validation" in datasets[paws_key].tags
    assert "view:target_given_source" in datasets[paws_key].tags

    assert isinstance(datasets[flores_key].input_path, InputName)
    assert datasets[flores_key].input_path.name == "source/shard-*.jsonl.gz"
    assert "split:devtest" in datasets[flores_key].tags
    assert "family:translation" in datasets[flores_key].tags
