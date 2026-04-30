# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.evals.paired_robustness_ppl import (
    DEFAULT_SAMPLE_CAP,
    ALL_PAIRED_TEXT_VIEWS,
    PairedRobustnessFamily,
    PairedTextView,
    linearized_text_views_for_example,
    paired_robustness_raw_steps,
    paired_robustness_raw_validation_sets,
    paired_robustness_slices,
)
from marin.execution.executor import InputName


def test_paired_robustness_slices_define_expected_held_out_inventory():
    slices = paired_robustness_slices()

    assert {(slice_.family, slice_.name, slice_.split) for slice_ in slices} == {
        (PairedRobustnessFamily.PARAPHRASE, "paws_labeled_final", "validation"),
        (PairedRobustnessFamily.PARAPHRASE, "paws_labeled_final", "test"),
        (PairedRobustnessFamily.TRANSLATION, "flores_eng_deu", "dev"),
        (PairedRobustnessFamily.TRANSLATION, "flores_eng_deu", "devtest"),
    }
    assert all(slice_.max_pairs <= DEFAULT_SAMPLE_CAP for slice_ in slices)
    assert all(
        slice_.tags_for_view(PairedTextView.TARGET_GIVEN_SOURCE)[0] == "paired_robustness_ppl" for slice_ in slices
    )


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


def test_paired_raw_validation_sets_register_every_view_for_selected_slices():
    slices = tuple(
        slice_
        for slice_ in paired_robustness_slices()
        if (slice_.family == PairedRobustnessFamily.PARAPHRASE and slice_.split == "validation")
        or (slice_.family == PairedRobustnessFamily.TRANSLATION and slice_.split == "devtest")
    )
    raw_steps = paired_robustness_raw_steps(slices=slices)
    datasets = paired_robustness_raw_validation_sets(slices=slices, raw_steps=raw_steps)

    assert set(datasets) == {slice_.dataset_key(view) for slice_ in slices for view in ALL_PAIRED_TEXT_VIEWS}

    for slice_ in slices:
        for view in ALL_PAIRED_TEXT_VIEWS:
            dataset = datasets[slice_.dataset_key(view)]
            assert isinstance(dataset.input_path, InputName)
            assert dataset.input_path.name == f"{view.value}/shard-*.jsonl.gz"
            assert set(slice_.tags_for_view(view)).issubset(dataset.tags)
