# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import tarfile
from pathlib import Path

from marin.execution.executor import InputName

from experiments.evals.paired_robustness_ppl import (
    ALL_PAIRED_TEXT_VIEWS,
    DEFAULT_SAMPLE_CAP,
    FLORES200_ARCHIVE_URL,
    PairedRobustnessFamily,
    PairedRobustnessMaterializeConfig,
    PairedRobustnessSourceType,
    PairedTextView,
    _flores200_archive_examples,
    linearized_text_views_for_example,
    paired_robustness_raw_steps,
    paired_robustness_raw_validation_sets,
    paired_robustness_slices,
)


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
    flores_slices = [slice_ for slice_ in slices if slice_.family == PairedRobustnessFamily.TRANSLATION]
    assert all(slice_.source_type == PairedRobustnessSourceType.FLORES200_ARCHIVE for slice_ in flores_slices)
    assert all(slice_.data_url == FLORES200_ARCHIVE_URL for slice_ in flores_slices)


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


def test_flores200_archive_examples_load_requested_language_pair(tmp_path: Path):
    archive_path = tmp_path / "flores200_dataset.tar.gz"
    _write_tar_text(
        archive_path,
        {
            "flores200_dataset/dev/eng_Latn.dev": "Hello.\nGoodbye.\n",
            "flores200_dataset/dev/deu_Latn.dev": "Hallo.\nAuf Wiedersehen.\n",
            "flores200_dataset/metadata_dev.tsv": (
                "URL\tdomain\ttopic\thas_image\thas_hyperlink\n"
                "https://example.test/1\tweb\tgreeting\tyes\tno\n"
                "https://example.test/2\tweb\tparting\tno\tyes\n"
            ),
        },
    )
    config = PairedRobustnessMaterializeConfig(
        name="flores_eng_deu",
        family=PairedRobustnessFamily.TRANSLATION,
        source_url="https://huggingface.co/datasets/facebook/flores",
        hf_dataset_id="facebook/flores",
        hf_dataset_name="eng_Latn-deu_Latn",
        split="dev",
        source_field="sentence_eng_Latn",
        target_field="sentence_deu_Latn",
        source_label="English",
        target_label="German",
        max_pairs=512,
        source_type=PairedRobustnessSourceType.FLORES200_ARCHIVE,
        data_url=archive_path.as_uri(),
    )

    examples = list(_flores200_archive_examples(config))

    assert examples == [
        {
            "id": 1,
            "URL": "https://example.test/1",
            "domain": "web",
            "topic": "greeting",
            "has_image": 1,
            "has_hyperlink": 0,
            "sentence_eng_Latn": "Hello.",
            "sentence_deu_Latn": "Hallo.",
        },
        {
            "id": 2,
            "URL": "https://example.test/2",
            "domain": "web",
            "topic": "parting",
            "has_image": 0,
            "has_hyperlink": 1,
            "sentence_eng_Latn": "Goodbye.",
            "sentence_deu_Latn": "Auf Wiedersehen.",
        },
    ]


def _write_tar_text(path: Path, members: dict[str, str]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, text in members.items():
            payload = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
