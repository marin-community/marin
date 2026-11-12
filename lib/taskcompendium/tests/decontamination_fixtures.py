# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private release receipts used by source-admission tests."""

from finestore.rl.curation import LexicalContaminationEvidence


def decontamination_evidence(dataset: str, revision: str, count: int) -> LexicalContaminationEvidence:
    return LexicalContaminationEvidence(
        version=1,
        input_dataset_id=dataset,
        input_dataset_revision=revision,
        converter_revision="converter-pin",
        text_renderer_version="public-task-v1",
        measured_tasks=count,
        contaminated_tasks=0,
        contaminated_fraction=0,
        max_overlap=0,
        mean_max_overlap=0,
        p95_max_overlap=0,
        matched_evals={},
        eval_corpus_version="eval-v1",
        eval_bloom_uri="s3://test/evals.bloom",
        eval_hash_index_uri="s3://test/evals.index",
        feature_filter_version=1,
        ngram_length=13,
        overlap_threshold=0.5,
        min_matched_features=2,
        drop_set_uris=[],
        task_flags_uri="s3://test/task-flags.jsonl",
        evidence_url="s3://test/lexical.json",
        receipt={
            "private": True,
            "format": "taskcompendium",
            "status": "clean",
            "dataset_id": dataset,
            "dataset_revision": revision,
            "retained_tasks": count,
            "dropped_tasks": 0,
            "drop_manifest_uri": None,
            "original_private_dataset_id": None,
            "original_private_dataset_revision": None,
        },
    )
