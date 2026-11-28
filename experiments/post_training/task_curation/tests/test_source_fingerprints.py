# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seeded fingerprint determinism, candidate ranking, and duplicate-group folding."""

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.fingerprints import semantic_digest
from taskcompendium.source_fingerprints import (
    DirectionalOverlap,
    DuplicateLabel,
    DuplicateReview,
    MeasuredOverlap,
    PairVerdict,
)

from experiments.post_training.task_curation.source_fingerprints import (
    candidate_pairs,
    fingerprint_source,
    pair_similarity,
)
from infra.marina.apps.rl_data_catalog.server.duplicate_folding import SourceStanding, fold_groups


def document(task_id: str, dataset: str, instruction: str, *, grader: dict | None = None) -> dict:
    grader = grader or {"kind": "none", "reason": "fixture"}
    return {
        "id": task_id,
        "context": {"events": ({"type": "message", "role": "user", "content": instruction},)},
        "environment_requirements": {},
        "answer_type": "text",
        "answer_format": {"kind": "plain_text"},
        "grader": grader,
        "source": {"dataset": dataset, "revision": "1", "row": task_id, "importer_revision": "1"},
        "labels": {"subset": "train"} if task_id.endswith("-labelled") else {},
    }


def fingerprint(documents, inputs):
    return fingerprint_source(iter(documents), inputs)


def test_fingerprint_sample_and_counts_are_seed_deterministic():
    documents = [document(f"task-{index}", "fixture/dataset", f"Solve problem number {index}.") for index in range(50)]
    inputs = {"s3://fixture/population/specs/part-000000.parquet": "a" * 64}
    first = fingerprint(documents, inputs)
    second = fingerprint(list(reversed(documents)), inputs)
    assert first.task_count == second.task_count == 50
    assert first.within_source_duplicates == second.within_source_duplicates == 0
    assert first.sample_sha256 == second.sample_sha256
    assert [sample.task_id for sample in first.sample] == [sample.task_id for sample in second.sample]
    assert first.instruction_minhash == second.instruction_minhash
    assert first.instruction_hashes == second.instruction_hashes


def test_fingerprint_counts_identical_contracts_once():
    documents = [document("task-duplicate", "fixture/dataset", "Solve the same problem.") for _ in range(3)]
    counts = fingerprint(documents, {"s3://fixture/population/specs/part-000000.parquet": "b" * 64})
    assert counts.task_count == 3
    assert counts.within_source_duplicates == 2


def test_reporting_subset_label_does_not_change_the_semantic_digest():
    """A task re-filed under another reporting subset keeps its task semantics."""

    def task(labels: dict) -> TaskSpec:
        return TaskSpec.model_validate(
            {
                **document("task-subset", "fixture/dataset", "Solve problem."),
                "labels": labels,
            }
        )

    train = task({"subset_id": "train"})
    heldout = task({"subset_id": "heldout"})
    assert semantic_digest(train, include_reference=True) == semantic_digest(heldout, include_reference=True)
    other_subset = task({"different_label": "value"})
    assert semantic_digest(train, include_reference=True) != semantic_digest(other_subset, include_reference=True)


def test_pair_similarity_ranks_by_the_minhash_and_hash_overlaps():
    shared = [document(f"task-{index}", "fixture/shared", f"Solve shared problem {index}.") for index in range(20)]
    left = fingerprint(shared[:20], {"s3://fixture/left/specs/part-000000.parquet": "d" * 64})
    other = fingerprint(
        [document(f"other-{index}", "fixture/other", f"Completely different task {index}.") for index in range(20)],
        {"s3://fixture/other/specs/part-000000.parquet": "e" * 64},
    )
    pair = pair_similarity("Left:source", left, "Right:source", other)
    assert pair.instruction_similarity < 0.2
    empty = fingerprint([], {})
    none = pair_similarity("Left:source", left, "Right:source", empty)
    assert none.rank_score == 0


def test_candidate_pairs_shortlist_deterministically_per_changed_source():
    fingerprints = {}
    for name in ("Alpha:source", "Beta:source", "Gamma:source"):
        documents = [
            document(f"{name.split(':')[0].lower()}-{index}", f"fixture/{name}", f"Shared problem {index}.")
            for index in range(15)
        ]
        fingerprints[name] = fingerprint(documents, {f"s3://fixture/{name}/specs/part-000000.parquet": "f" * 64})
    first = candidate_pairs(fingerprints, {"Alpha:source"}, 2)
    second = candidate_pairs(fingerprints, {"Alpha:source"}, 2)
    assert [(pair.left, pair.right) for pair in first] == [(pair.left, pair.right) for pair in second]
    assert len(first) == 2
    assert all("Alpha:source" in (pair.left, pair.right) for pair in first)


def _measured(left_fraction: float, right_fraction: float) -> MeasuredOverlap:

    return MeasuredOverlap(
        seed=0,
        sample_size=20,
        similarity_threshold=0.5,
        duplicate_threshold=0.9,
        partial_threshold=0.1,
        left_to_right=_direction(left_fraction),
        right_to_left=_direction(right_fraction),
    )


def _direction(fraction: float) -> DirectionalOverlap:
    return DirectionalOverlap(
        population=20,
        sampled=20,
        matched=int(fraction * 20),
        fraction=fraction,
        ci_low=fraction,
        ci_high=fraction,
        exact_matched=int(fraction * 20),
        exact_fraction=fraction,
        neighbors=[],
        neighbors_sha256="0" * 64,
        neighbor_receipts_uri="s3://fixture/neighbors",
    )


def test_duplicate_groups_fold_a_contained_source_under_its_canonical():
    reviews = [
        DuplicateReview(
            similarity=pair_similarity(
                "Small:source",
                fingerprint([document("task", "fixture/small", "Solve.")], {"s3://fixture/small": "1" * 64}),
                "Large:source",
                fingerprint([document("task", "fixture/large", "Solve.")], {"s3://fixture/large": "2" * 64}),
            ),
            verdict=PairVerdict(label=DuplicateLabel.DUPLICATE, overlap_fraction=1.0, reason="fixture"),
            prompt_sha256="0" * 64,
            model="fixture",
            completion_id="fixture",
            prompt_tokens=1,
            completion_tokens=1,
            measurement=_measured(1.0, 0.2),
        )
    ]
    standings = {
        "Small:source": SourceStanding(5),
        "Large:source": SourceStanding(500),
    }
    groups = fold_groups(reviews, standings)
    assert len(groups) == 1
    group = groups[0]
    assert group.canonical_source_id == "Large:source"
    assert [member.source_id for member in group.members] == ["Large:source", "Small:source"]


def test_duplicate_groups_never_fold_on_a_distinct_verdict():
    reviews = [
        DuplicateReview(
            similarity=pair_similarity(
                "Small:source",
                fingerprint([document("task", "fixture/small", "Solve.")], {"s3://fixture/small": "1" * 64}),
                "Large:source",
                fingerprint([document("task", "fixture/large", "Solve.")], {"s3://fixture/large": "2" * 64}),
            ),
            verdict=PairVerdict(label=DuplicateLabel.DISTINCT, overlap_fraction=0.0, reason="fixture"),
            prompt_sha256="0" * 64,
            model="fixture",
            completion_id="fixture",
            prompt_tokens=1,
            completion_tokens=1,
            measurement=_measured(1.0, 0.2),
        )
    ]
    standings = {"Small:source": SourceStanding(5), "Large:source": SourceStanding(500)}
    assert fold_groups(reviews, standings) == []
