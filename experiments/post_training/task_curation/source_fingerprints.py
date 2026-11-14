# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic source fingerprints and candidate ranking over normalized tasks."""

import hashlib
import itertools
import re
import unicodedata
from collections.abc import Iterable, Mapping
from heapq import nsmallest

import dupekit
import pyarrow as pa
from taskcompendium.identity import canonical_sha256, task_identity
from taskcompendium.models import TaskSpec, VerifyitGrader, format_conversation
from taskcompendium.pipeline.sampling import seeded_order, seeded_sample
from taskcompendium.source_fingerprints import (
    DuplicateGroup,
    DuplicateGroupMember,
    DuplicateLabel,
    DuplicateReview,
    FingerprintSample,
    PairSimilarity,
    SourceFingerprint,
    SourceLineage,
)

from infra.marina.apps.rl_data_catalog.server.source_identity import (
    canonical_source_id,
    canonical_source_map,
    original_source_id,
)

FINGERPRINT_SEED = 0
REVIEW_SAMPLE_SIZE = 20
REVIEW_INSTRUCTION_CHARACTERS = 1000
MINHASH_PERMUTATIONS = 64
MINHASH_NGRAM_SIZE = 5
HASH_SAMPLE_SIZE = 128
FINGERPRINT_BATCH_SIZE = 32


def normalized_instruction(task: TaskSpec) -> str:
    text = format_conversation(task.context.events)
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def reference_test_hashes(task: TaskSpec) -> tuple[str | None, str]:
    """Hash reference content and the executable grading contract separately."""
    reference = [resource.source.model_dump(mode="json") for resource in task.resources.oracle]
    if isinstance(task.grader, VerifyitGrader) and task.grader.mode in {"math", "mcq", "exact", "judge"}:
        reference.append(task.grader.parameters)
    grader = task.grader.model_dump(mode="json", exclude={"environment", "judge"})
    private_tests = [resource.source.model_dump(mode="json") for resource in task.resources.verifier]
    return (
        canonical_sha256(reference) if reference else None,
        canonical_sha256({"grader": grader, "files": private_tests}),
    )


def fingerprint_source(documents: Iterable[Mapping], inputs: dict[str, str]) -> SourceFingerprint:
    """Fingerprint the complete population and retain a fixed-seed review sample."""
    seen = set()
    duplicate_count = 0
    instructions, references, tests, task_ids, urls, lineage = set(), set(), set(), set(), set(), set()
    signature = [2**64 - 1] * MINHASH_PERMUTATIONS

    def samples() -> Iterable[FingerprintSample]:
        nonlocal duplicate_count, signature
        for batch in itertools.batched(documents, FINGERPRINT_BATCH_SIZE):
            tasks = [TaskSpec.model_validate(document.get("task", document)) for document in batch]
            texts = [normalized_instruction(task) for task in tasks]
            sketches = (
                dupekit.transform(
                    pa.RecordBatch.from_pylist([{"text": text} for text in texts]),
                    [
                        dupekit.Transformation.MinHash(
                            "text", "signature", MINHASH_PERMUTATIONS, MINHASH_NGRAM_SIZE, FINGERPRINT_SEED
                        )
                    ],
                )
                .column("signature")
                .to_pylist()
            )
            for task, text, sketch in zip(tasks, texts, sketches, strict=True):
                signature = [min(left, right) for left, right in zip(signature, sketch, strict=True)]
                key = task_identity(task)
                duplicate_count += key in seen
                seen.add(key)
                instruction_hash = canonical_sha256(text)
                instructions.add(instruction_hash)
                task_ids.add(canonical_sha256(task.id))
                task_ids.add(canonical_sha256({"dataset": task.source.dataset, "row": task.source.row}))
                urls.update(canonical_sha256(url) for url in re.findall(r"https?://[^\s]+", task.source.row))
                lineage.add((task.source.dataset, task.source.revision))
                reference_hash, tests_hash = reference_test_hashes(task)
                if reference_hash is not None:
                    references.add(reference_hash)
                tests.add(tests_hash)
                yield FingerprintSample(
                    task_id=task.id,
                    task_sha256=canonical_sha256(task.model_dump(mode="json")),
                    instruction_sha256=instruction_hash,
                    instruction=text[:REVIEW_INSTRUCTION_CHARACTERS],
                    instruction_characters=len(text),
                    reference_sha256=reference_hash,
                    tests_sha256=tests_hash,
                )

    count, selected = seeded_sample(
        samples(), size=REVIEW_SAMPLE_SIZE, key=lambda sample: seeded_order(sample.task_id, FINGERPRINT_SEED)
    )
    return SourceFingerprint(
        seed=FINGERPRINT_SEED,
        task_count=count,
        within_source_duplicates=duplicate_count,
        sample_sha256=canonical_sha256([sample.model_dump(mode="json") for sample in selected]),
        sample=selected,
        instruction_minhash=signature if count else [],
        instruction_hashes=nsmallest(HASH_SAMPLE_SIZE, instructions),
        reference_hashes=nsmallest(HASH_SAMPLE_SIZE, references),
        test_hashes=nsmallest(HASH_SAMPLE_SIZE, tests),
        task_id_hashes=nsmallest(HASH_SAMPLE_SIZE, task_ids),
        url_hashes=nsmallest(HASH_SAMPLE_SIZE, urls),
        lineage=[SourceLineage(dataset=dataset, revision=revision) for dataset, revision in sorted(lineage)],
        inputs=inputs,
    )


def sampled_hash_overlap(left: list[str], right: list[str]) -> float:
    """Compare the shared bottom-k region, avoiding bias from differently sized sets."""
    if not left or not right:
        return 0.0
    cutoff = min(left[-1], right[-1])
    a, b = {value for value in left if value <= cutoff}, {value for value in right if value <= cutoff}
    return len(a & b) / len(a | b) if a | b else 0.0


def pair_similarity(left_id: str, left: SourceFingerprint, right_id: str, right: SourceFingerprint) -> PairSimilarity:
    if not left.task_count or not right.task_count:
        return PairSimilarity(
            left=left_id,
            right=right_id,
            rank_score=0,
            instruction_similarity=0,
            instruction_overlap=0,
            reference_overlap=0,
            test_overlap=0,
            task_id_overlap=0,
            url_overlap=0,
            shared_lineage=False,
        )
    if (left.version, left.seed, len(left.instruction_minhash)) != (
        right.version,
        right.seed,
        len(right.instruction_minhash),
    ):
        raise ValueError("Cannot compare incompatible fingerprint schemes")
    instruction_similarity = sum(
        a == b for a, b in zip(left.instruction_minhash, right.instruction_minhash, strict=True)
    ) / len(left.instruction_minhash)
    instruction_overlap = sampled_hash_overlap(left.instruction_hashes, right.instruction_hashes)
    reference_overlap = sampled_hash_overlap(left.reference_hashes, right.reference_hashes)
    test_overlap = sampled_hash_overlap(left.test_hashes, right.test_hashes)
    task_id_overlap = sampled_hash_overlap(left.task_id_hashes, right.task_id_hashes)
    url_overlap = sampled_hash_overlap(left.url_hashes, right.url_hashes)
    shared_lineage = bool({item.dataset for item in left.lineage} & {item.dataset for item in right.lineage})
    score = max(
        instruction_overlap,
        url_overlap,
        0.7 * instruction_similarity + 0.2 * max(reference_overlap, test_overlap) + 0.1 * shared_lineage,
    )
    return PairSimilarity(
        left=left_id,
        right=right_id,
        rank_score=score,
        instruction_similarity=instruction_similarity,
        instruction_overlap=instruction_overlap,
        reference_overlap=reference_overlap,
        test_overlap=test_overlap,
        task_id_overlap=task_id_overlap,
        url_overlap=url_overlap,
        shared_lineage=shared_lineage,
    )


def candidate_pairs(
    fingerprints: dict[str, SourceFingerprint], changed_sources: set[str], top_k: int
) -> list[PairSimilarity]:
    """Shortlist at most top-k pairs per changed source, with deterministic ties."""
    fingerprints = canonical_source_map(fingerprints)
    changed_sources = {canonical_source_id(key) for key in changed_sources}
    if not changed_sources.issubset(fingerprints):
        raise ValueError("Changed sources need complete fingerprints before duplicate review")
    selected = {}
    for source_id in sorted(changed_sources, key=original_source_id):
        if not fingerprints[source_id].task_count:
            continue
        candidates = []
        for other_id in sorted(fingerprints.keys() - {source_id}, key=original_source_id):
            if not fingerprints[other_id].task_count:
                continue
            left, right = sorted((source_id, other_id), key=original_source_id)
            candidates.append(pair_similarity(left, fingerprints[left], right, fingerprints[right]))
        for pair in sorted(
            candidates,
            key=lambda pair: (-pair.rank_score, original_source_id(pair.left), original_source_id(pair.right)),
        )[:top_k]:
            selected[pair.left, pair.right] = pair
    return [selected[key] for key in sorted(selected, key=lambda pair: tuple(map(original_source_id, pair)))]


def duplicate_groups(reviews: list[DuplicateReview]) -> list[DuplicateGroup]:
    """Group advisory flags; absent canonical-pair estimates remain unknown."""
    edges = {}
    neighbors = {}
    for review in reviews:
        if review.verdict.label == DuplicateLabel.DISTINCT:
            continue
        pair = review.similarity
        edges[frozenset((pair.left, pair.right))] = review.verdict.overlap_fraction
        neighbors.setdefault(pair.left, set()).add(pair.right)
        neighbors.setdefault(pair.right, set()).add(pair.left)
    pending = set(neighbors)
    groups = []
    while pending:
        frontier, members = [min(pending, key=original_source_id)], set()
        while frontier:
            member = frontier.pop()
            if member in members:
                continue
            members.add(member)
            frontier.extend(neighbors[member] - members)
        pending -= members
        canonical = min(members, key=original_source_id)
        group_id = hashlib.sha256("\n".join(sorted(map(original_source_id, members))).encode()).hexdigest()[:20]
        groups.append(
            DuplicateGroup(
                group_id=group_id,
                canonical_source_id=canonical,
                members=[
                    DuplicateGroupMember(
                        source_id=member,
                        overlap_with_canonical=1.0 if member == canonical else edges.get(frozenset((canonical, member))),
                    )
                    for member in sorted(members, key=original_source_id)
                ],
            )
        )
    return sorted(groups, key=lambda group: group.group_id)
