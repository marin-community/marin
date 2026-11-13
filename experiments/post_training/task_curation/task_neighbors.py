# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Symmetric content overlap against complete converted source populations."""

import itertools
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from heapq import nsmallest

import dupekit
import pyarrow as pa
from marin.evaluation.eval_stats import wilson_interval
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.sampling import seeded_order
from taskcompendium.source_fingerprints import (
    DirectionalOverlap,
    DuplicateLabel,
    DuplicateReview,
    MeasuredOverlap,
    PairSimilarity,
    PairVerdict,
    TaskNeighbor,
)

from experiments.post_training.task_curation.source_fingerprints import (
    normalized_instruction,
    reference_test_hashes,
)

SAMPLE_SIZE = 4096
SEED = 0
SIMILARITY_THRESHOLD = 0.9
DUPLICATE_THRESHOLD = 0.98
PARTIAL_THRESHOLD = 0.01
PERMUTATIONS = 128
BAND_ROWS = 4
NGRAM_SIZE = 5
BOILERPLATE_MIN_CHARACTERS = 80


@dataclass(frozen=True)
class TaskContent:
    task_id: str
    instruction_hash: str
    text: str
    reference_hash: str | None
    tests_hash: str


def content_rows(documents: Iterable[Mapping]) -> Iterable[TaskContent]:
    """Keep complete instruction text and content hashes, without task-ID joins."""
    for document in documents:
        task = TaskSpec.model_validate(document.get("task", document))
        # System messages describe the interface, while user turns carry the task.
        messages = [
            event.content for event in task.context.events if isinstance(event, TextMessage) and event.role == "user"
        ]
        text = "\n".join(messages) if messages else normalized_instruction(task)
        reference, tests = reference_test_hashes(task)
        yield TaskContent(task.id, canonical_sha256(normalized_instruction(task)), text, reference, tests)


def strip_boilerplate(rows: list[TaskContent]) -> list[TaskContent]:
    """Remove repeated whole wrapper lines, retaining short and task-specific text.

    Only long lines shared by at least 90% of a content-seeded panel are stripped.
    A line repeated because the source has one unique instruction is retained.
    No common-prefix character trimming can erase a varying number or option.
    """
    panel = nsmallest(128, {row.text for row in rows}, key=lambda text: seeded_order(text, SEED))
    lines = Counter(
        line.strip()
        for text in panel
        for line in set(text.splitlines())
        if len(line.strip()) >= BOILERPLATE_MIN_CHARACTERS
    )
    wrappers = {
        line
        for line, count in lines.items()
        if len(panel) >= 8
        and count / len(panel) >= 0.9
        and re.search(
            r"\b(?:you are|your response|final answer|assistant|grader|output format|instructions|reasoning|boxed)\b",
            line.casefold(),
        )
    }
    result = []
    for row in rows:
        text = "\n".join(line for line in row.text.splitlines() if line.strip() not in wrappers).strip()
        # Empty text must never make unrelated wrapper-only tasks exact matches.
        text = text or row.text
        text = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
        result.append(TaskContent(row.task_id, row.instruction_hash, text, row.reference_hash, row.tests_hash))
    return result


def signatures(rows: list[TaskContent]) -> Iterable[list[int]]:
    for batch in itertools.batched(rows, 256):
        yield from dupekit.transform(
            pa.RecordBatch.from_pylist([{"text": row.text} for row in batch]),
            [dupekit.Transformation.MinHash("text", "signature", PERMUTATIONS, NGRAM_SIZE, SEED)],
        ).column("signature").to_pylist()


def shingles(text: str) -> set[str]:
    # Word shingles keep a changed answer option or number visible in short tasks.
    words = re.findall(r"\w+|[^\w\s]", text)
    width = min(3, len(words))
    return {" ".join(words[index : index + width]) for index in range(len(words) - width + 1)} if width else set()


class PopulationIndex:
    """An exact lookup and lazy MinHash/LSH search of every target task."""

    def __init__(self, rows: list[TaskContent], inputs: dict[str, str]):
        if not rows:
            raise ValueError("Overlap requires nonempty source populations")
        self.inputs_sha256 = canonical_sha256(inputs)
        self.rows = strip_boilerplate(rows)
        self.exact: dict[str, list[int]] = defaultdict(list)
        self.stripped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            self.exact[row.instruction_hash].append(index)
            self.stripped[canonical_sha256(row.text)].append(index)
        self.buckets: dict[tuple[int, tuple[int, ...]], list[int]] | None = None

    def sample(self, size: int) -> list[TaskContent]:
        # Including the row ordinal samples task occurrences, including repeated
        # instructions, without biasing toward tasks whose source IDs coincide.
        selected = nsmallest(
            size, enumerate(self.rows), key=lambda item: seeded_order(f"{item[0]}:{item[1].instruction_hash}", SEED)
        )
        return [row for _, row in selected]

    def neighbors(self, queries: list[TaskContent]) -> list[TaskNeighbor]:
        result: list[TaskNeighbor | None] = [None] * len(queries)
        missing = []
        for index, row in enumerate(queries):
            candidates = self.exact.get(row.instruction_hash) or self.stripped.get(canonical_sha256(row.text))
            if candidates:
                # References break content-identical ties only, never select a question.
                partner = min(
                    (self.rows[item] for item in candidates),
                    key=lambda candidate: (
                        candidate.reference_hash != row.reference_hash,
                        candidate.tests_hash != row.tests_hash,
                        candidate.task_id,
                    ),
                )
                result[index] = neighbor_receipt(row, partner, 1.0, "exact")
            else:
                missing.append(index)
        if missing:
            if self.buckets is None:
                self.buckets = defaultdict(list)
                for index, signature in enumerate(signatures(self.rows)):
                    for start in range(0, PERMUTATIONS, BAND_ROWS):
                        self.buckets[start, tuple(signature[start : start + BAND_ROWS])].append(index)
            for index, signature in zip(missing, signatures([queries[item] for item in missing]), strict=True):
                candidates = set()
                for start in range(0, PERMUTATIONS, BAND_ROWS):
                    candidates.update(self.buckets.get((start, tuple(signature[start : start + BAND_ROWS])), ()))
                query = queries[index]
                query_shingles = shingles(query.text)
                best, score = None, 0.0
                for candidate in sorted(candidates):
                    partner = self.rows[candidate]
                    other = shingles(partner.text)
                    union = query_shingles | other
                    similarity = len(query_shingles & other) / len(union) if union else 0.0
                    if similarity > score:
                        best, score = partner, similarity
                result[index] = neighbor_receipt(query, best, score, "lsh" if best else "unmatched")
        assert all(row is not None for row in result)
        return [row for row in result if row is not None]


def neighbor_receipt(query: TaskContent, partner: TaskContent | None, score: float, method: str) -> TaskNeighbor:
    return TaskNeighbor(
        task_id=query.task_id,
        partner_task_id=partner.task_id if partner else None,
        similarity=score,
        method=method,
        reference_match=(
            (query.reference_hash == partner.reference_hash)
            if partner and query.reference_hash is not None and partner.reference_hash is not None
            else None
        ),
        tests_match=(query.tests_hash == partner.tests_hash) if partner else None,
    )


def directional_overlap(source: PopulationIndex, target: PopulationIndex, sample_size: int) -> DirectionalOverlap:
    """Estimate task-occurrence overlap, and census exact matches independently."""
    if sample_size <= 0:
        raise ValueError("Sample size must be positive")
    queries = source.sample(sample_size)
    neighbors = target.neighbors(queries)
    matched = sum(row.similarity >= SIMILARITY_THRESHOLD for row in neighbors)
    exact = sum(row.instruction_hash in target.exact for row in source.rows)
    fraction = matched / len(queries)
    interval = wilson_interval(matched, len(queries))
    low, high = (fraction, fraction) if len(queries) == len(source.rows) else (interval.low, interval.high)
    return DirectionalOverlap(
        population=len(source.rows),
        sampled=len(queries),
        matched=matched,
        fraction=fraction,
        ci_low=low,
        ci_high=high,
        exact_matched=exact,
        exact_fraction=exact / len(source.rows),
        neighbors=neighbors,
    )


def measured_review(
    pair: PairSimilarity,
    left: DirectionalOverlap,
    right: DirectionalOverlap,
    sample_size: int,
    left_inputs_sha256: str,
    right_inputs_sha256: str,
) -> DuplicateReview:
    """Mechanical labels cannot be overridden by a language-model opinion."""
    fractions = [max(direction.fraction, direction.exact_fraction) for direction in (left, right)]
    lower = [max(direction.ci_low, direction.exact_fraction) for direction in (left, right)]
    if min(lower) >= DUPLICATE_THRESHOLD:
        label, fraction = DuplicateLabel.DUPLICATE, 1.0
    elif max(lower) >= PARTIAL_THRESHOLD:
        label = DuplicateLabel.PARTIAL_OVERLAP
        fraction = min(value for value in fractions if value > 0)
    else:
        label, fraction = DuplicateLabel.DISTINCT, 0.0
    measurement = MeasuredOverlap(
        seed=SEED,
        sample_size=sample_size,
        left_inputs_sha256=left_inputs_sha256,
        right_inputs_sha256=right_inputs_sha256,
        similarity_threshold=SIMILARITY_THRESHOLD,
        duplicate_threshold=DUPLICATE_THRESHOLD,
        partial_threshold=PARTIAL_THRESHOLD,
        left_to_right=left,
        right_to_left=right,
    )
    reason = (
        f"Content NN A→B={left.fraction:.4f} (95% CI {left.ci_low:.4f}-{left.ci_high:.4f}); "
        f"B→A={right.fraction:.4f} (95% CI {right.ci_low:.4f}-{right.ci_high:.4f}). "
        f"Full exact fractions={left.exact_fraction:.4f}/{right.exact_fraction:.4f}. "
        "LSH retrieval is approximate; distinct means below the 1% reporting threshold, not proof of zero reuse."
    )
    return DuplicateReview(
        similarity=pair,
        verdict=PairVerdict(label=label, overlap_fraction=fraction, reason=reason),
        prompt_sha256=canonical_sha256(
            {"pair": pair.model_dump(mode="json"), "measurement": measurement.model_dump(mode="json")}
        ),
        model="deterministic:content-nn-v2",
        completion_id="content-nn-v2",
        prompt_tokens=0,
        completion_tokens=0,
        measurement=measurement,
    )


def compare_populations(
    pair: PairSimilarity, left: PopulationIndex, right: PopulationIndex, sample_size: int = SAMPLE_SIZE
) -> DuplicateReview:
    return measured_review(
        pair,
        directional_overlap(left, right, sample_size),
        directional_overlap(right, left, sample_size),
        sample_size,
        left.inputs_sha256,
        right.inputs_sha256,
    )


def exact_reference_integrity(source: PopulationIndex, target: PopulationIndex) -> dict:
    """Census answer agreement for identical complete instructions, preserving conflicts."""
    checked, mismatch, missing = 0, 0, 0
    examples = []
    for row in source.rows:
        candidates = target.exact.get(row.instruction_hash, ())
        if not candidates:
            continue
        references = {target.rows[index].reference_hash for index in candidates}
        if row.reference_hash is None or references == {None}:
            missing += 1
            continue
        checked += 1
        if row.reference_hash not in references:
            mismatch += 1
            if len(examples) < 20:
                partner = target.rows[candidates[0]]
                examples.append(
                    {
                        "task_id": row.task_id,
                        "partner_task_id": partner.task_id,
                        "question": row.text[:2000],
                        "reference_sha256": row.reference_hash,
                        "partner_reference_sha256": partner.reference_hash,
                    }
                )
    return {
        "population": len(source.rows),
        "checked": checked,
        "mismatched": mismatch,
        "missing_reference": missing,
        "examples": examples,
    }
