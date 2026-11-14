# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Symmetric content overlap against complete converted source populations."""

import itertools
import re
import sys
import unicodedata
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from heapq import nsmallest

import dupekit
import pyarrow as pa
from marin.evaluation.eval_stats import wilson_interval
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec, TextMessage, format_conversation
from taskcompendium.pipeline.sampling import seeded_order
from taskcompendium.source_fingerprints import (
    CURRENT_OVERLAP_VERSION,
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
SHINGLE_CACHE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class TaskContent:
    task_id: str
    instruction_hash: str
    text: str
    reference_hash: str | None
    tests_hash: str
    state_hash: str


def content_rows(documents: Iterable[Mapping]) -> Iterable[TaskContent]:
    """Keep complete instruction text and content hashes, without task-ID joins."""
    for document in documents:
        task = TaskSpec.model_validate(document.get("task", document))
        text = format_conversation(task.context.events)
        state = [
            event.model_dump(mode="json")
            for event in task.context.events
            if not isinstance(event, TextMessage) or event.role == "assistant"
        ]
        reference, tests = reference_test_hashes(task)
        yield TaskContent(
            task.id, canonical_sha256(normalized_instruction(task)), text, reference, tests, canonical_sha256(state)
        )


def wrapper_lines(rows: list[TaskContent]) -> set[str]:
    """Find long wrapper lines shared by at least 90% of sampled texts."""
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
    return wrappers


def normalized_content(row: TaskContent, wrappers: set[str]) -> TaskContent:
    text = "\n".join(line for line in row.text.splitlines() if line.strip() not in wrappers).strip()
    text = text or row.text
    text = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return TaskContent(row.task_id, row.instruction_hash, text, row.reference_hash, row.tests_hash, row.state_hash)


def strip_boilerplate(rows: list[TaskContent]) -> list[TaskContent]:
    wrappers = wrapper_lines(rows)
    return [normalized_content(row, wrappers) for row in rows]


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


def lsh_band_keys(signature: list[int]) -> Iterable[tuple[int, tuple[int, ...]]]:
    for start in range(0, len(signature), BAND_ROWS):
        yield start, tuple(signature[start : start + BAND_ROWS])


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    intersection = len(left & right)
    union = len(left) + len(right) - intersection
    return intersection / union if union else 0.0


class PopulationIndex:
    """Match hashes and state before materializing normalized text or LSH buckets."""

    def __init__(
        self,
        rows: list[TaskContent],
        inputs: dict[str, str],
        normalized_loader: Callable[[], list[TaskContent]] | None = None,
        *,
        shingle_cache_bytes: int = SHINGLE_CACHE_BYTES,
    ):
        if not rows:
            raise ValueError("Overlap requires nonempty source populations")
        self.inputs_sha256 = canonical_sha256(inputs)
        self.raw_rows = rows
        self.exact: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            self.exact[row.instruction_hash].append(index)
        self.states = {row.state_hash for row in rows}
        self._normalized_loader = normalized_loader
        self._normalized: dict[int, TaskContent] = {}
        self._rows: list[TaskContent] | None = None
        self._wrappers: set[str] | None = None
        self.stripped: dict[str, list[int]] = defaultdict(list)
        self.buckets: dict[tuple[int, tuple[int, ...]], list[int]] | None = None
        self._shingle_cache_limit = shingle_cache_bytes
        self._shingle_cache: OrderedDict[int, tuple[set[str], int]] = OrderedDict()
        self._shingle_cache_bytes = 0

    def _candidate_shingles(self, index: int) -> set[str]:
        cached = self._shingle_cache.get(index)
        if cached is not None:
            self._shingle_cache.move_to_end(index)
            return cached[0]
        result = shingles(self.rows[index].text)
        size = sys.getsizeof(result) + sum(sys.getsizeof(value) for value in result)
        if size > self._shingle_cache_limit:
            return result
        while self._shingle_cache_bytes + size > self._shingle_cache_limit:
            _, (_, removed_bytes) = self._shingle_cache.popitem(last=False)
            self._shingle_cache_bytes -= removed_bytes
        self._shingle_cache[index] = result, size
        self._shingle_cache_bytes += size
        return result

    def normalized_row(self, row: TaskContent) -> TaskContent:
        if self._normalized_loader is not None:
            normalized = self._normalized_loader()
            self._normalized = {id(raw): ready for raw, ready in zip(self.raw_rows, normalized, strict=True)}
            self._normalized_loader = None
        if self._normalized:
            return self._normalized[id(row)]
        if self._wrappers is None:
            self._wrappers = wrapper_lines(self.raw_rows)
        return normalized_content(row, self._wrappers)

    @property
    def rows(self) -> list[TaskContent]:
        if self._rows is None:
            self._rows = [self.normalized_row(row) for row in self.raw_rows]
            for index, row in enumerate(self._rows):
                self.stripped[canonical_sha256((row.text, row.state_hash))].append(index)
        return self._rows

    def raw_sample(self, size: int) -> list[TaskContent]:
        selected = nsmallest(
            size, enumerate(self.raw_rows), key=lambda item: seeded_order(f"{item[0]}:{item[1].instruction_hash}", SEED)
        )
        return [row for _, row in selected]

    def sample(self, size: int) -> list[TaskContent]:
        return [self.normalized_row(row) for row in self.raw_sample(size)]

    def neighbors(self, queries: list[TaskContent], source: "PopulationIndex | None" = None) -> list[TaskNeighbor]:
        result: list[TaskNeighbor | None] = [None] * len(queries)
        prepared = {}
        missing = []
        for index, row in enumerate(queries):
            candidates = self.exact.get(row.instruction_hash)
            matching_rows = self.raw_rows
            if not candidates and row.state_hash not in self.states:
                result[index] = neighbor_receipt(row, None, 0.0, "unmatched")
                continue
            if not candidates:
                prepared[index] = source.normalized_row(row) if source is not None else row
                matching_rows = self.rows
                candidates = self.stripped.get(canonical_sha256((prepared[index].text, row.state_hash)))
            if candidates:
                partner = min(
                    (matching_rows[item] for item in candidates),
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
                    for key in lsh_band_keys(signature):
                        self.buckets[key].append(index)
            for index, signature in zip(missing, signatures([prepared[item] for item in missing]), strict=True):
                candidates = set()
                for key in lsh_band_keys(signature):
                    candidates.update(self.buckets.get(key, ()))
                query = prepared[index]
                query_shingles = shingles(query.text)
                best, score = None, 0.0
                for candidate in sorted(candidates):
                    partner = self.rows[candidate]
                    if partner.state_hash != query.state_hash:
                        continue
                    similarity = jaccard_similarity(query_shingles, self._candidate_shingles(candidate))
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
    queries = source.raw_sample(sample_size)
    neighbors = target.neighbors(queries, source)
    matched = sum(row.similarity >= SIMILARITY_THRESHOLD for row in neighbors)
    exact = sum(row.instruction_hash in target.exact for row in source.raw_rows)
    fraction = matched / len(queries)
    interval = wilson_interval(matched, len(queries))
    low, high = (fraction, fraction) if len(queries) == len(source.raw_rows) else (interval.low, interval.high)
    return DirectionalOverlap(
        population=len(source.raw_rows),
        sampled=len(queries),
        matched=matched,
        fraction=fraction,
        ci_low=low,
        ci_high=high,
        exact_matched=exact,
        exact_fraction=exact / len(source.raw_rows),
        neighbors=neighbors,
        neighbors_sha256=canonical_sha256([row.model_dump(mode="json") for row in neighbors]),
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
        prompt_sha256=measurement_digest(pair, measurement),
        model=f"deterministic:{CURRENT_OVERLAP_VERSION}",
        completion_id=CURRENT_OVERLAP_VERSION,
        prompt_tokens=0,
        completion_tokens=0,
        measurement=measurement,
    )


def measurement_digest(pair: PairSimilarity, measurement: MeasuredOverlap) -> str:
    """Bind complete sample evidence independently of shortlist scores or presentation."""
    if measurement.left_to_right.neighbors_sha256 is None or measurement.right_to_left.neighbors_sha256 is None:
        raise ValueError("Measured overlap requires complete neighbor digests")
    evidence = measurement.model_dump(
        mode="json",
        exclude={
            "left_to_right": {"neighbors", "neighbor_receipts_uri"},
            "right_to_left": {"neighbors", "neighbor_receipts_uri"},
        },
    )
    return canonical_sha256({"left": pair.left, "right": pair.right, "measurement": evidence})


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
    for row in source.raw_rows:
        candidates = target.exact.get(row.instruction_hash, ())
        if not candidates:
            continue
        references = {target.raw_rows[index].reference_hash for index in candidates}
        if row.reference_hash is None or references == {None}:
            missing += 1
            continue
        checked += 1
        if row.reference_hash not in references:
            mismatch += 1
            if len(examples) < 20:
                partner = target.raw_rows[candidates[0]]
                examples.append(
                    {
                        "task_id": row.task_id,
                        "partner_task_id": partner.task_id,
                        "question": source.normalized_row(row).text[:2000],
                        "reference_sha256": row.reference_hash,
                        "partner_reference_sha256": partner.reference_hash,
                    }
                )
    return {
        "population": len(source.raw_rows),
        "checked": checked,
        "mismatched": mismatch,
        "missing_reference": missing,
        "examples": examples,
    }
