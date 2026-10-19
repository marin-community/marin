"""Deterministic stratified panel construction and loading for NUPA5K."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import ijson

nupa = import_module("eval.chat_benchmarks.NUPA-Loose.eval_instruct")


NUPA5K_SIZE = 5_000
NUPA5K_TASK_COUNT = 44
NUPA5K_STRATUM_COUNT = 2_391
DATA_DIR = Path(__file__).with_name("data")
MANIFEST_PATH = DATA_DIR / "nupa5k_manifest.jsonl"
MANIFEST_SHA256 = "27f122972d76dc2a1170a3d105a9856b55beedabe037f697c1a1aa2baeb67036"


@dataclass(frozen=True, order=True)
class SourceIdentity:
    """Stable identity for one source record in a NUPA task/digit stratum."""

    task_name: str
    digit: int
    sha256: str

    def as_record_id(self, split: str) -> str:
        return f"{split}:{self.task_name}:{self.digit}:{self.sha256}"

    def to_dict(self) -> dict[str, str | int]:
        return {"task_name": self.task_name, "digit": self.digit, "sha256": self.sha256}


Stratum = tuple[str, int]


@dataclass(frozen=True)
class SourceStratum:
    """Source strings belonging to one task and digit length."""

    task_name: str
    digit: int
    texts: tuple[str, ...]


def source_text_sha256(text: str) -> str:
    """Return the identity digest for an exact NUPA source string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_nupa5k_identities(source: Path, *, panel_size: int = NUPA5K_SIZE) -> tuple[SourceIdentity, ...]:
    """Build ordered panel identities from nested source JSON."""
    unique_counts = {
        (stratum.task_name, stratum.digit): len({source_text_sha256(text) for text in stratum.texts})
        for stratum in iter_source_strata(source)
    }
    allocation = _round_robin_allocation(unique_counts, panel_size=panel_size)
    quotas = Counter(allocation)

    selected_digests = {}
    for source_stratum in iter_source_strata(source):
        stratum = (source_stratum.task_name, source_stratum.digit)
        if quota := quotas[stratum]:
            selected_digests[stratum] = sorted({source_text_sha256(text) for text in source_stratum.texts})[:quota]

    offsets: Counter[Stratum] = Counter()
    identities = []
    for task_name, digit in allocation:
        stratum = (task_name, digit)
        identities.append(SourceIdentity(task_name, digit, selected_digests[stratum][offsets[stratum]]))
        offsets[stratum] += 1
    return tuple(identities)


def write_manifest(path: Path, identities: Sequence[SourceIdentity]) -> str:
    """Write ordered identities as JSONL and return the file's SHA-256 digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for identity in identities:
            output.write(json.dumps(identity.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_nupa5k_manifest(path: Path = MANIFEST_PATH) -> tuple[SourceIdentity, ...]:
    """Load and validate the checked-in ordered NUPA5K identity manifest."""
    if path == MANIFEST_PATH:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != MANIFEST_SHA256:
            raise ValueError(f"NUPA5K manifest digest changed: expected {MANIFEST_SHA256}, found {digest}")

    identities = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            identities.append(
                SourceIdentity(
                    task_name=record["task_name"],
                    digit=int(record["digit"]),
                    sha256=record["sha256"],
                )
            )
    identities = tuple(identities)
    if len(identities) != NUPA5K_SIZE or len(set(identities)) != NUPA5K_SIZE:
        raise ValueError(f"NUPA5K manifest must contain {NUPA5K_SIZE} unique identities")
    if len({identity.task_name for identity in identities}) != NUPA5K_TASK_COUNT:
        raise ValueError(f"NUPA5K manifest must cover {NUPA5K_TASK_COUNT} tasks")
    if len({(identity.task_name, identity.digit) for identity in identities}) != NUPA5K_STRATUM_COUNT:
        raise ValueError(f"NUPA5K manifest must cover {NUPA5K_STRATUM_COUNT} strata")
    return identities


def load_panel_records(
    source: Path,
    *,
    split: str,
    identities: Sequence[SourceIdentity],
) -> list[dict[str, Any]]:
    """Resolve manifest identities against source JSON and return manifest-ordered records."""
    if len(set(identities)) != len(identities):
        raise ValueError("NUPA5K identities must be unique")
    targets: dict[str, dict[int, tuple[SourceIdentity, ...]]] = {}
    for identity in identities:
        by_digit = targets.setdefault(identity.task_name, {})
        by_digit[identity.digit] = (*by_digit.get(identity.digit, ()), identity)

    records_by_identity: dict[SourceIdentity, dict[str, Any]] = {}
    for task_name, digit_groups in iter_source_tasks(source):
        task_targets = targets.get(task_name)
        if task_targets is None:
            continue

        selected_texts: dict[str, list[str]] = {digit: [] for digit in digit_groups}
        selected_identities = []
        for digit_key, texts in digit_groups.items():
            digit = int(digit_key)
            stratum_targets = task_targets.get(digit, ())
            if not stratum_targets:
                continue
            target_digests = {identity.sha256 for identity in stratum_targets}
            matches = {}
            for text in texts:
                digest = source_text_sha256(text)
                if digest in target_digests and digest not in matches:
                    matches[digest] = text
            ordered = sorted(stratum_targets, key=lambda identity: identity.sha256)
            selected_texts[digit_key] = [matches[identity.sha256] for identity in ordered if identity.sha256 in matches]
            selected_identities.extend(identity for identity in ordered if identity.sha256 in matches)

        records = nupa.flatten_nupa_tasks({task_name: selected_texts}, split=split)
        for identity, record in zip(selected_identities, records, strict=True):
            record["id"] = identity.as_record_id(split)
            record["source_sha256"] = identity.sha256
            records_by_identity[identity] = record

    missing = [identity for identity in identities if identity not in records_by_identity]
    if missing:
        first = missing[0]
        raise ValueError(
            f"NUPA5K source is missing {len(missing)} manifest identities; first missing record is {first.as_record_id(split)}"
        )
    return [records_by_identity[identity] for identity in identities]


def iter_source_tasks(source: Path) -> Iterator[tuple[str, dict[str, list[str]]]]:
    """Yield validated NUPA tasks from nested source JSON."""
    with source.open("rb") as source_file:
        for task_name, value in ijson.kvitems(source_file, ""):
            yield task_name, nupa.validate_nupa_digit_groups(task_name, value)


def iter_source_strata(source: Path) -> Iterator[SourceStratum]:
    """Yield each nonempty task/digit stratum from nested source JSON."""
    for task_name, digit_groups in iter_source_tasks(source):
        for digit, texts in digit_groups.items():
            if texts:
                yield SourceStratum(task_name, int(digit), tuple(texts))


def _round_robin_allocation(unique_counts: Mapping[Stratum, int], *, panel_size: int) -> tuple[Stratum, ...]:
    if panel_size <= 0:
        raise ValueError("panel_size must be positive")
    strata = sorted(stratum for stratum, count in unique_counts.items() if count > 0)
    if sum(unique_counts[stratum] for stratum in strata) < panel_size:
        raise ValueError(f"cannot select {panel_size} unique source records")

    allocation = []
    round_index = 0
    while len(allocation) < panel_size:
        for stratum in strata:
            if unique_counts[stratum] > round_index:
                allocation.append(stratum)
                if len(allocation) == panel_size:
                    return tuple(allocation)
        round_index += 1
    return tuple(allocation)
