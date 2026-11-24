# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decontaminate normalized data against the eval items: a rarest-key scan, then a pair scorer.

Each stage is a step, resumable shard by shard:

1. Keys (:func:`decon_keys_step`, once). Every item of the eval item table (``ITEM_SCHEMA``)
   gets up to three key streams (``decon_keys.key_streams``); each stream's key is its rarest 20-character
   window of letters and digits, counted over reference shards. Candidate windows lie outside the prefix and
   suffix that all of an eval's streams share (its prompt text), and equally rare windows go to the one that
   the fewest other items hold. Writes ``keys.parquet`` and the locator table that the scan reads.
2. Per normalized source, three steps:

   - scan (:func:`decon_scan_step`): every record of every shard is read three ways (as is, and with JSON
     escapes read once and twice); each item whose key a reading holds gives a candidate pair, the record text
     around the item (``decon_keys.record_pairs``; records over ``MAX_DOC_CHARS`` are read in overlapping
     pieces);
   - score (:func:`decon_score_step`, on :class:`ScoreWorkers`): a :class:`PairScorer` gives each pair the
     probability that the excerpt holds the item;
   - mark (:func:`decon_mark_step`): a record is contaminated when one of its pairs scores at or above the
     cutoff. A cutoff change re-runs only this step.

The marks are a co-partitioned attributes dataset, one file per input shard in input order (the contract that
the datakit store reads): ``id``, ``partition_id``, ``contaminated``, ``max_score`` (the record's best pair
score, 0 without a candidate), and ``matched_items`` (the items of its pairs at or above the cutoff). A
flagged sample (``outputs/flagged_sample``) keeps up to ``flagged_sample_size`` contaminated records per
shard with the excerpt of their best pair.
"""

import functools
import io
import json
import logging
import math
import os
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple, Protocol

import numpy as np
import pandas as pd
import pyarrow as pa
from fray.types import EnvironmentConfig, GpuConfig, ResourceConfig, create_environment
from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_file
from zephyr.writers import write_parquet_file

from marin.datakit.decon_keys import (
    LOCATOR_COLUMNS,
    MAX_DOC_CHARS,
    WINDOW,
    Candidate,
    KeyStream,
    KeyTable,
    Windows,
    candidate_windows,
    key_streams,
    keys_table,
    long_record_pairs,
    record_hit_indices,
    record_pairs,
    template_bounds,
    text_stream,
)
from marin.datakit.decon_tables import parquet_files, read_frame, read_table, write_frame, write_table
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import DatakitArtifactPath
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec

logger = logging.getLogger(__name__)

DECON_ATTRIBUTES_VERSION = 6
# Bump when the key choice changes; the keys, and every scan, score and mark built on them, re-key.
KEYS_VERSION = 1
# Bump when the scan's pairs change (key matching, readings, excerpts).
SCAN_VERSION = 1
KEYS_FILENAME = "keys.parquet"
LOCATORS_FILENAME = "locators.parquet"
_WINDOWS_FILENAME = "windows.npz"
# Pair files that one score task reads and scores together.
SCORE_GROUP_FILES = 16

# The item table: one row per eval record, read by the keys and the scorer.
ITEM_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("eval", pa.string()),
        ("suite", pa.string()),
        ("question", pa.string()),
        ("choices", pa.list_(pa.string())),
        ("answer", pa.string()),
        ("prompt", pa.string()),
        ("passage", pa.string()),
        ("choices_mode", pa.string()),
        ("reason", pa.string()),
    ]
)
ATTRIBUTE_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("partition_id", pa.int64()),
        ("contaminated", pa.bool_()),
        ("max_score", pa.float64()),
        ("matched_items", pa.list_(pa.string())),
    ]
)
FLAGGED_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        ("max_score", pa.float64()),
        ("matched_items", pa.list_(pa.string())),
        ("matched_evals", pa.list_(pa.string())),
    ]
)
_SCORE_COLUMNS = ["record", "item", "eval", "right", "score"]


class DeconKeys(BaseModel):
    """Outcome of :func:`build_keys`: each item stream's key and the locator table the scan reads.

    Attributes:
        keys_path: ``keys.parquet``, one row per key stream (``decon_keys.KEY_COLUMNS``).
        locators_path: ``locators.parquet``, one row per keyed stream (``decon_keys.LOCATOR_COLUMNS``).
        items_dir: The item table the keys were built from; the scorer reads the item texts from it.
        width: Key window width in stream characters.
        counters: Item, stream and reference-shard counts.
    """

    keys_path: DatakitArtifactPath
    locators_path: DatakitArtifactPath
    items_dir: DatakitArtifactPath
    width: int
    counters: dict[str, int | float]


class DeconScan(BaseModel):
    """Outcome of :func:`scan_source`: the candidate pairs of one source.

    Attributes:
        pairs_dir: One file of ``decon_keys.Candidate`` rows per input shard with pairs, with the shard's name.
        pair_files: The names of the input shards with pairs.
        counters: ``decon/records``, ``decon/candidate_records``, ``decon/pairs`` and ``decon/long_records``.
    """

    pairs_dir: DatakitArtifactPath
    pair_files: list[str]
    counters: dict[str, int | float]


class DeconScores(BaseModel):
    """Outcome of :func:`score_pairs`: one file of scored pairs (``record``, ``item``, ``eval``, ``right``,
    ``score``) per pair file of the scan, with its name."""

    scores_dir: DatakitArtifactPath
    counters: dict[str, int | float]


class DeconAttributes(BaseModel):
    """Outcome of :func:`mark_source`: a co-partitioned attributes dataset (see the module docstring).

    Attributes:
        main_output_dir: Directory of the attribute files, one per input shard with the input's file name.
        flagged_output_dir: Directory of the flagged sample, one file per input shard that has a flag.
        num_partitions: Number of output partitions; matches the source.
        counters: ``decon/contaminated`` and ``decon/clean`` records.
    """

    version: str = f"v{DECON_ATTRIBUTES_VERSION}"
    main_output_dir: DatakitArtifactPath
    flagged_output_dir: DatakitArtifactPath
    num_partitions: int
    counters: dict[str, int | float]


# --- Keys -------------------------------------------------------------------------------------------------


class ItemStreams(NamedTuple):
    """One entry per item and key stream (``decon_keys.key_streams``), in parallel lists."""

    ids: list[str]
    kinds: list[str]
    streams: list[str]
    evals: list[str]


def item_streams(items_dir: str, width: int) -> ItemStreams:
    """Every key stream of every item of the item table."""
    ids, kinds, streams, evals = [], [], [], []
    for path in parquet_files(items_dir):
        for row in read_table(path, ["id", "eval", "question", "choices", "passage"]).to_pylist():
            for kind, stream in key_streams(row["question"], row["choices"] or [], row["passage"] or "", width).items():
                ids.append(row["id"])
                kinds.append(str(kind))
                streams.append(stream)
                evals.append(row["eval"])
    return ItemStreams(ids, kinds, streams, evals)


def shared_holders(windows: Windows) -> np.ndarray:
    """For each candidate window, how many items hold it: equally rare windows go to the one that the fewest other
    items share (``keys_table``'s tie order), which keeps keys off shared prompt text such as a common stem."""
    hashes, owners, _, ids, _ = windows
    _, item_numbers = np.unique(ids, return_inverse=True)
    items = item_numbers[owners].astype(np.int64)
    pairs = np.unique(np.stack([hashes.view(np.int64), items], axis=1), axis=0)
    distinct, holders = np.unique(pairs[:, 0], return_counts=True)
    return holders[np.searchsorted(distinct, hashes.view(np.int64))]


def located_questions(questions: dict[str, str], evals: dict[str, str], width: int = WINDOW) -> dict[str, str]:
    """Each item's question stream without the prefix and suffix that all of its eval's questions share (the
    prompt text, such as "Question: ... Answer:"), so that the scan finds a copy written without that text."""
    ids = list(questions)
    streams = [text_stream(questions[item]) for item in ids]
    bounds = template_bounds(streams, [evals[item] for item in ids], width)
    return {item: stream[low:high] for item, stream, (low, high) in zip(ids, streams, bounds, strict=True)}


def _save_windows(windows: Windows, path: str) -> None:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        hashes=windows.hashes,
        owners=windows.owners,
        offsets=windows.offsets,
        ids=windows.ids,
        kinds=windows.kinds,
    )
    StoragePath(path).write_bytes(buffer.getvalue())


@functools.lru_cache(maxsize=1)
def _load_windows(path: str) -> Windows:
    data = np.load(io.BytesIO(StoragePath(path).read_bytes()), allow_pickle=True)
    return Windows(data["hashes"], data["owners"], data["offsets"], data["ids"], data["kinds"])


def _make_count(windows_path: str, counts_dir: str, width: int) -> Callable[[Iterator[str], ShardInfo], Iterator[dict]]:
    """A ``map_shard`` function: in how many records of its reference shards each candidate window occurs."""

    def count(paths: Iterator[str], shard: ShardInfo) -> Iterator[dict]:
        target = StoragePath(prefix_join(counts_dir, f"part-{shard.shard_idx:05d}.npz"))
        if target.exists():
            return
        unique = np.unique(_load_windows(windows_path).hashes)
        found = np.zeros(len(unique), dtype=np.uint32)
        records = 0
        for path in paths:
            for record in load_file(InputFileSpec(path=path, columns=["text"])):
                text = str(record.get("text") or "")
                if len(text) > MAX_DOC_CHARS:
                    continue
                records += 1
                found[record_hit_indices(text, unique, width)] += 1
        nonzero = np.nonzero(found)[0]
        buffer = io.BytesIO()
        np.savez(buffer, hashes=unique[nonzero], counts=found[nonzero])
        target.write_bytes(buffer.getvalue())
        yield {"records": records}

    return count


def _window_counts(windows: Windows, counts_dir: str) -> np.ndarray:
    """The summed reference count of every candidate window."""
    parts_hashes, parts_counts = [], []
    for path in sorted(str(part) for part in StoragePath(prefix_join(counts_dir, "*.npz")).glob()):
        part = np.load(io.BytesIO(StoragePath(path).read_bytes()))
        parts_hashes.append(part["hashes"])
        parts_counts.append(part["counts"].astype(np.int64))
    if not parts_hashes or not sum(len(part) for part in parts_hashes):
        return np.zeros(len(windows.hashes), dtype=np.int64)
    seen, inverse = np.unique(np.concatenate(parts_hashes), return_inverse=True)
    totals = np.bincount(inverse, weights=np.concatenate(parts_counts)).astype(np.int64)
    slot = np.minimum(np.searchsorted(seen, windows.hashes), len(seen) - 1)
    return np.where(seen[slot] == windows.hashes, totals[slot], 0)


def locator_table(keys: pa.Table, items_dir: str, width: int) -> pa.Table:
    """One row per keyed stream of *keys*: its key-1 hash, the key window and its offset, and the item's question
    stream without its eval's shared prompt text, so a scan can cut the excerpt around the question (or, when the
    question is not found whole, around the key's first match)."""
    rows = {
        path: read_frame(path, ["id", "eval", "question", "choices", "passage"]) for path in parquet_files(items_dir)
    }
    frame = pd.concat(rows.values(), ignore_index=True)
    texts = {row.id: (row.question, list(row.choices), row.passage or "") for row in frame.itertuples(index=False)}
    evals = dict(zip(frame["id"], frame["eval"], strict=True))
    question_of = located_questions({item: text[0] for item, text in texts.items()}, evals, width)
    keyed = keys.select(["id", "kind", "key1_hash", "key1_offset"]).to_pandas()
    keyed = keyed[keyed["key1_hash"] != 0]
    columns: dict[str, list] = {name: [] for name in LOCATOR_COLUMNS}
    for item, kind, key1_hash, offset in zip(
        keyed["id"], keyed["kind"], keyed["key1_hash"], keyed["key1_offset"], strict=True
    ):
        question, choices, passage = texts[item]
        stream = key_streams(question, choices, passage, width)[KeyStream(kind)]
        values = (
            item,
            kind,
            evals[item],
            int(key1_hash),
            stream[offset : offset + width],
            int(offset),
            question_of[item],
            len(question),
        )
        for name, value in zip(LOCATOR_COLUMNS, values, strict=True):
            columns[name].append(value)
    return pa.table(
        {
            name: pa.array(values, pa.uint64()) if name == "key1_hash" else pa.array(values)
            for name, values in columns.items()
        }
    )


def _prepare_windows(items_dir: str, windows_path: str, width: int) -> dict[str, int]:
    """Write every item stream's candidate windows: those outside the prefix and suffix that all of its eval's
    streams of its kind share."""
    ids, kinds, streams, evals = item_streams(items_dir, width)
    groups = [f"{e}\t{k}" for e, k in zip(evals, kinds, strict=True)]
    hashes, owners, offsets = candidate_windows(streams, width, template_bounds(streams, groups, width))
    _save_windows(
        Windows(hashes, owners, offsets, np.array(ids, dtype=object), np.array(kinds, dtype=object)), windows_path
    )
    return {"items": len(set(ids)), "streams": len(ids)}


def _choose_keys(windows_path: str, counts_dir: str, items_dir: str, output_path: str, width: int) -> dict[str, int]:
    """Write each stream's keys from the summed window counts, and the locator table the scan reads."""
    candidates = _load_windows(windows_path)
    table = keys_table(candidates, _window_counts(candidates, counts_dir), width, shared_holders(candidates))
    write_table(table, prefix_join(output_path, KEYS_FILENAME))
    locators = locator_table(table, items_dir, width)
    write_table(locators, prefix_join(output_path, LOCATORS_FILENAME))
    return {"keyed_items": len(set(locators.column("item").to_pylist()))}


def _one_task(context: ZephyrContext, task: Callable[[], dict[str, int]], resources: ResourceConfig) -> dict[str, int]:
    """Run *task* as the only task of a pipeline, so that its memory is a worker's, not the driver's."""
    outcome = context.execute(
        Dataset.from_list([0]).map_shard(lambda _items, _shard: iter([task()])), map_task_resources=resources
    )
    return outcome.results[0]


def build_keys(
    *,
    items_dir: str,
    reference_shards: Sequence[str],
    output_path: str,
    width: int = WINDOW,
    worker_resources: ResourceConfig,
    max_workers: int,
    zephyr_context: ZephyrContext | None = None,
) -> DeconKeys:
    """Choose each item stream's key from its candidate windows' counts over *reference_shards*, and write the
    keys and the locator table under *output_path* (see the module docstring). Every phase runs on workers."""
    if not reference_shards:
        raise ValueError("build_keys needs reference shards")
    windows_path = prefix_join(output_path, _WINDOWS_FILENAME)
    counts_dir = prefix_join(output_path, "counts")
    context = zephyr_context or ZephyrContext(
        name="decon-keys", resources=worker_resources, max_workers=min(max_workers, len(reference_shards))
    )
    streams = _one_task(context, lambda: _prepare_windows(items_dir, windows_path, width), worker_resources)
    context.execute(
        Dataset.from_list(list(reference_shards)).map_shard(_make_count(windows_path, counts_dir, width)),
        map_task_resources=worker_resources,
    )
    keyed = _one_task(
        context, lambda: _choose_keys(windows_path, counts_dir, items_dir, output_path, width), worker_resources
    )
    logger.info("decon keys: %s, %s", streams, keyed)
    return DeconKeys(
        keys_path=prefix_join(output_path, KEYS_FILENAME),
        locators_path=prefix_join(output_path, LOCATORS_FILENAME),
        items_dir=items_dir,
        width=width,
        counters={**streams, **keyed, "reference_shards": len(reference_shards)},
    )


def reference_shards(normalized_dirs: Sequence[str], per_source: int, seed: int) -> list[str]:
    """*per_source* shards of each normalized directory, drawn with one random stream in the order given."""
    rng = random.Random(seed)
    picked = []
    for directory in normalized_dirs:
        shards = parquet_files(directory, "**/*.parquet")
        picked += rng.sample(shards, min(per_source, len(shards)))
    return picked


# --- Scoring ----------------------------------------------------------------------------------------------


class PairScorer(Protocol):
    """Scores pairs (columns ``item`` and ``right``). ``version`` names the scorer in the score step's identity;
    ``load`` returns, once per worker, the function from pairs to scores on *device* (``"cuda"`` or ``"cpu"``)."""

    @property
    def version(self) -> str: ...

    def load(self, items_dir: str, device: str) -> Callable[[pd.DataFrame], np.ndarray]: ...


@dataclass(frozen=True)
class ScoreWorkers:
    """The workers that score the pairs: one per *pairs_per_worker* pairs of a source, up to *max_workers*, of
    *resources*. A GPU in *resources* makes the scorer run on CUDA. The workers sync *extras* in place of the
    driver's environment (the gpu extra has the CUDA build of torch); ``()`` reuses the driver's."""

    resources: ResourceConfig
    max_workers: int
    pairs_per_worker: int
    extras: tuple[str, ...]

    @property
    def device(self) -> str:
        return "cuda" if isinstance(self.resources.device, GpuConfig) else "cpu"

    def environment(self) -> EnvironmentConfig | None:
        return create_environment(extras=list(self.extras)) if self.extras else None


@functools.lru_cache(maxsize=1)
def _loaded_scorer(scorer: PairScorer, items_dir: str, device: str) -> Callable[[pd.DataFrame], np.ndarray]:
    return scorer.load(items_dir, device)


def _make_score(
    scorer: PairScorer, items_dir: str, scores_dir: str, device: str
) -> Callable[[Iterator[list[str]], ShardInfo], Iterator[dict]]:
    """A ``map_shard`` function: score each group of pair files into ``<scores_dir>/<file name>``."""

    def score(groups: Iterator[list[str]], shard: ShardInfo) -> Iterator[dict]:
        for paths in groups:
            todo = [path for path in paths if not StoragePath(prefix_join(scores_dir, os.path.basename(path))).exists()]
            if not todo:
                continue
            frames = [read_frame(path) for path in todo]
            pairs = pd.concat(frames, ignore_index=True)
            pairs["score"] = _loaded_scorer(scorer, items_dir, device)(pairs)
            start = 0
            for path, frame in zip(todo, frames, strict=True):
                part = pairs.iloc[start : start + len(frame)]
                start += len(frame)
                write_frame(part[_SCORE_COLUMNS], prefix_join(scores_dir, os.path.basename(path)))
            counters.pipeline.update_counter("decon/scored_pairs", len(pairs))
            yield {"pairs": len(pairs)}

    return score


# --- Scan and mark ----------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _index(locators_path: str) -> tuple[KeyTable, pa.Table]:
    """The key-1 table over locator rows, and the locators; loaded once per worker process."""
    locators = read_table(locators_path)
    return KeyTable(locators.column("key1_hash").to_numpy(), np.arange(locators.num_rows)), locators


def _make_scan(locators_path: str, pairs_dir: str, stats_dir: str, width: int):
    """A ``map_shard`` function: each record's candidate pairs into ``<pairs_dir>/<file name>`` (only when it has
    some) and the shard's counts into ``<stats_dir>/<file name>.json``, its done marker."""

    def scan(paths: Iterator[str], shard: ShardInfo) -> Iterator[dict]:
        for path in paths:
            name = os.path.basename(path)
            done = StoragePath(prefix_join(stats_dir, f"{name}.json"))
            if done.exists():
                yield json.loads(done.read_text())
                continue
            table, locators = _index(locators_path)
            stats = {"file": name, "records": 0, "chars": 0, "candidate_records": 0, "pairs": 0, "long_records": 0}
            found: list[Candidate] = []
            for record in load_file(InputFileSpec(path=path, columns=["id", "text"])):
                text = str(record.get("text") or "")
                record_id = str(record["id"])
                stats["records"] += 1
                stats["chars"] += len(text)
                if len(text) > MAX_DOC_CHARS:
                    stats["long_records"] += 1
                    pairs = long_record_pairs(record_id, text, table, locators, width)
                else:
                    pairs = record_pairs(record_id, text, table, locators, width)
                stats["candidate_records"] += bool(pairs)
                found += pairs
            stats["pairs"] = len(found)
            if found:
                write_table(pa.Table.from_pylist([asdict(pair) for pair in found]), prefix_join(pairs_dir, name))
            counters.pipeline.update_counter("decon/candidate_records", stats["candidate_records"])
            counters.pipeline.update_counter("decon/pairs", stats["pairs"])
            done.write_text(json.dumps(stats))
            yield stats

    return scan


def _record_scores(
    scores_path: str, cutoff: float
) -> tuple[dict[str, tuple[float, str]], dict[str, list[tuple[str, str]]]]:
    """For one scores file: each record's best pair as (score, excerpt), and its pairs at or above *cutoff* as
    (item, eval)."""
    best: dict[str, tuple[float, str]] = {}
    matched: dict[str, list[tuple[str, str]]] = {}
    if not StoragePath(scores_path).exists():
        return best, matched
    columns = read_table(scores_path, _SCORE_COLUMNS).to_pydict()
    rows = zip(columns["record"], columns["item"], columns["eval"], columns["right"], columns["score"], strict=True)
    for record, item, eval_name, right, score in sorted(rows, key=lambda row: -float(row[4])):
        best.setdefault(str(record), (float(score), str(right)))
        hits = matched.setdefault(str(record), [])
        if float(score) >= cutoff and (str(item), str(eval_name)) not in hits:
            hits.append((str(item), str(eval_name)))
    return best, {record: hits for record, hits in matched.items() if hits}


def _mark_rows(
    source: str,
    partition: int,
    best: dict[str, tuple[float, str]],
    matched: dict[str, list[tuple[str, str]]],
    flagged: list[dict[str, Any]],
    flagged_sample_size: int,
    rng: random.Random,
) -> Iterator[dict[str, Any]]:
    """The attribute rows of *source* in input order; fills *flagged* with a reservoir sample of its
    contaminated records."""
    seen = 0
    for record in load_file(InputFileSpec(path=source, columns=["id"])):
        record_id = str(record["id"])
        score, right = best.get(record_id, (0.0, ""))
        hits = matched.get(record_id, [])
        items = [item for item, _ in hits]
        counters.pipeline.update_counter("decon/contaminated" if hits else "decon/clean", 1)
        if hits and flagged_sample_size:
            seen += 1
            row = {
                "id": record_id,
                "text": right,
                "max_score": score,
                "matched_items": items,
                "matched_evals": sorted({eval_name for _, eval_name in hits}),
            }
            if len(flagged) < flagged_sample_size:
                flagged.append(row)
            elif (slot := rng.randint(0, seen - 1)) < flagged_sample_size:
                flagged[slot] = row
        yield {
            "id": record_id,
            "partition_id": partition,
            "contaminated": bool(hits),
            "max_score": score,
            "matched_items": items,
        }


def _make_mark(scores_dir: str, output_dir: str, cutoff: float, flagged_sample_size: int, files: list[str]):
    """A ``map_shard`` function: one dense attribute file per input shard, in input order (see the module
    docstring), and the shard's flagged sample."""
    partition_of = {path: number for number, path in enumerate(files)}

    def mark(paths: Iterator[str], shard: ShardInfo) -> Iterator[dict]:
        rng = random.Random(shard.shard_idx)
        for path in paths:
            name = os.path.basename(path)
            best, matched = _record_scores(prefix_join(scores_dir, name), cutoff)
            flagged: list[dict[str, Any]] = []
            rows = _mark_rows(path, partition_of[path], best, matched, flagged, flagged_sample_size, rng)
            yield write_parquet_file(
                rows, output_path=prefix_join(output_dir, f"outputs/main/{name}"), schema=ATTRIBUTE_SCHEMA
            )
            if flagged:
                write_parquet_file(
                    iter(flagged),
                    output_path=prefix_join(output_dir, f"outputs/flagged_sample/{name}"),
                    schema=FLAGGED_SCHEMA,
                )

    return mark


def _shards(normalized: NormalizedData) -> list[str]:
    files = parquet_files(normalized.main_output_dir, "**/*.parquet")
    if not files:
        raise FileNotFoundError(f"no .parquet files under {normalized.main_output_dir}")
    names = [os.path.basename(path) for path in files]
    if len(set(names)) != len(names):
        raise ValueError(
            f"the shards under {normalized.main_output_dir} must have distinct file names; the outputs use them"
        )
    return files


def scan_source(
    *,
    normalized: NormalizedData,
    keys: DeconKeys,
    output_path: str,
    worker_resources: ResourceConfig,
    max_workers: int,
    zephyr_context: ZephyrContext | None = None,
) -> DeconScan:
    """The candidate pairs of every record of *normalized* (see the module docstring)."""
    files = _shards(normalized)
    pairs_dir = prefix_join(output_path, "pairs")
    context = zephyr_context or ZephyrContext(
        name="decon-scan", resources=worker_resources, max_workers=min(max_workers, len(files))
    )
    scan = context.execute(
        Dataset.from_list(files).map_shard(
            _make_scan(keys.locators_path, pairs_dir, prefix_join(output_path, "stats"), keys.width)
        ),
        map_task_resources=worker_resources,
    )
    return DeconScan(
        pairs_dir=pairs_dir,
        pair_files=sorted(stats["file"] for stats in scan.results if stats["pairs"]),
        counters={
            f"decon/{name}": sum(stats[name] for stats in scan.results)
            for name in ("records", "candidate_records", "pairs", "long_records")
        },
    )


def score_pairs(
    *, scan: DeconScan, keys: DeconKeys, scorer: PairScorer, score_workers: ScoreWorkers, output_path: str
) -> DeconScores:
    """Score every pair of *scan* on its own context of *score_workers*; fail unless every pair file is scored."""
    scores_dir = prefix_join(output_path, "scores")

    def unscored() -> list[str]:
        return [name for name in scan.pair_files if not StoragePath(prefix_join(scores_dir, name)).exists()]

    pending = [prefix_join(scan.pairs_dir, name) for name in unscored()]
    if pending:
        groups = [pending[i : i + SCORE_GROUP_FILES] for i in range(0, len(pending), SCORE_GROUP_FILES)]
        wanted = math.ceil(scan.counters["decon/pairs"] / score_workers.pairs_per_worker)
        ZephyrContext(
            name="decon-score",
            resources=score_workers.resources,
            worker_environment=score_workers.environment(),
            max_workers=max(min(score_workers.max_workers, len(groups), wanted), 1),
        ).execute(
            Dataset.from_list(groups).map_shard(_make_score(scorer, keys.items_dir, scores_dir, score_workers.device)),
            map_task_resources=score_workers.resources,
        )
    missing = unscored()
    if missing:
        raise RuntimeError(f"{len(missing)} pair files have no scores, such as {missing[:5]}")
    return DeconScores(scores_dir=scores_dir, counters={"decon/scored_pairs": scan.counters["decon/pairs"]})


def mark_source(
    *,
    normalized: NormalizedData,
    scores: DeconScores,
    cutoff: float,
    flagged_sample_size: int,
    output_path: str,
    worker_resources: ResourceConfig,
    max_workers: int,
    zephyr_context: ZephyrContext | None = None,
) -> DeconAttributes:
    """Mark the records of *normalized* whose best pair in *scores* reaches *cutoff* (see the module docstring)."""
    files = _shards(normalized)
    context = zephyr_context or ZephyrContext(
        name="decon-mark", resources=worker_resources, max_workers=min(max_workers, len(files))
    )
    marked = context.execute(
        Dataset.from_list(files).map_shard(
            _make_mark(scores.scores_dir, output_path, cutoff, flagged_sample_size, files)
        ),
        map_task_resources=worker_resources,
    )
    return DeconAttributes(
        main_output_dir=prefix_join(output_path, "outputs/main"),
        flagged_output_dir=prefix_join(output_path, "outputs/flagged_sample"),
        num_partitions=len(files),
        counters=dict(marked.counters),
    )


# --- Steps ------------------------------------------------------------------------------------------------


def decon_keys_step(
    *,
    name: str,
    items: StepSpec,
    sources: Mapping[str, StepSpec],
    shards_per_source: int,
    seed: int,
    worker_resources: ResourceConfig,
    max_workers: int,
    width: int = WINDOW,
    zephyr_context: ZephyrContext | None = None,
    output_path_prefix: str | None = None,
) -> StepSpec:
    """A StepSpec for :func:`build_keys` over the item table of *items* (a step whose output holds
    ``items/*.parquet``), with *shards_per_source* reference shards of each normalized source."""
    source_steps = [sources[source] for source in sorted(sources)]

    def run(output_path: str) -> DeconKeys:
        directories = [read_artifact(step.output_path, NormalizedData).main_output_dir for step in source_steps]
        return build_keys(
            items_dir=prefix_join(items.output_path, "items"),
            reference_shards=reference_shards(directories, shards_per_source, seed),
            output_path=output_path,
            width=width,
            worker_resources=worker_resources,
            max_workers=max_workers,
            zephyr_context=zephyr_context,
        )

    return StepSpec(
        name=name,
        fn=run,
        deps=[items, *source_steps],
        hash_attrs={"keys_version": KEYS_VERSION, "shards_per_source": shards_per_source, "seed": seed, "width": width},
        output_path_prefix=output_path_prefix,
    )


def decon_scan_step(
    *,
    name: str,
    normalized: StepSpec,
    keys: StepSpec,
    worker_resources: ResourceConfig,
    max_workers: int,
    zephyr_context: ZephyrContext | None = None,
    output_path_prefix: str | None = None,
) -> StepSpec:
    """A StepSpec for :func:`scan_source` over the output of *normalized*, with the keys of *keys*."""
    return StepSpec(
        name=name,
        fn=lambda output_path: scan_source(
            normalized=read_artifact(normalized.output_path, NormalizedData),
            keys=read_artifact(keys.output_path, DeconKeys),
            output_path=output_path,
            worker_resources=worker_resources,
            max_workers=max_workers,
            zephyr_context=zephyr_context,
        ),
        deps=[normalized, keys],
        hash_attrs={"scan_version": SCAN_VERSION},
        output_path_prefix=output_path_prefix,
    )


def decon_score_step(
    *,
    name: str,
    scan: StepSpec,
    keys: StepSpec,
    scorer: PairScorer,
    score_workers: ScoreWorkers,
    output_path_prefix: str | None = None,
) -> StepSpec:
    """A StepSpec for :func:`score_pairs` over the pairs of *scan*."""
    return StepSpec(
        name=name,
        fn=lambda output_path: score_pairs(
            scan=read_artifact(scan.output_path, DeconScan),
            keys=read_artifact(keys.output_path, DeconKeys),
            scorer=scorer,
            score_workers=score_workers,
            output_path=output_path,
        ),
        deps=[scan, keys],
        hash_attrs={"scorer": scorer.version},
        output_path_prefix=output_path_prefix,
    )


def decon_mark_step(
    *,
    name: str,
    normalized: StepSpec,
    scores: StepSpec,
    cutoff: float,
    flagged_sample_size: int,
    worker_resources: ResourceConfig,
    max_workers: int,
    zephyr_context: ZephyrContext | None = None,
    output_path_prefix: str | None = None,
) -> StepSpec:
    """A StepSpec for :func:`mark_source` over the output of *normalized*, with the scores of *scores*."""
    return StepSpec(
        name=name,
        fn=lambda output_path: mark_source(
            normalized=read_artifact(normalized.output_path, NormalizedData),
            scores=read_artifact(scores.output_path, DeconScores),
            cutoff=cutoff,
            flagged_sample_size=flagged_sample_size,
            output_path=output_path,
            worker_resources=worker_resources,
            max_workers=max_workers,
            zephyr_context=zephyr_context,
        ),
        deps=[normalized, scores],
        hash_attrs={
            "cutoff": cutoff,
            "flagged_sample_size": flagged_sample_size,
            "attribute_schema_version": DECON_ATTRIBUTES_VERSION,
        },
        output_path_prefix=output_path_prefix,
    )
