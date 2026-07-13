# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Precision analysis for the all-sources decon bloom.

For a sampled set of flagged corpus docs:

1. Pull ``(id, attributes.matched_hashes)`` from the decon output parquet.
2. Resolve each ``matched_hash`` to ``eval_id`` via the combined bloom's
   ``_bloom/eval_hash_index.parquet`` sidecar.
3. Resolve each ``eval_id`` to the originating eval text by reading the
   parquet eval corpus.
4. Resolve the corpus ``id`` back to the source text via the normalized
   corpus (`marin.datakit.sources.all_sources()[name].normalized` →
   ``main_output_dir``).
5. For each ``(corpus_text, eval_text, matched_hash)``, ask Claude
   whether the corpus doc substantively contains the eval text.

Outputs a TSV ``precision_<source>.tsv`` with columns
``corpus_id, eval_id, n_matched_hashes, max_overlap, verdict, rationale``
plus a precision summary on stdout.

Usage (after all_sources_decon.py has run):

    ANTHROPIC_API_KEY=... uv run python \\
        experiments/datakit/decontam/ops/precision_analysis.py \\
        --decon-output gs://.../datakit/decon/finepdfs/  \\
        --source-name finepdfs --sample-size 100

The decon output path is the ``output_path_prefix`` used in
``all_sources_decon.py`` joined with the source name; ``--source-name``
selects the matching :func:`marin.datakit.sources.all_sources` entry so
we can resolve the normalized-data location for corpus-text lookup.
"""

import argparse
import json
import logging
import os
import random
import re
from collections.abc import Iterator
from dataclasses import dataclass

import pyarrow.parquet as pq
import yaml
from marin.datakit.normalize import NormalizedData
from marin.datakit.sources import all_sources
from marin.execution.artifact import read_artifact
from pyarrow import fs as pa_fs
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging

from experiments.datakit.decontam.all_sources_decon import EVAL_ROOT

logger = logging.getLogger(__name__)

JUDGE_MODEL = "claude-sonnet-4-6"
JUDGE_PROMPT = """\
You are auditing a decontamination pipeline. We flagged a training-corpus
document as overlapping with an evaluation set, because at least one
13-gram in the corpus doc collides with the eval text.

Decide whether the flag is correct: does the corpus document
substantively contain content from the eval text (a near-verbatim copy,
a translation, a paraphrase, or a clearly-derived passage), or is the
overlap coincidental (common phrasing, boilerplate, an unrelated topic
that happens to share a phrase)?

Reply with a single JSON object:

  {{"verdict": "true_positive" | "false_positive", "rationale": "<one short sentence>"}}

EVAL TEXT (from {eval_id}):
{eval_text}

CORPUS DOCUMENT (id={corpus_id}; first {corpus_chars} chars):
{corpus_text}
"""

MAX_EVAL_CHARS = 4000
MAX_CORPUS_CHARS = 12000


@dataclass
class FlaggedRecord:
    corpus_id: str
    partition_id: int
    matched_hashes: list[int]
    max_overlap: float


def _read_anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    # Fall back to .marin.yaml (the project ships it with the key for CI).
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "..", "..", "..", ".marin.yaml")
    if os.path.exists(cand):
        with open(cand) as f:
            data = yaml.safe_load(f) or {}
        env = data.get("env", {})
        key = env.get("ANTHROPIC_API_KEY")
        if key:
            return key
    raise RuntimeError("ANTHROPIC_API_KEY not set and not found in .marin.yaml")


def _list_parquet_recursive(root: str) -> list[str]:
    gcs = pa_fs.GcsFileSystem()
    bare = root.removeprefix("gs://").rstrip("/")
    entries = gcs.get_file_info(pa_fs.FileSelector(bare, recursive=True))
    return sorted(f"gs://{e.path}" for e in entries if e.path.endswith(".parquet"))


def _iter_flagged_records(decon_output: str) -> Iterator[FlaggedRecord]:
    """Walk the decon output parquet files and yield contaminated rows."""
    files = _list_parquet_recursive(decon_output)
    logger.info("decon output: %d parquet shards under %s", len(files), decon_output)
    gcs = pa_fs.GcsFileSystem()
    for f in files:
        bare = f.removeprefix("gs://")
        pf = pq.ParquetFile(bare, filesystem=gcs)
        for batch in pf.iter_batches(columns=["id", "partition_id", "attributes"], batch_size=2048):
            ids = batch.column("id").to_pylist()
            partition_ids = batch.column("partition_id").to_pylist()
            attrs = batch.column("attributes").to_pylist()
            for i, p, a in zip(ids, partition_ids, attrs, strict=True):
                if a and a.get("contaminated"):
                    yield FlaggedRecord(
                        corpus_id=str(i),
                        partition_id=int(p),
                        matched_hashes=list(a.get("matched_hashes") or []),
                        max_overlap=float(a.get("max_overlap") or 0.0),
                    )


def _load_eval_hash_index(bloom_dir: str) -> dict[int, list[str]]:
    """``hash → list[eval_id]`` from the combined bloom's sidecar."""
    sidecar = bloom_dir.rstrip("/") + "/_bloom/eval_hash_index.parquet"
    logger.info("loading eval hash index from %s", sidecar)
    with StoragePath(sidecar).open("rb") as f:
        table = pq.read_table(f)
    by_hash: dict[int, list[str]] = {}
    for h, eid in zip(table.column("hash").to_pylist(), table.column("eval_id").to_pylist(), strict=True):
        by_hash.setdefault(int(h), []).append(str(eid))
    logger.info("hash index: %d unique hashes -> %d (hash, eval_id) rows", len(by_hash), table.num_rows)
    return by_hash


def _eval_id_to_text(eval_root: str, keep_ids: set[str]) -> dict[str, tuple[str, str]]:
    """Eval ``id → (text, parquet_path)``, restricted to ``keep_ids``.

    Streams the eval corpus parquet tree and only retains records whose
    ``id`` is requested. Early-exits when all ids are found. Avoids
    materializing all 1.8M eval records when we only need a handful.
    """
    files = _list_parquet_recursive(eval_root)
    logger.info(
        "eval corpus: %d parquet files under %s (filtering to %d ids)",
        len(files),
        eval_root,
        len(keep_ids),
    )
    gcs = pa_fs.GcsFileSystem()
    out: dict[str, tuple[str, str]] = {}
    for f_idx, f in enumerate(files, 1):
        bare = f.removeprefix("gs://")
        pf = pq.ParquetFile(bare, filesystem=gcs)
        for batch in pf.iter_batches(columns=["id", "text"], batch_size=2048):
            ids = batch.column("id").to_pylist()
            texts = batch.column("text").to_pylist()
            for eid, t in zip(ids, texts, strict=True):
                sid = str(eid)
                if sid in keep_ids:
                    out[sid] = (str(t), f)
        if len(out) >= len(keep_ids):
            logger.info("found all %d eval ids after %d/%d files", len(out), f_idx, len(files))
            return out
        if f_idx % 200 == 0:
            logger.info(
                "scanned %d/%d eval files, found %d/%d ids",
                f_idx,
                len(files),
                len(out),
                len(keep_ids),
            )
    logger.info("eval id index: %d / %d ids found", len(out), len(keep_ids))
    return out


def _corpus_id_to_text(source_name: str, sample: list[FlaggedRecord]) -> dict[str, str]:
    """Build ``corpus_id → text`` by opening only the parquet shards that contain ``sample`` ids.

    The decon output stores ``partition_id`` per record, which mirrors the
    source's ``part-NNNNN-of-NNNNN`` shard index (`marin.datakit.decon`
    contract). So we can index directly into the source corpus and skip
    the brute-force shard-scan that costs ~95 min on a 19 GB / 76-shard
    source like cp/biodiversity.
    """
    sources = all_sources()
    if source_name not in sources:
        raise KeyError(f"unknown source {source_name!r}; known: {sorted(sources)}")
    nd: NormalizedData = read_artifact(sources[source_name].normalized.output_path, NormalizedData)
    all_files = _list_parquet_recursive(nd.main_output_dir)
    n_total_shards = len(all_files)

    keep_ids = {r.corpus_id for r in sample}
    needed_partitions: set[int] = {r.partition_id for r in sample}
    files_by_partition: dict[int, str] = {}
    for f in all_files:
        # part-NNNNN-of-NNNNN.parquet → partition idx is the first NNNNN
        name = f.rsplit("/", 1)[-1]
        try:
            part_idx = int(name.split("-")[1])
        except (ValueError, IndexError):
            continue
        files_by_partition[part_idx] = f

    target_files = [files_by_partition[p] for p in sorted(needed_partitions) if p in files_by_partition]
    logger.info(
        "source %s: opening %d targeted shards (of %d total) for %d ids",
        source_name,
        len(target_files),
        n_total_shards,
        len(keep_ids),
    )
    gcs = pa_fs.GcsFileSystem()
    out: dict[str, str] = {}
    for f_idx, f in enumerate(target_files, 1):
        bare = f.removeprefix("gs://")
        pf = pq.ParquetFile(bare, filesystem=gcs)
        for batch in pf.iter_batches(columns=["id", "text"], batch_size=2048):
            ids = batch.column("id").to_pylist()
            texts = batch.column("text").to_pylist()
            for i, t in zip(ids, texts, strict=True):
                sid = str(i)
                if sid in keep_ids:
                    out[sid] = str(t)
        if f_idx % 10 == 0:
            logger.info(
                "scanned %d/%d targeted shards, found %d/%d ids",
                f_idx,
                len(target_files),
                len(out),
                len(keep_ids),
            )
        if len(out) >= len(keep_ids):
            logger.info("found all %d ids after %d/%d shards", len(out), f_idx, len(target_files))
            return out
    logger.info("corpus id index: %d / %d ids found", len(out), len(keep_ids))
    return out


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "...<truncated>"


def _judge(client, eval_id: str, eval_text: str, corpus_id: str, corpus_text: str) -> tuple[str, str]:
    """Return (verdict, rationale) from Claude.

    ``verdict`` is normalized to ``true_positive`` | ``false_positive`` | ``error``.
    """
    prompt = JUDGE_PROMPT.format(
        eval_id=eval_id,
        eval_text=_truncate(eval_text, MAX_EVAL_CHARS),
        corpus_id=corpus_id,
        corpus_chars=MAX_CORPUS_CHARS,
        corpus_text=_truncate(corpus_text, MAX_CORPUS_CHARS),
    )
    try:
        resp = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        return ("error", f"api: {type(exc).__name__}: {exc}")
    txt = resp.content[0].text if resp.content else ""
    m = re.search(r"\{[^{}]*\"verdict\"[^{}]*\}", txt, re.DOTALL)
    if not m:
        return ("error", f"unparseable: {txt[:200]}")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return ("error", f"json: {m.group(0)[:200]}")
    v = str(obj.get("verdict") or "").strip()
    if v not in ("true_positive", "false_positive"):
        return ("error", f"bad verdict: {v!r}")
    return (v, str(obj.get("rationale") or ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decon-output", required=True, help="GCS dir with decon parquet shards")
    parser.add_argument("--source-name", required=True, help="Key in marin.datakit.sources.all_sources()")
    parser.add_argument("--bloom-dir", required=True, help="GCS dir containing _bloom/eval_hash_index.parquet")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default=None, help="Output TSV path (default: precision_<source>.tsv)")
    args = parser.parse_args()

    configure_logging(logging.INFO)
    safe_source = args.source_name.replace("/", "_")
    report_path = args.report or f"precision_{safe_source}.tsv"

    # Stage 1: collect flagged records.
    flagged = list(_iter_flagged_records(args.decon_output))
    logger.info("flagged records in source %s: %d", args.source_name, len(flagged))
    if not flagged:
        print(f"no flagged records in {args.decon_output} -- nothing to judge")
        return
    rng = random.Random(args.seed)
    sample = rng.sample(flagged, min(args.sample_size, len(flagged)))

    # Stage 2: build the three lookup tables. Load hash index in full (small
    # enough), then derive the needed eval_id set for stream-filtering both
    # the eval corpus and the source corpus reads.
    hash_to_eval_ids = _load_eval_hash_index(args.bloom_dir)
    needed_eval_ids: set[str] = set()
    for r in sample:
        for h in r.matched_hashes:
            eids = hash_to_eval_ids.get(int(h))
            if eids:
                needed_eval_ids.add(eids[0])
                break
    eval_id_to_text = _eval_id_to_text(EVAL_ROOT, needed_eval_ids)
    corpus_id_to_text = _corpus_id_to_text(args.source_name, sample)

    # Stage 3: judge.
    from anthropic import Anthropic  # noqa: PLC0415  # optional dep: anthropic

    client = Anthropic(api_key=_read_anthropic_key())

    n_tp = n_fp = n_err = 0
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("corpus_id\teval_id\tn_matched_hashes\tmax_overlap\tverdict\trationale\n")
        for i, rec in enumerate(sample, 1):
            corpus_text = corpus_id_to_text.get(rec.corpus_id, "")
            if not corpus_text:
                f.write(
                    f"{rec.corpus_id}\t\t{len(rec.matched_hashes)}\t{rec.max_overlap:.3f}\terror\tcorpus text missing\n"
                )
                n_err += 1
                continue
            # Pick the first hash that resolves; for now we don't enumerate all eval matches.
            eval_id = None
            for h in rec.matched_hashes:
                eids = hash_to_eval_ids.get(int(h))
                if eids:
                    eval_id = eids[0]
                    break
            if eval_id is None or eval_id not in eval_id_to_text:
                f.write(
                    f"{rec.corpus_id}\t{eval_id or ''}\t{len(rec.matched_hashes)}\t"
                    f"{rec.max_overlap:.3f}\terror\teval text missing\n"
                )
                n_err += 1
                continue
            eval_text, _eval_path = eval_id_to_text[eval_id]
            verdict, rationale = _judge(client, eval_id, eval_text, rec.corpus_id, corpus_text)
            if verdict == "true_positive":
                n_tp += 1
            elif verdict == "false_positive":
                n_fp += 1
            else:
                n_err += 1
            f.write(
                f"{rec.corpus_id}\t{eval_id}\t{len(rec.matched_hashes)}\t{rec.max_overlap:.3f}\t{verdict}\t"
                f"{rationale.replace(chr(9), ' ').replace(chr(10), ' ')}\n"
            )
            if i % 25 == 0:
                logger.info("judged %d/%d (tp=%d fp=%d err=%d)", i, len(sample), n_tp, n_fp, n_err)

    n_judged = n_tp + n_fp
    precision = n_tp / n_judged if n_judged else 0.0
    print()
    print(f"summary: tp={n_tp} fp={n_fp} err={n_err} of {len(sample)} samples")
    print(f"precision (excluding errors): {precision:.3f}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
