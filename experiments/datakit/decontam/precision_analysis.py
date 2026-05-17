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
        experiments/datakit/decontam/precision_analysis.py \\
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
from marin.execution.artifact import Artifact
from pyarrow import fs as pa_fs
from rigging.filesystem import url_to_fs
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
        for batch in pf.iter_batches(columns=["id", "attributes"], batch_size=2048):
            ids = batch.column("id").to_pylist()
            attrs = batch.column("attributes").to_pylist()
            for i, a in zip(ids, attrs, strict=True):
                if a and a.get("contaminated"):
                    yield FlaggedRecord(
                        corpus_id=str(i),
                        matched_hashes=list(a.get("matched_hashes") or []),
                        max_overlap=float(a.get("max_overlap") or 0.0),
                    )


def _load_eval_hash_index(bloom_dir: str) -> dict[int, list[str]]:
    """``hash → list[eval_id]`` from the combined bloom's sidecar."""
    sidecar = bloom_dir.rstrip("/") + "/_bloom/eval_hash_index.parquet"
    logger.info("loading eval hash index from %s", sidecar)
    fs_, path = url_to_fs(sidecar)
    with fs_.open(path, "rb") as f:
        table = pq.read_table(f)
    by_hash: dict[int, list[str]] = {}
    for h, eid in zip(table.column("hash").to_pylist(), table.column("eval_id").to_pylist(), strict=True):
        by_hash.setdefault(int(h), []).append(str(eid))
    logger.info("hash index: %d unique hashes -> %d (hash, eval_id) rows", len(by_hash), table.num_rows)
    return by_hash


def _eval_id_to_text(eval_root: str) -> dict[str, tuple[str, str]]:
    """Eval ``id → (text, parquet_path)`` across the whole eval corpus."""
    files = _list_parquet_recursive(eval_root)
    logger.info("eval corpus: %d parquet files under %s", len(files), eval_root)
    gcs = pa_fs.GcsFileSystem()
    out: dict[str, tuple[str, str]] = {}
    for f in files:
        bare = f.removeprefix("gs://")
        pf = pq.ParquetFile(bare, filesystem=gcs)
        for batch in pf.iter_batches(columns=["id", "text"], batch_size=2048):
            ids = batch.column("id").to_pylist()
            texts = batch.column("text").to_pylist()
            for eid, t in zip(ids, texts, strict=True):
                out[str(eid)] = (str(t), f)
    logger.info("eval id index: %d records", len(out))
    return out


def _corpus_id_to_text(source_name: str) -> dict[str, str]:
    """Build ``corpus_id → text`` from the normalized source.

    Loads the source's NormalizedData artifact to find ``main_output_dir``,
    then reads every parquet shard's ``id`` + ``text`` columns. For huge
    corpora this is memory-heavy; callers should restrict via a SAMPLE_IDS
    set if needed. For now this is a single shot for the precision analysis
    of the sampled records only -- we filter in-loop.
    """
    sources = all_sources()
    if source_name not in sources:
        raise KeyError(f"unknown source {source_name!r}; known: {sorted(sources)}")
    nd: NormalizedData = Artifact.from_path(sources[source_name].normalized, NormalizedData)
    files = _list_parquet_recursive(nd.main_output_dir)
    logger.info("source %s: %d parquet shards under %s", source_name, len(files), nd.main_output_dir)
    gcs = pa_fs.GcsFileSystem()
    out: dict[str, str] = {}
    for f in files:
        bare = f.removeprefix("gs://")
        pf = pq.ParquetFile(bare, filesystem=gcs)
        for batch in pf.iter_batches(columns=["id", "text"], batch_size=2048):
            ids = batch.column("id").to_pylist()
            texts = batch.column("text").to_pylist()
            for i, t in zip(ids, texts, strict=True):
                out[str(i)] = str(t)
    logger.info("corpus id index: %d records", len(out))
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
    report_path = args.report or f"precision_{args.source_name}.tsv"

    # Stage 1: collect flagged records.
    flagged = list(_iter_flagged_records(args.decon_output))
    logger.info("flagged records in source %s: %d", args.source_name, len(flagged))
    if not flagged:
        print(f"no flagged records in {args.decon_output} -- nothing to judge")
        return
    rng = random.Random(args.seed)
    sample = rng.sample(flagged, min(args.sample_size, len(flagged)))

    # Stage 2: build the three lookup tables.
    hash_to_eval_ids = _load_eval_hash_index(args.bloom_dir)
    eval_id_to_text = _eval_id_to_text(EVAL_ROOT)
    sample_ids = {r.corpus_id for r in sample}
    corpus_id_to_text_full = _corpus_id_to_text(args.source_name)
    corpus_id_to_text = {k: v for k, v in corpus_id_to_text_full.items() if k in sample_ids}

    # Stage 3: judge.
    from anthropic import Anthropic

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
