# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fasttext quality-classifier baseline on the same corpus slice.

Runs the deployed-style fasttext inference (the path the fast-transformer replaced) on a
CPU Zephyr fleet so the throughput / $ comparison is apples-to-apples: identical doc set,
identical sharding. Reports docs/sec and chars/sec (fasttext's natural units).
"""

import argparse
import functools
import json
import logging
import os
import posixpath
import tempfile
import time
from collections.abc import Iterator

import fasttext
from fray.cluster import ResourceConfig
from rigging.filesystem import StoragePath, open_url
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.readers import DEFAULT_FILE_PATH_COLUMN, load_file
from zephyr.runners import InlineRunner
from zephyr.writers import ThreadedBatchWriter, write_parquet_file

from experiments.datakit.cluster.quality.fast_transformer.tpu_bench.common import (
    DEFAULT_CORPUS,
    FASTTEXT_MODEL,
    accumulate,
    resolve_dataset_path,
    write_result_json,
)

logger = logging.getLogger(__name__)

PREDICT_BATCH = 1000
MAX_TEXT_CHARS = 4000  # the deployed v0 fasttext cap (~1k tokens)


@functools.cache
def _load_model(model_path: str):
    fd, local = tempfile.mkstemp(suffix=".bin")
    with os.fdopen(fd, "wb") as out, open_url(model_path, "rb") as fh:
        out.write(fh.read())
    logger.info("loaded fasttext model from %s", model_path)
    return fasttext.load_model(local)


def _normalize(text: str) -> str:
    """fasttext.predict rejects literal newlines; strip + cap like the deployed path."""
    return (text or "")[:MAX_TEXT_CHARS].replace("\n", " ").replace("\r", " ")


def _writer(output_path: str, model_path: str):
    def writer(records: Iterator[dict], shard: ShardInfo) -> Iterator[dict]:
        model = _load_model(model_path)
        records = iter(records)
        first = next(records, None)
        if first is None:
            return
        shard_file = posixpath.basename(first[DEFAULT_FILE_PATH_COLUMN])
        out_file = f"{output_path.rstrip('/')}/outputs/main/{shard_file}"
        timing: dict[str, float] = {}
        result: dict = {}
        n_docs = n_chars = 0

        def _sink(items):
            result.update(write_parquet_file(items, output_path=out_file))

        with ThreadedBatchWriter(_sink) as w:
            batch: list[dict] = []

            def flush(batch):
                nonlocal n_docs, n_chars
                if not batch:
                    return
                texts = [_normalize(r.get("text") or "") for r in batch]
                with accumulate(timing, "predict"):
                    labels, probs = model.predict(texts, k=1)
                for r, lab, pr in zip(batch, labels, probs, strict=True):
                    w.submit({"id": r["id"], "score": float(pr[0]), "label": lab[0]})
                n_docs += len(batch)
                n_chars += sum(len(t) for t in texts)

            for r in (first, *records):
                batch.append(r)
                if len(batch) >= PREDICT_BATCH:
                    flush(batch)
                    batch = []
            flush(batch)

        counters.pipeline.update_counter("ft2/docs", n_docs)
        counters.pipeline.update_counter("ft2/chars", n_chars)
        counters.pipeline.update_counter("ft2/predict_ms", int(timing.get("predict", 0.0) * 1000))
        yield {"shard_file": shard_file, "docs": n_docs, **result}

    return writer


def run(
    *,
    corpus_glob: str | None,
    model_path: str | None,
    output_path: str,
    max_files: int,
    max_workers: int,
    cpu: int,
    result_json: str | None,
):
    corpus_glob = resolve_dataset_path(corpus_glob or DEFAULT_CORPUS)
    model_path = resolve_dataset_path(model_path or FASTTEXT_MODEL)
    files = sorted(str(m) for m in StoragePath(corpus_glob).glob())[:max_files]
    if not files:
        raise ValueError(f"no files matched {corpus_glob}")
    logger.info("C fasttext: %d files, %d workers, cpu=%d/worker", len(files), max_workers, cpu)
    pipeline = (
        Dataset.from_list(files)
        .flat_map(functools.partial(load_file, include_file_paths=True))
        .map_shard(_writer(output_path, model_path))
    )
    ctx = ZephyrContext(
        name="ft-c-fasttext",
        resources=ResourceConfig(cpu=cpu, ram=f"{cpu * 2}g"),
        max_workers=max_workers,
        stage_runner_factory=InlineRunner,
        heartbeat_timeout=1200,
    )
    t0 = time.time()
    agg = dict(ctx.execute(pipeline).counters)
    wall = time.time() - t0
    docs = agg.get("ft2/docs", 0)
    chars = agg.get("ft2/chars", 0)
    payload = {
        "stage": "C_fasttext",
        "files": len(files),
        "max_workers": max_workers,
        "cpu_per_worker": cpu,
        "wall_s": round(wall, 1),
        "docs": docs,
        "chars": chars,
        "docs_per_s": round(docs / wall, 1) if wall else 0,
        "mb_per_s": round(chars / 1e6 / wall, 2) if wall else 0,
        "predict_worker_s": round(agg.get("ft2/predict_ms", 0) / 1000, 1),
        "counters": agg,
    }
    print("BENCH " + json.dumps(payload), flush=True)
    if result_json:
        write_result_json(result_json, payload)
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default=None, help="text parquet glob; relative paths root at marin_prefix()")
    p.add_argument("--model-path", default=None, help="fasttext .bin; relative paths root at marin_prefix()")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-files", type=int, default=64)
    p.add_argument("--max-workers", type=int, default=32)
    p.add_argument("--cpu", type=int, default=8)
    p.add_argument("--result-json", default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(
        corpus_glob=args.corpus,
        model_path=args.model_path,
        output_path=args.out_dir,
        max_files=args.max_files,
        max_workers=args.max_workers,
        cpu=args.cpu,
        result_json=args.result_json,
    )


if __name__ == "__main__":
    main()
