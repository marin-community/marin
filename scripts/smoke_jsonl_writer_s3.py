# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the durable JSONL writer against an fsspec URI.

This is intentionally tiny: read words from /usr/share/dict, enqueue them through
JsonlChunkWriter, and verify the manifest was written. It is useful for checking
that a cluster image has working fsspec/s3fs credentials without involving a GPU
or a trainer.
"""

import argparse
import json
from pathlib import Path

import fsspec

from marin.monitoring.jsonl_writer import BackpressurePolicy, JsonlChunkWriter, JsonlChunkWriterConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test JsonlChunkWriter against local/S3/GCS storage")
    parser.add_argument("--output-uri", required=True, help="Destination directory, e.g. s3://bucket/prefix")
    parser.add_argument("--dict-dir", default="/usr/share/dict", help="Directory containing a word list")
    parser.add_argument("--max-words", type=int, default=10_000)
    parser.add_argument("--records-per-chunk", type=int, default=1_000)
    parser.add_argument("--max-queue-items", type=int, default=2_000)
    args = parser.parse_args()

    config = JsonlChunkWriterConfig(
        output_uri=args.output_uri,
        records_per_chunk=args.records_per_chunk,
        max_queue_items=args.max_queue_items,
        backpressure_policy=BackpressurePolicy.BLOCK,
        log_every=args.records_per_chunk,
    )
    word_path = _find_word_list(Path(args.dict_dir))
    print(f"Reading up to {args.max_words} words from {word_path}")
    print(f"Writing JSONL chunks to {args.output_uri}")

    with JsonlChunkWriter(config) as writer:
        for index, word in enumerate(_read_words(word_path, args.max_words)):
            writer.write({"index": index, "word": word, "source_path": str(word_path)})

    manifest_uri = f"{args.output_uri.rstrip('/')}/manifest.json"
    with fsspec.open(manifest_uri, "rt") as f:
        manifest = json.load(f)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    expected = min(args.max_words, _count_words(word_path, args.max_words))
    if manifest["records_written"] != expected:
        raise RuntimeError(f"expected {expected} records, wrote {manifest['records_written']}")


def _find_word_list(dict_dir: Path) -> Path:
    candidates = [dict_dir / "words", dict_dir / "web2", dict_dir / "linux.words"]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    for candidate in sorted(dict_dir.iterdir() if dict_dir.exists() else []):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no word list found under {dict_dir}")


def _read_words(path: Path, max_words: int):
    with path.open() as f:
        for line in f:
            word = line.strip()
            if not word:
                continue
            yield word
            max_words -= 1
            if max_words <= 0:
                return


def _count_words(path: Path, max_words: int) -> int:
    return sum(1 for _ in _read_words(path, max_words))


if __name__ == "__main__":
    main()
