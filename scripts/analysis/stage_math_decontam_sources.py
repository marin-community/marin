# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage math benchmark sources into one Datakit decontamination directory.

The output directory is meant to be passed directly as ``eval_data_sources`` to
Datakit decontamination.  Every staged data file is JSONL gzip with at least
``id`` and ``text`` fields; metadata lives in hidden sidecars so Datakit's
recursive reader skips it.

Default sources:

* ``HuggingFaceH4/MATH-500`` test split.
* ``openai/gsm8k`` main train and test splits.
* ``HuggingFaceH4/aime_2024`` train split.
"""

from __future__ import annotations

import argparse
import json
import logging
import posixpath
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from marin.utils import fsspec_mkdirs, load_dataset_with_backoff
from rigging.filesystem import atomic_rename, open_url, url_to_fs

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = "gs://marin-us-east5/scratch/ahmed/math-decontamin-dir"

MATH500_DATASET_ID = "HuggingFaceH4/MATH-500"
MATH500_REVISION = "ff5b202"
MATH500_SPLIT = "test"

GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
GSM8K_CONFIG = "main"
GSM8K_SPLITS = ("train", "test")

AIME24_DATASET_ID = "HuggingFaceH4/aime_2024"
AIME24_REVISION = "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"
AIME24_SPLIT = "train"

STAGED_FILENAME = "data.jsonl.gz"
MANIFEST_FILENAME = ".manifest.json"


@dataclass(frozen=True)
class SourceSpec:
    """One benchmark source split to stage."""

    name: str
    dataset_id: str
    revision: str
    split: str
    output_subdir: str
    renderer: Callable[[dict[str, Any]], str]
    id_fields: tuple[str, ...]
    config: str | None = None

    @property
    def output_file(self) -> str:
        return posixpath.join(self.output_subdir, STAGED_FILENAME)


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _required_string(record: dict[str, Any], field: str) -> str:
    value = _clean_string(record.get(field))
    if not value:
        raise ValueError(f"Record is missing required non-empty field {field!r}: {record}")
    return value


def _labeled_sections(sections: Sequence[tuple[str, str]]) -> str:
    blocks = [f"{label}:\n{text}" for label, text in sections if text]
    if not blocks:
        raise ValueError("Cannot render an empty decontamination record")
    return "\n\n".join(blocks)


def render_math500(record: dict[str, Any]) -> str:
    """Render a MATH-500 row as benchmark text."""
    return _labeled_sections(
        (
            ("Problem", _required_string(record, "problem")),
            ("Solution", _clean_string(record.get("solution"))),
            ("Answer", _clean_string(record.get("answer"))),
        )
    )


def render_gsm8k(record: dict[str, Any]) -> str:
    """Render a GSM8K row as benchmark text."""
    return _labeled_sections(
        (
            ("Question", _required_string(record, "question")),
            ("Answer", _clean_string(record.get("answer"))),
        )
    )


def render_aime24(record: dict[str, Any]) -> str:
    """Render an AIME24 row as benchmark text."""
    return _labeled_sections(
        (
            ("Problem", _required_string(record, "problem")),
            ("Solution", _clean_string(record.get("solution"))),
            ("Answer", _clean_string(record.get("answer") or record.get("expected_answer"))),
        )
    )


def _record_id(spec: SourceSpec, record: dict[str, Any], index: int) -> str:
    for field in spec.id_fields:
        value = _clean_string(record.get(field))
        if value:
            return f"{spec.name}:{spec.split}:{value}"
    return f"{spec.name}:{spec.split}:{index:08d}"


def _output_exists(path: str) -> bool:
    fs, resolved = url_to_fs(path)
    return fs.exists(resolved)


def _output_size(path: str) -> int:
    fs, resolved = url_to_fs(path)
    return int(fs.info(resolved)["size"])


def load_hf_records(spec: SourceSpec) -> Iterable[dict[str, Any]]:
    """Stream one pinned Hugging Face dataset split."""
    kwargs: dict[str, Any] = {
        "path": spec.dataset_id,
        "split": spec.split,
        "revision": spec.revision,
        "streaming": True,
    }
    if spec.config is not None:
        kwargs["name"] = spec.config
    return load_dataset_with_backoff(context=f"load {spec.dataset_id} {spec.split}", **kwargs)


def stage_source(
    *,
    spec: SourceSpec,
    output_root: str,
    force: bool,
    max_records: int | None = None,
    records: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stage one source split and return manifest metadata for it."""
    output_dir = posixpath.join(output_root.rstrip("/"), spec.output_subdir)
    output_file = posixpath.join(output_dir, STAGED_FILENAME)
    fsspec_mkdirs(output_dir, exist_ok=True)

    if _output_exists(output_file) and not force:
        logger.info("Skipping existing %s", output_file)
        return {
            "name": spec.name,
            "dataset_id": spec.dataset_id,
            "revision": spec.revision,
            "config": spec.config,
            "split": spec.split,
            "output_file": output_file,
            "status": "skipped_existing",
            "bytes_written": _output_size(output_file),
        }

    source_records = records if records is not None else load_hf_records(spec)
    record_count = 0
    with atomic_rename(output_file) as temp_path:
        with open_url(temp_path, "wt", encoding="utf-8", compression="gzip") as handle:
            for index, record in enumerate(source_records):
                staged = {
                    "id": _record_id(spec, record, index),
                    "text": spec.renderer(record),
                    "source": spec.name,
                    "provenance": {
                        "dataset_id": spec.dataset_id,
                        "revision": spec.revision,
                        "config": spec.config,
                        "split": spec.split,
                        "index": index,
                    },
                }
                json.dump(staged, handle, ensure_ascii=False)
                handle.write("\n")
                record_count += 1
                if max_records is not None and record_count >= max_records:
                    break

    bytes_written = _output_size(output_file)
    logger.info("Staged %s records to %s", record_count, output_file)
    return {
        "name": spec.name,
        "dataset_id": spec.dataset_id,
        "revision": spec.revision,
        "config": spec.config,
        "split": spec.split,
        "output_file": output_file,
        "status": "written",
        "record_count": record_count,
        "bytes_written": bytes_written,
    }


def build_source_specs(args: argparse.Namespace) -> list[SourceSpec]:
    """Build the source split list from parsed CLI args."""
    specs = [
        SourceSpec(
            name="math500",
            dataset_id=args.math500_dataset_id,
            revision=args.math500_revision,
            split=args.math500_split,
            output_subdir=posixpath.join("math500", args.math500_split),
            renderer=render_math500,
            id_fields=("unique_id", "id"),
        ),
    ]

    for split in args.gsm8k_splits:
        specs.append(
            SourceSpec(
                name=f"gsm8k:{args.gsm8k_config}",
                dataset_id=args.gsm8k_dataset_id,
                revision=args.gsm8k_revision,
                config=args.gsm8k_config,
                split=split,
                output_subdir=posixpath.join("gsm8k", args.gsm8k_config, split),
                renderer=render_gsm8k,
                id_fields=("id",),
            )
        )

    if not args.skip_aime24:
        specs.append(
            SourceSpec(
                name="aime24",
                dataset_id=args.aime24_dataset_id,
                revision=args.aime24_revision,
                split=args.aime24_split,
                output_subdir=posixpath.join("aime24", args.aime24_split),
                renderer=render_aime24,
                id_fields=("id", "unique_id"),
                config=args.aime24_config,
            )
        )

    return specs


def write_manifest(output_root: str, entries: list[dict[str, Any]], args: argparse.Namespace) -> str:
    """Write hidden manifest sidecar for reproducibility."""
    output_file = posixpath.join(output_root.rstrip("/"), MANIFEST_FILENAME)
    payload = {
        "schema_version": 1,
        "description": "Datakit-ready math benchmark sources for decontamination.",
        "output_root": output_root,
        "staged_filename": STAGED_FILENAME,
        "entries": entries,
        "args": {
            "force": args.force,
            "max_records_per_split": args.max_records_per_split,
            "skip_aime24": args.skip_aime24,
        },
    }
    fsspec_mkdirs(output_root, exist_ok=True)
    with atomic_rename(output_file) as temp_path:
        with open_url(temp_path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return output_file


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true", help="Overwrite staged JSONL files if they already exist.")
    parser.add_argument("--max-records-per-split", type=int, default=None, help="Smoke-test cap; omit for full staging.")

    parser.add_argument("--math500-dataset-id", default=MATH500_DATASET_ID)
    parser.add_argument("--math500-revision", default=MATH500_REVISION)
    parser.add_argument("--math500-split", default=MATH500_SPLIT)

    parser.add_argument("--gsm8k-dataset-id", default=GSM8K_DATASET_ID)
    parser.add_argument("--gsm8k-revision", default=GSM8K_REVISION)
    parser.add_argument("--gsm8k-config", default=GSM8K_CONFIG)
    parser.add_argument("--gsm8k-splits", nargs="+", default=list(GSM8K_SPLITS))

    parser.add_argument("--skip-aime24", action="store_true")
    parser.add_argument("--aime24-dataset-id", default=AIME24_DATASET_ID)
    parser.add_argument("--aime24-revision", default=AIME24_REVISION)
    parser.add_argument("--aime24-config", default=None)
    parser.add_argument("--aime24-split", default=AIME24_SPLIT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    entries = [
        stage_source(
            spec=spec,
            output_root=args.output_root,
            force=args.force,
            max_records=args.max_records_per_split,
        )
        for spec in build_source_specs(args)
    ]
    manifest_path = write_manifest(args.output_root, entries, args)
    logger.info("Wrote manifest to %s", manifest_path)
    logger.info("Datakit eval_data_sources root: %s", args.output_root)


if __name__ == "__main__":
    main()
