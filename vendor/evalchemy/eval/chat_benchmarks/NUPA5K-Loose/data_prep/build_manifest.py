"""Rebuild the checked-in NUPA5K identity manifest from the pinned source."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

from ..panel import MANIFEST_PATH, NUPA5K_SIZE, build_nupa5k_identities, write_manifest

nupa = import_module("eval.chat_benchmarks.NUPA-Loose.eval_instruct")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=nupa.SOURCE_DATASET_NAME)
    parser.add_argument("--revision", default=nupa.SOURCE_DATASET_REVISION)
    parser.add_argument("--split", default=nupa.DEFAULT_SPLIT)
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--panel-size", type=int, default=NUPA5K_SIZE)
    args = parser.parse_args()

    source = args.source_file or nupa.download_nupa_source(
        dataset_name=args.dataset_name,
        dataset_revision=args.revision,
        split=args.split,
    )
    identities = build_nupa5k_identities(source, panel_size=args.panel_size)
    digest = write_manifest(args.output, identities)
    print(f"Wrote {len(identities)} NUPA5K identities to {args.output} (sha256={digest})")


if __name__ == "__main__":
    main()
