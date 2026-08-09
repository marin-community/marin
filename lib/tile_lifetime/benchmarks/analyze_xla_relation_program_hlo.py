# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Emit a generic Relation/Contract/Map/Fold report from XLA HLO."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from tile_lifetime.xla_relation_program_recovery import recover_relation_programs


def main() -> None:
    """Analyze one plain-text or gzipped HLO dump."""
    parser = argparse.ArgumentParser()
    parser.add_argument("hlo", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = args.hlo.read_bytes()
    if args.hlo.suffix == ".gz":
        payload = gzip.decompress(payload)
    rendered = json.dumps(recover_relation_programs(payload.decode()).to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    args.output.write_text(rendered)


if __name__ == "__main__":
    main()
