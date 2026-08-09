# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Emit a structured Contract-pair/Map report from an XLA HLO dump."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from tile_lifetime.xla_hlo_recovery import recover_pair_map_regions


def main() -> None:
    """Analyze one plain-text or gzipped HLO dump."""
    parser = argparse.ArgumentParser()
    parser.add_argument("hlo", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = args.hlo.read_bytes()
    if args.hlo.suffix == ".gz":
        payload = gzip.decompress(payload)
    report = recover_pair_map_regions(payload.decode()).to_dict()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    args.output.write_text(rendered)


if __name__ == "__main__":
    main()
