#!/usr/bin/env python3
"""Export, unpack, and import a credential-free reviewed image handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from capability_pipeline.image_publication_handoff import (
    export_handoff,
    import_publication,
    unpack_handoff,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--status", type=Path, required=True)
    export.add_argument("--archive", type=Path, required=True)
    unpack = commands.add_parser("unpack")
    unpack.add_argument("--archive", type=Path, required=True)
    unpack.add_argument("--output", type=Path, required=True)
    imported = commands.add_parser("import")
    imported.add_argument("--status", type=Path, required=True)
    imported.add_argument("--archive", type=Path, required=True)
    imported.add_argument("--role", choices=("candidate", "private_verifier"), required=True)
    imported.add_argument("--publication", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_handoff(args.status, args.archive)
    elif args.command == "unpack":
        result = unpack_handoff(args.archive, args.output)
    else:
        result = import_publication(args.status, args.archive, args.role, args.publication)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - never print provider or credential text.
        print(json.dumps({"state": "failed", "error_type": type(error).__name__}))
        raise SystemExit(1) from None
