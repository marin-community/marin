# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect, store, and lower semantic task specifications."""

import argparse
import json
from pathlib import Path
from typing import BinaryIO, cast

import fsspec
import msgspec

from taskcompendium.execution import HarborTaskBinding
from taskcompendium.grading import source_verifier
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import tasktrove_verifier
from taskcompendium.serialization import (
    from_json,
    json_schema,
    read_parquet,
    renderings_from_json,
    specification_hash,
    write_parquet,
)


def _read(uri: str) -> bytes:
    with fsspec.open(uri, "rb") as stream:
        return cast(BinaryIO, stream).read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    schema = commands.add_parser("schema", help="Write the versioned TaskSpec JSON schema")
    schema.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate", help="Validate a specification against its pinned verifier ontology")
    validate.add_argument("input")
    pack = commands.add_parser("pack", help="Store JSON specifications as nested Parquet rows")
    pack.add_argument("inputs", nargs="+")
    pack.add_argument("--output", required=True)
    listing = commands.add_parser("list", help="List task identities and source provenance from Parquet")
    listing.add_argument("input")
    export = commands.add_parser("export", help="Lower one JSON specification to a runnable Harbor package")
    export.add_argument("input")
    export.add_argument("--renderings", required=True)
    export.add_argument("--binding", required=True, help="TaskCompendium HarborTaskBinding JSON")
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "schema":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(json_schema(), indent=2) + "\n")
    elif args.command == "validate":
        spec = from_json(_read(args.input))
        for step in spec.steps:
            verifier = tasktrove_verifier(step.verifier)
            if verifier is not None:
                source_verifier(verifier)
        print(json.dumps({"id": spec.id, "sha256": specification_hash(spec)}))
    elif args.command == "pack":
        count = write_parquet((from_json(_read(uri)) for uri in args.inputs), args.output)
        print(json.dumps({"rows": count, "output": args.output}))
    elif args.command == "list":
        for spec in read_parquet(args.input):
            print(
                json.dumps(
                    {
                        "id": spec.id,
                        "sha256": specification_hash(spec),
                        "source": msgspec.to_builtins(spec.metadata.source),
                    }
                )
            )
    elif args.command == "export":
        destination = lower_to_harbor(
            from_json(_read(args.input)),
            renderings_from_json(_read(args.renderings)),
            msgspec.json.decode(_read(args.binding), type=HarborTaskBinding),
            args.output,
        )
        print(destination)


if __name__ == "__main__":
    main()
