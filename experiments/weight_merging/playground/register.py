# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Register an immutable local checkpoint in an evaluation run's source manifest."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.weight_merging.playground.checkpoints import checkpoint_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--training-metadata", type=Path)
    parser.add_argument("--repo")
    parser.add_argument("--revision")
    args = parser.parse_args()
    path = args.path.resolve()
    paths_file = args.run / "model-paths.json"
    sources_file = args.run / "model-sources.json"
    paths = json.loads(paths_file.read_text())
    sources = json.loads(sources_file.read_text())
    if args.name in paths or args.name in sources:
        raise ValueError(f"Source already registered: {args.name}")
    source = {"path": str(path), "weights_sha256": {}}
    for filename in sorted(set(checkpoint_map(path).values())):
        with (path / filename).open("rb") as stream:
            source["weights_sha256"][filename] = hashlib.file_digest(stream, "sha256").hexdigest()
    source["config_sha256"] = hashlib.sha256((path / "config.json").read_bytes()).hexdigest()
    if args.training_metadata:
        source["training_metadata"] = str(args.training_metadata.resolve())
        source["training_metadata_sha256"] = hashlib.sha256(args.training_metadata.read_bytes()).hexdigest()
    if args.repo or args.revision:
        if not args.repo or not args.revision:
            raise ValueError("A public source needs both repository and revision")
        source.update(repo=args.repo, revision=args.revision)
    paths[args.name] = str(path)
    sources[args.name] = source
    sources_file.write_text(json.dumps(sources, indent=2) + "\n")
    paths_file.write_text(json.dumps(paths, indent=2) + "\n")
    print(json.dumps(source, indent=2))


if __name__ == "__main__":
    main()
