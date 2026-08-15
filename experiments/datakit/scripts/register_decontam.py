# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write the per-source decontamination path map that ``hero_data.decontam`` reads.

The decon step's identity folds in an eval bloom and a cross-source drop-set
stage, and both have moved since the run that marked this corpus, so rebuilding
the step from current code addresses a directory nothing wrote. The path is
therefore recorded, like the Harrier map, and this script records it: it lists
``datakit/decontam/`` and keeps the directory each registered source actually
produced.

One source can have several directories there, one per decon run. This script
picks between them by their recorded provenance, not by mtime: a directory is a
candidate only if the decon step that wrote it depended on the drop-set stage
given as ``--drop-set``, so naming the run's drop sets selects that run's
outputs across every source. List what is on storage first::

    uv run python -m experiments.datakit.scripts.register_decontam --list

Then pin a run. Nothing is written unless every registered source resolves,
because a map that covers most of the registry sends the store looking for a
stage that is not there, deep into a fleet run::

    uv run python -m experiments.datakit.scripts.register_decontam \\
        --drop-set datakit/decon_drop/_combined_b583a0aa

Run it where the data is -- the CoreWeave data region -- with ``MARIN_PREFIX``
set to ``s3://marin-us-east-02a/marin``. Paths are stored relative to that root,
so the map resolves under whatever prefix the reader is configured for.
"""

import argparse
import collections
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from marin.execution.artifact import read_record
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.datakit import hero_data

logger = logging.getLogger(__name__)

DECON_RELATIVE = "datakit/decontam"

# What a step writes to say a directory is its output rather than a parent.
ARTIFACT_FILENAME = ".artifact.json"


def discover(root: str) -> dict[str, list[str]]:
    """Map each registered source to the decon output directories it could own.

    Keyed by source name, valued by ``<source>_<hash>`` directories relative to
    ``root``. A source name is recovered by stripping the trailing ``_<hash>``,
    so a directory whose stripped name is not registered is ignored rather than
    guessed at.

    Walks one delimited level at a time rather than globbing. Source names nest
    one to three path levels under the stage root (``stack-v3``, ``cp/usgpo``,
    ``starcoder2/ir/python``), and a glob deep enough to reach the third level
    enumerates every Parquet shard under all of them -- millions of keys to
    answer a question about directory names. Descending only into prefixes that a
    registered name actually starts with costs one request per directory.
    """
    registered = set(hero_data.source_names())
    prefixes = {
        "/".join(parts[:depth]) for name in registered for parts in [name.split("/")] for depth in range(1, len(parts))
    }

    candidates: list[tuple[str, str, StoragePath]] = []
    pending = [""]
    while pending:
        relative_dir = pending.pop()
        for child in StoragePath(f"{root}/{relative_dir}" if relative_dir else root).ls():
            relative = f"{relative_dir}/{child.name}" if relative_dir else child.name
            source = relative.rsplit("_", 1)[0]
            # The two are not exclusive. ``finepdfs`` is a registered source *and*
            # the parent of eighteen more, so the run writes both
            # ``finepdfs_<hash>`` and ``finepdfs/<lang>_<hash>``. A name test alone
            # reads the parent as the source and never descends past it.
            if source in registered:
                candidates.append((source, relative, child))
            if relative in prefixes:
                pending.append(relative)

    # Carrying the artifact file is what makes a directory an output rather than a
    # parent, and it is the only part of this that costs a request per directory.
    with ThreadPoolExecutor(max_workers=32) as pool:
        is_output = pool.map(lambda entry: (entry[2] / ARTIFACT_FILENAME).exists(), candidates)

    found: dict[str, list[str]] = collections.defaultdict(list)
    for (source, relative, _), output in zip(candidates, is_output, strict=True):
        if output:
            found[source].append(relative)
    return dict(found)


def select(candidates: dict[str, list[str]], root: str, drop_set: str | None) -> dict[str, str]:
    """Keep one directory per source: the one whose decon step used ``drop_set``.

    With no ``drop_set`` a source is only resolved when it has exactly one
    directory. Ambiguity is reported rather than broken by a rule, because the
    directories differ by decon run and picking the wrong one silently marks the
    corpus against the wrong eval corpus.
    """
    selected: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    for source, relatives in sorted(candidates.items()):
        if drop_set is None:
            matching = relatives
        else:
            matching = [r for r in relatives if drop_set in _dependencies_of(f"{root}/{r}")]
        if len(matching) == 1:
            selected[source] = matching[0]
        else:
            ambiguous[source] = matching or relatives
    if ambiguous:
        for source, relatives in sorted(ambiguous.items())[:10]:
            logger.error("%s: %d candidate directories %s", source, len(relatives), relatives)
        raise SystemExit(
            f"{len(ambiguous)} sources did not resolve to one decon directory; pass --drop-set to select a run"
        )
    return selected


def _dependencies_of(path: str) -> set[str]:
    """Return the step outputs a decon directory was built from, prefix-relative."""
    record = read_record(path)
    if record is None:
        raise SystemExit(f"{path} has no artifact record, so the run that wrote it cannot be identified")
    return {dep.removeprefix(f"{hero_data.MANIFEST_PREFIX}/") for dep in record.dep_paths}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drop-set", help="keep only directories whose artifact reports this drop-set stage")
    parser.add_argument("--list", action="store_true", help="report what is on storage and exit")
    args = parser.parse_args(argv)

    configure_logging(logging.INFO)
    configure_coreweave_s3()

    root = f"{hero_data.MANIFEST_PREFIX}/{DECON_RELATIVE}"
    candidates = discover(root)
    registered = hero_data.source_names()
    logger.info("%d of %d registered sources have decon output under %s", len(candidates), len(registered), root)

    if args.list:
        for source, relatives in sorted(candidates.items()):
            logger.info("%s: %s", source, relatives)
        return

    selected = select(candidates, root, args.drop_set)
    missing = sorted(set(registered) - set(selected))
    if missing:
        raise SystemExit(
            f"{len(missing)} registered sources have no decon output, e.g. {missing[:5]}. "
            "A partial map sends the store looking for a stage that is not there."
        )

    destination = hero_data.decon_paths_path()
    destination.write_text(json.dumps(selected, indent=1, sort_keys=True) + "\n")
    logger.info("wrote %d paths to %s", len(selected), destination)


if __name__ == "__main__":
    main()
