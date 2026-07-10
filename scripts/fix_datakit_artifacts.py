# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite stale paths in the published datakit artifact.json files.

When the decon and dedup outputs were promoted from their original TTL
buckets to the canonical ``gs://marin-eu-west4/datakit/...`` paths, the
data was copied but the embedded ``output_dir`` / ``attr_dir`` fields in
``artifact.json`` were left pointing at the (now-stale) TTL paths.

This script rewrites those fields in place:

* Every ``gs://marin-eu-west4/datakit/decontam/**/artifact.json``:
  set ``output_dir`` to the artifact dir itself.
* ``gs://marin-eu-west4/datakit/dedup/dedup_v0_manual/artifact.json``:
  rewrite every ``sources.<main_dir>.attr_dir`` from
  ``gs://marin-eu-west4/tmp/ttl=2d/rav/datakit/dedup_783d0380/outputs/source_NNN``
  to ``gs://marin-eu-west4/datakit/dedup/dedup_v0_manual/outputs/source_NNN``.

Before rewriting any artifact, the original is copied to
``artifact.json.bak`` (idempotent: skips if a ``.bak`` already exists).

Default is dry-run; pass ``--apply`` to actually write.
"""

import argparse
import json
import logging
import os
import posixpath

from rigging.filesystem import StoragePath, url_to_fs

logger = logging.getLogger(__name__)


DECON_GLOB = "gs://marin-eu-west4/datakit/decontam/**/artifact.json"
DEDUP_ARTIFACT = "gs://marin-eu-west4/datakit/dedup/dedup_v0_manual/artifact.json"
DEDUP_NEW_OUTPUTS_ROOT = "gs://marin-eu-west4/datakit/dedup/dedup_v0_manual/outputs"

_BACKUP_SUFFIX = ".bak"


def _gs_url(p: str) -> str:
    return p if p.startswith("gs://") else f"gs://{p}"


def _read_json(path: str) -> dict:
    return json.loads(StoragePath(path).read_text())


def _backup_then_write_json(path: str, data: dict) -> None:
    """Copy ``path`` to ``path.bak`` (if no backup yet), then overwrite ``path`` with ``data``.

    Backup is idempotent: an existing ``.bak`` is preserved as the original
    snapshot rather than overwritten by an already-rewritten current value.
    """
    backup_path = f"{path}{_BACKUP_SUFFIX}"
    if not StoragePath(backup_path).exists():
        fs, src = url_to_fs(path)
        _, dst = url_to_fs(backup_path)
        fs.copy(src, dst)
    StoragePath(path).write_text(json.dumps(data))


def fix_decon(apply: bool) -> int:
    """Rewrite output_dir in every decon artifact.json. Returns the count rewritten."""
    paths = [_gs_url(str(m)) for m in StoragePath(DECON_GLOB).glob()]
    logger.info("decon: scanning %d artifact.json files", len(paths))
    n_changed = 0
    for path in paths:
        d = _read_json(path)
        art_dir = posixpath.dirname(path)
        old = d.get("output_dir")
        if old == art_dir:
            continue
        d["output_dir"] = art_dir
        n_changed += 1
        if n_changed <= 5 or n_changed % 20 == 0:
            logger.info("decon[%d]: %s -> %s", n_changed, old, art_dir)
        if apply:
            _backup_then_write_json(path, d)
    logger.info("decon: %d rewrites %s", n_changed, "applied" if apply else "(dry-run)")
    return n_changed


def fix_dedup(apply: bool) -> int:
    """Rewrite sources.*.attr_dir in the dedup artifact. Returns the count rewritten."""
    d = _read_json(DEDUP_ARTIFACT)
    n_changed = 0
    for src_key, entry in d["sources"].items():
        old = entry["attr_dir"]
        new = f"{DEDUP_NEW_OUTPUTS_ROOT}/{os.path.basename(old.rstrip('/'))}"
        if old == new:
            continue
        entry["attr_dir"] = new
        n_changed += 1
        if n_changed <= 3:
            logger.info("dedup[%d]: %s\n  %s -> %s", n_changed, src_key, old, new)
    logger.info("dedup: %d rewrites %s", n_changed, "applied" if apply else "(dry-run)")
    if apply and n_changed > 0:
        _backup_then_write_json(DEDUP_ARTIFACT, d)
    return n_changed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually write changes (default: dry-run).")
    args = parser.parse_args()

    n_decon = fix_decon(args.apply)
    n_dedup = fix_dedup(args.apply)
    logger.info("total: %d decon + %d dedup rewrites %s", n_decon, n_dedup, "applied" if args.apply else "(dry-run)")


if __name__ == "__main__":
    main()
