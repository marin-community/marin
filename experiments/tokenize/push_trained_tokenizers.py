# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Push trained tokenizer artifacts so cluster workers can load them (Track C).

``levanter.load_tokenizer(name_or_path)`` resolves a bare ``"org/name"``-shaped string by
staging a ``mirror://tokenizers/<name_or_path>/hf-hub-<hf_hub_version>/`` cache (rigging's
mirror filesystem, backed by ``$MARIN_PREFIX`` on the cluster) before falling back to the HF
Hub — see ``_stage_tokenizer`` in ``lib/levanter/src/levanter/tokenizers.py``. A raw ``s3://``
path does not work as ``name_or_path`` (``os.path.isdir`` is false for it, and it is not a valid
HF Hub repo id), so this pre-populates that mirror cache directly for each trained tokenizer
under ``trained/<name>``, matching exactly how the off-the-shelf baseline/SuperBPE arms resolve.
It also writes a second, human-browsable copy with no ``hf-hub-<version>`` path segment, at
``tokenizers/trained/<name>/tokenizer.json`` under the same prefix.

Run:
  uv run python -m experiments.tokenize.push_trained_tokenizers \\
      --tokenizers-dir experiments/tokenize/results/trained_tokenizers
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from huggingface_hub import __version__ as hf_hub_version
from rigging.filesystem import open_url

logger = logging.getLogger(__name__)

# Must match `_MIRROR_TOKENIZER_PREFIX` in lib/levanter/src/levanter/tokenizers.py.
_MIRROR_TOKENIZER_PREFIX = "tokenizers"
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


def arm_ref(name: str) -> str:
    """The ``TokenizerArm.ref`` a pushed tokenizer named ``name`` resolves under."""
    return f"trained/{name}"


def push_one(tokenizer_dir: str, name: str) -> dict:
    """Push one trained tokenizer's files to both the functional and manifest locations."""
    ref = arm_ref(name)
    cache_prefix = f"mirror://{_MIRROR_TOKENIZER_PREFIX}/{ref}/hf-hub-{hf_hub_version}"
    manifest_prefix = f"mirror://{_MIRROR_TOKENIZER_PREFIX}/{ref}"

    pushed = []
    for filename in _TOKENIZER_FILES:
        local_path = os.path.join(tokenizer_dir, filename)
        if not os.path.isfile(local_path):
            continue
        with open(local_path, "rb") as src:
            data = src.read()
        for prefix in (cache_prefix, manifest_prefix):
            with open_url(f"{prefix}/{filename}", "wb") as dst:
                dst.write(data)
        pushed.append(filename)

    logger.info("pushed %s (%s) -> %s and %s", name, pushed, cache_prefix, manifest_prefix)
    return {"name": name, "ref": ref, "files": pushed, "manifest_path": manifest_prefix}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizers-dir", required=True, help="directory of <name>/tokenizer.json subdirs")
    ap.add_argument("--out", default=None, help="write the push manifest as JSON here")
    args = ap.parse_args()

    manifest_path = os.path.join(args.tokenizers_dir, "manifest.json")
    with open(manifest_path) as f:
        trained = json.load(f)

    results = [push_one(row["tokenizer_dir"], row["name"]) for row in trained]

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
