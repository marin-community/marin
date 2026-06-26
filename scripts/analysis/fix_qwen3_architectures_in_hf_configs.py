# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Patch Delphi midtrain HF checkpoint configs: architectures -> ["Qwen3ForCausalLM"].

Background: marin PR #3092 — KEYS_TO_COPY_FROM_BASE_CONFIG in
levanter.compat.hf_checkpoints copies `architectures` from the base reference
config, clobbering the correct value. Weights are real Qwen3; only the
config.json label is wrong. This script fixes the label in-place on all
matching exported checkpoints under gs://marin-us-east5/.

Idempotent: skips configs already patched, so re-running picks up
in-flight runs as they finish.

Usage:
    python scripts/analysis/fix_qwen3_architectures_in_hf_configs.py            # dry-run (default)
    python scripts/analysis/fix_qwen3_architectures_in_hf_configs.py --apply    # actually write
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

GCS_GLOB = "gs://marin-us-east5/checkpoints/delphi-*/hf/step-*/config.json"
EXPECTED_MODEL_TYPE = "qwen3"
EXPECTED_VOCAB_SIZE = 128256
TARGET_ARCHITECTURES = ["Qwen3ForCausalLM"]
WRONG_ARCHITECTURES = ["LlamaForCausalLM"]
GSUTIL_NO_MATCH_MARKER = "matched no objects"

logger = logging.getLogger(__name__)


def list_config_paths(glob: str) -> list[str]:
    """Return GCS paths matching glob, or [] if no objects match."""
    result = subprocess.run(
        ["gsutil", "ls", glob],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if GSUTIL_NO_MATCH_MARKER in result.stderr.lower():
            return []
        raise RuntimeError(f"gsutil ls failed (rc={result.returncode}): {result.stderr}")
    return [line for line in result.stdout.splitlines() if line.startswith("gs://")]


def download(gcs_path: str, local_path: Path) -> None:
    subprocess.run(
        ["gsutil", "-q", "cp", gcs_path, str(local_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def upload(local_path: Path, gcs_path: str) -> None:
    subprocess.run(
        ["gsutil", "-q", "cp", str(local_path), gcs_path],
        check=True,
        capture_output=True,
        text=True,
    )


def patch_config(gcs_path: str, apply: bool) -> str:
    """Process one config.json. Returns a status string for logging."""
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "config.json"
        download(gcs_path, local)
        config = json.loads(local.read_text())

        model_type = config.get("model_type")
        vocab_size = config.get("vocab_size")
        if model_type != EXPECTED_MODEL_TYPE:
            return f"SKIP (model_type={model_type!r}, expected {EXPECTED_MODEL_TYPE!r})"
        if vocab_size != EXPECTED_VOCAB_SIZE:
            return f"SKIP (vocab_size={vocab_size}, expected {EXPECTED_VOCAB_SIZE})"

        current = config.get("architectures")
        if current == TARGET_ARCHITECTURES:
            return "OK (already patched)"
        if current != WRONG_ARCHITECTURES:
            return f"SKIP (architectures={current!r}, refusing to overwrite unfamiliar value)"

        if not apply:
            return f"WOULD-PATCH ({current} -> {TARGET_ARCHITECTURES})"

        config["architectures"] = TARGET_ARCHITECTURES
        local.write_text(json.dumps(config))
        upload(local, gcs_path)
        return "PATCHED"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Upload patched configs. Without this flag, runs in dry-run mode.",
    )
    parser.add_argument(
        "--glob",
        default=GCS_GLOB,
        help=f"GCS glob to match config.json files. Default: {GCS_GLOB}",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mode = "APPLY" if args.apply else "DRY-RUN"

    logger.info(f"[{mode}] discovering configs at {args.glob}")
    paths = list_config_paths(args.glob)
    logger.info(f"[{mode}] found {len(paths)} config.json file(s)")

    counts: dict[str, int] = {}
    for path in paths:
        status = patch_config(path, apply=args.apply)
        bucket = status.split(" ", 1)[0]
        counts[bucket] = counts.get(bucket, 0) + 1
        logger.info(f"  {status}: {path}")

    logger.info(f"[{mode}] summary: {dict(sorted(counts.items()))}")
    if not args.apply and counts.get("WOULD-PATCH", 0) > 0:
        logger.info("Re-run with --apply to write the changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
