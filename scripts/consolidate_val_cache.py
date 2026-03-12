"""Re-consolidate the visual-only validation shard caches into a proper TreeCache.

The original tokenization produced per-shard validation caches under
gs://marin-vlm/hf_85m_levanter_cache_v2/visual_only/validation/
but the consolidation step did not produce top-level TreeCache structure
(input_ids/, loss_weights/, shard_ledger.json), so Levanter cannot load it.

This script lists the existing shard caches, filters to those with a .success
sentinel, and runs consolidate_shard_caches to produce the merged TreeCache.

Usage:
    uv run scripts/consolidate_val_cache.py
"""

import logging

import fsspec
import numpy as np

from levanter.store.cache import consolidate_shard_caches

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VALIDATION_PATH = "gs://marin-vlm/hf_85m_levanter_cache_v2/visual_only/validation"


def main():
    fs, _ = fsspec.core.url_to_fs(VALIDATION_PATH)

    # List all shard subdirectories (part-*_val/)
    entries = fs.ls(VALIDATION_PATH, detail=False)
    shard_paths = [f"gs://{e}" for e in entries if fs.isdir(e)]
    logger.info("Found %d shard directories under %s", len(shard_paths), VALIDATION_PATH)

    # Batch-check .success sentinels via glob instead of per-path fs.exists
    success_files = set(fs.glob(f"{VALIDATION_PATH.removeprefix('gs://')}/*/.success"))
    existing = [p for p in shard_paths if f"{p.removeprefix('gs://')}/.success" in success_files]
    logger.info("Of those, %d have .success sentinel", len(existing))

    if not existing:
        logger.error("No valid shard caches found. Nothing to consolidate.")
        return

    exemplar = {
        "input_ids": np.zeros((0,), dtype=np.int32),
        "loss_weights": np.zeros((0,), dtype=np.float32),
    }

    logger.info("Consolidating %d shard caches into %s", len(existing), VALIDATION_PATH)
    consolidate_shard_caches(
        shard_cache_paths=existing,
        output_path=VALIDATION_PATH,
        exemplar=exemplar,
    )
    logger.info("Consolidation complete.")


if __name__ == "__main__":
    main()
