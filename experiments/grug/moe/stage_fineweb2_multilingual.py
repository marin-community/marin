# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage FineWeb2 multilingual *test*-split parquets from HuggingFace to GCS.

The perplexity-gap PPL reader can't open real ``hf://datasets/...`` paths (it
routes them through gcsfs and fails with ``b/hf%3A/o``), so the multilingual
datasets never score. This copies the pinned test-split parquets in-region
(run as a us-central2 Iris job) to ``gs://marin-us-central2/raw/fineweb2_multilingual/``
so ``fineweb2_multilingual_parquet_pattern`` can point at ``gs://`` like the
other bundles. ~95 configs, ~6.4 GB total.
"""

from concurrent.futures import ThreadPoolExecutor

import fsspec

from experiments.evals.fineweb2_multilingual import (
    FINEWEB2_DATASET_ID,
    FINEWEB2_MULTILINGUAL_EVAL_CONFIGS,
    FINEWEB2_PARQUET_REVISION,
)

_DST_PREFIX = "marin-us-central2/raw/fineweb2_multilingual"
_SPLIT = "test"


def _stage(config: str) -> tuple[str, int, int]:
    hf = fsspec.filesystem("hf")
    gs = fsspec.filesystem("gs")
    src_glob = f"hf://datasets/{FINEWEB2_DATASET_ID}@{FINEWEB2_PARQUET_REVISION}/{config}/{_SPLIT}/*.parquet"
    copied = skipped = 0
    for src in hf.glob(src_glob):
        name = src.rsplit("/", 1)[-1]
        dst = f"{_DST_PREFIX}/{config}/{_SPLIT}/{name}"
        if gs.exists("gs://" + dst):
            skipped += 1
            continue
        with hf.open(src, "rb") as f, gs.open("gs://" + dst, "wb") as g:
            while True:
                chunk = f.read(16 * 1024 * 1024)
                if not chunk:
                    break
                g.write(chunk)
        copied += 1
    return config, copied, skipped


def main() -> None:
    configs = FINEWEB2_MULTILINGUAL_EVAL_CONFIGS
    print(f"staging {len(configs)} fineweb2 configs ({_SPLIT} split) -> gs://{_DST_PREFIX}", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for config, copied, skipped in ex.map(_stage, configs):
            done += 1
            print(f"[{done}/{len(configs)}] {config}: copied={copied} skipped={skipped}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
