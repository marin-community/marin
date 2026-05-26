# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate the datakit MoE mixture bucket CSV from the store manifest.

Reads ``.artifact.json`` from the datakit store, applies the same 166/34
mixable/tail split as ``datakit_moe_mix.py``, and writes one row per mixture
component (bucket name, token count, epochs to reach each token budget).
"""

import argparse
import csv
import json

import fsspec

STORE = "gs://marin-us-central2/datakit/store_8ac06c74"

# MixtureDataset caps block size at 2**16; a bucket mixes on its own only if
# `token_share * MIXTURE_BLOCK_SIZE >= 1`. Buckets below that floor go to `tail`.
MIXTURE_BLOCK_SIZE = 65535

# Token budgets to report epoch counts for.
BUDGETS = {"epochs_for_8T": 8e12, "epochs_for_2T": 2e12}

DEFAULT_OUTPUT = "experiments/grug/moe/datakit_moe_mix_buckets_us_central2.csv"


def _bucket_name(cluster: int, quality: int) -> str:
    return f"c{cluster:02d}q{quality}"


def generate_rows(store: str) -> list[tuple[str, int]]:
    """Return (bucket_name, tokens) per mixture component, mixable buckets then `tail`."""
    with fsspec.open(f"{store}/.artifact.json") as fh:
        buckets = json.load(fh)["buckets"]

    total_tokens = sum(b["total_tokens"] for b in buckets)
    floor = total_tokens / MIXTURE_BLOCK_SIZE

    mixable = sorted(
        (b for b in buckets if b["total_tokens"] >= floor),
        key=lambda b: (b["cluster_id"], b["quality_bucket"]),
    )
    tail_tokens = sum(b["total_tokens"] for b in buckets if b["total_tokens"] < floor)

    rows = [(_bucket_name(b["cluster_id"], b["quality_bucket"]), b["total_tokens"]) for b in mixable]
    rows.append(("tail", tail_tokens))
    return rows


def write_csv(rows: list[tuple[str, int]], output: str) -> None:
    with open(output, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["bucket", "tokens", *BUDGETS])
        for name, tokens in rows:
            writer.writerow([name, tokens, *(round(budget / tokens, 4) for budget in BUDGETS.values())])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=STORE, help="datakit store path containing .artifact.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="CSV output path")
    args = parser.parse_args()

    rows = generate_rows(args.store)
    write_csv(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
