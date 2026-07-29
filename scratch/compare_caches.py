"""Compare the TPU-side (GCS) and GPU-side (S3/CoreWeave) contacts-v1 token caches.

The two caches were tokenized independently -- GCS on 2026.07.13.1 for the TPU sweeps, S3 on
2026.07.25 for exp153 -- and only their document/token *totals* have been checked. If the
token streams differ at all, every TPU-vs-GPU loss comparison is confounded before a single
step runs, so this is the cheapest thing to rule out.

Reads only ledgers plus the first chunk of the first shard: a few MB, not the 6 GB caches.
"""

import json

import fsspec

GCS = "gs://marin-us-east5/tokenized/contacts-v1/2026.07.13.1/train"
S3 = "s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/tokenized/contacts-v1/2026.07.25/train"
N_TOKENS = 4096


def ledger(fs, base: str) -> dict:
    with fs.open(f"{base}/shard_ledger.json", "rb") as handle:
        return json.load(handle)


def first_tokens(fs, base: str, n: int) -> list[int]:
    """Read the leading token ids out of the cache's first data chunk."""
    import numpy as np

    keys = sorted(k for k in fs.find(base) if "input_ids/data/" in k)
    if not keys:
        return []
    with fs.open(keys[0], "rb") as handle:
        raw = handle.read(n * 4)
    return np.frombuffer(raw, dtype=np.int32)[:n].tolist()


def summarize(label: str, fs, base: str) -> tuple[dict, list[int]]:
    led = ledger(fs, base)
    flat = {k: v for k, v in led.items() if not isinstance(v, (list, dict))}
    toks = first_tokens(fs, base, N_TOKENS)
    print(f"=== {label} ===")
    print(f"  {base}")
    print(f"  ledger: {flat}")
    print(f"  first {len(toks)} tokens: {toks[:16]}...")
    if toks:
        print(f"  sum={sum(toks)} min={min(toks)} max={max(toks)}")
    return flat, toks


def main() -> None:
    from rigging.filesystem.s3_compat import configure_coreweave_s3

    gcs = fsspec.filesystem("gs")
    gled, gtok = summarize("GCS (TPU sweeps)", gcs, GCS.replace("gs://", ""))

    configure_coreweave_s3()
    s3 = fsspec.filesystem("s3")
    sled, stok = summarize("S3 (CoreWeave exp153)", s3, S3.replace("s3://", ""))

    print("\n=== comparison ===")
    print(f"  ledger rows equal: {gled.get('total_num_rows') == sled.get('total_num_rows')}")
    if gtok and stok:
        same = gtok == stok
        print(f"  first {N_TOKENS} tokens identical: {same}")
        if not same:
            diffs = [i for i, (a, b) in enumerate(zip(gtok, stok)) if a != b]
            print(f"  first differing index: {diffs[0] if diffs else 'n/a'}  ({len(diffs)} of {len(gtok)} differ)")
    else:
        print("  could not read token data from one or both caches")


if __name__ == "__main__":
    main()
