"""Verify the token cache exp166cw pins matches the GCS cache #117 trained on.

exp166cw reads the contacts-v1 cache in place from the exp153 prefix rather than copying
it. Every exp166cw result is compared against a #117 number, so if those two caches differ
at all the whole experiment is confounded before a single step runs -- the same check that
ruled data out of the TPU-vs-GPU investigation.

Reads ledgers plus the leading tokens of the first data chunk: a few MB, not the 6 GB
caches.
"""

import json

import fsspec

GCS_TRAIN = "gs://marin-us-east5/tokenized/contacts-v1/2026.07.13.1/train"
GCS_VAL = "gs://marin-us-east5/tokenized/contacts-v1-val/2026.07.13.1/validation"
S3_TRAIN = "s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/tokenized/contacts-v1/2026.07.25/train"
S3_VAL = (
    "s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/tokenized/contacts-v1-val/2026.07.25/validation"
)
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


def summarize(label: str, fs, base: str) -> tuple[int | None, list[int]]:
    led = ledger(fs, base)
    rows = led.get("total_num_rows")
    toks = first_tokens(fs, base, N_TOKENS)
    print(f"  {label:<10} rows={rows} first16={toks[:16]}")
    return rows, toks


def compare(split: str, gcs_path: str, s3_path: str, gcs, s3) -> bool:
    print(f"\n=== {split} ===")
    g_rows, g_toks = summarize("GCS", gcs, gcs_path.replace("gs://", ""))
    s_rows, s_toks = summarize("S3", s3, s3_path.replace("s3://", ""))
    same_rows = g_rows == s_rows
    same_toks = bool(g_toks) and g_toks == s_toks
    print(f"  rows equal: {same_rows}   first {N_TOKENS} tokens identical: {same_toks}")
    if not same_toks and g_toks and s_toks:
        diffs = [i for i, (a, b) in enumerate(zip(g_toks, s_toks, strict=False)) if a != b]
        print(f"  first differing index: {diffs[0] if diffs else 'n/a'} ({len(diffs)} of {len(g_toks)} differ)")
    return same_rows and same_toks


def main() -> None:
    from rigging.filesystem.s3_compat import configure_coreweave_s3

    gcs = fsspec.filesystem("gs")
    configure_coreweave_s3()
    s3 = fsspec.filesystem("s3")

    train_ok = compare("train", GCS_TRAIN, S3_TRAIN, gcs, s3)
    val_ok = compare("validation", GCS_VAL, S3_VAL, gcs, s3)

    print(f"\nVERDICT: {'MATCH -- pinned cache is the cache #117 trained on' if train_ok and val_ok else 'MISMATCH'}")
    if not (train_ok and val_ok):
        raise SystemExit("pinned cache does not match GCS; do not launch until this is resolved")


if __name__ == "__main__":
    main()
