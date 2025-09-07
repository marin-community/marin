import random
import numpy as np
import tensorstore as ts
from typing import Any
from urllib.parse import urlparse
from transformers import AutoTokenizer


def open_zarr3_array(dir_path: str, context: ts.Context | None = None):
    """
    Open a Zarr v3 array directory with tensorstore.
    dir_path should be the directory that contains zarr.json.
    """
    parsed = urlparse(dir_path)

    return ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "gcs", "bucket": parsed.netloc, "path": parsed.path.lstrip("/")},
        },
        open=True,
        read=True,
        context=context,
    ).result()


def sample_flat_spans(
    field_root: str,  # e.g. ".../input_ids"
    n_samples: int = 5,
    span_len: int = 100,
    seed: int = 0,
    context: ts.Context | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    data_arr = open_zarr3_array(field_root + "/data", context=context)
    offs_arr = open_zarr3_array(field_root + "/offsets", context=context)

    # Get number of offsets (no data read)
    n_offsets = int(next(iter(offs_arr.domain.shape)))  # 1-D
    num_docs = n_offsets - 1
    if num_docs <= 0:
        raise ValueError("No documents found.")

    # Stream offsets in chunks; collect first n_samples docs with len >= span_len
    CHUNK = 10_000_000  # tune as needed; memory stays O(chunk)
    selected = []  # store tuples (doc_idx, start_off, end_off)

    pos = 0
    prev_last = None  # stitch boundaries across chunks
    done = False
    while pos < n_offsets and not done:
        end = min(n_offsets, pos + CHUNK + 1)  # +1 to compute diffs
        chunk = offs_arr[pos:end].read().result()  # small slice only

        base = pos
        if prev_last is not None:
            # Prepend last offset from previous chunk to compute first doc len here
            chunk = np.concatenate([prev_last, chunk])
            base = pos - 1

        lens = np.diff(chunk)  # doc lengths for this window
        # Iterate docs in this window in order and pick first eligible
        for idx in range(len(lens)):
            if lens[idx] >= span_len:
                di = base + idx
                start_off = int(chunk[idx])
                end_off = int(chunk[idx + 1])
                selected.append((di, start_off, end_off))
                if len(selected) >= n_samples:
                    done = True
                    break

        prev_last = chunk[-1:].copy()
        pos += CHUNK

    if not selected:
        raise ValueError(f"No documents at least {span_len} tokens long.")

    # Fetch and return the spans (only 100 tokens per sample)
    samples = []
    for di, s, e in selected:
        length = e - s
        rel = 0 if length == span_len else rng.randint(0, length - span_len)
        abs_start = s + rel
        toks = np.array(data_arr[abs_start : abs_start + span_len].read().result(), dtype=np.int64).tolist()
        samples.append(
            {
                "doc": int(di),
                "doc_len": int(length),
                "start_in_doc": int(rel),
                "abs_start": int(abs_start),
                "tokens": toks,
            }
        )
    return samples


if __name__ == "__main__":
    # EXAMPLE USAGE:
    # Field root that contains "data/zarr.json" and "offsets/zarr.json"
    # FIELD_ROOT = "gs://marin-us-central1/tokenized/dclm-wrap-qa-1024-dbd9ba/train/input_ids"
    FIELD_ROOT = "gs://marin-us-central1/tokenized/dclm-wrap-qa-1024-v2-160b8e/train/input_ids"

    # Optional: configure TensorStore context for better streaming behavior
    ctx = ts.Context(
        {
            "cache_pool": {"total_bytes_limit": 256 * 1024 * 1024},
            "data_copy_concurrency": {"limit": 8},
            "gcs_request_concurrency": {"limit": 16},
            "gcs_request_retries": {},
        }
    )

    samples = sample_flat_spans(
        field_root=FIELD_ROOT,
        n_samples=5,
        span_len=100,
        seed=42,
        context=ctx,
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    for i, s in enumerate(samples, 1):
        print(
            f"Sample {i}: doc={s['doc']} len={s['doc_len']} start_in_doc={s['start_in_doc']} abs_start={s['abs_start']}"
        )
        print(tokenizer.decode(s["tokens"]))
        print()
