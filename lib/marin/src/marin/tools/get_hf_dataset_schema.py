# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Hugging Face Dataset Schema Inspection Tool

For usage instructions and examples, see:
https://github.com/marin-community/marin/blob/main/docs/recipes/add_dataset.md
"""

import argparse
import json
import logging
import os
import time

import requests
from datasets import get_dataset_config_names

log = logging.getLogger(__name__)

HF_DATASET_SIZE_URL = "https://datasets-server.huggingface.co/size"
HF_DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"


def hf_auth_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    # Fall back to the cached token from `huggingface-cli login`
    try:
        from huggingface_hub import HfApi

        token = HfApi().token
        if token:
            return {"Authorization": f"Bearer {token}"}
    except Exception as e:
        log.warning("Failed to resolve HF token from huggingface_hub: %s", e)
    return {}


def get_dataset_size(dataset_name: str, config_name: str | None = None) -> dict:
    """Return dataset size metadata (rows/bytes) from the datasets-server size endpoint."""

    params = {"dataset": dataset_name}
    if config_name:
        params["config"] = config_name

    response = requests.get(HF_DATASET_SIZE_URL, params=params, headers=hf_auth_headers(), timeout=60)
    response.raise_for_status()

    payload = response.json()
    size_info = payload.get("size", {})

    def _summary_block() -> dict:
        if config_name:
            summary = size_info.get("config")
            if summary is not None:
                return summary
        summary = size_info.get("dataset")
        if summary is not None:
            return summary
        return {}

    def _matches_target(entry: dict) -> bool:
        if entry.get("dataset") not in {None, dataset_name}:
            return False
        if config_name and entry.get("config") != config_name:
            return False
        return True

    splits_summary = {
        entry["split"]: {
            "config": entry.get("config"),
            "num_rows": entry.get("num_rows"),
            "num_columns": entry.get("num_columns"),
            "num_bytes_parquet_files": entry.get("num_bytes_parquet_files"),
            "num_bytes_memory": entry.get("num_bytes_memory"),
        }
        for entry in size_info.get("splits", [])
        if entry.get("split") and _matches_target(entry)
    }

    return {
        "summary": _summary_block(),
        "configs": size_info.get("configs"),
        "splits": splits_summary,
        "partial": payload.get("partial", False),
    }


def fetch_rows_at_offsets(
    dataset_name: str,
    config_name: str,
    split: str,
    offsets: list[int],
) -> list[dict]:
    """Fetch rows at the given offsets, pacing requests to avoid rate limits."""
    headers = hf_auth_headers()
    rows = []
    for i, offset in enumerate(offsets):
        for attempt in range(7):
            resp = requests.get(
                HF_DATASET_ROWS_URL,
                params={
                    "dataset": dataset_name,
                    "config": config_name,
                    "split": split,
                    "offset": offset,
                    "length": 1,
                },
                headers=headers,
                timeout=30,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = float(resp.headers.get("Retry-After", 0))
                backoff = max(retry_after, 2 ** (attempt + 1))
                time.sleep(backoff)
                continue
            resp.raise_for_status()
            fetched = resp.json().get("rows", [])
            if not fetched:
                raise ValueError(f"No rows returned for {dataset_name} at offset {offset}")
            rows.append(fetched[0]["row"])
            break
        else:
            resp.raise_for_status()
        if i < len(offsets) - 1:
            time.sleep(0.2)
    return rows


MAX_STREAM_ROWS = 100_000


def _sample_streaming(
    dataset_name: str,
    config: str | None,
    split: str,
    n: int,
    seed: int,
) -> list[dict]:
    """Reservoir-sample *n* rows by streaming, capped at MAX_STREAM_ROWS."""
    import random

    from datasets import load_dataset

    rng = random.Random(seed)
    kwargs: dict = {"split": split, "streaming": True}
    if config:
        kwargs["name"] = config

    ds = load_dataset(dataset_name, **kwargs)
    reservoir: list[dict] = []
    for i, example in enumerate(ds):
        if i >= MAX_STREAM_ROWS:
            break
        if i < n:
            reservoir.append(dict(example))
        else:
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = dict(example)
    return reservoir


def sample_dataset(
    dataset_name: str,
    config: str | None = None,
    split: str = "train",
    n: int = 5,
    seed: int = 42,
) -> dict:
    """Fetch random samples from a HuggingFace dataset.

    Tries the dataset viewer /rows API first. If that fails, falls back to
    streaming with reservoir sampling (capped at 100k rows).

    Returns a dict with keys: dataset, config, split, schema,
    num_rows, seed, samples (list of raw row dicts).
    """
    import random

    rng = random.Random(seed)

    size_info = get_dataset_size(dataset_name, config)
    split_meta = size_info.get("splits", {}).get(split, {})
    num_rows = split_meta.get("num_rows")
    resolved_config = split_meta.get("config") or config

    if not num_rows:
        summary = size_info.get("summary", {})
        num_rows = summary.get("num_rows")

    if not resolved_config:
        configs = get_dataset_config_names(dataset_name)
        if len(configs) == 1:
            resolved_config = configs[0]
        elif "default" in configs:
            resolved_config = "default"
        else:
            raise ValueError(f"Config required. Available: {configs}")

    # Try the /rows API first
    if num_rows:
        try:
            actual_n = min(n, num_rows)
            offsets = rng.sample(range(num_rows), actual_n)
            rows = fetch_rows_at_offsets(dataset_name, resolved_config, split, offsets)
            if rows:
                return {
                    "dataset": dataset_name,
                    "config": resolved_config,
                    "split": split,
                    "schema": list(rows[0].keys()),
                    "num_rows": num_rows,
                    "seed": seed,
                    "samples": rows,
                }
        except Exception as e:
            log.warning("Dataset viewer /rows failed for %s, falling back to streaming: %s", dataset_name, e)

    # Fallback: stream with reservoir sampling
    log.info("Streaming %s (config=%s, split=%s) for reservoir sampling...", dataset_name, resolved_config, split)
    rows = _sample_streaming(dataset_name, resolved_config, split, n, seed)
    if not rows:
        raise ValueError(f"Dataset returned 0 rows: {dataset_name}")

    return {
        "dataset": dataset_name,
        "config": resolved_config,
        "split": split,
        "schema": list(rows[0].keys()),
        "num_rows": num_rows,
        "seed": seed,
        "samples": rows,
    }


def _get_total_file_bytes(dataset_name: str) -> int:
    """Return total file size in bytes from HuggingFace Hub metadata."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(dataset_name, files_metadata=True)
    return sum(s.size for s in info.siblings if s.size)


def estimate_tokens(
    dataset_name: str,
    text_field: str,
    config_name: str | None = None,
    split: str = "train",
    tokenizer: str = "meta-llama/Meta-Llama-3.1-8B",
    sample_size: int = 100,
    seed: int = 42,
) -> dict:
    """Sample random rows from a dataset, tokenize them, and extrapolate total token count.

    Picks ``sample_size`` random row offsets, fetches them via the HF
    datasets-server /rows API, tokenizes the ``text_field``, and multiplies
    the average tokens-per-row by the total row count from the /size endpoint.

    When the datasets-server only indexed a partial slice of the data (common
    for large non-parquet datasets), the row count is extrapolated from the
    ratio of total file size to indexed parquet size.
    """
    import random as _random

    from transformers import AutoTokenizer

    rng = _random.Random(seed)
    size_info = get_dataset_size(dataset_name, config_name)
    partial = size_info.get("partial", False)
    split_meta = size_info.get("splits", {}).get(split, {})
    indexed_rows = split_meta.get("num_rows")
    indexed_parquet_bytes = split_meta.get("num_bytes_parquet_files")
    resolved_config = split_meta.get("config") or config_name

    if indexed_rows is None:
        summary = size_info.get("summary", {})
        indexed_rows = summary.get("num_rows")

    if not indexed_rows:
        raise ValueError(f"Could not determine row count for {dataset_name} split={split}")

    # When partial, extrapolate total rows from file size ratio
    num_rows = indexed_rows
    num_rows_estimated = False
    if partial and indexed_parquet_bytes and indexed_parquet_bytes > 0:
        total_bytes = _get_total_file_bytes(dataset_name)
        if total_bytes > indexed_parquet_bytes:
            num_rows = round(indexed_rows * (total_bytes / indexed_parquet_bytes))
            num_rows_estimated = True

    if not resolved_config:
        configs = get_dataset_config_names(dataset_name)
        if len(configs) == 1:
            resolved_config = configs[0]
        elif "default" in configs:
            resolved_config = "default"
        else:
            raise ValueError(f"Config name required for /rows API. Available: {configs}")

    actual_sample_size = min(sample_size, indexed_rows)
    offsets = rng.sample(range(indexed_rows), actual_sample_size)

    rows = fetch_rows_at_offsets(dataset_name, resolved_config, split, offsets)
    if not rows:
        raise ValueError(f"Failed to fetch any rows from {dataset_name}")

    tok = AutoTokenizer.from_pretrained(tokenizer)
    token_counts = [len(tok.encode(str(row[text_field]))) for row in rows]

    avg_tokens = sum(token_counts) / len(token_counts)
    estimated_total = round(avg_tokens * num_rows)

    result = {
        "avg_tokens_per_row": round(avg_tokens, 1),
        "sample_size": len(token_counts),
        "num_rows": num_rows,
        "estimated_total_tokens": estimated_total,
        "tokenizer": tokenizer,
    }
    if num_rows_estimated:
        result["num_rows_estimated"] = True
        result["partial"] = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HuggingFace dataset inspection: sample rows, estimate tokens.",
        epilog="Example: %(prog)s roneneldan/TinyStories --n 3 --estimate_tokens --text_field text",
    )
    parser.add_argument("dataset_name", help='Dataset name (e.g., "roneneldan/TinyStories")')
    parser.add_argument("--config_name", help="Config name for datasets with multiple configs")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--text_field", help="Text field for token estimation")
    parser.add_argument("--n", type=int, default=5, help="Number of rows to sample (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--estimate_tokens", action="store_true", help="Estimate total token count")
    parser.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3.1-8B", help="Tokenizer for estimation")
    parser.add_argument("--sample_size", type=int, default=100, help="Rows for token estimation (default: 100)")
    args = parser.parse_args()

    result = sample_dataset(
        args.dataset_name,
        config=args.config_name,
        split=args.split,
        n=args.n,
        seed=args.seed,
    )

    output = {
        "dataset": result["dataset"],
        "config": result["config"],
        "split": result["split"],
        "schema": result["schema"],
        "num_rows": result["num_rows"],
        "seed": result["seed"],
        "num_samples": len(result["samples"]),
        "samples": result["samples"],
    }

    if args.estimate_tokens:
        text_field = args.text_field
        if not text_field:
            for candidate in ("text", "content", "document", "body", "code"):
                if candidate in result["schema"]:
                    text_field = candidate
                    break
        if not text_field:
            parser.error(f"--text_field required for token estimation. Schema: {result['schema']}")
        output["estimated_tokens"] = estimate_tokens(
            dataset_name=args.dataset_name,
            text_field=text_field,
            config_name=args.config_name,
            split=args.split,
            tokenizer=args.tokenizer,
            sample_size=args.sample_size,
            seed=args.seed,
        )

    print(json.dumps(output, indent=2, default=str))
