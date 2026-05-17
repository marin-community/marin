# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Anthropic warmup + batch caching probe.

Tests whether a sync 'warmup' call (which writes the cache) followed by a
batch of N-1 calls with the same cacheable prefix produces ~1 write + ~N-1
reads instead of the ~N/2 + ~N/2 race we saw with batch-only.

Flow:
  1. Submit 1 sync request via /v1/messages with cache_control: ephemeral on
     the prefix. Wait for it to complete. This writes the cache.
  2. Submit N-1 batch requests via /v1/messages/batches with the SAME prefix
     and same cache_control marker. These should all cache-hit since the
     cache was written by step 1 and is still warm (5-min TTL).
  3. Print per-request cache metrics and aggregate hit rate.

If the cache works as we hope, we should see N-1 of N-1 batch requests with
cache_read > 0 (only step 1 wrote the cache).

Usage:
    unset ANTHROPIC_API_KEY  # to bypass Claude Code session token
    set -a; source .env; source .env2; set +a
    PYENV_VERSION=3.12.0 uv run python -m experiments.posttrain.disagreement_primitive.diversity_gen.probe_claude_warmup_batch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import batch_anthropic as ba  # type: ignore
from experiments.posttrain.disagreement_primitive.diversity_gen.probe_claude_batch_cache import (
    CACHEABLE_SYSTEM_PROMPT,
    CACHEABLE_USER_PREAMBLE,
)


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--n-requests", type=int, default=20,
                    help="Total request count = 1 warmup + (n-1) batch")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--job-dir", type=Path, default=None)
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set — `unset ANTHROPIC_API_KEY; set -a; source .env; source .env2; set +a`")

    if args.job_dir is None:
        args.job_dir = Path(f"/tmp/claude_warmup_batch_probe_{_now_stamp()}")
    args.job_dir.mkdir(parents=True, exist_ok=True)

    print("== Anthropic warmup + batch caching probe ==")
    print(f"  model:      {args.model}")
    print(f"  n_requests: 1 warmup + {args.n_requests - 1} batch = {args.n_requests} total")
    print(f"  job_dir:    {args.job_dir}")

    # ---- Step 1: sync warmup call ----
    print()
    print("== step 1: sync warmup call (writes cache) ==")
    warmup_payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "system": [{"type": "text", "text": CACHEABLE_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": CACHEABLE_USER_PREAMBLE, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Generate warmup scenario #0. Pick any plausible target referent. Return a JSON object with the standard scenario fields."},
            ],
        }],
    }
    t0 = time.time()
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "anthropic-version": "2023-06-01",
            "x-api-key": api_key,
            "content-type": "application/json",
        },
        json=warmup_payload,
        timeout=300.0,
    )
    if resp.status_code != 200:
        print(f"  WARMUP FAILED: HTTP {resp.status_code}")
        print(f"  body: {resp.text[:400]}")
        return
    warmup_result = resp.json()
    warmup_usage = warmup_result.get("usage", {})
    warmup_elapsed = time.time() - t0
    (args.job_dir / "warmup_result.json").write_text(json.dumps(warmup_result, indent=2))
    print(f"  warmup OK in {warmup_elapsed:.1f}s")
    print(f"  warmup usage: input={warmup_usage.get('input_tokens',0)}  "
          f"cache_create={warmup_usage.get('cache_creation_input_tokens',0)}  "
          f"cache_read={warmup_usage.get('cache_read_input_tokens',0)}  "
          f"output={warmup_usage.get('output_tokens',0)}")
    if warmup_usage.get("cache_creation_input_tokens", 0) == 0:
        print("  WARNING: warmup didn't write to cache. Caching may not be engaging at all.")

    # ---- Step 2: submit batch with same cacheable prefix ----
    print()
    print(f"== step 2: submit batch with {args.n_requests - 1} requests sharing the same prefix ==")
    requests = []
    suffixes = [
        f"Generate scenario #{i+1} of {args.n_requests}. Pick a unique target "
        f"referent that has NOT been used before. Suggested seed index: {i+1}. "
        f"Return a JSON object with the standard scenario fields."
        for i in range(args.n_requests - 1)
    ]
    for i, suffix in enumerate(suffixes):
        user_content = CACHEABLE_USER_PREAMBLE + suffix
        req = ba.build_request(
            custom_id=f"batch_{i:03d}",
            model=args.model,
            system=CACHEABLE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=args.max_tokens,
            cache=True,
            cache_user_prefix=CACHEABLE_USER_PREAMBLE,
        )
        requests.append(req)

    state = ba.submit(api_key, requests, args.job_dir, name="batch")
    print(f"  submitted batch {state['batch_id']}")
    print(f"  polling...")
    ba.poll(api_key, args.job_dir, name="batch", interval=10.0, timeout=3600.0)
    print(f"  batch terminal — collecting results")
    entries = ba.collect(api_key, args.job_dir, name="batch")

    # ---- Step 3: aggregate ----
    print()
    print(f"== per-request batch cache metrics ==")
    print(f"  {'custom_id':<14s}  {'input':>8s}  {'cache_create':>14s}  {'cache_read':>12s}  {'output':>8s}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*14}  {'-'*12}  {'-'*8}")

    total_input = 0
    total_cache_create = 0
    total_cache_read = 0
    total_output = 0
    n_succeeded = 0
    n_with_cache_read = 0
    for entry in sorted(entries, key=lambda e: e.get("custom_id", "")):
        cid = entry.get("custom_id", "?")
        result = entry.get("result", {})
        if result.get("type") != "succeeded":
            print(f"  {cid:<14s}  [FAIL: {result.get('type','?')}]")
            continue
        n_succeeded += 1
        u = ba.usage_of(entry)
        ip = u.get("input_tokens", 0) or 0
        cc = u.get("cache_creation_input_tokens", 0) or 0
        cr = u.get("cache_read_input_tokens", 0) or 0
        op = u.get("output_tokens", 0) or 0
        total_input += ip
        total_cache_create += cc
        total_cache_read += cr
        total_output += op
        if cr > 0:
            n_with_cache_read += 1
        print(f"  {cid:<14s}  {ip:>8d}  {cc:>14d}  {cr:>12d}  {op:>8d}")

    print()
    print(f"== aggregate ==")
    print(f"  succeeded:                   {n_succeeded} / {len(entries)}")
    print(f"  batch requests cache-hit:    {n_with_cache_read} / {n_succeeded}")
    print(f"  total batch input:           {total_input:>10,d}")
    print(f"  total batch cache_creation:  {total_cache_create:>10,d}")
    print(f"  total batch cache_read:      {total_cache_read:>10,d}")
    print(f"  total batch output:          {total_output:>10,d}")
    print()

    # Cost comparison: warmup is sync (no batch discount); batch is 50% off everything.
    wu_ip = warmup_usage.get("input_tokens", 0)
    wu_cc = warmup_usage.get("cache_creation_input_tokens", 0)
    wu_cr = warmup_usage.get("cache_read_input_tokens", 0)
    wu_op = warmup_usage.get("output_tokens", 0)

    # Sonnet 4.6 rates (sync): input $3/M, cache_write 5m $3.75/M, cache_read $0.30/M, output $15/M
    # Sonnet 4.6 rates (batch, 50% off): input $1.50/M, cache_write $1.875/M, cache_read $0.15/M, output $7.50/M
    warmup_cost = (wu_ip * 3.00 + wu_cc * 3.75 + wu_cr * 0.30 + wu_op * 15.00) / 1_000_000
    batch_cost = (total_input * 1.50 + total_cache_create * 1.875 + total_cache_read * 0.15 + total_output * 7.50) / 1_000_000

    # Baseline: same N requests in batch without ANY caching (no cache_control markers)
    # would pay full input rate on every token. The total "input that would have been"
    # is `total_input + total_cache_create + total_cache_read`, all at batch rate.
    batch_no_cache_baseline = (
        (total_input + total_cache_create + total_cache_read) * 1.50 + total_output * 7.50
    ) / 1_000_000
    warmup_no_cache_baseline = (
        (wu_ip + wu_cc + wu_cr) * 3.00 + wu_op * 15.00
    ) / 1_000_000

    print(f"== cost comparison ==")
    print(f"  warmup actual:           ${warmup_cost:.5f}")
    print(f"  batch  actual:           ${batch_cost:.5f}")
    print(f"  TOTAL actual:            ${warmup_cost + batch_cost:.5f}")
    print()
    print(f"  warmup if NO caching:    ${warmup_no_cache_baseline:.5f}")
    print(f"  batch  if NO caching:    ${batch_no_cache_baseline:.5f}")
    print(f"  TOTAL if NO caching:     ${warmup_no_cache_baseline + batch_no_cache_baseline:.5f}")
    print()
    print(f"  savings: ${(warmup_no_cache_baseline + batch_no_cache_baseline) - (warmup_cost + batch_cost):.5f}")
    if (warmup_no_cache_baseline + batch_no_cache_baseline) > 0:
        savings_pct = 100 * (1 - (warmup_cost + batch_cost) / (warmup_no_cache_baseline + batch_no_cache_baseline))
        print(f"  savings: {savings_pct:.1f}%")

    print()
    if n_succeeded == 0:
        print("  VERDICT: probe failed — no batch results succeeded")
    elif n_with_cache_read >= n_succeeded - 1:
        print(f"  VERDICT: warmup pattern works — {n_with_cache_read} of {n_succeeded} batch requests cache-hit. "
              "Use this pattern (1 sync warmup → batch of remainder) for max caching.")
    elif n_with_cache_read >= n_succeeded // 2:
        print(f"  VERDICT: partial — {n_with_cache_read} of {n_succeeded} batch requests cache-hit. "
              "Some parallel writes still happened despite warmup. May need a longer wait before batch.")
    else:
        print(f"  VERDICT: warmup pattern did NOT help — only {n_with_cache_read} of {n_succeeded} cache-hit. "
              "Cache may have evicted or the warmup didn't actually write to cache.")


if __name__ == "__main__":
    main()
