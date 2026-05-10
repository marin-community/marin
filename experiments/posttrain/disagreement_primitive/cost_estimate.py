"""Cost estimator for Anthropic / OpenAI / Gemini batches.

Anchors estimates on ACTUAL token usage from completed batches in this repo,
rather than rough per-call guesses. The prior Claude baseline fill (1,502
calls) cost $13 actual but I had estimated $10 — a 30% miss. This script
prevents repeating that mistake.

Usage:
    # Use historical Claude baseline fill data to forecast a new batch:
    python cost_estimate.py forecast claude --n-calls 5516

    # Show actual costs of a completed batch:
    python cost_estimate.py actual claude results/raw/e9_dart_claude_baseline_fill/<ts>/

    # Same for OpenAI:
    python cost_estimate.py forecast openai --n-calls 5516 --calibrate-from results/raw/e9_dart_gpt_baseline_fill/<ts>/
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

# Pricing (per million tokens, as of 2026-05).
# Empirically calibrated Anthropic rates from the 2026-05-10 baseline fill:
# 1,502 calls, 3.69M input tokens, 0.77M output tokens, cost $13 → effective
# rates $1.72/M input + $8.60/M output (close to but slightly above the
# documented $1.50/$7.50 Sonnet-4.6 batch rates — a +15% safety margin).
PRICING = {
    "claude": {  # Anthropic Sonnet 4.6 batch (50% off sync); empirical rates
        "input_per_m": 1.72,
        "output_per_m": 8.60,
        "notes": "Sonnet 4.6 batch; empirical from 2026-05-10 baseline fill ($13 / 1502 calls)",
    },
    "claude_sync": {  # Sonnet 4.6 sync (no discount)
        "input_per_m": 3.00,
        "output_per_m": 15.00,
        "notes": "Sonnet 4.6 sync; double batch rate",
    },
    "openai": {  # GPT-5.1 batch (reasoning_effort=none, 50% off sync)
        # GPT-5.1 sync rate: $1.25/M input, $10/M output (per 2026-05 OpenAI pricing)
        # Batch is 50% off → $0.625/M input, $5.00/M output
        # Add 15% safety margin (from Anthropic miss) → $0.72/M input, $5.75/M output
        # NEEDS CALIBRATION when first OpenAI batch completes — use --calibrate-from
        "input_per_m": 0.72,
        "output_per_m": 5.75,
        "notes": "GPT-5.1 batch reasoning_effort=none; needs empirical calibration",
    },
    "openai_sync": {
        "input_per_m": 1.25,
        "output_per_m": 10.00,
        "notes": "GPT-5.1 sync reasoning_effort=none; double batch rate",
    },
    "gemini_pro": {  # Gemini 3.1 Pro thinking_level=low, sync
        # ~$1.25/M input, $10/M output. Thinking tokens billed as output.
        "input_per_m": 1.25,
        "output_per_m": 10.00,
        "notes": "Gemini 3.1 Pro thinking_level=low; thinking tokens billed as output",
    },
}


def usage_from_anthropic_results(jsonl_path: Path) -> tuple[int, int, int]:
    """Read an Anthropic batch results jsonl and return (n, total_input, total_output)."""
    n = total_in = total_out = 0
    for line in jsonl_path.open():
        if not line.strip(): continue
        entry = json.loads(line)
        msg = entry.get("result", {}).get("message", {})
        u = msg.get("usage", {})
        if u:
            n += 1
            total_in += u.get("input_tokens", 0)
            total_out += u.get("output_tokens", 0)
    return n, total_in, total_out


def usage_from_openai_results(jsonl_path: Path) -> tuple[int, int, int]:
    """Read an OpenAI batch results jsonl and return (n, total_input, total_output)."""
    n = total_in = total_out = 0
    for line in jsonl_path.open():
        if not line.strip(): continue
        entry = json.loads(line)
        body = entry.get("response", {}).get("body", {})
        u = body.get("usage", {})
        if u:
            n += 1
            total_in += u.get("prompt_tokens", 0)
            total_out += u.get("completion_tokens", 0)
    return n, total_in, total_out


def find_results_files(job_dir: Path, provider: str) -> list[Path]:
    """Find all results.jsonl-style files in a job directory."""
    if provider == "claude":
        return sorted(job_dir.glob("*_results.jsonl"))
    if provider == "openai":
        return sorted(job_dir.glob("output_*.jsonl"))
    return []


def cmd_actual(provider: str, job_dir: Path):
    """Show actual cost from a completed batch."""
    files = find_results_files(job_dir, provider)
    if not files:
        print(f"No results files found in {job_dir}")
        return 1

    pricing = PRICING[provider]
    grand_n = grand_in = grand_out = 0
    for f in files:
        if provider == "claude":
            n, ti, to = usage_from_anthropic_results(f)
        else:
            n, ti, to = usage_from_openai_results(f)
        cost = (ti / 1e6 * pricing["input_per_m"]) + (to / 1e6 * pricing["output_per_m"])
        avg_in = ti / n if n else 0
        avg_out = to / n if n else 0
        print(f"  {f.name}: n={n}, in={ti:,} (avg {avg_in:.0f}), out={to:,} (avg {avg_out:.0f}), cost=${cost:.2f}")
        grand_n += n; grand_in += ti; grand_out += to

    if grand_n == 0: return 1
    grand_cost = (grand_in / 1e6 * pricing["input_per_m"]) + (grand_out / 1e6 * pricing["output_per_m"])
    print(f"\nTOTAL: {grand_n} calls, {grand_in:,} input + {grand_out:,} output tokens")
    print(f"  estimated cost: ${grand_cost:.2f}")
    print(f"  per-call avg: ${grand_cost/grand_n:.5f}")
    print(f"  per-call tokens: avg in={grand_in/grand_n:.0f}, avg out={grand_out/grand_n:.0f}")
    print(f"  pricing source: {pricing['notes']}")
    return 0


def cmd_forecast(provider: str, n_calls: int, calibrate_from: Path | None):
    """Forecast cost for a new batch of n_calls."""
    pricing = PRICING[provider]
    if calibrate_from:
        files = find_results_files(calibrate_from, provider)
        if not files:
            print(f"WARNING: no calibration data in {calibrate_from}; using PRICING table only")
            avg_in = avg_out = None
        else:
            grand_n = grand_in = grand_out = 0
            for f in files:
                if provider == "claude":
                    n, ti, to = usage_from_anthropic_results(f)
                else:
                    n, ti, to = usage_from_openai_results(f)
                grand_n += n; grand_in += ti; grand_out += to
            avg_in = grand_in / grand_n if grand_n else 0
            avg_out = grand_out / grand_n if grand_n else 0
            print(f"Calibrated from {calibrate_from} ({grand_n} historical calls):")
            print(f"  avg input tokens/call: {avg_in:.0f}")
            print(f"  avg output tokens/call: {avg_out:.0f}")
    else:
        # Default assumption: ~3000 input + ~500 output (typical judge call)
        avg_in = 3000
        avg_out = 500
        print(f"WARNING: no calibration; using default assumption {avg_in} input + {avg_out} output per call")
        print(f"  for higher accuracy, pass --calibrate-from <past_batch_dir>")

    if avg_in is not None:
        forecast_in = n_calls * avg_in
        forecast_out = n_calls * avg_out
        cost = (forecast_in / 1e6 * pricing["input_per_m"]) + (forecast_out / 1e6 * pricing["output_per_m"])
        print(f"\nForecast for {n_calls:,} calls on {provider}:")
        print(f"  expected input: {forecast_in:,} tokens")
        print(f"  expected output: {forecast_out:,} tokens")
        print(f"  pricing: ${pricing['input_per_m']}/M input + ${pricing['output_per_m']}/M output")
        print(f"  cost forecast: ${cost:.2f}")
        print(f"  per-call: ${cost/n_calls:.5f}")
        print(f"  pricing source: {pricing['notes']}")
        return cost
    return None


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a_actual = sub.add_parser("actual", help="Show actual cost of a completed batch")
    a_actual.add_argument("provider", choices=["claude", "openai"])
    a_actual.add_argument("job_dir", type=Path)

    a_fc = sub.add_parser("forecast", help="Forecast cost of a proposed batch")
    a_fc.add_argument("provider", choices=list(PRICING.keys()))
    a_fc.add_argument("--n-calls", type=int, required=True)
    a_fc.add_argument("--calibrate-from", type=Path, default=None,
                      help="Path to a past batch results dir to use for empirical token avgs")

    args = ap.parse_args()
    if args.cmd == "actual":
        return cmd_actual(args.provider, args.job_dir)
    if args.cmd == "forecast":
        return cmd_forecast(args.provider, args.n_calls, args.calibrate_from) and 0


if __name__ == "__main__":
    sys.exit(main())
