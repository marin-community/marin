# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retry only first-wave model requests interrupted by upstream 429 rate limits.

Original response rows remain immutable. Recovered rows land at a separate URI
per model and must supersede only their matching, explicitly failed pair.
"""

import argparse
from decimal import Decimal

from experiments.post_training.capability_driven_envs.biotasks_library.freeze_combined_v2 import V2_ROOT
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave import (
    CAMPAIGN_START_USAGE_USD,
    MAX_CAMPAIGN_SPEND_USD,
    OUTPUT_ROOT,
    _key_budget,
    _model_path,
    _provider_estimate,
    _read_completed,
    _run_model,
    _verify_routes,
    load_eligible,
    load_panel,
)


def _is_rate_limited(row: dict) -> bool:
    """Distinguish provider 429 from bad tool arguments or model-level limits."""
    if row.get("trial", {}).get("status") != "ungraded":
        return False
    attempts = row.get("attempts", [])
    return bool(
        attempts
        and (attempts[-1].get("rollout", {}).get("failure") or {}).get("exception_type") == "RateLimitError"
        and (attempts[-1].get("rollout", {}).get("failure") or {}).get("diagnostics", {}).get("operation") == "model"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_workers > 4:
        parser.error("recovery concurrency must be between 1 and 4")
    tasks, task_sha = load_eligible(args.tasks or V2_ROOT + "taskspecs.jsonl")
    task_by_id = {task.id: task for task in tasks}
    models, manifest_sha = load_panel()
    for model in models:
        prefix = f"{args.output_root.rstrip('/')}/attempts/{_model_path(model)}/"
        original = _read_completed(prefix + "results.jsonl", tasks, model, manifest_sha)
        if original is None:
            raise RuntimeError(f"Missing original response matrix for {model.model}")
        pending = [task_by_id[row["task_id"]] for row in original if _is_rate_limited(row)]
        if not pending:
            continue
        output = prefix + "rate-limit-recovery1.jsonl"
        recovered = _read_completed(output, pending, model, manifest_sha)
        if recovered is None:
            usage, remaining = _key_budget()
            reserve = max(Decimal("5"), _provider_estimate(model) * Decimal(len(pending)) / Decimal(len(tasks)))
            spent = usage - CAMPAIGN_START_USAGE_USD
            if spent + reserve > MAX_CAMPAIGN_SPEND_USD or reserve > remaining:
                raise RuntimeError(f"Recovery budget gate for {model.model}: spent={spent} reserve={reserve}")
            print(f"RECOVER model={model.model} rate_limit_pairs={len(pending)} reserve_usd={reserve}", flush=True)
            _run_model(pending, model, manifest_sha, output, args.max_workers)
            recovered = _read_completed(output, pending, model, manifest_sha)
            if recovered is None:
                raise RuntimeError(f"Missing recovery output for {model.model}")
        _verify_routes(recovered, model, output)
        remaining_429 = sum(_is_rate_limited(row) for row in recovered)
        print(
            f"RECOVERED model={model.model} rows={len(recovered)} remaining_rate_limit={remaining_429} output={output}",
            flush=True,
        )
    print(f"FIRST_WAVE_RECOVERY_COMPLETE task_sha={task_sha} panel_sha={manifest_sha}", flush=True)


if __name__ == "__main__":
    main()
