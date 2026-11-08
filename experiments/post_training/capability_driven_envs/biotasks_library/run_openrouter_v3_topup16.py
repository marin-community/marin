# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pilot additional routes; select at most 13 later without resampling pilots.

Uses the original additional20 campaign's immutable $40 key-usage baseline,
so both pilots and full runs share one spending envelope.
"""

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import fsspec
from taskcompendium.models import TaskSpec

from experiments.post_training.capability_driven_envs.biotasks_library.effective20_batches import full_batches
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave import (
    PinnedModel,
    _key_budget,
    _model_path,
    _provider_estimate,
    _read_completed,
    _verify_routes,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave import (
    load_panel as load_first_panel,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_first_wave_v3 import load_cohort
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_v3_additional20 import (
    MAX_NEW_SPEND,
    _run,
    load_pilot_ids,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_v3_additional20 import (
    ROOT as INITIAL_ROOT,
)
from experiments.post_training.capability_driven_envs.biotasks_library.run_openrouter_v3_additional20 import (
    load_panel as load_initial_panel,
)

PANEL = Path(__file__).with_name("openrouter_v3_topup16.json")
ROOT = INITIAL_ROOT.replace("openrouter-additional20-v1/", "openrouter-topup16-v1/")
CAMPAIGN_ID = "biotasks/combined/v3/calibration/openrouter-topup16-v1"


def load_panel() -> tuple[list[PinnedModel], str]:
    """Validate a supplementary, fully disjoint list of pinned routes."""
    data = PANEL.read_bytes()
    manifest = json.loads(data)
    models = [PinnedModel(**row) for row in manifest["models"]]
    old, _ = load_first_panel()
    initial, _ = load_initial_panel()
    if (
        manifest["panel_id"] != "biotasks-v3-openrouter-topup16-pilot-v1"
        or manifest["task_snapshot_sha256"] != "b893bd40fb1c6676619b32cd11a180969605b2b69a7a3b5b6df1974ae4bbb8d6"
        or manifest["prior_response_sha256"] != "acf9b5aff6a47230e2e0d0c751554e03969d4af9fbbb22099154db1aed039cda"
        or Decimal(manifest["shared_spend_ceiling_usd"]) != MAX_NEW_SPEND
        or len(models) != 16
        or len({m.model for m in models}) != len(models)
        or {m.model for m in models} & ({m.model for m in old} | {m.model for m in initial})
        or any(m.context_window < 65536 or not m.provider_tag or not m.provider_name for m in models)
    ):
        raise ValueError("Top-up panel is overlapping or incomplete")
    return models, hashlib.sha256(data).hexdigest()


def budget_gate(model: PinnedModel, pairs: int) -> None:
    """Check the original campaign's frozen baseline, including all pilots."""
    with fsspec.open(INITIAL_ROOT + "budget.json", "rt") as source:
        budget = json.load(source)
    _, initial_sha = load_initial_panel()
    if budget["panel_sha256"] != initial_sha or Decimal(budget["max_new_spend_usd"]) != MAX_NEW_SPEND:
        raise ValueError("Original campaign spending baseline changed")
    usage, remaining = _key_budget()
    spent = usage - Decimal(budget["baseline_usage_usd"])
    reserve = max(Decimal("1"), _provider_estimate(model) * Decimal(pairs) / Decimal(247))
    if spent < 0 or spent + reserve > MAX_NEW_SPEND or reserve > remaining:
        raise RuntimeError(f"Shared budget gate: spent={spent}, reserve={reserve}, remaining={remaining}")
    print(f"BUDGET spent_usd={spent} reserve_usd={reserve} remaining_usd={remaining}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--selected-manifest", type=Path, help="Required for full run; versioned effective-20 selection")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 8:
        parser.error("max-workers must be in [1,8]")
    if not args.pilot_only and args.selected_manifest is None:
        parser.error("full run needs a frozen effective-20 selection manifest")
    models, panel_sha = load_panel()
    selection = json.loads(args.selected_manifest.read_text()) if args.selected_manifest else None
    if selection is not None:
        selected = selection["topup_models"]
        if (
            selection["selection_id"] != "biotasks-v3-openrouter-effective20-v2"
            or selection["topup_panel_sha256"] != panel_sha
            or len(selected) != 13
            or len(set(selected)) != 13
            or not set(selected) <= {m.model for m in models}
            or set(selection["batch_sizes"]) != set(selected) | set(selection["initial_models"])
        ):
            raise ValueError("Invalid effective20 selection of 13 top-up models")
        models = [m for m in models if m.model in selected]
    cohort = load_cohort()
    tasks = {row["task_id"]: TaskSpec.model_validate(row["task"]) for row in cohort.rows}
    pilot_ids = load_pilot_ids(set(tasks))
    for model in models:
        prefix = ROOT + f"attempts/{_model_path(model)}/"
        pilot = [tasks[item] for item in pilot_ids]
        pilot_uri = prefix + "pilot.jsonl"
        completed = _read_completed(pilot_uri, pilot, model, panel_sha)
        if completed is None:
            budget_gate(model, len(pilot))
            _run(
                pilot,
                model,
                panel_sha,
                pilot_uri,
                min(args.max_workers, 2),
                campaign_id=CAMPAIGN_ID,
                name_prefix="biotasks-topup16",
            )
            completed = _read_completed(pilot_uri, pilot, model, panel_sha)
            if completed is None:
                raise RuntimeError(f"Missing top-up pilot: {model.model}")
        try:
            _verify_routes(completed, model, pilot_uri)
        except RuntimeError as error:
            if "Generation metadata still unavailable" not in str(error):
                raise
            print(f"PILOT_METADATA_UNAVAILABLE model={model.model}; no full batch", flush=True)
            continue
        if not any(row.get("provider_evidence") for row in completed):
            print(f"PILOT_UNROUTED model={model.model}; no full batch", flush=True)
            continue
        print(f"PILOT model={model.model} completed=12", flush=True)
        if args.pilot_only:
            continue
        rest = [tasks[item] for item in sorted(tasks) if item not in pilot_ids]
        assert selection is not None
        for full_uri, batch in full_batches(rest, model.model, prefix, selection["batch_sizes"]):
            existing = _read_completed(full_uri, batch, model, panel_sha)
            if existing is None:
                budget_gate(model, len(batch))
                _run(
                    batch,
                    model,
                    panel_sha,
                    full_uri,
                    args.max_workers,
                    campaign_id=CAMPAIGN_ID,
                    name_prefix="biotasks-topup16",
                )
                existing = _read_completed(full_uri, batch, model, panel_sha)
                if existing is None:
                    raise RuntimeError(f"Missing full top-up batch: {model.model}")
            _verify_routes(existing, model, full_uri)
        print(f"DONE model={model.model} pairs=230", flush=True)
    print("TOPUP16_PILOT_COMPLETE" if args.pilot_only else "TOPUP_SELECTED_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
