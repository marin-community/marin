# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch one reviewed Delphi true-midtraining cooldown cell.

This launcher consumes the explicit YAML plans in ``configs/``. It refuses to
choose checkpoints dynamically: the checkpoint candidate row must already be
marked ``review_status: approved`` and the per-mix cell must point at that row.

Usage:

    uv run python experiments/midtrain_specs/true_midtrain/nemotron_math_only/launcher.py \\
        --base 3e18 --mix p33m67 --cooldown-ratio 0.20 --tpu v6e-4 --dry-run

Submit the coordinator through Iris; the coordinator stages the checkpoint,
writes the manifest/config, submits the TPU child, and waits for it.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import draccus
import levanter.config  # noqa: F401 - registers draccus codecs used by Levanter config objects
import yaml
from levanter.optim import AdamHConfig
from marin.midtraining import (
    LLAMA3_TOKENIZER,
    ComputeProfile,
    CooldownMode,
    CooldownResume,
    CrossRegionCopyPolicy,
    MidtrainSpec,
    append_to_attempt_group,
    build_launch_request,
    build_manifest_row,
    build_run_identity,
    preflight,
    render_train_lm_config,
    resolve_midtrain_spec,
    stage_cooldown_checkpoint,
    submit_launch,
    validate_midtrain_spec,
    write_manifest,
    write_train_config,
)

from experiments.delphi_models import DELPHI_BANNED_SUBSTRINGS, get_delphi_model
from experiments.midtrain_specs import DELPHI_MIDTRAIN_MIXES, LEGACY_PROVENANCE, load_legacy_data_section
from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "configs"
WANDB_PROJECT = "delphi-midtraining"
DEFAULT_OUTPUT_REGION = "us-east5"
DEFAULT_CONTAINER_RAM = "256g"
APPROVED_REVIEW_STATUS = "approved"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Delphi flops key, e.g. 3e18.")
    parser.add_argument("--mix", required=True, choices=list(DELPHI_MIDTRAIN_MIXES))
    parser.add_argument("--cooldown-ratio", required=True, type=float, choices=[0.30, 0.20, 0.10])
    parser.add_argument("--tpu", required=True, help="TPU type for the training child, e.g. v6e-4.")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--output-region", default=DEFAULT_OUTPUT_REGION)
    parser.add_argument("--ram", default=DEFAULT_CONTAINER_RAM)
    parser.add_argument(
        "--allow-cross-region-stage",
        action="store_true",
        help="Allow copying the reviewed native checkpoint into the run output region.",
    )
    parser.add_argument("--stage-budget-gb", type=int, default=0)
    parser.add_argument(
        "--stage-reason",
        default="reviewed true-midtraining cooldown checkpoint staging",
        help="Audit reason recorded in the cooldown stage record when cross-region staging is allowed.",
    )
    parser.add_argument(
        "--no-preemptible-child",
        action="store_false",
        dest="child_preemptible",
        help="Force the nested TPU training child onto non-preemptible capacity when that capacity exists.",
    )
    parser.set_defaults(child_preemptible=True)
    parser.add_argument("--dry-run", action="store_true", help="Build and print the plan; do not stage or submit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    candidate = reviewed_candidate(args.base, args.cooldown_ratio)
    cell = planned_cell(args.mix, args.base, args.cooldown_ratio, candidate["candidate_id"])
    spec = build_spec(
        candidate=candidate,
        cell=cell,
        mix=args.mix,
        tpu_type=args.tpu,
        attempt=args.attempt,
        output_region=args.output_region,
        ram=args.ram,
        child_preemptible=args.child_preemptible,
    )
    resolved = resolve_midtrain_spec(spec)
    validate_midtrain_spec(resolved)

    if args.dry_run:
        print_plan(resolved, candidate)
        return 0

    cross_region = CrossRegionCopyPolicy(
        allowed=args.allow_cross_region_stage,
        budget_gb=args.stage_budget_gb,
        reason=args.stage_reason,
    )
    stage_record = stage_cooldown_checkpoint(spec, cross_region_copy=cross_region)
    report = preflight(resolved, cross_region_copy=cross_region)
    if not report.ok:
        print(f"FAIL {spec.run.run_id}")
        for failure in report.failures:
            print(f"  fail: {failure}")
        return 1
    for warning in report.warnings:
        print(f"warn {spec.run.run_id}: {warning}")

    row = build_manifest_row(resolved, report, stage_record=stage_record, status="launched")
    write_manifest(row, output_path=spec.run.output_path)
    write_train_config(resolved)
    append_to_attempt_group(row, region=spec.run.output_region)
    result = submit_launch(build_launch_request(resolved))
    print(f"submitted {spec.run.run_id}  (steps={resolved.num_train_steps}, tpu={spec.compute.tpu_type})")

    while True:
        try:
            status = result.wait(raise_on_failure=False)
        except Exception as exc:
            print(f"  {spec.run.run_id}: wait raised {exc!r}; retrying so the coordinator does not exit early")
            time.sleep(30)
            continue
        status_value = getattr(status, "value", str(status))
        if status_value in {"pending", "running"}:
            print(f"  {spec.run.run_id}: wait returned non-terminal status {status_value!r}; retrying")
            time.sleep(30)
            continue
        print(f"  {spec.run.run_id}: {status_value}")
        return 0 if status_value == "succeeded" else 1


def reviewed_candidate(base: str, cooldown_ratio: float) -> dict[str, Any]:
    data = load_yaml(CONFIG_DIR / "checkpoint_candidates.yaml")
    candidate_id = candidate_id_for(base, cooldown_ratio)
    matches = [row for row in data["checkpoint_candidates"] if row["candidate_id"] == candidate_id]
    if not matches:
        raise ValueError(f"No checkpoint candidate {candidate_id!r} in checkpoint_candidates.yaml")
    candidate = matches[0]
    if candidate["review_status"] != APPROVED_REVIEW_STATUS:
        raise ValueError(
            f"Checkpoint candidate {candidate_id!r} has review_status={candidate['review_status']!r}; "
            f"set review_status: {APPROVED_REVIEW_STATUS} after human review before launching."
        )
    if not candidate["suggested_required_artifacts_ok"]:
        raise ValueError(f"Checkpoint candidate {candidate_id!r} is not artifact-clean")
    return candidate


def planned_cell(mix: str, base: str, cooldown_ratio: float, candidate_id: str) -> dict[str, Any]:
    data = load_yaml(CONFIG_DIR / f"{mix}.yaml")
    if data["launch_contract"]["require_candidate_review_status"] != APPROVED_REVIEW_STATUS:
        raise ValueError(f"{mix}.yaml launch_contract must require approved candidates")
    matches = [
        cell
        for cell in data["cells"]
        if cell["base"] == base
        and float(cell["cooldown_ratio"]) == cooldown_ratio
        and cell["checkpoint_candidate_id"] == candidate_id
    ]
    if not matches:
        raise ValueError(f"No planned cell for base={base}, mix={mix}, cooldown_ratio={cooldown_ratio}")
    return matches[0]


def build_spec(
    *,
    candidate: dict[str, Any],
    cell: dict[str, Any],
    mix: str,
    tpu_type: str,
    attempt: int,
    output_region: str,
    ram: str,
    child_preemptible: bool,
) -> MidtrainSpec:
    base = get_delphi_model(candidate["base"])
    run = build_run_identity(
        logical_cell_id=cell["cell_id"],
        attempt=attempt,
        output_region_name=output_region,
        wandb_project=WANDB_PROJECT,
    )
    mode = CooldownMode(
        resume=CooldownResume(
            pretrain_checkpoint_path=candidate["suggested_checkpoint_path"],
            resume_step=int(candidate["suggested_step"]),
            staged_output_path=run.output_path,
        )
    )
    return MidtrainSpec(
        base=base,
        run=run,
        compute=ComputeProfile(
            tpu_type=tpu_type,
            batch_size=base.batch_size,
            ram=ram,
            regions=(output_region,),
            preemptible=child_preemptible,
        ),
        mode=mode,
        tokenizer=LLAMA3_TOKENIZER,
        model_config=model_config_for_base(base),
        optimizer_config=optimizer_config_for_cooldown(base),
        data_section_override=load_legacy_data_section(mix),
        data_section_provenance=LEGACY_PROVENANCE[mix],
        banned_substrings=frozenset(DELPHI_BANNED_SUBSTRINGS),
        expected_min_step=int(candidate["suggested_step"]),
        extra_tags=(
            "cooldown_midtraining",
            "true_midtraining",
            "dataset:nemotron_math_only",
            f"mix:{mix}",
            f"cooldown_ratio:{candidate['cooldown_ratio']:.2f}",
            f"target_step:{candidate['target_step']}",
            f"resume_step:{candidate['suggested_step']}",
            f"checkpoint_relation:{candidate['suggested_relation_to_target']}",
            f"checkpoint_candidate:{candidate['candidate_id']}",
        ),
    )


def model_config_for_base(base: Any) -> dict[str, Any]:
    cfg = completed_adamh_heuristic._build_model_config(hidden_size=base.hidden_dim)
    return {"type": "qwen3", **draccus.encode(cfg)}


def optimizer_config_for_cooldown(base: Any) -> dict[str, Any]:
    cfg = AdamHConfig(
        learning_rate=base.peak_lr,
        adam_lr=base.peak_adam_lr,
        beta1=base.beta1,
        beta2=base.beta2,
        epsilon=base.epsilon,
        max_grad_norm=base.max_grad_norm,
        weight_decay=base.weight_decay,
        warmup=base.warmup_fraction,
        decay=base.decay_fraction,
        min_lr_ratio=base.min_lr_ratio,
        lr_schedule=base.lr_schedule,
        nesterov=base.nesterov,
    )
    return {"type": "adamH", **draccus.encode(cfg)}


def print_plan(resolved: Any, candidate: dict[str, Any]) -> None:
    spec = resolved.spec
    tags = render_train_lm_config(resolved)["trainer"]["tracker"]["tags"]
    print(f"plan {spec.run.run_id}")
    print(f"  output_path: {spec.run.output_path}")
    print(f"  base: {spec.base.flops_key}")
    print(f"  tpu: {spec.compute.tpu_type}")
    print(f"  mix: {spec.extra_tags[3]}")
    print(f"  target_step: {candidate['target_step']}")
    print(f"  resume_step: {candidate['suggested_step']}")
    print(f"  checkpoint_delta: {candidate['suggested_step_delta']} ({candidate['suggested_relation_to_target']})")
    print(f"  checkpoint_path: {candidate['suggested_checkpoint_path']}")
    print("  wandb_tags:")
    for tag in tags:
        print(f"    - {tag}")


def candidate_id_for(base: str, cooldown_ratio: float) -> str:
    return f"delphi-{base}-cooldown{round(cooldown_ratio * 100):02d}"


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
