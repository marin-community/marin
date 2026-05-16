# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Midtraining sweep for Delphi bases 3e18 -> 2e20 at K=0.20.

Sweep shape: 6 bases x 3 mixes x 4 LR factors = 72 cells.

Mode: CPT — `initialize_from_hf: <repo>@<revision>`. Each base streams its
HF weights directly via Levanter's `RepoRef.from_string` path (free
ingress; no GCS staging). Optimizer state starts fresh; the schedule
warms up over the first 10% of the CPT step count.

Math validation set: the ``data:`` block is lifted verbatim from each
mix's canonical 1e21 K=0.20 reference run (see
``experiments/midtrain_specs/data_sections/<mix>.json``). This makes the
held-out math val partition byte-identical to the 1e21/1e22 K=0.20 sweep
so cross-scale loss numbers are directly comparable.

Usage (one cell at a time, no batch CLI):

    uv run python experiments/midtrain_specs/delphi_small_cpt_k020.py \\
        --base 3e18 --mix p33m67 --lr 0.5

To plan/launch all 72 (parallelized one TPU slice per base):

    uv run python experiments/midtrain_specs/delphi_small_cpt_k020.py --dry-run
    uv run python experiments/midtrain_specs/delphi_small_cpt_k020.py --base 3e18

The script never auto-runs the full grid; you select per invocation.
"""

import argparse
import dataclasses
import logging
import time
from collections.abc import Iterator

from marin.midtraining import (
    LLAMA3_TOKENIZER,
    BudgetPolicy,
    CheckpointSourceKind,
    ComputeProfile,
    CptInit,
    CptMode,
    MidtrainSpec,
    append_to_attempt_group,
    build_launch_request,
    build_manifest_row,
    build_run_identity,
    preflight,
    resolve_midtrain_spec,
    submit_launch,
    validate_midtrain_spec,
    write_manifest,
    write_train_config,
)

from experiments.delphi_models import (
    DELPHI_2E19,
    DELPHI_2E20,
    DELPHI_3E18,
    DELPHI_3E19,
    DELPHI_9E18,
    DELPHI_9E19,
    DELPHI_BANNED_SUBSTRINGS,
    DelphiModel,
)
from experiments.midtrain_specs import (
    DELPHI_MIDTRAIN_MIXES,
    LEGACY_PROVENANCE,
    load_legacy_data_section,
)
from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

BASES: dict[str, DelphiModel] = {
    "3e18": DELPHI_3E18,
    "9e18": DELPHI_9E18,
    "2e19": DELPHI_2E19,
    "3e19": DELPHI_3E19,
    "9e19": DELPHI_9E19,
    "2e20": DELPHI_2E20,
}

LR_FACTORS: tuple[float, ...] = (0.33, 0.5, 0.67, 0.83)

# Per-base TPU recommendation. v5p-8 fits all of them; bigger slices for the
# larger bases cut wall-clock proportionally. Override per-cell via --tpu.
DEFAULT_TPU: dict[str, str] = {
    "3e18": "v5p-8",
    "9e18": "v5p-8",
    "2e19": "v5p-8",
    "3e19": "v5p-8",
    "9e19": "v5p-16",
    "2e20": "v5p-32",
}

REGION: str = "us-east5"
WANDB_PROJECT: str = "delphi-midtraining"
BUDGET: BudgetPolicy = BudgetPolicy.pretrain_fraction(0.20)


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------


def _data_section_for_mix(mix: str) -> dict:
    """Load the canonical 1e21 data section, returning a fresh copy each call."""
    return load_legacy_data_section(mix)


def _model_config_for_base(base: DelphiModel) -> dict:
    """Render the Qwen3 architecture dict from the heuristic at base.hidden_dim."""
    cfg = completed_adamh_heuristic._build_model_config(hidden_size=base.hidden_dim)
    return dataclasses.asdict(cfg)


def _optimizer_config_for_base(base: DelphiModel, *, lr_factor: float) -> dict:
    """Build an AdamH config dict from the base's pretrain hparams + CPT lr_factor.

    All non-LR fields are pretrain-verbatim from
    ``experiments/delphi_models.py`` (which itself reads from
    ``.executor_info``). Only ``learning_rate`` and ``adam_lr`` are scaled
    by ``lr_factor``; the rest passes through.
    """
    return {
        "type": "adam_h",
        "learning_rate": base.peak_lr * lr_factor,
        "adam_lr": base.peak_adam_lr * lr_factor,
        "beta1": base.beta1,
        "beta2": base.beta2,
        "epsilon": base.epsilon,
        "max_grad_norm": base.max_grad_norm,
        "weight_decay": base.weight_decay,
        "warmup": base.warmup_fraction,
        "decay": base.decay_fraction,
        "min_lr_ratio": base.min_lr_ratio,
        "lr_schedule": base.lr_schedule,
        "nesterov": base.nesterov,
    }


def _logical_cell_id(base_key: str, mix: str, lr_factor: float) -> str:
    return f"delphi-{base_key}-{mix}-k0p20-lr{round(lr_factor * 100):02d}"


def build_spec(
    *,
    base_key: str,
    mix: str,
    lr_factor: float,
    tpu_type: str | None = None,
    attempt: int = 1,
) -> MidtrainSpec:
    """Construct a :class:`MidtrainSpec` for one cell of the sweep."""
    base = BASES[base_key]
    tpu = tpu_type or DEFAULT_TPU[base_key]
    cell_id = _logical_cell_id(base_key, mix, lr_factor)
    run = build_run_identity(
        logical_cell_id=cell_id,
        attempt=attempt,
        output_region_name=REGION,
        wandb_project=WANDB_PROJECT,
    )
    return MidtrainSpec(
        base=base,
        run=run,
        compute=ComputeProfile(tpu_type=tpu, batch_size=base.batch_size, regions=(REGION,)),
        mode=CptMode(
            init=CptInit(
                source_kind=CheckpointSourceKind.HF_WEIGHTS,
                hf_repo=base.hf_repo,
                hf_revision=base.hf_revision,
            ),
            budget=BUDGET,
            lr_factor=lr_factor,
        ),
        tokenizer=LLAMA3_TOKENIZER,
        model_config=_model_config_for_base(base),
        optimizer_config=_optimizer_config_for_base(base, lr_factor=lr_factor),
        data_section_override=_data_section_for_mix(mix),
        data_section_provenance=LEGACY_PROVENANCE[mix],
        banned_substrings=frozenset(DELPHI_BANNED_SUBSTRINGS),
        extra_tags=("sweep:delphi-small-cpt-k020", f"mix:{mix}"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _expand(bases: tuple[str, ...], mixes: tuple[str, ...], lrs: tuple[float, ...]) -> Iterator[tuple]:
    for b in bases:
        for m in mixes:
            for lr in lrs:
                yield b, m, lr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch Delphi 3e18→2e20 K=0.20 midtraining cells.")
    parser.add_argument("--base", action="append", choices=list(BASES), help="Repeat to select multiple bases.")
    parser.add_argument(
        "--mix",
        action="append",
        choices=list(DELPHI_MIDTRAIN_MIXES),
        help="Repeat to select mixes.",
    )
    parser.add_argument("--lr", action="append", type=float, help="Repeat to select LR factors.")
    parser.add_argument("--tpu", default=None, help="Override TPU type for the selected cells.")
    parser.add_argument("--attempt", type=int, default=1, help="Attempt number for fresh restarts.")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; no submission.")
    parser.add_argument(
        "--launch-spacing-seconds",
        type=int,
        default=30,
        help="Seconds between submissions (Iris coordinator-collision guard).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    selected_bases = tuple(args.base or BASES.keys())
    selected_mixes = tuple(args.mix or DELPHI_MIDTRAIN_MIXES)
    selected_lrs = tuple(args.lr or LR_FACTORS)
    cells = list(_expand(selected_bases, selected_mixes, selected_lrs))
    print(f"{len(cells)} cells selected (bases={selected_bases}, mixes={selected_mixes}, lrs={selected_lrs})")

    for idx, (base_key, mix, lr) in enumerate(cells):
        spec = build_spec(base_key=base_key, mix=mix, lr_factor=lr, tpu_type=args.tpu, attempt=args.attempt)
        resolved = resolve_midtrain_spec(spec)
        validate_midtrain_spec(resolved)
        report = preflight(resolved)
        cell_label = f"[{idx + 1}/{len(cells)}] {spec.run.run_id}"
        if not report.ok:
            print(f"FAIL {cell_label}")
            for failure in report.failures:
                print(f"  fail: {failure}")
            return 1
        for warning in report.warnings:
            print(f"warn {cell_label}: {warning}")
        if args.dry_run:
            print(f"plan {cell_label}  (tpu={spec.compute.tpu_type}, steps={resolved.num_train_steps})")
            continue
        row = build_manifest_row(resolved, report, status="launched")
        write_manifest(row, output_path=spec.run.output_path)
        write_train_config(resolved)
        append_to_attempt_group(row, region=spec.run.output_region)
        submit_launch(build_launch_request(resolved))
        print(f"submitted {cell_label}  (steps={resolved.num_train_steps})")
        if idx + 1 < len(cells):
            time.sleep(args.launch_spacing_seconds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
