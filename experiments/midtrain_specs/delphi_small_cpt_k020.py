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

Usage — one cell per invocation, three selectors REQUIRED:

    uv run python experiments/midtrain_specs/delphi_small_cpt_k020.py \\
        --base 3e18 --mix p33m67 --lr 0.5

The script refuses to run without all three selectors so the redesign-doc
rule "one logical training cell per launcher invocation" cannot be
accidentally violated. To run a multi-cell sweep, loop in the shell /
driver, not in this script.
"""

import argparse
import logging
import os
import re
import time

import draccus
import levanter.config  # noqa: F401 — side effect: registers draccus codecs (timedelta, RepoRef, etc.)
from iris.cluster.client.job_info import get_job_info
from iris.cluster.constraints import WellKnownAttribute
from levanter.optim import AdamHConfig
from marin.midtraining import (
    CPT_DEFAULT_DECAY,
    CPT_DEFAULT_WARMUP_FRACTION,
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
    DELPHI_3E20,
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
    "3e20": DELPHI_3E20,
}

LR_FACTORS: tuple[float, ...] = (0.33, 0.5, 0.67, 0.83)

# Per-base TPU recommendation. Smaller bases default to v5p-8; larger bases use
# larger slices where v5p-8 is not allowlisted or would be too slow.
# Override per-cell via --tpu, but only within ALLOWED_TPUS_PER_BASE.
DEFAULT_TPU: dict[str, str] = {
    "3e18": "v5p-8",
    "9e18": "v5p-8",
    "2e19": "v5p-8",
    "3e19": "v5p-8",
    "9e19": "v5p-16",
    "2e20": "v5p-32",
    "3e20": "v5p-16",
}

# Per-base TPU allowlist. Picking a size outside this set fails preflight so
# we don't quietly run a 2e20 cell on a v5p-8 (too small to fit) or burn a
# v5p-64 on a 3e18 cell. Extending the allowlist is a deliberate edit.
#
# The v6e entries below are deliberately for short throughput/HBM probes, not
# endorsements for full sweeps. Keep probe runs visibly tagged/suffixed so W&B
# charts do not get mixed into quality comparisons by accident.
ALLOWED_TPUS_PER_BASE: dict[str, frozenset[str]] = {
    "3e18": frozenset({"v5p-8", "v5p-16"}),
    "9e18": frozenset({"v5p-8", "v5p-16", "v6e-4", "v6e-8"}),
    "2e19": frozenset({"v5p-8", "v5p-16", "v6e-4", "v6e-8"}),
    "3e19": frozenset({"v5p-8", "v5p-16", "v5p-32", "v6e-4", "v6e-8"}),
    "9e19": frozenset({"v5p-16", "v5p-32", "v5p-64", "v6e-8"}),
    "2e20": frozenset({"v5p-8", "v5p-16", "v5p-32", "v5p-64"}),
    "3e20": frozenset({"v5p-16", "v5p-32", "v5p-64"}),
}

# Identity env vars that the redesign doc forbids — run_id must come from
# --base/--mix/--lr/--attempt, never from the shell. Refuse to launch if any
# are present so we never silently inherit a stale run id from a parent shell.
DISALLOWED_ENV_NAMES: frozenset[str] = frozenset({"RUN_ID", "WANDB_RUN_ID"})
DISALLOWED_ENV_PREFIXES: tuple[str, ...] = ("MIDTRAIN_", "TRUE_MIDTRAIN_")
RUN_SUFFIX_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")

REGION: str = "us-east5"
WANDB_PROJECT: str = "delphi-midtraining"
BUDGET: BudgetPolicy = BudgetPolicy.pretrain_fraction(0.20)
DEFAULT_CONTAINER_RAM: str = "256g"


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------


def _data_section_for_mix(mix: str) -> dict:
    """Load the canonical 1e21 data section, returning a fresh copy each call."""
    return load_legacy_data_section(mix)


def _model_config_for_base(base: DelphiModel) -> dict:
    """Render the Qwen3 architecture dict from the heuristic at base.hidden_dim.

    Uses ``draccus.encode`` so nested ChoiceRegistry types (e.g. the rope
    config) get their ``type:`` discriminator key. Prepends ``type: qwen3``
    at the top level (draccus.encode emits the body without the registry
    tag for the top-level instance).
    """
    cfg = completed_adamh_heuristic._build_model_config(hidden_size=base.hidden_dim)
    return {"type": "qwen3", **draccus.encode(cfg)}


def _optimizer_config_for_base(base: DelphiModel, *, lr_factor: float) -> dict:
    """Build an AdamH config dict from the base's pretrain hparams + CPT lr_factor.

    Non-LR / non-schedule fields are pretrain-verbatim from
    ``experiments/delphi_models.py`` (which itself reads from ``.executor_info``).
    Only ``learning_rate`` and ``adam_lr`` are scaled by ``lr_factor``; ``beta*``,
    ``epsilon``, ``max_grad_norm``, ``weight_decay``, ``min_lr_ratio``,
    ``lr_schedule``, and ``nesterov`` pass through verbatim from the base.

    **Warmup and decay are CPT-policy values**, not pretrain values: every CPT
    cell uses ``CPT_DEFAULT_WARMUP_FRACTION`` (0.10) and ``CPT_DEFAULT_DECAY``
    (``None`` — Levanter sentinel for full decay after warmup). In Levanter
    this produces a strict triangular schedule (10% linear warmup → 90%
    linear decay to ``min_lr_ratio * peak``; no stable plateau) for every
    ``num_train_steps`` value, matching the legacy Delphi CPT triangular
    schedule from ``exp_delphi_math_10b_midtrain.py`` while avoiding that
    file's fixed-token warmup degeneration on small bases. **Do not** pass a
    float for ``decay`` (e.g. ``0.9``): Levanter floors fractional stage
    lengths independently, so some step counts can get a 1-step stable
    plateau. See the constant docstring in
    ``lib/marin/src/marin/midtraining/modes.py`` for the rationale.
    """
    cfg = AdamHConfig(
        learning_rate=base.peak_lr * lr_factor,
        adam_lr=base.peak_adam_lr * lr_factor,
        beta1=base.beta1,
        beta2=base.beta2,
        epsilon=base.epsilon,
        max_grad_norm=base.max_grad_norm,
        weight_decay=base.weight_decay,
        warmup=CPT_DEFAULT_WARMUP_FRACTION,
        decay=CPT_DEFAULT_DECAY,
        min_lr_ratio=base.min_lr_ratio,
        lr_schedule=base.lr_schedule,
        nesterov=base.nesterov,
    )
    return {"type": "adamH", **draccus.encode(cfg)}


def _logical_cell_id(base_key: str, mix: str, lr_factor: float, *, run_suffix: str | None = None) -> str:
    cell_id = f"delphi-{base_key}-{mix}-k0p20-lr{round(lr_factor * 100):02d}"
    if run_suffix is None:
        return cell_id
    _check_run_suffix(run_suffix)
    return f"{cell_id}-{run_suffix}"


def build_spec(
    *,
    base_key: str,
    mix: str,
    lr_factor: float,
    tpu_type: str | None = None,
    attempt: int = 1,
    run_suffix: str | None = None,
    probe_steps: int | None = None,
    ram: str = DEFAULT_CONTAINER_RAM,
) -> MidtrainSpec:
    """Construct a :class:`MidtrainSpec` for one cell of the sweep."""
    _check_selectors(base_key=base_key, mix=mix, lr_factor=lr_factor)
    _check_probe_args(run_suffix=run_suffix, probe_steps=probe_steps)
    base = BASES[base_key]
    tpu = tpu_type or DEFAULT_TPU[base_key]
    _check_tpu_allowed(base_key, tpu)
    cell_id = _logical_cell_id(base_key, mix, lr_factor, run_suffix=run_suffix)
    run = build_run_identity(
        logical_cell_id=cell_id,
        attempt=attempt,
        output_region_name=REGION,
        wandb_project=WANDB_PROJECT,
    )
    return MidtrainSpec(
        base=base,
        run=run,
        compute=ComputeProfile(tpu_type=tpu, batch_size=base.batch_size, ram=ram, regions=(REGION,)),
        mode=CptMode(
            init=CptInit(
                source_kind=CheckpointSourceKind.HF_WEIGHTS,
                hf_repo=base.hf_repo,
                hf_revision=base.hf_revision,
            ),
            budget=_budget_policy(probe_steps),
        ),
        tokenizer=LLAMA3_TOKENIZER,
        model_config=_model_config_for_base(base),
        optimizer_config=_optimizer_config_for_base(base, lr_factor=lr_factor),
        data_section_override=_data_section_for_mix(mix),
        data_section_provenance=LEGACY_PROVENANCE[mix],
        banned_substrings=frozenset(DELPHI_BANNED_SUBSTRINGS),
        extra_tags=tuple(
            tag
            for tag in (
                "sweep:delphi-small-cpt-k020",
                f"mix:{mix}",
                f"run_suffix:{run_suffix}" if run_suffix else None,
                "probe:throughput-hbm" if probe_steps is not None else None,
                f"probe_steps:{probe_steps}" if probe_steps is not None else None,
                "do_not_compare:quality" if probe_steps is not None else None,
            )
            if tag is not None
        ),
    )


def _budget_policy(probe_steps: int | None) -> BudgetPolicy:
    if probe_steps is None:
        return BUDGET
    return BudgetPolicy.fixed_steps(probe_steps, label=f"probe{probe_steps}steps")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _reject_stale_env_vars() -> None:
    """Refuse to run if any identity-overriding env var is set.

    The redesign doc requires run_id to derive solely from the CLI selectors
    (--base/--mix/--lr/--attempt). A stray env var in the calling shell
    silently rewriting the run_id was a recurring foot-gun in earlier
    iterations.
    """
    bad = sorted(
        k for k in os.environ if k in DISALLOWED_ENV_NAMES or any(k.startswith(p) for p in DISALLOWED_ENV_PREFIXES)
    )
    if bad:
        raise RuntimeError(
            f"Refusing to launch with identity env vars set: {bad}. "
            "The launcher derives run_id and W&B id from --base/--mix/--lr/--attempt only. "
            "Unset these before invoking."
        )


def _reject_preemptible_coordinator() -> None:
    """Refuse a parent coordinator running on preemptible Iris capacity."""
    info = get_job_info()
    if info is None:
        return
    if info.worker_id and "-preemptible-" in info.worker_id:
        raise RuntimeError(
            f"Refusing to launch a midtraining coordinator on preemptible worker {info.worker_id!r}. "
            "Launch the outer Iris job as CPU-only without --preemptible (or pass --no-preemptible). "
            "The nested TPU training child remains preemptible; the coordinator must stay alive to "
            "avoid cascade-killing the child."
        )
    for constraint in info.constraints:
        if constraint.key == WellKnownAttribute.PREEMPTIBLE and constraint.values[0].value == "true":
            raise RuntimeError(
                "Refusing to launch a midtraining coordinator with the Iris constraint preemptible=true. "
                "Remove --preemptible from the outer CPU job (or pass --no-preemptible). The nested TPU "
                "training child is the preemptible job; the coordinator should be stable CPU capacity."
            )


def _is_iris_retry_attempt() -> bool:
    info = get_job_info()
    return info is not None and info.attempt_id > 0


def _check_selectors(*, base_key: str, mix: str, lr_factor: float) -> None:
    if base_key not in BASES:
        raise ValueError(f"Unknown base {base_key!r}; expected one of {sorted(BASES)}")
    if mix not in DELPHI_MIDTRAIN_MIXES:
        raise ValueError(f"Unknown mix {mix!r}; expected one of {sorted(DELPHI_MIDTRAIN_MIXES)}")
    if lr_factor not in LR_FACTORS:
        raise ValueError(f"Unknown lr_factor {lr_factor!r}; expected one of {LR_FACTORS}")


def _check_tpu_allowed(base_key: str, tpu: str) -> None:
    allowed = ALLOWED_TPUS_PER_BASE[base_key]
    if tpu not in allowed:
        raise RuntimeError(
            f"TPU {tpu!r} is not in the allowlist for base {base_key}: {sorted(allowed)}. "
            "Either pick an allowed TPU or extend ALLOWED_TPUS_PER_BASE deliberately."
        )


def _check_run_suffix(run_suffix: str) -> None:
    if not RUN_SUFFIX_PATTERN.match(run_suffix):
        raise ValueError(
            f"run_suffix {run_suffix!r} must match {RUN_SUFFIX_PATTERN.pattern}; "
            "use lowercase letters, digits, and hyphens only."
        )
    if re.search(r"-a[0-9]{3}$", run_suffix):
        raise ValueError("run_suffix must not end with an attempt suffix like '-a001'")


def _check_probe_args(*, run_suffix: str | None, probe_steps: int | None) -> None:
    if probe_steps is None:
        return
    if probe_steps <= 0:
        raise ValueError(f"probe_steps must be positive, got {probe_steps!r}")
    if run_suffix is None or not run_suffix.startswith("probe-"):
        raise ValueError("probe_steps requires a run_suffix starting with 'probe-'")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Launch ONE Delphi K=0.20 midtraining cell. "
            "Multi-cell sweeps must loop in the driver shell — never in this script."
        )
    )
    parser.add_argument("--base", required=True, choices=list(BASES), help="Pretrain base flops key.")
    parser.add_argument(
        "--mix",
        required=True,
        choices=list(DELPHI_MIDTRAIN_MIXES),
        help="Midtraining data mixture id (p33m67 / p50m50 / p67m33).",
    )
    parser.add_argument(
        "--lr",
        required=True,
        type=float,
        choices=list(LR_FACTORS),
        help="LR multiplier on the base's pretrain peak LR.",
    )
    parser.add_argument(
        "--tpu",
        default=None,
        help="Override TPU type. Defaults to DEFAULT_TPU[base]; must be in ALLOWED_TPUS_PER_BASE[base].",
    )
    parser.add_argument("--attempt", type=int, default=1, help="Attempt number for fresh restarts.")
    parser.add_argument(
        "--ram",
        default=DEFAULT_CONTAINER_RAM,
        help="Container RAM for the TPU child. Defaults to 256g after 9e18/2e19 HF-save exit-137 OOMs.",
    )
    parser.add_argument(
        "--run-suffix",
        default=None,
        help="Optional safe suffix appended before -aNNN in the W&B/run id, e.g. bench-v6e4.",
    )
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=None,
        help=(
            "Run a clearly tagged fixed-step hardware probe instead of the full K=0.20 budget. "
            "Requires --run-suffix probe-*."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan only; no submission.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    _reject_stale_env_vars()
    if not args.dry_run:
        _reject_preemptible_coordinator()

    spec = build_spec(
        base_key=args.base,
        mix=args.mix,
        lr_factor=args.lr,
        tpu_type=args.tpu,
        attempt=args.attempt,
        run_suffix=args.run_suffix,
        probe_steps=args.probe_steps,
        ram=args.ram,
    )
    resolved = resolve_midtrain_spec(spec)
    validate_midtrain_spec(resolved)
    report = preflight(resolved, allow_existing_matching_manifest=_is_iris_retry_attempt())
    label = spec.run.run_id
    if not report.ok:
        print(f"FAIL {label}")
        for failure in report.failures:
            print(f"  fail: {failure}")
        return 1
    for warning in report.warnings:
        print(f"warn {label}: {warning}")

    if args.dry_run:
        print(f"plan {label}  (tpu={spec.compute.tpu_type}, ram={spec.compute.ram}, steps={resolved.num_train_steps})")
        return 0

    row = build_manifest_row(resolved, report, status="launched")
    write_manifest(row, output_path=spec.run.output_path)
    write_train_config(resolved)
    append_to_attempt_group(row, region=spec.run.output_region)
    result = submit_launch(build_launch_request(resolved))
    print(f"submitted {label}  (steps={resolved.num_train_steps}, tpu={spec.compute.tpu_type}, ram={spec.compute.ram})")

    # Block on this single child until it terminates. The iris executor
    # cascade-kills children when their parent reaches SUCCEEDED
    # (lib/iris/.../transitions.py — "Succeeded jobs always cascade"), so the
    # parent must outlive the child. With one-cell-per-invocation each iris
    # job run is its own top-level coordinator that lives for exactly one
    # training run.
    while True:
        try:
            status = result.wait(raise_on_failure=False)
        except Exception as exc:
            print(f"  {label}: wait raised {exc!r}; retrying so the coordinator does not exit early")
            time.sleep(30)
            continue
        status_value = getattr(status, "value", str(status))
        if status_value in {"pending", "running"}:
            print(f"  {label}: wait returned non-terminal status {status_value!r}; retrying")
            time.sleep(30)
            continue
        print(f"  {label}: {status_value}")
        return 0 if status_value == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
