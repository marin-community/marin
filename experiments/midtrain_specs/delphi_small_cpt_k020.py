# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delphi CPT midtraining launcher across the 3e18 -> 1e22 ladder.

Budget: defaults to K=0.20 of each base's own pretrain tokens
(``pretrain_fraction(0.20)``, iso-FLOP — bigger bases get more tokens). Pass
``--budget-tokens N`` for a fixed math-token budget shared across every base
(iso-token ladder: ``num_train_steps = round(N / (batch_size * 4096))`` per
base). The cell id then carries ``tok<label>`` (e.g. ``tok1b``) instead of
``k0p20`` so iso-token and K=0.20 runs never collide.

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
    default_budget_label,
    preflight,
    resolve_midtrain_spec,
    submit_launch,
    validate_midtrain_spec,
    write_manifest,
    write_train_config,
)

from experiments.delphi_models import (
    DELPHI_1E21,
    DELPHI_1E22,
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
    "1e21": DELPHI_1E21,
    "1e22": DELPHI_1E22,
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
    "1e21": "v5p-64",
    "1e22": "v5p-256",
}

# Per-base TPU allowlist. Picking a size outside this set fails preflight so
# we don't quietly run a 2e20 cell on a v5p-8 (too small to fit) or burn a
# v5p-64 on a 3e18 cell. Extending the allowlist is a deliberate edit.
#
# The v6e entries below are deliberately for short throughput/HBM probes, not
# endorsements for full sweeps. Keep probe runs visibly tagged/suffixed so W&B
# charts do not get mixed into quality comparisons by accident.
ALLOWED_TPUS_PER_BASE: dict[str, frozenset[str]] = {
    # v4 entries: 32 GB HBM/chip (v4-8 = 4 chips/128 GB, v4-32 = 16 chips/512 GB).
    # CPT loads the model from HF, so any region's slices work; cap activation
    # memory with --per-device-parallelism on the bigger bases (see iso-token
    # ladder notes in .agents/logbooks/debug_midtrain.md).
    "3e18": frozenset({"v5p-8", "v5p-16", "v6e-8", "v4-8"}),
    "9e18": frozenset({"v5p-8", "v5p-16", "v5p-32", "v6e-4", "v6e-8", "v4-8"}),
    "2e19": frozenset({"v5p-8", "v5p-16", "v5p-32", "v6e-4", "v6e-8", "v4-8"}),
    "3e19": frozenset({"v5p-8", "v5p-16", "v5p-32", "v6e-4", "v6e-8", "v4-8"}),
    "9e19": frozenset({"v5p-8", "v5p-16", "v5p-32", "v5p-64", "v6e-8", "v4-8", "v4-16"}),
    "2e20": frozenset({"v5p-8", "v5p-16", "v5p-32", "v5p-64", "v4-8", "v4-16"}),
    "3e20": frozenset({"v5p-8", "v5p-16", "v5p-32", "v5p-64", "v6e-8", "v4-8", "v4-16", "v4-32"}),
    "1e21": frozenset(
        {"v5p-8", "v5p-16", "v5p-32", "v5p-64", "v5p-128", "v5p-256", "v5p-512", "v4-8", "v4-16", "v4-32"}
    ),
    "1e22": frozenset(
        {"v5p-8", "v5p-16", "v5p-32", "v5p-64", "v5p-128", "v5p-256", "v5p-512"}
        | {"v4-16", "v4-32", "v4-64", "v4-128", "v6e-16", "v6e-32"}
    ),
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


def _logical_cell_id(
    base_key: str, mix: str, lr_factor: float, *, budget_tag: str = "k0p20", run_suffix: str | None = None
) -> str:
    cell_id = f"delphi-{base_key}-{mix}-{budget_tag}-lr{round(lr_factor * 100):02d}"
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
    budget_tokens: int | None = None,
    ram: str = DEFAULT_CONTAINER_RAM,
    per_device_parallelism: int = -1,
    region: str = REGION,
    child_preemptible: bool = True,
    expected_min_step: int | None = None,
) -> MidtrainSpec:
    """Construct a :class:`MidtrainSpec` for one cell of the sweep.

    ``region`` sets the TPU compute region and the checkpoint output region. The
    math/val data block is region-independent (it pins the us-east5 cache for the
    byte-identical val carve-out), and CPT streams model weights from HF, so a
    cross-region ``region`` only moves compute + checkpoints, not the model load.
    """
    _check_selectors(base_key=base_key, mix=mix, lr_factor=lr_factor)
    _check_probe_args(run_suffix=run_suffix, probe_steps=probe_steps)
    _check_budget_args(probe_steps=probe_steps, budget_tokens=budget_tokens)
    base = BASES[base_key]
    tpu = tpu_type or DEFAULT_TPU[base_key]
    _check_tpu_allowed(base_key, tpu)
    budget_tag = f"tok{default_budget_label(budget_tokens)}" if budget_tokens is not None else "k0p20"
    cell_id = _logical_cell_id(base_key, mix, lr_factor, budget_tag=budget_tag, run_suffix=run_suffix)
    run = build_run_identity(
        logical_cell_id=cell_id,
        attempt=attempt,
        output_region_name=region,
        wandb_project=WANDB_PROJECT,
    )
    return MidtrainSpec(
        base=base,
        run=run,
        expected_min_step=expected_min_step,
        compute=ComputeProfile(
            tpu_type=tpu,
            batch_size=base.batch_size,
            ram=ram,
            per_device_parallelism=per_device_parallelism,
            regions=(region,),
            preemptible=child_preemptible,
        ),
        mode=CptMode(
            init=CptInit(
                source_kind=CheckpointSourceKind.HF_WEIGHTS,
                hf_repo=base.hf_repo,
                hf_revision=base.hf_revision,
            ),
            budget=_budget_policy(probe_steps, budget_tokens),
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
                "sweep:delphi-cpt-isotoken" if budget_tokens is not None else "sweep:delphi-small-cpt-k020",
                f"budget:{budget_tag}" if budget_tokens is not None else None,
                f"mix:{mix}",
                f"run_suffix:{run_suffix}" if run_suffix else None,
                "probe:throughput-hbm" if probe_steps is not None else None,
                f"probe_steps:{probe_steps}" if probe_steps is not None else None,
                "do_not_compare:quality" if probe_steps is not None else None,
            )
            if tag is not None
        ),
    )


def _budget_policy(probe_steps: int | None, budget_tokens: int | None) -> BudgetPolicy:
    if probe_steps is not None:
        return BudgetPolicy.fixed_steps(probe_steps, label=f"probe{probe_steps}steps")
    if budget_tokens is not None:
        return BudgetPolicy.fixed_tokens(budget_tokens, label=default_budget_label(budget_tokens))
    return BUDGET


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


def _check_budget_args(*, probe_steps: int | None, budget_tokens: int | None) -> None:
    if probe_steps is not None and budget_tokens is not None:
        raise ValueError("Pass at most one of --probe-steps / --budget-tokens; they select different CPT budgets.")
    if budget_tokens is not None and budget_tokens <= 0:
        raise ValueError(f"budget_tokens must be positive, got {budget_tokens!r}")


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
        "--per-device-parallelism",
        type=int,
        default=-1,
        help=(
            "Levanter trainer.per_device_parallelism (per-device microbatch; -1 = auto = "
            "global_batch / num_devices). Set a small positive value to enable gradient "
            "accumulation when the auto per-device batch is too large for the slice — e.g. "
            "1e22 (batch 1024) on small slices overflows int32 activation indexing; use 16."
        ),
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
    parser.add_argument(
        "--budget-tokens",
        type=int,
        default=None,
        help=(
            "Iso-token CPT: train a fixed math-token budget shared across every base "
            "(e.g. 1000000000 for 1B) instead of the default K=0.20 pretrain-fraction budget. "
            "num_train_steps = round(budget_tokens / (batch_size * 4096)) per base. "
            "The cell id carries tok<label> (e.g. tok1b). Mutually exclusive with --probe-steps."
        ),
    )
    parser.add_argument(
        "--region",
        default=REGION,
        help=(
            "Compute + checkpoint-output region (default us-east5). The data block stays the "
            "us-east5 cache (byte-identical val) and CPT loads the model from HF, so a cross-region "
            "value only moves compute + checkpoints, not the model load. Use when the default region "
            "is capacity/quota-blocked (e.g. --region us-central1)."
        ),
    )
    parser.add_argument(
        "--no-child-preempt",
        dest="child_preemptible",
        action="store_false",
        default=True,
        help=(
            "Request the TPU child on non-preemptible (reserved) capacity instead of preemptible. "
            "Use for long single runs that thrash on preemption-recovery (e.g. 9.7B 1e22), at the cost "
            "of competing for scarcer reserved quota."
        ),
    )
    parser.add_argument(
        "--expected-min-step",
        type=int,
        default=None,
        help=(
            "Resume an existing run id (same --attempt) from its GCS checkpoint instead of refusing the "
            "fresh launch. Set to a step at or below the latest saved checkpoint; preflight verifies a "
            "checkpoint >= this step exists, then training resumes from it (GCS ckpt + W&B step counter). "
            "Use to move a preempted cell onto fresh capacity without losing progress."
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
        budget_tokens=args.budget_tokens,
        ram=args.ram,
        per_device_parallelism=args.per_device_parallelism,
        region=args.region,
        child_preemptible=args.child_preemptible,
        expected_min_step=args.expected_min_step,
    )
    resolved = resolve_midtrain_spec(spec)
    validate_midtrain_spec(resolved, allow_cross_region_data=args.region != REGION)
    report = preflight(
        resolved,
        allow_existing_matching_manifest=_is_iris_retry_attempt(),
        allow_cross_region_data=args.region != REGION,
    )
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
