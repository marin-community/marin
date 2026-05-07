# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline VEP eval of the DNA bolinas parameter-scaling-sweep checkpoints.

For each of the 8 models produced by `run_parameter_scaling_sweep` in
`experiments.dna.exp109_bolinas_scaling_sweep`, this script:

1. Reconstructs the training ExecutorStep (without running it) to derive its
   GCS output path via the marin executor's content-hash addressing.
2. Confirms the final checkpoint exists at the expected step, where the
   expected step is read from the training step's own config (off-by-one for
   levanter's 0-indexed step convention). Incomplete runs are warned about
   and skipped — only fully-trained models are evaluated.
3. In eval mode, builds one `ExecutorStep` per eligible model and submits
   them via `executor_main`. Each step runs the TraitGym Mendelian VEP eval
   harness in its own TPU job, so all 8 evals can run in parallel and each
   gets a fresh wandb process (avoiding the global-state singleton bug from
   running multiple `run_eval_harness_main` calls in one process).

Modes (env vars):
    VERIFY_ONLY=yes              Resolve and print final checkpoint paths for
                                 all 8 models, validate region, and exit.
                                 Runs locally — no executor, no iris job.
    CHECKPOINT_FORMAT=levanter   Use the Levanter-format checkpoint at
                                 `<output>/checkpoints/step-N/` (default).
    CHECKPOINT_FORMAT=hf         Use the HF-format checkpoint at
                                 `<output>/hf/step-N/`.

`marin_prefix()` must resolve to the same prefix the sweep ran under (e.g.
`MARIN_PREFIX=gs://marin-us-east5`), and any sweep-time env vars that affect
tokenize/training step hashes (e.g. `PIN_TOKENIZE_REGION`,
`SCALING_TPU_TYPES`) must match the sweep run.

Submission:

    # Verify (local laptop, no iris):
    SSL_CERT_FILE=$(uv run python -c 'import certifi; print(certifi.where())') \\
    MARIN_PREFIX=gs://marin-us-east5 VERIFY_ONLY=yes \\
    uv run python experiments/dna/exp109_bolinas_sweep_eval.py

    # Eval (CPU coordinator on iris; per-step TPU workers):
    source .env && uv run iris --config lib/iris/examples/marin.yaml job run \\
        --no-wait --zone us-east5-a \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HUGGING_FACE_HUB_TOKEN $HUGGING_FACE_HUB_TOKEN \\
        -- python experiments/dna/exp109_bolinas_sweep_eval.py

The eval-mode iris invocation is bare on purpose — TPU and lm-eval extras
are declared per step via `remote(...)` (see `_build_eval_step`).
"""

import logging
import os
from dataclasses import dataclass

import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import discover_latest_checkpoint
from levanter.eval_harness import EvalHarnessMainConfig, LmEvalHarnessConfig, run_eval_harness_main
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import Executor, ExecutorStep, executor_main
from marin.execution.remote import remote
from rigging.filesystem import check_path_in_region, marin_prefix

from experiments.dna.exp109_bolinas_scaling_sweep import (
    SCALING_BATCH_SIZE,
    SCALING_HIDDEN_SIZES,
    SCALING_VERSION,
    TOKENIZER,
    ScalingTrainStep,
    _format_params,
    _model_seq_len,
    _scaling_target_tokens,
    build_scaling_train_steps,
)
from experiments.evals.task_configs import TRAITGYM_MENDELIAN_V2_255, convert_to_levanter_task_config

logger = logging.getLogger(__name__)

EVAL_VERSION = "v0.2"
EXPECTED_REGION = "us-east5"
EVAL_TPU_TYPE = "v5p-8"
EVAL_PIP_DEPENDENCY_GROUPS = ("tpu", "eval")
CHECKPOINT_FORMATS = ("levanter", "hf")


def _verify_only_mode() -> bool:
    value = os.getenv("VERIFY_ONLY", "no").lower()
    if value not in ("yes", "no"):
        raise ValueError(f"VERIFY_ONLY must be 'yes' or 'no', got {value!r}")
    return value == "yes"


def _checkpoint_format() -> str:
    value = os.getenv("CHECKPOINT_FORMAT", "levanter").lower()
    if value not in CHECKPOINT_FORMATS:
        raise ValueError(f"CHECKPOINT_FORMAT must be one of {CHECKPOINT_FORMATS}, got {value!r}")
    return value


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """A scaling-sweep training step paired with its checkpoint resolution.

    `final_checkpoint` is set only when training is fully complete (the latest
    Levanter checkpoint reached `expected_step`); incomplete runs leave it
    `None` so callers can warn-and-skip. `checkpoint_format` indicates which
    on-disk format (`"levanter"` or `"hf"`) `final_checkpoint` points at.
    """

    train: ScalingTrainStep
    output_path: str
    expected_step: int
    latest_step: int | None  # latest Levanter step actually present, for diagnostics
    final_checkpoint: str | None
    checkpoint_format: str


def _production_num_train_steps() -> int:
    """Production num_train_steps for the scaling sweep — used to construct the steps."""
    target_tokens = _scaling_target_tokens()
    return target_tokens // (SCALING_BATCH_SIZE * _model_seq_len())


def _expected_final_step(ts: ScalingTrainStep) -> int:
    """The step number levanter stamps onto the final checkpoint of a complete run.

    Levanter's train loop is ``while state.step < num_train_steps`` (see
    `lib/levanter/src/levanter/trainer.py:523`), so state.step ranges
    0..num_train_steps-1 and the last saved checkpoint is `step-{N-1}`.
    Reading num_train_steps off the step's own config keeps this in sync if
    the sweep config changes.
    """
    return ts.step.config.train_config.trainer.num_train_steps - 1


def _build_train_steps() -> list[ScalingTrainStep]:
    """Reproduce the production scaling sweep step list to derive matching paths."""
    return build_scaling_train_steps(num_train_steps=_production_num_train_steps(), hidden_sizes=SCALING_HIDDEN_SIZES)


def _build_executor() -> Executor:
    prefix = marin_prefix()
    return Executor(prefix=prefix, executor_info_base_path=os.path.join(prefix, "experiments"))


def _parse_step(checkpoint_dir: str) -> int:
    """Extract N from a path ending in ``step-N`` (or ``step-N/``)."""
    name = checkpoint_dir.rstrip("/").rsplit("/", 1)[-1]
    assert name.startswith("step-"), f"unexpected checkpoint dir name: {name!r}"
    return int(name[len("step-") :])


def _resolve_checkpoints(
    executor: Executor,
    train_steps: list[ScalingTrainStep],
    checkpoint_format: str,
) -> list[ResolvedCheckpoint]:
    """Resolve each training step to its final checkpoint, if complete.

    Uses `discover_latest_checkpoint` on the Levanter checkpoints dir (which
    has the per-step ``metadata.json`` it keys on) to find the actual final
    step number. A run is "complete" iff that step equals the expected final
    step the sweep was configured for. The returned `final_checkpoint` path is
    either the Levanter step dir as-is, or its HF sibling at
    ``.../hf/step-N``, depending on `checkpoint_format`.
    """
    resolved: list[ResolvedCheckpoint] = []
    for ts in train_steps:
        executor.compute_version(ts.step, is_pseudo_dep=False)
        output_path = executor.output_paths[ts.step]
        expected_step = _expected_final_step(ts)
        latest_levanter = discover_latest_checkpoint(os.path.join(output_path, "checkpoints"))

        latest_step: int | None = None
        final_checkpoint: str | None = None
        if latest_levanter is not None:
            latest_step = _parse_step(latest_levanter)
            if latest_step == expected_step:
                if checkpoint_format == "levanter":
                    final_checkpoint = latest_levanter
                else:  # "hf"
                    final_checkpoint = os.path.join(output_path, "hf", f"step-{latest_step}")

        resolved.append(
            ResolvedCheckpoint(
                train=ts,
                output_path=output_path,
                expected_step=expected_step,
                latest_step=latest_step,
                final_checkpoint=final_checkpoint,
                checkpoint_format=checkpoint_format,
            )
        )
    return resolved


def _status(r: ResolvedCheckpoint) -> tuple[str, str]:
    """Classify a resolved checkpoint. Returns (status, detail)."""
    if r.latest_step is None:
        return "MISSING", f"no Levanter checkpoints under {r.output_path}/checkpoints"
    if r.final_checkpoint is None:
        return "INCOMPLETE", f"latest step={r.latest_step}, expected={r.expected_step}"
    try:
        check_path_in_region("checkpoint", r.final_checkpoint, EXPECTED_REGION)
    except ValueError as e:
        return "REGION_MISMATCH", str(e)
    return "OK", r.final_checkpoint


def _print_verify_summary(resolved: list[ResolvedCheckpoint]) -> None:
    fmt = resolved[0].checkpoint_format
    print("=" * 110)
    print(f"Verify mode: final {fmt}-format checkpoints for scaling sweep {SCALING_VERSION}")
    print(f"Expected final step: {resolved[0].expected_step}   Expected region: {EXPECTED_REGION}")
    print("=" * 110)
    counts: dict[str, int] = {}
    for r in resolved:
        status, detail = _status(r)
        counts[status] = counts.get(status, 0) + 1
        label = f"h{r.train.hidden_size:<5} {r.train.num_params / 1e6:>7.1f}M"
        print(f"  {label}  [{status:<15}]  {detail}")
    print("=" * 110)
    summary = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
    print(f"Summary: {summary}")
    bad = sum(c for s, c in counts.items() if s != "OK")
    if bad:
        raise SystemExit(f"verify failed: {bad}/{len(resolved)} models not ready for eval")


def _build_eval_step(resolved: ResolvedCheckpoint) -> ExecutorStep:
    """Build a per-model eval ExecutorStep.

    The step name (`evals/dna/<run_name>`) ends with the wandb run name so
    GCS layout and wandb runs share a stable identifier. Each step runs in
    its own TPU job — separate process per eval, which avoids the wandb
    global-state singleton bug observed when calling `run_eval_harness_main`
    multiple times in one process.
    """
    assert resolved.final_checkpoint is not None, f"No final checkpoint for {resolved.train.run_name}"
    run_name = (
        f"dna-bolinas-scaling-eval-{SCALING_VERSION}"
        f"-p{_format_params(resolved.train.num_params)}"
        f"-vep-{EVAL_VERSION}"
    )
    eval_config = EvalHarnessMainConfig(
        eval_harness=LmEvalHarnessConfig(
            task_spec=convert_to_levanter_task_config([TRAITGYM_MENDELIAN_V2_255]),
            include_path="experiments/evals/custom_tasks",
            max_packed_segments=1,
        ),
        tokenizer=TOKENIZER,
        checkpoint_path=str(resolved.final_checkpoint),
        checkpoint_is_hf=(resolved.checkpoint_format == "hf"),
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                name=run_name,
                tags=[
                    "dna",
                    "bolinas",
                    "scaling",
                    "vep_eval",
                    SCALING_VERSION,
                    EVAL_VERSION,
                    f"hidden={resolved.train.hidden_size}",
                    f"params={resolved.train.num_params}",
                    f"step={resolved.expected_step}",
                    f"format={resolved.checkpoint_format}",
                ],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
        ),
        model=resolved.train.model_config,
    )
    return ExecutorStep(
        name=os.path.join("evals", "dna", run_name),
        fn=remote(
            run_eval_harness_main,
            resources=ResourceConfig.with_tpu(EVAL_TPU_TYPE),
            pip_dependency_groups=list(EVAL_PIP_DEPENDENCY_GROUPS),
        ),
        config=eval_config,
    )


def run_sweep_eval() -> None:
    checkpoint_format = _checkpoint_format()
    train_steps = _build_train_steps()
    executor = _build_executor()
    resolved = _resolve_checkpoints(executor, train_steps, checkpoint_format)

    if _verify_only_mode():
        _print_verify_summary(resolved)
        return

    eligible: list[ResolvedCheckpoint] = []
    for r in resolved:
        status, detail = _status(r)
        if status == "OK":
            eligible.append(r)
        else:
            logger.warning(
                "skipping h=%d (%.1fM params): %s — %s",
                r.train.hidden_size,
                r.train.num_params / 1e6,
                status,
                detail,
            )
    if not eligible:
        raise RuntimeError(
            f"No eligible checkpoints to evaluate (0/{len(resolved)}). Run with VERIFY_ONLY=yes for details."
        )

    eval_steps = [_build_eval_step(r) for r in eligible]
    logger.info("Submitting %d eval steps (skipped %d non-OK)", len(eval_steps), len(resolved) - len(eval_steps))
    executor_main(
        steps=eval_steps,
        description=f"DNA Bolinas scaling-eval {SCALING_VERSION} ({EVAL_VERSION})",
    )


if __name__ == "__main__":
    run_sweep_eval()
