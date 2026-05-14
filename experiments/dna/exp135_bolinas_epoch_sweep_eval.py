# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline TraitGym eval for every Levanter checkpoint of every mix in exp135.

One Fray job per ``(run, step)``. Results land at
``gs://<tmp>/<run>/step-<N>.json`` (a 30-day TTL temp bucket).
Re-runs of this script skip evals whose JSON already exists; the worker
double-checks before submitting any compute. A final merge produces
``gs://<tmp>/merged.json``.

XLA compilation is reused across eval jobs via the shared
``JAX_COMPILATION_CACHE_DIR`` that ``resolve_training_env`` sets by default.
Each fresh ``(model arch, mesh)`` pair pays one compile; every subsequent
checkpoint with the same shape hits the cache.

Inspired by ``experiments/grug/base/launch.py`` + ``experiments/grug/dispatch.py``:
no Executor, no status files — work-done signal is just the result JSON's
presence at a deterministic path.

Submitting
----------

The script runs as a CPU-only iris coordinator that submits one Fray TPU
sub-job per checkpoint. Region must be pinned to where exp135 wrote its
checkpoints (``us-east5``) so ``marin_prefix()`` resolves to the right
bucket via the GCE metadata server.

Full sweep::

    .venv/bin/iris --cluster marin job run \\
        --user eczech \\
        --memory 16g --enable-extra-resources \\
        --extra cpu \\
        --region us-east5 \\
        --no-wait \\
        -e JAX_PLATFORMS cpu \\
        -e PYTHONUNBUFFERED 1 \\
        --job-name exp135-eval-$(date -u +%Y%m%d-%H%M%S) \\
        -- python -u experiments/dna/exp135_bolinas_epoch_sweep_eval.py

Subsetting (e.g. for a smoke / single-checkpoint warmup)::

    -e EVAL_MIX_NAMES downstream_only \\
    -e EVAL_STEPS 125 \\

Env knobs:

- ``EVAL_MIX_NAMES``  Comma-separated mix names. Defaults to all in
  ``MIX_CONFIGS``.
- ``EVAL_STEPS``  Comma-separated checkpoint steps. Defaults to all.
"""

import json
import os
from collections.abc import Callable, Sequence
from typing import Any

import fsspec
import jax
import jmp
from fray import ResourceConfig
from fray import client as fray_client
from fray.types import Entrypoint, JobRequest, create_environment
from levanter.eval_harness import EvalHarnessMainConfig, LmEvalHarnessConfig, run_eval_harness_main
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.py_utils import FailSafeJSONEncoder
from marin.execution.remote import _sanitize_job_name
from marin.training.training import extras_for_resources, resolve_training_env
from rigging.filesystem import marin_prefix, marin_temp_bucket

from experiments.dna.exp135_bolinas_epoch_sweep import (
    EPOCHS,
    MIX_CONFIGS,
    TOKENIZER,
    _format_params,
    _model_config,
    _num_params,
)
from experiments.dna.exp135_bolinas_epoch_sweep import (
    VERSION as TRAIN_VERSION,
)
from experiments.evals.task_configs import TRAITGYM_MENDELIAN_V2_255, convert_to_levanter_task_config

# =============================================================================
# Constants
# =============================================================================

EVAL_VERSION = "v0.3"
TMP_BUCKET = marin_temp_bucket(ttl_days=30, prefix=f"exp135-eval-{EVAL_VERSION}")
EVAL_RESOURCES = ResourceConfig.with_tpu("v6e-4", ram="64g")
FS = fsspec.filesystem("gs")


# =============================================================================
# Discovery
# =============================================================================


def _run_name(idx: int, mix_name: str) -> str:
    params_label = _format_params(_num_params())
    return f"dna-bolinas-epoch-{TRAIN_VERSION}-p{params_label}-e{EPOCHS}-i{idx}-{mix_name}"


def _result_path(run: str, step: int) -> str:
    return f"{TMP_BUCKET}/{run}/step-{step}.json"


def _find_checkpoints(run: str) -> list[tuple[int, str]]:
    """Locate the training output dir for ``run`` and list its complete ``step-N/`` dirs.

    Training writes to ``{marin_prefix()}/checkpoints/{run}-{hash}/``. Marin's
    ``bake_output_path`` pins the Levanter checkpointer's ``base_path`` to a
    ``checkpoints/`` subdir, so step dirs land at
    ``{train_dir}/checkpoints/step-N/``. ``metadata.json`` is written last by
    Levanter — its presence marks a complete checkpoint.

    Uses ``fs.ls`` (one-level listing, client-side filter) rather than
    ``fs.glob`` because globs expand wildcards via additional listings that
    have stalled for many minutes on the marin checkpoints prefix.
    """
    entries = FS.ls(f"{marin_prefix()}/checkpoints", detail=False)
    matches = [e for e in entries if e.rsplit("/", 1)[-1].startswith(f"{run}-")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one training output for {run}, got {matches}")
    [train_dir] = matches
    out = []
    for d in FS.ls(f"{train_dir}/checkpoints", detail=False):
        name = d.rsplit("/", 1)[-1]
        if not name.startswith("step-"):
            continue
        if FS.exists(f"{d}/metadata.json"):
            out.append((int(name.split("-", 1)[1]), f"gs://{d}"))
    return sorted(out)


# =============================================================================
# Fray submission (copied from experiments.defaults._submit_train_job)
# =============================================================================


def _submit_eval_job(
    name: str,
    entrypoint_callable: Callable[..., None],
    args: Sequence[Any],
    resources: ResourceConfig,
    env_vars: dict[str, str] | None,
):
    """Resolve env, build a JobRequest, submit to Iris, return the handle.

    Does NOT wait — caller is responsible for ``handle.wait(...)`` so the
    coordinator can fan out many sub-jobs in parallel.

    Worker extras: ``extras_for_resources(resources)`` (e.g. ``tpu``) plus
    ``lm_eval`` so the eval harness library is available on the TPU pod.
    Mirrors the training step's ``pip_dependency_groups=["tpu", "lm_eval"]``.
    """
    resolved_env_vars = dict(env_vars or {})
    env = resolve_training_env(resolved_env_vars, resources)
    extras = [*extras_for_resources(resources), "lm_eval"]
    job_request = JobRequest(
        name=_sanitize_job_name(name),
        entrypoint=Entrypoint.from_callable(entrypoint_callable, args=list(args)),
        resources=resources,
        environment=create_environment(env_vars=env, extras=extras),
    )
    return fray_client.current_client().submit(job_request)


# =============================================================================
# Worker entrypoint
# =============================================================================


def _eval_on_worker(run: str, step: int, ckpt_path: str, out_path: str) -> None:
    """Load one Levanter checkpoint, run the harness, persist results JSON."""
    if FS.exists(out_path):
        return

    outputs = run_eval_harness_main(
        EvalHarnessMainConfig(
            eval_harness=LmEvalHarnessConfig(
                task_spec=convert_to_levanter_task_config([TRAITGYM_MENDELIAN_V2_255]),
                include_path="experiments/evals/custom_tasks",
                max_packed_segments=1,
            ),
            tokenizer=TOKENIZER,
            checkpoint_path=ckpt_path,
            checkpoint_is_hf=False,
            trainer=TrainerConfig(tracker=NoopConfig(), mp=jmp.get_policy("p=f32,c=bfloat16")),
            model=_model_config(),
        )
    )
    if jax.process_index() != 0 or outputs is None:
        return
    with FS.open(out_path, "w") as f:
        # outputs["configs"] holds task class references; plain json.dump chokes on
        # ABCMeta. FailSafeJSONEncoder is the same encoder Levanter uses internally.
        json.dump(
            {"run": run, "step": step, "ckpt": ckpt_path, "results": outputs},
            f,
            indent=2,
            cls=FailSafeJSONEncoder,
        )


# =============================================================================
# Submit / merge
# =============================================================================


def _submit_one(run: str, step: int, ckpt_path: str):
    """Submit one eval Fray sub-job; return the handle, or ``None`` if the
    result JSON is already present (idempotence skip)."""
    out_path = _result_path(run, step)
    if FS.exists(out_path):
        return None
    return _submit_eval_job(
        name=f"exp135-eval-{run}-step-{step}",
        entrypoint_callable=_eval_on_worker,
        args=[run, step, ckpt_path, out_path],
        resources=EVAL_RESOURCES,
        env_vars={
            "WANDB_MODE": "disabled",
            "HF_DATASETS_TRUST_REMOTE_CODE": "1",
            "HF_ALLOW_CODE_EVAL": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
    )


def _merge() -> str:
    rows = []
    for p in sorted(FS.glob(f"{TMP_BUCKET}/**/step-*.json")):
        with FS.open(p) as f:
            rows.append(json.load(f))
    out = f"{TMP_BUCKET}/merged.json"
    with FS.open(out, "w") as f:
        json.dump(rows, f, indent=2)
    return out


def _selected_mixes() -> list[tuple[int, "object"]]:
    raw = os.getenv("EVAL_MIX_NAMES")
    if not raw:
        return list(enumerate(MIX_CONFIGS))
    wanted = {s.strip() for s in raw.split(",")}
    return [(i, m) for i, m in enumerate(MIX_CONFIGS) if m.name in wanted]


def _selected_steps(steps: list[tuple[int, str]]) -> list[tuple[int, str]]:
    raw = os.getenv("EVAL_STEPS")
    if not raw:
        return steps
    wanted = {int(s.strip()) for s in raw.split(",")}
    return [s for s in steps if s[0] in wanted]


def main() -> None:
    # Fan-out: submit every eval first, collect handles, then wait on all.
    # Handles are waited in submission order — if an earlier job is slowest
    # we block until it finishes, but every other job is running in parallel.
    handles = []
    for idx, mix in _selected_mixes():
        run = _run_name(idx, mix.name)
        for step, ckpt in _selected_steps(_find_checkpoints(run)):
            handle = _submit_one(run, step, ckpt)
            if handle is not None:
                handles.append(handle)
    print(f"submitted {len(handles)} eval sub-jobs; waiting for all to finish", flush=True)
    for handle in handles:
        handle.wait(raise_on_failure=True)
    print(f"merged -> {_merge()}")


if __name__ == "__main__":
    main()
