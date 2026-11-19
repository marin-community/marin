# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the two one-hour Snowball arms of the offline Mochi distillation ablation.

Both arms copy the E12 one-hour FP32 template (10/05 BF16 weights upcast to FP32, fresh optimizer, 60 updates,
constant 2e-5, batch 32, accumulation 4, data 8 x context 8, 64 H100 on cw-us-east-02a, production priority, the
template's runtime profile). The general stream, its seed and the substituted row slots are identical; 480 of the
1,920 rows (25%) come from the Mochi-scored slice. Arm A trains those rows on hard labels only (alpha 0); arm B
adds 0.5 x the soft cross-entropy to Mochi's renormalised top 20. Checkpoints are not saved.

    uv run --env-file .env python -m experiments.post_training.teacher_distill.launch_arms --template <dir> --arm a
"""

import argparse
import json
import os
from dataclasses import fields, replace
from pathlib import Path

import draccus
from dotenv import dotenv_values
from fray.iris_backend import (
    convert_constraints,
    convert_entrypoint,
    convert_environment,
    convert_resources,
    resolve_coscheduling,
    wrap_multiprocess,
)
from fray.types import Entrypoint, create_environment
from iris.cli.connect import open_iris_client
from iris.jax.compile_cache import configure_jax_compilation_cache
from iris.rpc import job_pb2
from levanter.data.text.datasets import LmDataConfig
from marin.training.training import resolve_training_env
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.grug.moe.optimizer import GrugMoeMuonHConfig  # noqa: F401 -- register optimizer
from experiments.post_training.long_loop.runs.e12.mixed_data import E12DataConfig
from experiments.post_training.long_loop.sft_runtime.train import (
    GrugRunConfig,
    _apply_hero_ep_runtime_defaults,
    _run_grug_local,
)

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_ID = "e12-grown-fresh-onehour-fp32-treatment-20261118"
TEMPLATE_PREFIX = "tmp/ttl=30d/long-loop-436/e12/grown-fresh-onehour-fp32-treatment-20261118"
ROWS = "s3://marin-us-east-02a/tmp/ttl=14d/teacher-distill-504/rows"
ARMS = {"a": ("hard", 0.0), "b": ("soft", 0.5)}
TOP = 20
SUBSTITUTED_ROWS = 480
BLOCK = 1920


def runtime_env(profile: dict) -> dict:
    """The template's worker environment: profile knobs, node-local caches, no shard autotuning."""
    env = {k: v for k, v in os.environ.items() if k.startswith(("XLA_", "JAX_", "MALLOC_")) or k == "LD_PRELOAD"}
    env.pop("JAX_PLATFORMS", None)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    for name, value in profile["worker_environment"].items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    flags = [f for f in env.get("XLA_FLAGS", "").split() if f.partition("=")[0] not in profile["xla_flags"]]
    flags.extend(name + "=" + value for name, value in profile["xla_flags"].items() if value is not None)
    env["XLA_FLAGS"] = " ".join(flags)
    env["MARIN_PREFIX"] = "s3://marin-us-east-02a/marin"
    env["MARIN_TEMP_PREFIX"] = "s3://marin-us-east-02a"
    return env


def worker(config, output, settings):
    original = {f.name: getattr(config.data, f.name) for f in fields(LmDataConfig) if f.init}
    config = replace(config, data=E12DataConfig(**original, **settings))
    configure_jax_compilation_cache()
    if int(os.environ["IRIS_MULTIGPU_PROCESS_INDEX"]) == 0:
        (StoragePath(output) / "consumed-config.json").write_text(json.dumps(draccus.encode(config), indent=2))
    _run_grug_local(config)


def build(template: Path, arm: str, date: str):
    label, alpha = ARMS[arm]
    run_id = f"task504-distill-{arm}-{label}-{date}"
    text = (template / "config.json").read_text()
    text = text.replace(TEMPLATE_PREFIX, f"tmp/ttl=14d/teacher-distill-504/runs/{run_id}").replace(TEMPLATE_ID, run_id)
    raw = json.loads(text)
    trainer = raw["trainer"]
    trainer["save_checkpoints"] = False
    trainer["distill_alpha"] = alpha
    trainer["trainer"]["tracker"]["group"] = "task504-mochi-distill"
    trainer["trainer"]["tracker"]["tags"] = ["task504", "mochi-distill", f"alpha-{alpha}"]
    config = draccus.decode(GrugRunConfig, raw)
    settings = json.loads((template / "data-settings.json").read_text())
    # A scoring time cap drops whole cells, so substitute exactly the rows that were assembled.
    rows = json.loads(StoragePath(f"{ROWS}/train-summary.json").read_text())["rows"]
    if not 0 < rows <= SUBSTITUTED_ROWS:
        raise ValueError(f"unexpected teacher-scored train rows: {rows}")
    settings.update(
        repair_cache=f"{ROWS}/train",
        repair_row_weight=rows / BLOCK,
        outer_block_size=BLOCK,
        repair_once=True,
        control_only=False,
        teacher_topk=TOP,
        teacher_validation_cache=f"{ROWS}/heldout",
    )
    profile = json.loads((template / "runtime-profile.json").read_text())
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=True, processes_per_task=8, moe_implementation="sonic", remat_mode=config.model.remat_mode
    )
    env = resolve_training_env(base_env=runtime_env(profile), resources=config.resources)
    output = str(raw["native_timing_root"]).removesuffix("/timing")
    resource = convert_resources(config.resources)
    entry = wrap_multiprocess(
        convert_entrypoint(Entrypoint.from_callable(worker, args=[config, output, settings])), resource, 8
    )
    kwargs = dict(
        name=run_id,
        entrypoint=entry,
        resources=resource,
        environment=convert_environment(
            create_environment(workspace=str(ROOT), env_vars=env, extras=["gpu"]), config.resources.device
        ),
        constraints=convert_constraints(config.resources),
        coscheduling=resolve_coscheduling(config.resources, 8),
        replicas=8,
        # Both arms may not fit at once; the second can queue behind the first.
        scheduling_timeout=Duration.from_seconds(10800),
        timeout=Duration.from_seconds(14400),
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
        priority_band=job_pb2.PRIORITY_BAND_PRODUCTION,
        existing_job_policy=job_pb2.EXISTING_JOB_POLICY_ERROR,
        termination_grace=Duration.from_seconds(60),
    )
    return kwargs, {"run_id": run_id, "alpha": alpha, "output": output, "rows": rows, "settings": settings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True, help="the one-hour FP32 treatment packet directory")
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--date", default="20261119")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    configure_coreweave_s3()
    kwargs, plan = build(args.template, args.arm, args.date)
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return
    for key, value in dotenv_values(ROOT.parent / "marin" / ".env").items():
        if key in ("CW_KEY_ID", "CW_KEY_SECRET", "WANDB_API_KEY") and value:
            kwargs["environment"].secrets[key] = value
    if not all(k in kwargs["environment"].secrets for k in ("CW_KEY_ID", "CW_KEY_SECRET", "WANDB_API_KEY")):
        raise ValueError("missing CoreWeave or W&B credentials")
    with open_iris_client(config_file=ROOT / "lib/iris/config/marin.yaml", workspace=ROOT) as client:
        job = client.submit(**kwargs)
    print(f"submitted {job.job_id}")


if __name__ == "__main__":
    main()
