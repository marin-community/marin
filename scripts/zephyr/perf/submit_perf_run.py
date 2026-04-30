# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit one leg (control or treatment) of a zephyr perf gate ferry on Iris.

Caller is responsible for setting up the working directory at the right SHA
(typically a `git worktree add ../zephyr-perf-{control,treatment} <sha>`); this
script does **not** check out code, it only fires `iris job run` from
``--cwd``.

Outputs a single JSON line on stdout::

    {"job_id": "...", "label": "control", "gate": "1",
     "ferry_module": "experiments.ferries.datakit_ferry",
     "status_path": "gs://...", "run_id": "...", "cwd": "..."}

The status JSON path is the standard ``FERRY_STATUS_PATH`` contract (see
``experiments/ferries/datakit_ferry.py``); the ferry writes ``status`` and
``marin_prefix`` there on completion.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import logging
import shlex
import subprocess
import sys

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class GateConfig:
    ferry_module: str
    region: str | None
    memory: str
    disk: str
    cpu: str
    extra: str
    priority: str

    @property
    def iris_args(self) -> list[str]:
        args = [
            f"--memory={self.memory}",
            f"--disk={self.disk}",
            f"--cpu={self.cpu}",
            f"--extra={self.extra}",
            f"--priority={self.priority}",
        ]
        if self.region:
            args.append(f"--region={self.region}")
        return args


# Mirrors the canonical workflow YAMLs:
#   .github/workflows/marin-datakit-smoke.yaml          -> Gate 1
#   .github/workflows/marin-datakit-nemotron-ferry.yaml -> Gate 2
GATES: dict[str, GateConfig] = {
    "1": GateConfig(
        ferry_module="experiments.ferries.datakit_ferry",
        region=None,
        memory="2G",
        disk="4G",
        cpu="1",
        extra="cpu",
        priority="production",
    ),
    "2": GateConfig(
        ferry_module="experiments.ferries.datakit_nemotron_ferry",
        region="europe-west4",
        memory="3G",
        disk="5G",
        cpu="1",
        extra="cpu",
        priority="production",
    ),
}


def _build_run_id(pr: int, gate: str, label: str) -> str:
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"zephyr-perf-pr{pr}-g{gate}-{label}-{ts}"


def _build_status_path(pr: int, run_id: str, override: str | None) -> str:
    if override:
        return override
    return "gs://marin-us-central1/tmp/ttl=7d/zephyr-perf/" f"pr{pr}/{run_id}/ferry_run_status.json"


def submit(
    *,
    gate: str,
    label: str,
    pr: int,
    cwd: str,
    iris_config: str,
    status_path_override: str | None,
    wandb_entity: str,
    wandb_project: str,
    dry_run: bool,
) -> dict[str, object]:
    if gate not in GATES:
        raise ValueError(f"Unknown gate {gate!r}; expected one of {sorted(GATES)}")
    if label not in {"control", "treatment"}:
        raise ValueError(f"Unknown label {label!r}; expected control|treatment")

    cfg = GATES[gate]
    run_id = _build_run_id(pr, gate, label)
    status_path = _build_status_path(pr, run_id, status_path_override)

    cmd = [
        ".venv/bin/iris",
        f"--config={iris_config}",
        "job",
        "run",
        "--no-wait",
        *cfg.iris_args,
        "-e",
        "SMOKE_RUN_ID",
        run_id,
        "-e",
        "FERRY_STATUS_PATH",
        status_path,
        "-e",
        "WANDB_ENTITY",
        wandb_entity,
        "-e",
        "WANDB_PROJECT",
        wandb_project,
        "-e",
        "WANDB_API_KEY",
        "$WANDB_API_KEY",
        "-e",
        "HF_TOKEN",
        "$HF_TOKEN",
        "--",
        "python",
        "-m",
        cfg.ferry_module,
    ]

    if dry_run:
        logger.info("dry-run cmd (cwd=%s): %s", cwd, " ".join(shlex.quote(c) for c in cmd))
        job_id = "DRY-RUN"
    else:
        out = subprocess.check_output(cmd, cwd=cwd, text=True)
        job_id = out.strip().splitlines()[-1].strip()

    return {
        "job_id": job_id,
        "label": label,
        "gate": gate,
        "ferry_module": cfg.ferry_module,
        "status_path": status_path,
        "run_id": run_id,
        "cwd": cwd,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True, choices=sorted(GATES))
    parser.add_argument("--label", required=True, choices=("control", "treatment"))
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--cwd", required=True, help="Path to a worktree at the target SHA.")
    parser.add_argument(
        "--iris-config",
        default="lib/iris/examples/marin.yaml",
        help="Iris cluster config; resolved relative to --cwd.",
    )
    parser.add_argument(
        "--status-out",
        default=None,
        help="Override FERRY_STATUS_PATH. Default lives under gs://marin-us-central1/tmp/ttl=7d/.",
    )
    parser.add_argument("--wandb-entity", default="marin-community")
    parser.add_argument("--wandb-project", default="marin")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = submit(
        gate=args.gate,
        label=args.label,
        pr=args.pr,
        cwd=args.cwd,
        iris_config=args.iris_config,
        status_path_override=args.status_out,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        dry_run=args.dry_run,
    )
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
