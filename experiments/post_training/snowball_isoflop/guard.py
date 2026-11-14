# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Enforce aggregate experiment budgets independently of training managers."""

import argparse
import json
import logging
import time
from dataclasses import dataclass
from enum import StrEnum

import draccus
from iris.cli.connect import open_iris_client
from iris.client.client import IrisClient, iris_ctx
from iris.cluster.types import JobName
from iris.resources.state import TERMINAL_TASK_STATES, TaskState
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.snowball_isoflop.budget import allocation_ledger


class Phase(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    Q5 = "Q5"


CEILINGS = {Phase.A: 2200, Phase.B: 144, Phase.C: 384, Phase.D: 384, Phase.Q5: 1152}
POLL_SECONDS = 30
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhaseBudget:
    phase: Phase
    roots: tuple[str, ...]
    reserved_h100_hours: float
    ceiling_h100_hours: float


@dataclass(frozen=True)
class GuardConfig:
    budgets: tuple[PhaseBudget, ...]
    closed: bool


def budget_snapshot(client: IrisClient, config: GuardConfig) -> dict:
    seen = set()
    seen_phases = set()
    phases = []
    for budget in config.budgets:
        if budget.phase in seen_phases:
            raise ValueError("Each phase requires one aggregate budget")
        seen_phases.add(budget.phase)
        if not 0 <= budget.reserved_h100_hours < budget.ceiling_h100_hours <= CEILINGS[budget.phase]:
            raise ValueError("Guard configuration exceeds an approved phase ceiling")
        ledgers = []
        for name in budget.roots:
            root = JobName.from_wire(name)
            if root.parent is not None or not name.startswith(f"/benfeuer/snowball-isoflop-{budget.phase}"):
                raise ValueError("Budget guard root is outside its approved phase")
            if name in seen:
                raise ValueError("Budget guard cannot count a root twice")
            seen.add(name)
            ledgers.append(allocation_ledger(client, root))
        active_gpus = 8 * sum(
            TaskState(task["state"]) not in TERMINAL_TASK_STATES for ledger in ledgers for task in ledger["tasks"]
        )
        observed = sum(ledger["h100_hours"] for ledger in ledgers)
        charged = observed + budget.reserved_h100_hours
        phases.append(
            {
                "phase": budget.phase,
                "ledgers": ledgers,
                "observed_h100_hours": observed,
                "reserved_h100_hours": budget.reserved_h100_hours,
                "charged_h100_hours": charged,
                "ceiling_h100_hours": budget.ceiling_h100_hours,
                "active_gpu_reservation": active_gpus,
                "limit_reached": charged + active_gpus / 60 >= budget.ceiling_h100_hours,
            }
        )
    return {"phases": phases, "closed": config.closed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--status-root", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config_path = StoragePath(args.config)
    manager = StoragePath(args.status_root)
    config = draccus.decode(GuardConfig, json.loads(config_path.read_text()))
    if not args.run:
        with open_iris_client(cluster_name="marin", workspace=None) as client:
            print(json.dumps(budget_snapshot(client, config), indent=2))
        return
    context = iris_ctx()
    if context.client is None:
        raise RuntimeError("The live budget guard must run inside its Iris CPU job")
    cancelled_roots = set()
    while True:
        config = draccus.decode(GuardConfig, json.loads(config_path.read_text()))
        report = budget_snapshot(context.client, config)
        (manager / "latest.json").write_text(json.dumps(report, indent=2))
        violations = [phase for phase in report["phases"] if phase["limit_reached"]]
        for phase in violations:
            for ledger in phase["ledgers"]:
                if ledger["root"] not in cancelled_roots and any(
                    TaskState(task["state"]) not in TERMINAL_TASK_STATES for task in ledger["tasks"]
                ):
                    context.client.job(JobName.from_wire(ledger["root"])).cancel()
                    cancelled_roots.add(ledger["root"])
                    logger.error("Stopped registered root %s at phase %s limit", ledger["root"], phase["phase"])
        if violations:
            (manager / "limit-reached.json").write_text(json.dumps(report, indent=2))
        if config.closed and not any(phase["active_gpu_reservation"] for phase in report["phases"]):
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
