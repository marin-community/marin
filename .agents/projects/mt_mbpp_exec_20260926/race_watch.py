"""Cancel both racing MT-MBPP runs (europe-west4 and the second us-east5 plan) once every checkpoint-language pair of
Proportional, UniMax-8 and MARINER has completed generation in at least one plan (Calvin, 2026-09-26: race both,
cancel the slower duplicates). Checks every 5 minutes; logs coverage to race_watch.log beside this file."""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from marin.evaluation.olmo_base_eval.components import MT_MBPP_SUBTASKS

from experiments.domain_phase_mix import evaluate_table9_accuracy as inference

HERE = Path(__file__).resolve().parent
PLANS = [HERE / "plan_euw4.json", HERE / "plan_east5b.json", HERE / "plan_east5.json"]
RACING = ["/calvinxu/mt-mbpp-accuracy-v6e4-full-euw4-20260926", "/calvinxu/mt-mbpp-accuracy-v6e4-full-east5b-20260926"]
LOG = HERE / "race_watch.log"
IRIS = ["uv", "run", "--no-sync", "iris", "--config", "lib/iris/config/marin.yaml"]


def log(message: str) -> None:
    with LOG.open("a") as handle:
        handle.write(f"{datetime.now().strftime('%H:%M:%S')} {message}\n")


def state(job: str) -> str:
    out = subprocess.run([*IRIS, "job", "describe", job], capture_output=True, text=True).stdout
    return next((line.split()[1] for line in out.splitlines() if line.startswith("State:")), "unknown")


def main() -> None:
    plans = [json.loads(p.read_text()) for p in PLANS]
    names = [r["name"] for r in plans[0]["rows"]]
    log(f"watching {len(names)} checkpoints x {len(MT_MBPP_SUBTASKS)} tasks")
    while True:
        covered, by_plan = set(), {p.stem: 0 for p in PLANS}
        for path, plan in zip(PLANS, plans, strict=True):
            for row in plan["rows"]:
                for task in MT_MBPP_SUBTASKS:
                    if inference.completed_task(plan, row, task, 0) is not None:
                        covered.add((row["name"], task))
                        by_plan[path.stem] += 1
        total = len(names) * len(MT_MBPP_SUBTASKS)
        log(f"covered {len(covered)}/{total} {by_plan}")
        if len(covered) == total:
            for job in RACING:
                s = state(job)
                if s in ("running", "pending"):
                    result = subprocess.run([*IRIS, "job", "cancel", "--exact", job], capture_output=True, text=True)
                    log(f"cancelled {job} (was {s}): {result.stdout.strip()[:200]}")
                else:
                    log(f"{job} already {s}")
            log("DONE")
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
