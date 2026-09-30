"""Partition a fresh catalog run across independently checkpointed Iris workers."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from .catalog import canonical_sha256, load_pilot, validate_pilot
from .runtime import sha256

ROOT = Path(__file__).resolve().parents[1]
LEDGER_ROOT = Path.home() / ".cache/capability-pipeline/fleet-launches"
SCHEMA = "capability-generation-fleet-v1"


def _write(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _controller() -> dict[str, str]:
    paths = [*(p for p in sorted((ROOT / "capability_pipeline").rglob("*"))
               if "__pycache__" not in p.parts and p.suffix in {".py", ".json"}),
             *sorted((ROOT / "scripts").glob("*.*")),
             *sorted((ROOT / "vendor/task_spec").rglob("*")),
             *sorted((ROOT / "docs/build_acceptance").glob("*.md")),
             ROOT / "docs/build_acceptance_001.md", ROOT / "docs/task_contract.md",
             ROOT / "data/revocations.json"]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths if p.is_file()}


STAGES = ("generate", "propose")
TIERS = ("interactive", "bulk")


def create_plan(pilot: Path, output: Path, run_prefix: str, *, workers: int = 4,
                concurrency: int = 256, cpu: int = 16, memory: str = "128g",
                stage: str = "generate", tier: str = "interactive") -> dict:
    """Preserve every capability and its ten slots, balancing whole capabilities."""
    source = load_pilot(pilot)
    rows = sorted(source["capabilities"], key=lambda row: row["capability_id"])
    if not 1 <= workers <= min(len(rows), concurrency) or cpu < 1:
        raise ValueError("workers, CPU and concurrency must be positive; shards cannot be empty")
    if not re.fullmatch(r"[1-9][0-9]*[gm]", memory):
        raise ValueError("memory must be an explicit positive g/m quantity")
    if stage not in STAGES or tier not in TIERS:
        raise ValueError(f"stage must be one of {STAGES} and tier one of {TIERS}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", run_prefix) or any(
        part in {"", ".", ".."} for part in run_prefix.split("/")
    ):
        raise ValueError("run prefix must be a safe relative path")
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "input-pilot.json", source)
    identity = {"source_sha256": sha256(output / "input-pilot.json"), "controller": _controller(),
                "run_prefix": run_prefix, "workers": workers, "concurrency": concurrency,
                "cpu": cpu, "memory": memory, "stage": stage, "tier": tier}
    fleet_id = canonical_sha256(identity)[:20]
    shards = []
    for index in range(workers):
        part = copy.deepcopy(source)
        part["capabilities"] = rows[index::workers]
        ids = {row["capability_id"] for row in part["capabilities"]}
        if "learning_progression" in part:
            progression = part["learning_progression"]
            progression["edges"] = [edge for edge in progression["edges"] if edge["dependent_id"] in ids]
            progression["edges_sha256"] = canonical_sha256(progression["edges"])
            progression["scope_subject_ids"] = sorted({row["subject_id"] for row in part["capabilities"]})
        validate_pilot(part)
        name = f"shard-{index:03d}"
        path = output / f"{name}.json"
        _write(path, part)
        shards.append({
            "name": name, "pilot": path.name, "pilot_sha256": sha256(path),
            "capability_ids": sorted(ids), "slots": 10 * len(ids),
            "concurrency": concurrency // workers + (index < concurrency % workers),
            "output": f"{run_prefix}/generation-{fleet_id}/{name}", "cpu": cpu, "memory": memory,
        })
    plan = {"schema_version": SCHEMA, "identity": identity, "fleet_id": fleet_id,
            "source_sha256": identity["source_sha256"],
            "controller": identity["controller"], "aggregate_concurrency": concurrency,
            "capabilities": len(rows), "slots": len(rows) * 10, "shards": shards,
            "scope": "fresh independent capability runs; not a migration or repair-budget reset"}
    _write(output / "plan.json", plan)
    return plan


def validate_plan(root: Path) -> dict:
    plan = json.loads((root / "plan.json").read_text())
    if plan.get("schema_version") != SCHEMA or plan.get("controller") != _controller():
        raise ValueError("fleet controller changed; prepare an explicit new plan")
    identity = plan["identity"]
    if canonical_sha256(identity)[:20] != plan["fleet_id"] or identity["source_sha256"] != plan["source_sha256"] or identity["controller"] != plan["controller"]:
        raise ValueError("fleet generation identity differs")
    if len(plan["shards"]) != identity["workers"] or plan["aggregate_concurrency"] != identity["concurrency"]:
        raise ValueError("fleet allocation identity differs")
    if sha256(root / "input-pilot.json") != plan["source_sha256"]:
        raise ValueError("fleet source manifest changed")
    source = load_pilot(root / "input-pilot.json")
    expected = {row["capability_id"]: row for row in source["capabilities"]}
    observed = {}
    for shard in plan["shards"]:
        if shard["output"] != f"{identity['run_prefix']}/generation-{plan['fleet_id']}/{shard['name']}" or shard["cpu"] != identity["cpu"] or shard["memory"] != identity["memory"]:
            raise ValueError("fleet worker differs from generation identity")
        if shard["pilot"] != shard["name"] + ".json" or not re.fullmatch(r"shard-[0-9]{3,}", shard["name"]):
            raise ValueError("unsafe fleet shard path")
        path = root / shard["pilot"]
        if sha256(path) != shard["pilot_sha256"]:
            raise ValueError("fleet shard changed")
        part = load_pilot(path)
        ids = [row["capability_id"] for row in part["capabilities"]]
        expected_part = copy.deepcopy(source)
        expected_part["capabilities"] = [expected[key] for key in ids]
        if "learning_progression" in expected_part:
            progression = expected_part["learning_progression"]
            progression["edges"] = [edge for edge in progression["edges"] if edge["dependent_id"] in ids]
            progression["edges_sha256"] = canonical_sha256(progression["edges"])
            progression["scope_subject_ids"] = sorted({expected[key]["subject_id"] for key in ids})
        if part != expected_part:
            raise ValueError("fleet changed source context or learning progression")
        if type(shard["concurrency"]) is not int or shard["concurrency"] < 1 or type(shard["cpu"]) is not int or shard["cpu"] < 1:
            raise ValueError("fleet worker resources must be positive")
        if not re.fullmatch(r"[1-9][0-9]*[gm]", shard["memory"]) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", shard["output"]) or any(part in {"", ".", ".."} for part in shard["output"].split("/")):
            raise ValueError("fleet memory or output path is invalid")
        if sorted(ids) != shard["capability_ids"] or shard["slots"] != len(ids) * 10:
            raise ValueError("fleet shard accounting differs")
        for row in part["capabilities"]:
            key = row["capability_id"]
            if key in observed or row != expected.get(key):
                raise ValueError("fleet duplicates or alters a capability")
            observed[key] = row
    if observed != expected or plan["capabilities"] != len(expected) or plan["slots"] != 10 * len(expected):
        raise ValueError("fleet does not cover the entire source manifest")
    if sum(row["concurrency"] for row in plan["shards"]) != plan["aggregate_concurrency"]:
        raise ValueError("fleet concurrency accounting differs")
    if len({row["output"] for row in plan["shards"]}) != len(plan["shards"]):
        raise ValueError("fleet workers must have independent output prefixes")
    return plan


def _require_unchanged(root: Path, plan: dict, plan_hash: str) -> None:
    """Cheap drift check against the identities validate_plan already proved.

    validate_plan's semantic deep-compare reloads the whole source manifest and
    deep-copies it once per shard.  Running it again inside every submission
    thread is O(shards^2) GIL-bound work: at 100 shards over the 1,999-capability
    catalog that is ~10,000 deep copies of a 9 MB manifest before the first job
    is submitted.  The file identities it binds are recorded in the plan, so
    re-hashing every input detects the same drift in linear time.
    """
    if sha256(root / "plan.json") != plan_hash:
        raise ValueError("fleet plan changed before submission")
    if sha256(root / "input-pilot.json") != plan["source_sha256"]:
        raise ValueError("fleet source manifest changed")
    for shard in plan["shards"]:
        if sha256(root / shard["pilot"]) != shard["pilot_sha256"]:
            raise ValueError("fleet shard changed")
    if _controller() != plan["controller"]:
        raise ValueError("fleet controller changed; prepare an explicit new plan")


def submit_plan(root: Path, *, launch=None, parallel: int = 16) -> dict:
    """Submit all shards together; ambiguous prior launches are never repeated.

    ``parallel`` bounds LOCAL submitter processes only.  Each ``submit.sh`` runs
    uv, gcloud and the Iris CLI, and a hundred of them at once on a laptop that
    is already swapping get OOM-killed into ``submission_failed`` receipts that
    are never retried.  It does not pace the cluster: Iris federation delivers
    jobs at its own rate, and every shard is still submitted in this call.
    """
    if parallel < 1:
        raise ValueError("submission parallelism must be positive")
    root = root.resolve()
    plan = validate_plan(root)
    plan_hash = sha256(root / "plan.json")
    launch = launch or subprocess.run
    ledger = LEDGER_ROOT / plan["fleet_id"]
    ledger.mkdir(parents=True, exist_ok=True)
    # Plans written before stage/tier existed ran generate at interactive tier.
    identity_stage = plan["identity"].get("stage", "generate")
    identity_tier = plan["identity"].get("tier", "interactive")

    def one(shard):
        _require_unchanged(root, plan, plan_hash)
        name = shard["name"]
        receipt = ledger / f"{name}.submission.json"
        started = ledger / f"{name}.started.json"
        local_receipt = root / receipt.name
        identity = {"plan_sha256": plan_hash, "shard": name}
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved.get("identity") != identity:
                raise ValueError("fleet submission receipt identity changed")
            if not local_receipt.exists():
                _write(local_receipt, saved)
            return saved
        if started.exists() or (root / started.name).exists():
            return {"identity": identity, "state": "pending_submission_observation"}
        try:
            _write(started, identity)
        except FileExistsError:
            return {"identity": identity, "state": "pending_submission_observation"}
        _write(root / started.name, identity)
        env = dict(os.environ)
        marin = Path(env.get("MARIN", str(Path.home() / "openathena/marin")))
        staging = marin / "capability-pipeline-staging"
        env.update(CPU=str(shard["cpu"]), MEM=shard["memory"], CAPABILITY_SOURCE_ROOT=str(ROOT),
                   CAPABILITY_REQUIRE_EMPTY_DESTINATION="1",
                   STAGING_ROOT=str(staging),
                   CAPABILITY_SUBMISSION_LOCK=str(staging / f".fleet-{plan_hash[:20]}-{name}.lock"))
        # stage and tier are part of the frozen identity, so a plan cannot be
        # resubmitted under a different stage or tier than it was validated with.
        command = [str(ROOT / "scripts/submit.sh"), "--stage", identity_stage, "--pilot",
                   str(root / shard["pilot"]), "--out", shard["output"], "--concurrency",
                   str(shard["concurrency"]), "--tier", identity_tier,
                   "--run-name", f"cap-fleet-{plan_hash[:12]}-{name}"]
        result = launch(command, env=env, cwd=ROOT, capture_output=True, check=False)
        saved = {"identity": identity, "state": "submitted" if result.returncode == 0 else "submission_failed",
                 "exit_code": result.returncode, "output": shard["output"],
                 "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
                 "stderr_sha256": hashlib.sha256(result.stderr).hexdigest()}
        try:
            _require_unchanged(root, plan, plan_hash)
        except (ValueError, OSError, KeyError):
            saved["state"] = "pending_source_drift"
        _write(receipt, saved)
        _write(local_receipt, saved)
        return saved

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(plan["shards"]), parallel)) as pool:
        receipts = list(pool.map(one, plan["shards"]))
    return {"schema_version": SCHEMA, "plan_sha256": plan_hash,
            "submitted": sum(row["state"] == "submitted" for row in receipts),
            "workers": len(receipts), "receipts": receipts,
            "scope": "submission only; task outcomes require each shard's coverage report"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("plan")
    make.add_argument("--pilot", type=Path, required=True)
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--run-prefix", required=True)
    make.add_argument("--workers", type=int, default=4)
    make.add_argument("--concurrency", type=int, default=256)
    make.add_argument("--cpu", type=int, default=16)
    make.add_argument("--memory", default="128g")
    make.add_argument("--stage", choices=STAGES, default="generate")
    make.add_argument("--tier", choices=TIERS, default="interactive")
    submit = commands.add_parser("submit")
    submit.add_argument("--plan-dir", type=Path, required=True)
    submit.add_argument("--parallel", type=int, default=16,
                        help="concurrent local submit.sh processes (does not pace the cluster)")
    args = parser.parse_args(argv)
    if args.command == "plan":
        value = create_plan(args.pilot, args.out, args.run_prefix, workers=args.workers,
                            concurrency=args.concurrency, cpu=args.cpu, memory=args.memory,
                            stage=args.stage, tier=args.tier)
        print(json.dumps({"workers": len(value["shards"]), "slots": value["slots"]}))
        return 0
    report = submit_plan(args.plan_dir, parallel=args.parallel)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["submitted"] == report["workers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
