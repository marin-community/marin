# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the acceptance measurement: every sampled proposal authored once in each arm, one allocation per pair.

``run`` drives one shard of one wave. Its allocations are the wave's capabilities with ``rank % shards == shard``,
each in every arm, so the arms of a capability share a host and a time window. Each allocation runs in its own child
process (``allocation``), at most ``--concurrency`` at a time with sliding-window dispatch; the child runs the stack's
``StageCoordinator`` on a local root and writes ``records.jsonl``. The parent then uploads the allocation's tree and
records to ``<output>/allocations/<arm>/<capability>/`` and writes ``SEALED.json`` LAST, for an accepted, rejected or
abandoned terminal. A held allocation stays unsealed with its tree uploaded; the next ``run`` restores the tree and
the coordinator resumes it. A sealed allocation is never run again. Each try is ``attempts/<n>.json``; after
``MAX_ATTEMPTS`` tries an unsealed allocation is reported and left for an explicit decision.

``manifest.json`` (written once per output, and checked by every later shard) fixes the arms' file shas and policy
digests, the git sha, the sample's sha, the models and the shared settings; a shard whose manifest differs refuses
to run, so a measurement never mixes two configurations.

``dry-run`` renders the arms on proposals with no model and no machine (``dryrun.py``) and refuses arms that differ
outside their declared surfaces.

    uv run --package marin-taskforge --frozen python -m \\
        experiments.post_training.capability_driven_envs.acceptance_measure.driver run --wave 1 --shard 0 --shards 11 ...
"""

import asyncio
import json
import logging
import os
import shutil
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import click
from iris.cluster.client import get_job_info
from rigging.log_setup import configure_logging
from shellbox.machine import Backend
from taskforge.llm.glm_client import RelayEndpoint
from taskforge.sandbox.factories import MachineHost
from taskforge.sandbox.image_builder import REGISTRY_CREDENTIAL_ENV
from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage40_build_with_program.run import ResearchRequirement
from taskforge.stage_coordinator.run import run_workpieces
from taskforge.stage_coordinator.workpiece_machine import Terminal

from experiments.post_training.capability_driven_envs.acceptance_measure.dryrun import dry_run
from experiments.post_training.capability_driven_envs.acceptance_measure.pipeline import (
    LEDGER_DIR,
    POOL,
    Host,
    clients,
    coordinator_for,
    live_services,
)
from experiments.post_training.capability_driven_envs.acceptance_measure.proposals import read_proposal
from experiments.post_training.capability_driven_envs.acceptance_measure.records import (
    RECORDS_FILE,
    Allocation,
    write_records,
)
from experiments.post_training.capability_driven_envs.acceptance_measure.runtime import (
    INFRASTRUCTURE_RETRY,
    arm_runtime,
    shared_settings,
)
from experiments.post_training.capability_driven_envs.acceptance_measure.sample import (
    _filesystem,
    sha256_hex,
    wave_capabilities,
)
from experiments.post_training.capability_driven_envs.acceptance_measure.treatment import (
    ARM_NAMES,
    ARMS_DIR,
    load_arms,
    load_treatment,
)

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
SEAL_FILE = "SEALED.json"
SPEC_FILE = "allocation.json"
TERMINAL_FILE = "terminal.json"
TREE_DIR = "tree"
ATTEMPTS_DIR = "attempts"
ALLOCATIONS_DIR = "allocations"
PROPOSAL_FILE = "proposal.json"
MAX_ATTEMPTS = 3
SEALED = frozenset({Terminal.ACCEPTED, Terminal.REJECTED, Terminal.ABANDONED})
PARALLEL_KEY_ENV = "TASKFORGE_PARALLEL_KEY"
CW_KEY_ENVS = ("CW_KEY_ID", "CW_KEY_SECRET")
MODULE = "experiments.post_training.capability_driven_envs.acceptance_measure.driver"
IDENTITY_FIELDS = ("format", "git_sha", "sample_sha256", "arms", "models", "shared", "pool")


@dataclass(frozen=True)
class AllocationSpec:
    """What a child needs to run one allocation; written as ``allocation.json``. Secrets stay in the environment."""

    allocation: Allocation
    proposal_file: str
    root: str
    host: Host


# --- storage -----------------------------------------------------------------------------------------------------


class Store:
    """The output root, local or on S3, addressed by paths relative to it."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.fs, self.root = _filesystem(self.url)

    def path(self, relative: str) -> str:
        return f"{self.root}/{relative}"

    def exists(self, relative: str) -> bool:
        return self.fs.exists(self.path(relative))

    def read(self, relative: str) -> bytes:
        return self.fs.cat_file(self.path(relative))

    def write(self, relative: str, data: bytes) -> None:
        target = self.path(relative)
        self.fs.makedirs(target.rsplit("/", 1)[0], exist_ok=True)
        self.fs.pipe_file(target, data)

    def upload_tree(self, local: Path, relative: str) -> int:
        """Upload every file under ``local`` to ``relative/<path>``, one key per file; returns the file count."""
        files = sorted(p for p in local.rglob("*") if p.is_file())
        for path in files:
            target = self.path(f"{relative}/{path.relative_to(local)}")
            self.fs.makedirs(target.rsplit("/", 1)[0], exist_ok=True)
            self.fs.put_file(str(path), target)
        return len(files)

    def download_tree(self, relative: str, local: Path) -> int:
        prefix = self.path(relative)
        if not self.fs.exists(prefix):
            return 0
        found = [p for p in self.fs.find(prefix) if not p.endswith("/")]
        for remote in found:
            target = local / remote[len(prefix) + 1 :]
            target.parent.mkdir(parents=True, exist_ok=True)
            self.fs.get_file(remote, str(target))
        return len(found)

    def count(self, relative: str) -> int:
        prefix = self.path(relative)
        return len(self.fs.find(prefix)) if self.fs.exists(prefix) else 0


def read_source(url: str) -> bytes:
    fs, path = _filesystem(url)
    return fs.cat_file(path)


# --- manifest ----------------------------------------------------------------------------------------------------


def git_sha_of(value: str) -> str:
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise click.BadParameter(f"--git-sha takes a full 40-character sha, got {value!r}")
    return value


def manifest(sample_bytes: bytes, git_sha: str) -> dict:
    """The run's identity. It always covers every arm, so jobs that each run one arm share one manifest."""
    treatments = {name: load_treatment(ARMS_DIR / f"{name}.json") for name in ARM_NAMES}
    return {
        "format": "acceptance-measure-manifest-v1",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "sample_sha256": sha256_hex(sample_bytes),
        "arms": {name: t.manifest() for name, t in treatments.items()},
        "models": {name: arm_runtime(t).models for name, t in treatments.items()},
        "shared": shared_settings(),
        "pool": str(POOL),
    }


def manifest_digest(document: Mapping) -> str:
    return sha256_hex(json.dumps({k: document[k] for k in IDENTITY_FIELDS}, sort_keys=True))


def ensure_manifest(store: Store, document: dict) -> str:
    """Write the manifest when the output has none, else refuse one whose identity differs; returns its digest."""
    digest = manifest_digest(document)
    if store.exists(MANIFEST_FILE):
        existing = json.loads(store.read(MANIFEST_FILE))
        if manifest_digest(existing) != digest:
            differs = [k for k in IDENTITY_FIELDS if existing.get(k) != document.get(k)]
            raise click.ClickException(f"{store.url}/{MANIFEST_FILE} was written for another configuration: {differs}")
        return digest
    store.write(MANIFEST_FILE, (json.dumps(document | {"digest": digest}, indent=2) + "\n").encode())
    return digest


# --- allocations -------------------------------------------------------------------------------------------------


def shard_allocations(sample: Mapping, wave: str, shard: int, shards: int, arms: Sequence[str]) -> list[Allocation]:
    """The allocations of ``shard``: the wave's capabilities with ``rank % shards == shard``, in every arm."""
    entries = sample["sample"] if wave == "all" else wave_capabilities(sample, int(wave))
    return [
        Allocation(arm, e["capability"], e["domain"], e["rank"], e["slot"], e["proposal"], e["sha256"])
        for e in entries
        if e["rank"] % shards == shard
        for arm in arms
    ]


def allocation_prefix(allocation: Allocation) -> str:
    return f"{ALLOCATIONS_DIR}/{allocation.key}"


def attempts_of(store: Store, allocation: Allocation) -> int:
    return store.count(f"{allocation_prefix(allocation)}/{ATTEMPTS_DIR}")


async def run_one(store: Store, allocation: Allocation, host: Host, work_dir: Path, manifest_sha: str) -> str:
    """Run, upload and (on a final terminal) seal one allocation; returns what happened, for the shard's summary."""
    prefix = allocation_prefix(allocation)
    if store.exists(f"{prefix}/{SEAL_FILE}"):
        return "sealed_before"
    attempt = attempts_of(store, allocation)
    if attempt >= MAX_ATTEMPTS:
        logger.warning("%s: %d attempts without a seal; left for an explicit decision", allocation.key, attempt)
        return "attempts_exhausted"
    local = work_dir / allocation.arm / allocation.capability
    shutil.rmtree(local, ignore_errors=True)
    root = local / TREE_DIR
    root.mkdir(parents=True)
    restored = store.download_tree(f"{prefix}/{TREE_DIR}", root)
    raw = read_source(allocation.proposal)
    read_proposal(raw, allocation.sha256)
    (local / PROPOSAL_FILE).write_bytes(raw)
    spec = AllocationSpec(allocation, str(local / PROPOSAL_FILE), str(root), host)
    (local / SPEC_FILE).write_text(json.dumps(asdict(spec), default=str, indent=2))
    start = {
        "attempt": attempt,
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "restored_files": restored,
        "manifest": manifest_sha,
    }
    store.write(f"{prefix}/{ATTEMPTS_DIR}/{attempt}.json", json.dumps(start).encode())
    log = local / "allocation.log"
    with log.open("wb") as handle:
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-m",
            MODULE,
            "allocation",
            "--spec",
            str(local / SPEC_FILE),
            stdout=handle,
            stderr=handle,
        )
        code = await child.wait()
    terminal = json.loads((local / TERMINAL_FILE).read_text()) if (local / TERMINAL_FILE).exists() else None
    files = store.upload_tree(root, f"{prefix}/{TREE_DIR}")
    store.write(f"{prefix}/allocation.log", log.read_bytes()[-2_000_000:])
    ended = {
        "ended": datetime.now(UTC).isoformat(timespec="seconds"),
        "exit": code,
        "terminal": terminal,
        "files": files,
    }
    store.write(f"{prefix}/{ATTEMPTS_DIR}/{attempt}.json", json.dumps(start | ended).encode())
    if terminal is None or Terminal(terminal["terminal"]) not in SEALED:
        logger.info("%s: attempt %d ended %s, exit %d; not sealed", allocation.key, attempt, terminal, code)
        return "unsealed"
    records = (root / RECORDS_FILE).read_bytes()
    store.write(f"{prefix}/{RECORDS_FILE}", records)
    seal = terminal | {
        "records_sha256": sha256_hex(records),
        "files": files,
        "attempt": attempt,
        "manifest": manifest_sha,
    }
    store.write(f"{prefix}/{SEAL_FILE}", (json.dumps(seal, indent=2) + "\n").encode())
    logger.info("%s: sealed %s after attempt %d", allocation.key, terminal["terminal"], attempt)
    return f"sealed_{terminal['terminal']}"


async def run_shard(
    store: Store, allocations: Sequence[Allocation], host: Host, work_dir: Path, manifest_sha: str, concurrency: int
) -> dict:
    """Sliding-window dispatch: a new allocation starts the moment one ends."""
    outcomes: dict[str, str] = {}
    pending = list(allocations)
    running: set[asyncio.Task] = set()
    keys: dict[asyncio.Task, str] = {}
    while pending or running:
        while pending and len(running) < concurrency:
            allocation = pending.pop(0)
            task = asyncio.create_task(run_one(store, allocation, host, work_dir, manifest_sha))
            running.add(task)
            keys[task] = allocation.key
        done, running = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            outcomes[keys[task]] = task.result()
            logger.info(
                "allocation %s: %s (%d of %d ended)", keys[task], outcomes[keys[task]], len(outcomes), len(allocations)
            )
    return outcomes


# --- the child -----------------------------------------------------------------------------------------------------


def _secret(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise click.ClickException(f"{name} is not set")
    return value


async def run_allocation(spec: AllocationSpec) -> Terminal:
    allocation = spec.allocation
    runtime = arm_runtime(load_treatment(ARMS_DIR / f"{allocation.arm}.json"))
    proposal = read_proposal(Path(spec.proposal_file).read_bytes(), allocation.sha256)
    root = Path(spec.root)
    root.mkdir(parents=True, exist_ok=True)
    research = ResearchRequirement.ENFORCED if proposal.header.research else ResearchRequirement.OFF
    host = spec.host
    secrets = {} if host.parallel_key_env is None else {host.parallel_key_env: _secret(host.parallel_key_env)}
    async with (
        clients(host, runtime.models, _secret(host.token_env)) as models,
        live_services(host, root / LEDGER_DIR, secrets) as services,
    ):
        coordinator = coordinator_for(runtime, root, host, models, services, research)
        terminals = await run_workpieces(coordinator, _one(allocation.workpiece_id, proposal), INFRASTRUCTURE_RETRY)
    write_records(root, allocation, proposal)
    return terminals[allocation.workpiece_id]


async def _one(workpiece_id: str, proposal: TaskProposal) -> AsyncIterator[tuple[str, TaskProposal]]:
    yield workpiece_id, proposal


def load_spec(path: Path) -> AllocationSpec:
    raw = json.loads(path.read_text())
    host = raw["host"]
    return AllocationSpec(
        Allocation(**raw["allocation"]),
        raw["proposal_file"],
        raw["root"],
        Host(**(host | {"host": MachineHost(host["host"]), "backends": tuple(Backend(b) for b in host["backends"])})),
    )


# --- CLI -----------------------------------------------------------------------------------------------------------


@click.group()
def main() -> None:
    configure_logging(logging.INFO)


@main.command()
@click.option("--sample", "sample_path", required=True, help="sample.json from sample.py (local or s3://).")
@click.option("--output", required=True, help="The measurement's output root (local or s3://).")
@click.option("--wave", type=click.Choice(["1", "2", "all"]), required=True)
@click.option("--shard", type=click.IntRange(min=0), required=True)
@click.option("--shards", type=click.IntRange(min=1), required=True)
@click.option("--arms", default=",".join(ARM_NAMES), show_default=True)
@click.option("--git-sha", required=True, help="The commit the job runs, for the manifest.")
@click.option("--relay-job", help="The GLM relay job (inside Iris); exactly one of this and --base-url.")
@click.option("--base-url", help="An OpenAI-compatible base URL ending in /v1 (a laptop port-forward).")
@click.option(
    "--controller-url", envvar="IRIS_CONTROLLER_URL", help="The Iris controller, for gVisor machines and image builds."
)
@click.option("--image-cluster", default="cw-rno2a", show_default=True)
@click.option("--concurrency", type=click.IntRange(min=1), required=True, help="Allocations running at once.")
@click.option("--work-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--allow-incomplete-population", is_flag=True, help="Run a sample drawn while the n=25 run was in flight.")
def run(
    sample_path: str, output: str, wave: str, shard: int, shards: int, arms: str, git_sha: str, relay_job: str | None,
    base_url: str | None, controller_url: str | None, image_cluster: str, concurrency: int, work_dir: Path,
    allow_incomplete_population: bool,
) -> None:  # fmt: skip
    """Run one shard of one wave; print the outcome counts."""
    if shard >= shards:
        raise click.BadParameter("--shard must be below --shards")
    names = tuple(arms.split(","))
    if unknown := set(names) - set(ARM_NAMES):
        raise click.BadParameter(f"unknown arms {sorted(unknown)}; the arms are {', '.join(ARM_NAMES)}")
    sample_bytes = read_source(sample_path)
    sample = json.loads(sample_bytes)
    if not sample["population"]["complete"] and not allow_incomplete_population:
        raise click.ClickException(
            "the sample was drawn from an incomplete n=25 run; redraw it, or pass --allow-incomplete-population"
        )
    store = Store(output)
    manifest_sha = ensure_manifest(store, manifest(sample_bytes, git_sha_of(git_sha)))
    allocations = shard_allocations(sample, wave, shard, shards, names)
    if not allocations:
        raise click.ClickException(f"shard {shard} of {shards} holds no allocation of wave {wave}")
    host = Host(
        host=MachineHost.IRIS if relay_job else MachineHost.LAPTOP,
        backends=(Backend.SHELLSIM, Backend.GVISOR) if relay_job else (Backend.SHELLSIM,),
        controller_url=controller_url,
        relay_job=relay_job,
        base_url=base_url,
        parallel_key_env=PARALLEL_KEY_ENV if os.environ.get(PARALLEL_KEY_ENV) else None,
        # Inside an Iris task the controller is the task's own cluster, which refuses a pin naming itself.
        image_cluster=image_cluster if relay_job and get_job_info() is None else None,
        registry_credential_env=REGISTRY_CREDENTIAL_ENV if relay_job else None,
    )
    preflight(host, names)
    click.echo(f"manifest {manifest_sha[:16]}; shard {shard}/{shards} wave {wave}: {len(allocations)} allocations")
    outcomes = asyncio.run(run_shard(store, allocations, host, work_dir, manifest_sha, concurrency))
    counts: dict[str, int] = {}
    for outcome in outcomes.values():
        counts[outcome] = counts.get(outcome, 0) + 1
    click.echo(json.dumps({"shard": shard, "allocations": len(allocations), "outcomes": counts}))


def preflight(host: Host, arms: Sequence[str]) -> None:
    """Fail before any allocation starts: every secret the run reads is set, and the relay routes every model."""
    for name in (
        host.token_env,
        *CW_KEY_ENVS,
        *(() if host.registry_credential_env is None else (host.registry_credential_env,)),
    ):
        _secret(name)
    if host.relay_job is None:
        return
    models = sorted({m for name in arms for m in arm_runtime(load_treatment(ARMS_DIR / f"{name}.json")).models.values()})
    for model in models:
        endpoint = RelayEndpoint(host.relay_job, _secret(host.token_env), POOL, model).current()
        click.echo(f"relay {host.relay_job} routes {model} at {endpoint.base_url}")


@main.command()
@click.option("--spec", "spec_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
def allocation(spec_path: Path) -> None:
    """Run one allocation (the child of ``run``); write ``terminal.json`` beside the spec."""
    spec = load_spec(spec_path)
    terminal = asyncio.run(run_allocation(spec))
    events = json.loads((Path(spec.root) / "workpieces" / spec.allocation.workpiece_id / "events.json").read_text())[
        "events"
    ]
    last = next((e for e in reversed(events) if e["kind"] == "terminal"), {"attrs": {}})
    record = {"terminal": str(terminal), "kind": last["attrs"].get("kind"), "reason": last["attrs"].get("reason")}
    (spec_path.parent / TERMINAL_FILE).write_text(json.dumps(record) + "\n")


@main.command("dry-run")
@click.option("--sample", "sample_path", required=True, help="sample.json; the first --proposals entries are rendered.")
@click.option("--proposals", "count", type=click.IntRange(min=1), default=3, show_default=True)
@click.option("--output", type=click.Path(file_okay=False, path_type=Path), required=True)
def dry_run_command(sample_path: str, count: int, output: Path) -> None:
    """Render every arm on the first sampled proposals with no model, diff the arms, write ``diff.json``."""
    sample = json.loads(read_source(sample_path))
    proposals = [
        (e["capability"], read_proposal(read_source(e["proposal"]), e["sha256"])) for e in sample["sample"][:count]
    ]
    summary = asyncio.run(dry_run(load_arms(), proposals, output))
    for pair, entry in summary["pairs"].items():
        click.echo(f"{pair}: {len(entry['changed_keys'])} keys changed; surfaces {', '.join(entry['changed_surfaces'])}")
    click.echo(f"no unintended differences; {output / 'diff.json'}")


@main.command("manifest")
@click.option("--sample", "sample_path", required=True)
@click.option("--git-sha", required=True)
def manifest_command(sample_path: str, git_sha: str) -> None:
    """Print the manifest a run would write, with its digest."""
    document = manifest(read_source(sample_path), git_sha_of(git_sha))
    click.echo(json.dumps(document | {"digest": manifest_digest(document)}, indent=2))


if __name__ == "__main__":
    main()
