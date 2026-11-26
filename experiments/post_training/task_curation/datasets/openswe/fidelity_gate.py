# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The OpenSWE fidelity gate: the upstream-built image against the converted path.

Ruling 2026-11-26 6:03 AM (option a): qualification is the fidelity gate. Per
sampled task (at least 20 rows per condensed python base, deterministic seed)
BOTH arms run the oracle and the no-op control on Daytona, following the
cohort-grader precedent — the verifier runs on the task machine:

- the UPSTREAM arm builds the row's Dockerfile with the repository checkout as
  the build context, exactly as upstream built images. Only the never-published
  ``openswe-python-<v>`` FROM binds to a fresh in-region build of the upstream
  ``prepare_baseimg.py`` recipe (the upstream-fidelity base); every other byte
  passes through, including the ``COPY repo`` build.
- the CONDENSED arm builds the conversion's own Dockerfile bytes (the
  digest-pinned condensed base, build-time clone+checkout) and stages the exact
  package files the conversion ships (``tests/test.sh``, ``tests/eval_script.sh``,
  ``solution/*``).

Both arms must score the oracle 1 and the no-op 0 — identical outcomes. The
oracle runs in place where the image keeps its git repository (``git reset
--hard`` + ``git clean -fd``, keeping the ignored build artifacts a production
agent machine carries); otherwise a fresh machine from the same arm build
stages the oracle. Any divergence, unscored verdict, or build failure is a
typed exclusion with per-row evidence, and the driver exits nonzero — fail
closed.
"""

import asyncio
import json
import logging
import os
import random
import shlex
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

from shellbox.backends.daytona.machine import DaytonaDeclarativeMachineFactory
from shellbox.image import DockerfileSource
from shellbox.machine import Command, Machine, MachineFactory, MachineSpec, NetworkPolicy

from experiments.post_training.task_curation.datasets.openswe import bases, convert, grader
from experiments.post_training.task_curation.datasets.openswe.dockerfile_translation import (
    CONDA_BASE_PATTERN,
    DIGEST_PIN_PATTERN,
    dockerfile_instructions,
)
from experiments.post_training.task_curation.datasets.openswe.sources import OpenSweRow, keep_release_rows

LOGGER = logging.getLogger(__name__)

ARM_UPSTREAM = "upstream"
ARM_CONDENSED = "condensed"
GATE_SEED = 20261126
SAMPLES_PER_BASE = 20
ORACLE_TIMEOUT_SEC = 600.0
GIT_TIMEOUT_SEC = 120.0
CREATE_ATTEMPTS = 3
CREATE_BACKOFF_SECONDS = (15.0, 45.0)
EVIDENCE_OUTPUT_TAIL = 3000

STAGED_FILES = ("tests/test.sh", "tests/eval_script.sh", "solution/gold.patch", "solution/solve.sh")
UPSTREAM_BINDINGS_FILENAME = "upstream-fidelity-bindings.json"


class FidelityExclusion(Exception):
    """A typed per-row exclusion with evidence; the row leaves the qualified population."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class ControlRecord:
    """One oracle or no-op control run with its verdict evidence."""

    control: str
    reward: int
    eval_exit_status: str | None
    protocol: str
    machine: str
    output_tail: str

    def as_dict(self) -> dict:
        return dict(
            control=self.control,
            reward=self.reward,
            eval_exit_status=self.eval_exit_status,
            protocol=self.protocol,
            machine=self.machine,
            output_tail=self.output_tail,
        )


@dataclass(frozen=True)
class ArmRecord:
    """One arm's controls, or the typed failure that kept it from completing."""

    arm: str
    noop: ControlRecord | None = None
    oracle: ControlRecord | None = None
    failure_reason: str | None = None
    failure_detail: str | None = None

    def as_dict(self) -> dict:
        return dict(
            arm=self.arm,
            noop=self.noop.as_dict() if self.noop else None,
            oracle=self.oracle.as_dict() if self.oracle else None,
            failure_reason=self.failure_reason,
            failure_detail=self.failure_detail,
        )


@dataclass(frozen=True)
class TaskRecord:
    """One sampled row's gate decision with per-row evidence."""

    instance_id: str
    base: str
    status: str
    exclusion_reason: str | None = None
    exclusion_detail: str | None = None
    arms: dict[str, ArmRecord] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(
            instance_id=self.instance_id,
            base=self.base,
            status=self.status,
            exclusion_reason=self.exclusion_reason,
            exclusion_detail=self.exclusion_detail,
            arms={name: record.as_dict() for name, record in self.arms.items()},
        )


def row_base(row: OpenSweRow) -> str | None:
    """The row's ``openswe-python-<version>`` root base, or None when it names none."""
    for instruction in dockerfile_instructions(row.dockerfile):
        if instruction.name != "FROM":
            continue
        fields = [token for token in instruction.argument.split() if not token.startswith("--")]
        if not fields:
            continue
        match = CONDA_BASE_PATTERN.fullmatch(fields[0])
        if match:
            return match.group("version")
    return None


def _literal_conda_froms(row: OpenSweRow) -> bool:
    """True when every root FROM the row declares is a literal reference.

    Variable FROMs never convert (typed rejection), so the gate records them as
    unbaseable instead of sampling rows that cannot reach admission.
    """
    for instruction in dockerfile_instructions(row.dockerfile):
        if instruction.name != "FROM":
            continue
        fields = [token for token in instruction.argument.split() if not token.startswith("--")]
        if fields and "$" in fields[0]:
            return False
    return True


def sample_by_base(
    rows: list[OpenSweRow], *, per_base: int, seed: int = GATE_SEED
) -> tuple[dict[str, list[OpenSweRow]], dict[str, int]]:
    """A deterministic sample per python base plus the skipped-row census.

    Rows whose FROM is not a literal ``openswe-python-<version>`` reference (no
    condensed base, or a variable reference the converter types rejects) never
    sample; the census records how many each group holds.
    """
    grouped: dict[str, list[OpenSweRow]] = {}
    skipped: dict[str, int] = {"other_base": 0, "variable_from": 0}
    for row in rows:
        if not _literal_conda_froms(row):
            skipped["variable_from"] += 1
            continue
        base = row_base(row)
        if base is None:
            skipped["other_base"] += 1
        else:
            grouped.setdefault(base, []).append(row)
    sampler = random.Random(seed)
    samples: dict[str, list[OpenSweRow]] = {}
    for base in sorted(grouped):
        population = list(grouped[base])
        sampler.shuffle(population)
        samples[base] = population[:per_base]
    return samples, skipped


def upstream_dockerfile(text: str, upstream_bases: dict[str, str]) -> str:
    """Bind the never-published upstream bases; every other byte passes through.

    Only root ``FROM openswe-python-<version>`` references are rewritten to the
    upstream-fidelity digest pins — the build context (the repository checkout)
    and the ``COPY repo`` instructions stay verbatim. A variable reference is a
    typed exclusion; other registry bases pass through for Daytona to pull.
    """
    lines = text.splitlines()
    stages: set[str] = set()
    for instruction in dockerfile_instructions(text):
        if instruction.name != "FROM":
            continue
        fields = [token for token in instruction.argument.split() if not token.startswith("--")]
        if not fields:
            raise FidelityExclusion("empty_from", f"line {instruction.line_number}: an empty FROM instruction")
        base, stage = fields[0], None
        if len(fields) >= 3 and fields[1].upper() == "AS":
            stage = fields[2]
        if base.lower() in stages or base == "scratch":
            if stage:
                stages.add(stage.lower())
            continue
        match = CONDA_BASE_PATTERN.fullmatch(base)
        if match:
            pinned = upstream_bases.get(match.group("version"))
            if not pinned or not DIGEST_PIN_PATTERN.fullmatch(pinned):
                raise FidelityExclusion(
                    "unbound_upstream_base",
                    f"line {instruction.line_number}: no digest-pinned upstream-fidelity base covers "
                    f"Python {match.group('version')}",
                )
            lines[instruction.line_number - 1] = lines[instruction.line_number - 1].replace(base, pinned, 1)
        elif "$" in base:
            raise FidelityExclusion(
                "variable_from", f"line {instruction.line_number}: FROM {base!r} is not a literal reference"
            )
        if stage:
            stages.add(stage.lower())
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _last_workdir(text: str) -> str:
    for instruction in reversed(dockerfile_instructions(text)):
        if instruction.name == "WORKDIR":
            return instruction.argument.strip()
    return "/testbed"


def _instance_slug(instance_id: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in instance_id)


def clone_context(row: OpenSweRow, root: Path) -> Path:
    """The upstream build context: the repository checkout at base_commit.

    Upstream built images with the checkout itself as the context
    (``scripts/build_images.py``), so the context carries ``repo/`` and the
    Dockerfile at its root.
    """
    checkout = root / _instance_slug(row.instance_id) / "repo"
    checkout.parent.mkdir(parents=True, exist_ok=True)
    clone = subprocess.run(
        ["git", "clone", "--quiet", f"https://github.com/{row.repo}.git", str(checkout)],
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SEC,
    )
    if clone.returncode:
        raise FidelityExclusion("clone_failed", f"git clone {row.repo}: {clone.stderr.strip()[-400:]}")
    check = subprocess.run(
        ["git", "-C", str(checkout), "checkout", "--quiet", row.base_commit],
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SEC,
    )
    if check.returncode:
        raise FidelityExclusion("checkout_failed", f"checkout {row.base_commit}: {check.stderr.strip()[-400:]}")
    return checkout.parent


def arm_contexts(
    row: OpenSweRow,
    condensed_bases: dict[str, str],
    upstream_bases: dict[str, str],
    scratch: Path,
) -> tuple[dict[str, tuple[Path, str]], Path]:
    """Both arms' build contexts and workdirs, plus the staged task files.

    The package renders through the production converter first: a row the
    converter types rejects never gates (it never reaches admission), and a
    convertible row gates on exactly the bytes the conversion ships.
    """
    try:
        files = convert.package_files(row, condensed_bases)
    except convert.OpenSweRowExcluded as excluded:
        raise FidelityExclusion(f"converter_{excluded.reason}", excluded.detail) from excluded
    staged_dir = scratch / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)
    for name in STAGED_FILES:
        (staged_dir / name.replace("/", "_")).write_bytes(files[name])

    condensed_text = files["environment/Dockerfile"].decode()
    condensed_dir = scratch / "condensed" / _instance_slug(row.instance_id)
    condensed_dir.mkdir(parents=True, exist_ok=True)
    (condensed_dir / "Dockerfile").write_text(condensed_text)

    upstream_text = upstream_dockerfile(row.dockerfile, upstream_bases)
    upstream_dir = clone_context(row, scratch / "upstream")
    (upstream_dir / "Dockerfile").write_text(upstream_text)

    contexts = {
        ARM_CONDENSED: (condensed_dir, _last_workdir(condensed_text)),
        ARM_UPSTREAM: (upstream_dir, _last_workdir(upstream_text)),
    }
    return contexts, staged_dir


async def create_machine(factory: MachineFactory, context: Path, workdir: str) -> Machine:
    """One task machine from a build context, retried through transient failures."""
    spec = MachineSpec(
        source=DockerfileSource(context, context / "Dockerfile"),
        workdir=workdir,
        network=NetworkPolicy.ALLOW,
        cpus=convert.CPUS,
        memory_mb=convert.MEMORY_MB,
        storage_mb=convert.STORAGE_MB,
        startup_timeout=convert.BUILD_TIMEOUT_SEC,
    )
    for attempt in range(CREATE_ATTEMPTS):
        try:
            return await factory.create(spec)
        except Exception as error:
            if attempt == CREATE_ATTEMPTS - 1:
                raise
            LOGGER.warning("machine create attempt %d failed: %s", attempt + 1, str(error)[:300])
            await asyncio.sleep(CREATE_BACKOFF_SECONDS[min(attempt, len(CREATE_BACKOFF_SECONDS) - 1)])
    raise AssertionError("unreachable")


async def _download(machine: Machine, source: str) -> bytes | None:
    target = Path(tempfile.gettempdir()) / f"gate-{uuid.uuid4().hex}"
    try:
        await machine.download(source, target)
        return target.read_bytes()
    except Exception:
        return None
    finally:
        target.unlink(missing_ok=True)


async def stage_task_files(machine: Machine, staged_dir: Path) -> None:
    """Install the task package's tests and solution at the production paths."""
    await machine.run(Command(("mkdir", "-p", "/tests", "/solution"), timeout=60.0))
    for name in STAGED_FILES:
        await machine.upload(staged_dir / name.replace("/", "_"), f"/{name}")
    await machine.run(Command(("chmod", "+x", *STAGED_FILES), timeout=60.0))


def _parse_reward(control: str, reward_bytes: bytes | None, output_tail: str) -> int:
    if reward_bytes is None:
        raise FidelityExclusion(
            "verifier_unscored",
            f"the {control} control's reward file is missing; output tail: {output_tail[-600:]}",
        )
    try:
        return int(reward_bytes.decode().strip())
    except ValueError as error:
        raise FidelityExclusion(
            "verifier_unscored",
            f"the {control} control's reward file is malformed: {reward_bytes[:64]!r}",
        ) from error


async def run_verifier(machine: Machine, *, control: str, machine_name: str) -> ControlRecord:
    """Run the fail-closed verifier and read its recorded verdict.

    The verifier script always exits 0 — the reward file arbitrates. The
    recorded eval exit status and the output tail ride as evidence, and the
    reward file must agree with the output's own score.
    """
    await machine.run(Command(("bash", "/tests/test.sh"), timeout=convert.VERIFIER_TIMEOUT_SEC))
    reward_bytes = await _download(machine, "/logs/verifier/reward.txt")
    status_bytes = await _download(machine, "/logs/verifier/eval-exit-status.txt")
    output_bytes = await _download(machine, "/logs/verifier/eval-output.txt")
    output = (output_bytes or b"").decode(errors="replace")
    reward = _parse_reward(control, reward_bytes, output[-EVIDENCE_OUTPUT_TAIL:])
    recomputed = grader.reward(output)
    if reward != recomputed:
        raise FidelityExclusion(
            "reward_divergence",
            f"the {control} control's reward file records {reward} but its eval output scores {recomputed}; "
            f"output tail: {output[-600:]}",
        )
    return ControlRecord(
        control=control,
        reward=reward,
        eval_exit_status=(status_bytes or b"").decode(errors="replace").strip() or None,
        protocol="verifier",
        machine=machine_name,
        output_tail=output[-EVIDENCE_OUTPUT_TAIL:],
    )


def _machine_name(machine: Machine) -> str:
    return str(getattr(machine, "name", "") or "machine")


async def _in_place_oracle(machine: Machine, workdir: str, machine_name: str) -> ControlRecord | None:
    """The oracle on the machine the no-op just ran on, or None for the fresh fallback.

    ``git reset --hard`` + ``git clean -fd`` restores the recorded bug state
    while keeping the ignored build artifacts the image build left.
    """
    probe = await machine.run(Command(("git", "-C", workdir, "rev-parse", "--show-toplevel"), timeout=GIT_TIMEOUT_SEC))
    root = probe.stdout.decode(errors="replace").strip()
    if probe.exit_code != 0 or not root:
        return None
    reset = await machine.run(
        Command(
            ("sh", "-c", f"git -C {shlex.quote(root)} reset --hard HEAD && git -C {shlex.quote(root)} clean -fd"),
            timeout=GIT_TIMEOUT_SEC,
        )
    )
    if reset.exit_code != 0:
        return None
    solve = await machine.run(Command(("bash", "/solution/solve.sh"), timeout=ORACLE_TIMEOUT_SEC))
    if solve.exit_code != 0:
        return None
    record = await run_verifier(machine, control="oracle", machine_name=machine_name)
    return replace(record, protocol="in_place")


async def _fresh_oracle(factory: MachineFactory, context: Path, workdir: str, staged_dir: Path) -> ControlRecord:
    """The oracle control on a fresh machine from the same arm build."""
    machine = await create_machine(factory, context, workdir)
    machine_name = _machine_name(machine)
    try:
        await stage_task_files(machine, staged_dir)
        solve = await machine.run(Command(("bash", "/solution/solve.sh"), timeout=ORACLE_TIMEOUT_SEC))
        if solve.exit_code != 0:
            detail = solve.stderr.decode(errors="replace")[-600:] or solve.stdout.decode(errors="replace")[-600:]
            raise FidelityExclusion("oracle_apply_failed", f"solve.sh exited {solve.exit_code}: {detail}")
        record = await run_verifier(machine, control="oracle", machine_name=machine_name)
        return replace(record, protocol="fresh")
    finally:
        await machine.close()


async def run_arm(arm: str, context: Path, workdir: str, staged_dir: Path, factory: MachineFactory) -> ArmRecord:
    """Both controls for one arm, or the typed failure that kept them from running."""
    try:
        machine = await create_machine(factory, context, workdir)
    except Exception as error:
        return ArmRecord(arm=arm, failure_reason="build_failed", failure_detail=str(error)[:1200])
    machine_name = _machine_name(machine)
    try:
        await stage_task_files(machine, staged_dir)
        noop = await run_verifier(machine, control="noop", machine_name=machine_name)
        oracle = await _in_place_oracle(machine, workdir, machine_name)
    finally:
        await machine.close()
    if oracle is None:
        oracle = await _fresh_oracle(factory, context, workdir, staged_dir)
    return ArmRecord(arm=arm, noop=noop, oracle=oracle)


def decide(instance_id: str, base: str, arms: dict[str, ArmRecord]) -> TaskRecord:
    """The gate decision for one row from its two arms' evidence. Fails closed."""
    for arm, record in arms.items():
        if record.failure_reason is not None:
            return TaskRecord(
                instance_id,
                base,
                "excluded",
                f"{arm}_{record.failure_reason}",
                record.failure_detail,
                arms,
            )
    for arm, record in arms.items():
        if record.noop is None or record.oracle is None:
            return TaskRecord(
                instance_id, base, "excluded", f"{arm}_control_missing", f"{arm} ran no complete control", arms
            )
    for arm, record in arms.items():
        if record.noop.reward != 0:
            detail = f"{arm} no-op scored {record.noop.reward}; output tail: {record.noop.output_tail[-600:]}"
            return TaskRecord(instance_id, base, "excluded", f"{arm}_noop_mismatch", detail, arms)
    for arm, record in arms.items():
        if record.oracle.reward != 1:
            detail = f"{arm} oracle scored {record.oracle.reward}; output tail: {record.oracle.output_tail[-600:]}"
            return TaskRecord(instance_id, base, "excluded", f"{arm}_oracle_mismatch", detail, arms)
    return TaskRecord(instance_id, base, "qualified", None, None, arms)


async def evaluate_task(
    row: OpenSweRow,
    condensed_bases: dict[str, str],
    upstream_bases: dict[str, str],
    factory: MachineFactory,
    scratch_root: Path,
) -> TaskRecord:
    """Gate one sampled row: render, run both arms, decide fail-closed."""
    base = row_base(row) or "other"
    scratch = scratch_root / _instance_slug(row.instance_id)
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        contexts, staged_dir = arm_contexts(row, condensed_bases, upstream_bases, scratch)
        arms: dict[str, ArmRecord] = {}
        for arm, (context, workdir) in contexts.items():
            try:
                arms[arm] = await run_arm(arm, context, workdir, staged_dir, factory)
            except FidelityExclusion as excluded:
                arms[arm] = ArmRecord(arm=arm, failure_reason=excluded.reason, failure_detail=excluded.detail)
        return decide(row.instance_id, base, arms)
    except FidelityExclusion as excluded:
        return TaskRecord(row.instance_id, base, "excluded", excluded.reason, excluded.detail, {})
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


async def run_gate(
    samples: dict[str, list[OpenSweRow]],
    condensed_bases: dict[str, str],
    upstream_bases: dict[str, str],
    factory: MachineFactory,
    *,
    report_dir: Path,
    concurrency: int = 4,
) -> dict:
    """Gate every sampled row under a bounded sandbox concurrency.

    Every sampled row produces a record — qualified or a typed exclusion with
    evidence. An unexpected error is itself a typed exclusion row: the gate
    never silently drops a sampled task.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    counts = {"qualified": 0, "excluded": 0}
    per_reason: dict[str, int] = {}
    per_base: dict[str, dict[str, int]] = {}

    async def one(row: OpenSweRow) -> TaskRecord:
        async with semaphore:
            try:
                return await evaluate_task(row, condensed_bases, upstream_bases, factory, report_dir / "scratch")
            except Exception as error:
                LOGGER.exception("unexpected gate failure for %s", row.instance_id)
                return TaskRecord(
                    row.instance_id,
                    row_base(row) or "other",
                    "excluded",
                    "unexpected_error",
                    str(error)[:1200],
                    {},
                )

    pending = [one(row) for base in sorted(samples) for row in samples[base]]
    with (report_dir / "rows.jsonl").open("w") as stream:
        for completed in asyncio.as_completed(pending):
            record = await completed
            stream.write(json.dumps(record.as_dict(), sort_keys=True) + "\n")
            stream.flush()
            state = "qualified" if record.status == "qualified" else "excluded"
            counts[state] += 1
            if state == "excluded":
                reason = record.exclusion_reason or "untyped"
                per_reason[reason] = per_reason.get(reason, 0) + 1
            base_counts = per_base.setdefault(record.base, {"qualified": 0, "excluded": 0})
            base_counts[state] += 1
    summary = {
        "sampled": len(pending),
        "qualified": counts["qualified"],
        "excluded": counts["excluded"],
        "exclusions_by_reason": dict(sorted(per_reason.items())),
        "per_base": {base: dict(sorted(base_counts.items())) for base, base_counts in sorted(per_base.items())},
        "state": "ok" if counts["excluded"] == 0 else "mismatched",
    }
    (report_dir / "fidelity-gate-result.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    import argparse  # noqa: PLC0415

    from shellbox.backends.daytona.connection import DaytonaConnection  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condensed-bindings", required=True, help="The staged condensed base bindings URI.")
    parser.add_argument("--upstream-bindings", required=True, help="The staged upstream-fidelity base bindings URI.")
    parser.add_argument("--report-dir", type=Path, required=True, help="The local report directory.")
    parser.add_argument("--keep-release-dir", help="A local checkout of the keep-only release.")
    parser.add_argument("--per-base", type=int, default=SAMPLES_PER_BASE)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--bases", help="A comma-separated base filter (e.g. 3.9,3.11).")
    parser.add_argument("--seed", type=int, default=GATE_SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    api_key = os.environ.get("DAYTONA_API_KEY")
    if not api_key:
        raise SystemExit("the gate runs on Daytona: set DAYTONA_API_KEY (the eval account) before launching")
    from rigging.filesystem.s3_compat import configure_coreweave_s3  # noqa: PLC0415

    configure_coreweave_s3()
    condensed_bases = bases.load_condensed_bases(args.condensed_bindings)
    upstream_bases = bases.load_bindings(args.upstream_bindings, UPSTREAM_BINDINGS_FILENAME)
    rows = keep_release_rows(args.keep_release_dir)
    samples, skipped = sample_by_base(rows, per_base=args.per_base, seed=args.seed)
    if args.bases:
        keep = set(args.bases.split(","))
        samples = {base: population for base, population in samples.items() if base in keep}
    LOGGER.info(
        "gate sample: %d rows across %d bases; skipped %s",
        sum(len(population) for population in samples.values()),
        len(samples),
        skipped,
    )
    (args.report_dir / "sampling.json").write_text(
        json.dumps(
            {
                "per_base": {
                    base: [row.instance_id for row in population] for base, population in sorted(samples.items())
                },
                "skipped": skipped,
                "seed": args.seed,
                "per_base_count": args.per_base,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    connection = DaytonaConnection(
        api_key=api_key,
        api_url=os.environ.get("DAYTONA_API_URL", "https://app.daytona.io/api"),
        target=os.environ.get("DAYTONA_TARGET", "us"),
    )
    factory = DaytonaDeclarativeMachineFactory(client_factory=connection.client_factory())
    summary = asyncio.run(
        run_gate(
            samples,
            condensed_bases,
            upstream_bases,
            factory,
            report_dir=args.report_dir,
            concurrency=args.concurrency,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["state"] != "ok":
        raise SystemExit(f"fidelity gate mismatched: {summary['exclusions_by_reason']}")


if __name__ == "__main__":
    main()
