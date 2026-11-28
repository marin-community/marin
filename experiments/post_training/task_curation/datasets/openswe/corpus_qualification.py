# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The full-corpus oracle qualification for the repaired OpenSWE population.

Coordinator ruling 2026-11-28: the fidelity gate's 220-row sample excludes only
its own rows; at a 44 percent sample exclusion rate the unsampled remainder
would publish tasks whose environment cannot build or whose grader cannot pass
its own oracle — dead tasks that always score 0. Before admission, every row of
the repaired population runs the oracle arm (and the no-op arm, which rides the
same machine) through the gate's own one-arm machinery, sharded across the
Daytona EVAL account within the org quota, and every failing row leaves with a
typed exclusion. Only oracle-qualified rows admit.

A shard is an index-range split of the qualified population: the classifier's
repairable and clean rows, minus the gate root's own exclusions, minus rows the
converter rejects (those are conversion exclusions, not qualification
failures). Each shard writes the gate's report shapes — rows.jsonl and
fidelity-gate-result.json — so the merged corpus report is a gate root the
conversion consumes unchanged.
"""

import asyncio
import json
import logging
import shutil
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.task_curation.datasets.openswe import bases, convert, grader_repair
from experiments.post_training.task_curation.datasets.openswe.fidelity_gate import (
    ARM_CONDENSED,
    STAGED_FILES,
    ArmRecord,
    FidelityExclusion,
    TaskRecord,
    _instance_slug,
    _last_workdir,
    run_arm,
)
from experiments.post_training.task_curation.datasets.openswe.sources import OpenSweRow

LOGGER = logging.getLogger(__name__)


def condensed_context(row: OpenSweRow, condensed_bases: dict[str, str], scratch: Path) -> tuple[Path, str, Path]:
    """The condensed arm's build context, workdir and staged files.

    The gate's ``arm_contexts`` builds both arms; the corpus qualification runs
    the release's own grader only, so only the condensed half is built.
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
    return condensed_dir, _last_workdir(condensed_text), staged_dir


def decide_condensed(instance_id: str, base: str, arm) -> TaskRecord:
    """The qualification decision for one row from its condensed-arm evidence.

    Fails closed: the no-op must score 0 and the oracle must score 1 — the
    grader distinguishes the agent's diff from the reference solution.
    """
    if arm.failure_reason is not None:
        return TaskRecord(instance_id, base, "excluded", arm.failure_reason, arm.failure_detail, {"condensed": arm})
    if arm.noop is None or arm.oracle is None:
        return TaskRecord(
            instance_id,
            base,
            "excluded",
            "condensed_control_missing",
            "the arm ran no complete control",
            {"condensed": arm},
        )
    if arm.noop.reward != 0:
        detail = f"condensed no-op scored {arm.noop.reward}; tail: {arm.noop.output_tail[-600:]}"
        return TaskRecord(instance_id, base, "excluded", "condensed_noop_mismatch", detail, {"condensed": arm})
    if arm.oracle.reward != 1:
        detail = f"condensed oracle scored {arm.oracle.reward}; output tail: {arm.oracle.output_tail[-600:]}"
        return TaskRecord(instance_id, base, "excluded", "condensed_oracle_mismatch", detail, {"condensed": arm})
    return TaskRecord(instance_id, base, "qualified", None, None, {"condensed": arm})


class StubMachine:
    """The control-sequence state machine the driver's plumbing runs against
    when the Daytona sandbox factory is stubbed: staging succeeds, the
    verifier's reward follows the golden application (0 before solve.sh, 1
    after), the recorded output carries the matching OPENSWE_EXIT_CODE marker,
    and everything else exits 0. The factory never touches Daytona."""

    def __init__(self, index: int):
        self.name = f"stub-machine-{index}"
        self.golden_applied = False
        self.commands: list[str] = []

    async def run(self, command):
        from shellbox.machine import ExitReason, Result  # noqa: PLC0415

        self.commands.append(" ".join(command.argv)[:200])
        if command.argv[:2] == ("bash", "/solution/solve.sh"):
            self.golden_applied = True
        if "rev-parse" in " ".join(command.argv):
            return Result(
                exit_code=0,
                stdout=b"/testbed\n",
                stderr=b"",
                stdout_truncated=False,
                stderr_truncated=False,
                reason=ExitReason.EXITED,
            )
        return Result(
            exit_code=0,
            stdout=b"",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
            reason=ExitReason.EXITED,
        )

    async def upload(self, source: Path, target: str) -> None:
        return None

    async def download(self, source: str, target: Path) -> None:
        marker = b"0" if self.golden_applied else b"2"
        if source.endswith("reward.txt"):
            payload = b"1" if self.golden_applied else b"0"
        elif source.endswith("eval-exit-status.txt"):
            payload = b"0"
        else:
            payload = (
                ">>>>> Start Test Output\n"
                "stub eval output\n"
                f"OPENSWE_EXIT_CODE={marker.decode()}\n"
                ">>>>> End Test Output\n"
            ).encode()
        target.write_bytes(payload)

    async def close(self) -> None:
        return None


class StubMachineFactory:
    """The sandbox factory the local end-to-end run uses: every create returns
    a StubMachine; no Daytona account is touched."""

    def __init__(self) -> None:
        self.created = 0

    async def create(self, spec):
        self.created += 1
        return StubMachine(self.created)


def qualification_population(
    rows: list[OpenSweRow], excluded_instance_ids: frozenset[str]
) -> tuple[list[OpenSweRow], dict[str, int]]:
    """The rows the corpus run qualifies: convertible, repair-or-clean, not
    already excluded by the gate's sample run."""
    counts = {"scanned": len(rows), "sample_records": 0, "converter_unrepairable": 0, "population": 0}
    population: list[OpenSweRow] = []
    for row in rows:
        if row.instance_id in excluded_instance_ids:
            counts["sample_records"] += 1
            continue
        classification = grader_repair.classify_grader(row)
        if classification.verdict == "unrepairable":
            counts["converter_unrepairable"] += 1
            continue
        population.append(row)
    counts["population"] = len(population)
    return population, counts


def shard_population(population: list[OpenSweRow], shard_index: int, shard_count: int) -> list[OpenSweRow]:
    """A deterministic index-range split; shard boundaries never overlap."""
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard index {shard_index} outside 0..{shard_count - 1}")
    return population[shard_index::shard_count]


async def run_shard(
    samples: list[OpenSweRow],
    condensed_bases: dict[str, str],
    factory,
    *,
    report_dir: Path,
    concurrency: int = 4,
) -> dict:
    """Qualify every row of the shard under a bounded sandbox concurrency."""
    report_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    counts = {"qualified": 0, "excluded": 0}
    per_reason: dict[str, int] = {}

    async def one(row: OpenSweRow) -> TaskRecord:
        async with semaphore:
            base = "corpus"
            scratch = report_dir / "scratch" / _instance_slug(row.instance_id)
            scratch.mkdir(parents=True, exist_ok=True)
            try:
                context, workdir, staged_dir = condensed_context(row, condensed_bases, scratch)
                try:
                    arm = await run_arm(ARM_CONDENSED, context, workdir, staged_dir, factory)
                except FidelityExclusion as excluded:
                    arm = ArmRecord(arm=ARM_CONDENSED, failure_reason=excluded.reason, failure_detail=excluded.detail)
                return decide_condensed(row.instance_id, base, arm)
            except FidelityExclusion as excluded:
                return TaskRecord(row.instance_id, base, "excluded", excluded.reason, excluded.detail, {})
            except Exception as error:
                LOGGER.exception("unexpected qualification failure for %s", row.instance_id)
                return TaskRecord(row.instance_id, base, "excluded", "unexpected_error", str(error)[:1200], {})
            finally:
                shutil.rmtree(scratch, ignore_errors=True)

    pending = [one(row) for row in samples]
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
    summary = {
        "sampled": len(pending),
        "qualified": counts["qualified"],
        "excluded": counts["excluded"],
        "exclusions_by_reason": dict(sorted(per_reason.items())),
        "state": "ok" if counts["excluded"] == 0 else "mismatched",
        "shard": True,
    }
    (report_dir / "fidelity-gate-result.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    import argparse  # noqa: PLC0415
    import os  # noqa: PLC0415

    from shellbox.backends.daytona.connection import DaytonaConnection  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--condensed-bindings", required=True)
    parser.add_argument("--gate-root", required=True, help="The gate root whose exclusions the corpus run carries.")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Run the plumbing end to end against the stubbed sandbox factory (no Daytona account).",
    )
    parser.add_argument(
        "--limit", type=int, help="Qualify only the first N population rows (the stubbed end-to-end run)."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.stub:
        api_key = "stub"
    else:
        api_key = os.environ.get("DAYTONA_API_KEY")
        if not api_key:
            raise SystemExit(
                "the qualification runs on Daytona: set DAYTONA_API_KEY (the eval account) before launching"
            )
    configure_coreweave_s3()
    from experiments.post_training.task_curation.datasets.openswe.fidelity_gate import (  # noqa: PLC0415
        DaytonaDeclarativeMachineFactory,
        load_gate_evidence,
    )
    from experiments.post_training.task_curation.datasets.openswe.sources import keep_release_rows  # noqa: PLC0415

    condensed_bases = bases.load_condensed_bases(args.condensed_bindings)
    rows = keep_release_rows()
    evidence = load_gate_evidence(args.gate_root)
    # The gate's own sample rows keep their records (qualified and excluded
    # alike); the corpus run covers the rest of the population, so the merged
    # report carries one evidence record per row without re-running the sample.
    population, counts = qualification_population(rows, frozenset(evidence.excluded) | evidence.qualified)
    if args.limit is not None:
        population = population[: args.limit]
    shard = shard_population(population, args.shard_index, args.shard_count)
    LOGGER.info(
        "shard %d/%d: %d of %d population rows (scanned %d, sample records %d, converter-unrepairable %d)",
        args.shard_index,
        args.shard_count,
        len(shard),
        counts["population"],
        counts["scanned"],
        counts["sample_records"],
        counts["converter_unrepairable"],
    )
    (args.report_dir / "sampling.json").write_text(
        json.dumps(
            {
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "population": counts["population"],
                "shard_rows": len(shard),
                "instance_ids": [row.instance_id for row in shard],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if args.stub:
        factory = StubMachineFactory()
    else:
        connection = DaytonaConnection(
            api_key=api_key,
            api_url=os.environ.get("DAYTONA_API_URL", "https://app.daytona.io/api"),
            target=os.environ.get("DAYTONA_TARGET", "us"),
        )
        factory = DaytonaDeclarativeMachineFactory(client_factory=connection.client_factory())
    summary = asyncio.run(
        run_shard(shard, condensed_bases, factory, report_dir=args.report_dir, concurrency=args.concurrency)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
