# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select and package accepted forge tasks as a TaskCompendium cohort for the RL Data Atlas.

Forge authors each task from a capability proposal and validates it before accepting it: authored
controls replay through the grader, three solver trials run per validation round, two adversaries
try to make the grader credit a wrong answer and refuse a right one, and a review agent reads the
solver transcripts for task defects. Defects send the task back to its author. An accepted task then
runs five more solver trials and a final review. The forge export
(``experiments/forge/export_tasks.py`` on the forge branch) writes one record per accepted task:
the workpiece without its author session, and the rollout spec its confirmation trials ran.

This module reads those records and:

1. keeps the *clean* accepts: no notes (defects accepted with the task) and no missing evidence;
2. selects ``--target`` of them, coverage first: every capability's best task, then every
   capability's second best, and so on, breaking the last partial tier by score;
3. writes ``lowered-tasks.parquet`` (``path``, ``lowered_task_spec`` envelopes) for the selected
   tasks, ``selection.jsonl`` (every candidate with its score and why it was or was not selected)
   and ``manifest.json`` (counts, hashes and the selection rule).

Run with CoreWeave credentials exported::

    uv run python -m experiments.post_training.task_curation.forge_release \
        --records s3://marin-us-east-02a/marin/forge-viz/v1 --target 4000 --output /tmp/forge-release
"""

import argparse
import concurrent.futures
import gzip
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import CommandSemantics, TaskSpec

from experiments.post_training.task_curation.lowered_envelope import LOWERED_TASK_SPEC_COLUMN, LoweredSpec, MachineShape

RECORD_SCHEMA = "forge-task-record/1"
ACCEPTED_LIST = "inputs/accepted.tsv"
READ_THREADS = 64
LOWERED_FILENAME = "lowered-tasks.parquet"
SELECTION_FILENAME = "selection.jsonl"
MANIFEST_FILENAME = "manifest.json"
SELECTION_RULE = (
    "Clean accepts only (no notes, no missing evidence). Within each capability, tasks are ranked by score; "
    "every capability's rank-1 task is taken, then every rank-2 task, and so on, and the last partial rank "
    "is filled by score. Score = 1 for an in-band solve rate over the eight confirmation trials "
    "(0.25 <= p <= 0.75) + 4p(1-p) (largest at p = 0.5, where a trial says most about the policy) + 0.25 "
    "when the accepted submission was not a revert to an earlier one + 0.1 x min(controls, 20)/20; ties "
    "break on the SHA-256 of the task id."
)
CONTROLS_CREDIT_CAP = 20
# Copies of ``taskforge.sandbox.images.GRADER_BASE_IMAGE`` and the serialized ``taskforge.spec.draft.LOCAL_CPYTHON``:
# taskforge is not a dependency of this package. A drift fails closed, since ``restored_task`` refuses any restored
# machine that differs from its authored spec.
GRADER_BASE_IMAGE = "docker.io/library/python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1"
EMPTY_PACKAGES = {"pypi_packages": [], "tools": []}
ENVIRONMENTS = {CommandSemantics.SHELL_SIMULATOR: "shellsim", CommandSemantics.LINUX_PROCESS: "linux", None: "reasoning"}
"""A task's environment, by the command semantics of its (restored) task machine."""


class Band(StrEnum):
    IN_BAND = "in_band"
    TOO_EASY = "too_easy"
    TOO_HARD = "too_hard"


class Choice(StrEnum):
    SELECTED = "selected"
    NOT_CLEAN = "not_clean"
    COVERAGE_TIER = "below_coverage_tier"


@dataclass(frozen=True)
class TaskSummary:
    """What selection reads from one exported forge task record."""

    id: str
    capability: str
    arm: str
    clean: bool
    passes: int
    trials: int
    band: Band
    modify_rounds: int
    reverted: bool
    controls: int
    verification: str
    environment: str

    @property
    def solve_rate(self) -> float:
        return self.passes / self.trials

    @property
    def score(self) -> float:
        p = self.solve_rate
        return (
            float(self.band is Band.IN_BAND)
            + 4 * p * (1 - p)
            + 0.25 * (not self.reverted)
            + 0.1 * min(self.controls, CONTROLS_CREDIT_CAP) / CONTROLS_CREDIT_CAP
        )

    @property
    def tiebreak(self) -> str:
        return hashlib.sha256(self.id.encode()).hexdigest()


@dataclass(frozen=True)
class Candidate:
    summary: TaskSummary
    choice: Choice
    rank: int | None


def trial_passed(grade: Mapping | None) -> bool:
    """Forge's pass rule (``forge.stages.rules.passed``) over a serialized grade."""
    if grade is None or grade["status"] == "submission_failure":
        return False
    if grade.get("passed") is not None:
        return grade["passed"]
    reward, score_max = grade.get("reward") or 0.0, grade.get("score_max", 1.0)
    return reward >= score_max - 1e-9


def task_summary(record: Mapping) -> TaskSummary:
    if record["schema"] != RECORD_SCHEMA:
        raise ValueError(f"{record.get('id')}: record schema {record['schema']!r}, expected {RECORD_SCHEMA!r}")
    workpiece = record["workpiece"]
    confirmation = workpiece["evidence"][-1]
    if workpiece["terminal"] != "accepted" or not confirmation["confirmation"]:
        raise ValueError(f"{record['id']}: not an accepted, confirmed workpiece")
    final = workpiece["decisions"][-1]
    if final["kind"] != "accept":
        raise ValueError(f"{record['id']}: last decision is {final['kind']}, not accept")
    header = workpiece["proposal"]["header"]
    submission = workpiece["submission"]
    trials = confirmation["solver"]
    return TaskSummary(
        id=record["id"],
        capability=record["capability"],
        arm=record["arm"],
        clean=not workpiece["notes"] and not workpiece["missing_evidence"],
        passes=sum(trial_passed(trial["grade"]) for trial in trials),
        trials=len(trials),
        band=Band(final["band"]),
        modify_rounds=sum(decision["kind"] == "modify" for decision in workpiece["decisions"]),
        reverted=any(decision.get("revert") for decision in workpiece["decisions"] if decision["kind"] == "accept"),
        controls=len(submission["controls"]),
        verification=submission["verification"] or header["verification"],
        environment=ENVIRONMENTS[restored_task(record)["environment_requirements"].get("command_semantics")],
    )


def select(summaries: Iterable[TaskSummary], target: int) -> list[Candidate]:
    """Coverage-first selection of ``target`` clean tasks; see ``SELECTION_RULE``."""
    by_capability: dict[str, list[TaskSummary]] = defaultdict(list)
    out: list[Candidate] = []
    for summary in summaries:
        if summary.clean:
            by_capability[summary.capability].append(summary)
        else:
            out.append(Candidate(summary, Choice.NOT_CLEAN, None))
    ranked: list[tuple[int, TaskSummary]] = []
    for tasks in by_capability.values():
        tasks.sort(key=lambda t: (-t.score, t.tiebreak))
        ranked.extend(enumerate(tasks, 1))
    clean = len(ranked)
    if clean < target:
        raise ValueError(f"only {clean} clean tasks for a target of {target}")
    ranked.sort(key=lambda item: (item[0], -item[1].score, item[1].tiebreak))
    out += [
        Candidate(summary, Choice.SELECTED if index < target else Choice.COVERAGE_TIER, rank)
        for index, (rank, summary) in enumerate(ranked)
    ]
    return out


def machine_shape(machine: Mapping | None) -> MachineShape:
    return MachineShape() if machine is None else MachineShape.from_mapping(machine)


def restored_machine(requirements: Mapping | None) -> Mapping | None:
    """Undo forge's local-sandbox hosting of one machine.

    The campaign ran with ``FORGE_LOCAL_SANDBOX=1``, under which taskforge's ``host_requirements`` rewrites a machine
    naming ``GRADER_BASE_IMAGE`` into a Linux process with no image and an empty package spec, run in the driver's
    own sandbox. Forge never authors that shape, so it maps back to the image the author named.
    """
    if (
        requirements is not None
        and requirements.get("command_semantics") == CommandSemantics.LINUX_PROCESS
        and requirements.get("docker_image") is None
        and requirements.get("package_spec") == EMPTY_PACKAGES
    ):
        return {**requirements, "docker_image": GRADER_BASE_IMAGE, "package_spec": None}
    return requirements


def machine_identity(requirements: Mapping | None) -> tuple | None:
    if requirements is None:
        return None
    return (requirements.get("command_semantics"), requirements.get("docker_image"), requirements.get("package_spec"))


def restored_task(record: Mapping) -> dict:
    """The confirmed task spec with its machines restored to the environments the author declared."""
    spec = record["rollout_spec"]
    if spec is None:
        raise ValueError(f"{record['id']}: no confirmation rollout spec was recorded")
    task = dict(spec["task"])
    authored = record["workpiece"]["submission"]["task"]
    task["environment_requirements"] = restored_machine(task["environment_requirements"])
    if task["grader"].get("environment") is not None:
        task["grader"] = {**task["grader"], "environment": restored_machine(task["grader"]["environment"])}
    for field, hosted, declared in (
        ("environment_requirements", task["environment_requirements"], authored["environment_requirements"]),
        ("grader.environment", task["grader"].get("environment"), authored["grader"].get("environment")),
    ):
        if machine_identity(hosted) != machine_identity(declared):
            raise ValueError(
                f"{record['id']}: restored {field} {machine_identity(hosted)} != authored {machine_identity(declared)}"
            )
    return task


def lowered_spec(record: Mapping) -> LoweredSpec:
    """The confirmed task and its machine shapes, with each machine's authored environment restored."""
    spec = record["rollout_spec"]
    if spec is None:
        raise ValueError(f"{record['id']}: no confirmation rollout spec was recorded")
    return LoweredSpec(
        TaskSpec.model_validate(restored_task(record)),
        machine_shape(spec["task_machine"]),
        machine_shape(spec["verifier_machine"]),
    )


def read_records(root: str) -> list[dict]:
    base = StoragePath(root)
    rows = (base / ACCEPTED_LIST).read_text().splitlines()
    identities = [row.split("\t")[1] for row in rows if row]
    if not identities:
        raise ValueError(f"{root}/{ACCEPTED_LIST} lists no tasks")

    def read(identity: str) -> dict:
        return json.loads(gzip.decompress((base / "records" / identity / "task.json.gz").read_bytes()))

    with concurrent.futures.ThreadPoolExecutor(READ_THREADS) as pool:
        return list(pool.map(read, identities))


def selection_rows(selection: Sequence[Candidate]) -> list[dict]:
    return [
        {**asdict(item.summary), "score": round(item.summary.score, 6), "choice": item.choice, "rank": item.rank}
        for item in selection
    ]


def write_release(records: Sequence[Mapping], target: int, output: Path, records_root: str) -> dict:
    """Write the cohort for the selected tasks and return its manifest."""
    selection = select((task_summary(record) for record in records), target)
    chosen = {item.summary.id for item in selection if item.choice is Choice.SELECTED}
    lowered = [
        {"path": record["id"], LOWERED_TASK_SPEC_COLUMN: json.dumps(lowered_spec(record).envelope())}
        for record in sorted(records, key=lambda r: r["id"])
        if record["id"] in chosen
    ]
    output.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("path", pa.string()), (LOWERED_TASK_SPEC_COLUMN, pa.string())])
    pq.write_table(pa.Table.from_pylist(lowered, schema=schema), output / LOWERED_FILENAME)
    rows = selection_rows(selection)
    (output / SELECTION_FILENAME).write_text("".join(json.dumps(row) + "\n" for row in rows))
    counts: dict[str, int] = defaultdict(int)
    for item in selection:
        counts[item.choice] += 1
    selected = [item.summary for item in selection if item.choice is Choice.SELECTED]
    manifest = {
        "records": records_root,
        "candidates": len(selection),
        "choices": dict(counts),
        "selected": len(lowered),
        "capabilities": len({s.capability for s in selected}),
        "bands": {band: sum(s.band is band for s in selected) for band in Band},
        "verification": {
            kind: sum(s.verification == kind for s in selected) for kind in {s.verification for s in selected}
        },
        "environment": {
            kind: sum(s.environment == kind for s in selected) for kind in {s.environment for s in selected}
        },
        "rule": SELECTION_RULE,
        "lowered_sha256": hashlib.sha256((output / LOWERED_FILENAME).read_bytes()).hexdigest(),
    }
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", required=True, help="the forge export root holding inputs/ and records/")
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    manifest = write_release(read_records(args.records), args.target, args.output, args.records)
    print(json.dumps({key: value for key, value in manifest.items() if key != "rule"}, indent=2))


if __name__ == "__main__":
    main()
