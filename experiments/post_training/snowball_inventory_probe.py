# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the complete inventory and exercise source graders without policy inference."""

import asyncio
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from importlib.metadata import distribution
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.core import Harness
from rolloutengine.harness.token_prefix import TokenPrefixHarness
from rolloutengine.machines import _machine_spec
from rolloutengine.models import TaskRolloutSpec
from shellbox.machine import Backend, MachineFactory
from skyrl_gym.rollout_specs import is_shell_harness, shell_harness
from taskcompendium.grading import grade_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    ConversationTrace,
    GradingAttempt,
    ScriptGrader,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
)
from taskcompendium.runtime.grading import grade_empty_in_sandbox
from verifyit.modes.grade_judge import JudgeConnection

if sys.platform == "linux":
    from shellbox.backends.local.machine import LocalMachineFactory

EXPECTED_ROWS = 100_000
CORE_REVISION = "6e2fe1839ea067be147f1ce06aced63a96670f7a"
FEATURE_DISTRIBUTIONS = ("marin-taskcompendium", "marin-rolloutengine", "marin-verifyit")
INVALID_ANSWER = "snowball_inventory_probe_invalid_answer"


@dataclass(frozen=True)
class SourceProbe:
    source_id: str
    task_id: str
    status: str
    reward: float | None


@dataclass
class ProbeReport:
    status: str = "running"
    phase: str = "package_identity"
    sha256: str | None = None
    row_count: int = 0
    engine_checks: int = 0
    contract_source_id: str | None = None
    contract_task_id: str | None = None
    source_counts: dict[str, int] = field(default_factory=dict)
    package_revisions: dict[str, str] = field(default_factory=dict)
    checks: list[SourceProbe] = field(default_factory=list)
    error_type: str | None = None


def package_revisions() -> dict[str, str]:
    revisions = {}
    for name in FEATURE_DISTRIBUTIONS:
        metadata = distribution(name).read_text("direct_url.json")
        if metadata is None:
            raise ValueError("Feature package is missing immutable Git source identity")
        revision = json.loads(metadata).get("vcs_info", {}).get("commit_id")
        if revision != CORE_REVISION:
            raise ValueError("Feature package differs from the selected core source revision")
        revisions[name] = revision
    return revisions


def inventory_representatives(
    path: Path,
    report: ProbeReport,
    factories: Mapping[str, MachineFactory],
    token_prefix: TokenPrefixHarness,
    judge_connection: JudgeConnection,
) -> tuple[dict[str, TaskRolloutSpec], TaskSpec | None]:
    scripts = {}
    native = None
    counts = Counter()
    engines: dict[str, RolloutEngine] = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["source_id", "task_rollout_spec"], batch_size=256):
        for row in batch.to_pylist():
            spec = TaskRolloutSpec.model_validate_json(row["task_rollout_spec"])
            source = row["source_id"]
            report.contract_source_id = source
            report.contract_task_id = spec.task.id
            if spec.harness not in engines:
                harness: Harness
                if is_shell_harness(spec.harness):
                    harness = shell_harness(spec.harness)
                elif spec.harness == "token_prefix":
                    harness = token_prefix
                else:
                    raise ValueError("The inventory has an unconfigured harness")
                engines[spec.harness] = RolloutEngine(factories, {spec.harness: harness})
            engines[spec.harness].check(spec.model_copy(update={"judge_connection": judge_connection}))
            report.engine_checks += 1
            counts[source] += 1
            report.row_count += 1
            if isinstance(spec.task.grader, ScriptGrader):
                if spec.verifier_machine is None or spec.verifier_machine.backend != Backend.LOCAL:
                    raise ValueError("Script grading probe requires an explicit local verifier machine")
                scripts.setdefault(source, spec)
            elif (
                isinstance(spec.task.grader, VerifyitGrader)
                and spec.task.grader.environment is None
                and spec.task.grader.mode == "structure_closure"
            ):
                native = spec.task if native is None else native
    report.source_counts = dict(sorted(counts.items()))
    if report.row_count != EXPECTED_ROWS or report.engine_checks != EXPECTED_ROWS:
        raise ValueError("The supplied pool did not pass all 100000 engine contract checks")
    if not scripts or native is None:
        raise ValueError("The supplied inventory lacks script or native closure tasks")
    report.contract_source_id = None
    report.contract_task_id = None
    return scripts, native


async def probe(
    pool_uri: str,
    expected_sha256: str,
    report: ProbeReport,
    task_sessions: dict,
    judge_model: str,
) -> None:
    report.package_revisions = package_revisions()
    report.phase = "inventory_validation"
    with TemporaryDirectory(prefix="snowball-inventory-probe-") as directory:
        path = Path(directory) / "train.parquet"
        with StoragePath(pool_uri).open("rb") as source, path.open("wb") as target:
            shutil.copyfileobj(source, target)
        with path.open("rb") as source:
            report.sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        if report.sha256 != expected_sha256.lower():
            raise ValueError("The pool differs from its expected SHA-256")
        report.phase = "local_sandbox_capability"
        if sys.platform != "linux":
            raise RuntimeError("Local sandbox capability requires a Linux host")
        if task_sessions.get("local") is None:
            raise ValueError("The inventory requires the configured local machine backend")
        factory = LocalMachineFactory()
        report.phase = "engine_contract_validation"
        token_prefix = TokenPrefixHarness(**task_sessions["token_prefix"])
        remote_judge = task_sessions["remote_judge"]
        judge_connection = JudgeConnection(remote_judge["base_url"], os.environ[remote_judge["api_key_env"]])
        os.environ["VERIFYIT_JUDGE_MODEL"] = judge_model
        scripts, native = inventory_representatives(
            path, report, {Backend.LOCAL: factory}, token_prefix, judge_connection
        )
    report.phase = "script_grading"
    for source, spec in sorted(scripts.items()):
        assert isinstance(spec.task.grader, ScriptGrader) and spec.verifier_machine is not None
        grade = await grade_empty_in_sandbox(
            spec.task,
            factory,
            _machine_spec(spec.task.grader.environment, spec.verifier_machine),
        )
        report.checks.append(SourceProbe(source, spec.task.id, grade.status, grade.reward))
        if grade.status != Outcome.GRADED:
            raise ValueError("A representative script grader did not produce a grade")
    report.phase = "native_grading"
    attempt = GradingAttempt(
        ConversationTrace(events=(*native.context.events, TextMessage(role="assistant", content=INVALID_ANSWER)))
    )
    grade = grade_answer(native, attempt)
    report.checks.append(SourceProbe(native.source.dataset, native.id, grade.status, grade.reward))
    if grade.status != Outcome.GRADED:
        raise ValueError("The native closure grader did not produce a grade")
    report.status = "passed"
    report.phase = "complete"


def write_report(output_uri: str, report: ProbeReport) -> None:
    with StoragePath(output_uri).open("wt") as target:
        json.dump(asdict(report), target, indent=2, allow_nan=False)
        target.write("\n")


@click.command(help=__doc__)
@click.option("--pool-uri", required=True, help="Exact URI of the complete compiled train.parquet.")
@click.option("--expected-sha256", required=True)
@click.option("--output-uri", required=True, help="Destination JSON report URI.")
@click.option("--task-sessions-json", required=True, help="Exact configured training task sessions without credentials.")
@click.option("--judge-model", required=True)
def main(pool_uri: str, expected_sha256: str, output_uri: str, task_sessions_json: str, judge_model: str) -> None:
    report = ProbeReport()
    try:
        asyncio.run(probe(pool_uri, expected_sha256, report, json.loads(task_sessions_json), judge_model))
    except Exception as error:
        report.status = "failed"
        report.error_type = type(error).__name__
        write_report(output_uri, report)
        raise click.ClickException(f"{report.phase} failed ({report.error_type})") from None
    write_report(output_uri, report)
    click.echo(
        json.dumps(
            {
                "status": report.status,
                "row_count": report.row_count,
                "checks": len(report.checks),
                "engine_checks": report.engine_checks,
            }
        )
    )


if __name__ == "__main__":
    main()
