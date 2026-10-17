# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assess original Harbor archives with the canonical TaskCompendium quality reviewer."""

import hashlib
import io
import json
import os
import tarfile
from contextlib import ExitStack
from functools import partial
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq
from finestore.schema import arrow_schema
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.inference.openai_chat import OpenAIChatClient
from taskcompendium.pipeline.assessment import (
    PACKET_SCHEMA,
    AssessmentInputs,
    AssessmentSampling,
    SamplePopulation,
    SelectedTask,
)
from taskcompendium.pipeline.assessment_review import ASSESSMENT_RUBRIC, AssessmentModel, grade_assessment
from taskcompendium.pipeline.chat_requests import chat_output
from zephyr.context import ZephyrContext

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_REQUEST_TIMEOUT, resolve_glm_base_url

PREVIEW_CHARACTERS = 48_000
FILE_PREVIEW_CHARACTERS = 16_000
HARBOR_RUBRIC = """
The evidence is an original Harbor task, not a TaskSpec conversion. Resource roles are explicit:
instruction.md, environment files and setup_files are public; tests are private verifier files;
solution files are private oracle evidence. Harbor uploads setup_files to /setup_files before
the agent runs, and tests to /tests for verification. The agent performs the task inside the
declared environment; repository checkout or dependency installation in public instructions is
part of the task. Inspect the original grader and original instructions. A passing oracle shows
one reference execution succeeded, not that the task is fair, leak-free or impossible to exploit.
The cohort label admitted means admitted to review, not admitted to training. Check execution.status:
when oracle validation was skipped, there is no evidence of a successful reference execution.
Do not infer oracle success from the cohort label or treat a skipped check as a failed execution.
"""


def archive_evidence(payload: bytes, staged: Path, budget: int) -> tuple[list[dict], list[str], int]:
    """Check executed file bytes and prepare visibility-labelled bounded previews."""
    records = []
    omissions = []
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
        members = sorted(archive.getmembers(), key=lambda m: (not m.name.endswith("instruction.md"), m.name))
        for member in members:
            if member.isdir():
                continue
            name = member.name.removeprefix("./")
            if not member.isfile() or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"Unsupported archive member: {name}")
            stream = archive.extractfile(member)
            assert stream is not None
            data = stream.read()
            if (staged / name).read_bytes() != data:
                raise ValueError(f"Executed task differs from source: {staged / name}")
            role = "oracle" if name.startswith("solution/") else "verifier" if name.startswith("tests/") else "public"
            record = {"path": name, "role": role, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                omissions.append(f"Binary file omitted: {name} ({len(data)} bytes)")
            else:
                allowance = min(FILE_PREVIEW_CHARACTERS, budget)
                record.update(text=text[:allowance], characters=len(text), truncated=len(text) > allowance)
                budget -= min(allowance, len(text))
                if len(text) > allowance:
                    omissions.append(f"Text truncated: {name}; retained {allowance}/{len(text)} characters")
            records.append(record)
    return records, omissions, budget


def prepare(source: Path, results: tuple[Path, ...], output: Path, task_paths: tuple[str, ...]) -> None:
    """Freeze review packets only for successful, byte-matched Daytona oracle trials."""
    observed = {}
    attempts = {}
    for directory in results:
        for path in directory.glob("*/result.json"):
            result = json.loads(path.read_text())
            name = Path(result["config"]["task"]["path"]).name
            if name in observed:
                previous = observed[name][1]
                if previous["exception_info"] is None:
                    raise ValueError(f"Cannot replace a scored oracle result: {name}")
                if previous["config"]["task"]["path"] != result["config"]["task"]["path"]:
                    raise ValueError(f"Oracle retry changed the staged task path: {name}")
            attempts.setdefault(name, []).append(hashlib.sha256(path.read_bytes()).hexdigest())
            observed[name] = (path, result)
    packets = []
    decisions = []
    selected = set(task_paths)
    for batch in pq.ParquetFile(source).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            name = row["path"]
            if selected and name not in selected:
                continue
            if name not in observed:
                raise ValueError(f"Oracle has not finished: {name}")
            result_path, result = observed[name]
            config = result["config"]
            verifier = result["verifier_result"]
            passed = (
                result["exception_info"] is None
                and verifier is not None
                and verifier["rewards"] == {"reward": 1.0}
                and config["agent"]["name"] == "oracle"
                and config["environment"]["type"] == "daytona"
            )
            decisions.append(
                {
                    "task_path": name,
                    "oracle_passed": passed,
                    "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                    "attempts_sha256": attempts[name],
                }
            )
            if not passed:
                continue
            staged = Path(config["task"]["path"])
            files, omissions, budget = archive_evidence(row["task_binary"], staged, PREVIEW_CHARACTERS)
            oracle, oracle_omissions, _ = archive_evidence(row["solution_binary"], staged, budget)
            identity = f"{row['source']}/{name}"
            evidence = {
                "files": files,
                "oracle_files": oracle,
                "execution": {
                    "status": "oracle_passed",
                    "backend": "daytona",
                    "agent": "oracle",
                    "verifier_result": verifier,
                    "snapshot": config["environment"]["kwargs"]["snapshot_template_name"],
                    "result_sha256": decisions[-1]["result_sha256"],
                },
                "source": {
                    "task_archive_sha256": hashlib.sha256(row["task_binary"]).hexdigest(),
                    "solution_archive_sha256": hashlib.sha256(row["solution_binary"]).hexdigest(),
                },
            }
            packets.append(
                {
                    "task_id": identity,
                    "cohort": "admitted",
                    "packet_json": json.dumps(evidence),
                    "omissions": omissions + oracle_omissions,
                    "exclusion_reason": None,
                }
            )
    if selected != {d["task_path"] for d in decisions} and selected:
        raise ValueError("Requested task paths were not all found")
    if not packets:
        raise ValueError("No passing oracle packets")
    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pylist(packets, schema=PACKET_SCHEMA), output / "packets.parquet")
    manifest = {
        "source": source.name,
        "mode": "original-harbor-assessment",
        "status": "oracle-passing candidates; quality not yet certified",
        "processed_rows": len(decisions),
        "oracle_passing_rows": len(packets),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "decisions": decisions,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def prepare_without_oracle(source: Path, staged: Path, output: Path, reason: str) -> None:
    """Prepare byte-matched quality evidence with an explicit oracle-validation exemption."""
    packets = []
    identities = set()
    for batch in pq.ParquetFile(source).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            name = Path(row["path"])
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe task path: {name}")
            identity = f"{row['source']}/{row['path']}"
            if identity in identities:
                raise ValueError(f"Duplicate task identity: {identity}")
            identities.add(identity)
            files, omissions, budget = archive_evidence(row["task_binary"], staged / name, PREVIEW_CHARACTERS)
            oracle, oracle_omissions, _ = archive_evidence(row["solution_binary"], staged / name, budget)
            evidence = {
                "files": files,
                "oracle_files": oracle,
                "execution": {"status": "oracle_validation_skipped", "reason": reason},
                "source": {
                    "task_archive_sha256": hashlib.sha256(row["task_binary"]).hexdigest(),
                    "solution_archive_sha256": hashlib.sha256(row["solution_binary"]).hexdigest(),
                },
            }
            packets.append(
                {
                    "task_id": identity,
                    "cohort": "admitted",
                    "packet_json": json.dumps(evidence),
                    "omissions": omissions + oracle_omissions,
                    "exclusion_reason": None,
                }
            )
    if not packets:
        raise ValueError("No quality-review candidates")
    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pylist(packets, schema=PACKET_SCHEMA), output / "packets.parquet")
    manifest = {
        "source": source.name,
        "mode": "original-harbor-assessment",
        "status": "oracle validation skipped; quality not yet certified",
        "oracle_validation": "skipped",
        "oracle_validation_reason": reason,
        "processed_rows": len(packets),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


@click.group(help=__doc__)
def main() -> None:
    pass


@main.command("prepare")
@click.option("--source", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--results", type=click.Path(path_type=Path, exists=True), multiple=True, required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--task-path", "task_paths", multiple=True)
def prepare_command(source: Path, results: tuple[Path, ...], output: Path, task_paths: tuple[str, ...]) -> None:
    prepare(source, results, output, task_paths)


@main.command("prepare-without-oracle")
@click.option("--source", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--staged", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--reason", required=True)
def prepare_without_oracle_command(source: Path, staged: Path, output: Path, reason: str) -> None:
    prepare_without_oracle(source, staged, output, reason)


@main.command("review")
@click.option("--inputs", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--relay-job", required=True)
@click.option("--model-revision", required=True)
@click.option("--concurrency", type=int, required=True)
def review(inputs: Path, output: Path, relay_job: str, model_revision: str, concurrency: int) -> None:
    packets = pq.read_table(inputs / "packets.parquet").to_pylist()
    metadata = AssessmentInputs(
        source=json.loads((inputs / "manifest.json").read_text())["source"],
        scope="Every candidate in the supplied packet file; execution evidence is stated separately per task",
        sampling=AssessmentSampling(admitted=len(packets), excluded=0),
        populations=[SamplePopulation(cohort="admitted", population=len(packets), sample_count=len(packets))],
        sample=[SelectedTask(task_id=row["task_id"], cohort="admitted") for row in packets],
        packet_paths=[str(inputs / "packets.parquet")],
    )
    pq.write_table(
        pa.Table.from_pylist([metadata.model_dump(mode="json")], schema=arrow_schema(AssessmentInputs)),
        inputs / "inputs.parquet",
    )
    client = OpenAIChatClient(
        resolve_glm_base_url(relay_job), os.environ[GLM_BULK_TOKEN_ENV], timeout=GLM_REQUEST_TIMEOUT
    )
    with ExitStack() as scope:
        local_client = LocalClient()
        scope.callback(local_client.shutdown)
        scope.enter_context(set_current_client(local_client))
        context = scope.enter_context(
            ZephyrContext(client=local_client, max_workers=1, chunk_storage_prefix=str(output / "chunks"))
        )
        grade_assessment(
            context,
            curated_root=str(inputs),
            inputs_root=str(inputs),
            rubric=ASSESSMENT_RUBRIC + HARBOR_RUBRIC,
            model=AssessmentModel("glm-5.3", model_revision),
            submit=partial(chat_output, client, max_concurrent=concurrency),
            cache_root=str(output / "review-cache"),
            output_root=str(output),
        )


if __name__ == "__main__":
    main()
