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
from urllib.parse import urlsplit
from urllib.request import url2pathname

import click
import finelog.types
import finestore.layout
import iris.rpc.job_pb2
import pyarrow.parquet as pq
import rigging.filesystem.storage_path
import rolloutengine.engine
import shellbox.machine
import taskcompendium.models
import verifyit.modes.grade_judge
from cloud.iris.launch_config import load_launch_config
from marinskyrl.model_manifest import MODEL_MANIFEST_FILENAME, ModelManifest
from marinskyrl.task_machine_factories import task_machine_factories
from omegaconf import DictConfig, OmegaConf
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.provenance import launch_provenance
from rolloutengine.engine import RolloutEngine
from rolloutengine.machines import _machine_spec
from rolloutengine.models import TaskRolloutSpec
from shellbox.machine import Backend, MachineFactory
from skyrl_gym.rollout_specs import restore_harness
from skyrl_train.dataset.tasks import TaskDataset
from skyrl_train.tokenizer import tokenizer_from_config
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
from transformers import PreTrainedTokenizerBase
from verifyit.modes.grade_judge import MODEL_ENV, JudgeConnection

EXPECTED_ROWS = 100_000
FEATURE_DISTRIBUTIONS = {
    "marin-finelog": finelog.types,
    "marin-finestore": finestore.layout,
    "marin-iris": iris.rpc.job_pb2,
    "marin-rigging": rigging.filesystem.storage_path,
    "marin-shellbox": shellbox.machine,
    "marin-taskcompendium": taskcompendium.models,
    "marin-rolloutengine": rolloutengine.engine,
    "marin-verifyit": verifyit.modes.grade_judge,
}
INVALID_ANSWER = "snowball_inventory_probe_invalid_answer"


@dataclass(frozen=True)
class SourceProbe:
    source_id: str
    task_id: str
    status: str
    reward: float | None


@dataclass(frozen=True)
class ValidationTask:
    task_id: str
    evaluation_family: str
    contract_id: str
    source_contract_id: str
    dataset_uid: str


@dataclass
class ProbeReport:
    status: str = "running"
    phase: str = "package_identity"
    sha256: str | None = None
    launch_config_sha256: str | None = None
    runtime_tree_hash: str | None = None
    runtime_lock_sha256: str | None = None
    launch_provenance: dict = field(default_factory=dict)
    model_manifest_identity: str | None = None
    tokenizer_template_sha256: str | None = None
    model_repository: str | None = None
    model_revision: str | None = None
    row_count: int = 0
    engine_checks: int = 0
    validation_sha256: str | None = None
    validation_rows: int = 0
    validation_engine_checks: int = 0
    validation_family_counts: dict[str, int] = field(default_factory=dict)
    validation_tasks: list[ValidationTask] = field(default_factory=list)
    training_loader_rows: int | None = None
    contract_source_id: str | None = None
    contract_task_id: str | None = None
    source_counts: dict[str, int] = field(default_factory=dict)
    package_revisions: dict[str, str] = field(default_factory=dict)
    factory_backends: list[str] = field(default_factory=list)
    checks: list[SourceProbe] = field(default_factory=list)
    error_type: str | None = None


def runtime_source_evidence() -> dict:
    """Record declared launch context and the copied workspace's nested dependency lock."""
    provenance = launch_provenance()
    if not provenance.tree_hash:
        raise ValueError("Runtime launch provenance must identify its source tree")
    lock = Path(__file__).resolve().parents[2] / "vendor/skyrl/uv.lock"
    return {
        "runtime_tree_hash": provenance.tree_hash,
        "runtime_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "launch_provenance": asdict(provenance),
    }


def package_revisions() -> dict[str, str]:
    """Identify actual imports, rejecting sources outside their installed distribution."""
    workspace = Path(__file__).resolve().parents[2]
    revisions = {}
    for name, imported_module in FEATURE_DISTRIBUTIONS.items():
        package = distribution(name)
        if imported_module.__file__ is None:
            raise ValueError("Feature package has no imported source file")
        imported_file = Path(imported_module.__file__).resolve(strict=True)
        metadata = package.read_text("direct_url.json")
        source = json.loads(metadata) if metadata is not None else {}
        if source.get("dir_info", {}).get("editable") is True:
            uri = urlsplit(source["url"])
            if (
                uri.scheme != "file"
                or uri.netloc not in {"", "localhost"}
                or not uri.path.startswith("/")
                or uri.query
                or uri.fragment
            ):
                raise ValueError("Editable feature package must name a local file URI")
            installed_root = Path(url2pathname(uri.path)).resolve(strict=True)
            expected_root = (workspace / "lib" / name.removeprefix("marin-")).resolve(strict=True)
            if installed_root != expected_root:
                raise ValueError("Editable feature package is not installed from this Marin workspace")
            imported_file.relative_to(installed_root / "src")
            imported_path = imported_file.relative_to(workspace).as_posix()
            revisions[name] = f"editable:{imported_path};sha256:{hashlib.sha256(imported_file.read_bytes()).hexdigest()}"
            continue
        owned_files = {Path(package.locate_file(path)).resolve() for path in package.files or ()}
        if imported_file not in owned_files:
            raise ValueError("Feature package import is shadowed outside its installed distribution")
        file_hash = hashlib.sha256(imported_file.read_bytes()).hexdigest()
        if metadata is None:
            revisions[name] = f"registry:{package.version};import-sha256:{file_hash}"
            continue
        revision = source.get("vcs_info", {}).get("commit_id")
        if isinstance(revision, str) and len(revision) == 40 and all(c in "0123456789abcdef" for c in revision):
            revisions[name] = f"git:{revision};import-sha256:{file_hash}"
            continue
        archive_hash = source.get("archive_info", {}).get("hashes", {}).get("sha256")
        if isinstance(archive_hash, str) and len(archive_hash) == 64:
            revisions[name] = f"archive-sha256:{archive_hash};import-sha256:{file_hash}"
            continue
        raise ValueError("Feature package is not installed from the frozen registry or immutable source")
    return revisions


def validated_probe_config(path: Path, report: ProbeReport) -> DictConfig:
    """Record exact launch bytes and the current workspace's declared provenance."""
    document = path.read_bytes()
    report.launch_config_sha256 = hashlib.sha256(document).hexdigest()
    with TemporaryDirectory(prefix="snowball-probe-config-") as directory:
        snapshot = Path(directory) / "launch.yaml"
        snapshot.write_bytes(document)
        config = load_launch_config(snapshot)
    evidence = runtime_source_evidence()
    report.runtime_tree_hash = evidence["runtime_tree_hash"]
    report.runtime_lock_sha256 = evidence["runtime_lock_sha256"]
    report.launch_provenance = evidence["launch_provenance"]
    return config


def verify_initializer_model(
    model_uri: str, repository: str, revision: str, expected_identity: str | None = None
) -> ModelManifest:
    """Reject a named initializer whose immutable model manifest names another source.

    This checks manifest provenance, without rehashing model weights or proving
    serving behavior. Generic smoke and continuation runs have no named-source gate.
    """
    uri = prefix_join(model_uri, MODEL_MANIFEST_FILENAME)
    with StoragePath(uri).open("rt") as source:
        manifest = ModelManifest.from_mapping(json.load(source), uri)
    if (manifest.model_id, manifest.revision) != (repository, revision):
        raise ValueError("Model manifest source differs from the selected initializer")
    if expected_identity is not None and manifest.identity != expected_identity:
        raise ValueError("Model manifest content differs from the pinned initializer")
    return manifest


def verify_tokenizer_template(tokenizer: PreTrainedTokenizerBase, expected_sha256: str | None) -> str:
    """Identify the active serving template and reject drift from the frozen baseline."""
    actual = hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("Active tokenizer template differs from the pinned initializer")
    return actual


def preflight_tokenizer(config: DictConfig) -> PreTrainedTokenizerBase:
    """Use pinned tokenizer metadata and the serving template before GPU staging."""
    tokenizer_config = OmegaConf.merge(
        config.skyrl,
        {
            "trainer": {
                "policy": {
                    "model": {
                        "tokenizer_path": str(config.inputs.model.tokenizer_uri),
                        "tokenizer_revision": str(config.inputs.model.tokenizer_revision),
                    }
                }
            }
        },
    )
    return tokenizer_from_config(tokenizer_config)


def inventory_representatives(
    path: Path,
    report: ProbeReport,
    factories: Mapping[str, MachineFactory],
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
                harness = restore_harness(spec.harness)
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
    launch_config: Path,
    expected_sha256: str,
    report: ProbeReport,
    expected_validation_sha256: str | None,
    validation_family_counts: dict[str, int] | None = None,
    model_repository: str | None = None,
    model_revision: str | None = None,
    expected_model_manifest_identity: str | None = None,
    expected_tokenizer_template_sha256: str | None = None,
) -> None:
    report.phase = "launch_configuration"
    config = validated_probe_config(launch_config, report)
    if (model_repository is None) != (model_revision is None):
        raise ValueError("Named initializer admission requires both repository and revision")
    if expected_model_manifest_identity is not None and model_repository is None:
        raise ValueError("Pinned initializer content requires its named repository and revision")
    if model_repository is not None:
        assert model_revision is not None
        report.phase = "initializer_identity"
        manifest = verify_initializer_model(
            str(config.inputs.model.uri), model_repository, model_revision, expected_model_manifest_identity
        )
        report.model_manifest_identity = manifest.identity
        report.model_repository = manifest.model_id
        report.model_revision = manifest.revision
    report.phase = "package_identity"
    report.package_revisions = package_revisions()
    task_sessions = config.skyrl.environment.task_sessions
    if task_sessions.get("local") is None:
        raise ValueError("The composed training configuration did not enable the local machine backend")
    if len(config.inputs.train_data) != 1:
        raise ValueError("The inventory smoke requires one complete training pool")
    source = config.inputs.train_data[0]
    pool_uri = prefix_join(source.uri, source.relative_path)
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
        factories = task_machine_factories(task_sessions)
        report.factory_backends = sorted(factories)
        factory = factories[Backend.LOCAL]
        report.phase = "engine_contract_validation"
        remote_judge = task_sessions["remote_judge"]
        judge_connection = JudgeConnection(remote_judge["base_url"], os.environ[remote_judge["api_key_env"]])
        # Verifyit selects the model at the process boundary; its per-call connection
        # carries only credentials. Keep the composed model without changing task graders.
        os.environ[MODEL_ENV] = config.runtime.task_env[MODEL_ENV]
        scripts, native = inventory_representatives(path, report, factories, judge_connection)
        validation_canaries = {}
        tokenizer = preflight_tokenizer(config)
        if expected_tokenizer_template_sha256 is not None:
            report.phase = "tokenizer_identity"
            report.tokenizer_template_sha256 = verify_tokenizer_template(tokenizer, expected_tokenizer_template_sha256)
        training_dataset = TaskDataset(
            [str(path)],
            tokenizer,
            int(config.skyrl.trainer.max_prompt_length),
            num_workers=1,
            chat_template_kwargs=dict(config.skyrl.generator.chat_template_kwargs),
        )
        report.training_loader_rows = len(training_dataset)
        if report.training_loader_rows != EXPECTED_ROWS:
            raise ValueError("The selected initializer's training loader dropped inventory rows")
        if config.inputs.validation_data:
            report.phase = "validation_contracts"
            if len(config.inputs.validation_data) != 1 or expected_validation_sha256 is None:
                raise ValueError("The validation panel requires one pinned source and its expected hash")
            validation = config.inputs.validation_data[0]
            validation_uri = prefix_join(validation.uri, validation.relative_path)
            validation_path = Path(directory) / "validation.parquet"
            with StoragePath(validation_uri).open("rb") as source, validation_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            with validation_path.open("rb") as source:
                report.validation_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
            if report.validation_sha256 != expected_validation_sha256:
                raise ValueError("The validation panel differs from its expected SHA-256")
            engines = {}
            counts = Counter()
            tasks = []
            columns = ["task_rollout_spec"]
            if validation_family_counts is not None:
                columns.extend(["task_id", "evaluation_family", "contract_id", "source_contract_id"])
            for batch in pq.ParquetFile(validation_path).iter_batches(columns=columns, batch_size=256):
                for row in batch.to_pylist():
                    spec = TaskRolloutSpec.model_validate_json(row["task_rollout_spec"])
                    if spec.harness not in engines:
                        harness = restore_harness(spec.harness)
                        engines[spec.harness] = RolloutEngine(factories, {spec.harness: harness})
                    engines[spec.harness].check(spec.model_copy(update={"judge_connection": judge_connection}))
                    report.validation_rows += 1
                    report.validation_engine_checks += 1
                    if validation_family_counts is not None:
                        if row["task_id"] != spec.task.id:
                            raise ValueError("Validation pairing ID differs from its runtime task")
                        family = row["evaluation_family"]
                        counts[family] += 1
                        tasks.append(row)
                        if isinstance(spec.task.grader, ScriptGrader):
                            if spec.verifier_machine is None or spec.verifier_machine.backend != Backend.DAYTONA:
                                raise ValueError("Original panel graders require their source Daytona backend")
                            validation_canaries.setdefault(family, spec)
            if validation_family_counts is None:
                raise ValueError("Validation requires explicit fixed-panel family counts")
            if validation_family_counts is not None:
                assert tokenizer is not None
                if counts != validation_family_counts or len({row["task_id"] for row in tasks}) != len(tasks):
                    raise ValueError("Full-panel family counts or unique pairing IDs differ")
                if set(validation_canaries) != set(validation_family_counts) - {"structural"}:
                    raise ValueError("Every original script-grading family requires a validation canary")
                validation_dataset = TaskDataset(
                    [str(validation_path)],
                    tokenizer,
                    int(config.skyrl.trainer.max_prompt_length),
                    num_workers=1,
                    chat_template_kwargs=dict(config.skyrl.generator.chat_template_kwargs),
                )
                if len(validation_dataset) != len(tasks):
                    raise ValueError("The selected initializer's validation loader dropped panel rows")
                admitted = validation_dataset.collate_fn(list(validation_dataset))
                for expected, actual in zip(tasks, admitted, strict=True):
                    if expected["task_id"] != actual["input"].spec.task.id:
                        raise ValueError("Validation loader changed original panel ordering")
                    report.validation_tasks.append(
                        ValidationTask(
                            expected["task_id"],
                            expected["evaluation_family"],
                            expected["contract_id"],
                            expected["source_contract_id"],
                            actual["uid"],
                        )
                    )
                report.validation_family_counts = dict(counts)
        elif expected_validation_sha256 is not None:
            raise ValueError("A validation hash was supplied without a configured validation panel")
    report.phase = "script_grading"
    for source, spec in sorted(scripts.items()):
        assert isinstance(spec.task.grader, ScriptGrader) and spec.verifier_machine is not None
        grade = await grade_empty_in_sandbox(
            spec.task,
            factory,
            _machine_spec(spec.task.grader.environment, spec.verifier_machine),
        )
        report.checks.append(SourceProbe(source, spec.task.id, grade.status, grade.reward))
        if grade.status != Outcome.GRADED or grade.reward != 0:
            raise ValueError("A representative script grader did not reject an empty submission with a usable zero")
    report.phase = "native_grading"
    assert isinstance(native.grader, VerifyitGrader)
    prefix = native.grader.parameters["prefix"]
    expected = native.grader.parameters["expected"]
    if not isinstance(prefix, str) or not isinstance(expected, str) or not expected.startswith(prefix):
        raise ValueError("Native reference is not a continuation of its supplied prefix")
    for answer, expected_reward in ((expected[len(prefix) :], 1.0), (INVALID_ANSWER, 0.0)):
        attempt = GradingAttempt(
            ConversationTrace(events=(*native.context.events, TextMessage(role="assistant", content=answer)))
        )
        grade = grade_answer(native, attempt)
        report.checks.append(SourceProbe(native.source.dataset, native.id, grade.status, grade.reward))
        if grade.status != Outcome.GRADED or grade.reward != expected_reward:
            raise ValueError("The current native grader did not distinguish valid and invalid continuations")
    report.phase = "validation_grading"
    for family, spec in sorted(validation_canaries.items()):
        assert isinstance(spec.task.grader, ScriptGrader) and spec.verifier_machine is not None
        grade = await grade_empty_in_sandbox(
            spec.task,
            factories[spec.verifier_machine.backend],
            _machine_spec(spec.task.grader.environment, spec.verifier_machine),
            timeout=spec.limits.verifier_timeout,
        )
        report.checks.append(SourceProbe(family, spec.task.id, grade.status, grade.reward))
        if grade.status != Outcome.GRADED or grade.reward != 0:
            raise ValueError("An original validation grader did not reject the empty submission with a usable zero")
    report.status = "passed"
    report.phase = "complete"


def write_report(output_uri: str, report: ProbeReport) -> None:
    with StoragePath(output_uri).open("wt") as target:
        json.dump(asdict(report), target, indent=2, allow_nan=False)
        target.write("\n")


@click.command(help=__doc__)
@click.option("--launch-config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--expected-sha256", required=True)
@click.option("--expected-validation-sha256")
@click.option("--validation-family-counts", help="Required explicit JSON family counts for a full development panel.")
@click.option("--output-uri", required=True, help="Destination JSON report URI.")
@click.option("--model-repository")
@click.option("--model-revision")
@click.option("--expected-model-manifest-identity")
@click.option("--expected-tokenizer-template-sha256")
def main(
    launch_config: Path,
    expected_sha256: str,
    expected_validation_sha256: str | None,
    validation_family_counts: str | None,
    output_uri: str,
    model_repository: str | None,
    model_revision: str | None,
    expected_model_manifest_identity: str | None,
    expected_tokenizer_template_sha256: str | None,
) -> None:
    report = ProbeReport()
    try:
        counts = json.loads(validation_family_counts) if validation_family_counts is not None else None
        asyncio.run(
            probe(
                launch_config,
                expected_sha256,
                report,
                expected_validation_sha256,
                counts,
                model_repository,
                model_revision,
                expected_model_manifest_identity,
                expected_tokenizer_template_sha256,
            )
        )
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
