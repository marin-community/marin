# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenSWE keep-only rows as native Harbor packages behind the exit-code grader.

Per task the converter emits the native package the landed chain serves
(``native_task`` → base-image pinning → ``convert_cohort_task``):

- ``instruction.md`` — the row's ``problem_statement`` verbatim.
- ``environment/Dockerfile`` — the row recipe with the translated bytes (the
  condensed base binding replaces the unpublished upstream base; the repository
  copy becomes a build-time clone+checkout of ``base_commit``).
- ``tests/eval_script.sh`` — the upstream eval_script byte-verbatim: the
  selection and grading authority (ruling 2026-11-26 6:03 AM, option a).
- ``tests/test.sh`` — the profile's fail-closed verifier around it.
- ``solution/gold.patch`` + ``solution/solve.sh`` — the oracle.

Nothing is synthesized: the rows carry no test IDs (empty ``FAIL_TO_PASS``/
``PASS_TO_PASS`` upstream) and the grader reads the recorded exit code, never a
test list. A row whose bytes the profile cannot express faithfully is a typed
exclusion with evidence.
"""

import base64
import hashlib
from collections.abc import Callable

import tomlkit
from taskcompendium.identity import task_spec_sha256
from taskcompendium.models import Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, RawRow

from experiments.post_training.task_curation.datasets.native_harbor import native_task
from experiments.post_training.task_curation.datasets.openswe import grader
from experiments.post_training.task_curation.datasets.openswe import grader_repair
from experiments.post_training.task_curation.datasets.openswe.dockerfile_translation import (
    DockerfileRejected,
    translate_dockerfile,
)
from experiments.post_training.task_curation.datasets.openswe.sources import (
    KEEP_RELEASE_REPO,
    KEEP_RELEASE_REVISION,
    UPSTREAM_DATASET,
    UPSTREAM_REVISION,
    OpenSweRow,
)
from experiments.post_training.task_curation.native_source_conversion import NativeRowConversion

# The profile's declared execution profile: the Daytona per-sandbox cap the lane's
# SWE waves run at (ruling 49; the fidelity gate qualifies every admitted row at
# exactly this profile). OpenSWE rows declare no execution limits upstream, so
# there is no deviation to record — these are the profile's own declarations.
CPUS = 4
MEMORY_MB = 8192
STORAGE_MB = 10240
BUILD_TIMEOUT_SEC = 3600.0
VERIFIER_TIMEOUT_SEC = 2400.0
AGENT_TIMEOUT_SEC = 1800.0


class OpenSweRowExcluded(Exception):
    """A typed per-row exclusion with evidence; the row leaves the converted population."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def _task_toml(
    row: OpenSweRow,
    workdir: str,
    base_group: str | None,
    declared_bases: tuple[str, ...],
    grader_repair_note: dict | None = None,
) -> str:
    metadata: dict = {
        "author_name": "OpenSWE (Marin conversion)",
        "author_email": "dataset@GAIR-NLP",
        "category": "debugging",
        "instance_id": row.instance_id,
        "repo": row.repo,
        "base_commit": row.base_commit,
        "upstream_image": row.image_name,
        "upstream_dataset": UPSTREAM_DATASET,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_bases": list(declared_bases),
    }
    if base_group is not None:
        metadata["condensed_base_python"] = base_group
    if grader_repair_note is not None:
        metadata["grader_repair"] = grader_repair_note
    config = {
        "metadata": metadata,
        "verifier": {"timeout_sec": VERIFIER_TIMEOUT_SEC},
        "agent": {"timeout_sec": AGENT_TIMEOUT_SEC},
        "environment": {
            "cpus": CPUS,
            "memory_mb": MEMORY_MB,
            "storage_mb": STORAGE_MB,
            "build_timeout_sec": BUILD_TIMEOUT_SEC,
            "allow_internet": True,
            "workdir": workdir,
        },
    }
    return tomlkit.dumps(config)


def package_files(
    row: OpenSweRow, condensed_bases: dict[str, str], resolve: Callable[[str], str] | None = None
) -> dict[str, bytes]:
    """Render one keep-only row as native package files.

    Raises ``OpenSweRowExcluded`` with typed evidence when the row cannot convert
    faithfully; every emitted file otherwise carries the upstream bytes.
    ``resolve`` digest-pins any non-condensed registry base through the shared
    resolver (the conversion entry stages it; tests may fake it).
    """
    if not row.dockerfile.strip():
        raise OpenSweRowExcluded("empty_dockerfile", "The row carries no Dockerfile")
    if not row.eval_script.strip():
        raise OpenSweRowExcluded("empty_eval_script", "The row carries no eval_script")
    if not (row.patch or "").strip():
        raise OpenSweRowExcluded("missing_gold_patch", "The row does not provide its original solution patch")
    if "OPENSWE_EXIT_CODE" not in row.eval_script:
        raise OpenSweRowExcluded(
            "ungradeable_eval_script",
            "The upstream eval_script never records an OPENSWE_EXIT_CODE verdict",
        )
    # The no-op grader scan (coordinator ruling 2026-11-27): a grader that
    # applies the golden patch before testing passes an empty-diff agent — a
    # broken grader fails closed. Repairable graders drop the golden-patch
    # application so the agent's diff is what gets tested; unrepairable ones
    # leave the converted population as typed exclusions.
    grader_classification = grader_repair.classify_grader(row)
    if grader_classification.verdict == "unrepairable":
        raise OpenSweRowExcluded(grader_classification.reason or "grader passes no-op", grader_classification.detail)
    eval_script = row.eval_script
    grader_repair_note = None
    if grader_classification.verdict == "repairable":
        eval_script = grader_repair.repaired_script(row, grader_classification)
        grader_repair_note = {
            "applied": True,
            "golden_sites": len(grader_classification.golden_sites),
            "golden_matches": sorted({site.match for site in grader_classification.golden_sites}),
            "resets_scoped": len(grader_classification.reset_sites),
        }
    try:
        translation = translate_dockerfile(
            row.dockerfile,
            repo=row.repo,
            base_commit=row.base_commit,
            condensed_bases=condensed_bases,
            resolve_base=resolve,
        )
    except DockerfileRejected as error:
        raise OpenSweRowExcluded("dockerfile_rejected", str(error)) from error
    task_metadata_extra = None if grader_repair_note is None else grader_repair_note
    files = {
        "task.toml": _task_toml(
            row, translation.workdir, translation.base_group, translation.declared_bases, task_metadata_extra
        ).encode(),
        "instruction.md": row.problem_statement.encode(),
        "environment/Dockerfile": translation.text.encode(),
        "tests/test.sh": grader.verifier_script(),
        "tests/eval_script.sh": eval_script.encode(),
        "solution/gold.patch": row.patch.encode(),
        "solution/solve.sh": grader.solve_script(translation.workdir),
    }
    return files


def row_to_raw(row: OpenSweRow, files: dict[str, bytes]) -> RawRow:
    """Wrap the rendered package in the native import's row shape."""
    decoded = {
        "files": {name: base64.b64encode(content).decode() for name, content in files.items()},
        "file_metadata": {name: {"mode": "0755" if name.endswith(".sh") else "0644", "mtime_ns": 0} for name in files},
        "archive_links": {},
        "directory_metadata": {},
        "archive_sha256": (
            hashlib.sha256("".join(f"{name}\0{files[name].hex()}\0" for name in sorted(files)).encode()).hexdigest()
        ),
    }
    source = Source(
        dataset=KEEP_RELEASE_REPO,
        revision=KEEP_RELEASE_REVISION,
        row=row.instance_id,
        importer_revision=UPSTREAM_REVISION,
    )
    return RawRow(id=f"openswe-{row.instance_id}", source=source, data=decoded)


def convert_row(
    row: OpenSweRow,
    condensed_bases: dict[str, str],
    context: ConversionContext,
    resolve: Callable[[str], str] | None = None,
) -> NativeRowConversion | ImportRejection | OpenSweRowExcluded:
    """Convert one keep-only row into a pinned, cohort-converted TaskSpec.

    The shared chain runs after the package renders: base-image pinning (the
    registry resolver with its digest cache), the static blocker audit, and the
    cohort conversion to the executable ScriptGrader form. ``resolve`` overrides
    the registry resolver (tests stage a fake); the production default is the
    shared resolver. Registry refusals and static blockers become typed
    per-row exclusions instead of failing the whole conversion.
    """
    from experiments.post_training.task_curation.native_source_conversion import (  # noqa: PLC0415
        BasePinMissing,
        BasePinRejected,
        CohortConversionRejected,
        convert_native_source_row,
    )

    try:
        files = package_files(row, condensed_bases, resolve=resolve)
    except OpenSweRowExcluded as excluded:
        return excluded
    task = native_task(row_to_raw(row, files), context)
    if isinstance(task, ImportRejection):
        return task
    try:
        if resolve is not None:
            return convert_native_source_row(task, input_task_sha256=task_spec_sha256(task), resolve=resolve)
        return convert_native_source_row(task, input_task_sha256=task_spec_sha256(task))
    except BasePinRejected as error:
        # BasePinMissing subclasses BasePinRejected: a missing tag is a missing image;
        # any other refusal fails closed as a rejected pin.
        reason = "base_image_missing" if isinstance(error, BasePinMissing) else "base_pin_rejected"
        return OpenSweRowExcluded(reason, str(error))
    except CohortConversionRejected as error:
        return OpenSweRowExcluded("cohort_conversion_rejected", str(error))
