"""Freeze and run a worker-local GLM proposal-to-construction pipeline.

This is deliberately a thin composition boundary: proposal admission remains in
``cli.propose`` and construction remains in ``synthesis.synthesize``.  It never
turns a rejected proposal into an admitted one, and it does not infer coverage
from the accepted subset.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
from typing import Any

from . import cli, synthesis
from .inference import atomic_json, digest
from .runtime import sha256, tree_sha256

SCHEMA = "capability-generate-v1"
_PROPOSAL_FILES = (
    "accepted.json",
    "rejected.json",
    "null.json",
    "plans.json",
    "proposals.json",
    "report.json",
    "run.json",
)
_SYNTHESIS_FILES = ("report.json", "run.json", "tasks.json")


class GenerateError(ValueError):
    """The frozen orchestration boundary cannot be safely resumed."""


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise GenerateError(f"invalid required artifact: {path}") from error


def _files(directory: Path, names: tuple[str, ...]) -> dict[str, str]:
    result = {}
    for name in names:
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise GenerateError(f"required stage artifact is missing or unsafe: {path}")
        result[name] = sha256(path)
    return result


def _synthesis_state(construction_dir: Path) -> str | None:
    """``report.json["state"]`` of a construction directory.

    ``synthesize`` exits 0 once every item is terminal (a failed item is an
    outcome); only ``"complete"`` means every item was quality accepted.
    """
    try:
        report = json.loads((construction_dir / "report.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    state = report.get("state") if isinstance(report, dict) else None
    return state if isinstance(state, str) else None


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise GenerateError(f"refusing to overwrite frozen receipt: {path}")
    atomic_json(path, value)


def _settings(args) -> dict[str, Any]:
    """Only behavior-affecting values are frozen in the run identity."""
    settings = {
        "concurrency": args.concurrency,
        "tier": args.tier,
        "proposal_repair_rounds": args.proposal_repair_rounds,
        "construction_max_repair_rounds": _construction_max_repair_rounds(),
        "proposal_hold_seconds": args.proposal_hold_seconds,
        "structured_output": args.structured_output,
        "omp": args.omp,
        "model": args.model,
        "session_time": args.session_time,
        "max_continuations": args.max_continuations,
        "validation_timeout": args.validation_timeout,
        # Staging paths change on a durable worker resume.  Bind content and
        # the pinned source lock, never their transient absolute locations.
        "controller_package_sha256": _controller_package_sha256(),
        "taskcompendium_lock_sha256": sha256(synthesis.SOURCE_LOCK),
        "runtime_runner_sha256": _optional_file_sha(args.runtime_runner),
        "daytona_tools_tree_sha256": _optional_tree_sha(
            args.daytona_tools or os.environ.get("CAPABILITY_DAYTONA_TOOLS")
        ),
        "research_overlay_sha256": _optional_file_sha(
            args.research_overlay
            or os.environ.get("CAPABILITY_OMP_CONFIG")
            or os.environ.get("RESEARCH_OVERLAY")
        ),
    }
    adoption = _adoption_inputs(args)
    if adoption is not None:
        snapshot, archive, receipt = adoption
        settings["proposal_adoption"] = {
            "snapshot_capture_sha256": _optional_file_sha(
                str(snapshot / "snapshot-capture.json")
            ),
            "snapshot_pull_sha256": _optional_file_sha(
                str(snapshot / "pull-manifest.json")
            ),
            "source_archive_sha256": _optional_file_sha(str(archive)),
            "launch_receipt_sha256": _optional_file_sha(str(receipt)),
        }
    return settings


def _adoption_inputs(args) -> tuple[Path, Path, Path] | None:
    values = tuple(
        getattr(args, name, None)
        for name in (
            "adopt_proposal_checkpoint",
            "adoption_source_archive",
            "adoption_launch_receipt",
        )
    )
    if all(value is None for value in values):
        return None
    if any(not isinstance(value, str) or not value for value in values):
        raise GenerateError(
            "proposal adoption requires checkpoint, source archive and launch receipt together"
        )
    paths = tuple(Path(value) for value in values)
    if any(path.is_symlink() for path in paths) or not paths[0].is_dir():
        raise GenerateError("proposal adoption inputs are missing or linked")
    return paths


def _construction_max_repair_rounds() -> int:
    try:
        rounds = int(os.environ.get("CAPABILITY_MAX_REPAIR_ROUNDS", "2"))
    except ValueError as error:
        raise GenerateError(
            "CAPABILITY_MAX_REPAIR_ROUNDS must be an integer"
        ) from error
    if not 0 <= rounds <= 5:
        raise GenerateError(
            "CAPABILITY_MAX_REPAIR_ROUNDS must be between zero and five"
        )
    return rounds


def _optional_file_sha(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_file() or path.is_symlink():
        raise GenerateError("configured file input is missing or unsafe")
    return sha256(path)


def _optional_tree_sha(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if (
        not path.is_dir()
        or path.is_symlink()
        or any(p.is_symlink() for p in path.rglob("*"))
    ):
        raise GenerateError("configured directory input is missing or unsafe")
    return tree_sha256(path)


def _controller_package_sha256() -> str:
    """Bind executable controller sources, excluding nondeterministic bytecode."""
    package = Path(__file__).resolve().parent
    files = {
        path.relative_to(package).as_posix(): sha256(path)
        for path in package.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".py", ".json"}
    }
    return digest(dict(sorted(files.items())))


def _freeze_input(root: Path, pilot: Path) -> Path:
    """Copy the complete pilot manifest once, and reject source drift on resume."""
    return _freeze_named_input(root, "input-pilot.json", pilot)


def _freeze_named_input(root: Path, name: str, source: Path) -> Path:
    if not source.is_file() or source.is_symlink():
        raise GenerateError("frozen input source is missing or unsafe")
    frozen = root / name
    source_bytes = source.read_bytes()
    if frozen.exists() or frozen.is_symlink():
        if frozen.is_symlink() or frozen.read_bytes() != source_bytes:
            raise GenerateError("frozen pilot manifest changed")
    else:
        with frozen.open("xb") as stream:
            stream.write(source_bytes)
    # Parse after copying so malformed source data never masquerades as frozen input.
    _json(frozen)
    return frozen


def _freeze_run(root: Path, pilot: Path, args) -> dict[str, Any]:
    path = root / "generate-run.json"
    settings = _settings(args)
    existing = _json(path) if path.exists() else None
    if (
        existing is not None
        and "proposal_adoption" not in settings
        and "proposal_adoption" in existing.get("settings", {})
    ):
        # After the adopted proposal tree has been sealed, a normal durable
        # resume needs that tree, not a second download of the source checkpoint.
        receipt = _validated_proposal_receipt(root, existing)
        provenance = receipt.get("adoption", {})
        frozen = existing["settings"]["proposal_adoption"]
        if provenance.get("original_source_archive_sha256") != frozen.get(
            "source_archive_sha256"
        ) or provenance.get("original_launch_receipt_sha256") != frozen.get(
            "launch_receipt_sha256"
        ):
            raise GenerateError("adopted proposal resume provenance differs")
        settings["proposal_adoption"] = frozen
    identity = {
        "schema_version": SCHEMA,
        "pilot_sha256": sha256(pilot),
        "settings": settings,
    }
    identity["identity_sha256"] = digest(identity)
    if path.exists() or path.is_symlink():
        existing = _json(path)
        if existing != identity:
            raise GenerateError("generate input identity or settings changed")
    else:
        atomic_json(path, identity)
    return identity


def _validated_proposal_receipt(root: Path, identity: dict[str, Any]) -> dict[str, Any]:
    proposal = root / "proposal"
    receipt_path = root / "proposal-receipt.json"
    receipt = _json(receipt_path)
    if (
        receipt.get("schema_version") != SCHEMA
        or receipt.get("stage") != "proposal"
        or receipt.get("identity_sha256") != identity["identity_sha256"]
        or receipt.get("exit_code") not in (0, 2)
        or receipt.get("files") != _files(proposal, _PROPOSAL_FILES)
        or receipt.get("tree_sha256") != tree_sha256(proposal)
    ):
        raise GenerateError("frozen proposal receipt does not match proposal outputs")
    accepted = _accepted_or_empty(proposal / "accepted.json")
    if receipt.get("accepted_count") != len(accepted):
        raise GenerateError("frozen proposal accepted inventory changed")
    return receipt


def _accepted_or_empty(path: Path) -> list[dict[str, Any]]:
    """An all-rejected/null proposal run is a valid, non-constructing result."""
    document = _json(path)
    if document == []:
        return []
    return synthesis.load_accepted(path)


def _proposal_args(args, pilot: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        pilot=str(pilot),
        out=str(output),
        seed_run=None,
        concurrency=args.concurrency,
        tier=args.tier,
        repair_rounds=args.proposal_repair_rounds,
        hold_seconds=args.proposal_hold_seconds,
        structured_output=args.structured_output,
        limit=None,
    )


def _synthesis_args(args, accepted: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        accepted=str(accepted),
        out=str(output),
        concurrency=args.concurrency,
        tier=args.tier,
        limit=None,
        omp=args.omp,
        model=args.model,
        session_time=args.session_time,
        max_continuations=args.max_continuations,
        validation_timeout=args.validation_timeout,
        taskcompendium_source=args.taskcompendium_source,
        runtime_runner=args.runtime_runner,
        daytona_tools=args.daytona_tools,
        research_overlay=args.research_overlay,
        retry_infrastructure=False,
        infrastructure_health_receipt=None,
        retry_adversary=False,
    )


def _coverage(
    manifest: Path, proposal: Path, synthesis_roots: list[Path]
) -> dict[str, Any]:
    """Use independent accounting when present; never infer it from acceptance."""
    base = {
        "manifest_sha256": sha256(manifest),
        "proposal_tree_sha256": tree_sha256(proposal),
        "synthesis_tree_sha256": [tree_sha256(root) for root in synthesis_roots],
    }
    try:
        from .coverage import CoverageError, coverage_report
    except ImportError:
        return {**base, "state": "pending", "reason": "coverage module is unavailable"}
    try:
        report = coverage_report(
            manifest, proposal_root=proposal, synthesis_roots=synthesis_roots
        )
    except CoverageError as error:
        return {**base, "state": "failed", "reason": str(error)}
    if (
        not isinstance(report, dict)
        or type(report.get("accounting_complete")) is not bool
    ):
        return {
            **base,
            "state": "failed",
            "reason": "coverage module returned an invalid report",
        }
    return {
        **base,
        "state": "complete" if report["accounting_complete"] else "incomplete",
        "report": report,
    }


def generate(args) -> int:
    adoption = _adoption_inputs(args)
    pilot = Path(args.pilot).resolve()
    if not pilot.is_file() or pilot.is_symlink():
        raise GenerateError("pilot manifest is missing or unsafe")
    if args.concurrency < 1 or args.proposal_repair_rounds < 0:
        raise GenerateError(
            "concurrency must be positive; proposal repair rounds nonnegative"
        )
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or root.is_relative_to(pilot.parent) and root == pilot.parent:
        raise GenerateError("generate output is unsafe")
    with (root / ".controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen_pilot = _freeze_input(root, pilot)
        identity = _freeze_run(root, frozen_pilot, args)
        proposal_dir = root / "proposal"
        construction_dir = root / "construction"
        proposal_receipt_path = root / "proposal-receipt.json"

        # Once construction exists, proposals must come exclusively from the
        # frozen receipt; no proposal generation is permitted on resume.
        if construction_dir.exists() and not proposal_receipt_path.is_file():
            raise GenerateError("construction exists without a frozen proposal receipt")
        if adoption is not None and not proposal_receipt_path.exists():
            from .proposal_adoption import validate_and_adopt

            snapshot, archive, launch = adoption
            validate_and_adopt(
                snapshot_dir=snapshot,
                source_archive=archive,
                launch_receipt=launch,
                pilot=frozen_pilot,
                destination=root,
                target_identity=identity,
            )
            if not proposal_receipt_path.is_file():
                raise GenerateError(
                    "proposal adoption did not produce its frozen receipt"
                )
        if proposal_receipt_path.is_file():
            proposal_receipt = _validated_proposal_receipt(root, identity)
        else:
            if proposal_dir.exists() and (
                proposal_dir.is_symlink() or not proposal_dir.is_dir()
            ):
                raise GenerateError("proposal directory is unsafe")
            proposal_dir.mkdir(exist_ok=True)
            # ``propose`` resumes its StageStore cache.  This is permitted only
            # before a construction directory exists, checked above.
            exit_code = cli.propose(_proposal_args(args, frozen_pilot, proposal_dir))
            if exit_code not in (0, 2):
                raise GenerateError(
                    f"proposal stage returned unexpected exit code {exit_code}"
                )
            # A normal exit 2 means portfolio iteration is incomplete, but its
            # accepted siblings remain exactly what synthesize validates.
            accepted = _accepted_or_empty(proposal_dir / "accepted.json")
            proposal_receipt = {
                "schema_version": SCHEMA,
                "stage": "proposal",
                "identity_sha256": identity["identity_sha256"],
                "exit_code": exit_code,
                "accepted_count": len(accepted),
                "files": _files(proposal_dir, _PROPOSAL_FILES),
                "tree_sha256": tree_sha256(proposal_dir),
            }
            _write_new(proposal_receipt_path, proposal_receipt)

        accepted_path = proposal_dir / "accepted.json"
        accepted = _accepted_or_empty(accepted_path)
        construction_receipt_path = root / "construction-receipt.json"
        construction_progress_path = root / "construction-progress.json"
        if construction_receipt_path.is_file():
            receipt = _json(construction_receipt_path)
            if (
                receipt.get("schema_version") != SCHEMA
                or receipt.get("stage") != "construction"
                or receipt.get("identity_sha256") != identity["identity_sha256"]
                or receipt.get("proposal_receipt_sha256")
                != sha256(proposal_receipt_path)
                or receipt.get("files") != _files(construction_dir, _SYNTHESIS_FILES)
                or receipt.get("tree_sha256") != tree_sha256(construction_dir)
            ):
                raise GenerateError(
                    "frozen construction receipt does not match outputs"
                )
            exit_code = receipt.get("exit_code")
            if exit_code not in (0, 2):
                raise GenerateError(
                    "frozen construction receipt has an invalid exit code"
                )
            synthesis_state = _synthesis_state(construction_dir)
        elif not accepted:
            # Preserve the explicit all-rejected/null disposition.  Do not
            # invoke toolchain resolution or an OMP model session.
            receipt = {
                "schema_version": SCHEMA,
                "stage": "construction",
                "identity_sha256": identity["identity_sha256"],
                "proposal_receipt_sha256": sha256(proposal_receipt_path),
                "exit_code": 2,
                "state": "skipped_no_accepted_proposals",
                "files": {},
                "tree_sha256": None,
            }
            atomic_json(construction_progress_path, receipt)
            exit_code = 2
            synthesis_state = None
        else:
            if construction_dir.exists() and (
                construction_dir.is_symlink() or not construction_dir.is_dir()
            ):
                raise GenerateError("construction directory is unsafe")
            construction_dir.mkdir(exist_ok=True)
            exit_code = synthesis.synthesize(
                _synthesis_args(args, accepted_path, construction_dir)
            )
            if exit_code not in (0, 2):
                raise GenerateError(
                    f"construction stage returned unexpected exit code {exit_code}"
                )
            synthesis_state = _synthesis_state(construction_dir)
            receipt = {
                "schema_version": SCHEMA,
                "stage": "construction",
                "identity_sha256": identity["identity_sha256"],
                "proposal_receipt_sha256": sha256(proposal_receipt_path),
                "exit_code": exit_code,
                "synthesis_state": synthesis_state,
                "files": _files(construction_dir, _SYNTHESIS_FILES),
                "tree_sha256": tree_sha256(construction_dir),
            }
            if exit_code == 0 and synthesis_state == "complete":
                _write_new(construction_receipt_path, receipt)
            else:
                # A `needs_continuation` synthesis report is not immutable
                # terminal evidence, even when synthesize exited 0 (exit 0 now
                # means every item is terminal, not that every item was
                # accepted).  A later worker invocation resumes this same
                # directory and its retained semantic budgets; coverage below
                # may still freeze it as honest terminal accounting.
                atomic_json(construction_progress_path, receipt)

        coverage = _coverage(
            frozen_pilot, proposal_dir, [] if not accepted else [construction_dir]
        )
        covered = coverage.get("state") == "complete"
        # Honest exhausted rejections are terminal accounting too. Preserve the
        # construction exit code and yield separately; do not ask an operator to
        # resume a budget-exhausted task merely to obtain a zero worker exit.
        complete = covered
        construction_complete = (
            bool(accepted) and exit_code == 0 and synthesis_state == "complete"
        )
        if covered and accepted and not construction_receipt_path.exists():
            _write_new(construction_receipt_path, receipt)
        report = {
            "schema_version": SCHEMA,
            "proposal": proposal_receipt,
            "construction": receipt,
            "coverage": coverage,
            "generation_state": (
                "complete"
                if construction_complete
                else "complete_no_accepted_proposals"
                if not accepted and covered
                else "complete_with_rejections"
                if covered
                else "needs_continuation"
            ),
            "training_ready_quality": {
                "state": "unassessed",
                "reason": "synthesis acceptance and slot coverage do not establish broader training-ready quality",
            },
            "state": "complete" if complete else "needs_continuation",
        }
        atomic_json(root / "report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0 if complete else 2


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "generate", help="Run frozen GLM proposal and construction stages"
    )
    parser.add_argument(
        "--pilot", required=True, help="pilot manifest consumed by proposal generation"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--adopt-proposal-checkpoint",
        help="verified original proposal snapshot for a fresh controller migration",
    )
    parser.add_argument(
        "--adoption-source-archive", help="original immutable controller source archive"
    )
    parser.add_argument(
        "--adoption-launch-receipt",
        help="original launch receipt binding source identity",
    )
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument(
        "--tier", choices=("interactive", "bulk"), default="interactive"
    )
    parser.add_argument("--proposal-repair-rounds", type=int, default=1)
    parser.add_argument("--proposal-hold-seconds", type=int, default=3600)
    parser.add_argument(
        "--structured-output", choices=("json_schema", "off"), default="json_schema"
    )
    parser.add_argument("--omp", default="omp")
    parser.add_argument("--model", default="glm-orion/glm-5.3")
    parser.add_argument("--session-time", type=int, default=28_800)
    parser.add_argument("--max-continuations", type=int, default=8)
    parser.add_argument("--validation-timeout", type=int, default=14_400)
    parser.add_argument("--taskcompendium-source")
    parser.add_argument("--runtime-runner")
    parser.add_argument("--daytona-tools")
    parser.add_argument("--research-overlay")
    parser.set_defaults(func=generate)
