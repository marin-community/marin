"""Run one hash-bound private-runtime image migration revalidation."""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
from pathlib import Path

from capability_pipeline.image_migration import (
    prepare_image_migration_revalidation,
    run_prepared_image_migration,
    validate_image_migration_bundle,
)
from capability_pipeline.image_pointer_migration import (
    validate_pointer_migration_bundle,
)
from capability_pipeline.synthesis import (
    OfficialToolchain,
    OMPAgent,
    SynthesisError,
    _synthesize_attempt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Revalidate one reviewed, image-only TaskSpec migration"
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-image")
    parser.add_argument("--expected-candidate-image")
    parser.add_argument("--expected-verifier-image")
    parser.add_argument("--expected-bundle-manifest-sha256", required=True)
    parser.add_argument("--expected-migration-receipt-sha256", required=True)
    parser.add_argument("--taskcompendium-source")
    parser.add_argument("--runtime-runner")
    parser.add_argument("--daytona-tools")
    parser.add_argument("--omp", default="omp")
    parser.add_argument("--model", default="glm-orion/glm-5.3")
    parser.add_argument("--session-time", type=int, default=28800)
    parser.add_argument("--max-continuations", type=int, default=8)
    parser.add_argument("--research-overlay")
    parser.add_argument("--validation-timeout", type=int, default=14400)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.session_time < 1 or args.validation_timeout < 1:
        raise SynthesisError("timeouts must be positive")
    if args.max_continuations < 0:
        raise SynthesisError("max continuations cannot be negative")
    bundle = Path(args.bundle).resolve()
    output = Path(args.out).resolve()
    pointer_values = (args.expected_candidate_image, args.expected_verifier_image)
    if args.expected_image and any(pointer_values):
        raise SynthesisError(
            "single-image and pointer-image expectations are mutually exclusive"
        )
    if args.expected_image:
        plan = validate_image_migration_bundle(
            bundle,
            expected_image=args.expected_image,
            expected_manifest_sha256=args.expected_bundle_manifest_sha256,
            expected_receipt_sha256=args.expected_migration_receipt_sha256,
        )
    elif all(pointer_values):
        plan = validate_pointer_migration_bundle(
            bundle,
            expected_images={
                "candidate": args.expected_candidate_image,
                "verifier": args.expected_verifier_image,
            },
            expected_manifest_sha256=args.expected_bundle_manifest_sha256,
            expected_receipt_sha256=args.expected_migration_receipt_sha256,
        )
    else:
        raise SynthesisError(
            "provide either --expected-image or both pointer image expectations"
        )

    executable = shutil.which(args.omp) if os.sep not in args.omp else args.omp
    if not executable or not Path(executable).is_file():
        raise SynthesisError(f"OMP executable is unavailable: {args.omp}")
    runtime_runner = (
        Path(args.runtime_runner).resolve() if args.runtime_runner else None
    )
    if runtime_runner and (
        not runtime_runner.is_file() or not os.access(runtime_runner, os.X_OK)
    ):
        raise SynthesisError("runtime runner must be executable")
    daytona_value = args.daytona_tools or os.environ.get("CAPABILITY_DAYTONA_TOOLS")
    daytona_tools = Path(daytona_value).resolve() if daytona_value else None
    if daytona_tools:
        missing = [
            name
            for name in ("dt.py", "dt.sh", "validate_env.py", "verify.py", "adapter.py")
            if not (daytona_tools / name).is_file()
        ]
        if missing:
            raise SynthesisError(
                "Daytona tool directory is incomplete: " + ", ".join(missing)
            )
    overlay_value = (
        args.research_overlay
        or os.environ.get("CAPABILITY_OMP_CONFIG")
        or os.environ.get("RESEARCH_OVERLAY")
    )
    research_overlay = Path(overlay_value).resolve() if overlay_value else None
    if research_overlay and not research_overlay.is_file():
        raise SynthesisError(f"research overlay is unavailable: {research_overlay}")

    toolchain = OfficialToolchain.resolve(bundle, args.taskcompendium_source)
    agent = OMPAgent(
        str(executable),
        args.model,
        args.session_time,
        args.max_continuations,
        research_overlay,
    )
    prepared = prepare_image_migration_revalidation(plan, output)
    with (output / ".controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def validate_once(item, root):
            return _synthesize_attempt(
                item,
                root,
                agent,
                toolchain,
                runtime_runner,
                args.validation_timeout,
                daytona_tools,
            )

        result = run_prepared_image_migration(prepared, validate_once)
    return 0 if result.get("state") == "quality_accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
