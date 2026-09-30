#!/usr/bin/env python3
"""Run one completed-checkpoint validation attempt without construction repair."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
from pathlib import Path

from capability_pipeline.checkpoint_revalidation import (
    prepare_checkpoint_revalidation,
    run_prepared_checkpoint_revalidation,
    validate_checkpoint_bundle,
)
from capability_pipeline.synthesis import (
    OfficialToolchain,
    OMPAgent,
    SynthesisError,
    _synthesize_attempt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-bundle-manifest-sha256", required=True)
    parser.add_argument("--expected-request-sha256", required=True)
    parser.add_argument("--new-controller-provenance", type=Path, required=True)
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
    if args.session_time < 1 or args.validation_timeout < 1 or args.max_continuations < 0:
        raise SynthesisError("invalid checkpoint-revalidation time limit")
    plan = validate_checkpoint_bundle(args.bundle, expected_manifest_sha256=args.expected_bundle_manifest_sha256, expected_request_sha256=args.expected_request_sha256)
    try:
        provenance = json.loads(args.new_controller_provenance.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SynthesisError("new controller provenance is unreadable") from error
    executable = shutil.which(args.omp) if os.sep not in args.omp else args.omp
    if not executable or not Path(executable).is_file():
        raise SynthesisError(f"OMP executable is unavailable: {args.omp}")
    runtime_runner = Path(args.runtime_runner).resolve() if args.runtime_runner else None
    if runtime_runner and (not runtime_runner.is_file() or not os.access(runtime_runner, os.X_OK)):
        raise SynthesisError("runtime runner must be executable")
    daytona_value = args.daytona_tools or os.environ.get("CAPABILITY_DAYTONA_TOOLS")
    daytona_tools = Path(daytona_value).resolve() if daytona_value else None
    if daytona_tools:
        missing = [name for name in ("dt.py", "dt.sh", "validate_env.py", "verify.py", "adapter.py") if not (daytona_tools / name).is_file()]
        if missing:
            raise SynthesisError("Daytona tool directory is incomplete: " + ", ".join(missing))
    overlay_value = args.research_overlay or os.environ.get("CAPABILITY_OMP_CONFIG") or os.environ.get("RESEARCH_OVERLAY")
    overlay = Path(overlay_value).resolve() if overlay_value else None
    if overlay and not overlay.is_file():
        raise SynthesisError("research overlay is unavailable")
    toolchain = OfficialToolchain.resolve(Path(args.bundle), args.taskcompendium_source)
    agent = OMPAgent(str(executable), args.model, args.session_time, args.max_continuations, overlay)
    prepared = prepare_checkpoint_revalidation(plan, args.out, new_controller=provenance)
    with (prepared.root / ".controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_prepared_checkpoint_revalidation(
            prepared,
            lambda item, root: _synthesize_attempt(
                item,
                root / plan.construction_root,
                agent,
                toolchain,
                runtime_runner,
                args.validation_timeout,
                daytona_tools,
            ),
        )
    return 0 if result.get("state") == "quality_accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
