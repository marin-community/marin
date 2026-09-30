#!/usr/bin/env python3
"""Exercise the patched pinned Harbor runner without inference or a real trial."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import taskcompendium.harbor.runner as native_runner

from capability_pipeline.composite_extension import (
    IMPORT_PATH,
    PATCHED_RUNNER_SHA256,
    UNSUPPORTED_SPECIFICATION,
)
from scripts.run_composite_probe import build

HARBOR_REVISION = "93147ea9e07b04ec8d2eb5afd2916386f1aacc69"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def run(output: Path) -> dict:
    if sha256(Path(native_runner.__file__)) != PATCHED_RUNNER_SHA256:
        raise RuntimeError("runner smoke requires the exact patched source overlay")

    class Distribution:
        @staticmethod
        def read_text(name: str) -> str:
            if name != "direct_url.json":
                raise ValueError("unexpected distribution metadata request")
            return json.dumps({"vcs_info": {"commit_id": HARBOR_REVISION}})

    class Result:
        pass

    class FakeTrial:
        agent = object()

        @staticmethod
        async def run() -> Result:
            return Result()

    class TrialFactory:
        calls = 0

        @classmethod
        async def create(cls, config) -> FakeTrial:
            del config
            cls.calls += 1
            return FakeTrial()

    native_runner.distribution = lambda name: Distribution()
    native_runner.TrialConfig.model_validate = lambda value: value
    native_runner.Trial = TrialFactory
    _, package = build(
        output,
        "https://example.invalid/v1",
        "glm-5.3",
        "glm",
    )
    try:
        await native_runner.run_trial(
            package,
            {"verifier": {"import_path": "ordinary:Verifier"}},
            output / "trials",
            "wrong-verifier",
        )
    except RuntimeError as error:
        if "declared verifier adapter" not in str(error):
            raise
    else:
        raise RuntimeError("patched runner accepted an undeclared verifier")
    if TrialFactory.calls:
        raise RuntimeError("wrong verifier reached Harbor Trial.create")
    result = await native_runner.run_trial(
        package,
        {"verifier": {"import_path": IMPORT_PATH}},
        output / "trials",
        "declared-verifier",
    )
    if not isinstance(result, Result) or TrialFactory.calls != 1:
        raise RuntimeError(
            "guarded package did not reach Harbor Trial.create exactly once"
        )
    evidence = {
        "schema_version": "composite-runner-integration-v1",
        "state": "passed",
        "inference_executed": False,
        "daytona_executed": False,
        "wrong_verifier_rejected_before_trial": True,
        "guarded_package_reached_trial_create": True,
        "ordinary_specification_is_sentinel": (
            package / "specification.json"
        ).read_bytes()
        == UNSUPPORTED_SPECIFICATION,
        "preserved_specification_sha256": sha256(
            package / "composite-specification.json"
        ),
        "manifest_sha256": sha256(package / "manifest.json"),
        "patched_runner_sha256": sha256(Path(native_runner.__file__)),
    }
    evidence_path = output / "runner-smoke.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    evidence = asyncio.run(run(parser.parse_args().out.resolve()))
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
