# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the pinned fixture normalizer and run one fixture audit."""

import argparse
import os
import subprocess
from pathlib import Path

SHUTTLE_TEST_OPT_LABEL = "@shuttle_mlir//:shuttle-test-opt"


def build_and_resolve_normalizer(
    *,
    bazel: Path,
    xla_source: Path,
    output_user_root: Path,
    repository_cache: Path,
    jobs: int,
    ram_mb: int,
) -> Path:
    """Build and resolve the one test-only fixture normalizer output."""
    startup_flags = [f"--output_user_root={output_user_root}"]
    command_flags = [
        f"--repository_cache={repository_cache}",
        f"--jobs={jobs}",
        f"--local_cpu_resources={jobs}",
        f"--local_ram_resources={ram_mb}",
        "--noshow_progress",
        "--show_result=0",
    ]
    subprocess.run(
        [str(bazel), *startup_flags, "build", *command_flags, SHUTTLE_TEST_OPT_LABEL],
        cwd=xla_source,
        check=True,
    )
    result = subprocess.run(
        [str(bazel), *startup_flags, "cquery", "--output=files", SHUTTLE_TEST_OPT_LABEL],
        cwd=xla_source,
        check=True,
        capture_output=True,
        text=True,
    )
    output_files = result.stdout.splitlines()
    if len(output_files) != 1 or not output_files[0]:
        raise RuntimeError(f"{SHUTTLE_TEST_OPT_LABEL} produced {len(output_files)} output paths")
    relative_path = Path(output_files[0])
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError(f"{SHUTTLE_TEST_OPT_LABEL} produced an unsafe output path: {relative_path}")
    if relative_path.name != "shuttle-test-opt":
        raise RuntimeError(f"{SHUTTLE_TEST_OPT_LABEL} output must be named shuttle-test-opt, found {relative_path.name}")
    normalizer = xla_source / relative_path
    if not normalizer.is_file() or not os.access(normalizer, os.X_OK):
        raise RuntimeError(f"{SHUTTLE_TEST_OPT_LABEL} output is not executable: {normalizer}")
    return normalizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bazel", required=True, type=Path)
    parser.add_argument("--xla-source", required=True, type=Path)
    parser.add_argument("--output-user-root", required=True, type=Path)
    parser.add_argument("--repository-cache", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=int)
    parser.add_argument("--ram-mb", required=True, type=int)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--verifier", type=Path)
    arguments = parser.parse_args()
    if arguments.jobs <= 0 or arguments.ram_mb <= 0:
        parser.error("jobs and ram-mb must be positive")

    normalizer = build_and_resolve_normalizer(
        bazel=arguments.bazel,
        xla_source=arguments.xla_source,
        output_user_root=arguments.output_user_root,
        repository_cache=arguments.repository_cache,
        jobs=arguments.jobs,
        ram_mb=arguments.ram_mb,
    )
    print(f"fixture_audit_normalizer={normalizer}", flush=True)
    subprocess.run(
        [str(arguments.python), str(arguments.generator), "--normalizer", str(normalizer)],
        check=True,
    )
    if arguments.verifier is not None:
        subprocess.run(
            [str(arguments.python), str(arguments.verifier), "--normalizer", str(normalizer)],
            check=True,
        )
        print(f"fixture_verifier={arguments.verifier}")
    print("fixture_audit=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
