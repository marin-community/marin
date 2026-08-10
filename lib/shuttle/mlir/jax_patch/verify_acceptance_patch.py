# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the guarded observer bridge selection in a patched JAX checkout."""

import argparse
import re
import subprocess
from collections import Counter
from collections.abc import Collection
from pathlib import Path

PINNED_JAX_REVISION = "619764c15117fbefc4ba13ab941871cb514c23f6"
DEFINE = "--define=SHUTTLE_TEST_OBSERVER=1"
BRIDGE = "@shuttle_mlir//:ShuttlePythonObserverTestBridge"
ADAPTER = "@shuttle_mlir//:ShuttleXlaRegistryAdapter"
_CQUERY_LABEL = re.compile(
    r"^(?P<label>(?:@{1,2}[A-Za-z0-9._+~-]+)?//[^\s():]*:[^\s():]+)(?: \((?:[0-9a-f]{7,64}|null)\))?$"
)


def parse_cquery_labels(output: str) -> tuple[str, ...]:
    """Parse Bazel cquery label output and discard configuration suffixes."""
    labels: list[str] = []
    for line in output.splitlines():
        match = _CQUERY_LABEL.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed cquery label line: {line!r}")
        labels.append(match.group("label"))
    if not labels:
        raise ValueError("cquery returned no labels")
    return tuple(labels)


def require_cquery_labels(output: str, required: Collection[str]) -> frozenset[str]:
    """Return parsed labels after requiring each exact dependency label."""
    labels = parse_cquery_labels(output)
    required_labels = frozenset(required)
    if len(required_labels) != len(required):
        raise ValueError("required cquery labels contain duplicates")
    label_counts = Counter(labels)
    missing = required_labels - set(label_counts)
    if missing:
        raise ValueError(f"cquery omitted required labels: {sorted(missing)}")
    duplicates = {label: label_counts[label] for label in sorted(required_labels) if label_counts[label] != 1}
    if duplicates:
        raise ValueError(f"cquery repeated required labels: {duplicates}")
    return frozenset(labels)


def run_query(
    bazel: Path,
    jax_source: Path,
    output_user_root: Path,
    xla_source: Path,
    shuttle_mlir: Path,
    arguments: list[str],
) -> str:
    return subprocess.run(
        [
            str(bazel),
            f"--output_user_root={output_user_root}",
            "cquery",
            f"--override_repository=xla={xla_source}",
            f"--override_repository=shuttle_mlir={shuttle_mlir}",
            *arguments,
        ],
        cwd=jax_source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bazel", required=True, type=Path)
    parser.add_argument("--jax-source", required=True, type=Path)
    parser.add_argument("--xla-source", required=True, type=Path)
    parser.add_argument("--shuttle-mlir", required=True, type=Path)
    parser.add_argument("--output-user-root", required=True, type=Path)
    arguments = parser.parse_args()

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=arguments.jax_source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PINNED_JAX_REVISION:
        parser.error(f"JAX revision is {revision}, expected {PINNED_JAX_REVISION}")
    source = (arguments.jax_source / "jaxlib" / "jax.cc").read_text()
    guarded_include = '#ifdef SHUTTLE_TEST_OBSERVER\n#include "shuttle/Testing/PythonObserverTestBridge.h"\n#endif'
    guarded_registration = (
        "#ifdef SHUTTLE_TEST_OBSERVER\n" "  mlir::shuttle::testing::registerShuttleObserverTestBindings(m);\n" "#endif"
    )
    if source.count(guarded_include) != 1 or source.count(guarded_registration) != 1:
        parser.error("jax.cc does not contain exactly one guarded test bridge binding")

    selected_build = run_query(
        arguments.bazel,
        arguments.jax_source,
        arguments.output_user_root,
        arguments.xla_source,
        arguments.shuttle_mlir,
        [DEFINE, "--output=build", "//jaxlib:_jax_pywrap_library"],
    )
    if selected_build.count('"-DSHUTTLE_TEST_OBSERVER"') != 1:
        parser.error("acceptance configuration did not select the _jax compile define")
    try:
        require_cquery_labels(
            run_query(
                arguments.bazel,
                arguments.jax_source,
                arguments.output_user_root,
                arguments.xla_source,
                arguments.shuttle_mlir,
                [DEFINE, "--output=label", "deps(//jaxlib:_jax_pywrap_library)"],
            ),
            (BRIDGE, ADAPTER),
        )
    except ValueError as error:
        parser.error(f"invalid acceptance _jax cquery evidence: {error}")

    default_build = run_query(
        arguments.bazel,
        arguments.jax_source,
        arguments.output_user_root,
        arguments.xla_source,
        arguments.shuttle_mlir,
        ["--output=build", "//jaxlib:_jax_pywrap_library"],
    )
    try:
        default_deps = require_cquery_labels(
            run_query(
                arguments.bazel,
                arguments.jax_source,
                arguments.output_user_root,
                arguments.xla_source,
                arguments.shuttle_mlir,
                ["--output=label", "deps(//jaxlib:_jax_pywrap_library)"],
            ),
            (ADAPTER,),
        )
    except ValueError as error:
        parser.error(f"invalid ordinary _jax cquery evidence: {error}")
    if "-DSHUTTLE_TEST_OBSERVER" in default_build or BRIDGE in default_deps:
        parser.error("ordinary _jax configuration unexpectedly includes the test bridge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
