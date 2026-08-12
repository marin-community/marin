# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute audited inputs and Shuttle-lowered StableHLO bitwise on JAX CPU."""

import argparse
import runpy
import subprocess
from pathlib import Path

import jax
import numpy as np
from jax._src import compiler, xla_bridge
from jax._src.interpreters import mlir
from jaxlib import xla_client
from jaxlib.mlir import ir

PIPELINES = (
    "shuttle-stablehlo-source-ordered-pipeline",
    "shuttle-stablehlo-fast-pipeline",
)
MAP_FIXTURES = (
    ("f32-map-shape-ops.mlir", ((3,), (2, 2))),
    ("f32-mapped-singleton-broadcast.mlir", ((7, 1), (1, 13), (1, 7))),
)


def load_fixtures():
    path = Path(__file__).with_name("regenerate-jax-fixtures.py")
    return runpy.run_path(str(path))["FIXTURES"]


def compile_stablehlo(source: str):
    context = mlir.make_ir_context()
    with context:
        module = ir.Module.parse(source)
        backend = xla_bridge.get_backend("cpu")
        devices = xla_client.DeviceList(tuple(backend.devices()))
        options = compiler.get_compile_options(1, 1, backend=backend)
        executable = compiler.backend_compile_and_load(backend, module, devices, options, [])
    return context, executable


def execute(source: str, inputs: list[np.ndarray]) -> list[np.ndarray]:
    context, executable = compile_stablehlo(source)
    del context
    device_inputs = [jax.device_put(value) for value in inputs]
    return [np.asarray(value) for value in executable.execute(device_inputs)]


def fixed_inputs(shapes: tuple[tuple[int, ...], ...]) -> list[np.ndarray]:
    values = []
    for ordinal, shape in enumerate(shapes):
        size = int(np.prod(shape))
        start = np.float32(-0.75 + ordinal * 0.125)
        stop = np.float32(0.875 + ordinal * 0.125)
        values.append(np.linspace(start, stop, size, dtype=np.float32).reshape(shape))
    return values


def positive_inputs(shapes: tuple[tuple[int, ...], ...]) -> list[np.ndarray]:
    values = []
    for ordinal, shape in enumerate(shapes):
        size = int(np.prod(shape))
        start = np.float32(0.25 + ordinal * 0.125)
        stop = np.float32(1.0 + ordinal * 0.125)
        values.append(np.linspace(start, stop, size, dtype=np.float32).reshape(shape))
    return values


def lower(source: Path, shuttle_opt: Path, pipeline: str) -> str:
    return subprocess.run(
        [str(shuttle_opt), f"--{pipeline}", str(source)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuttle-opt", required=True, type=Path)
    arguments = parser.parse_args()
    fixtures = load_fixtures()
    fixture_directory = Path(__file__).parent
    fixture_inputs = [(fixture.filename, fixture.shapes, fixed_inputs) for fixture in fixtures]
    fixture_inputs.extend((filename, shapes, positive_inputs) for filename, shapes in MAP_FIXTURES)
    for filename, shapes, input_factory in fixture_inputs:
        path = fixture_directory / filename
        source = path.read_text()
        inputs = input_factory(shapes)
        reference = execute(source, inputs)
        for pipeline in PIPELINES:
            actual = execute(lower(path, arguments.shuttle_opt, pipeline), inputs)
            if len(actual) != len(reference) or any(
                not np.array_equal(expected, result) for expected, result in zip(reference, actual, strict=True)
            ):
                parser.error(f"CPU bitwise parity failed: {filename} ({pipeline})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
