# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a study's analysis script on a run's checkpoints and W&B history, and the tiny synthetic run its gate tests.

A study script (``experiments/belay/studies/X####.py``) defines ``analyze(checkpoints, history, out_dir)``:

- ``checkpoints`` maps each saved step to that step's parameters (each model array's checkpoint path, ``params/...``,
  to a NumPy array), read from storage only when a step is looked up, so a script can walk every step over training;
- ``history`` is the run's W&B log, one dict per logged row (``_step`` and the metrics logged at it), so statistics
  the run logged during training (``train/layer_out/*``, say) need no checkpoint;
- the script writes ``report.md`` and any PNG figures it links into ``out_dir``. It reads nothing else and changes no
  model code.

    python -m experiments.belay.study_checkpoint run --script <path> --input <checkpoints dir> --history <json> \
        --out <dir>
    python -m experiments.belay.study_checkpoint selftest --script <path>     # the code-change gate's CPU test

``run`` is what ``B analyze`` submits as an Iris CPU job next to the checkpoints; it copies the report to ``--out``
(the study's ``report/`` directory in the cluster's bucket). ``selftest`` builds a tiny fast_track run (two
checkpoints written with the same writer training uses, and a short synthetic history) and runs the script on it.
"""

import argparse
import importlib.util
import json
import re
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import equinox as eqx
import fsspec
import jax
import numpy as np
import tensorstore as ts
from levanter.checkpoint_manifest import read_manifest
from levanter.tensorstore_serialization import (
    ARRAY_DRIVER,
    KVSTORE_DRIVER,
    build_kvstore_spec,
    tree_serialize_leaves_tensorstore,
)

from experiments.belay.model_probe import tiny_config, tiny_mesh
from experiments.grug.fast_track.model import Transformer

PARAMS_PREFIX = "params/"
REPORT_FILE = "report.md"
REPORT_SUFFIXES = (".md", ".png")


STEP_DIR = re.compile(r"step-(\d+)$")


def checkpoint_steps(checkpoints: str) -> dict[int, str]:
    """A run's saved checkpoints by step: each ``step-N`` directory under ``checkpoints`` (or the one step directory
    given), with a manifest."""
    if read_manifest(checkpoints) is not None:
        match = STEP_DIR.search(checkpoints.rstrip("/"))
        return {int(match.group(1)) if match else 0: checkpoints}
    fs, root = fsspec.core.url_to_fs(checkpoints)
    base = checkpoints.rstrip("/")
    steps = {}
    for directory in fs.ls(root, detail=False):
        match = STEP_DIR.search(directory.rstrip("/"))
        if match and read_manifest(f"{base}/{match.group(0)}") is not None:
            steps[int(match.group(1))] = f"{base}/{match.group(0)}"
    if not steps:
        raise SystemExit(f"refusing: {checkpoints} holds no complete step-N checkpoint")
    return dict(sorted(steps.items()))


class Checkpoints(Mapping[int, dict[str, np.ndarray]]):
    """A run's saved checkpoints by step, in step order; a step's parameters are read when it is looked up."""

    def __init__(self, steps: dict[int, str]) -> None:
        self._steps = steps

    def __getitem__(self, step: int) -> dict[str, np.ndarray]:
        return read_params(self._steps[step])

    def __iter__(self) -> Iterator[int]:
        return iter(self._steps)

    def __len__(self) -> int:
        return len(self._steps)


def read_params(checkpoint: str) -> dict[str, np.ndarray]:
    """Every array under ``params/`` in a checkpoint step directory, by its checkpoint path, read through the
    checkpoint's manifest (no model config or template needed)."""
    manifest = read_manifest(checkpoint)
    if manifest is None:
        raise SystemExit(f"refusing: {checkpoint} has no checkpoint manifest (give a step-N directory)")
    params = {}
    for entry in manifest.arrays:
        if not entry.path.startswith(PARAMS_PREFIX):
            continue
        spec = {
            "driver": ARRAY_DRIVER,
            "kvstore": {"driver": KVSTORE_DRIVER, "base": build_kvstore_spec(checkpoint), "path": entry.path},
        }
        params[entry.path] = np.asarray(ts.open(spec, read=True).result().read().result())
    if not params:
        raise SystemExit(f"refusing: {checkpoint} has no {PARAMS_PREFIX} arrays")
    return params


def write_tiny_checkpoint(directory: Path) -> Path:
    """A tiny fast_track model's parameters, written as training writes a checkpoint (``params/...`` arrays and a
    manifest)."""
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(tiny_config(), key=jax.random.key(0))
        tree_serialize_leaves_tensorstore(str(directory), {"params": eqx.filter(model, eqx.is_array)})
    return directory


def load_analyze(script: Path) -> Callable[[Checkpoints, list[dict], Path], None]:
    spec = importlib.util.spec_from_file_location(f"belay_study_{script.stem}", script)
    if spec is None or spec.loader is None:
        raise SystemExit(f"refusing: cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    analyze = getattr(module, "analyze", None)
    if not callable(analyze):
        raise SystemExit(f"refusing: {script} defines no analyze(checkpoints, history, out_dir)")
    return analyze


def run_analysis(script: Path, checkpoints: str, history: list[dict], out_dir: Path) -> list[Path]:
    """Run the script on a run's checkpoints and W&B history into ``out_dir``; returns the report files. Refuses an
    empty report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    load_analyze(script)(Checkpoints(checkpoint_steps(checkpoints)), history, out_dir)
    report = out_dir / REPORT_FILE
    if not report.exists() or not report.read_text().strip():
        raise SystemExit(f"refusing: {script.name} wrote no {REPORT_FILE}")
    return sorted(p for p in out_dir.iterdir() if p.suffix in REPORT_SUFFIXES)


# The selftest's synthetic history: rows a short run logs, with a training loss and one per-layer statistic.
SELFTEST_HISTORY = [
    {"_step": step, "train/loss": 6.0 - step / 100, "train/layer_out/rms_L0": 1.0 + step / 1000}
    for step in range(0, 201, 100)
]


def selftest(script: Path) -> list[Path]:
    """The gate's CPU test: the script on a tiny synthetic run (checkpoints at two steps, a short history) must write
    a report."""
    with tempfile.TemporaryDirectory() as work:
        checkpoints = Path(work) / "checkpoints"
        for step in (100, 200):
            write_tiny_checkpoint(checkpoints / f"step-{step}")
        files = run_analysis(script, str(checkpoints), SELFTEST_HISTORY, Path(work) / "report")
        return [Path(f.name) for f in files]


def upload(files: list[Path], destination: str) -> None:
    for file in files:
        with fsspec.open(f"{destination.rstrip('/')}/{file.name}", "wb") as out:
            out.write(file.read_bytes())


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--script", type=Path, required=True)
    run.add_argument("--input", required=True, help="a run's checkpoints directory, or one step-N directory in it")
    run.add_argument("--history", required=True, help="the run's W&B history: a JSON list of logged rows")
    run.add_argument("--out", required=True, help="where the report files go")
    sub.add_parser("selftest").add_argument("--script", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "selftest":
        print(f"ok: {args.script} wrote {', '.join(str(f) for f in selftest(args.script))} on a tiny checkpoint")
        return
    with tempfile.TemporaryDirectory() as work:
        with fsspec.open(args.history) as f:
            history = json.load(f)
        files = run_analysis(args.script, args.input, history, Path(work))
        upload(files, args.out)
    print(f"wrote {', '.join(f.name for f in files)} to {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
