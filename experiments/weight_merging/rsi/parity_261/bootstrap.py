"""Patch only the job-private pinned TPU environment, then dispatch its entrypoint."""

import hashlib
import importlib.util
import runpy
import sys
from pathlib import Path

from patch_v4_tiles import patch

HERE = Path(__file__).resolve().parent
root = Path(importlib.util.find_spec("tpu_inference").origin).parent
patch(root / "kernels/ragged_paged_attention/v3/kernel.py")
runpy.run_path(str(HERE / "trace_bootstrap.py"), run_name="task261_patch_only")
target = root / "worker/tpu_worker.py"
source = target.read_text()
needle = "        self.model_runner.load_model()\n"
replacement = (
    needle
    + '        print("TASK261_POST_LOAD_HBM", [(str(d), d.memory_stats()) for d in jax.local_devices()], flush=True)\n'
)
pristine = source.replace(replacement, needle)
if hashlib.sha256(pristine.encode()).hexdigest() != "3d7676f197cfe16c37ac0f8758932e0aa2a975f01e38ef129a22465981165bdd":
    raise RuntimeError("Unexpected TPU worker source; refusing HBM instrumentation")
if source != pristine.replace(needle, replacement):
    assert source == pristine and source.count(needle) == 1
    target.write_text(pristine.replace(needle, replacement))

if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    if command == "python":
        path, *arguments = arguments
        sys.argv = [path, *arguments]
        runpy.run_path(path, run_name="__main__")
    elif command == "vllm":
        from vllm.entrypoints.cli.main import main

        sys.argv = ["vllm", *arguments]
        main()
    else:
        raise ValueError(f"Unsupported private entrypoint {command}")
