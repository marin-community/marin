"""Patch TPUModelRunner to add supports_mm_inputs attribute.

The Ray executor's execute_model_ray checks worker.model_runner.supports_mm_inputs
but TPUModelRunner doesn't define it. Add it as False (TPU doesn't support MM yet).
"""

import os

BASE = "/workspace/tpu_inference/tpu_inference"
PATH = os.path.join(BASE, "runner/tpu_runner.py")

with open(PATH) as f:
    code = f.read()

# Add supports_mm_inputs after mesh init
target = '        logger.info(f"Init mesh | mesh={self.mesh}")'
if target in code:
    code = code.replace(
        target,
        target + "\n        self.supports_mm_inputs = False  # TPU doesn't support MM inputs via Ray yet"
    )
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED tpu_runner.py: added supports_mm_inputs = False")
else:
    print("SKIP tpu_runner.py: target pattern not found")
