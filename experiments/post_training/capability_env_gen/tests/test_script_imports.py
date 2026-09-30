import os
import subprocess
import sys
from pathlib import Path


def test_controller_scripts_are_not_shadowed_by_another_checkout(tmp_path):
    root = Path(__file__).resolve().parents[1]
    foreign = tmp_path / "scripts"
    foreign.mkdir()
    (foreign / "__init__.py").write_text("raise RuntimeError('foreign scripts imported')\n")
    result = subprocess.run(
        [sys.executable, "-c",
         "import scripts.capture_task_images as module; print(module.__file__)"],
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": os.pathsep.join((str(root), str(tmp_path))),
                          "PYTHONSAFEPATH": "1"},
        capture_output=True, text=True, check=True,
    )
    assert Path(result.stdout.strip()) == root / "scripts/capture_task_images.py"
