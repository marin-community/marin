# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.swe_rebench_trace import contree_pipeline


def test_tracer_injection_uploads_python_sitecustomize():
    env, files = contree_pipeline._tracer_injection()

    assert env == {"PYTHONPATH": "/pytracer"}
    assert files == {"/pytracer/sitecustomize.py": str(contree_pipeline.PYTRACER_DIR / "sitecustomize.py")}
