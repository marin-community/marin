#!/usr/bin/env python3
from marin.run.ray_run import parse_pip_requirements


def test_parse_pip_requirements():
    """
    Test that parse_pip_requirements properly handles extras and pinned packages.
    """
    assert parse_pip_requirements("numpy,scipy,sympy") == ["numpy", "scipy", "sympy"]
    assert parse_pip_requirements("numpy,scipy[extras1,extras2],sympy") == ["numpy", "scipy[extras1,extras2]", "sympy"]
    assert parse_pip_requirements("numpy,scipy==1.8.0,sympy") == ["numpy", "scipy==1.8.0", "sympy"]
    assert parse_pip_requirements("numpy==2.0.0,scipy[extras1,extras2],sympy") == [
        "numpy==2.0.0",
        "scipy[extras1,extras2]",
        "sympy",
    ]
    assert parse_pip_requirements(
        "numpy==2.0.0,datatrove[io,processing] @ git+https://github.com/nelson-liu/datatrove@tqdm_loggable,scipy"
    ) == [
        "numpy==2.0.0",
        "datatrove[io,processing] @ git+https://github.com/nelson-liu/datatrove@tqdm_loggable",
        "scipy",
    ]
