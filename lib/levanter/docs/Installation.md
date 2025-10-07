# Installation

If you are using GPUs, please follow the installation steps [here instead](Getting-Started-GPU.md).
If you're using a TPU, more complete documentation for setting that up is available [here](Getting-Started-TPU-VM.md).

## Installing as a library
You can install Levanter into your Python environment using `pip` or `uv`.

```bash
# as a dependency
uv add levanter
# or, into an environment
uv pip install levanter
# or, the old fashioned way
pip install levanter
```

Or, you can install it from source:
```bash
uv add git+https://github.com/stanford-crfm/levanter.git
# or
uv pip install git+https://github.com/stanford-crfm/levanter.git
```

## Setting up a Levanter environment
If you're running Levanter primarily from the Levanter working directory, Levanter only requires that you [install `uv`](https://docs.astral.sh/uv/getting-started/installation/).

First, pull the repo
```bash
git clone https://github.com/stanford-crfm/levanter.git
cd levanter
```

You can then run Levanter commands directly using
```bash
uv run --extra gpu <command>  # On GPU
uv run --extra tpu <command>  # On TPU
uv run <command>              # On CPU
```

For instance, run the tests with
```bash
uv run --extra gpu --extra test pytest
```

Alternatively, you can just create the environment with
```bash
uv sync --extra gpu  # add extras as needed
```

and activate the resulting venv with
```bash
source .venv/bin/activate
```

This environment will be installed in "editable mode", so you can also use this as a development environment.
