"""Shared eval preset catalog.

Each preset is a flat dict of eval-run defaults (datasets, n_concurrent,
agent_parser, agent_kwargs, etc.), stored as one ``<name>.yaml`` per preset
in this package directory.
"""

from pathlib import Path

import yaml

_PRESET_DIR = Path(__file__).parent


def load_presets() -> dict[str, dict]:
    """Load every ``*.yaml`` in this package, keyed by filename stem."""
    presets: dict[str, dict] = {}
    for path in sorted(_PRESET_DIR.glob("*.yaml")):
        with path.open() as f:
            presets[path.stem] = yaml.safe_load(f)
    return presets


def get_preset(name: str) -> dict:
    """Return a single preset by name. Raises KeyError if unknown."""
    return load_presets()[name]
