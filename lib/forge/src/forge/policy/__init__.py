# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The prompts and settings of a run as one flat mapping from dotted key to value.

The prompt file ``prompts/<group>/<name>.md`` is the key ``<group>.<name>``; ``policy.json`` holds the settings. A
prompt file ends with one newline, which the loader drops; templates render with ``str.format``.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated

from pydantic import PlainSerializer

from forge.model.content_hash import digest

PROMPTS_DIR = "prompts"
SETTINGS_FILE = "policy.json"
PROMPT_SUFFIX = ".md"

type Setting = bool | int | float | str | tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Prompt:
    """A template loaded from a prompt file: ``template`` is the file's text without its final newline."""

    template: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.template.encode()).hexdigest()

    def render(self, **fields: object) -> str:
        return self.template.format(**fields)


@dataclass(frozen=True, slots=True)
class Policy:
    """Every prompt and setting of a run, by dotted key; a JSON list setting is held as a tuple, so no caller can
    change a setting of a shared policy through the value it read."""

    entries: Annotated[Mapping[str, Prompt | Setting], PlainSerializer(dict)]

    def __post_init__(self) -> None:
        frozen = {key: tuple(value) if isinstance(value, list) else value for key, value in self.entries.items()}
        object.__setattr__(self, "entries", MappingProxyType(frozen))

    def prompt(self, key: str) -> Prompt:
        return self._entry(key, Prompt)

    def setting[T](self, key: str, kind: type[T]) -> T:
        return self._entry(key, kind)

    def _entry[T](self, key: str, kind: type[T]) -> T:
        value = self.entries[key]
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            raise ValueError(f"policy entry {key} is {value!r}, not a {kind.__name__}")
        return value

    def stage_digest(self, prefix: str) -> str:
        """The sha256 of the sorted entries under ``prefix``: a prompt by its template's sha256, a setting by value."""
        under = {key: value for key, value in self.entries.items() if key.startswith(f"{prefix}.")}
        return digest(
            {key: value.sha256 if isinstance(value, Prompt) else value for key, value in sorted(under.items())}
        )

    def overlaid(self, *overlays: Mapping[str, object]) -> "Policy":
        """This policy with each overlay's values in place, in order; a string replaces a prompt's template.

        Raises:
            ValueError: an overlay names a key this policy lacks, or gives a key a value of another type.
        """
        entries = dict(self.entries)
        for overlay in overlays:
            for key, value in overlay.items():
                if key not in entries:
                    raise ValueError(f"the policy has no entry {key}")
                old = entries[key]
                new = Prompt(value) if isinstance(old, Prompt) and isinstance(value, str) else value
                new = float(new) if type(old) is float and type(new) is int else new
                new = tuple(new) if isinstance(new, list) else new
                if type(new) is not type(old):
                    raise ValueError(f"policy entry {key} is a {type(old).__name__}, not {value!r}")
                entries[key] = new
        return Policy(entries)


def load_policy(defaults_dir: Path) -> Policy:
    """The prompts under ``defaults_dir/prompts`` and the settings of ``defaults_dir/policy.json``.

    Raises:
        ValueError: a prompt file does not end with a newline, or a setting shares a prompt's key.
    """
    prompts: dict[str, Prompt | Setting] = {}
    for path in sorted((defaults_dir / PROMPTS_DIR).rglob(f"*{PROMPT_SUFFIX}")):
        text = path.read_text()
        if not text.endswith("\n"):
            raise ValueError(f"{path} does not end with a newline")
        prompts[".".join(path.relative_to(defaults_dir / PROMPTS_DIR).with_suffix("").parts)] = Prompt(text[:-1])
    settings = json.loads((defaults_dir / SETTINGS_FILE).read_text())
    shared = sorted(set(prompts) & set(settings))
    if shared:
        raise ValueError(f"{defaults_dir / SETTINGS_FILE} sets prompt keys: {', '.join(shared)}")
    return Policy(prompts | settings)


def parse_overlay(text: str) -> dict[str, object]:
    """A driver's ``--policy`` argument: a JSON object from dotted key to value."""
    overlay = json.loads(text)
    if not isinstance(overlay, dict):
        raise ValueError(f"a policy overlay is a JSON object, not {text!r}")
    return overlay


def resolution_lines(declared: Policy, resolved: Policy) -> list[str]:
    """``POLICY <key> declared=<value> resolved=<value>`` for every entry, a prompt shown by its sha256 prefix."""

    def shown(value: Prompt | Setting) -> str:
        return f"sha256:{value.sha256[:12]}" if isinstance(value, Prompt) else json.dumps(value)

    return [
        f"POLICY {key} declared={shown(declared.entries[key])} resolved={shown(value)}"
        for key, value in sorted(resolved.entries.items())
    ]
