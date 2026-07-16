"""Resolver for the unified per-model vLLM serve config.

Extracted from OT-Agent ``model_config/resolver.py``. The YAML data files live
in the sibling ``models/`` directory (``serve/models/<org>/<slug>.yaml``).

Resolution merge order (later wins, most-specific):
    base intrinsics -> subsystem(s)[0] -> ... -> subsystem(s)[-1] -> hardware variant

Falls back to regex patterns (``models/_patterns.yaml``) when no per-model
file exists, preserving the eval registry's size-inference defaults.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

MODEL_CONFIG_DIR = Path(__file__).resolve().parent / "models"

INTRINSIC_FIELDS = frozenset({
    "trust_remote_code",
    "hf_overrides",
    "limit_mm_per_prompt",
    "max_model_len",
    "tool_call_parser",
    "reasoning_parser",
})


def _slugify(model: str) -> tuple[str, str]:
    parts = model.split("/", 1)
    if len(parts) == 2:
        org, rest = parts
    else:
        org, rest = "_unaffiliated", model
    slug = re.sub(r"[^\w.\-]", "_", rest)
    return org, slug


def find_model_file(model: str) -> Optional[Path]:
    org, slug = _slugify(model)
    candidate = MODEL_CONFIG_DIR / org / f"{slug}.yaml"
    return candidate if candidate.is_file() else None


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _strip_internal_keys(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("model", "subsystems", "variants", "notes")}


def _resolve_patterns(model: str, subsystem: str, hardware: Optional[str]) -> dict:
    patterns_path = MODEL_CONFIG_DIR / "_patterns.yaml"
    if not patterns_path.is_file():
        return {}
    data = _load_yaml(patterns_path)
    active_profile = hardware or "default"
    for pat in data.get("patterns", []):
        regex = pat.get("match")
        if not regex:
            continue
        profiles = pat.get("profiles") or ["default"]
        if active_profile not in profiles:
            continue
        if re.search(regex, model):
            return _strip_internal_keys(
                {k: v for k, v in pat.items() if k not in ("match", "profiles", "subsystems")}
            )
    return {}


def resolve_model_config(
    model: str,
    subsystem: str = "eval",
    hardware: Optional[str] = None,
    subsystems: Optional[Sequence[str]] = None,
) -> dict:
    chain: List[str] = []
    if subsystems:
        chain = list(subsystems)
    if subsystem not in chain:
        chain.insert(0, subsystem)

    model_file = find_model_file(model)
    if model_file is None:
        return _resolve_patterns(model, subsystem, hardware)

    data = _load_yaml(model_file)
    merged = _strip_internal_keys(data)

    subs_block = data.get("subsystems", {})
    for sub in chain:
        sub_overlay = subs_block.get(sub)
        if sub_overlay:
            variant_block = sub_overlay.get("variants", {})
            sub_fields = {k: v for k, v in sub_overlay.items() if k != "variants"}
            merged = _deep_merge(merged, sub_fields)
            if hardware and hardware in variant_block:
                merged = _deep_merge(merged, variant_block[hardware])

    return merged


def load_all_model_configs(subsystem: str = "eval", hardware: Optional[str] = None) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for org_dir in sorted(MODEL_CONFIG_DIR.iterdir()):
        if not org_dir.is_dir() or org_dir.name.startswith("_") or org_dir.name.startswith("."):
            continue
        for f in sorted(org_dir.glob("*.yaml")):
            data = _load_yaml(f)
            model_id = data.get("model")
            if not model_id:
                continue
            out[model_id] = resolve_model_config(model_id, subsystem=subsystem, hardware=hardware)
    return out


__all__ = [
    "MODEL_CONFIG_DIR",
    "INTRINSIC_FIELDS",
    "resolve_model_config",
    "load_all_model_configs",
    "find_model_file",
]
