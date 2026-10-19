"""Canonical identities for resumable evaluation requests.

``from_run_inputs`` hashes controlling files, resolves available model revisions
and normalizes decoding aliases. ``RunFingerprint`` excludes operational settings
and provenance, then hashes the canonical JSON payload. Source and request hashes
validate replayed items independently of that provenance.

This module uses only the Python standard library.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

SCHEMA_VERSION = 1

# from_run_inputs resolves these material fields, hashing controlling files
# before constructing the canonical payload.

# IN the hashed set — any change must refuse a resume (material delta).
MATERIAL_FIELDS = (
    "rendered_config",        # the rendered eval config (resolved, not the raw CLI)
    "template_digest",        # rendered chat-template string hash
    "apply_chat_template",    # bool — template on/off is material
    "grader_source_digest",   # sha256 of the grader source
    "grader_version",         # pinned grader version tag
    "model_repo",             # HF repo id
    "model_revision",         # HF commit / revision
    "task_name",
    "task_data_digest",       # sha256 of the loaded data file (debug [:2] slice => different digest)
    "temperature",
    "top_p",
    "max_tokens",             # incl. max_new_tokens / max_gen_toks (normalized in from_run_inputs)
    "do_sample",
    "n",
    "num_samples",
    "seed_set",               # full seed tuple; AIME24 n_repeat + per-repeat derivation
    "max_model_len",
    "num_fewshot",
    "passk_batch_size",       # pass@k problem-batch size
)

# Structurally EXCLUDED from the hashed set — operational settings and provenance.
# Replayed source and request hashes enforce logical identity across code/package updates.
# (TP / world_size is excluded here for comparability; the FineStore manager checks
#  world_size separately before restoring rank-local request rows.)
IGNORED_FIELDS = (
    "implementation",
    "dependencies",
    "job_name",
    "run_tag",
    "started_at",
    "wall_clock",
    "slurm_job_id",
    "batch_size",
    "max_batch_size",
    "gpu_memory_utilization",
    "tensor_parallel_size",
    "world_size",
    "rank",
    "verbosity",
    "log_level",
)

# Decoding-param ALIASES -> the canonical MATERIAL key. lm-eval, vLLM and the chat_benchmarks use
# different spellings for the same generation cap (`max_gen_toks` in lm-eval gen_kwargs,
# `max_new_tokens` in the chat_benchmarks, `max_tokens` in the OpenAI-style sampling params). They
# are the SAME material input, so normalize to one key — two configs that differ only by alias
# spelling must produce the identical fingerprint.
_DECODING_ALIASES = {
    "max_gen_toks": "max_tokens",
    "max_new_tokens": "max_tokens",
    "max_tokens": "max_tokens",
}


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def digest_file(path: Optional[str | Path]) -> Optional[str]:
    """Content-hash a single controlling FILE (task data, grader source).

    Returns ``None`` when ``path`` is ``None`` (the input is genuinely absent for this run),
    so the field simply does not enter the hashed set. A path that is given but missing on disk
    is a hard error — silently treating a missing controlling file as "no input" would let a
    resume validate against a different effective config (fail-safe toward raising, not toward a
    false resume). Reads in binary so the digest is encoding-agnostic and byte-exact.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"controlling file for fingerprint not found: {p}")
    return _sha256_bytes(p.read_bytes())


def digest_files(paths: Optional[Iterable[str | Path]]) -> Optional[str]:
    """Content-hash an ordered SET of controlling files as one digest (e.g. a multi-file grader).

    Full paths determine sort order. Each file contributes its basename and raw
    content bytes, each followed by a NUL separator. The digest changes when the
    file set or contents change, independently of the caller's iteration order.
    """
    if paths is None:
        return None
    items = sorted(Path(p) for p in paths)
    if not items:
        return None
    h = hashlib.sha256()
    for p in items:
        if not p.is_file():
            raise FileNotFoundError(f"controlling file for fingerprint not found: {p}")
        h.update(p.name.encode("utf-8"))
        h.update(b"\x00")
        h.update(p.read_bytes())
        h.update(b"\x00")
    return "sha256:" + h.hexdigest()


def normalize_decoding(decoding: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Collapse decoding-param aliases onto the canonical MATERIAL keys.

    - ``max_gen_toks`` / ``max_new_tokens`` / ``max_tokens`` -> ``max_tokens``;
    - everything else (``temperature``, ``top_p``, ``do_sample``, ``n``, ``num_samples``) passes
      through under its own MATERIAL name.

    Conflicting aliases (e.g. ``max_gen_toks=64`` AND ``max_tokens=128`` in one config) raise —
    an ambiguous material input must not be silently coalesced.
    """
    out: Dict[str, Any] = {}
    if not decoding:
        return out
    seen_max: Dict[str, Any] = {}
    for k, v in decoding.items():
        if v is None:
            continue
        if k in _DECODING_ALIASES:
            canon = _DECODING_ALIASES[k]
            if canon in seen_max and seen_max[canon] != v:
                raise ValueError(
                    f"conflicting decoding aliases for {canon!r}: {seen_max[canon]!r} vs {v!r} "
                    f"(from {k!r}) — cannot canonicalize an ambiguous material input"
                )
            seen_max[canon] = v
            out[canon] = v
        else:
            out[k] = v
    return out


@dataclass
class RunFingerprint:
    """Resolved run inputs + the canonical-payload / content-hash machinery.

    ``inputs`` holds the RESOLVED MATERIAL set (files already content-hashed into ``*_digest``
    keys, decoding aliases already normalized). Use :meth:`from_run_inputs` to build one from raw
    run inputs; the bare constructor is for tests / round-tripping a stored ``fingerprint.json``.
    """

    inputs: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def _canonical_payload(self) -> Dict[str, Any]:
        """Strip the cosmetic (IGNORED) fields; keep everything else, sorted by key.

        New material fields enter the hash by default; only explicitly ignored
        operational settings and provenance are excluded.
        """
        ignored = set(IGNORED_FIELDS)
        return {k: self.inputs[k] for k in sorted(self.inputs) if k not in ignored}

    def value(self) -> str:
        """Return a single SHA-256 digest over the canonical payload."""
        payload = json.dumps(self._canonical_payload(), sort_keys=True, separators=(",", ":"))
        return _sha256_text(payload)

    def to_json(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "value": self.value(),
            "canonical_payload": self._canonical_payload(),
        }

    def matches(self, other: "RunFingerprint") -> bool:
        return self.value() == other.value()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RunFingerprint):
            return NotImplemented
        return self.matches(other)

    def __hash__(self) -> int:
        return hash(self.value())

    def diff_material(self, other: "RunFingerprint") -> Dict[str, Any]:
        """Material fields that changed vs `other` (for the refuse error message)."""
        a, b = self._canonical_payload(), other._canonical_payload()
        changed: Dict[str, Any] = {}
        for k in set(a) | set(b):
            if a.get(k) != b.get(k):
                changed[k] = {"this": a.get(k), "other": b.get(k)}
        return changed

    @classmethod
    def from_run_inputs(
        cls,
        *,
        model_repo: Optional[str] = None,
        model_revision: Optional[str] = None,
        task_name: Optional[str] = None,
        task_data_path: Optional[str | Path] = None,
        task_data_digest: Optional[str] = None,
        grader_source_path: Optional[str | Path | Sequence[str | Path]] = None,
        grader_source_digest: Optional[str] = None,
        grader_version: Optional[str] = None,
        rendered_config: Optional[Any] = None,
        apply_chat_template: Optional[bool] = None,
        chat_template: Optional[str] = None,
        chat_template_digest: Optional[str] = None,
        decoding: Optional[Dict[str, Any]] = None,
        seed_set: Optional[Sequence[Any]] = None,
        max_model_len: Optional[int] = None,
        num_fewshot: Optional[int] = None,
        passk_batch_size: Optional[int] = None,
        extra_material: Optional[Dict[str, Any]] = None,
    ) -> "RunFingerprint":
        """Resolve raw run inputs into the curated MATERIAL set, then build the fingerprint.

        Content-hashing of the controlling FILES:
          - ``task_data_path``  -> ``task_data_digest`` (sha256 of the loaded data file; a debug
            ``[:2]`` slice that writes a different file, or a different file, changes the digest).
            A precomputed ``task_data_digest`` may be passed directly (e.g. for an in-memory HF
            dataset that has no single file — the caller hashes its serialized rows and passes the
            digest, noted in the close-out).
          - ``grader_source_path`` -> ``grader_source_digest`` (sha256 of the grader source; accepts
            a single path or a sequence for a multi-file grader). A precomputed digest may be passed.
          - ``chat_template`` (the rendered template STRING) -> ``template_digest``; or pass
            ``chat_template_digest`` directly. When ``apply_chat_template`` is false this is absent.

        Decoding ALIASES are normalized (``normalize_decoding``) before they enter the set, so
        ``max_gen_toks`` / ``max_new_tokens`` / ``max_tokens`` collapse to one key.

        Model revision is taken as given (the caller resolves the HF commit where available — see
        ``resolve_model_revision``). ``rendered_config`` is stored as-is (already-resolved dict /
        string); it is part of the hashed set so any resolved-config change refuses.
        """
        resolved: Dict[str, Any] = {}

        # --- model -------------------------------------------------------------------------
        if model_repo is not None:
            resolved["model_repo"] = model_repo
        if model_revision is not None:
            resolved["model_revision"] = model_revision

        # --- task + data digest ------------------------------------------------------------
        if task_name is not None:
            resolved["task_name"] = task_name
        data_digest = task_data_digest if task_data_digest is not None else digest_file(task_data_path)
        if data_digest is not None:
            resolved["task_data_digest"] = data_digest

        # --- grader source digest + version -----------------------------------------------
        if grader_source_digest is not None:
            resolved["grader_source_digest"] = grader_source_digest
        elif grader_source_path is not None:
            if isinstance(grader_source_path, (str, Path)):
                resolved["grader_source_digest"] = digest_file(grader_source_path)
            else:
                resolved["grader_source_digest"] = digest_files(grader_source_path)
        if grader_version is not None:
            resolved["grader_version"] = grader_version

        # --- rendered config ---------------------------------------------------------------
        if rendered_config is not None:
            resolved["rendered_config"] = rendered_config

        # --- chat template -----------------------------------------------------------------
        if apply_chat_template is not None:
            resolved["apply_chat_template"] = bool(apply_chat_template)
        if chat_template_digest is not None:
            resolved["template_digest"] = chat_template_digest
        elif chat_template is not None:
            resolved["template_digest"] = _sha256_text(chat_template)

        # --- decoding params (alias-normalized) --------------------------------------------
        for k, v in normalize_decoding(decoding).items():
            resolved[k] = v

        # --- remaining scalar material -----------------------------------------------------
        if seed_set is not None:
            resolved["seed_set"] = list(seed_set)
        if max_model_len is not None:
            resolved["max_model_len"] = max_model_len
        if num_fewshot is not None:
            resolved["num_fewshot"] = num_fewshot
        if passk_batch_size is not None:
            resolved["passk_batch_size"] = passk_batch_size

        if extra_material:
            for k, v in extra_material.items():
                if k in IGNORED_FIELDS:
                    raise ValueError(f"extra_material key {k!r} is in the IGNORED partition")
                resolved[k] = v

        return cls(inputs=resolved)


def resolve_model_revision(
    model_repo: Optional[str],
    revision: Optional[str] = None,
    *,
    allow_network: bool = False,
) -> Optional[str]:
    """Resolve an HF model revision to a pinned commit where possible.

    - If ``revision`` is already a 40-char hex commit, return it unchanged.
    - If a non-pinned ``revision`` (branch/tag, or ``None``) is given and ``allow_network`` is
      True, resolve the commit via ``huggingface_hub`` (best-effort; falls back to the given
      revision on any failure or under ``HF_HUB_OFFLINE``).
    - Otherwise return ``revision`` (or ``None``) unchanged — under the cluster's
      ``HF_HUB_OFFLINE=1`` we do NOT hit the network; the caller's pinned revision (or the model
      dir's own commit, resolved upstream) is the material input.

    Kept side-effect-light and import-lazy so the module stays pure-stdlib for unit tests.
    """
    if revision and len(revision) == 40 and all(c in "0123456789abcdef" for c in revision.lower()):
        return revision
    if not allow_network or not model_repo:
        return revision
    try:  # pragma: no cover - network path not exercised in CPU tests
        from huggingface_hub import HfApi

        return HfApi().model_info(model_repo, revision=revision).sha or revision
    except Exception:
        return revision
