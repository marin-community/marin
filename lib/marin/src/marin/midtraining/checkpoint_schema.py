# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed validators that a Levanter native checkpoint matches the
model class declared by a midtraining spec.

Background (2026-05-27 root-cause):

The materialization helper (``scripts/materialize_delphi_prefix_checkpoint.py``)
silently decoded untyped ``model:`` payloads in old Delphi ``.executor_info``
files as ``type: llama``. That built a Llama param tree (no ``q_norm`` /
``k_norm`` slots). When the loader pulled a Qwen3-native checkpoint into
that smaller tree, the on-disk q/k RMSNorm arrays were silently dropped;
training proceeded for thousands of steps under the wrong (no-QK-norm)
forward pass; the written checkpoint looked complete by file-presence
checks (``manifest.ocdbt``, ``metadata.json``, ``d/``) but its OCDBT
kvstore was missing 12+ keys per layer. The failure only surfaced when a
downstream cooldown launcher tried to load the result with the correct
``type: qwen3`` config — by which point ~6 cells had already been launched
and killed cleanly with confusing ``q_norm``-restore errors.

These helpers catch that bug class at two new points:

- **Before launch** (preflight): enumerate OCDBT keys of the staged source
  checkpoint and assert the expected class-specific arrays are present.
- **After save** (launcher post-train hook): same check on the final
  cooled-down checkpoint, in case anything in the load → train → save
  pipeline silently degraded the param tree.

Key listing goes through ``tensorstore`` so it works on any Levanter
checkpoint regardless of region or driver. Tests inject a fake
``list_keys`` callable to avoid touching GCS.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Final

logger = logging.getLogger(__name__)

#: Substrings that identify Qwen3's per-attention-layer QK-norm scale vectors in
#: the OCDBT keyspace. Levanter writes them as ``...attn.q_norm.weight`` and
#: ``...attn.k_norm.weight`` (path may be dotted or slashed depending on the
#: pytree leaf-path encoder; substring match is robust to either).
_QWEN3_Q_NORM_MARKER: Final[str] = "q_norm"
_QWEN3_K_NORM_MARKER: Final[str] = "k_norm"

#: Model types whose param tree is a strict superset of Llama. The bug class
#: in the docstring depends on this asymmetry: silently degrading the type
#: drops these arrays. Add new entries here as more architectures are checked
#: against Delphi-class trees.
_MODEL_TYPES_WITH_QK_NORM: Final[frozenset[str]] = frozenset({"qwen3"})


ListCheckpointKeys = Callable[[str], tuple[str, ...]]


def default_list_checkpoint_keys(checkpoint_dir: str) -> tuple[str, ...]:
    """List the OCDBT kvstore entries inside ``<checkpoint_dir>/`` via tensorstore.

    The returned keys are decoded UTF-8 paths into the checkpoint's
    pytree, e.g.
    ``model.layers.0.self_attn.q_norm.weight/.zarray``. Both the metadata
    keys and array-chunk keys are returned; callers should substring-match
    rather than try to reconstruct exact pytree paths.

    Requires:
        - ``tensorstore`` installed (it is, via Levanter)
        - GCS read access if ``checkpoint_dir`` is a ``gs://`` URI

    Raises:
        RuntimeError: if the kvstore cannot be opened. The caller should
            surface the failure rather than silently treating it as "no
            keys" — that would re-introduce the silent-degradation pattern.
    """
    import tensorstore as ts

    base = checkpoint_dir.rstrip("/") + "/"
    spec = {"driver": "ocdbt", "base": base}
    try:
        kvstore = ts.KvStore.open(spec).result()
        keys = kvstore.list().result()
    except Exception as exc:
        raise RuntimeError(
            f"Could not list OCDBT keys under {checkpoint_dir!r}; "
            "checkpoint may be unreachable or missing manifest.ocdbt. "
            "Refusing to proceed silently — see "
            "lib/marin/src/marin/midtraining/checkpoint_schema.py."
        ) from exc
    return tuple(sorted(k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k) for k in keys))


def assert_qwen3_qk_norm_present(
    checkpoint_dir: str,
    *,
    num_layers: int,
    list_keys: ListCheckpointKeys = default_list_checkpoint_keys,
) -> None:
    """Fail closed if a checkpoint that should be Qwen3 is missing QK-norm arrays.

    ``num_layers`` is taken from the base-model registry (e.g.
    ``DelphiModel.num_layers``) so we can give an actionable error that
    states how many arrays we expected. We don't require an exact key
    count because OCDBT may shard a single logical array across multiple
    storage keys and the (metadata + chunk) key count per array depends
    on the driver. We only require that the substring ``q_norm`` and the
    substring ``k_norm`` each appear in at least one OCDBT key — the bug
    class produces zero such keys, never an intermediate count.

    Raises:
        ValueError: with the exact missing-marker(s) named, instructing
            the operator to re-materialize the checkpoint with the
            Qwen3-fixed helper.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers!r}")
    keys = list_keys(checkpoint_dir)
    q_norm_hits = sum(1 for k in keys if _QWEN3_Q_NORM_MARKER in k)
    k_norm_hits = sum(1 for k in keys if _QWEN3_K_NORM_MARKER in k)

    missing_markers: list[str] = []
    if q_norm_hits == 0:
        missing_markers.append(_QWEN3_Q_NORM_MARKER)
    if k_norm_hits == 0:
        missing_markers.append(_QWEN3_K_NORM_MARKER)

    if missing_markers:
        raise ValueError(
            f"Checkpoint {checkpoint_dir!r} is missing Qwen3 QK-norm arrays "
            f"({', '.join(missing_markers)}). Total OCDBT keys: {len(keys)}; "
            f"keys matching 'q_norm': {q_norm_hits}, matching 'k_norm': {k_norm_hits}; "
            f"expected at least {num_layers} of each. "
            "This is the silent-type-degradation bug class (commit 5afac0bdf): "
            "the checkpoint was almost certainly produced from a Llama-decoded "
            "config that dropped the Qwen3 q_norm/k_norm arrays at load time. "
            "Re-materialize with scripts/materialize_delphi_prefix_checkpoint.py "
            "(the helper now fails closed unless source decodes as Qwen3Config), "
            "and update the candidate row in "
            "experiments/midtrain_specs/true_midtrain/nemotron_math_only/configs/"
            "checkpoint_candidates.yaml to point at the new path."
        )


def assert_checkpoint_complete_for_model_type(
    checkpoint_dir: str,
    *,
    model_type: str,
    num_layers: int,
    list_keys: ListCheckpointKeys = default_list_checkpoint_keys,
) -> None:
    """Dispatch on declared model type and assert class-specific invariants.

    Currently only Qwen3 has a class-specific check (the QK-norm scale arrays).
    Other declared types pass through without further key validation, but
    the dispatch still verifies the type string is non-empty so a missing
    type discriminator on the spec is itself a failure.
    """
    if not model_type:
        raise ValueError(
            f"assert_checkpoint_complete_for_model_type requires a non-empty model_type; got {model_type!r}. "
            "An empty discriminator is the same bug class — downstream decoders silently fall back to a default."
        )
    if model_type in _MODEL_TYPES_WITH_QK_NORM:
        assert_qwen3_qk_norm_present(checkpoint_dir, num_layers=num_layers, list_keys=list_keys)
    else:
        logger.info(
            "checkpoint_schema: no class-specific OCDBT key check registered for model_type=%r; "
            "verified only that the discriminator was set. Add an assertion here when a new "
            "architecture with extra-arrays-vs-Llama is used.",
            model_type,
        )
