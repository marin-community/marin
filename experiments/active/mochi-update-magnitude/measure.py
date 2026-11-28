# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure per-update weight movement between Mochi checkpoints.

Reads the fixed sample (``registry.SAMPLE``) from both sides of every pair in
``registry.PAIRS``, computes per-tensor and per-group movement statistics, and
writes JSONL rows plus a summary to the output prefix. Checkpoint access is
read-only; nothing is copied through the Mac.

Usage (in region, CPU job):
  python experiments/active/mochi-update-magnitude/measure.py \
      --pairs sft-exp3-0to2100 --out s3://.../results
Smoke test (one tiny tensor, local):
  ... --pairs sft-hero-0to6916 --limit-tensors 2 --out /tmp/mum-results
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import posixpath
import time
from dataclasses import dataclass

import fsspec
import numpy as np
import registry
from megatron_map import LEVANTER_SHAPES, build_mapping, qkv_row_index
from readers import (
    LevanterReader,
    MegatronInventory,
    SafetensorsReader,
    configure_object_storage,
    load_megatron_block,
)
from registry import (
    CHECKPOINTS,
    LEVANTER,
    MEGATRON,
    OUTPUT_PREFIX,
    SAFETENSORS,
    SAMPLE,
    Checkpoint,
    Pair,
    Sample,
)

logger = logging.getLogger("mochi-update-magnitude")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def bf16_round(a: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (nearest-even) and back, matching HF exports."""
    u = a.astype(np.float32).view(np.uint32)
    rounded = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


@dataclass
class SliceStats:
    elements: int
    sum_w2: float
    sum_d2: float
    median_rel_elem: float
    frac_bf16_changed: float
    rel_sample: np.ndarray  # strided subsample of |delta| / (|W| + eps)


def slice_stats(a: np.ndarray, b: np.ndarray, lands: str, max_sample: int = 400_000) -> SliceStats:
    a = np.ascontiguousarray(a, dtype=np.float32).ravel()
    b = np.ascontiguousarray(b, dtype=np.float32).ravel()
    if a.shape != b.shape:
        raise ValueError(f"element-aligned shapes differ: {a.shape} vs {b.shape}")
    if lands == "bf16":
        a = bf16_round(a)
        b = bf16_round(b)
    delta = b.astype(np.float64) - a.astype(np.float64)
    sum_w2 = float(np.square(a.astype(np.float64)).sum())
    sum_d2 = float(np.square(delta).sum())
    changed = bf16_round(a) != bf16_round(b)
    frac_changed = float(np.count_nonzero(changed)) / a.size if a.size else 0.0
    abs_w = np.abs(a)
    abs_delta = np.abs(delta.astype(np.float32))
    stride = max(1, a.size // max_sample)
    rel_sample = abs_delta[stride - 1 :: stride] / (abs_w[stride - 1 :: stride] + np.float32(1e-30))
    return SliceStats(
        elements=int(a.size),
        sum_w2=sum_w2,
        sum_d2=sum_d2,
        median_rel_elem=float(np.median(rel_sample)) if rel_sample.size else 0.0,
        frac_bf16_changed=frac_changed,
        rel_sample=rel_sample,
    )


# ---------------------------------------------------------------------------
# Fixed sample as read requests
# ---------------------------------------------------------------------------

# Levanter tensor name -> (HF export name template, needs transpose).
# HF nn.Linear stores [out, in]; Levanter Einsum stores [in, out].
HF_MAP = {
    "token_embed": ("model.embed_tokens.weight", False),
    "output_proj": ("lm_head.weight", True),
    "embed_norm/weight": ("model.embed_norm.weight", False),
    "final_norm/weight": ("model.norm.weight", False),
    "embed_gated_norm/w_down": ("model.embed_gated_norm.down_proj.weight", True),
    "embed_gated_norm/w_up": ("model.embed_gated_norm.up_proj.weight", False),
    "final_gated_norm/w_down": ("model.final_gated_norm.down_proj.weight", True),
    "final_gated_norm/w_up": ("model.final_gated_norm.up_proj.weight", False),
    "stacked_blocks/stacked/rms_attn/weight": ("model.layers.{L}.input_layernorm.weight", False),
    "stacked_blocks/stacked/rms_mlp/weight": ("model.layers.{L}.post_attention_layernorm.weight", False),
    "stacked_blocks/stacked/attn/w_q": ("model.layers.{L}.self_attn.q_proj.weight", True),
    "stacked_blocks/stacked/attn/w_k": ("model.layers.{L}.self_attn.k_proj.weight", True),
    "stacked_blocks/stacked/attn/w_v": ("model.layers.{L}.self_attn.v_proj.weight", True),
    "stacked_blocks/stacked/attn/w_o": ("model.layers.{L}.self_attn.o_proj.weight", True),
    "stacked_blocks/stacked/attn/attn_gate": ("model.layers.{L}.self_attn.attn_gate.weight", False),
    "stacked_blocks/stacked/attn/sconv_k/weight": ("model.layers.{L}.self_attn.sconv_k.weight", False),
    "stacked_blocks/stacked/sconv_attn/weight": ("model.layers.{L}.sconv_attn.weight", False),
    "stacked_blocks/stacked/sconv_mlp/weight": ("model.layers.{L}.sconv_mlp.weight", False),
    "stacked_blocks/stacked/attn_gated_norm/w_down": ("model.layers.{L}.attn_gated_norm.down_proj.weight", True),
    "stacked_blocks/stacked/attn_gated_norm/w_up": ("model.layers.{L}.attn_gated_norm.up_proj.weight", False),
    "stacked_blocks/stacked/mlp_gated_norm/w_down": ("model.layers.{L}.mlp_gated_norm.down_proj.weight", True),
    "stacked_blocks/stacked/mlp_gated_norm/w_up": ("model.layers.{L}.mlp_gated_norm.up_proj.weight", False),
    "stacked_blocks/stacked/mlp/router": ("model.layers.{L}.mlp.router.weight", True),
    "stacked_blocks/stacked/mlp/router_bias": ("model.layers.{L}.mlp.router.bias", False),
    "stacked_blocks/stacked/mlp/latent_norm/weight": ("model.layers.{L}.mlp.latent_norm.weight", False),
    "stacked_blocks/stacked/mlp/w_latent_down": ("model.layers.{L}.mlp.latent_down_proj.weight", True),
    "stacked_blocks/stacked/mlp/w_latent_up": ("model.layers.{L}.mlp.latent_up_proj.weight", True),
    "stacked_blocks/stacked/shared/{N}/w_gate": ("model.layers.{L}.shared_experts.{N}.gate_proj.weight", True),
    "stacked_blocks/stacked/shared/{N}/w_up": ("model.layers.{L}.shared_experts.{N}.up_proj.weight", True),
    "stacked_blocks/stacked/shared/{N}/w_down": ("model.layers.{L}.shared_experts.{N}.down_proj.weight", True),
    "stacked_blocks/stacked/mlp/expert_mlp/w_gate": ("model.layers.{L}.mlp.experts.{E}.gate_proj.weight", True),
    "stacked_blocks/stacked/mlp/expert_mlp/w_up": ("model.layers.{L}.mlp.experts.{E}.up_proj.weight", True),
    "stacked_blocks/stacked/mlp/expert_mlp/w_down": ("model.layers.{L}.mlp.experts.{E}.down_proj.weight", True),
}


@dataclass(frozen=True)
class SliceRequest:
    """One element-aligned slice to read from both sides of a pair."""

    tensor: str
    group: str
    subgroup: str
    layer: int | None
    slices: tuple  # Levanter-orientation index spec
    hf_slices: tuple  # index spec in HF orientation (pre-transpose)


@dataclass(frozen=True)
class ExpertRequest:
    """A group of expert slabs: layers [start, start+count) x experts [estart, estart+ecount)."""

    part: str  # w_gate | w_up | w_down
    layer_start: int
    layer_count: int
    expert_start: int
    expert_count: int


def simple_requests(sample: Sample) -> list[SliceRequest]:
    reqs: list[SliceRequest] = []

    def add(tensor, group, subgroup, layer=None, slices=(), hf_slices=()):
        reqs.append(SliceRequest(tensor, group, subgroup, layer, tuple(slices), tuple(hf_slices)))

    for tensor in registry.GROUPS["norm_scales"]["rms"]:
        add(tensor, "norm_scales", "rms")
    for tensor in registry.GROUPS["norm_scales"]["gated_norm_projections"]:
        if "stacked_blocks" in tensor:
            for layer in sample.layers:
                add(tensor, "norm_scales", "gated_norm_projections", layer=layer, slices=(layer,), hf_slices=())
        else:
            add(tensor, "norm_scales", "gated_norm_projections")
    for layer in sample.router_layers:
        add("stacked_blocks/stacked/mlp/router", "router", "router", layer=layer, slices=(layer,), hf_slices=())
        add(
            "stacked_blocks/stacked/mlp/router_bias",
            "router",
            "router_bias",
            layer=layer,
            slices=(layer,),
            hf_slices=(),
        )
    for layer in sample.layers:
        add(
            "stacked_blocks/stacked/attn/w_q",
            "attention",
            "w_q",
            layer=layer,
            slices=(layer, slice(0, sample.attention_rows)),
            hf_slices=(slice(0, sample.attention_rows),),
        )
        add(
            "stacked_blocks/stacked/attn/w_k",
            "attention",
            "w_k",
            layer=layer,
            slices=(layer, slice(0, 2 * sample.attention_rows)),
            hf_slices=(slice(0, 2 * sample.attention_rows),),
        )
        add(
            "stacked_blocks/stacked/attn/w_v",
            "attention",
            "w_v",
            layer=layer,
            slices=(layer, slice(0, 2 * sample.attention_rows)),
            hf_slices=(slice(0, 2 * sample.attention_rows),),
        )
        add(
            "stacked_blocks/stacked/attn/w_o",
            "attention",
            "w_o",
            layer=layer,
            slices=(layer, slice(None), slice(0, sample.attention_rows)),
            hf_slices=(slice(None), slice(0, sample.attention_rows)),
        )
        for n in (0, 1):
            for part in ("w_gate", "w_up"):
                add(
                    f"stacked_blocks/stacked/shared/{n}/{part}",
                    "shared_dense_mlp",
                    "shared_experts",
                    layer=layer,
                    slices=(layer, slice(0, sample.attention_rows)),
                    hf_slices=(slice(0, sample.attention_rows),),
                )
            add(
                f"stacked_blocks/stacked/shared/{n}/w_down",
                "shared_dense_mlp",
                "shared_experts",
                layer=layer,
                slices=(layer, slice(0, sample.attention_rows)),
                hf_slices=(slice(None), slice(0, sample.attention_rows)),
            )
        add(
            "stacked_blocks/stacked/mlp/w_latent_down",
            "shared_dense_mlp",
            "dense_latent",
            layer=layer,
            slices=(layer, slice(0, sample.attention_rows)),
            hf_slices=(slice(0, sample.attention_rows),),
        )
        add(
            "stacked_blocks/stacked/mlp/w_latent_up",
            "shared_dense_mlp",
            "dense_latent",
            layer=layer,
            slices=(layer, slice(0, sample.attention_rows)),
            hf_slices=(slice(0, sample.attention_rows),),
        )
    for layer in (0, 16, 32, 47):
        add(
            "stacked_blocks/stacked/attn/attn_gate",
            "attention_conv_gate",
            "conv_gate",
            layer=layer,
            slices=(layer, slice(0, sample.attention_rows)),
            hf_slices=(slice(0, sample.attention_rows),),
        )
        for name in (
            "stacked_blocks/stacked/attn/sconv_k/weight",
            "stacked_blocks/stacked/sconv_attn/weight",
            "stacked_blocks/stacked/sconv_mlp/weight",
        ):
            add(name, "attention_conv_gate", "conv_gate", layer=layer, slices=(layer,), hf_slices=(layer,))
    for start, width in sample.embed_row_bands:
        add(
            "token_embed",
            "embed_output",
            "token_embed",
            slices=(slice(start, start + width),),
            hf_slices=(slice(start, start + width),),
        )
        add(
            "output_proj",
            "embed_output",
            "output_proj",
            slices=(slice(0, sample.attention_rows), slice(start, start + width)),
            hf_slices=(slice(start, start + width),),
        )
    return reqs


def expert_requests(sample: Sample) -> list[ExpertRequest]:
    reqs = []
    for layer_start, layer_count in sample.expert_layer_groups:
        reqs.append(ExpertRequest("w_gate", layer_start, layer_count, sample.experts[0], len(sample.experts)))
        reqs.append(ExpertRequest("w_up", layer_start, layer_count, sample.experts[0], len(sample.experts)))
        reqs.append(ExpertRequest("w_down", layer_start, layer_count, sample.experts[0], len(sample.experts)))
    return reqs


# ---------------------------------------------------------------------------
# Side readers (return Levanter-orientation arrays)
# ---------------------------------------------------------------------------


class LevSide:
    def __init__(self, checkpoint: Checkpoint, sample: Sample):
        self.checkpoint = checkpoint
        self.reader = LevanterReader(checkpoint)
        self.sample = sample

    def read(self, req: SliceRequest) -> np.ndarray:
        return self.reader.read(req.tensor, req.slices)

    def read_experts(self, req: ExpertRequest) -> dict[tuple[int, int], np.ndarray]:
        d2, w2, d3, w3 = self.sample.expert_area
        out = {}
        for layer in range(req.layer_start, req.layer_start + req.layer_count):
            for expert in range(req.expert_start, req.expert_start + req.expert_count):
                out[(layer, expert)] = self.reader.read(
                    f"stacked_blocks/stacked/mlp/expert_mlp/{req.part}",
                    (layer, expert, slice(d2, d2 + w2), slice(d3, d3 + w3)),
                )
        return out


class HFSide:
    def __init__(self, checkpoint: Checkpoint, sample: Sample):
        self.checkpoint = checkpoint
        self.reader = SafetensorsReader(checkpoint)
        self.sample = sample
        self._cache: dict[tuple, np.ndarray] = {}

    def _hf_name(self, tensor: str, layer: int | None) -> str:
        template, _ = HF_MAP[tensor]
        name = template.replace("{L}", str(layer if layer is not None else -1))
        name = name.replace("{N}", "0")
        return name

    def _expected_stacked_shape(self, tensor: str) -> tuple[int, ...] | None:
        if tensor in LEVANTER_SHAPES:
            shape, _ = LEVANTER_SHAPES[tensor]
            return tuple(shape)
        return None

    def _read_one(
        self, tensor: str, layer: int | None, hf_slices, expected=None, expert: int | None = None
    ) -> np.ndarray:
        _, transpose = HF_MAP[tensor]
        name = self._hf_name(tensor, layer)
        if expert is not None:
            name = name.replace("{E}", str(expert))
        key = (name, hf_slices)
        if key not in self._cache:
            self._cache[key] = self.reader.read_f32(name, hf_slices)
        arr = self._cache[key]
        if expected is None and layer is not None:
            stacked = self._expected_stacked_shape(tensor)
            if stacked is not None and len(stacked) > 1:
                # Narrow the full per-layer expectation to the requested HF slice.
                narrowed = []
                for dim, size in enumerate(stacked[1:]):
                    spec = hf_slices[dim] if dim < len(hf_slices) else slice(None)
                    if isinstance(spec, int):
                        narrowed.append(1)
                    else:
                        start, stop, _ = spec.indices(size)
                        narrowed.append(stop - start)
                expected = tuple(narrowed)
        if expected is not None and arr.shape != tuple(expected):
            if tuple(arr.shape) == tuple(expected)[::-1] and len(expected) == 2:
                logger.info(
                    "orientation: %s layer %s HF shape %s is the transpose of expected %s",
                    tensor,
                    layer,
                    list(arr.shape),
                    list(expected),
                )
                transpose = not transpose
                arr = np.ascontiguousarray(arr.T)
            else:
                raise ValueError(f"{tensor} layer {layer}: HF shape {arr.shape} vs expected {expected}")
        if transpose:
            arr = np.ascontiguousarray(arr.T)
        return arr

    def read(self, req: SliceRequest) -> np.ndarray:
        tensor = req.tensor
        stacked = self._expected_stacked_shape(tensor)
        if req.layer is None and stacked is not None and len(stacked) > 1 and stacked[0] == 48:
            # Stacked tensor read whole: concatenate the per-layer HF tensors.
            per_layer = [self._read_one(tensor, layer, req.hf_slices) for layer in range(48)]
            return np.ascontiguousarray(np.stack(per_layer))
        return self._read_one(tensor, req.layer, req.hf_slices)

    def read_experts(self, req: ExpertRequest) -> dict[tuple[int, int], np.ndarray]:
        d2, w2, d3, w3 = self.sample.expert_area
        out = {}
        for layer in range(req.layer_start, req.layer_start + req.layer_count):
            for expert in range(req.expert_start, req.expert_start + req.expert_count):
                matrix = self._read_one(
                    f"stacked_blocks/stacked/mlp/expert_mlp/{req.part}",
                    layer,
                    (),
                    expected=(3072, 3072),
                    expert=expert,
                )
                out[(layer, expert)] = np.ascontiguousarray(matrix[d2 : d2 + w2, d3 : d3 + w3])
        return out


class MegatronSide:
    """RL side: reads FP32 master weights for mapped keys via chunk streaming.

    Mapping entries carry the DCP key, the Megatron global shape, a transpose
    flag (Megatron keeps the HF export orientation) and an optional fused-axis
    range (q|k|v rows; gate|up rows) selecting this canonical tensor out of a
    fused Megatron tensor.
    """

    def __init__(self, checkpoint: Checkpoint, mapping: dict):
        self.checkpoint = checkpoint
        self.mapping = mapping  # canonical tensor name -> entry
        self.sample = SAMPLE

    def _entry(self, name: str) -> dict:
        entry = self.mapping.get(name)
        if entry is None:
            raise KeyError(f"no Megatron mapping for {name}")
        return entry

    def _lev_shape(self, entry: dict) -> tuple[int, ...]:
        shape = entry["shape"]
        perm = _axis_perm(len(shape), entry["transpose"])
        fused = entry.get("fused")
        lev = []
        for p in perm:  # lev dim order follows perm
            extent = shape[p]
            if fused and p == fused["dim"] and fused.get("layout") != "qkv":
                extent = fused["stop"] - fused["start"]
            if fused and p == fused["dim"] and fused.get("layout") == "qkv":
                extent = fused["lev_extent"]
            lev.append(extent)
        return tuple(lev)

    def _plan(self, entry: dict, lev_slices: tuple) -> tuple[tuple, tuple]:
        """Translate a Levanter-orientation slice into a Megatron-orientation slice."""
        perm = _axis_perm(len(entry["shape"]), entry["transpose"])
        fused = entry.get("fused")
        lev_shape = self._lev_shape(entry)
        meg_slices: list = [slice(None)] * len(entry["shape"])
        for i, p in enumerate(perm):
            spec = lev_slices[i] if i < len(lev_slices) else slice(None)
            if isinstance(spec, int):
                spec = slice(spec, spec + 1)
            start, stop, _ = spec.indices(lev_shape[i])
            if fused and p == fused["dim"]:
                if fused.get("layout") == "qkv":
                    meg_slices[p] = slice(None)  # read the whole fused axis; gather below
                    continue
                start += fused["start"]
                stop += fused["start"]
            meg_slices[p] = slice(start, stop)
        normalized = tuple(lev_slices[i] if i < len(lev_slices) else slice(None) for i in range(len(lev_shape)))
        return tuple(meg_slices), normalized

    def read(self, req: SliceRequest) -> np.ndarray:
        entry = self._entry(req.tensor)
        meg_slices, lev_slices = self._plan(entry, req.slices)
        block = load_megatron_block(self.checkpoint, entry["key"], meg_slices)
        lev_block = np.transpose(block, _axis_perm(block.ndim, entry["transpose"]))
        fused = entry.get("fused")
        if fused and fused.get("layout") == "qkv":
            # Megatron interleaves the fused qkv rows per query group; gather the
            # requested part into natural head order along the fused output axis.
            rows = qkv_row_index(fused["part"])
            fused_axis = _axis_perm(block.ndim, entry["transpose"]).index(fused["dim"])
            index: list = [slice(None)] * block.ndim
            index[fused_axis] = rows
            lev_block = np.ascontiguousarray(lev_block[tuple(index)])
        out = lev_block[tuple(slice(None) for _ in lev_slices)]
        return np.ascontiguousarray(out.reshape(_requested_shape(lev_slices, self._lev_shape(entry))))

    def read_experts(self, req: ExpertRequest) -> dict[tuple[int, int], np.ndarray]:
        d2, w2, d3, w3 = self.sample.expert_area
        out = {}
        for layer in range(req.layer_start, req.layer_start + req.layer_count):
            for expert in range(req.expert_start, req.expert_start + req.expert_count):
                name = f"stacked_blocks/stacked/mlp/expert_mlp/{req.part}"
                out[(layer, expert)] = self.read(
                    SliceRequest(
                        name,
                        "expert_banks",
                        req.part,
                        layer,
                        (layer, expert, slice(d2, d2 + w2), slice(d3, d3 + w3)),
                        (),
                    )
                )
        return out


def _axis_perm(n_dims: int, transpose: bool) -> tuple[int, ...]:
    """Axis permutation from Megatron to Levanter orientation.

    Stacked leading dims stay in place; projection axes swap. 2-D transposed
    tensors reverse fully ([vocab, hidden] <-> [hidden, vocab]).
    """
    if not transpose:
        return tuple(range(n_dims))
    if n_dims == 2:
        return (1, 0)
    if n_dims == 3:
        return (0, 2, 1)
    if n_dims == 4:
        return (0, 1, 3, 2)
    raise ValueError(f"unsupported ndim {n_dims}")


def _requested_shape(lev_slices: tuple, lev_shape: tuple[int, ...]) -> tuple[int, ...]:
    out = []
    for i, s in enumerate(lev_slices):
        if isinstance(s, int):
            out.append(1)
        else:
            start, stop, _ = s.indices(lev_shape[i])
            out.append(stop - start)
    return tuple(out)


# Orientation-ambiguous tensors: square projections and fused gate|up halves.
# The probe measures each candidate on a <=2-update anchor pair; the correct
# orientation gives a ~1e-5 relative delta, a wrong one ~1e0.
ORIENTABLE = (
    ("stacked_blocks/stacked/attn/w_o", "transpose"),
    ("stacked_blocks/stacked/mlp/expert_mlp/w_down", "transpose"),
    ("stacked_blocks/stacked/mlp/expert_mlp/w_gate", "fused"),
    ("stacked_blocks/stacked/mlp/expert_mlp/w_up", "fused"),
    ("stacked_blocks/stacked/shared/0/w_down", "transpose"),
    ("stacked_blocks/stacked/shared/1/w_down", "transpose"),
    ("stacked_blocks/stacked/shared/0/w_gate", "fused"),
    ("stacked_blocks/stacked/shared/0/w_up", "fused"),
    ("stacked_blocks/stacked/shared/1/w_gate", "fused"),
    ("stacked_blocks/stacked/shared/1/w_up", "fused"),
)


def _flipped_entry(entry: dict, kind: str) -> dict:
    entry = copy.deepcopy(entry)
    if kind == "transpose":
        entry["transpose"] = not entry["transpose"]
    elif kind == "fused":
        start, stop = entry["fused"]["start"], entry["fused"]["stop"]
        span = stop - start
        if start == 0:
            entry["fused"]["start"], entry["fused"]["stop"] = span, 2 * span
        else:
            entry["fused"]["start"], entry["fused"]["stop"] = 0, span
    return entry


def orientation_probe(pair: Pair, sample: Sample, mapping_a, mapping_b) -> dict:
    """Resolve ambiguous orientations on a short-update anchor pair.

    Returns {canonical_name: {"kind": ..., "scores": {...}, "flipped": bool}};
    the caller applies it with ``apply_orientation``.
    """
    a_ckpt, b_ckpt = CHECKPOINTS[pair.side_a], CHECKPOINTS[pair.side_b]
    side_a = make_side(a_ckpt, sample, mapping_a)
    requests = {req.tensor: req for req in simple_requests(sample)}
    choices = {}
    for name, kind in ORIENTABLE:
        req = requests.get(name)
        if req is None:
            continue
        try:
            arr_a = side_a.read(req)
            entry = (mapping_b or mapping_a).get(name)
            scores = {}
            for label, variant in (("default", entry), ("flipped", _flipped_entry(entry, kind))):
                trial = MegatronSide(b_ckpt if CHECKPOINTS[pair.side_b].kind == MEGATRON else a_ckpt, {name: variant})
                arr_b = trial.read(req)
                stats = slice_stats(arr_a, arr_b, "as-stored")
                scores[label] = (stats.sum_d2 / stats.sum_w2) ** 0.5 if stats.sum_w2 else 0.0
            flipped = scores["flipped"] < scores["default"] * 0.5
            choices[name] = {"kind": kind, "scores": scores, "flipped": flipped}
            logger.info(
                "orientation %s (%s): default=%.3e flipped=%.3e -> %s",
                name,
                kind,
                scores["default"],
                scores["flipped"],
                "FLIP" if flipped else "keep",
            )
        except Exception as error:  # probe is best-effort
            logger.warning("orientation probe failed for %s: %s", name, error)
    return choices


def apply_orientation(mapping: dict, choices: dict) -> dict:
    mapping = copy.deepcopy(mapping)
    for name, choice in choices.items():
        if choice.get("flipped") and name in mapping:
            entry = mapping[name]
            if choice["kind"] == "transpose":
                entry["transpose"] = not entry["transpose"]
            elif choice["kind"] == "fused":
                lo, hi = entry["fused"]["start"], entry["fused"]["stop"]
                span = hi - lo
                if lo == 0:
                    entry["fused"]["start"], entry["fused"]["stop"] = span, 2 * span
                else:
                    entry["fused"]["start"], entry["fused"]["stop"] = 0, span
    return mapping


# ---------------------------------------------------------------------------
# Pair measurement
# ---------------------------------------------------------------------------


def make_side(checkpoint: Checkpoint, sample: Sample, mapping: dict | None):
    if checkpoint.kind == LEVANTER:
        return LevSide(checkpoint, sample)
    if checkpoint.kind == SAFETENSORS:
        return HFSide(checkpoint, sample)
    if checkpoint.kind == MEGATRON:
        if mapping is None:
            raise ValueError(f"Megatron checkpoint {checkpoint.id} needs a key mapping")
        return MegatronSide(checkpoint, mapping)
    raise ValueError(f"unknown checkpoint kind {checkpoint.kind}")


def measure_pair(
    pair: Pair, sample: Sample, mapping_a=None, mapping_b=None, limit_tensors: int | None = None
) -> tuple[list[dict], list[dict]]:
    @dataclass
    class _Rec:
        label: str
        group: str
        subgroup: str
        layer: int | None
        expert: int | None

    a_ckpt, b_ckpt = CHECKPOINTS[pair.side_a], CHECKPOINTS[pair.side_b]
    side_a = make_side(a_ckpt, sample, mapping_a)
    side_b = make_side(b_ckpt, sample, mapping_b)
    lands_options = ("as-stored", "bf16")

    group_acc: dict[tuple[str, str], dict] = {}
    rows: list[dict] = []

    def record(req_like, arr_a, arr_b, label):
        for lands in lands_options:
            try:
                stats = slice_stats(arr_a, arr_b, lands)
            except ValueError as error:
                rows.append({"pair": pair.id, "tensor": label, "error": f"alignment: {error}"})
                return
            rel = (stats.sum_d2 / stats.sum_w2) ** 0.5 if stats.sum_w2 else 0.0
            rows.append(
                {
                    "pair": pair.id,
                    "stage": pair.stage,
                    "updates": pair.updates,
                    "config": pair.config,
                    "lands": lands,
                    "tensor": label,
                    "group": req_like.group,
                    "subgroup": req_like.subgroup,
                    "layer": getattr(req_like, "layer", None),
                    "expert": getattr(req_like, "expert", None),
                    "elements": stats.elements,
                    "rel_norm": rel,
                    "rel_norm_per_update": rel / pair.updates if pair.updates else rel,
                    "median_rel_elem": stats.median_rel_elem,
                    "frac_bf16_changed": stats.frac_bf16_changed,
                }
            )
            acc = group_acc.setdefault(
                (req_like.group, lands),
                {
                    "sum_w2": 0.0,
                    "sum_d2": 0.0,
                    "elements": 0,
                    "frac_num": 0.0,
                    "tensors": set(),
                    "rel_elems": [],
                },
            )
            acc["sum_w2"] += stats.sum_w2
            acc["sum_d2"] += stats.sum_d2
            acc["elements"] += stats.elements
            acc["frac_num"] += stats.frac_bf16_changed * stats.elements
            acc["tensors"].add(label.split("[")[0])
            if stats.rel_sample.size:
                acc["rel_elems"].append(stats.rel_sample)

    requests = simple_requests(sample)
    if limit_tensors:
        requests = requests[:limit_tensors]
    for index, req in enumerate(requests):
        started = time.monotonic()
        try:
            arr_a = side_a.read(req)
            arr_b = side_b.read(req)
        except Exception as error:
            logger.warning("slice failed %s %s: %s", pair.id, req.tensor, error)
            rows.append({"pair": pair.id, "tensor": req.tensor, "layer": req.layer, "error": str(error)[:200]})
            continue
        record(_Rec(req.tensor, req.group, req.subgroup, req.layer, None), arr_a, arr_b, req.tensor)
        if index % 25 == 0:
            logger.info("pair %s simple %d/%d (%.1fs)", pair.id, index, len(requests), time.monotonic() - started)

    for req in expert_requests(sample):
        started = time.monotonic()
        try:
            slabs_a = side_a.read_experts(req)
            slabs_b = side_b.read_experts(req)
        except Exception as error:
            logger.warning("expert group failed %s %s: %s", pair.id, req.part, error)
            rows.append({"pair": pair.id, "tensor": f"expert_mlp/{req.part}", "error": str(error)[:200]})
            continue
        common = sorted(set(slabs_a) & set(slabs_b))
        for layer, expert in common:
            label = f"expert_mlp/{req.part}[L{layer}E{expert}]"
            record(
                _Rec(label, "expert_banks", req.part, layer, expert),
                slabs_a[(layer, expert)],
                slabs_b[(layer, expert)],
                label,
            )
        logger.info(
            "pair %s experts %s layers %d..%d: %d slabs (%.1fs)",
            pair.id,
            req.part,
            req.layer_start,
            req.layer_start + req.layer_count - 1,
            len(common),
            time.monotonic() - started,
        )

    summary_rows = []
    for (group, lands), acc in group_acc.items():
        rel = (acc["sum_d2"] / acc["sum_w2"]) ** 0.5 if acc["sum_w2"] else 0.0
        pooled = np.concatenate(acc["rel_elems"]) if acc["rel_elems"] else np.array([])
        summary_rows.append(
            {
                "pair": pair.id,
                "stage": pair.stage,
                "updates": pair.updates,
                "config": pair.config,
                "lands": lands,
                "group": group,
                "elements": acc["elements"],
                "tensors": sorted(acc["tensors"]),
                "group_rel_norm": rel,
                "group_rel_norm_per_update": rel / pair.updates if pair.updates else rel,
                "group_median_rel_elem": float(np.median(pooled)) if pooled.size else None,
                "group_frac_bf16_changed": acc["frac_num"] / acc["elements"] if acc["elements"] else None,
            }
        )
    return rows, summary_rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def write_outputs(out_dir: str, stamp: str, rows: list[dict], summary: list[dict], pairs: list[Pair]) -> tuple:
    rows_path = posixpath.join(out_dir, f"rows-{stamp}.jsonl")
    summary_path = posixpath.join(out_dir, f"summary-{stamp}.json")
    fs, _ = fsspec.url_to_fs(rows_path)
    fs.makedirs(out_dir, exist_ok=True)
    with fs.open(rows_path, "wb") as handle:
        for row in rows:
            handle.write((json.dumps(row) + "\n").encode())
    meta = {
        "sample": SAMPLE.describe(),
        "pairs": [
            {"id": p.id, "a": p.side_a, "b": p.side_b, "updates": p.updates, "config": p.config, "note": p.note}
            for p in pairs
        ],
        "summary": summary,
    }
    with fs.open(summary_path, "wb") as handle:
        handle.write(json.dumps(meta, indent=1, default=str).encode())
    return rows_path, summary_path


def load_mapping(mapping_dir: str, checkpoint: Checkpoint) -> dict:
    path = posixpath.join(mapping_dir, f"{checkpoint.id}-mapping.json")
    fs, _ = fsspec.url_to_fs(path)
    try:
        with fs.open(path, "rb") as handle:
            return json.load(handle)
    except FileNotFoundError:
        pass
    # Live-row checkpoints can roll between jobs: rebuild the mapping from the
    # checkpoint's own metadata and cache it for the rest of the job.
    logger.info("building mapping on the fly for %s", checkpoint.id)
    mapping, notes = build_mapping(MegatronInventory(checkpoint).entries())
    logger.info("mapping for %s: %s", checkpoint.id, notes[-1])
    _save_mapping(mapping_dir, checkpoint.id, mapping)
    return mapping


def _save_mapping(mapping_dir: str, checkpoint_id: str, mapping: dict) -> None:
    path = posixpath.join(mapping_dir, f"{checkpoint_id}-mapping.json")
    fs, _ = fsspec.url_to_fs(path)
    fs.makedirs(mapping_dir, exist_ok=True)
    with fs.open(path, "wb") as handle:
        handle.write(json.dumps(mapping, indent=1, default=str).encode())
    logger.info("saved mapping %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="*", default=None, help="pair ids; default: all")
    parser.add_argument("--stage", choices=["rl", "sft"], default=None)
    parser.add_argument("--out", default=posixpath.join(OUTPUT_PREFIX, "results"))
    parser.add_argument("--mapping-dir", default=posixpath.join(OUTPUT_PREFIX, "inventory"))
    parser.add_argument(
        "--limit-tensors", type=int, default=None, help="read only the first N simple slice requests (smoke tests)"
    )
    args = parser.parse_args()

    configure_object_storage()

    pairs = registry.pairs_for(args.stage, args.pairs)
    logger.info("measuring %d pairs: %s", len(pairs), [p.id for p in pairs])

    # Resolve ambiguous Megatron-side orientations on the shortest-update anchor
    # pair before measuring anything.
    anchor = next(
        (
            p
            for p in pairs
            if p.updates <= 2 and (CHECKPOINTS[p.side_a].kind == MEGATRON or CHECKPOINTS[p.side_b].kind == MEGATRON)
        ),
        None,
    )
    if anchor and not args.limit_tensors:
        logger.info("orientation probe on %s", anchor.id)
        mapping_a = (
            load_mapping(args.mapping_dir, CHECKPOINTS[anchor.side_a])
            if CHECKPOINTS[anchor.side_a].kind == MEGATRON
            else None
        )
        mapping_b = (
            load_mapping(args.mapping_dir, CHECKPOINTS[anchor.side_b])
            if CHECKPOINTS[anchor.side_b].kind == MEGATRON
            else None
        )
        choices = orientation_probe(anchor, SAMPLE, mapping_a, mapping_b)
        for side, mapping in (("a", mapping_a), ("b", mapping_b)):
            checkpoint = CHECKPOINTS[anchor.side_a] if side == "a" else CHECKPOINTS[anchor.side_b]
            if checkpoint.kind == MEGATRON and mapping:
                _save_mapping(args.mapping_dir, checkpoint.id, apply_orientation(mapping, choices))

    all_rows: list[dict] = []
    all_summary: list[dict] = []
    for pair in pairs:
        mapping_a = (
            load_mapping(args.mapping_dir, CHECKPOINTS[pair.side_a])
            if CHECKPOINTS[pair.side_a].kind == MEGATRON
            else None
        )
        mapping_b = (
            load_mapping(args.mapping_dir, CHECKPOINTS[pair.side_b])
            if CHECKPOINTS[pair.side_b].kind == MEGATRON
            else None
        )
        started = time.monotonic()
        rows, summary = measure_pair(pair, SAMPLE, mapping_a, mapping_b, args.limit_tensors)
        all_rows.extend(rows)
        all_summary.extend(summary)
        logger.info("pair %s done in %.1fs (%d rows)", pair.id, time.monotonic() - started, len(rows))

    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    rows_path, summary_path = write_outputs(args.out, stamp, all_rows, all_summary, pairs)
    logger.info("wrote %s", rows_path)
    logger.info("wrote %s", summary_path)


if __name__ == "__main__":
    main()
