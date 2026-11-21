# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared tokenization, shifted assistant-only masking, packing, and CW S3 helpers.

Matches the E12-B Levanter TreeCache layout:
- Unpadded rows: ``input_ids`` (int32), ``loss_weights`` (float32), equal length,
  starting with BOS (128000) and ending with EOS (128001) at weight 0.0.
- Shifted mask: ``loss_weights[i] == 1.0`` iff ``input_ids[i + 1]`` belongs to a
  selected assistant span after ``<|start_header_id|>assistant<|end_header_id|>\\n``
  (token IDs ``[128006, 78191, 128007, 198]``), through and including ``<|eot_id|>`` (128009).
- Packed rows: greedy concatenation in row order without splitting rows, padded to
  ``SEQ_LEN = 262144`` with EOS (128001) at weight 0.0, written via ``SerialCacheWriter``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from levanter.store.cache import CacheMetadata, SerialCacheWriter, TreeCache
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

TOKENIZER_ID = "marin-community/marin-tokenizer-tools"
TOKENIZER_REV = "3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09"

BOS_ID = 128000
EOS_ID = 128001
START_HEADER_ID = 128006
END_HEADER_ID = 128007
EOT_ID = 128009
ASSISTANT_ID = 78191
NEWLINE_ID = 198
ASSISTANT_HEADER_IDS = [START_HEADER_ID, ASSISTANT_ID, END_HEADER_ID, NEWLINE_ID]

SEQ_LEN = 262144
EXEMPLAR = {
    "input_ids": np.zeros(0, np.int32),
    "loss_weights": np.zeros(0, np.float32),
}


def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_mask_runs(mask: list[int]) -> list[tuple[int, int]]:
    """Return ``(start, end)`` half-open index intervals where ``mask`` is 1."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def render_shifted_row(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool,
    tools: list[dict[str, Any]] | None = None,
    select_turns: str = "last",
    training_template: str | None = None,
) -> dict[str, Any]:
    """Render ``messages`` with the pinned Marin chat template and shifted assistant-only weights.

    Args:
        tokenizer: Loaded Marin ``AutoTokenizer`` (``marin-tokenizer-tools@3f6879ac``).
        messages: Chat messages list.
        enable_thinking: Explicit boolean (``True`` for ``/think``, ``False`` for ``/nothink``).
        tools: Optional OpenAI-format tools list.
        select_turns: ``"last"`` (supervise only the terminal assistant turn, as in Skill A)
            or ``"all"`` (supervise every assistant turn, as in Skill B 1- and 2-turn items).
        training_template: Optional explicit Jinja template string; if provided, verifies exact
            agreement with the tokenizer's default template under explicit ``enable_thinking``.
    """
    if select_turns not in ("last", "all"):
        raise ValueError(f"select_turns must be 'last' or 'all', got {select_turns!r}")

    template_kwargs: dict[str, Any] = {"enable_thinking": enable_thinking}
    if tools is not None:
        template_kwargs["tools"] = tools

    enc_kwargs = dict(template_kwargs)
    if training_template is not None:
        enc_kwargs["chat_template"] = training_template

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        **enc_kwargs,
    )
    ids = [int(x) for x in encoded["input_ids"]]
    raw_mask = [int(x) for x in encoded["assistant_masks"]]

    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **enc_kwargs)
    if training_template is not None:
        default_rendered = tokenizer.apply_chat_template(messages, tokenize=False, **template_kwargs)
        if rendered != default_rendered:
            raise ValueError("Training and default chat templates disagree under explicit enable_thinking")

    if not ids or ids[0] != BOS_ID:
        raise ValueError(f"Rendered row must start with BOS ({BOS_ID}), got {ids[:4]}")
    if EOS_ID in ids:
        raise ValueError(f"EOS ({EOS_ID}) appeared inside rendered conversation before boundary append")

    runs = find_mask_runs(raw_mask)
    if not runs:
        raise ValueError("Rendered conversation has zero assistant spans")

    selected_runs = [runs[-1]] if select_turns == "last" else runs
    selected_mask = [0] * len(ids)
    for start, end in selected_runs:
        if start < len(ASSISTANT_HEADER_IDS) or ids[start - len(ASSISTANT_HEADER_IDS) : start] != ASSISTANT_HEADER_IDS:
            raise ValueError(
                f"Assistant span at {start} is not preceded by assistant header {ASSISTANT_HEADER_IDS}; "
                f"got {ids[max(0, start - 4) : start]}"
            )
        if ids[end - 1] != EOT_ID:
            raise ValueError(f"Assistant span [{start}, {end}) does not end with EOT ({EOT_ID}); got {ids[end - 1]}")
        for idx in range(start, end):
            selected_mask[idx] = 1

    # Append EOS boundary token with zero weight, then apply shifted causal target convention:
    # weight[i] scores token[i + 1].
    ids.append(EOS_ID)
    selected_mask.append(0)
    weights = [float(v) for v in selected_mask[1:]] + [0.0]

    if len(ids) != len(weights):
        raise ValueError("input_ids and loss_weights length mismatch")
    if weights[-1] != 0.0 or ids[-1] != EOS_ID:
        raise ValueError("Final token must be EOS (128001) with weight 0.0")
    for start, _end in selected_runs:
        if weights[start - 1] != 1.0 or ids[start - 1] != NEWLINE_ID:
            raise ValueError(f"Expected shifted weight 1.0 at newline token before span start {start}")
    if select_turns == "last" and any(weights[: selected_runs[0][0] - 1]):
        raise ValueError("Non-terminal turn carries non-zero loss weight under select_turns='last'")

    # Round-trip decode each selected span via shifted weights.
    decoded_spans = [
        tokenizer.decode([ids[i + 1] for i in range(start - 1, end - 1) if weights[i] > 0.0], skip_special_tokens=False)
        for start, end in selected_runs
    ]
    all_weighted_targets = [ids[i + 1] for i in range(len(ids) - 1) if weights[i] > 0.0]
    expected_targets = [ids[i] for start, end in selected_runs for i in range(start, end)]
    if all_weighted_targets != expected_targets:
        raise ValueError("Shifted loss_weights target tokens do not match selected assistant spans")

    return {
        "input_ids": ids,
        "loss_weights": weights,
        "sequence_tokens": len(ids),
        "target_tokens": len(expected_targets),
        "selected_spans": [[s, e] for s, e in selected_runs],
        "rendered_text": rendered,
        "decoded_targets": decoded_spans,
    }


def pack_rows_greedy(
    rows: list[dict[str, Any]],
    *,
    seq_len: int = SEQ_LEN,
) -> list[tuple[dict[str, np.ndarray], dict[str, Any]]]:
    """Pack unpadded rows greedily in row order into fixed ``seq_len`` arrays (never splitting a row)."""
    packs: list[tuple[dict[str, np.ndarray], dict[str, Any]]] = []
    ids: list[int] = []
    weights: list[float] = []
    members: list[str] = []
    seg_lengths: list[int] = []

    def finish() -> None:
        used = len(ids)
        pad = seq_len - used
        sup_count = sum(1 for v in weights if v > 0.0)
        eff_sum = float(sum(weights))
        packs.append(
            (
                {
                    "input_ids": np.asarray(ids + [EOS_ID] * pad, dtype=np.int32),
                    "loss_weights": np.asarray(weights + [0.0] * pad, dtype=np.float32),
                },
                {
                    "rows": list(members),
                    "segment_lengths": list(seg_lengths),
                    "real_positions": used,
                    "padding_positions": pad,
                    "target_tokens": sup_count,
                    "supervised_token_count": sup_count,
                    "effective_loss_weight_sum": round(eff_sum, 4),
                },
            )
        )
        ids.clear()
        weights.clear()
        members.clear()
        seg_lengths.clear()

    for row in rows:
        row_ids = row["input_ids"]
        row_w = row["loss_weights"]
        if len(row_ids) != len(row_w):
            raise ValueError(f"Row {row['id']} length mismatch")
        if len(row_ids) > seq_len:
            raise ValueError(f"Row {row['id']} length {len(row_ids)} exceeds seq_len={seq_len}")
        if row_ids[0] != BOS_ID or row_ids[-1] != EOS_ID or row_w[-1] != 0.0:
            raise ValueError(f"Row {row['id']} violates BOS/EOS/final-weight=0 invariant")
        if ids and len(ids) + len(row_ids) > seq_len:
            finish()
        ids.extend(row_ids)
        weights.extend(row_w)
        members.append(row["id"])
        seg_lengths.append(len(row_ids))
    if ids:
        finish()
    return packs


def write_and_verify_levanter_cache(
    rows: list[dict[str, Any]],
    cache_dir: Path,
    *,
    seq_len: int = SEQ_LEN,
    loss_weight_scale: float = 1.0,
) -> dict[str, Any]:
    """Pack ``rows`` into a Levanter ``TreeCache``, verify exact array readback and per-segment mask round-trip."""
    if loss_weight_scale <= 0.0:
        raise ValueError(f"loss_weight_scale must be positive, got {loss_weight_scale}")
    if cache_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache directory: {cache_dir}")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)

    packs = pack_rows_greedy(rows, seq_len=seq_len)
    with SerialCacheWriter(
        str(cache_dir),
        EXEMPLAR,
        metadata=CacheMetadata({"input_ids_key": "input_ids", "loss_weights_key": "loss_weights"}),
    ) as writer:
        writer.write_batch([arrays for arrays, _ in packs])

    reader = TreeCache.load(str(cache_dir), EXEMPLAR).as_sync_dataset()
    if len(reader) != len(packs):
        raise ValueError(f"Readback pack count mismatch: {len(reader)} != {len(packs)}")

    scale_f32 = np.float32(loss_weight_scale)
    row_by_id = {r["id"]: r for r in rows}
    verified_segments = 0
    for idx, (want, meta) in enumerate(packs):
        got = reader[idx]
        for key in want:
            np.testing.assert_array_equal(want[key], got[key])
        got_ids = got["input_ids"]
        got_w = got["loss_weights"]
        if not np.all((got_w == 0.0) | (got_w == scale_f32)):
            raise ValueError(f"Pack {idx} has loss_weights outside {{0.0, {loss_weight_scale}}}")
        pos = 0
        for rid, slen in zip(meta["rows"], meta["segment_lengths"], strict=True):
            src = row_by_id[rid]
            seg_ids = got_ids[pos : pos + slen]
            seg_w = got_w[pos : pos + slen]
            np.testing.assert_array_equal(seg_ids, np.asarray(src["input_ids"], dtype=np.int32))
            np.testing.assert_array_equal(seg_w, np.asarray(src["loss_weights"], dtype=np.float32))
            if int(seg_ids[-1]) != EOS_ID or float(seg_w[-1]) != 0.0:
                raise ValueError(f"Pack {idx} segment {rid} missing EOS boundary with zero weight")
            pos += slen
            verified_segments += 1
        if pos < seq_len:
            if not np.all(got_ids[pos:] == EOS_ID) or not np.all(got_w[pos:] == 0.0):
                raise ValueError(f"Pack {idx} padding region corrupted")

    total_src_sup = sum(sum(1 for v in r["loss_weights"] if v > 0.0) for r in rows)
    total_pack_sup = sum(m["supervised_token_count"] for _, m in packs)
    if total_src_sup != total_pack_sup:
        raise ValueError(f"Supervised token count conservation failed: {total_src_sup} != {total_pack_sup}")

    total_src_w = float(sum(sum(r["loss_weights"]) for r in rows))
    total_pack_w = float(sum(m["effective_loss_weight_sum"] for _, m in packs))
    expected_w = total_src_sup * float(loss_weight_scale)
    if abs(total_src_w - expected_w) > 1e-2 or abs(total_pack_w - expected_w) > 1e-2:
        raise ValueError(
            f"Effective weight conservation failed: src={total_src_w}, pack={total_pack_w}, exp={expected_w}"
        )

    total_real = sum(m["real_positions"] for _, m in packs)
    total_capacity = len(packs) * seq_len
    return {
        "packed_rows": len(packs),
        "source_rows": len(rows),
        "verified_segments": verified_segments,
        "loss_weight_scale": float(loss_weight_scale),
        "target_tokens": total_src_sup,
        "supervised_token_count": total_src_sup,
        "effective_loss_weight_sum": round(total_src_w, 4),
        "real_positions": total_real,
        "padding_positions": total_capacity - total_real,
        "packing_utilization": round(total_real / total_capacity, 6) if total_capacity else 0.0,
        "target_token_density": round(total_src_sup / total_capacity, 6) if total_capacity else 0.0,
        "readback_exact": True,
        "row_manifest": [meta for _, meta in packs],
    }


def assert_cw_s3_endpoint(uri: str) -> tuple[Any, str, str]:
    """Configure CoreWeave S3 routing and assert the resolved endpoint host is ``cwobject.com``."""
    configure_coreweave_s3()
    fs, base = filesystem_for(uri)
    endpoint = (
        (getattr(fs, "client_kwargs", None) or {}).get("endpoint_url")
        or (getattr(fs, "storage_options", None) or {}).get("endpoint_url")
        or ""
    )
    host = endpoint.split("//")[-1].split("/")[0].split(":")[0]
    if not (host == "cwobject.com" or host.endswith(".cwobject.com")):
        raise RuntimeError(f"Refusing non-CoreWeave S3 endpoint for {uri}: {endpoint!r} (host={host!r})")
    return fs, base, endpoint


def upload_dir_to_cw_s3(local_dir: Path, dest_uri: str, receipt_path: Path | None = None) -> dict[str, Any]:
    """Upload ``local_dir`` to CoreWeave S3 (refusing existing prefix) and verify SHA-256 on readback."""
    fs, base, endpoint = assert_cw_s3_endpoint(dest_uri)
    if fs.exists(base):
        raise FileExistsError(f"Refusing to overwrite existing remote prefix: {dest_uri}")
    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"No files found under {local_dir}")

    for p in files:
        rel = p.relative_to(local_dir).as_posix()
        fs.put_file(str(p), f"{base}/{rel}")

    entries = []
    for p in files:
        rel = p.relative_to(local_dir).as_posix()
        local_sha = sha256_file(p)
        remote_sha = hashlib.sha256(fs.cat_file(f"{base}/{rel}")).hexdigest()
        if local_sha != remote_sha:
            raise RuntimeError(f"Remote readback SHA-256 mismatch for {rel}: {remote_sha} != {local_sha}")
        entries.append({"path": rel, "bytes": p.stat().st_size, "sha256": local_sha})

    listed = sorted(x for x in fs.find(base))
    if len(listed) != len(files):
        raise RuntimeError(f"Remote file count mismatch: listed {len(listed)} != uploaded {len(files)}")

    receipt = {
        "dest_uri": dest_uri,
        "endpoint_url": endpoint,
        "objects": len(entries),
        "bytes": sum(e["bytes"] for e in entries),
        "readback_verified": True,
        "files": entries,
    }
    if receipt_path is not None:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
