# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build, verify, audit, pack (Levanter TreeCache), and optionally upload Skill A and Skill B datasets.

CPU-only pipeline matching E12-B's Parquet + TreeCache layout:
  1. Generate and deterministically verify examples across train/validation/test splits until exact pack targets.
  2. Write zstd-compressed Parquet splits with ``input_ids: list<int32>`` and ``loss_weights: list<float32>``.
  3. Run standalone Parquet row-audit (``row-audit-receipt.json``).
  4. Run 8-gram / domain / tool disjointness audit against Probe 1/2/3 + optional Micro210/BFCL
     (``disjointness-receipt.json``).
  5. Pack into ``SEQ=262144`` Levanter ``TreeCache`` with exact readback & per-segment round-trip
     (``cache/packing-manifest.json``).
  6. Optionally upload ``cache/`` to CoreWeave S3 (asserting ``cwobject.com`` endpoint, non-overwrite, SHA-256 readback).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from .common import (
    ASSISTANT_HEADER_IDS,
    BOS_ID,
    EOS_ID,
    EOT_ID,
    SEQ_LEN,
    sha256_file,
    upload_dir_to_cw_s3,
    write_and_verify_levanter_cache,
)
from .decon_audit import extract_bfcl_names_and_payloads, run_disjointness_audit
from .skill_a_ordinal import generate_skill_a_example
from .skill_a_ordinal import to_parquet_row as skill_a_to_row
from .skill_b_abstention import (
    SKILL_B_CUE_PARAPHRASES,
    generate_skill_b_example,
)
from .skill_b_abstention import to_parquet_row as skill_b_to_row

# Module-level worker globals for multiprocessing speed (load tokenizer once per worker process)
_WORKER_TOK: Any = None
_WORKER_TEMPLATE: str | None = None
_WORKER_EXTRA_BLOCKED: set[str] | None = None


def _init_worker(tokenizer_path: str, template_path: str | None, extra_blocked_tools: list[str] | None) -> None:
    global _WORKER_TOK, _WORKER_TEMPLATE, _WORKER_EXTRA_BLOCKED
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _WORKER_TOK = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    _WORKER_TEMPLATE = Path(template_path).read_text(encoding="utf-8") if template_path else None
    _WORKER_EXTRA_BLOCKED = set(extra_blocked_tools) if extra_blocked_tools else None


def _gen_one_skill_a(args: tuple[str, int]) -> dict[str, Any]:
    split, idx = args
    ex = generate_skill_a_example(_WORKER_TOK, split=split, index=idx, training_template=_WORKER_TEMPLATE)
    return skill_a_to_row(ex)


def _gen_one_skill_b(args: tuple[str, int]) -> dict[str, Any]:
    split, idx = args
    ex = generate_skill_b_example(
        _WORKER_TOK,
        split=split,
        index=idx,
        training_template=_WORKER_TEMPLATE,
        extra_blocked_tools=_WORKER_EXTRA_BLOCKED,
    )
    return skill_b_to_row(ex)


def _trim_rows_to_exact_packs(
    rows: list[dict[str, Any]], target_packs: int, seq_len: int = SEQ_LEN
) -> list[dict[str, Any]]:
    """Keep the exact prefix of ``rows`` whose greedy packing produces exactly ``target_packs`` full/near-full packs."""
    selected: list[dict[str, Any]] = []
    packs_closed = 0
    cur_used = 0
    for row in rows:
        rlen = len(row["input_ids"])
        if cur_used and cur_used + rlen > seq_len:
            packs_closed += 1
            cur_used = 0
            if packs_closed >= target_packs:
                break
        selected.append(row)
        cur_used += rlen
    final_packs = packs_closed + (1 if cur_used > 0 else 0)
    if final_packs != target_packs:
        raise ValueError(f"Generated rows produced {final_packs} packs, need exact target_packs={target_packs}")
    return selected


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> str:
    table = pa.Table.from_pylist(rows)
    table = table.set_column(
        table.schema.get_field_index("input_ids"),
        "input_ids",
        pa.array([r["input_ids"] for r in rows], type=pa.list_(pa.int32())),
    )
    table = table.set_column(
        table.schema.get_field_index("loss_weights"),
        "loss_weights",
        pa.array([r["loss_weights"] for r in rows], type=pa.list_(pa.float32())),
    )
    pq.write_table(table, path, compression="zstd")
    return sha256_file(path)


def _apply_loss_weight_scale(rows: list[dict[str, Any]], scale: float) -> list[dict[str, Any]]:
    """Scale binary ``{0.0, 1.0}`` loss weights to ``{0.0, scale}`` without touching ``input_ids``."""
    if scale <= 0.0:
        raise ValueError(f"loss_weight_scale must be positive, got {scale}")
    out: list[dict[str, Any]] = []
    for r in rows:
        w = r["loss_weights"]
        if any(v not in (0.0, 1.0) for v in w):
            raise ValueError(f"{r['id']}: source loss_weights must be binary {{0.0, 1.0}} before scaling")
        scaled = [float(scale) if v > 0.0 else 0.0 for v in w]
        out.append({**r, "loss_weights": scaled})
    return out


def audit_parquet_splits(
    skill: str,
    out_dir: Path,
    tokenizer_path: Path,
    *,
    loss_weight_scale: float = 1.0,
) -> dict[str, Any]:
    """Standalone readback row-audit matching E12-B audit_parquet.py plus full shifted-mask checks."""
    if loss_weight_scale <= 0.0:
        raise ValueError(f"loss_weight_scale must be positive, got {loss_weight_scale}")
    tok = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    split_ids: dict[str, set[str]] = {}
    summary: dict[str, Any] = {}

    for split in ("train", "validation", "test"):
        ppath = out_dir / f"{split}.parquet"
        tbl = pq.read_table(ppath)
        if tbl.schema.field("input_ids").type != pa.list_(pa.int32()):
            raise ValueError(f"{split}: input_ids type is {tbl.schema.field('input_ids').type}, expected list<int32>")
        if tbl.schema.field("loss_weights").type != pa.list_(pa.float32()):
            raise ValueError(
                f"{split}: loss_weights type is {tbl.schema.field('loss_weights').type}, expected list<float32>"
            )
        rows = tbl.to_pylist()
        ids_set = {r["id"] for r in rows}
        if len(ids_set) != len(rows):
            raise ValueError(f"{split}: duplicate row IDs")
        split_ids[split] = ids_set

        seq_lens: list[int] = []
        tgt_lens: list[int] = []
        mode_ctr: collections.Counter[str] = collections.Counter()
        stratum_ctr: collections.Counter[str] = collections.Counter()
        needle_ctr: collections.Counter[str] = collections.Counter()
        band_ctr: collections.Counter[str] = collections.Counter()

        for r in rows:
            ids = r["input_ids"]
            w = r["loss_weights"]
            if len(ids) != len(w) or len(ids) > SEQ_LEN:
                raise ValueError(f"{r['id']}: invalid length {len(ids)}")
            if ids[0] != BOS_ID or ids[-1] != EOS_ID or w[-1] != 0.0:
                raise ValueError(f"{r['id']}: BOS/EOS/final-weight invariant failed")
            if EOS_ID in ids[:-1]:
                raise ValueError(f"{r['id']}: embedded EOS token inside unpadded row")
            nz = [i for i, v in enumerate(w) if v > 0.0]
            if not nz:
                raise ValueError(f"{r['id']}: zero supervised tokens")
            if any(v != 0.0 and abs(v - loss_weight_scale) > 1e-5 for v in w):
                raise ValueError(f"{r['id']}: loss_weights outside {{0.0, {loss_weight_scale}}}")
            if len(nz) != int(r["target_tokens"]):
                raise ValueError(f"{r['id']}: target_tokens column disagrees with supervised_token_count")
            if abs(sum(w) - len(nz) * loss_weight_scale) > 1e-2:
                raise ValueError(f"{r['id']}: sum(loss_weights) disagrees with scale * supervised_token_count")
            # Every contiguous run of weighted positions must start right after ASSISTANT_HEADER_IDS
            # (meaning ids[run_start - 3 : run_start + 1] == ASSISTANT_HEADER_IDS) and score EOT at its end.
            run_starts = [nz[0]] + [nz[j] for j in range(1, len(nz)) if nz[j] != nz[j - 1] + 1]
            run_ends = [nz[j - 1] for j in range(1, len(nz)) if nz[j] != nz[j - 1] + 1] + [nz[-1]]
            for rs, re_idx in zip(run_starts, run_ends, strict=True):
                if ids[rs - 3 : rs + 1] != ASSISTANT_HEADER_IDS:
                    raise ValueError(f"{r['id']}: weighted span at {rs} not aligned to assistant header")
                if ids[re_idx + 1] != EOT_ID:
                    raise ValueError(f"{r['id']}: weighted span ending at {re_idx} does not score EOT")
            if skill == "skill_a" and len(run_starts) != 1:
                raise ValueError(f"{r['id']}: Skill A must have exactly 1 supervised (terminal) assistant turn")

            # Spot-verify decoded targets contain no think markers in /nothink mode
            if r["mode"] == "nothink":
                decoded = tok.decode([ids[i + 1] for i in nz], skip_special_tokens=False)
                if "<|start_think|>" in decoded or "<|end_think|>" in decoded:
                    raise ValueError(f"{r['id']}: think marker leaked into /nothink target")

            seq_lens.append(len(ids))
            tgt_lens.append(int(r["target_tokens"]))
            mode_ctr[r["mode"]] += 1
            if "stratum" in r:
                stratum_ctr[r["stratum"]] += 1
            if "n_needles" in r:
                needle_ctr[str(r["n_needles"])] += 1
            if "band" in r:
                band_ctr[r["band"]] += 1

        seq_lens.sort()
        tgt_lens.sort()
        cue_audit: dict[str, Any] | None = None
        if skill == "skill_b":
            # Independently detect cue presence directly in user turn 1 & turn 2 text (do not trust flag alone).
            fam_counts: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])  # [with_cue, total]
            stratum_cue: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
            abstain_cue = call_cue = abstain_nocue = call_nocue = 0
            miss_cue = pos_cue = miss_nocue = pos_nocue = 0
            t2_cue = t2_total = 0
            for r in rows:
                msgs = json.loads(r["messages_json"])
                u1 = msgs[0]["content"]
                text_has_cue = any(p in u1 for p in SKILL_B_CUE_PARAPHRASES)
                if text_has_cue != bool(r["has_cue"]):
                    raise ValueError(f"{r['id']}: has_cue flag disagrees with user turn 1 cue text")
                st = r["stratum"]
                stratum_cue[st][0] += int(text_has_cue)
                stratum_cue[st][1] += 1
                if st.startswith("missing_required_arg"):
                    fam = "missing_required_arg"
                    is_abstain = True
                    miss_cue += int(text_has_cue)
                    miss_nocue += int(not text_has_cue)
                elif st == "domain_mismatch_abstain":
                    fam = "domain_mismatch_abstain"
                    is_abstain = True
                else:
                    fam = "positive_controls"
                    is_abstain = False
                    pos_cue += int(text_has_cue)
                    pos_nocue += int(not text_has_cue)
                fam_counts[fam][0] += int(text_has_cue)
                fam_counts[fam][1] += 1
                if text_has_cue:
                    abstain_cue += int(is_abstain)
                    call_cue += int(not is_abstain)
                else:
                    abstain_nocue += int(is_abstain)
                    call_nocue += int(not is_abstain)
                if st == "missing_required_arg_twoturn":
                    u2 = msgs[2]["content"]
                    u2_cue = any(p in u2 for p in SKILL_B_CUE_PARAPHRASES)
                    if u2_cue != bool(r["has_turn2_cue"]):
                        raise ValueError(f"{r['id']}: has_turn2_cue flag disagrees with user turn 2 text")
                    t2_cue += int(u2_cue)
                    t2_total += 1
            p_abs_cue = abstain_cue / max(1, abstain_cue + call_cue)
            p_abs_nocue = abstain_nocue / max(1, abstain_nocue + call_nocue)
            p_miss_cue = miss_cue / max(1, miss_cue + pos_cue)
            p_miss_nocue = miss_nocue / max(1, miss_nocue + pos_nocue)
            if abs(p_abs_cue - p_abs_nocue) > 0.02 or abs(p_miss_cue - p_miss_nocue) > 0.02:
                raise ValueError(
                    f"{split}: Skill B cue shortcut detected: P(abstain|cue)={p_abs_cue:.4f} vs "
                    f"P(abstain|no_cue)={p_abs_nocue:.4f}; missing_vs_pos {p_miss_cue:.4f} vs {p_miss_nocue:.4f}"
                )
            cue_audit = {
                "p_abstain_given_cue": round(p_abs_cue, 4),
                "p_abstain_given_no_cue": round(p_abs_nocue, 4),
                "abs_diff_overall": round(abs(p_abs_cue - p_abs_nocue), 4),
                "counts_overall": {
                    "abstain_cue": abstain_cue,
                    "call_cue": call_cue,
                    "abstain_no_cue": abstain_nocue,
                    "call_no_cue": call_nocue,
                },
                "p_missing_abstain_given_cue_vs_positive": round(p_miss_cue, 4),
                "p_missing_abstain_given_no_cue_vs_positive": round(p_miss_nocue, 4),
                "abs_diff_missing_vs_positive": round(abs(p_miss_cue - p_miss_nocue), 4),
                "by_stratum_family": {
                    k: {"with_cue": v[0], "total": v[1], "p_cue": round(v[0] / v[1], 4)}
                    for k, v in sorted(fam_counts.items())
                },
                "by_stratum": {
                    k: {"with_cue": v[0], "total": v[1], "p_cue": round(v[0] / v[1], 4)}
                    for k, v in sorted(stratum_cue.items())
                },
                "turn2_completion": {
                    "with_cue": t2_cue,
                    "total": t2_total,
                    "p_turn2_cue": round(t2_cue / max(1, t2_total), 4),
                    "p_call_given_turn2_cue": 1.0,
                },
            }
        summary[split] = {
            "parquet_sha256": sha256_file(ppath),
            "rows": len(rows),
            "loss_weight_scale": float(loss_weight_scale),
            "total_sequence_tokens": sum(seq_lens),
            "supervised_token_count": sum(tgt_lens),
            "total_supervised_target_tokens": sum(tgt_lens),
            "effective_loss_weight_sum": round(sum(tgt_lens) * float(loss_weight_scale), 4),
            "mean_sequence_tokens": round(sum(seq_lens) / len(seq_lens), 1),
            "min_sequence_tokens": seq_lens[0],
            "median_sequence_tokens": seq_lens[len(seq_lens) // 2],
            "max_sequence_tokens": seq_lens[-1],
            "mean_target_tokens": round(sum(tgt_lens) / len(tgt_lens), 1),
            "min_target_tokens": tgt_lens[0],
            "max_target_tokens": tgt_lens[-1],
            "modes": dict(mode_ctr),
            **({"strata": dict(stratum_ctr)} if stratum_ctr else {}),
            **({"cue_balance_audit": cue_audit} if cue_audit is not None else {}),
            **({"by_needles": dict(needle_ctr)} if needle_ctr else {}),
            **({"by_band": dict(band_ctr)} if band_ctr else {}),
        }

    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if split_ids[a] & split_ids[b]:
            raise ValueError(f"Row ID overlap between {a} and {b}")

    return {
        "pass": True,
        "skill": skill,
        "loss_weight_scale": float(loss_weight_scale),
        "checks": [
            "parquet list<int32>/list<float32> schema",
            "split row-ID disjointness",
            "BOS=128000 / EOS=128001 boundary / final loss_weight=0.0",
            f"loss_weights strictly in {{0.0, {float(loss_weight_scale)}}}",
            "shifted mask aligned to assistant header [128006, 78191, 128007, 198] through EOT=128009",
            "supervised_token_count and effective_loss_weight_sum exact conservation",
            "zero think-marker leak in /nothink targets",
        ],
        "splits": summary,
    }


def build_skill_dataset(
    *,
    skill: str,
    out_dir: Path,
    tokenizer_path: Path,
    template_path: Path | None,
    train_packs: int,
    val_packs: int,
    test_packs: int,
    probe_paths: dict[str, Path],
    micro210_path: Path | None,
    bfcl_root: Path | None,
    workers: int,
    upload_dest: str | None,
    loss_weight_scale: float = 1.0,
    from_parquet_dir: Path | None = None,
) -> dict[str, Any]:
    t0 = time.monotonic()
    if loss_weight_scale <= 0.0:
        raise ValueError(f"loss_weight_scale must be positive, got {loss_weight_scale}")
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_shas: dict[str, str] = {}
    if from_parquet_dir is not None:
        for split in ("train", "validation", "test"):
            src_rows = pq.read_table(from_parquet_dir / f"{split}.parquet").to_pylist()
            scaled_rows = _apply_loss_weight_scale(src_rows, loss_weight_scale)
            parquet_shas[split] = _write_parquet(scaled_rows, out_dir / f"{split}.parquet")
            print(
                f"[{skill}:{split}] rescaled {len(scaled_rows)} rows (w={loss_weight_scale}) "
                f"(sha256={parquet_shas[split][:12]})",
                flush=True,
            )
    else:
        extra_blocked_tools: list[str] | None = None
        if skill == "skill_b" and bfcl_root is not None and bfcl_root.is_dir():
            _, bfcl_names = extract_bfcl_names_and_payloads(bfcl_root)
            extra_blocked_tools = sorted(bfcl_names)

        # Oversample candidate counts safely to guarantee exact pack target fill before greedy trimming.
        rows_per_pack_est = 12 if skill == "skill_a" else 560
        split_targets = {"train": train_packs, "validation": val_packs, "test": test_packs}
        gen_fn = _gen_one_skill_a if skill == "skill_a" else _gen_one_skill_b

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(str(tokenizer_path), str(template_path) if template_path else None, extra_blocked_tools),
        ) as pool:
            for split, n_packs in split_targets.items():
                # Generate in batches until greedy packing exceeds n_packs, then trim to exact n_packs.
                rows: list[dict[str, Any]] = []
                next_idx = 0
                batch_sz = max(n_packs * rows_per_pack_est, 20 if skill == "skill_a" else 600)
                while True:
                    tasks = [(split, i) for i in range(next_idx, next_idx + batch_sz)]
                    rows.extend(pool.map(gen_fn, tasks, chunksize=4 if skill == "skill_a" else 64))
                    next_idx += batch_sz
                    packs_closed = 0
                    cur = 0
                    for r in rows:
                        rl = len(r["input_ids"])
                        if cur and cur + rl > SEQ_LEN:
                            packs_closed += 1
                            cur = 0
                        cur += rl
                    total_packs = packs_closed + (1 if cur > 0 else 0)
                    if total_packs > n_packs or (total_packs == n_packs and cur >= int(SEQ_LEN * 0.90)):
                        break
                    batch_sz = max(
                        rows_per_pack_est * max(1, n_packs - total_packs + 1),
                        12 if skill == "skill_a" else 500,
                    )

                trimmed = _trim_rows_to_exact_packs(rows, n_packs, SEQ_LEN)
                if loss_weight_scale != 1.0:
                    trimmed = _apply_loss_weight_scale(trimmed, loss_weight_scale)
                parquet_shas[split] = _write_parquet(trimmed, out_dir / f"{split}.parquet")
                print(
                    f"[{skill}:{split}] wrote {len(trimmed)} rows -> {n_packs} packs "
                    f"(w={loss_weight_scale}, sha256={parquet_shas[split][:12]})",
                    flush=True,
                )

    # 1. Row audit
    row_audit = audit_parquet_splits(skill, out_dir, tokenizer_path, loss_weight_scale=loss_weight_scale)
    (out_dir / "row-audit-receipt.json").write_text(json.dumps(row_audit, indent=2) + "\n")

    # 2. 8-gram disjointness audit
    decon_tok = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    decon = run_disjointness_audit(
        [out_dir / f"{s}.parquet" for s in ("train", "validation", "test")],
        probe_paths=probe_paths,
        tokenizer=decon_tok,
        micro210_path=micro210_path,
        bfcl_root=bfcl_root,
    )
    (out_dir / "disjointness-receipt.json").write_text(json.dumps(decon, indent=2) + "\n")
    if not decon["pass"]:
        raise RuntimeError(f"{skill}: disjointness audit FAILED (see {out_dir / 'disjointness-receipt.json'})")

    # 3. Pack into Levanter TreeCache with exact readback & per-segment verification
    cache_root = out_dir / "cache"
    packing_reports: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        ppath = out_dir / f"{split}.parquet"
        rows = pq.read_table(ppath, columns=["id", "input_ids", "loss_weights"]).to_pylist()
        rep = write_and_verify_levanter_cache(
            rows,
            cache_root / split,
            seq_len=SEQ_LEN,
            loss_weight_scale=loss_weight_scale,
        )
        rep["source_parquet_sha256"] = parquet_shas[split]
        packing_reports[split] = rep
    (cache_root / "packing-manifest.json").write_text(json.dumps(packing_reports, indent=2) + "\n")

    # 4. Optional upload to CoreWeave S3 with endpoint assertion and SHA-256 readback
    upload_receipt = None
    if upload_dest:
        upload_receipt = upload_dir_to_cw_s3(cache_root, upload_dest, out_dir / "upload-receipt.json")

    elapsed = round(time.monotonic() - t0, 1)
    summary = {
        "skill": skill,
        "loss_weight_scale": float(loss_weight_scale),
        "elapsed_seconds": elapsed,
        "row_audit_pass": row_audit["pass"],
        "disjointness_pass": decon["pass"],
        "packing": {s: {k: v for k, v in r.items() if k != "row_manifest"} for s, r in packing_reports.items()},
        "upload": (
            {k: upload_receipt[k] for k in ("dest_uri", "endpoint_url", "objects", "bytes", "readback_verified")}
            if upload_receipt
            else None
        ),
    }
    (out_dir / "build-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", choices=("skill_a", "skill_b"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument("--train-packs", type=int, required=True)
    parser.add_argument("--val-packs", type=int, default=2)
    parser.add_argument("--test-packs", type=int, default=2)
    parser.add_argument("--probe1", type=Path, required=True)
    parser.add_argument("--probe2", type=Path, required=True)
    parser.add_argument("--probe3", type=Path, required=True)
    parser.add_argument("--probe3-v2", type=Path, default=None)
    parser.add_argument("--micro210", type=Path, default=None)
    parser.add_argument("--bfcl-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--upload-dest", type=str, default=None)
    parser.add_argument(
        "--loss-weight-scale",
        type=float,
        default=1.0,
        help="Positive float multiplier on supervised loss_weights (e.g. 30.0 for ~1.39%% loss share; dose knob).",
    )
    parser.add_argument(
        "--from-parquet-dir",
        type=Path,
        default=None,
        help="Optional existing w=1.0 Parquet directory to rescale and re-audit/re-pack with identical input_ids.",
    )
    args = parser.parse_args()

    probe_paths = {"probe1": args.probe1, "probe2": args.probe2, "probe3": args.probe3}
    if args.probe3_v2 is not None:
        probe_paths["probe3_v2"] = args.probe3_v2

    build_skill_dataset(
        skill=args.skill,
        out_dir=args.output_dir,
        tokenizer_path=args.tokenizer,
        template_path=args.template,
        train_packs=args.train_packs,
        val_packs=args.val_packs,
        test_packs=args.test_packs,
        probe_paths=probe_paths,
        micro210_path=args.micro210,
        bfcl_root=args.bfcl_root,
        workers=args.workers,
        upload_dest=args.upload_dest,
        loss_weight_scale=args.loss_weight_scale,
        from_parquet_dir=args.from_parquet_dir,
    )


if __name__ == "__main__":
    main()
