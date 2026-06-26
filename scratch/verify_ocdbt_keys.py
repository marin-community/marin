# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""DECISIVE OCDBT-level verification of materialized Delphi prefix checkpoints.

Lists the real leaf keys in each checkpoint's OCDBT kvstore (via the repo's
tested ``marin.midtraining.checkpoint_schema.default_list_checkpoint_keys``)
and confirms, against the KNOWN-GOOD canonical source checkpoint:

  - q_norm / k_norm keys present  (Qwen3 QK-norm — the 2026-05-27 silent-drop bug)
  - opt_state keys present        (optimizer moments preserved for cooldown resume)
  - model.embeddings / transformer / lm_head present
  - prefix key SET == source key SET (no missing/extra leaves)

Writes a JSON verdict to /tmp/ocdbt_result.json. Read-only; no GCS writes.
"""
from __future__ import annotations

import json

from marin.midtraining.checkpoint_schema import default_list_checkpoint_keys

ROOT = "gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints"
SRC = {
    "3e18": (
        "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+18-d1024-L11-B8-adamh_scaling_v6/checkpoints/step-20000"
    ),
    "9e18": (
        "gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+18-d1152-L12-B16-adamh_scaling_v6/checkpoints/step-30000"
    ),
    "2e19": (
        "gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+19-d1408-L15-B16-adamh_scaling_v6/checkpoints/step-30000"
    ),
    "3e19": (
        "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+19-d1536-L16-B32-adamh_scaling_v6/checkpoints/step-20000"
    ),
}
CK = {
    "3e18": ("26134", "29868"),
    "9e18": ("31021", "35453"),
    "2e19": ("38587", "44100"),
    "3e19": ("26609", "30411"),
}


def summarize(keys: tuple[str, ...]) -> dict:
    return {
        "n": len(keys),
        "q_norm": sum(1 for k in keys if "q_norm" in k),
        "k_norm": sum(1 for k in keys if "k_norm" in k),
        "opt_state": sum(1 for k in keys if "opt_state" in k),
        "emb": sum(1 for k in keys if "embeddings" in k or "embed" in k),
        "lm_head": sum(1 for k in keys if "lm_head" in k),
        "transformer": sum(1 for k in keys if "transformer" in k or "layers" in k or "blocks" in k),
    }


def main() -> int:
    result: dict = {"bases": {}, "all_ok": True}
    for base, (s70, s80) in CK.items():
        src = SRC[base]
        src_keys = set(default_list_checkpoint_keys(src))
        src_sum = summarize(tuple(src_keys))
        entry: dict = {"source": {"path": src, **src_sum}, "prefixes": {}}
        print(f"\n== {base} ==")
        print(
            f"  SOURCE n={src_sum['n']} q_norm={src_sum['q_norm']} k_norm={src_sum['k_norm']} "
            f"opt_state={src_sum['opt_state']} emb={src_sum['emb']} lm_head={src_sum['lm_head']} "
            f"transformer={src_sum['transformer']}"
        )
        for label, step in (("70", s70), ("80", s80)):
            path = f"{ROOT}/delphi-{base}-prefixes-qwen3/checkpoints/step-{step}"
            try:
                pk = set(default_list_checkpoint_keys(path))
            except Exception as exc:
                print(f"  [ERROR] {label}% step-{step}: {exc!r}")
                entry["prefixes"][label] = {"error": repr(exc)}
                result["all_ok"] = False
                continue
            s = summarize(tuple(pk))
            same_set = pk == src_keys
            missing = sorted(src_keys - pk)
            extra = sorted(pk - src_keys)
            qk_ok = s["q_norm"] > 0 and s["k_norm"] > 0
            opt_ok = s["opt_state"] > 0
            ok = qk_ok and opt_ok and same_set
            result["all_ok"] = result["all_ok"] and ok
            entry["prefixes"][label] = {
                "path": path,
                "step": step,
                **s,
                "same_keyset_as_source": same_set,
                "n_missing_vs_source": len(missing),
                "n_extra_vs_source": len(extra),
                "qk_ok": qk_ok,
                "opt_ok": opt_ok,
                "ok": ok,
                "missing_sample": missing[:8],
                "extra_sample": extra[:8],
            }
            print(
                f"  [{'OK ' if ok else 'FAIL'}] {label}% step-{step}: n={s['n']} "
                f"q_norm={s['q_norm']} k_norm={s['k_norm']} opt_state={s['opt_state']} "
                f"same_keyset={same_set} missing={len(missing)} extra={len(extra)}"
            )
            if missing:
                print(f"        MISSING sample: {missing[:6]}")
            if extra:
                print(f"        EXTRA sample: {extra[:6]}")
        result["bases"][base] = entry
    with open("/tmp/ocdbt_result.json", "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(
        "\nOCDBT RESULT:",
        (
            "ALL COMPLETE (qk_norm + opt_state + keyset==source)"
            if result["all_ok"]
            else "INCOMPLETE — see FAIL/ERROR rows"
        ),
    )
    print("WROTE /tmp/ocdbt_result.json")
    return 0 if result["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
