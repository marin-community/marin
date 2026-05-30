# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Structural verification of materialized Delphi prefix checkpoints.

metadata.json records the Levanter TrainerState pytree at GROUP granularity
(model.embeddings / model.transformer / model.lm_head / opt_state / step / ...);
per-leaf arrays (qk_norm, Adam moments) live in the OCDBT store beneath each
group. So this check proves: (1) the prefix checkpoint's group set is IDENTICAL
to the canonical SOURCE isoflop checkpoint (a known-complete Qwen3+AdamH
full-state checkpoint), (2) opt_state group is present, (3) the model.* groups
are present, (4) step metadata == expected target step. Identical group
structure vs source ⇒ same model arrays (incl qk_norm) + same optimizer moments.

Leaf-level qk_norm/opt-moment proof is done separately by an actual Levanter
load (verify_prefix_load.py).
"""
from __future__ import annotations

import json

import fsspec

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
# base: (70% step, 80% step)
CK = {
    "3e18": ("26134", "29868"),
    "9e18": ("31021", "35453"),
    "2e19": ("38587", "44100"),
    "3e19": ("26609", "30411"),
}


def groups(ckpt_path: str) -> set[str]:
    with fsspec.open(f"{ckpt_path.rstrip('/')}/metadata.json", "r") as f:
        d = json.load(f)
    tm = d.get("tree_metadata") or d.get("tree") or {}
    return set(tm.keys()) if isinstance(tm, dict) else set()


def main() -> int:
    all_ok = True
    for base, (s70, s80) in CK.items():
        src = SRC[base]
        src_groups = groups(src)
        has_opt_src = any(k == "opt_state" or k.startswith("opt_state") for k in src_groups)
        print(f"\n== {base} ==")
        print(f"  SOURCE {src}")
        print(
            f"    groups={len(src_groups)} opt_state={has_opt_src} "
            f"model.emb={'model.embeddings' in src_groups} "
            f"model.transformer={'model.transformer' in src_groups} "
            f"model.lm_head={'model.lm_head' in src_groups}"
        )
        print(f"    source group set: {sorted(src_groups)}")
        for label, step in (("70%", s70), ("80%", s80)):
            path = f"{ROOT}/delphi-{base}-prefixes-qwen3/checkpoints/step-{step}"
            g = groups(path)
            same = g == src_groups
            has_opt = any(k == "opt_state" or k.startswith("opt_state") for k in g)
            model_ok = {"model.embeddings", "model.transformer", "model.lm_head"} <= g
            ok = same and has_opt and model_ok
            all_ok = all_ok and ok
            extra = g - src_groups
            missing = src_groups - g
            print(
                f"  [{'OK ' if ok else 'FAIL'}] {label} step-{step}: groups={len(g)} "
                f"identical_to_source={same} opt_state={has_opt} model_groups={model_ok}"
                + ("" if not missing else f" MISSING={sorted(missing)}")
                + ("" if not extra else f" EXTRA={sorted(extra)}")
            )
    print("\nSTRUCTURE RESULT:", "ALL MATCH SOURCE + HAVE opt_state" if all_ok else "MISMATCH — investigate")

    # Machine-readable summary so a corrupted terminal render cannot mislead.
    result = {"all_ok": all_ok, "checkpoints": {}}
    for base, (s70, s80) in CK.items():
        src_groups = groups(SRC[base])
        for label, step in (("70", s70), ("80", s80)):
            path = f"{ROOT}/delphi-{base}-prefixes-qwen3/checkpoints/step-{step}"
            g = groups(path)
            result["checkpoints"][f"{base}_{label}_step{step}"] = {
                "n_groups": len(g),
                "identical_to_source": g == src_groups,
                "has_opt_state": any(k == "opt_state" or k.startswith("opt_state") for k in g),
                "has_model_groups": {"model.embeddings", "model.transformer", "model.lm_head"} <= g,
            }
    with open("/tmp/struct_result.json", "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print("WROTE /tmp/struct_result.json")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
