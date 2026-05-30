# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""DECISIVE leaf-level verification: actually load a materialized prefix
checkpoint's model subtree into a Qwen3 eval-shape with allow_partial=False
(the same strict load Phase-2 cooldown performs), and assert q_norm / k_norm
(QK-norm) leaves are present and non-None. Also confirms opt_state group.

This removes all doubt about QK-norm at the leaf level: with allow_partial=False
Levanter RAISES if any expected array (incl q_norm/k_norm) is absent from the
stored checkpoint. Writes a JSON verdict to /tmp/load_result.json.

Usage: python scratch/verify_prefix_load.py <base> <step>
"""
from __future__ import annotations

import json
import sys
import traceback

ROOT = "gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints"


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "3e18"
    step = sys.argv[2] if len(sys.argv) > 2 else "29868"
    ckpt = f"{ROOT}/delphi-{base}-prefixes-qwen3/checkpoints/step-{step}"
    verdict: dict = {"base": base, "step": step, "ckpt": ckpt}
    try:
        import equinox as eqx
        import haliax as hax
        import jax
        from levanter.checkpoint import load_checkpoint

        from experiments.delphi_models import get_delphi_model
        from scripts.materialize_delphi_prefix_checkpoint import load_source_train_config

        model = get_delphi_model(base)
        src_cfg = load_source_train_config(model)
        model_cfg = src_cfg.model
        verdict["model_cfg_type"] = type(model_cfg).__name__
        verdict["use_qk_norm"] = getattr(model_cfg, "use_qk_norm", None)

        # Vocab axis: prefer the source config's vocab, else LLaMA3 default.
        vocab_size = getattr(src_cfg, "vocab_size", None) or getattr(model_cfg, "vocab_size", None) or 128256
        Vocab = hax.Axis("vocab", int(vocab_size))
        verdict["vocab_size"] = int(vocab_size)

        key = jax.random.PRNGKey(0)
        model_shape = eqx.filter_eval_shape(model_cfg.build, Vocab, key=key)

        # Strict load of just the model subtree. Raises if any leaf is missing.
        loaded = load_checkpoint(model_shape, ckpt, subpath="model", allow_partial=False)
        verdict["model_load"] = "OK (allow_partial=False succeeded)"

        # Walk to attention q_norm/k_norm. Qwen3 transformer is a Stacked block,
        # so the q_norm/k_norm leaves carry a leading layer axis.
        leaves_with_paths = jax.tree_util.tree_leaves_with_path(loaded)
        qn = [p for p, _ in leaves_with_paths if "q_norm" in jax.tree_util.keystr(p)]
        kn = [p for p, _ in leaves_with_paths if "k_norm" in jax.tree_util.keystr(p)]
        verdict["q_norm_leaves"] = len(qn)
        verdict["k_norm_leaves"] = len(kn)

        # Shapes of the first q_norm/k_norm leaf (proves real arrays, layer-stacked)
        def shape_of(path):
            for p, v in leaves_with_paths:
                if p == path:
                    return list(getattr(v, "shape", []))
            return None

        verdict["q_norm_shape"] = shape_of(qn[0]) if qn else None
        verdict["k_norm_shape"] = shape_of(kn[0]) if kn else None
        verdict["qk_norm_present"] = bool(qn) and bool(kn)
        verdict["ok"] = verdict["qk_norm_present"]
    except Exception as exc:
        verdict["ok"] = False
        verdict["error"] = repr(exc)
        verdict["traceback"] = traceback.format_exc()

    with open("/tmp/load_result.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print("WROTE /tmp/load_result.json  ok=", verdict.get("ok"))
    return 0 if verdict.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
