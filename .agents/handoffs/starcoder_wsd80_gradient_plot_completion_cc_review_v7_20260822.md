PASS_AFTER_BLOCKERS_RESOLVED

# StarCoder WSD80 gradient plot completion: final CC review

## Review provenance

- Reviewer: Claude Code through the OAuth subscription for `plambdafour@proton.me`
- Billing preflight: `stripe_subscription`; `ANTHROPIC_API_KEY` removed from the child environment
- Model: `claude-opus-5[1m]`, maximum effort
- Permissions: read-only (`Read`, `Grep`, and `Glob`; no shell or write tools)
- Session: `381ecd81-7322-48f6-a7e7-fb640f74ecd2`
- Scope: frozen design closure, authorization lifecycle, staged execution, additive recovery semantics, and zero-LR handling

## Verdict

No launch-blocking or scientific blocker remains. The release recovers 288 saved-checkpoint groups without retraining and does not alter the frozen v10 inference.

The review identified and verified fixes for three fail-closed packaging defects:

1. Every object read by the frozen full training design is now hash-pinned and included in the workspace contract: the design manifest plus its 19 artifacts.
2. The authorization sidecar is excluded only from the preauthorization path set. Authorization creates it; launch and audit require and verify it.
3. The complete reachable experiment-module import closure is present, including the previously omitted `experiments/simple_train_config.py` leaf.

The reviewer independently verified the 224 common target-source additions, 32 H5 target-tail additions, and 32 H5 source-tail additions against the sealed v10 manifest. It also confirmed that final target-update cosine is structurally undefined because the learning rate is zero, rather than a recoverable missing value.

## Operational requirements

- Run under `UV_FROZEN=1` from the detached historical runtime tree.
- Include exactly `infra/pulumi/pyproject.toml` and `infra/pulumi/src` after excluding the broader `infra` tree.
- Run and pass `--mode audit --stage 1` before submitting Stage 2; that audit creates the immutable Stage-1 runtime-environment baseline required by later stages.
- Continue through the frozen 8, 16, 32, and 232-row stages only after the exact prior-stage audit passes.

## Residual risks

The release checks import-path presence manually; a future revision should derive the closure mechanically. Some historical launcher sources are presence-pinned rather than content-pinned, but any load-bearing drift fails against the row-level `train_config_sha256` identity before a result can be accepted. Neither issue blocks this frozen release.
