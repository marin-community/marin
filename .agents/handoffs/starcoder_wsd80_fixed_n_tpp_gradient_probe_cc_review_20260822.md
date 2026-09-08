# Fixed-N TPP gradient-onset probe review

Claude Opus 5 reviewed the freeze and runtime modules adversarially through the `plambdafour@proton.me` OAuth subscription with API-key authentication disabled.

The first review found that the adapted repair kernel would read an unexported `SUPPORT_STARCODER` name after completing gradient computation. The runtime now exports it and fails locally unless every dynamic `freeze.*` name used by either the mechanism kernel or historical runtime is available.

The final review verified:

- the local remote-canary entrypoint applies the new adapter before delegating;
- the primary estimand uses the shared 0.55T, 0.70T, 0.80T, and 0.90T grid rather than incomparable fixed-step offsets;
- the manifest contains 256 rows, with a 16-row cross-cell preflight;
- 208 rows match historical precision and 48 disclosed r3 rows use 16 of 64 blocks and 8 of 32 optimizer draws;
- runtime, source, checkpoint, reference-batch, and output identities fail closed;
- the preflight exercises exact and reduced precision paths before four disjoint 64-row parents may run concurrently;
- the runner does not read endpoint outcomes; and
- all storage, checkpoints, parents, and TPU work remain in us-central1.

No hard blocker remains. The compressed workspace bundle must still be measured before submission, and the final whole-panel audit remains mandatory before scientific analysis.

VERDICT: PASS
