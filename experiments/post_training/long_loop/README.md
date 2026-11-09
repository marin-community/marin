# Long-loop prototype

Snowball SFT → asynchronous RLOO-N (GS8) → Micro210 → public HF checkpoint and
provenance archive. The launcher takes one content-pinned run spec. Dataset
manifests and stage configurations are explicit inputs, resolved relative to it.

```bash
uv run python -m experiments.post_training.long_loop \
  --spec experiments/post_training/long_loop/specs/round1.yaml \
  --validation-output /tmp/long-loop/full --version 2026.11.10
```

Add `--smoke` to use the same inputs with the spec's scale overrides. Add
`--run --max-concurrent 1` only for an admitted run. The resolved JSON records the
spec digest, complete configs, dataset identities, scale changes and blockers.
Strict SkyRL composition runs on the actual graph documents before allocation.
The example is intentionally **not launchable yet**: the admitted agentic cohort,
heldout/exclusion evidence and several runtime provenance bindings are pending.
A printed blocked plan is not a successful dry-run or permission to launch.

The smoke keeps the full SFT weight schedule, 262k SFT packing, 102,400-token RL
window, CP4, GS8 and 32 agentic / 32 non-agentic consumed groups per update.
Only update counts and evaluation scale may change. The old plumbing smoke used
32k SFT, the non-agentic quick-loop selection and short-context RL. It verified
stage handoffs but did not qualify the intended training geometry or data mix.
Those historical commands remain reproducible at their original Git pins;
the current entrypoint requires `--spec`.

The `.11` full SFT attempt inherited the old smoke's fixed initial mixture weights.
Its source components matched the reproduction, but it omitted the 250-step weight
schedule. It failed after its first update in the checkpoint sampling helper and
has not been restarted. The supplied SFT manifest restores all 19 schedule phases.
50B denotes scheduled packed positions, not 50B distinct supervised tokens.

All jobs remain on RNO2A at **interactive** priority, with at most **256 H100**.
Missing inputs and failed qualification gates stop the run. A failed stage is not
replayed automatically. Keep this PR draft; no merge approval is implied.

## Historical plumbing evidence

The completed plumbing run verified 20 supported benchmark paths, including the
BixBench repair. This was a runtime test, not a full Micro210 performance result.

The historical Micro210 runtime originated from #10933. Exact source and dataset
pins, consumed configs and retained traces are in the verified archive evidence.
The selected plumbing model is public at
[Snowball-LongLoop-Plumbing-Step3-2026.11.09-r1](https://huggingface.co/open-athena/Snowball-LongLoop-Plumbing-Step3-2026.11.09-r1),
revision `638e48507118e60d09b64be50230f795a50deff9`.

Campaign status and evidence live in
`/Users/benfeuer/Documents/experiments/active/long-loop-prototype/`:
`STATUS.md`, `DESIGN.md`, `artifacts/archive-hf-smoke-20261109/`, and
`artifacts/run-spec-20261109/`. The original copy-based archive job was cancelled;
the replacement HF publication and evidence archive verified. Historical trace
sampling and telemetry gaps remain documented there.
