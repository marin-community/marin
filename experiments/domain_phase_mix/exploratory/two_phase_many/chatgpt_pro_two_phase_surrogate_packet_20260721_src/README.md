# ChatGPT Pro Two-Phase Surrogate Packet

This packet is a portable, task-focused snapshot for a long-running ChatGPT Pro modeling effort. It contains canonical one- and two-phase swarm tables, the complete 3e18 development-heldout archive available at build time, standalone reference models, exact historical source snapshots, prior negative results, and a prompt adapted from the search-control principles in OpenAI's Cycle Double Cover prompt.

## Use

1. Upload `chatgpt_pro_two_phase_surrogate_packet_20260721.zip` to a fresh ChatGPT Pro task.
2. Paste `PROMPT_TO_CHATGPT_PRO.md` as the task prompt.
3. Ask ChatGPT Pro to return the requested self-contained solution zip, not only a prose report.

The prompt is intentionally not line-wrapped within paragraphs so it can be pasted without hard line breaks.

## Verify Locally

From the extracted packet root:

```bash
uv run --no-project --script standalone_code/inspect_packet.py
uv run --no-project --script standalone_code/test_packet.py
shasum -a 256 -c CHECKSUMS.sha256
```

Run a representative fit:

```bash
uv run --no-project --script standalone_code/reproduce_fit.py --dataset delphi_3e18_two_phase_fit --target uncheatable_bpb --model compact_retained_state --output-dir outputs/compact_uncheatable
```

## Read Order

1. `PROMPT_TO_CHATGPT_PRO.md`
2. `docs/CHATGPT_PRO_PROTOCOL.md`
3. `docs/SCIENTIFIC_BRIEF.md`
4. `docs/DATA_DICTIONARY.md`
5. `docs/MODELS.md`
6. `evidence/mechanistic_surrogate_discovery/final_synthesis/final_report.md`
7. `evidence/compact_raw_optimum_validation/report.md`
8. `evidence/compact_sample_efficiency/report.md`

## Boundaries

The 3e18 archive is exposed development and falsification evidence, not an IID test set or untouched confirmation set. The currently running Compact sub-280 raw-optimum panel is not included. Any recommendation from this packet remains provisional until it succeeds on a new preregistered untouched panel.

No operational credentials, private storage prefixes, local home paths, W&B access, GCS access, Iris access, or Fieldbook installation are required.
