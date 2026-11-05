# What's in a base? Reproduction files

This directory and `experiments/evaluation/mini_policy_runtime/` hold the code behind the report
"What's in a base? The changing face of the base model" (2026-11-05). The scripts are kept as they
ran. Some use local paths from the original workstation; change them before you run them.

## Shared runtime

All GPU work ran on Iris `cw-rno2a`:

- Mini840 served the model under test on 2 H100s, with a separate 8-H100 judge.
- Snowball probes, trace scoring and the decode ablation used vLLM at TP1 x DP4 with expert
  parallelism on 4 H100s, with a whole 8-H100 node reserved per job. The decode ablation used
  vLLM `39e62869`, which is newer than the campaign build.
- Qwen probes used transformers (sdpa), not vLLM.

- `mini_policy_runtime/generational_perplexity.py` scores probe strings and full traces by teacher
  forcing. It supports `--template-family template`, `--enable-thinking`, `--chat-template-file`
  and `--traces` / `--trace-output` for per-token NLL.
- `mini_policy_runtime/generational_perplexity_serve.sh MIRROR LABEL [args]` starts vLLM
  (TP1 x DP4, expert parallel) on the mirror, then runs the probe script against it.
- `mini_policy_runtime/chat-templates/` holds the Snowball SFT template
  (`grug-67b-a2b-datakit-sft.jinja`), which both Snowball checkpoints use, and the Kimi-Linear
  template.
- `mini_policy_runtime/generation-configs/qwen35-35b-a3b-base-im-end-stop/` holds the
  `generation_config.json` that adds `<|im_end|>` (248046) as a stop token for the Qwen3.5 base
  (#10901).

## Report sections and their sources

| Section | Code | Data |
| --- | --- | --- |
| 1. Mini840 results; stop-token rerun | Mini840 launcher on this branch (#10804); `audits/` | Tracker and audits in the reports and payloads gists |
| 2–5. Probes: format, think tag, dialect, canaries | `generational_perplexity.py` | Probe payloads gist |
| 3. Snowball think marker | `generational_perplexity.py` (`start_think_marker` probes) | `tm-v1-*.json` in the payloads gist |
| 1. Turn coherence | `turn_coherence/turn_coherence.py` | Mini840 dumps (below) |
| 6. Trace likelihood | `generational_perplexity.py --traces`, `trace_likelihood/analyze_traces.py` | Trace set in the [traces gist](https://gist.github.com/penfever/3a411b1289e9dbcb8d9edfc36edb8eaf); save it as `mini_policy_runtime/trace-sets/ifbench-11-authors-100-prompts.jsonl` (the repository ignores `*.jsonl`) |
| 6. Snowball SFT improbability | `sft_improbability/*.py` | Per-token scores (below) |
| 6. Decode-setting ablation | `mini_policy_runtime/decode_ablation_*`, `decode_ablation/*` | Ablation tables in the reports gist |

## Public data

- [Reports and tables](https://gist.github.com/penfever/2422c4b74ffc7644f282fdc050dfd914): analysis,
  Mini840 tracker and state log, and the turn-coherence, SFT-improbability, non-greedy,
  decode-ablation, think-marker and trace-likelihood reports.
- [Probe payloads and audits](https://gist.github.com/penfever/c83170094ba051f12694ede515dd83e2).
- [IFBench trace set](https://gist.github.com/penfever/3a411b1289e9dbcb8d9edfc36edb8eaf).

## Large data (Marin-internal object storage, region RNO2A)

- Mini840 run root: `s3://marin-rno2a/marin/experiments/chat-aligned-base-evals/2026.10.31.150013/`.
- Qwen3.5 base `<|im_end|>` rerun:
  `s3://marin-rno2a/marin/checkpoints/mini840-e2e-qwen35-35b-a3b-base-im-end-stop/2026.11.04.4`.
- Trace-likelihood per-token scores and decode-ablation generations and scores (permanent copy):
  `s3://marin-rno2a/marin/experiments/base-model-evals/whats-in-a-base-2026.11.05/`.

## Notes

- The trace set replaces two email addresses in model responses with `[REDACTED-EMAIL]`.
- The IFBench traces were sampled at each author's own defaults. See the report's section 6 and
  the decode ablation for the controlled comparison.
