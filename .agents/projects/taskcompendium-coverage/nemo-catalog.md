# NeMo Gym coverage catalog

Effort: medium. Date: 2026-09-13. This is a bounded source audit, not an
environment validation: no server, verifier, agent, dataset download, or source
code was executed. The pinned upstream is
[`NVIDIA-NeMo/Gym@1e668906`](https://github.com/NVIDIA-NeMo/Gym/tree/1e668906d2e69a9e8ee9aaafc60050a4025d9688).

## Reading the catalog

`nemo-catalog.json` records one current canonical resource-server family per
requested capability, rather than treating agent-specific `environments/`
wrappers as different benchmarks. Wrapper paths exist only for Calendar v2,
Code Gen, Indian Banking, Instruction Following, Reasoning Gym, and Workplace
Assistant among this requested set. A missing wrapper is an observed absence at
the pinned tree, not a claim that the resource server is unusable.

The source revision is the live Hugging Face dataset commit returned by
`hf datasets info` on this date. It identifies current public content, while
the Gym commit identifies the verifier/configuration audited. Configs that only
name a GitLab artifact are marked unknown rather than assigned a guessed Hub
revision. The included fixture key is a bounded candidate for a future suite;
when upstream provides no stable ID, retain the raw canonical row and use its
content hash as provenance.

For the selected no-ID math, function-pivot, and Calendar fixtures, the JSON
catalog uses `jsonl_record_index: 0` and the SHA-256 of `jq -cS` canonical JSON.
The competitive-code selection is the first committed fixture row with
`hash_id: c69268d8bdb4da0685d7b187c88296c1`. Structured Outputs v4 row
`DS1-E974FB8A` is marked only as a supplementary final-action case.

## Capability result

Most rows reuse existing TaskCompendium semantics: answer-only exact/rule/
library scoring (MCQA, math, instruction following, Reasoning Gym, Calendar)
or an executable private code verifier (competitive code, SWE-Gym/SWE-bench).
They do not justify separate importer architectures.

Three execution shapes are missing or need an explicit adapter, each recorded
separately in the catalog.

1. **Final predicted action, no execution.** The conversational tool-use and
   SWE pivots judge a next function/terminal action structurally. A declared
   `function_call_batch` is an unordered comparison target, not an instruction
   to run tools in parallel. Structured Outputs v4 is also non-executing: one
   function call is the submitted answer. These need a final-action rendering
   and a private structural verifier, never an artificial shell/MCP runtime.
2. **Deterministic stateful domain tools.** Workplace Assistant and indirect
   prompt injection actually invoke in-process tools against a fresh seeded
   state. They need a provider/harness adapter that declares tool availability,
   preserves a private seed and verifier, and treats the semantic task as
   provider-independent. They should not become six source-specific importers.
3. **Reactive user conversation.** Indian Banking combines stateful tools with
   an LLM customer simulator and optional judge. It cannot be represented as
   fixed ordered static steps without losing behavior. It needs a distinct
   conversational-environment interface, exactly the deferred class called out
   in TaskCompendium's package guidance.

Google Search also executes real tools, but its state is the credentialed,
changing web. Its answer verifier is simple; a faithful source adapter is a
separate nondeterministic web provider with explicit credentials and sequential
tool policy, not a portable replay task.

## Fidelity cautions

- `code_gen` uses LiveCodeBench's remote checker and says the verifier itself
  has no sandbox. Do not claim container portability without a pinned checker
  runtime.
- Mini-SWE-Agent is an agent harness; SWE-bench owns the per-instance sandbox
  and test evaluation. Keep harness selection out of a semantic task record.
- Calendar is described as multi-turn, but the audited server receives already
  materialized conversation history and grades one final JSON response. It has
  no runtime user simulator.
- Structured Outputs explicitly scores schema adherence only. A schema-valid
  hallucination is source-correct for this benchmark.
- The current Indian Banking config requires a non-policy model for both user
  simulation and judging; source behavior is not fully reproducible from its
  static dataset revision alone.
- Workplace id `0` is materialized as a pinned fixture with GitHub Contents
  blob and raw SHA-256 provenance for its source record, server contract, six
  tool modules, and six CSV seed files. The Apache-2.0 provider uses those
  bounded modules plus `pandas`, without the NeMo Gym training or server stack.
  Its private verifier rejects malformed gold as an invalid task; ordinary
  agent tool errors remain source-visible observations. It preserves the
  source's scope: calendar, email, analytics plots, project-management, and
  CRM mutable tables are compared; company-directory output, immutable
  analytics rows, and tool-result text are not reward targets.

## Source ledger

| Family | Primary source used | Dataset/source status |
| --- | --- | --- |
| MCQA | [README](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/mcqa/README.md) | `nvidia/Nemotron-RL-knowledge-mcqa@62a1eec1f952723eab2ee3832222f533b8138067` |
| Math | [app](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/math_with_judge/app.py) | `nvidia/Nemotron-RL-math-OpenMathReasoning@5ef69384b65f13f08c35b73ddd5e7bf5e4621043` |
| Instruction following | [app](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/instruction_following/app.py) | `nvidia/Nemotron-RL-instruction_following@3b253899665cb71334bb54c14eb5d91751beaad7` |
| Code generation | [app](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/code_gen/app.py) | `nvidia/Nemotron-RL-coding-competitive_coding@ae1f446f299823ea3c4c00217942b53787278b31` |
| Function pivot | [README](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/single_step_tool_use_with_argument_comparison/README.md) | `nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1@9643c8103d7bfbc2d7fc4d15991d6739c612ff58` |
| Workplace | [app](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/workplace_assistant/app.py) | `nvidia/Nemotron-RL-agent-workplace_assistant@c86a908379e0a361a573c395e175d3c1aa128e6c` |
| SWE pivot | [app](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/swe_pivot/app.py) | Hub candidate `nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1@4947a3c8ea803413a65f9eca14a96ef521b2ddf5`; config uses GitLab, split join unverified |
| Mini-SWE/SWE-bench | [mini harness](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/responses_api_agents/mini_swe_agent/README.md) | `SWE-Gym/SWE-Gym@bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb`; validation `princeton-nlp/SWE-bench_Verified@c104f840cc67f8b6eec6f759ebc8b2693d585d4a` |
| Google Search | [app](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/google_search/app.py) | `nvidia/Nemotron-RL-knowledge-web_search-mcqa@0feb60b8347ff5389139ada2d8e52a8bdad434f8` |
| Reasoning Gym | [app](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/reasoning_gym/app.py) | `nvidia/Nemotron-RL-ReasoningGym-v1@ad2c929b2dfd64ca30afb4d60e2f69a5a1919c4d` |
| Structured v4 | [README](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/structured_outputs/README.md) | v4 config is GitLab-only; older text source is not a v4 revision |
| Calendar v2 | [config](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/calendar/configs/calendar_v2.yaml) | `nvidia/Nemotron-RL-Instruction-Following-Calendar-v2@556a3b1ab3eb12bab38327bf0ec4cdbeab452338` |
| Indian Banking | [app](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/indian_banking/app.py) | `NPCI/nemo-gym-indian-banking@b58b627a3b366bc0fdc5064bc66cb59e92dee2c3` |
| Indirect injection | [verifier](https://github.com/NVIDIA-NeMo/Gym/blob/1e668906d2e69a9e8ee9aaafc60050a4025d9688/resources_servers/indirect_prompt_injection/verifier.py) | GitLab-only in config; README says full release is future work |
