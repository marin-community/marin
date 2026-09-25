# SFT source quality spot check

All 247 registered SFT sources were graded. Ten chat records were scored for 245 sources; the two remaining Penfever OpenCode stores contained only two and six records, all of which were scored. The distribution is 35 A, 193 B, 14 C, 5 D, and 0 F. These grades screen for behavior worth imitating; they do not measure task success or training effect.

## Method

GPT-6-luna reviewers scored up to ten bounded chat excerpts per source against the rubric below. The registry snapshot is checkout `82620ad62b` plus the local `identity-data` registration and exclusions of three Penfever traces that teach a different model identity. The CSV lists the 247 included names. Of these, 106 (105 Penfever execution-trace sources and AgentTrove) were scored earlier from pinned Hugging Face revisions or structured chat rows accepted by their adapters in regional storage.

For the other normalized-chat sources, the sampler computed SHA-256 over the source name, used the digest modulo the shard count and row-group count, then read ten consecutive rows from that row group. The identity source was sampled at one offset per dataset decile from its pinned Hugging Face revision. These deterministic convenience samples are not uniform corpus-wide random samples. The grade CSV records each row ID and either a source revision or a normalized-chat artifact hash; the artifact manifest records the full path and sampled file. The math v3 and instruction-following chat v3 samples used completed normalized artifacts `80323c41` and `5634cbcc`.

| Dimension | 2 points | 1 point | 0 points |
| --- | --- | --- | --- |
| Task grounding | Actions address the task and its constraints | Mostly on task, with drift | Wrong task or invented requirements |
| Substantive progress | Appropriate actions or answer make useful progress | Partial progress or detours | No useful progress or unproductive loop |
| Evidence and correction | Uses relevant evidence or a sound derivation and corrects visible errors | Plausible answer with limited visible support | Ignores contrary evidence or makes an unsupported claim |
| Ending and handoff | Coherent completion or honest unresolved state | Weak handoff or unfinished continuation | False success claim or misleading ending |
| Training signal | Reasoning and tool use are clear and worth imitating | Duplication, boilerplate, or harness ritual | Severe repetition, unsupported certainty, or parser leakage |

Judge behavior in the context of the visible request and conversation format. A direct answer, worked solution, refusal, or code artifact can make full progress without tools. Require a check or correction only when the task and visible evidence call for one; a concise answer need not show a test. A complete answer is a coherent ending, while unfinished work needs an honest handoff. Score the usefulness and clarity of the response, not its length or tool count. An excerpt that omits a check leaves that check unknown. A terminal tool call such as `functions.finish` can be a valid ending when the environment defines it as completion.

Reviewers judged whether visible reasoning supports the answer and whether a claim contradicts visible tool output. They did not systematically verify final answers or run code. Where present, `result` and `verifier_output` are post-run metadata. They describe the outcome; they are not messages the assistant saw. A failed or unverified episode is not penalized merely for failing. A hidden verifier failure does not establish that the assistant ignored an error. In the Penfever terminal traces, the JSON command's `analysis` field is visible output, not a parsed thinking turn.

| Grade | Rule |
| --- | --- |
| A | Median 9–10, at least eight rows score 8+, and no repeated severe defect |
| B | Median at least 7 but the A rule is not met; at least eight rows score 5+, and at most one severe defect |
| C | Median 5–6.5, or useful rows mixed with several low-quality examples |
| D | Median 3–4.5, or repeated misleading endings, loops, or weak task grounding |
| F | Median below 3, or pervasive behavior that should not be imitated |

A severe defect is a false completion claim, an unsupported claim of a consequential action, or a clearly contradictory answer. Repeated severe defects can cap the letter grade even when the median is higher; the CSV flags affected rows. For example, `swe-zero-12m` is D despite median 9 because three of ten rows claim completion after saying no repository edit was made. For the two small stores, the reviewers applied the same row rubric to every record and assigned a provisional letter grade from the observed pattern; the ten-row count thresholds do not apply. The source grades in the 2026-09-18 mixture notes came from one record per source and used no written rubric, so they are not directly comparable.

## Results

| Source group | Sources | A | B | C | D | F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nemotron SFT | 3 | 0 | 3 | 0 | 0 | 0 |
| Nemotron SFT v3 agentic_v1 | 2 | 0 | 2 | 0 | 0 | 0 |
| Nemotron SFT v3 agentic_v2 | 3 | 0 | 3 | 0 | 0 | 0 |
| Nemotron SFT v3 arc_agi_v1 | 8 | 0 | 4 | 1 | 3 | 0 |
| Nemotron SFT v3 competitive_programming_v2 | 4 | 0 | 4 | 0 | 0 | 0 |
| Nemotron SFT v3 cuda_v1 | 1 | 0 | 1 | 0 | 0 | 0 |
| Nemotron SFT v3 finance_v1 | 1 | 0 | 1 | 0 | 0 | 0 |
| Nemotron SFT v3 instruction_following_chat_v2 | 2 | 1 | 1 | 0 | 0 | 0 |
| Nemotron SFT v3 instruction_following_chat_v3 | 2 | 1 | 1 | 0 | 0 | 0 |
| Nemotron SFT v3 math_proofs_v1 | 1 | 0 | 1 | 0 | 0 | 0 |
| Nemotron SFT v3 math_proofs_v2 | 1 | 0 | 1 | 0 | 0 | 0 |
| Nemotron SFT v3 math_v2 | 3 | 1 | 2 | 0 | 0 | 0 |
| Nemotron SFT v3 math_v3 | 1 | 0 | 1 | 0 | 0 | 0 |
| Nemotron SFT v3 math_v4 | 1 | 0 | 1 | 0 | 0 | 0 |
| Nemotron SFT v3 multilingual_v1 | 18 | 2 | 15 | 1 | 0 | 0 |
| Nemotron SFT v3 multilingual_v2 | 12 | 3 | 5 | 3 | 1 | 0 |
| Nemotron SFT v3 opencode_v1 | 6 | 0 | 5 | 1 | 0 | 0 |
| Nemotron SFT v3 safety_v1 | 1 | 1 | 0 | 0 | 0 | 0 |
| Nemotron SFT v3 safety_v2 | 1 | 0 | 1 | 0 | 0 | 0 |
| Nemotron SFT v3 science_v2 | 4 | 0 | 4 | 0 | 0 | 0 |
| Nemotron SFT v3 swe_v1 | 1 | 1 | 0 | 0 | 0 | 0 |
| Nemotron SFT v3 swe_v2 | 2 | 1 | 0 | 1 | 0 | 0 |
| OpenSWE Traces | 13 | 0 | 13 | 0 | 0 | 0 |
| Other SFT sources | 19 | 6 | 12 | 0 | 1 | 0 |
| Penfever GLM 5.2 Terminus 2 | 13 | 8 | 5 | 0 | 0 | 0 |
| Penfever MiniMax M27 131K | 44 | 2 | 41 | 1 | 0 | 0 |
| Penfever Qwen 3.5 122B 32K | 48 | 4 | 43 | 1 | 0 | 0 |
| Penfever Qwen 3.5 122B OpenCode | 32 | 4 | 23 | 5 | 0 | 0 |
| Total | 247 | 35 | 193 | 14 | 5 | 0 |

The [grade data](sft-agent-trace-quality-2026-09-25-grades.csv) contain the component scores and row IDs, and the [artifact manifest](sft-agent-trace-quality-2026-09-25-artifacts.csv) records sampled normalized stores. “Other SFT sources” groups 19 stand-alone registries spanning conversation, math, code, compaction, and function calling; each is named in the per-source table.

The tail audit raised nine of 22 initial low source grades. The five D grades that remain include three ARC partitions with unsupported or contradictory inferred rules, a Hindi coding partition whose sampled replies give prose approaches without the requested code, and `swe-zero-12m`. In the last source, three sampled traces report that no repository edit was made and then claim “Task complete”; one other row claims a dependency update while its displayed output still shows the old version.

## Limits

Ten adjacent rows from one normalized shard may overrepresent a topic or task type. Reviewers saw bounded excerpts, not full code bodies or every tool result. The 22 sources initially rated C, D, or F in the main normalized-store pass were resampled with response tails and regraded; the remaining later samples already included tails. A source grade does not certify answer correctness, tool execution, task reward, or suitability for a training mix. The two small stores have especially uncertain source grades because all available records still cover little variety. Recheck full traces and outcome distributions before using these grades as a hard data filter.

## Per-source grades

### Nemotron SFT

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft/sft_code` | 10 | B | 8 |
| `nemotron_sft/sft_general` | 10 | B | 8 |
| `nemotron_sft/sft_math` | 10 | B | 7 |

### Nemotron SFT v3 agentic_v1

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/agentic_v1/interactive_agent` | 10 | B | 8 |
| `nemotron_sft_v3/agentic_v1/tool_calling` | 10 | B | 8 |

### Nemotron SFT v3 agentic_v2

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/agentic_v2/interactive_agent` | 10 | B | 8 |
| `nemotron_sft_v3/agentic_v2/search` | 10 | B | 8 |
| `nemotron_sft_v3/agentic_v2/tool_calling` | 10 | B | 9 |

### Nemotron SFT v3 arc_agi_v1

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/arc_agi_v1/large_no_reasoning_no_tools` | 10 | C | 8 |
| `nemotron_sft_v3/arc_agi_v1/large_reasoning_and_tools` | 10 | B | 8 |
| `nemotron_sft_v3/arc_agi_v1/large_reasoning_no_tools` | 10 | D | 3 |
| `nemotron_sft_v3/arc_agi_v1/large_tools_no_reasoning` | 10 | B | 7 |
| `nemotron_sft_v3/arc_agi_v1/small_no_reasoning_no_tools` | 10 | D | 3 |
| `nemotron_sft_v3/arc_agi_v1/small_reasoning_and_tools` | 10 | B | 8 |
| `nemotron_sft_v3/arc_agi_v1/small_reasoning_no_tools` | 10 | D | 3 |
| `nemotron_sft_v3/arc_agi_v1/small_tools_no_reasoning` | 10 | B | 7 |

### Nemotron SFT v3 competitive_programming_v2

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/competitive_programming_v2/competitive_coding_cpp` | 10 | B | 8 |
| `nemotron_sft_v3/competitive_programming_v2/competitive_coding_python` | 10 | B | 8 |
| `nemotron_sft_v3/competitive_programming_v2/exercism` | 10 | B | 8 |
| `nemotron_sft_v3/competitive_programming_v2/text_to_sql` | 10 | B | 8 |

### Nemotron SFT v3 cuda_v1

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/cuda_v1/train` | 10 | B | 8 |

### Nemotron SFT v3 finance_v1

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/finance_v1/train` | 10 | B | 8 |

### Nemotron SFT v3 instruction_following_chat_v2

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/instruction_following_chat_v2/reasoning_off` | 10 | B | 8.5 |
| `nemotron_sft_v3/instruction_following_chat_v2/reasoning_on` | 10 | A | 9 |

### Nemotron SFT v3 instruction_following_chat_v3

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/instruction_following_chat_v3/chat` | 10 | A | 9 |
| `nemotron_sft_v3/instruction_following_chat_v3/instruction_following` | 10 | B | 7 |

### Nemotron SFT v3 math_proofs_v1

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/math_proofs_v1/lean` | 10 | B | 7 |

### Nemotron SFT v3 math_proofs_v2

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/math_proofs_v2/train` | 10 | B | 8 |

### Nemotron SFT v3 math_v2

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/math_v2/high` | 10 | A | 9 |
| `nemotron_sft_v3/math_v2/low` | 10 | B | 9 |
| `nemotron_sft_v3/math_v2/medium` | 10 | B | 8 |

### Nemotron SFT v3 math_v3

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/math_v3/train` | 10 | B | 8 |

### Nemotron SFT v3 math_v4

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/math_v4/train` | 10 | B | 7 |

### Nemotron SFT v3 multilingual_v1

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/multilingual_v1/code_de` | 10 | B | 7 |
| `nemotron_sft_v3/multilingual_v1/code_es` | 10 | B | 7 |
| `nemotron_sft_v3/multilingual_v1/code_fr` | 10 | B | 7 |
| `nemotron_sft_v3/multilingual_v1/code_it` | 10 | B | 7 |
| `nemotron_sft_v3/multilingual_v1/code_ja` | 10 | B | 7 |
| `nemotron_sft_v3/multilingual_v1/code_zh` | 10 | B | 7 |
| `nemotron_sft_v3/multilingual_v1/math_de` | 10 | B | 7 |
| `nemotron_sft_v3/multilingual_v1/math_es` | 10 | B | 8 |
| `nemotron_sft_v3/multilingual_v1/math_fr` | 10 | B | 7 |
| `nemotron_sft_v3/multilingual_v1/math_it` | 10 | C | 6.5 |
| `nemotron_sft_v3/multilingual_v1/math_ja` | 10 | B | 7 |
| `nemotron_sft_v3/multilingual_v1/math_zh` | 10 | B | 7 |
| `nemotron_sft_v3/multilingual_v1/stem_de` | 10 | B | 8 |
| `nemotron_sft_v3/multilingual_v1/stem_es` | 10 | B | 8 |
| `nemotron_sft_v3/multilingual_v1/stem_fr` | 10 | A | 9 |
| `nemotron_sft_v3/multilingual_v1/stem_it` | 10 | B | 8 |
| `nemotron_sft_v3/multilingual_v1/stem_ja` | 10 | B | 8 |
| `nemotron_sft_v3/multilingual_v1/stem_zh` | 10 | A | 9 |

### Nemotron SFT v3 multilingual_v2

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/multilingual_v2/code_hi` | 10 | D | 4 |
| `nemotron_sft_v3/multilingual_v2/code_ja` | 10 | C | 6 |
| `nemotron_sft_v3/multilingual_v2/code_ko` | 10 | C | 6 |
| `nemotron_sft_v3/multilingual_v2/code_pt` | 10 | C | 6 |
| `nemotron_sft_v3/multilingual_v2/math_hi` | 10 | B | 7 |
| `nemotron_sft_v3/multilingual_v2/math_ja` | 10 | B | 7 |
| `nemotron_sft_v3/multilingual_v2/math_ko` | 10 | B | 7 |
| `nemotron_sft_v3/multilingual_v2/math_pt` | 10 | B | 8 |
| `nemotron_sft_v3/multilingual_v2/stem_hi` | 10 | A | 9 |
| `nemotron_sft_v3/multilingual_v2/stem_ja` | 10 | A | 9 |
| `nemotron_sft_v3/multilingual_v2/stem_ko` | 10 | A | 9 |
| `nemotron_sft_v3/multilingual_v2/stem_pt` | 10 | B | 10 |

### Nemotron SFT v3 opencode_v1

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/opencode_v1/agent_skills` | 10 | B | 8 |
| `nemotron_sft_v3/opencode_v1/agent_skills_question_tool` | 10 | C | 7 |
| `nemotron_sft_v3/opencode_v1/bash_only_tool` | 10 | B | 8 |
| `nemotron_sft_v3/opencode_v1/bash_only_tool_skills` | 10 | B | 8 |
| `nemotron_sft_v3/opencode_v1/general` | 10 | B | 7 |
| `nemotron_sft_v3/opencode_v1/question_tool` | 10 | B | 8 |

### Nemotron SFT v3 safety_v1

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/safety_v1/train` | 10 | A | 9 |

### Nemotron SFT v3 safety_v2

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/safety_v2/train` | 10 | B | 9.5 |

### Nemotron SFT v3 science_v2

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/science_v2/rqa` | 10 | B | 8 |
| `nemotron_sft_v3/science_v2/so` | 10 | B | 8 |
| `nemotron_sft_v3/science_v2/syn_mcq` | 10 | B | 8 |
| `nemotron_sft_v3/science_v2/vendor` | 10 | B | 9 |

### Nemotron SFT v3 swe_v1

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/swe_v1/r2e_gym` | 10 | A | 10 |

### Nemotron SFT v3 swe_v2

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `nemotron_sft_v3/swe_v2/agentless` | 10 | C | 6.5 |
| `nemotron_sft_v3/swe_v2/openhands_swe` | 10 | A | 9 |

### OpenSWE Traces

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `open_swe_traces/v1_0/openhands/minimax_m25/swe_rebench_v2` | 10 | B | 8 |
| `open_swe_traces/v1_0/openhands/qwen35_122b/swe_rebench_v2` | 10 | B | 8 |
| `open_swe_traces/v1_0/sweagent/minimax_m25/swe_rebench_v2` | 10 | B | 8 |
| `open_swe_traces/v1_0/sweagent/qwen35_122b/swe_rebench_v2` | 10 | B | 8 |
| `open_swe_traces/v1_1/minisweagent/qwen36_27b/scale_swe` | 10 | B | 8 |
| `open_swe_traces/v1_1/minisweagent/qwen36_27b/swe_rebench_v2` | 10 | B | 8 |
| `open_swe_traces/v1_1/openhands/deepseek_v4_flash/scale_swe` | 10 | B | 8 |
| `open_swe_traces/v1_1/openhands/qwen36_27b/scale_swe` | 10 | B | 8 |
| `open_swe_traces/v1_1/openhands/qwen36_27b/swe_rebench_v2` | 10 | B | 8 |
| `open_swe_traces/v1_1/sweagent/qwen36_27b/scale_swe` | 10 | B | 8 |
| `open_swe_traces/v1_1/sweagent/qwen36_27b/swe_rebench_v2` | 10 | B | 8 |
| `open_swe_traces/v1_2/minisweagent/qwen38_27b/scale_swe` | 10 | B | 8 |
| `open_swe_traces/v1_2/minisweagent/qwen38_27b/swe_rebench_v2` | 10 | B | 8 |

### Other SFT sources

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `agenttrove` | 10 | B | 7.5 |
| `agenttrove-glm53-compactions` | 10 | A | 9 |
| `coderforge` | 10 | B | 8 |
| `davinci-dev/env-native` | 10 | A | 9 |
| `glm-5.2-kernelgym-rollouts` | 10 | B | 8 |
| `gpt-oss-rollouts` | 10 | B | 9 |
| `identity-data` | 10 | A | 9 |
| `massive_function_calling` | 10 | A | 9 |
| `nemotron-terminal` | 10 | B | 8 |
| `numinamath-1.5` | 10 | A | 9 |
| `numinamath-tir` | 10 | B | 8 |
| `openthoughts4-code-glm-5.2-n4` | 10 | B | 8 |
| `superior-reasoning` | 10 | B | 9 |
| `swe-rebench-openhands` | 10 | B | 8 |
| `swe-zero-12m` | 10 | D | 9 |
| `synthetic-1` | 10 | A | 9 |
| `synthetic-misconceptions-conversations` | 10 | B | 8 |
| `ultrachat-persona-conversations` | 10 | B | 8 |
| `wildchat-glm53-format-completions` | 10 | B | 7.5 |

### Penfever GLM 5.2 Terminus 2

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `penfever-traces/glm52-terminus2/exp_rpt_crosscodeeval-csharp-v4` | 10 | B | 8 |
| `penfever-traces/glm52-terminus2/exp_rpt_curriculum-easy` | 10 | A | 9 |
| `penfever-traces/glm52-terminus2/exp_rpt_curriculum-medium` | 10 | B | 8 |
| `penfever-traces/glm52-terminus2/exp_rpt_e2egit-large` | 10 | A | 9 |
| `penfever-traces/glm52-terminus2/exp_rpt_e2egit-v2` | 10 | A | 9 |
| `penfever-traces/glm52-terminus2/exp_rpt_nemotron-cpp-v2` | 10 | B | 7 |
| `penfever-traces/glm52-terminus2/exp_rpt_stack-pytest-large-v2` | 10 | A | 9 |
| `penfever-traces/glm52-terminus2/exp_rpt_stack-pytest-v2` | 10 | A | 9 |
| `penfever-traces/glm52-terminus2/exp_rpt_unitsyn-python-v3` | 10 | B | 8 |
| `penfever-traces/glm52-terminus2/nemotron-gym-agent-calendar` | 10 | A | 9 |
| `penfever-traces/glm52-terminus2/nemotron-gym-instruction-following-structured` | 10 | A | 9 |
| `penfever-traces/glm52-terminus2/nemotron-gym-knowledge-web-search-mcqa` | 10 | A | 9 |
| `penfever-traces/glm52-terminus2/nl2bash-tasks-cleaned-oracle` | 10 | B | 7 |

### Penfever MiniMax M27 131K

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `penfever-traces/minimax-m27-131k/code-contests-noblock` | 10 | A | 9 |
| `penfever-traces/minimax-m27-131k/exp_rle_minimal_instructions-v3` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/exp_rpt_codenet-python-v2` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/exp_rpt_crosscodeeval-csharp-v4` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/exp_rpt_curriculum-easy` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/exp_rpt_curriculum-medium` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/exp_rpt_e2egit-large` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/exp_rpt_e2egit-v2` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/exp_rpt_ghactions-v3` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/exp_rpt_methods2test-large-v2` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/exp_rpt_methods2test-large-v3` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/exp_rpt_nemotron-cpp` | 10 | B | 7.5 |
| `penfever-traces/minimax-m27-131k/exp_rpt_nemotron-junit` | 10 | B | 7.5 |
| `penfever-traces/minimax-m27-131k/exp_rpt_pr` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/exp_rpt_pymethods2test-large` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/exp_rpt_pymethods2test-v3` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/exp_rpt_stack-bash-v3` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/exp_rpt_stack-junit-v6` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/exp_rpt_stack-pytest-large` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/exp_rpt_stack-pytest-v2` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/exp_rpt_unitsyn-python-large` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/exp_rpt_unitsyn-python-v3` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/inferredbugs-sandboxes-verifier` | 10 | B | 7.5 |
| `penfever-traces/minimax-m27-131k/llm-verifier-freelancer` | 10 | C | 6.5 |
| `penfever-traces/minimax-m27-131k/mix_h10_reward_binary-v2` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/mix_h10_reward_proportional-v2` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/mix_h10_reward_staged-v2` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/mix_h11_single_skill_only-v2` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/mix_h1_struggle_zone-v2` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/mix_h2_language_balanced-v2` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/mix_h2_language_proportional` | 10 | B | 7 |
| `penfever-traces/minimax-m27-131k/mix_h4_binary_easy` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/mix_h8_original_tests-v2` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/nemotron-code-oracle-filtered` | 10 | B | 9 |
| `penfever-traces/minimax-m27-131k/nemotron-gym-agent-calendar` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/nemotron-gym-agent-workplace-v2` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/nemotron-gym-competitive-coding` | 10 | A | 9 |
| `penfever-traces/minimax-m27-131k/nemotron-gym-instruction-following-calendar` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/nemotron-gym-instruction-following-structured` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/nemotron-gym-knowledge-web-search-mcqa` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/nemotron-gym-math-advanced-calculations-v3` | 10 | B | 8 |
| `penfever-traces/minimax-m27-131k/nl2bash-tasks-cleaned-oracle` | 10 | B | 7 |
| `penfever-traces/minimax-m27-131k/selfinstruct-naive-sandboxes-2-verified` | 10 | B | 7 |
| `penfever-traces/minimax-m27-131k/swegym-tasks-patched-validated-v5` | 10 | B | 7.5 |

### Penfever Qwen 3.5 122B 32K

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `penfever-traces/qwen35-122b-32k/code-contests-noblock` | 10 | A | 9 |
| `penfever-traces/qwen35-122b-32k/exp_rle_minimal_instructions-v3` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/exp_rpt_codenet-python-v2` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/exp_rpt_crosscodeeval-csharp-v4` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/exp_rpt_curriculum-easy` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/exp_rpt_curriculum-medium` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/exp_rpt_e2egit-large` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/exp_rpt_e2egit-v2` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/exp_rpt_ghactions-v3` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/exp_rpt_methods2test-large-v2` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/exp_rpt_methods2test-large-v3` | 10 | A | 9 |
| `penfever-traces/qwen35-122b-32k/exp_rpt_nemotron-junit` | 10 | B | 7.5 |
| `penfever-traces/qwen35-122b-32k/exp_rpt_pr` | 10 | B | 7 |
| `penfever-traces/qwen35-122b-32k/exp_rpt_pymethods2test-large` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/exp_rpt_pymethods2test-v3` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/exp_rpt_stack-bash-v3` | 10 | B | 7 |
| `penfever-traces/qwen35-122b-32k/exp_rpt_stack-junit-v6` | 10 | B | 7 |
| `penfever-traces/qwen35-122b-32k/exp_rpt_stack-pytest-large` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/exp_rpt_stack-pytest-v2` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/exp_rpt_unitsyn-python-large` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/exp_rpt_unitsyn-python-v3` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/inferredbugs-sandboxes-verifier` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/llm-verifier-freelancer` | 10 | B | 7 |
| `penfever-traces/qwen35-122b-32k/mix_h10_reward_binary-v2` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/mix_h10_reward_proportional-v2` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/mix_h10_reward_staged-v2` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/mix_h11_single_skill_only-v2` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/mix_h1_struggle_zone-v2` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/mix_h2_language_balanced-v2` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/mix_h2_language_proportional` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/mix_h4_binary_easy` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/mix_h8_original_tests-v2` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/nemotron-code-oracle-filtered` | 10 | A | 9 |
| `penfever-traces/qwen35-122b-32k/nemotron-gym-agent-calendar` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/nemotron-gym-agent-workplace-v2` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/nemotron-gym-competitive-coding` | 10 | A | 9 |
| `penfever-traces/qwen35-122b-32k/nemotron-gym-instruction-following-calendar` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/nemotron-gym-instruction-following-structured` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/nemotron-gym-instruction-following-v2` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/nemotron-gym-knowledge-mcqa` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/nemotron-gym-knowledge-openqa-v2` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/nemotron-gym-knowledge-web-search-mcqa` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/nemotron-gym-math-advanced-calculations-v3` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/nemotron-gym-safety-v2` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/nemotron-math-oracle-filtered` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-32k/nl2bash-tasks-cleaned-oracle` | 10 | B | 7 |
| `penfever-traces/qwen35-122b-32k/selfinstruct-naive-sandboxes-2-verified` | 10 | B | 7 |
| `penfever-traces/qwen35-122b-32k/swesmith-oracle-filtered` | 10 | C | 6 |

### Penfever Qwen 3.5 122B OpenCode

| Source | Rows | Grade | Median / 10 |
| --- | ---: | :---: | ---: |
| `penfever-traces/qwen35-122b-131k-opencode/code-contests-noblock` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rle_adversarial` | 10 | C | 7.5 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_crosscodeeval-csharp-v4` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_crosscodeeval-java` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_curriculum-easy` | 10 | B | 9 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_curriculum-hard` | 10 | B | 9 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_curriculum-medium` | 10 | B | 9 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_e2egit-large` | 10 | A | 9 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_e2egit-v2` | 10 | B | 9 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_ghactions-v3` | 10 | B | 8.5 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_issue` | 10 | B | 9 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_methods2test-large-v3` | 2 | C | 5 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_multifile` | 10 | B | 9.5 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_nemotron-cpp-v2` | 10 | C | 7 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_nemotron-junit` | 10 | B | 9 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_pr` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_pymethods2test-large` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_pymethods2test-v3` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_stack-junit-v6` | 6 | C | 5 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_stack-pytest-large` | 10 | B | 7.5 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_stack-pytest-v2` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_unitsyn-python-large` | 10 | B | 7.5 |
| `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_unitsyn-python-v3` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-131k-opencode/inferredbugs-sandboxes-verifier` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-131k-opencode/llm-verifier-freelancer` | 10 | C | 6.5 |
| `penfever-traces/qwen35-122b-131k-opencode/mix_h4_binary_easy` | 10 | B | 8 |
| `penfever-traces/qwen35-122b-131k-opencode/nemotron-code-oracle-filtered` | 10 | B | 9 |
| `penfever-traces/qwen35-122b-131k-opencode/nemotron-gym-agent-calendar` | 10 | B | 9 |
| `penfever-traces/qwen35-122b-131k-opencode/nemotron-gym-instruction-following-structured` | 10 | A | 9 |
| `penfever-traces/qwen35-122b-131k-opencode/nemotron-gym-knowledge-web-search-mcqa` | 10 | A | 9 |
| `penfever-traces/qwen35-122b-131k-opencode/nemotron-gym-math-advanced-calculations-v3` | 10 | B | 9 |
| `penfever-traces/qwen35-122b-131k-opencode/nl2bash-tasks-cleaned-oracle` | 10 | A | 10 |
