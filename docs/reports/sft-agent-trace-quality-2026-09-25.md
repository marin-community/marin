# SFT agent trace quality spot check

Ten converted chat examples from each of 112 registered Terminus sources or partitions were scored for behavior worth imitating in SFT. Fourteen sources received A, 94 B, and four C. None received D or F. This is an exploratory screen of trace behavior, not a measure of task success or training effect. The source grades in the 2026-09-18 mixture notes came from one record per source and used no written rubric; the grades here are not directly comparable.

## Method

We sampled ten accepted structured-chat examples per source or partition, using pinned dataset revisions and the Terminus adapters from [PR #9434](https://github.com/marin-community/marin/pull/9434). The review covers 107 Penfever repositories, AgentTrove, and four Nemotron Terminal partitions. Thirty-seven Penfever sources used source-local hashed offsets from Hugging Face; 70 used hashed offsets within the first 100 accepted examples in regional storage. AgentTrove was stratified across its three task sources (4 exp_tas, 3 r2egym, 3 swesmith). The Nemotron samples were spread across subject files; dataset adapters used four code, three math, and three SWE examples. These are deterministic convenience samples, not uniform corpus-wide random samples.

GPT-6-luna reviewers applied the rubric to bounded excerpts of the task, analyses, commands, observations, ending, and post-run outcome. Full command bodies and code correctness were not audited. Parser acceptance is measured separately from content quality. Row identifiers, revisions, score components, and sample offsets are in the [grade data](sft-agent-trace-quality-2026-09-25-grades.csv).

Each example gets 0, 1, or 2 points on five dimensions:

| Dimension | 2 points | 1 point | 0 points |
| --- | --- | --- | --- |
| Task grounding | Actions address the task and its constraints | Mostly on task, with drift | Wrong task or invented requirements |
| Substantive progress | Appropriate actions or answer make useful progress | Partial progress or detours | No useful progress or unproductive loop |
| Evidence and correction | Checks relevant output and responds to visible errors | Limited checks | Ignores visible contrary evidence or claims success without support |
| Ending and handoff | Coherent completion or honest unresolved state | Weak handoff or unfinished continuation | False success claim or misleading ending |
| Training signal | Reasoning and tool use are clear and worth imitating | Duplication, boilerplate, or harness ritual | Severe repetition, unsupported certainty, or parser leakage |

The source's `result` and `verifier_output` fields are post-run metadata. They describe the outcome; they are not messages the assistant saw. A failed or unverified episode is not penalized merely for failing. A hidden verifier failure does not establish that the assistant ignored an error. The JSON command's `analysis` field is visible output, not a parsed thinking turn.

We assign source grades from the ten row scores:

| Grade | Rule |
| --- | --- |
| A | Median 9–10, at least eight rows score 8+, and no serious harmful pattern |
| B | Median at least 7 but the A rule is not met; at least eight rows score 5+, and at most one serious harmful example |
| C | Median 5–6.5, or useful rows mixed with several low-quality examples |
| D | Median 3–4.5, or repeated misleading endings, loops, or weak task grounding |
| F | Median below 3, or pervasive behavior that should not be imitated |

Severe defects can cap the letter grade even when the median is higher. All 112 sources supplied ten accepted rows.

## Results

| Source group | Sources | A | B | C | D/F |
| --- | ---: | ---: | ---: | ---: | ---: |
| Penfever MiniMax M27 131K | 45 | 2 | 42 | 1 | 0 |
| Penfever Qwen 3.5 122B 32K | 49 | 4 | 44 | 1 | 0 |
| Penfever GLM 5.2 Terminus 2 | 13 | 8 | 5 | 0 | 0 |
| AgentTrove | 1 | 0 | 1 | 0 | 0 |
| Nemotron Terminal partitions | 4 | 0 | 2 | 2 | 0 |
| Total | 112 | 14 | 94 | 4 | 0 |

The four C sources warrant a closer read before using them at full weight:

| Source | Median | Sampled behavior |
| --- | ---: | --- |
| `penfever-traces/minimax-m27-131k/llm-verifier-freelancer` | 6.5 | A storefront job starts a broad implementation and remains unfinished. Another asks for a codebase audit without providing a repository; the assistant identifies the missing input. These freelance postings often fit the terminal harness poorly. |
| `penfever-traces/qwen35-122b-32k/swesmith-oracle-filtered` | 6 | A Moto change is focused and reports passing tests, while another trace stops after a basic test and before checking duplicate IDs. Several rows have limited visible validation or handoff. |
| `nemotron/skill_based_medium` | 5 | Three rows complete useful data tasks. Others end during exploration, dependency conflict resolution, or debugging without a result. |
| `nemotron/skill_based_mixed` | 5 | Two rows score 2/10: one reads files without a diagnosis, and another checks packages before implementing the requested solver. Most other rows make partial progress. |

Several higher-grade examples show useful behavior. Nemotron dataset-adapter `code.parquet#row=6` revises a mistaken expected value, checks the inversion-count solution exhaustively, and scores 9/10. In contrast, `code.parquet#row=13` stops at compilation with its own DFS concern unresolved and scores 6/10. AgentTrove scores B (median 7.5): sampled Tezos advice is mostly coherent, while two sampled repository tasks remain unfinished. Its long shell-written answers are also a weak tool-use pattern. Competitive-programming rows often show focused implementation and sample checks; repository rows sometimes repeat completion checklists or assert checks whose returned output is absent from the bounded excerpt.

The grade data preserve the ten scores for every source. The tables below give each source's grade and median.

## Limits

This small, deterministic sample can miss rare loops, harmful replies, or failures concentrated later in a corpus. The source grade judges the visible conversational behavior, including honest unsuccessful attempts. It does not certify code correctness, tool execution, task reward, or suitability for a particular training mix. In particular, post-run `result` and `verifier_output` may report failure without showing that the assistant saw that failure. The compact review excerpts also limit how confidently a claimed test can be corroborated. Recheck full traces and outcome distributions before using these grades as a hard data filter.

## Per-source grades

### Penfever MiniMax M27 131K

| Source | Grade | Median / 10 |
| --- | :---: | ---: |
| `code-contests-noblock` | A | 9 |
| `exp_rle_minimal_instructions-v3` | B | 8 |
| `exp_rpt_codenet-python-v2` | B | 8 |
| `exp_rpt_crosscodeeval-csharp-v4` | B | 8 |
| `exp_rpt_curriculum-easy` | B | 8 |
| `exp_rpt_curriculum-medium` | B | 8 |
| `exp_rpt_e2egit-large` | B | 8 |
| `exp_rpt_e2egit-v2` | B | 8 |
| `exp_rpt_ghactions-v3` | B | 8 |
| `exp_rpt_methods2test-large-v2` | B | 8 |
| `exp_rpt_methods2test-large-v3` | B | 8 |
| `exp_rpt_nemotron-cpp` | B | 7.5 |
| `exp_rpt_nemotron-junit` | B | 7.5 |
| `exp_rpt_pr` | B | 8 |
| `exp_rpt_pymethods2test-large` | B | 8 |
| `exp_rpt_pymethods2test-v3` | B | 8 |
| `exp_rpt_stack-bash-v3` | B | 8 |
| `exp_rpt_stack-junit-v6` | B | 8 |
| `exp_rpt_stack-pytest-large` | B | 8 |
| `exp_rpt_stack-pytest-v2` | B | 8 |
| `exp_rpt_unitsyn-python-large` | B | 8 |
| `exp_rpt_unitsyn-python-v3` | B | 8 |
| `inferredbugs-sandboxes-verifier` | B | 7.5 |
| `llm-verifier-freelancer` | C | 6.5 |
| `mix_h10_reward_binary-v2` | B | 8 |
| `mix_h10_reward_proportional-v2` | B | 8 |
| `mix_h10_reward_staged-v2` | B | 8 |
| `mix_h11_single_skill_only-v2` | B | 8 |
| `mix_h1_struggle_zone-v2` | B | 8 |
| `mix_h2_language_balanced-v2` | B | 8 |
| `mix_h2_language_proportional` | B | 7 |
| `mix_h4_binary_easy` | B | 8 |
| `mix_h8_original_tests-v2` | B | 8 |
| `nemotron-code-oracle-filtered` | B | 9 |
| `nemotron-gym-agent-calendar` | B | 8 |
| `nemotron-gym-agent-workplace-v2` | B | 8 |
| `nemotron-gym-competitive-coding` | A | 9 |
| `nemotron-gym-identity-following-v2` | B | 8 |
| `nemotron-gym-instruction-following-calendar` | B | 8 |
| `nemotron-gym-instruction-following-structured` | B | 8 |
| `nemotron-gym-knowledge-web-search-mcqa` | B | 8 |
| `nemotron-gym-math-advanced-calculations-v3` | B | 8 |
| `nl2bash-tasks-cleaned-oracle` | B | 7 |
| `selfinstruct-naive-sandboxes-2-verified` | B | 7 |
| `swegym-tasks-patched-validated-v5` | B | 7.5 |

### Penfever Qwen 3.5 122B 32K

| Source | Grade | Median / 10 |
| --- | :---: | ---: |
| `code-contests-noblock` | A | 9 |
| `exp_rle_minimal_instructions-v3` | B | 8 |
| `exp_rpt_codenet-python-v2` | B | 8 |
| `exp_rpt_crosscodeeval-csharp-v4` | B | 8 |
| `exp_rpt_curriculum-easy` | B | 8 |
| `exp_rpt_curriculum-medium` | B | 8 |
| `exp_rpt_e2egit-large` | B | 8 |
| `exp_rpt_e2egit-v2` | B | 8 |
| `exp_rpt_ghactions-v3` | B | 8 |
| `exp_rpt_methods2test-large-v2` | B | 8 |
| `exp_rpt_methods2test-large-v3` | A | 9 |
| `exp_rpt_nemotron-junit` | B | 7.5 |
| `exp_rpt_pr` | B | 7 |
| `exp_rpt_pymethods2test-large` | B | 8 |
| `exp_rpt_pymethods2test-v3` | B | 8 |
| `exp_rpt_stack-bash-v3` | B | 7 |
| `exp_rpt_stack-junit-v6` | B | 7 |
| `exp_rpt_stack-pytest-large` | B | 8 |
| `exp_rpt_stack-pytest-v2` | B | 8 |
| `exp_rpt_unitsyn-python-large` | B | 8 |
| `exp_rpt_unitsyn-python-v3` | B | 8 |
| `inferredbugs-sandboxes-verifier` | B | 8 |
| `llm-verifier-freelancer` | B | 7 |
| `mix_h10_reward_binary-v2` | B | 8 |
| `mix_h10_reward_proportional-v2` | B | 8 |
| `mix_h10_reward_staged-v2` | B | 8 |
| `mix_h11_single_skill_only-v2` | B | 8 |
| `mix_h1_struggle_zone-v2` | B | 8 |
| `mix_h2_language_balanced-v2` | B | 8 |
| `mix_h2_language_proportional` | B | 8 |
| `mix_h4_binary_easy` | B | 8 |
| `mix_h8_original_tests-v2` | B | 8 |
| `nemotron-code-oracle-filtered` | A | 9 |
| `nemotron-gym-agent-calendar` | B | 8 |
| `nemotron-gym-agent-workplace-v2` | B | 8 |
| `nemotron-gym-competitive-coding` | A | 9 |
| `nemotron-gym-identity-following-v2` | B | 8 |
| `nemotron-gym-instruction-following-calendar` | B | 8 |
| `nemotron-gym-instruction-following-structured` | B | 8 |
| `nemotron-gym-instruction-following-v2` | B | 8 |
| `nemotron-gym-knowledge-mcqa` | B | 8 |
| `nemotron-gym-knowledge-openqa-v2` | B | 8 |
| `nemotron-gym-knowledge-web-search-mcqa` | B | 8 |
| `nemotron-gym-math-advanced-calculations-v3` | B | 8 |
| `nemotron-gym-safety-v2` | B | 8 |
| `nemotron-math-oracle-filtered` | B | 8 |
| `nl2bash-tasks-cleaned-oracle` | B | 7 |
| `selfinstruct-naive-sandboxes-2-verified` | B | 7 |
| `swesmith-oracle-filtered` | C | 6 |

### Penfever GLM 5.2 Terminus 2

| Source | Grade | Median / 10 |
| --- | :---: | ---: |
| `exp_rpt_crosscodeeval-csharp-v4` | B | 8 |
| `exp_rpt_curriculum-easy` | A | 9 |
| `exp_rpt_curriculum-medium` | B | 8 |
| `exp_rpt_e2egit-large` | A | 9 |
| `exp_rpt_e2egit-v2` | A | 9 |
| `exp_rpt_nemotron-cpp-v2` | B | 7 |
| `exp_rpt_stack-pytest-large-v2` | A | 9 |
| `exp_rpt_stack-pytest-v2` | A | 9 |
| `exp_rpt_unitsyn-python-v3` | B | 8 |
| `nemotron-gym-agent-calendar` | A | 9 |
| `nemotron-gym-instruction-following-structured` | A | 9 |
| `nemotron-gym-knowledge-web-search-mcqa` | A | 9 |
| `nl2bash-tasks-cleaned-oracle` | B | 7 |

### AgentTrove

| Source | Grade | Median / 10 |
| --- | :---: | ---: |
| `agenttrove` | B | 7.5 |

### Nemotron Terminal

| Source | Grade | Median / 10 |
| --- | :---: | ---: |
| `nemotron/dataset_adapters` | B | 8 |
| `nemotron/skill_based_easy` | B | 9 |
| `nemotron/skill_based_medium` | C | 5 |
| `nemotron/skill_based_mixed` | C | 5 |
