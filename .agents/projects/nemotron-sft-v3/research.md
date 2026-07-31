# Nemotron post-training v3 source audit

Date: 2026-07-30

Effort: high. The audit stopped after the official collection and dataset cards,
pinned repository metadata, bounded row and identifier samples, existing Marin
sources, Echo, and GitHub history agreed on the version relationships. No bulk
dataset download was used.

## Decisions

- Register 21 non-superseded SFT-stage repositories from the official collection
  under 74 versioned source keys. This includes the 16 repositories named in
  [issue #7826](https://github.com/marin-community/marin/issues/7826) and seven
  omitted repositories, minus Safety-v1 and Math-v3.
- Register Safety-v2 without Safety-v1. Safety-v2 reuses the v1 English prompts
  and responses, applies additional filtering, and adds six translated subsets.
- Register Math-v4 without Math-v3. NVIDIA calls v4 a direct replacement. Keep
  Math-v2 because it is the 7-million-trajectory seed corpus, not an earlier
  generation of the same v3/v4 response data.
- Register Multilingual-v2 Japanese without the corresponding v1 Japanese
  lanes. Keep the other Multilingual-v1 languages.
- Register Math-Proofs-v1 and v2. V1 contains Lean 4 formal proofs; v2 contains
  natural-language proof, verification, and meta-verification traces over 5,752
  AoPS problems drawn from v1. NVIDIA describes v2 as an extension, not a
  replacement for v1.
- Register Agentic-v1, Competitive-Programming-v1, SWE-v1, and Finance-v1.
  Their later related releases regenerate trajectories or add task families;
  NVIDIA does not identify them as superseded.
- Keep SWE-v2 and SWE-v3 separate. NVIDIA does not describe v3 as a replacement,
  and bounded UUID samples found no exact trajectory overlap.
- Keep every ARC-AGI config addressable. The `small_*` and `large_*` configs are
  overlapping wide and deep samples from the same successful-run pool. A mixture
  should choose one size family rather than union both. Exact matching rows
  receive the same content ID, which permits downstream cross-source deduplication.
- Keep every OpenCode split addressable. The six splits contain alternate
  trajectories over substantially shared questions, so mixture weights should
  not treat their task counts as independent.
- The Instruction-Following-Chat-v3 `chat` source contains only rows with a
  non-null user prompt. NVIDIA withholds LMSYS and WildChat prompts and provides
  a reconstruction script that requires separately authorized source datasets.
  The transform counts and drops those incomplete rows instead of writing
  promptless training documents.

## Evidence

The [official v3 collection](https://huggingface.co/collections/nvidia/nemotron-post-training-v3)
contains all 16 repositories from issue #7826. It also contains earlier versions
and adjacent SFT repositories, so collection membership is not evidence that the
members are disjoint.

The [Safety-v2 card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Safety-v2)
states that its English prompts and responses are the same as Safety-v1. A
bounded 4 MiB sample found 29 shared UUIDs; all 29 had byte-equivalent
`messages`. Safety-v2 has 43,521 English rows after additional filtering, plus
German, Spanish, French, Italian, Japanese, and Chinese translations.

The [Math-v4 card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v4)
calls v4 a direct replacement for earlier releases. Math-v3 has 3,638,783
DeepSeek V3.2/Speciale trajectories; Math-v4 has 545,431 DeepSeek-V4-Pro
trajectories. Bounded samples found no shared UUID, problem-text, or message
hashes, but both versions derive prompts from Math-v2. Math-v4 contains 285,516
`cot` and 259,915 `tir` rows; the registry preserves both through the advertised
`train` split.

The [Multilingual-v2 card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Multilingual-v2)
recommends replacing Multilingual-v1 Japanese with the v2 Japanese data.
Bounded Japanese samples found no shared UUID, user-text, or reasoning hashes,
so this is a quality supersession rather than an exact repack. Multilingual-v1
also reuses UUIDs across translated languages: pairwise samples commonly shared
8 to 10 of 10 seed UUIDs.

The [ARC-AGI-v1 card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-ARC-AGI-v1)
describes `small_*` as one solution per problem and `large_*` as a deeper sample
of the same successful-run pool. Bounded matching-config samples found exact
full-row overlap, including 26 of 100 `reasoning_and_tools` rows.

The [Instruction-Following-Chat-v3 repository](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v3)
ships `prepare_chat_prompts.py`. The script restores deliberately nulled LMSYS
and WildChat seed prompts by loading `lmsys/lmsys-chat-1m` and
`allenai/WildChat-1M`. Samples across the 17 GB chat file found null first-user
content at several offsets. The `instruction_following` split is self-contained
and additive to v2.

The [SWE-v3 card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3)
does not claim that v3 replaces SWE-v2. Samples covering 29,748 v3 UUIDs found
no intersection with 32 nested SWE-v2 UUIDs, 67,074 Nebius SWE-rebench
trajectory IDs, or 4,611 sampled CoderForge IDs. Different identifier schemes
mean task-level overlap with SWE-rebench, R2E-Gym, CoderForge, and SWE-ZERO is
still unresolved.

The [OpenCode-v1 manifest](https://huggingface.co/datasets/nvidia/Nemotron-SFT-OpenCode-v1/resolve/556d5237acff203f3e1a0be49428634c3606cda2/manifest.json)
lists six trajectory variants. In the first 100 rows per split, pairwise
question overlap ranged from 39 to 67 rows. Full trajectories differ by tool
configuration, skills, and `AGENTS.md` context.

## Additions outside issue #7826

The official collection has seven additional SFT-stage repositories that are
not named in issue #7826:

- `Nemotron-Math-Proofs-v1`
- `Nemotron-Math-Proofs-v2`
- `Nemotron-Agentic-v1`
- `Nemotron-Competitive-Programming-v1`
- `Nemotron-Math-v2`
- `Nemotron-SWE-v1`
- `Nemotron-SpecializedDomains-Finance-v1`

All seven are registered. Math-v2 exposes high, medium, and low reasoning lanes;
the three high Parquet shards form one source key. Competitive-Programming-v1
exposes C++, Python, and InfiniByte lanes; each key combines its two physical
JSONL shards. This prevents storage shards from becoming separate mixture
components.

The [Math-Proofs-v2 card](https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v2)
reports 82,737 rows and 5,000,839,123 tokens at revision
`7665d7f1d006fd89aa852a9dab8060c60b63f814`. That revision repairs incorrect
`subset` labels in the revision used by closed PR #5971. Its single
`default/train` source retains `proof`, `verification`, and `meta-verification`
as row metadata.

The [Finance-v1 card](https://huggingface.co/datasets/nvidia/Nemotron-SpecializedDomains-Finance-v1)
reports 326,698 synthetic Q&A rows grounded in SEC filings. Bounded samples had
system, user, and assistant messages with non-empty assistant
`reasoning_content`. Finance-v1 does not overlap the existing
`nemotron_specialized_v1_1/economics` multiple-choice pretraining source.

## Negative results and prior work

- No data-file LFS object IDs matched across the audited collection repositories or the
  audited earlier NVIDIA versions. This rules out whole-file renames, not
  row-level or seed-task overlap.
- Bounded samples found no exact row repack between Math-v3 and Math-v4,
  Multilingual-v1 and v2 Japanese, Science-v1 and v2, or SWE-v2 and v3.
- Existing `nemotron_sft/sft_code`, `sft_general`, and `sft_math` sources refer
  to Nemotron-Pretraining-SFT-v1. That gated, pretraining-stage corpus is not a
  rename of the post-training v3 collection.
- The older Nemotron Post-Training v1/v2 aggregate repositories predate these
  versioned releases. The gated v2 data prevented an exact row comparison.
- Prior adapters in [#5968](https://github.com/marin-community/marin/pull/5968),
  [#5969](https://github.com/marin-community/marin/pull/5969),
  [#5970](https://github.com/marin-community/marin/pull/5970),
  [#5971](https://github.com/marin-community/marin/pull/5971),
  [#5972](https://github.com/marin-community/marin/pull/5972),
  [#5978](https://github.com/marin-community/marin/pull/5978),
  [#5979](https://github.com/marin-community/marin/pull/5979), and
  [#6270](https://github.com/marin-community/marin/pull/6270) were closed
  without merging. They informed schema checks but do not register sources on
  the current main branch.
