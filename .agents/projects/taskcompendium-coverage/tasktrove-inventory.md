# TaskTrove `.8` inventory and bounded fidelity audit

The current authoritative release is the HF mirror `open-athena/task-trove`, backed by `s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.9`. Its manifest reports `input_tasks=1,739,326`, `clean_tasks=1,449,686`, 93 source keys, and 43 kept source details. The direct HF Parquet mirror has 1,449,686 rows. See the current authenticated manifest and sample evidence in [tasktrove-release-09-inventory-2026-09-14.md](tasktrove-release-09-inventory-2026-09-14.md).

The remainder of this file is a historical `.8` bounded-fidelity audit retained for provenance. Do not use its old HF catalog or POC counts as the current release inventory.

An independent fresh check did recover the **raw input** catalog without reading the
clean release: HTTP-range Parquet footer reads against
`open-thoughts/TaskTrove@0292300` succeeded for 93 default source files and 40
deprecated files. The default source footers sum to 1,739,326 rows, with the
producer's 93 source decisions dividing them into 43 `keep` sources (1,526,908
raw rows) and 50 `drop` sources (212,418 raw rows). These remain input counts;
the authoritative clean membership and post-filter counts are in the S3
release manifest/ledger. The full command evidence and per-source table are in
[tasktrove-release-access-2026-09-13.md](tasktrove-release-access-2026-09-13.md).

The local POC still provides **recorded sample provenance**. Its manifest names `.8`, records fixture SHA-256 values, and exports `tasktrove-<fixture-row>` tasks. That is weaker than a remote membership check: fixture `task.toml` files record the TaskTrove source/path and upstream identity but no clean-release version, while `build_poc.py` supplies the fixture filename row to `read_archive()`, which creates the `.8` source label. The candidate JSON therefore calls these `fixture_row`, not `release_row`.

The producer code at [`ccc5ff24`](https://github.com/marin-community/marin/blob/ccc5ff24cd16112a1e68c5dc6ca9f5ee4f6a52c2/experiments/post_training/tasktrove/pipeline.py) pins `.8`, raw `open-thoughts/TaskTrove@0292300`, and verifier [`b2b68d8`](https://github.com/marin-community/marin/tree/b2b68d8b0a770cdc0ab3903780172c4b3eea81b1/lib/tasktrove-verify). The local TaskCompendium source corrects the supplied inspected-converter SHA to [`bef70bb`](https://github.com/marin-community/marin/commit/bef70bb8584d7e0c1391d88c9eacb1605b551a90): `1391d`, rather than the prompt's `139d`/`1398d`. Git confirms `ccc5ff` is an ancestor of that revision and the verifier is an ancestor of the producer.

The raw Hub catalog has 133 source directories. Its viewer currently reports 1,604,128 default and 296,667 deprecated rows, which conflicts with the producer-era input count; it is useful only to establish source presence, never as a clean `.8` count. The Hub catalog is an explicitly non-authoritative fallback now that the S3 manifest is available.

| Area | raw / producer finding | TaskCompendium status |
| --- | --- | --- |
| MCQA | `knowledge-mcqa-v2` kept via `nemotron_mcqa` | Supported answer importer |
| NeMo math | OpenMathReasoning, Stack Overflow and v5 kept | Present in TaskTrove; not supported by the current math importer, which only accepts `all_puzzles` |
| Instruction following | IFEval and structured paths kept | Present; no TaskTrove importer |
| Code generation | Direct NeMo `competitive-coding-v2` is kept with 50 hidden stdin/stdout cases; Codeforces and DCAgent are separate sources with the same shape | Direct NeMo source is unsupported; generic Codeforces/pytest paths are supported with immutable toolchain images |
| Function-calling pivot | dropped: frozen transcript, exact next-call matching | Unsupported |
| Workplace | raw `agent-workplace-v2` exists but has no `ccc5ff` source-policy entry | Inclusion and count unknown; unsupported |
| SWE | Direct NeMo SWE pivot is dropped because no repository is supplied; rebench and swesmith are non-NeMo sources | No direct NeMo SWE support; generic repository paths remain outside current `tasktrove_coding` scope |
| Web search | dropped: prompt promises a tool that is absent | Unsupported |
| ReasoningGym | kept with library scoring | Present; no TaskTrove importer |
| Structured outputs | kept; JSON/YAML/TOML have stronger checks | JSON/XML importer exists, but both recorded examples show semantic false positives |
| Calendar v2 | kept as static scheduling: final JSON is checked against pre-materialized expected events, with no live calendar tool | Present; no TaskTrove importer. This static contract comes from the source, rather than a conversion loss |
| Indian banking | no literal raw source-directory name | The upstream `NPCI/nemo-gym-indian-banking` join is `task_id`, but no TaskTrove inclusion is proven; this does not rule out a task mentioning banking |
| Indirect prompt injection | dropped: any unrelated action passes the negative-only checker | Unsupported |

The original converter registry is at [`registry.py`](https://github.com/marin-community/marin/blob/ccc5ff24cd16112a1e68c5dc6ca9f5ee4f6a52c2/experiments/post_training/tasktrove/converters/registry.py). Relevant implementations are [`nemotron_gym.py`](https://github.com/marin-community/marin/blob/ccc5ff24cd16112a1e68c5dc6ca9f5ee4f6a52c2/experiments/post_training/tasktrove/converters/nemotron_gym.py), [`nemotron_openqa.py`](https://github.com/marin-community/marin/blob/ccc5ff24cd16112a1e68c5dc6ca9f5ee4f6a52c2/experiments/post_training/tasktrove/converters/nemotron_openqa.py), [`nemotron_reasoning.py`](https://github.com/marin-community/marin/blob/ccc5ff24cd16112a1e68c5dc6ca9f5ee4f6a52c2/experiments/post_training/tasktrove/converters/nemotron_reasoning.py), [`nemotron_structured_outputs.py`](https://github.com/marin-community/marin/blob/ccc5ff24cd16112a1e68c5dc6ca9f5ee4f6a52c2/experiments/post_training/tasktrove/converters/nemotron_structured_outputs.py), [`agent_calendar.py`](https://github.com/marin-community/marin/blob/ccc5ff24cd16112a1e68c5dc6ca9f5ee4f6a52c2/experiments/post_training/tasktrove/converters/agent_calendar.py), and [`prompt_injection.py`](https://github.com/marin-community/marin/blob/ccc5ff24cd16112a1e68c5dc6ca9f5ee4f6a52c2/experiments/post_training/tasktrove/converters/prompt_injection.py).

The per-family JSON evidence anchors resolve to the pinned [`source_verdicts.json`](https://github.com/marin-community/marin/blob/ccc5ff24cd16112a1e68c5dc6ca9f5ee4f6a52c2/experiments/post_training/tasktrove/source_verdicts.json). Raw-catalog source-presence comes from the exact [Hub revision endpoint](https://huggingface.co/api/datasets/open-thoughts/TaskTrove/revision/0292300) and its [Viewer size endpoint](https://datasets-server.huggingface.co/size?dataset=open-thoughts%2FTaskTrove&revision=0292300); it has high confidence for requested raw metadata, but cannot recover the historical `.8` clean membership.

Five concrete fixture candidates and their exact upstream joins are in [tasktrove-inventory.json](tasktrove-inventory.json). They deliberately test one actual source shape per support boundary, rather than proposing new importers. The decisive fidelity result is `structured/json-row-16636.tar.gz`: its JSON-schema import accepted an invented home-buying answer with reward 1. `structured/xml-row-16634.tar.gz` has the companion failure: invented values and nonnumeric text pass a name-only XML check. Both are recorded in [the problematic-task ledger](../../logbooks/taskcompendium-problematic-tasks.md). A supported syntax path is not evidence that the retained task faithfully measures extraction.
