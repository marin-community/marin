# Deterministic tool conversion

## Scope and limits — read before reuse

This is a tool-calling SFT campaign experiment, not a general conversion library.
It supports only:

- The campaign's pinned Exp3 selection, additional-source selection, and full-corpus
  target selection. `run.py` checks their canonical JSON SHA-256 values before
  reading shards or writing outputs; edited catalogs and unknown schemas fail.
- The pinned Llama-3-style chat serialization and tokenizer, SHA-256
  `6a61ca2105a54e9984169b409581f31cc50330d1b47104af13c221b05e462e45`.
  Required IDs are `<|start_header_id|>` = 128006, `<|end_header_id|>` = 128007,
  `<tool_call>` = 128005 and `assistant` = 78191. The reviewed tokenizer supplies
  the remaining message, response and reasoning delimiters.
- The tool names and argument contracts in the mapping table below, including
  `stateful_python_code_exec` / `execute_bash` → `bash` and `think` / `submit` /
  `finish` → native reasoning or completion. The preserved real-tool set is the
  explicit `REAL_TOOLS` allowlist in `convert.py`.
- CoreWeave `marin-us-east-02a` credential/signing-region configuration, with the
  campaign's pinned input caches and separate receipt outputs. This is not a
  general storage or cluster launcher.
- The hardcoded source exclusions: names containing `terminus`, exact
  `agenttrove`, and Nemotron path components `interactive_agent` or `tool_calling`.

It does **not** support other chat templates or tokenizers, other harness tool
sets, Terminus-2 conversion, ARC evaluation, multiple or parallel native calls
within one assistant turn, mixed-order call/reasoning cases, rich notebook output,
or any operation left residual by the mapping rules. Sequential notebook cells
across turns remain supported and share one persistent kernel. Recognized but
unsupported campaign traces remain residual; unknown formats, tool sets,
tokenizers and selection schemas raise `CampaignScopeError` with a pointer here.
Explicitly excluded sources and tool-free rows receive no conversion.

Do not reuse for other datasets; write a new converter, or generalize this one in a reviewed PR first.

## Conversion contract

`experiments.tool_conversion.run` converts pinned serialized conversations into
separate, receipt-bound archives. It never executes source programs or asks an
LLM to verify a deterministic change. Acceptance means that the operation mapping,
argument shape, call/result pairing, serialization and assistant mask passed the
adapter checks. Historical observations remain historical; this does not certify
the original task's correctness or reproduce its environment.

## Selection and mapping

Skip Nemotron `interactive_agent` and `tool_calling` sources, Terminus-2 sources
and terminal JSON protocols before conversion. Skip documents with no native
calls. Preserve real OpenCode, Pi, Codex, OpenHands and SWE-agent tools, including
`str_replace_editor`. A failed deterministic precondition leaves the complete
original trace in the residual selection; no partially converted row is accepted.

| Source operation | Target | Deterministic preconditions |
| --- | --- | --- |
| `stateful_python_code_exec(code)` | `bash(command)` | Declared notebook execution; valid Python; text results; no timeout/error output |
| `python_interpreter(script)` | `bash(command)` | Same notebook contract |
| `jupyter_kernel_run(cell_content)` | `bash(command)` | Same notebook contract |
| `sandbox_exec(snippet)` | `bash(command)` | Same notebook contract |
| `evaluate_code(source_code)` | `bash` running `python3 -c` | Declared stdout semantics; only one Python operation in the trace |
| `execute_bash(command)` | `bash(command)` | One foreground program per call, no shell session mutations, polling or interactive input |
| `think(thought)` | Assistant reasoning | One control call in the turn; only an empty or recognized logging acknowledgement; no second reasoning segment |
| `finish(message)` | Final assistant content | Final operation, string message, no environment observation |
| `submit()` | Final assistant turn | Final operation, empty arguments, no environment observation; preceding answer text remains |
| `arc_python_executor(code)` | Residual | The source contract tests `transform(grid)` against task grids and reports metrics; a bare Python command cannot reproduce it |
| `commit_solution`, `explicit_solution_with_augmentations` | Residual | Require grounded ARC evaluator, task inputs and selection state |
| `web-search(query)` | Residual | Requires an actual configured search backend and a compatible response contract |
| Unrecognized tool sets | Error | Outside the pinned campaign contract; see the scope section |

The notebook mapping writes a visible Python client and starts one local IPython
kernel inside the first Bash call. Later Bash calls connect to that same kernel.
The original code string is passed through a quoted, collision-checked heredoc.
This preserves notebook state and final-expression display without replaying
previous cells. The execution environment must provide Python, `ipykernel` and
`jupyter_client`, plus the original task's packages and files. Setup is folded
into the first call; no fictitious setup response is inserted.

The target is the `command` argument shared by Pi Bash and OpenCode Bash.
The implementation does not translate timeout units between those harnesses.
The converter reads JSON function declarations from `<tools>` blocks in the
serialized system turns. Their `description` fields identify notebook versus
stdout behavior; no separate declaration configuration overrides the source. Control
mappings use the source's declared logging/completion behavior: substantive
`submit` results, failed submissions, and nonfinal submissions remain residual.
Removed logging acknowledgements stay in the original archived text and the
change ledger. Other result contents, user content and unrelated turns remain
unchanged. IDs present in serialized calls survive; when the source template
omitted IDs, exported messages use deterministic positional IDs.

## Run a pinned selection

The CLI bootstraps CoreWeave S3 credentials for TensorStore and requires a
configured CoreWeave account, including when the selection/tokenizer files are
local. This entry point targets regional CoreWeave caches; it is not a generic
GCS launcher.
Run from the repository's experiment environment. Its `marin-verifyit[schema]`
dependency supplies JSON Schema validation; this experiment adds no dependency
to the Marin library package.

Input shards are Levanter TreeStores with an `input_ids` field containing
complete serialized conversation rows. The input JSON contains
`tokenizer_sha256` and a `sources` list. Each entry has
`source`, `input_path`, `expected_rows`, and optionally `source_revision` and
`expected_tokens`. Source is the dataset catalog identifier; each list entry
is one shard, and `--shard-start`/`--shard-stop` select its zero-based,
half-open list range. Paths identify immutable input shards. For science data that has already undergone a separate repair pass, the
selection owner must supply the accepted repair cache paths. Converting the
older source would restore unrepaired text or excluded examples.

```bash
uv run python -m experiments.tool_conversion.run \
  --selection selection.json --tokenizer tokenizer.json \
  --output s3://BUCKET/NEW-PREFIX/r0 \
  --shard-start 0 --shard-stop 1 --batch-size 128
```

Use a separate prefix per worker and conversion version. A changed catalog,
configuration or converter hash cannot resume an existing prefix. Never use a prefix owned by another producer, or alter its progress cursor
or per-batch work claims. `--row-limit` bounds a pilot per
shard and is part of the pinned configuration. The tokenizer's native special
token IDs are checked before selection.

Each batch contains `dispositions.json`, original source payloads for accepted
rows, a candidate, a change ledger, an export and a verification result. The
export contains text, structured messages and tools, token IDs, an assistant-only
loss mask and member hashes. Root `provenance.json` records converter file hashes
and tokenizer/catalog identity. Per-batch receipts bind compressed archive bytes
to their SHA-256 and post-commit cursor. Receipt creation is the commit point;
resumption skips committed batches. Deterministic archive construction permits a
failed upload or receipt write to be reconciled without inference replay.

Consumers must require a verified disposition, a nonempty verification record,
valid member hashes and an actual original/candidate call difference. A producer
status alone does not establish that calls changed. The overlay replaces accepted rows in the consumer's input dataset. Join
`shard`, `row`, and `input_ids_sha256` from batch dispositions to the original
input, apply each replacement once, then rerun the consumer's quality and mask
checks. `mapping_unresolved` records preserve the original locator for residual
processing. Skipped sources and
`tool_free`/`real_tools_only` rows receive no replacement. This runner
never edits or deletes the input dataset. Corpus counts by tool overlap; exact document counts must come
from the row dispositions rather than summing per-tool counts.
