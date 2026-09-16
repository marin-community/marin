PASS_AFTER_BLOCKERS_RESOLVED

# CC review: StarCoder WSD80 gradient plot completion v6

Claude Code was invoked twice through `claude -p` with `ANTHROPIC_API_KEY`
removed, the OAuth account `plambdafour@proton.me` on subscription billing,
`claude-opus-5` at maximum effort, and read-only `Read`, `Grep`, and `Glob`
access. The second pass re-read the corrective implementation after every
initial blocker was addressed.

## Initial blockers and closure

1. Runtime loading initially failed to rehash the superseded v5 release and its
   failure marker. It now verifies both, matching the v1-v4 lineage checks.
2. The frozen-lock mechanism depended on environment inheritance in the
   historical launcher. Direct inspection of commit `377ad16d` confirmed that
   its nested `subprocess.run` supplies no `env` override, so `UV_FROZEN=1` is
   inherited by `uv run iris`.
3. A byte-identical lock alone did not prove a successful dry run. The exact
   filtered-workspace reproduction exited zero after installing 210 packages
   and reached Iris CLI argument handling while preserving `uv.lock` at
   1,383,710 bytes and SHA-256 `f6a62fcbb29a82ecd51ce841ac05065357c9c0f2320554e7bdc8684495421e99`.
4. The submission requirement was prose-only. The release now records the
   required environment and command prefix, and runtime loading fails if that
   frozen contract drifts.
5. The three packaging exclusions lacked an exact inventory assertion. Freeze
   now requires all three exclusions to be consumed and exactly 1,006 retained
   historical source rows.

## Reviewer assessment

The three omitted files cannot affect numerical probe results:

- Iris stamps `_build_info.py` only for its client-revision freshness handshake.
- Iris excludes both dashboard HTML files by its normal bundle policy; the v5
  parent imported the probe stack and reached the source gate without them.
- `uv.lock` remains included and hash-checked byte-for-byte. Only the three
  deterministic packaging artifacts are excluded.

The failed v5 release is truthfully recorded, hash-pinned, reverified at runtime,
and has an empty result root. Row and group identities are unchanged, while v6
uses a fresh schema, artifact version, result root, authorization sidecar, and
runtime baseline. The reviewer independently checked that the staged
source-only comparison counts close exactly at 288 rows and 1,408 comparisons.

The residual risk is operational only: omitting `UV_FROZEN=1` would reproduce a
parent-side lock-hash failure before any probe child, not produce corrupted
numbers. The reviewer recommended freezing, authorizing, and submitting only the
eight-row Stage-1 canary under the exact frozen-lock command.

VERDICT: PASS_AFTER_BLOCKERS_RESOLVED
