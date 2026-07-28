# FA4 Phase-B build blockers

## TL;DR

`origin/main` at `ac6b03aef` had a live CUTLASS 4.6.0 compile bug in the
Marin-owned FA4 segmented backward. CUTLASS 4.6.0 removed
`cute.make_fragment`; the B200 backward configuration traces five calls to that
missing symbol. This branch applies `5833e329e`'s exact +5/-5 migration to
`cute.make_rmem_tensor`.

Give that migration slot B2 in `sequence.md`, before the current QuACK backend
entry, and renumber the current B2-B4 entries to B3-B5. B1 must combine one
dependency extraction from `538381606` with the root Pyrefly configuration hunk
from `5cf76b64a`; neither source commit is safe to cherry-pick.

## Evidence baseline

The worktree was fast-forwarded without local changes from the plan's
`origin/main` snapshot `1c631c4c0` to `ac6b03aef`, the current `origin/main` on
2026-07-28. The inspected `sequence.md` is blob
`c17f497989daf72b466467028d7d0b2a39aad2ab` from project commit `5337b0444`.
Its Phase-B table is at `sequence.md:106-137`.

Before the code edit, `_fa4_cute_segmented_bwd.py` contained 28 occurrences of
the string `make_fragment` and no `make_rmem_tensor`. Twenty-three occurrences
are still-valid APIs such as `cute.make_fragment_like` and
`TiledMma.make_fragment_A/B/C`. The five broken calls were the top-level
`cute.make_fragment(...)` allocations now at
`lib/levanter/src/levanter/grug/attention/_fa4_cute_segmented_bwd.py:765-766`,
`:1139`, `:1191`, and `:1275`.

## Blocker 1: CUTLASS 4.6.0 breaks the segmented backward

### Verdict

This was a live bug on `main`, not a latent API inconsistency. Importing the
module does not evaluate the five calls because they are inside CuTe kernel
method bodies. CuTe tracing of the configured B200 backward does evaluate them
and fails because CUTLASS 4.6.0 has no top-level `cutlass.cute.make_fragment`.
No GPU run was needed to establish the missing Python API, and no GPU compile
was claimed.

The version and route are concrete:

- `lib/levanter/pyproject.toml:89-91` pins
  `nvidia-cutlass-dsl[cu13]==4.6.0` and allows `flash-attn-4` from b16 through
  `<4.1`. `uv.lock:7623-7631` resolves CUTLASS 4.6.0 with wheel SHA256
  `e3e0e4d8df20d82c8401fa013f4d82021f41daa5fca3d24b55d4a677f2308ca8`.
- In the published 4.5.2 wheel,
  `cutlass/cute/tensor.py:887-895` defines `make_fragment` as a deprecated alias
  for `make_rmem_tensor`, and `cutlass/cute/__init__.py:145-153,277-285`
  exports both names.
- In the published 4.6.0 wheel,
  `cutlass/cute/__init__.py:158-165,308-314` exports
  `make_rmem_tensor`, `make_rmem_tensor_like`, and the retained
  `make_fragment_like`; it does not import or export `make_fragment`.
- `lib/levanter/src/levanter/grug/attention/_fa4_cute_config.py:47-53`
  maps SM100/B200 backward to path arch 120.
  `_fa4_cute_kernels.py:994-1007` selects
  `SegmentedFlashAttentionBackwardSm120`, and
  `_fa4_cute_kernels.py:1042-1063` constructs it. That class inherits the
  affected kernel body from `SegmentedFlashAttentionBackwardSm80`
  (`_fa4_cute_segmented_bwd.py:75,1621`).

### Change and merge-plan slot

Commit `5833e329ea99f4976c2abe4957fef077aa59c691` changes only the five top-level
allocations. The working-tree patch has the same stable patch ID,
`999b35d42f776bf1fe7545d17b36f06e4eff6831`, as that commit. After the edit,
the file has five `cute.make_rmem_tensor(` calls and no
`cute.make_fragment(` calls.

Insert this row after B1:

| # | Commit | From | Size |
|--:|---|---|--:|
| B2 | Migrate the FA4 segmented backward to the CUTLASS 4.6 register-memory API | `5833e329e` | +5 / -5 |

Renumber the current QuACK backend, embedding gather, and FA4 segment-bound
entries from B2-B4 to B3-B5. B1 changes dependencies but does not exercise the
kernel; B2 must precede the first GPU validation of the shared substrate.

### FA4 wheel b16/b23 landmine

The CUTLASS migration does not require a `flash-attn-4` upgrade. It is correct
with the locked b16 wheel and is independent of the wheel's constructor keyword
rename.

The wheel issue is real and separate:

- `uv.lock:2145-2161` resolves `flash-attn-4==4.0.0b16`.
  In that wheel, `flash_bwd_sm90.py:48-75` and
  `flash_bwd_sm100.py:50-68` accept `subtile_factor`.
- In b23, those parameters are named `q_subtile_factor`
  (`flash_bwd_sm90.py:49-76`, `flash_bwd_sm100.py:52-71`).
  Main still passes `subtile_factor` from the THD SM90 and SM100 launchers at
  `lib/levanter/src/levanter/grug/attention/_fa4_thd.py:398-436` and from the
  native SM90 launcher at `_fa4_cute_kernels.py:1241-1267`. A b23 upgrade would
  fail while constructing each affected backward object.
- b23 metadata requires `nvidia-cutlass-dsl==4.6.0.dev0`, while main and
  `quack-kernels==0.6.1` require stable `==4.6.0`. The b23 wheel therefore
  needs a coordinated dependency and launcher migration; it cannot replace b16
  in B1's lock update.

Both Marin FA4 backends import code from the wheel:
`_fa4_thd.py:57-78` imports upstream forward, backward, preprocess, and
postprocess modules, while `_fa4_cute_kernels.py:972-980,1208-1215` imports
helpers and native SM90 classes. The five migrated allocations remain in the
Marin-owned segmented SM80/SM120 class, so the wheel imports do not alter the
CUTLASS 4.6.0 verdict.

## Blocker 2: corrected B1 definition

The attribution in `sequence.md:110` is incomplete:

- `git diff-tree 538381606` names only
  `lib/levanter/pyproject.toml`, `lib/marin/pyproject.toml`, and `uv.lock`.
  The commit has no root `pyproject.toml` diff and no Pyrefly hunk.
- In `538381606`, the Levanter GPU block changes the CUTLASS range and adds
  `quack-kernels[cu13]==0.6.1`; the Marin GPU block only changes the CUTLASS
  range. Main already has the stricter stable CUTLASS pin in both blocks at
  `lib/levanter/pyproject.toml:90` and `lib/marin/pyproject.toml:123`.
- `5cf76b64a` adds the five-line root `pyproject.toml` hunk: one comment plus
  `cutlass`, `cutlass.*`, `quack`, and `quack.*` in
  `tool.pyrefly.ignore-missing-imports`. Main's list at
  `pyproject.toml:322-365` lacks all four patterns.
- The published `quack-kernels==0.6.1` metadata requires stable
  `nvidia-cutlass-dsl==4.6.0`, including for its `cu13` extra. This matches
  main's exact pin.

B1 must consist of exactly these changes against its eventual current-main
base:

1. Add `quack-kernels[cu13]==0.6.1; sys_platform == 'linux'` to
   `lib/levanter/pyproject.toml`'s `gpu` extra, using the requirement from
   `538381606`.
2. Add `cutlass`, `cutlass.*`, `quack`, and `quack.*` to the root
   `tool.pyrefly.ignore-missing-imports` list, with the explanatory comment from
   `5cf76b64a`.
3. Regenerate the current base's `uv.lock` minimally so Levanter records the
   direct QuACK requirement and the resolved QuACK package moves from main's
   transitive 0.5.0 (`uv.lock:9611-9625`) to 0.6.1. Preserve
   `flash-attn-4==4.0.0b16`.

Do not take either CUTLASS range edit from `538381606`; do not change
`lib/marin/pyproject.toml`; do not copy that commit's stale whole-lock image;
and do not cherry-pick `5cf76b64a`. B1 is a two-commit extraction followed by a
fresh minimal lock update.

## Other Phase-B checks

The B2 source warning in the plan remains accurate. The validated FSDP and EP
tips have identical blobs for all three backend files:

- `sonic_cute.py`: 272 lines, blob `4d53627060`
- `quack_moe_cute.py`: 172 lines, blob `d8b6b520be`
- `quack_symmetric_cute.py`: 117 lines, blob `628f77fdb2`

The additional implementation trap is B1's lock regeneration. The declared
`flash-attn-4>=4.0.0b16,<4.1` range admits b23, but the current code and stable
CUTLASS/QuACK pins do not. Preserve the b16 resolution unless a separate b23
migration updates the dependency pins and all three `subtile_factor` call
sites.

## CPU-only verification

On untouched `origin/main@ac6b03aef`, the literal `uv run pyrefly` command
exited 2 because a fresh worktree environment did not contain a `pyrefly`
executable. Pyrefly 1.0 also requires the `check` subcommand. The equivalent
pinned invocation,
`uv run --with pyrefly==1.0.0 pyrefly check`, reported
`0 errors (416 suppressed, 534 warnings not shown)` before the edit and the
same result after the code and report changes. The literal command still exits
2 after the edit for the same missing-executable reason, so neither Pyrefly
signal regressed.

The exact requested attention command,
`/home/marin/projects/marin/.venv/bin/pytest lib/levanter/tests -k "fa4 or attention" -q`,
ran 85 tests successfully and skipped 20, then exited 1 on an unrelated
collection error because the shared CPU environment lacks `tensorboardX` for
`lib/levanter/tests/test_tensorboard.py`. Repeating the same selection with only
that file ignored passed: 85 passed, 20 skipped. Seven skips were the FA4/CuTe
GPU tests (five in `test_fa4_cute_attention.py`, two in
`grug/test_attention.py`); the other 13 required Trackio, multiple devices,
Transformer Engine, or TPU. No FA4 kernel compiled, and no GPU job was
submitted.

`./infra/pre-commit.py --fix --files` passed for the migrated kernel and this
report.
