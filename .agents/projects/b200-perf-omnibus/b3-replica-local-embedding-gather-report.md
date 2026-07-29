# B3 replica-local embedding gather report

## Built

B3 fully replicates the Grug MoE token-embedding table and performs the lookup
inside a batch-sharded `shard_map`. Each replica indexes its local tokens without
assembling the hidden dimension across the global mesh. The model uses this lookup
in place of `Array.at[token_ids].get(out_sharding=...)`.

The regression test compiles the lookup on a four-device CPU mesh whose
`replica_dcn` axis has size four. It compares the result with NumPy and rejects
`all-to-all`, `all-gather`, `all-reduce`, `collective-permute`, and
`reduce-scatter` in the optimized HLO. The test runs in the repository-default
pytest selection.

## Diff size

The functional diff against `origin/main` is +31/−3:

- `experiments/grug/moe/model.py`: +25/−2
- `lib/levanter/src/levanter/grug/sharding.py`: +6/−1

The estimate was +32/−6. The implementation is within the estimate; the three
fewer deletions come from retaining `main`'s loop implementation instead of
applying an unrelated formatting hunk from the source commit. The regression
guard adds +47 test lines. This report is excluded from those counts.

## Verification

- `uv run pytest
  tests/test_grug_variant_contracts.py::test_grug_moe_embedding_lookup_hlo_has_no_collectives`
  passed on CPU. The compiled module reported `num_partitions=4`, the result
  matched NumPy exactly, and the optimized HLO contained none of the five
  collective opcode families checked by the test.
- A separate control compiled the old sharded-table lookup on the same four-CPU
  mesh. Its optimized HLO contained both `all-gather` and `all-to-all`. Replacing
  only the new table sharding with the old `P("model", Pbatch[0])` layout also
  makes the new test fail during lowering because the table no longer satisfies
  the replica-local `shard_map` contract.
- `./infra/pre-commit.py --all-files --fix` passed, including the repository's
  pinned Pyrefly check.
- `uv run pyrefly check` passed with 0 errors, 415 baseline suppressions, and 505
  warnings not shown. The requested `uv run pyrefly` spelling is not valid with
  Pyrefly 1.0.0: it prints command help and exits 2 because `check` is now a
  required subcommand.
- `uv run pytest` selected 1,274 default CPU-safe tests. The result was 1,253
  passed, 17 skipped, 47 deselected, 5 xfailed, and 1 failed. The failure was
  `test_grug_base_run_emits_expected_metrics_with_json_tracker`, in untouched
  `experiments/grug/base/model.py` while concatenating differently sharded label
  slices before any embedding lookup. A fresh pytest process with
  `Pembed_vocab` restored to its pre-B3 value reproduced the same failure. B3's
  HLO regression test passed in the full run. The unrelated Grug-base label path
  was left unchanged.

No GPU or cluster job was run. The source commit records 30 successful steps at
512 GPUs, but this extraction only revalidated numerical behavior and collective
absence on the CPU HLO path.

## Dropped

The source commit includes a formatting-only change in a scanned-block branch
that no longer exists on `main`. Applying the commit without committing produced
a conflict there. The extraction retains `main`'s current loop and applies only
the embedding helper, its call site, and the embedding-table sharding change.

No architecture experiments, FA4 metadata changes, smoke scripts, or neighboring
omnibus items were included.

## Plan differences and uncertainty

`sequence.md` describes `bdf61d7ed` as the cleanest cherry-pick in the series,
but it does not cherry-pick cleanly onto the assigned `origin/main` at
`6ce4a7e68`; its unrelated scanned-block formatting hunk conflicts with code
removed from `main`. The intended B3 hunks apply without semantic adaptation.

The CPU HLO guard exercises a four-way `replica_dcn` mesh and detects the old
collective regression, but it does not reproduce an eight-rack NCCL rendezvous.
The assignment prohibits submitting a cluster job, so the 512-GPU production
result remains source-commit evidence rather than a result reproduced here.

The repository-default pytest selection is not fully green because of the
pre-existing Grug-base failure described above. Fixing that path would expand
this commit beyond B3.
