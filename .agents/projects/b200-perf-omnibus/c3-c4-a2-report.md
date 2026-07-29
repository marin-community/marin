# C3/C4/A2 extraction report

## Result

C3 shards MoE attention, shared-expert MLP, lm-head parameters, and their
optimizer state over the combined `("data", "expert")` FSDP axes. The shared
sharding module now names the dense and EP lm-head layouts separately as
`Plm_head_dense` and `Plm_head_ep`. Dense Grug uses the former; the Grug MoE,
June TPU MoE, and Snowball call sites use the latter.

C4 forces activations onto the canonical expert-axis batch spec before the EP
`shard_map` boundary. Its regression test inspects the local dispatch input and
checks that the buffer has 16 rows for 8 local tokens at both expert-axis widths
2 and 4.

A2 resolves `SCALE_CAPACITY_FACTOR` in the EP scale launcher, stores it in
`GrugModelConfig`, and uses one canonical default of 1.0 in both the experiment
and low-level MoE paths.

## Diff size

- C3 is +80/-17 including a +53/-0 test. Excluding that test, the functional
  diff is +27/-17, or 44 changed lines against the plan's approximately 31
  lines (+23/-8). The extra churn names both lm-head layouts and updates every
  existing consumer instead of leaving an overloaded `Plm_head`.
- C4 is +113/-0 including +102/-0 tests. Its functional diff is +11/-0,
  exactly the sequence estimate.
- A2 is +61/-8 including +47/-1 tests. Its functional diff is +14/-7 against
  the approximately +8 estimate; the additional lines centralize the default
  and validate that the launcher resolves the environment before model
  construction.

## Verification

`./infra/pre-commit.py --all-files --fix` passes. The configured Pyrefly command,
`uv run pyrefly check --baseline .pyrefly-baseline.json`, reports 0 errors
(415 baseline suppressions). The exact documented command `uv run pyrefly`
exits with usage status 2 under Pyrefly 1.0.0 because that version requires a
subcommand; the repository's pre-commit entry point also invokes `pyrefly
check`.

Focused C3/C4 validation with eight CPU devices passes 7 tests. This includes
the non-expert weight `PartitionSpec` assertions, the dense-Grug lowering
contract on its expert-less abstract mesh, the EP dispatch-boundary activation
spec, and the dispatch-buffer width-invariance assertion. The same focused
selection plus Snowball parity passes 11 tests and skips the multi-device
boundary test on the default one-device CPU host.

The default `uv run pytest` selection completed with 1,255 passes, 17 skips, 47
deselections, 5 expected failures, and one failure. The failure is
`test_grug_base_run_emits_expected_metrics_with_json_tracker`: JAX 0.11 rejects
the concatenate in dense Grug's shifted-label construction because
`token_ids[:, 1:]` retains `P(("replica_dcn", "data"), None)` while
`token_ids[:, :1] * 0` becomes replicated. It reproduces in isolation. The
failing line is unchanged from `origin/main`, and C3 preserves the dense
lm-head spec byte-for-byte as `P(Pbatch[0], "model")`, so this is not caused by
the C3 layout split. I did not fold an unrelated shifted-label fix into this
series.

No GPU or cluster job was run. The `PartitionSpec` and JAXPR assertions validate
the requested static sharding contracts, but this work does not include a GB200
compile, HLO profile, memory measurement, or throughput measurement.

## Dropped scope

I extracted the C3, C4, and A2 behavior instead of cherry-picking
`54bbe3d23`. I left out `moe_latent_dim`, latent dispatch, all-to-all remat
markers, and the other launcher changes from that commit. Latent MoE is
explicitly negative in evidence card E11, and the sequence assigns the remat
work separately. I also did not add a compatibility alias, mesh-axis runtime
branch, vendored file, smoke script, or cluster validation.

## Plan corrections and uncertainties

The sequence's C3 claim that `P(Pfsdp, "model")` is a strict generalization of
`P("data", "model")` is incomplete. The two are equivalent at EP1 only when the
mesh retains an `"expert"` axis of size 1. Dense Grug's lowering mesh has no
expert axis, so the EP spec is invalid there. The implementation uses separate
static names for the dense and EP layouts rather than deriving either layout
from the runtime mesh.

An ignore-respecting symbol search initially missed Snowball's multiline import
under `lib/`. Pyrefly exposed that missed consumer. Snowball is an MoE model and
now uses `Plm_head_ep`; an ignore-independent follow-up search found no
remaining `Plm_head` references.
