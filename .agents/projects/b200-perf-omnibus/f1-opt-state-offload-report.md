# F1: MuonH optimizer-state host offload

## Result

F1 is implemented as an opt-in `SCALE_OFFLOAD_OPT_STATE=1` scale-launcher flag.
The training state keeps named-sharded optimizer arrays in JAX `pinned_host`
memory between steps. The compiled train step moves them to `device` after the
backward pass, performs the optimizer update, then returns the updated state to
`pinned_host`. Parameter arrays remain device-resident.

`GrugTrainerConfig.offload_opt_state` defaults to `False`. The launcher
documentation records the measured model-size split: d6144 EP64 used host
offload on Grace-Blackwell, while d5120 required a 135 GiB pinned-host arena and
measured 19.694% MFU. The default H100/PCIe path is also documented as a
configuration where the flag should remain unset.

The recorded 253.0K tok/s at 22.7% MFU to 255.3K tok/s at 23.1% MFU change is
not an isolated measurement of F1. That arm also enabled QB routing, XSA,
attention gating, and GatedNorm. This implementation does not attribute the
bundled +0.4 percentage-point result to host offload alone.

## Extraction

The source was `cff962d730`. I extracted only its optimizer-state memory-kind
transfers and adapted the flag to the current `GrugTrainerConfig` and scale
launcher. The source helper used `jax.typeof(leaf).sharding`; retaining that
detail is necessary because traced optimizer leaves do not expose concrete
sharding through `leaf.sharding`.

I dropped all `SCALE_ATTN_GATE`, `SCALE_XSA`, `SCALE_MOE_QB`, and
`SCALE_NO_HYPERBALL` hunks, along with the source smoke-script edit. I also
dropped the source commit's changes to the model return signature and
`GrugTrainState.pending_qb_betas`; main already has the current QB state and
metrics path.

The functional diff is +40/-3:

- `experiments/grug/moe/train.py`: +35/-3
- `experiments/grug/moe/launch_cw_scale.py`: +5/-0

The plan estimated approximately +45/-6, so the functional diff is within the
requested factor-of-two range. The behavior regression adds 62 test lines. This
report is excluded from the functional count.

## E1 correction and related main history

The struck E1 premise is false on `origin/main@6ce4a7e68`.
`36104c763` introduced the hardcoded Grug MoE baseline with GatedNorm, XSA,
attention gating, and QB routing. `90f3c2f8f` moved
`pending_qb_betas` into `GrugTrainState` and applies it inside the compiled step.
Both are ancestors of main. The XSA implementation was subsequently fixed for
GQA by `6dcb07e2b` and for GQA sharding by `060b043d0`, also ancestors of main.
The current `GrugModelConfig` docstring states that GatedNorm, XSA, and QB
routing are hardcoded, and there is no `qb_routing=False` launcher option.

The research lineage is different. `cff962d730` has merge base `696eb370d` with
main and carries `qb_routing: bool = False`; its scale launcher sets that field
only when `SCALE_MOE_QB=1`. Therefore a descendant run on that lineage with the
environment variable unset takes the QB-off path. I verified the code semantics
and branch ancestry. I did not audit every recorded submit environment, so this
does not establish the QB state of any specific measurement whose complete
environment is unavailable.

The other `cff962d730` features that the series discusses have also diverged:

- attention gating, XSA, GatedNorm, and QB routing already landed on main by the
  independent commits above and need no extraction from `cff962d730`;
- `SCALE_NO_HYPERBALL` is absent from main, but no numbered item in the series
  depends on it;
- `SCALE_OFFLOAD_OPT_STATE` was absent from main before this F1 change.

The sequence statement that F1 and E1 should be extracted together is stale.
The source hunks are interleaved, but main's independent QB implementation lets
F1 use the existing training state without importing any E1 code.

## Verification

The focused offload regression executes optimizer initialization on CPU with an
explicit Grug mesh, compares every initialized optimizer value against the
device-resident control, and checks that all non-scalar optimizer arrays use
`pinned_host`. It then lowers a complete offloaded train step and checks the
structured output sharding: all non-scalar optimizer arrays return in
`pinned_host` memory.

Commands and results:

- `uv run --package marin-levanter --group test --with pytest-timeout pytest
  tests/test_grug_variant_contracts.py -k 'optimizer_state_offload or
  grug_variant_one_step_contract' -q`: 3 passed.
- `./infra/pre-commit.py --all-files --fix`: clean.
- `uv run pyrefly check`: 0 errors.
- `uv run pytest`: 1,253 passed, 17 skipped, 5 xfailed, 47 deselected, and 1
  failure.

The default-suite failure is in the untouched
`test_grug_base_run_emits_expected_metrics_with_json_tracker`. Running it alone
reproduces the same `ShardingTypeError` at
`experiments/grug/base/model.py:227`: the explicit CPU path concatenates
`P(("replica_dcn", "data"), None)` and `P(None, None)` token slices. F1 changes
only `experiments/grug/moe`, and the focused MoE contract tests pass.

## Unverified

No GPU or cluster job was submitted. The full offloaded update was lowered but
not executed on Grace-Blackwell, so the device-to-host transfer timing, pinned
arena size, and HBM reduction were not remeasured. Attempting to execute the
offloaded update on the CPU backend aborted inside the JAX runtime; this is why
the checked-in CPU regression stops after lowering the full update. The flag is
off by default, and the intended runtime remains Grace-Blackwell.
