# Large solver protocol activation

This is the protocol-binding API reference. The live campaign's catalog,
FACET admission and ownership checks are recorded in
[difficulty #93](https://github.com/marin-community/marin-logbooks/issues/93).

The default Large protocol retains its frozen 16,384-token output cap. V2 keeps
thinking enabled and the same model, revision and 65,536-token context. Each
request gets `max_tokens = 65,536 - actual_prompt_tokens - 64`; requests leaving
less than 16,384 output tokens fail closed as typed context-budget outcomes.
Select `--large-protocol-id large-context-minus-prompt-20261127-v2` for `status`,
`plan` or `reconcile`. The launcher carries that selection into the regional
controller and its attempt actions. Other solvers retain their original protocol.

A v2 controller requires an immutable [ProtocolActivation](protocols.py) under
`v1/protocol-activations/large/` in its ledger. Its `qualification` and
`native_cleanup` fields are URI/SHA256 references. The qualification artifact is
the completed strict 16-case canary record: every response parses, every
length-censored response consumes its full dynamic allowance, emitted answers
validate against their schemas, and the engines survive. Empty normal stops are
permitted model zeros; HTTP errors fail qualification. The record must report
all sixteen responses qualified, `passed=true` and `native_server_closed=true`.
The cleanup artifact lists native job IDs, job states, failure counts and task
states: the parent must be SUCCEEDED, and the engine jobs and their tasks must
be SUCCEEDED or KILLED with zero failures. Models, layout and budget constants
are checked against the frozen Large identity.

Stage the record with `write_protocol_activation(ledger, record)` after those
checks pass. Replays retain its fixed timestamp and exact evidence. Once it
exists, a controller started with the legacy default fails closed. The first
reconcile also requires initial supersession links. Generate those with
`protocol_supersessions` over the current rows, selecting Large, the original
protocol and v2, with `activation_path(ledger)` as the evidence URI and
`record.activated_at` as the timestamp; append them through `ledger.append_rows`.
This links historical scored and unscored v1 attempts with task identities,
without rewriting their grades or costs. Reconciles append the same links for
later v1 finishes and preserve old in-flight holds until those attempts finish.

A measured pool receipt declares the same `solver_protocol_id` as its controller
and worker action. A missing field denotes the legacy protocol. Launch a v2 TPU
owner with `engine_pool --solver-protocol-id` set to the same v2 identifier. The
default owner retains the legacy receipt format. The controller withholds a
mismatched pool; workers reject a mismatch before issuing requests to it.

The R15 reader at `8883623a213d` cannot decode v2 row extensions. Inspect each
old job's entrypoint and frozen bundle before cutover. Quiesce the controller and
any outstanding full-ledger readers, including admission rechecks and importers;
check old sealers' frozen partition sets before allowing new rows. Frozen
`runtime`/`ledger_worker` attempt workers append v1 evidence without decoding v2
partitions. Preserve them after validating their immutable action, source/task
hashes and original resolved pool binding. Pending workers still need those
bindings checked before they begin. This offline compatibility result does not
clear the current live reader population.

Use `rigging.filesystem.distributed_lock.create_lock` and `lease_refresh` on
`<ledger>/v1/reconcile.lock` from a regional Iris job to fence new admissions
during handover. The maintained lease owns its timeout and refresh interval;
existing worker appends continue outside that lease. Coordinate the old root's
quiescence, reader checks and replacement with the lane operator.

For this campaign, stage activation and initial supersessions before reinstalling
`backend-repair-pending-FWeindel-validated-tasks-facet-v2-20261124.json` from the
legacy ledger's `v1/repair-verifications-parked/` into the current ledger's
`v1/repair-verifications/`. Preserve SHA256
`7df7007b5e7c44f360b648a8f7f9e93d0ec0508eedd5fc272a18f2c3085dcfc0` byte-for-byte.
The campaign's `facet-parked-verification-installed.json` identifies both ledgers
and the catalog-specific FACET gate. Make that the final cutover-owned ledger
write, then launch with the explicit v2 flag. These APIs do not perform the live
handover, validate the catalog/FACET gate, or stop legacy jobs themselves.
