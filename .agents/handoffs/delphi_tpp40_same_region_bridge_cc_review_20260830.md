# Delphi TPP40 same-region bridge fallback review

## Objective

Unblock the production Europe TPP40 launch using the already frozen one-pair
bridge (`run_order=2`) without waiting for unavailable East5 `v6e-8`
evaluation capacity. The scientific comparison remains the canonical East5
`v5p-8` trajectory versus the new Europe `v6e-8` trajectory. Only evaluation
placement changes: both checkpoints are evaluated on Europe `v6e-8` using the
same region-local Uncheatable and Table-9 payloads.

The existing numerical contract remains unchanged:

- phase-boundary Uncheatable absolute paired delta <= 0.002 BPB;
- endpoint Uncheatable absolute paired delta <= 0.002 BPB;
- endpoint Table-9 macro absolute paired delta <= 0.002 BPB;
- all required metrics and idempotent reruns must complete.

The frozen contract is:

`experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_acceptance_contract_v3.json`

## Proposed fallback

1. Mirror exactly three immutable canonical East5 artifacts into Europe using
   `gcloud storage rsync --recursive --checksums-only` (not Storage Transfer
   Service): Orbax steps 21855 and 27335, plus endpoint HF step 27335. Total is
   about 10.05 GB.
2. Require a path-independent inventory equality check for every mirrored tree:
   relative object name, byte size, and CRC32C must match exactly. Persist both
   source and destination identities and the aggregate manifest hash.
3. Create a dedicated Europe-side reference evaluator. It reconstructs the
   frozen `run_order=2` run spec and model identity, reads only the audited
   Europe mirror, uses the exact Europe validation caches/request set already
   accepted as payload-equivalent, and emits outputs under a new Europe prefix.
4. Build a new path manifest before reading results. Logical side names remain
   `east5` and `europe`; the East logical side records canonical source paths,
   mirrored checkpoint paths, mirror identities, and evaluator placement.
5. Compare the two Europe-evaluated results under the unchanged v3 thresholds.
   This removes evaluator-region/hardware variation and therefore isolates
   training-deployment drift more cleanly than the original cross-region eval.
6. Prove idempotence by rerunning the unchanged Europe reference evaluator and
   Europe candidate evaluator commands after completion, requiring zero child
   jobs and unchanged result inventories. Training idempotence remains checked
   against the existing frozen East reference command and Europe training
   command.
7. Keep the current East5 eval parents alive as supplementary evidence, but do
   not let their capacity queue block the same-region bridge or production.

## Production sequencing

After the same-region one-pair gate passes:

1. Re-materialize the already reviewed deterministic 280-row assignment and
   require byte-for-byte identity with candidate `assignment_v1.json`.
2. Re-run the exact resolved-path audit and require the already frozen payload
   SHA `0b9d659602f9385ba9172773762e0d2fb15900ec74ace0887ab31ea7dcbafbc7`.
3. Re-run strict launch safety and dry runs.
4. Submit the reviewed disjoint East5 plus Europe production parents.

## Review questions

1. Is same-region evaluation scientifically valid, and is it stronger than
   waiting for separate East/Europe evaluators for this accelerator bridge?
2. Does the mirror integrity contract sufficiently prove that the East model
   being evaluated is exactly the canonical East `run_order=2` model?
3. What must be frozen before the Europe bridge result exists to prevent
   outcome-contingent gate changes?
4. Is any part of the proposed idempotence evidence weaker than the reviewed v3
   gate?
5. Return a clear GO/NO-GO verdict and list only launch-blocking corrections
   separately from nonblocking polish.
