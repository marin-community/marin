# StarCoder WSD80 LR-onset central2/v4 launch review

## Scope

Read-only launch review of the deployment-only port in
`experiments/domain_phase_mix/launch_starcoder_wsd80_lr_onset_dense_surfaces.py`.
The frozen 644-row scientific design remains unchanged. The new deployment uses
`v4-8` in `us-central2-b`, region-local caches under
`gs://marin-us-central2`, and separate checkpoint and W&B identities.

Review environment: Claude Code subscription, `claude-opus-5`, maximum effort,
read-only tools, with `ANTHROPIC_API_KEY` removed from the child environment.

## Initial blocker

The first review blocked the proposed command because reducing coordinator RAM
from 32 GiB to 12 GiB made the non-preemptible CPU workers eligible but did not
exclude higher-priority preemptible TPU workers. It also recommended enforcing,
rather than merely recording, the central2 object fingerprint.

## Repairs

- Every canary and full parent command uses `--no-preemptible`, `--cpu 2`,
  `--memory 12GB`, `--disk 20GB`, `--region us-central2`, and
  `--zone us-central2-b`.
- Iris propagates only region and zone constraints to children. The parent-only
  non-preemptible constraint therefore does not prevent the `v4-8` training
  children from using preemptible capacity.
- The central2 validator now recomputes a canonical final-object manifest over
  relative path, byte size, CRC32C, and MD5. It requires exactly 1,618 objects,
  344,101,194,077 bytes, and SHA256
  `fa8d0341b94ce701e8cf115ae695e522a8cb13d99a1ec5d42d49080a514a870c`.
  Only `___temp/**`, `shard_ledger.json`, and `shard_ledger.json.bak` are
  excluded. The 1,618 retained rows are identical between central1 and
  central2.

## Final review

The final review verified that:

- `--no-preemptible` is a hard routing constraint and excludes the
  `v4-preemptible` pool for the parent.
- The 2 CPU / 12 GiB / 20 GiB parent fits the non-preemptible central2 CPU
  group, which is preferred over the non-preemptible reserved-v4 fallback.
- Child jobs independently reconstruct their `v4-8` resource constraints and
  remain eligible for preemptible training capacity.
- Source identity, ledger completion, consolidated layout, document count,
  shard-row identities, tokenizer metadata, finite-support bounds, and the new
  object fingerprint all fail closed before `run`.
- The canary and full-stage launches share artifact identity, so a successful
  canary is reused, while failed rows rerun safely.
- Central2 checkpoint paths, W&B IDs, group, version, and GCS roots are disjoint
  from the historical central1 deployment.
- The stale central1 release is intentionally provenance-only and fails closed
  under the new deployment-aware loader.

No scientific, locality, idempotency, or reproducibility blocker remains.

VERDICT: PASS
