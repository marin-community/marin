# StarCoder interactive migration preflight

Read-only checks at 21:29–21:31 UTC on 12 September 2026 passed.

- The original `launch_starcoder_tpp10_refinement.py` reconstructs the archived plan exactly, preserving all 45 run names, fingerprints and output paths. Plan hash: `99ca724bc3bde33e80afb07373e47ddc0d7e44f22e15cd41da3daa3de2f4ef40`. All 27 training/runtime/tokenizer source pins and the original release are unchanged (`frozen_recipe_replay.json`).
- Forty artifacts pass exact fingerprint, permanent final checkpoint, frozen runtime, native final PALOMA metric, and matching finished W&B checks (`pre_migration_artifacts.json`).
- Only target p55/p60/p65/p80 and matched p80 (trainer 20260911, subset 20260914) remain incomplete. All five Iris leaves are PENDING with no worker (`pre_migration_tasks.csv`).
- The four saved target checkpoints are unchanged at steps 10759, 6958, 2283, 728. Their full listed object metadata, including generations, sizes, CRC32C and MD5, are recorded in `pre_migration_checkpoints.json`. The matched point has no checkpoint yet.
- All five pending output leases were still active and owned by the current CPU coordinator. After the root task cancels the old tree, it must confirm terminal descendants and no active leases before submitting. Do not delete lock files or force duplicate work.

The simplest new submission uses the original refinement launcher directly, without `resume_starcoder_tpp10_refinement_batch.py` or its `BatchTrainingClient` priority override. Use a new parent name with explicit INTERACTIVE priority, the same regional coordinator resources, the existing refinement release and plan, and `--max-concurrent 45`. The original launcher's `pending_training_steps` should select exactly five unfinished recipes and reuse all 40 successes; confirm this in the new parent receipt. No source or scientific configuration edit is required. Inspect actual new child priorities and region after dispatch.

Root owns cancellation, submission, post-cancellation lease verification and Fieldbook mutations. This review made no job changes.
