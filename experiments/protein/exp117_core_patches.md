# Core levanter patches on this branch

Fixes to `lib/levanter` carried on this branch, needed to train/checkpoint this workload on Marin
TPUs. Each is a genuine upstream bug or gap, not experiment glue. Ordered by blast radius.

## Multi-host HF-checkpoint save — `compat/hf_checkpoints.py` (`6aa8e46cb`)

**Symptom.** Any slice spanning >1 VM host (16-chip `v5litepod-16`/`v6e-16`; multi-host `v5p-16`/`v5p-32`)
crashed at the periodic HF checkpoint save:
`RuntimeError: Fetching value for jax.Array that spans non-addressable (non process local) devices`.
Single-host slices were unaffected (their arrays are already host-local), so it only surfaced at scale.

**Cause.** `save_pretrained` deshards each weight with `reshard(w, PartitionSpec())` then `np.asarray(w)`.
Under the trainer's default **`AxisType.Auto`** mesh the deshard does not make the array
host-addressable, so `np.asarray` fails whenever the array spans more than one process.

**Rejected fix — `use_explicit_mesh_axes=True`.** It makes the `reshard` replicate correctly (verified),
but breaks *training* with a `ShardingTypeError` in RMSNorm: the model/haliax code is not
sharding-type-clean under explicit axes. Not viable without deep haliax work, and it would change the
training mesh — undesirable when the whole point is to measure memory under the real mesh.

**Fix.** Replace `np.asarray(v)` with `jax.experimental.multihost_utils.process_allgather(v, tiled=True)`
— a collective that gathers each weight to host on **every** process (only process 0 uploads;
`temp_dir_before_upload` already guards that). Save path only; the training mesh stays Auto. Verified
with a 2-process CPU repro and an end-to-end multi-host run that finished with checkpoints, HF
checkpoints, and evals.

## Skip eager HF reference-tokenizer fetch — `main/train_lm.py` (`69bea0a68`)

**Symptom.** Workers with unreliable HF egress hung indefinitely on a blocked socket at startup.

**Cause.** For `HFCompatConfig` models, `main()` builds `hf_checkpoint_converter()`, whose constructor
eagerly loads the model's reference tokenizer from HF Hub — then discards it one line later via
`.replaced(tokenizer=...)`.

**Fix.** Seed a throwaway model-config copy with the already-loaded training tokenizer so
`_infer_tokenizer` returns it unchanged and never touches HF. Guarded to configs exposing a tokenizer
field (Llama/Qwen); others keep the eager path.

## Sharded cache reader field enumeration — `store/cache.py` (`f370e52cd`)

**Symptom.** Reads of a sharded `TreeCache` failed with
`Sharded cache ledger missing input_ids/0 count for shard ...`.

**Cause.** The sharded reader rebuilt its field tree from the raw exemplar without `heuristic_is_leaf`,
so a Python-list exemplar (the text tokenizer's output) was descended into `input_ids/0`, `input_ids/1`,
… — keys the ledger (which records only `input_ids`) doesn't have.

**Fix.** Pass `is_leaf=heuristic_is_leaf` so the sharded reader enumerates fields the same way the
writer and the materialized reader do.

## `as_hf_tokenizer` honors `@revision` suffix — `tokenizers.py` (`5af673533`)

**Symptom.** HF checkpoint export crashed on the worker with an HF repo-id validation error (`@` not
in the allowed character set).

**Cause.** The branch pins HF tokenizers as `repo@revision` (so the local cache key forks on revision).
`HfMarinTokenizer.as_hf_tokenizer` / `KitokenMarinTokenizer.as_hf_tokenizer` passed the raw
`repo@revision` string straight to `AutoTokenizer.from_pretrained`.

**Fix.** Parse the suffix with the existing `_parse_repo_revision` helper and pass it as the `revision`
kwarg.

## Revision-pinned tokenizer caching — `tokenizers.py`, `data/text/datasets.py` (`65d6ad542`)

**Symptom.** A tokenizer repo re-pushed with a larger vocab (2840 → 2849 tokens) poisoned the shared
local tokenizer cache, so existing distance-masked checkpoints could no longer be loaded.

**Fix.** `load_tokenizer("repo@revision")` snapshots a specific HF revision and includes it in the local
cache key, so different revisions of the same repo don't collide. Callers pin the legacy revision and
pin the tokenize-step output paths so the new key doesn't trigger a from-scratch retokenize.
