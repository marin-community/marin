# exp160 zoonomia v1/v2 — launch ops log

**Session date:** 2026-05-05
**Task:** Launch [bolinas-dna#160](https://github.com/Open-Athena/bolinas-dna/issues/160) (sanity-check zoonomia projection pipeline by training on `zoonomia-v1-v1` and `zoonomia-v1-v2`).
**Branch:** `gonzalo/dna-exp160-zoonomia-v1-v2`
**Final iris job:** `/gonzalo/exp160-prod-v0.6` (after 5 prior submission attempts)

This log captures the issues encountered while taking exp160 from "exp142 mirror" to a healthy launch, so the next person doesn't rediscover them.

---

## Issue 1 — `default_tokenize` requests cpu=4 but the only CPU pool on Marin is n2-highmem-2 (2 cpu)

**Symptom.** Tokenize coordinator children pend forever:
```
Scheduler: Insufficient CPU (need 4 cores, available 0.5 cores)
```

**Root cause.** `experiments/defaults.py:285` sets `default_tokenize` resources to `cpu=4, ram="16g", disk="10g"` — but `lib/iris/examples/marin.yaml` only defines one CPU `scale_groups` entry (`cpu_vm_e2_highmem_2_ondemand` → `n2-highmem-2`, 2 vCPU). A 4-core request cannot fit on a 2-core worker, and the autoscaler reports `Demand=0` (it has no larger pool to scale into).

**Why exp142 didn't hit this.** The exp142 tokenize children scheduled successfully on `2026-04-29` — the `tokenized-bolinas-v5-v3{1,2,3}-char-bos` outputs were either produced under a different cluster config or were cached from a much earlier run. New zoonomia datasets don't have any pre-existing cache.

**Fix.** In `experiments/dna/exp160_zoonomia_v1_v2.py`, override the tokenize coordinator footprint:
```python
_TOKENIZE_RESOURCES = ResourceConfig.with_cpu(cpu=1, ram="4g", disk="10g")

def _tokenize(name, dataset, dataset_format):
    return default_tokenize(..., resources=_TOKENIZE_RESOURCES)
```
The coordinator is pure orchestration — it submits zephyr workers that do the heavy lifting at their own resource spec. cpu=1/ram=4g is plenty.

**Iterations needed.** Tried cpu=2 first (commit `cbcbd3440`) → still didn't fit (parent already takes 0.5 of one worker, leaving 1.5; child needs 2). Dropped to cpu=1 (commit `fb26f0712`). Resources don't go into the executor's content hash, so the lower-resource cache is interchangeable with higher-resource caches.

**Followup worth considering.** Either (a) raise `default_tokenize`'s default to fit on n2-highmem-2 by default, (b) add a 4-cpu pool to `marin.yaml`, or (c) document the override pattern somewhere experiment authors will see it.

---

## Issue 2 — us-central1-a CPU pool at `max_slices=6` and saturated by other users' coordinators

**Symptom.** Even at cpu=1, children stayed pending in us-central1-a:
```
Scheduler: Insufficient CPU (need 1 cores, available 0.5 cores) - 6 worker(s)
Insufficient memory (need 4.0GB, available 0.8GB) - 1 worker(s)
Autoscaler: Unsatisfied autoscaler demand: no_capacity:
  cpu_vm_e2_highmem_2_ondemand-us-central1-a=at_max_slices
```

**Root cause.** `marin.yaml`'s `cpu_vm_e2_highmem_2_ondemand` has `max_slices: 6` per zone. All 6 us-central1-a workers were occupied by other users' small coordinators (`tonyhlee`, `kevin`, etc.) at the time. Autoscaler can't grow the pool past max.

**Fix.** Switch the iris `--zone` constraint from `us-central1-a` to `us-east5-a` (which has comparable `max_slices=6` but better TPU headroom too: `tpu_v5p_8` had 32 ready vs us-central1-a's 3). Tokenizations scheduled within ~1 minute.

**Trade-off.** [#142 op note](https://github.com/Open-Athena/bolinas-dna/issues/142#issuecomment-4347439263) explains that pinning a zone is needed so checkpoint resume after preempt doesn't break across regions. us-east5-a is also a single zone, so the protection still holds.

**Followup worth considering.** Document the "if us-central1-a is full, try us-east5-a" lever somewhere — or add a checker that picks the least-loaded zone at submit time.

---

## Issue 3 — zoonomia HF datasets use `sequence` field, not `seq` like genomes-v5

**Symptom.** Tokenization for `bolinas-dna/zoonomia-v1-v1` (and v2) failed in zephyr workers with:
```
File "/app/lib/levanter/src/levanter/data/text/_batch_tokenizer.py", line 246, in __call__
    texts = [example[self.text_field] for example in batch]
KeyError: 'seq'
```

Val tokenizations on the genomes-v5 datasets (which use `seq`) succeeded fine on the same submission, narrowing it to the new zoonomia datasets.

**Root cause.** `DNALmDatasetFormat`'s `text_key` defaults to `"seq"` — correct for the genomes-v5 datasets used by exp136/exp142. The new bolinas-dna#158 zoonomia projection pipeline uses field name **`sequence`** instead.

Confirmed by streaming one example from HF:
```python
ds = load_dataset('bolinas-dna/zoonomia-v1-v1', split='train', streaming=True)
list(next(iter(ds)).keys())
# ['query_name', 'species', 't_chrom', 't_start', 't_end',
#  't_strand', 't_src_size', 'sequence', 'augmentation']
```

**Fix.** Override `text_key` in `TRAIN_FORMAT` only (val sets stay on the default `seq`):
```python
TRAIN_FORMAT = DNALmDatasetFormat(text_key="sequence", lowercase_weight=0.01)
```
Commit: `05adc5791`.

**Followup worth considering.** Either (a) the zoonomia pipeline should rename the field to `seq` for consistency with other Bolinas DNA datasets, or (b) `DNALmDatasetFormat` should auto-detect a sane default, or (c) future experiments authoring against zoonomia-v1+ datasets need to know about this.

---

## Issue 4 — training fails immediately with `WANDB_API_KEY must be set in the environment`

**Symptom.** First training child (`checkpoints-dna-bolinas-zoonomia-v1-v2-v0.1-zoonomia_v2`) launched, did initial syncing/installing, then crashed:
```
File "/app/lib/marin/src/marin/training/training.py", line 370, in _check_for_wandb_key
    raise ValueError(...)
ValueError: WANDB_API_KEY must be set in the environment.
```

**Root cause.** `marin.training.training._check_for_wandb_key` requires the env var to be present in either `train_config.env` or `os.environ`. The iris worker container doesn't bake it in; we have to forward it at iris submit time. The exp142 launch presumably did this manually.

**Fix.** Pull the key out of `~/.netrc` (where wandb stores it by default) and forward via `-e`:
```bash
export WANDB_API_KEY=$(uv run python -c \
  "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")
uv run iris ... -e WANDB_API_KEY "$WANDB_API_KEY" -- python experiments/dna/exp160_zoonomia_v1_v2.py
```

**Followup worth considering.** Document the `-e WANDB_API_KEY` requirement in the DNA experiment launch playbook, or have the iris worker image source it from a secret manager.

---

## Issue 5 — `zoonomia_v1` tokenize coordinator OOMs at 4g RAM during shutdown

**Symptom.** After 60+/64 shards completed and "Coordinator shutdown complete" logged, the tokenize task died with:
```
Container was OOM killed by the kernel
```

**Root cause.** The coordinator's shutdown/aggregation phase pushes peak memory above 4g for the larger whole-genome dataset (`bolinas-dna/zoonomia-v1-v1`, 27.8 GB / 64 shards). The TSS-proximal subset (`zoonomia-v1-v2`) tokenized fine at 4g.

**Fix.** Bump `_TOKENIZE_RESOURCES` ram from 4g → 12g (commit `7cfa45f26`). 12g leaves headroom under the 16g VM cap and still packs alongside the parent's 1g.

**Recovery without disturbing v0.6.** Rather than kill the running `zoonomia_v2` training in v0.6 (which was healthy at step 580/10000), launched a side-car job restricted to v1:
```bash
SWEEP_DATASETS=zoonomia_v1 → exp160-v1only-v0.1
```
The two arms then proceed in parallel under different parent jobs.

---

## Sequence of submissions

| Job | Resources change | Zone | Outcome |
|---|---|---|---|
| `/gonzalo/exp160-prod-v0.1` | cpu=4 (default) | us-central1-a | All 8 tokenize children pending, killed at 21:48Z |
| `/gonzalo/exp160-prod-v0.2` | cpu=2 | us-central1-a | Still pending, killed |
| `/gonzalo/exp160-prod-v0.3` | cpu=1 | us-central1-a | Still pending (cluster saturated), killed |
| `/gonzalo/exp160-prod-v0.4` | cpu=1 | **us-east5-a** | Tokenizations scheduled and ran! `KeyError: 'seq'` on zoonomia datasets, killed |
| `/gonzalo/exp160-prod-v0.5` | + `text_key="sequence"` | us-east5-a | zoonomia_v2 tokenize succeeded; training failed with `WANDB_API_KEY` missing |
| `/gonzalo/exp160-prod-v0.6` | + `-e WANDB_API_KEY` at submit | us-east5-a | zoonomia_v2 training going well (step 580+, loss 1.27→1.26 at 00:19Z); zoonomia_v1 tokenize OOM'd at 4g RAM during shutdown — see Issue 5 |
| `/gonzalo/exp160-v1only-v0.1` | + `ram="12g"`, `SWEEP_DATASETS=zoonomia_v1` | us-east5-a | Submitted as a side-car to redo zoonomia_v1 only without disturbing v0.6's running v2 training |

---

## Commits on `gonzalo/dna-exp160-zoonomia-v1-v2` branch

```
05adc5791 [dna] exp160: zoonomia datasets use 'sequence' field, not 'seq'
fb26f0712 [dna] exp160: drop tokenize cpu to 1 to pack alongside small jobs
cbcbd3440 [dna] exp160: lower tokenize coordinator cpu to fit n2-highmem-2 pool
8383d6622 [dna] Add exp160 zoonomia v1/v2 sanity-check
```

The three follow-up commits (`cbcbd3440`, `fb26f0712`, `05adc5791`) are infrastructure workarounds discovered live; squash-on-merge or keep depending on PR convention.

---

## Reproducer for the v0.6 launch

```bash
git checkout gonzalo/dna-exp160-zoonomia-v1-v2

export WANDB_API_KEY=$(uv run python -c \
  "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")

uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --extra marin:tpu \
  --user gonzalo \
  --zone us-east5-a \
  --max-retries 2 \
  --job-name exp160-prod-v0.6 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python experiments/dna/exp160_zoonomia_v1_v2.py
```
