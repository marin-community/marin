# exp75 — throughput by slice × band

Reference for slice selection. Same recipe on every slice/band (global batch 128,
seq 8192 → ~1.05M tokens/step), so all numbers are directly comparable.

## Metrics — track all four, **decide on `wts`**

Per W&B run (`eric-czech/marin`, group `exp75-contacts-v1-tune`):

| metric | what | from W&B keys |
|---|---|---|
| **`wts`** | **wall-clock tok/s** — tokens / elapsed clock from train start, incl. **in-run preemption/dead time** | `throughput/total_tokens` ÷ (`_timestamp` − `createdAt`) |
| `ats` | active tok/s — tokens / time the run process was alive/connected | `throughput/total_tokens` ÷ `_runtime` |
| `tts` | training tok/s — compute throughput during steps | `throughput/tokens_per_second` |
| `mfu` | model FLOPs utilization (%) | `throughput/mfu` |

Ordering always `tts ≥ ats ≥ wts` (compute ⊂ active ⊂ wall). **`wts` is the
decision metric** — the real "how fast will this finish" rate, folding in slice
speed *and* preemption. Diagnostics: `tts` = upside *if always available*, `ats`
isolates in-run overhead (eval/loading); `wts ≪ tts` flags a fast-but-preempted
slice; **MFU is mostly irrelevant** but explains *why* v6e big slices underperform.

**Verified semantics (don't assume — these were checked):** `throughput/total_tokens`
is **cumulative** (`= step × 1.05M`; ratio 1.000 across runs). `_runtime` is
**active time and excludes dead/preempted time** — proven on a preempted run where a
gap advanced wall +34,929 s but `_runtime` only +179 s. `tts`/`mfu` are
**instantaneous** (flat across steps), so they're averaged, not summed. Each point is
a **single wandb object** (resumes its own id through preemption — 0 multi-object
points), so per-slice `wts = Σtokens / Σwall` sums over *distinct* points (no
double-count). **Caveat:** `createdAt` is *training start* (post-scheduling), so `wts`
misses iris **queue/pending** time before a run starts — scarce-slice scheduling
latency (e.g. v5p-64) is handled separately by the ≥1 h relocate rule. Re-measure
periodically; never assume bigger = faster.

### How to compute it (do it exactly this way every time)

Every input is a **single W&B summary scalar per run** — no history scan, no
per-step work. Per run, then token-weighted per slice:

```python
import wandb
from datetime import datetime, timezone

def to_unix(iso):                      # run.created_at is an ISO string, not a summary key
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).replace(tzinfo=timezone.utc).timestamp()

api = wandb.Api()
runs = api.runs("eric-czech/marin", filters={"group": "exp75-contacts-v1-tune"})
agg = {}                               # key = (tpu, band)
for r in runs:
    if not r.name.endswith("-v1"):     # current campaign only (skip v0/v0.1 smoke)
        continue
    s = r.summary
    tok  = s.get("throughput/total_tokens")        # cumulative tokens (= step * 1.05M)
    rt   = s.get("_runtime")                        # ACTIVE seconds (excludes dead time)
    wall = s.get("_timestamp") - to_unix(r.created_at)   # elapsed incl. preemption
    if not tok or not rt or wall <= 0:             # skip not-yet-started runs
        continue
    tts = s.get("throughput/tokens_per_second")     # instantaneous rate
    mfu = s.get("throughput/mfu")                    # instantaneous %
    tags = dict(x.split("=", 1) for x in (r.tags or []) if "=" in x)
    a = agg.setdefault((tags.get("tpu"), tags.get("band")),
                       dict(tok=0, wall=0, rt=0, tts=[], mfu=[]))
    a["tok"] += tok; a["wall"] += wall; a["rt"] += rt          # SUM cumulative scalars
    a["tts"].append(tts); a["mfu"].append(mfu)                 # AVERAGE rates (never sum)

# per slice:
#   wts  = a.tok / a.wall          <- DECISION METRIC, rank slices by this (desc)
#   ats  = a.tok / a.rt
#   tts  = mean(a.tts)             # mfu = mean(a.mfu)
#   dead% = (a.wall - a.rt) / a.wall
```

Summing `tok`/`wall`/`rt` across a slice's runs is safe **because each point is one
wandb object** (0 multi-object points). If that ever changes (a point splits into
separate `crashed`+resumed objects), dedup per point first (cumulative `tok` would
double-count). `tts`/`mfu` are rates → **average, never sum**.

All tables backfilled & verified ~18:00Z, **ranked by `wts`** (token-weighted over
all v1 runs on each slice). `dead% = (wall − _runtime)/wall` = fraction of elapsed
clock the run was preempted/dead.

## Interactive (single-host, counts toward budget)

| slice | chips | wts | ats | tts | mfu | dead% | n | notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `v6e-8` | 8 | **~71k** | ~71k | ~73k | 14% | 0% | 4 | reliable (one budget-bounced cell ran 65% dead → wts 25k) |
| `v5p-8` | 4 | **~41k** | ~50k | ~54k | 40% | 18% | 4 | some preemption |
| `v6e-4` | 4 | **~40k** | ~40k | ~41k | 15% | 0% | 3 | reliable, slowest |

## Batch (multi-host, off-budget)

| slice | chips | hosts | wts | ats | tts | mfu | dead% | n | notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `v5p-64` | 32 | 8 | **~192k** | ~294k | ~333k | 31% | 34% | 1 | top wts but scarce/preempted (8-host gang) |
| `v5p-32` | 16 | 4 | **~154k** | ~171k | ~198k | 37% | 10% | 4 | **best risk-adjusted — high wts + reliable** |
| `v6e-16` | 16 | 4 | **~111k** | ~124k | ~130k | 12% | 11% | 4 | solid, reliable |
| `v5p-16` | 8 | 2 | **~95k** | ~97k | ~102k | 38% | 2% | 3 | most reliable; modest |
| `v6e-32` | 32 | 8 | **~74k** | ~195k | ~213k | 10% | **62%** | 3 | **AVOID — 62% dead; wts ≈ interactive despite 213k tts** |

**The wts re-ranking flips the tts story.** `v6e-32` is #2 by tts (213k) but its
8-host gang is preempted **62% of the wall clock**, so its real `wts` (~74k) is the
*worst* batch slice — no better than plain interactive `v6e-8`. **Stop sending work
to v6e-32.** Favored order by wts: **`v5p-64` > `v5p-32` > `v6e-16` > `v5p-16` ≫
`v6e-32`**; `v5p-32` is the best risk-adjusted pick (high wts, only 10% dead). v5p
MFU ~31–40% vs v6e ~10–15% explains why the v6e gangs underperform their chip count.
