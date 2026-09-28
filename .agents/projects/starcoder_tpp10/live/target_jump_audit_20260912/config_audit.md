# StarCoder TPP10 target configuration audit, 2026-09-12

The target BPB curve combines two metric definitions. Pilot p50/p70/p90 endpoints use the historical evaluator. Resumed refinement p55/p60/p65/p80 endpoints use BPB schema 2. The actual interactive workspace contains the changed evaluator, while the frozen recipe and runtime guard omit that file. Cross-run W&B configuration comparisons find only intended mixture-weight and run-identity differences. The evidence supports an evaluation discontinuity; it does not establish a training discontinuity.

## Metric and source evidence

All rows below finish at logged step 11,490. Loss is PALOMA token-average negative log likelihood in nats. The original and schema-2 evaluator use the same loss accumulator, so this loss remains comparable across these evaluator versions.

| Target StarCoder weight | Cohort / final evaluator | Token loss | Reported BPB |
|---:|---|---:|---:|
| 50% | Pilot / historical | 1.330683351 | 0.794240773 |
| 55% | Resumed refinement / schema 2 | 1.321192384 | 0.770134926 |
| 60% | Resumed refinement / schema 2 | 1.319997787 | 0.769438505 |
| 65% | Resumed refinement / schema 2 | 1.315052986 | 0.766556144 |
| 70% | Pilot / historical | 1.313351631 | 0.784388363 |
| 80% | Resumed refinement / schema 2 | 1.321655989 | 0.770405114 |
| 90% | Pilot / historical | 1.336890817 | 0.798773348 |

The 70% endpoint has lower token loss than the 65% endpoint, despite its higher displayed BPB. See [fresh_target_sources.json](fresh_target_sources.json) for the freshly read metric rows and schema markers.

The original evaluator computes batch BPB, then averages those values with scored-token weights. Schema 2 sums loss bits and scored bytes before dividing. For batch loss sum \(L_b\), scored tokens \(T_b\), and scored bytes \(B_b\), the two metrics are:

$$
\operatorname{BPB}_{old}=\log_2(e)\frac{\sum_b T_b L_b/B_b}{\sum_b T_b},\qquad
\operatorname{BPB}_{2}=\log_2(e)\frac{\sum_b L_b}{\sum_b B_b}.
$$

They can differ even with one evaluation dataset and unchanged batch boundaries. The loss-per-token numerator and weighting are unchanged. The schema-2 change also fixes hierarchical BPB aggregation and adds `eval/bpb_schema_version=2`. It does not alter evaluation model execution, example selection, shuffle, token loss, or training optimization. See [the exact evaluator diff](eval_pilot_to_interactive.diff), historical evaluator lines 553–590, and interactive evaluator lines 558–594 and 613–624.

## Bundle lineage

| Workspace | Evidence | Evaluator SHA-256 / schema |
|---|---|---|
| Pilot | Local retained ZIP SHA-256 `4567a8171518c174da02918c41161850cd63bb465dc8510ea80a6138483395b5`, matching submitted-bundle receipt | `b67cc96335f667982a0fe43618215734140c28392be215a88177979c712dee2c` / historical |
| Original refinement | Local retained ZIP SHA-256 `8100461f4555644152f7a576754a5d25095ebe40b827dd6ffea4354024881fff`, matching submitted-bundle receipt | Same historical evaluator |
| Batch recovery preflight | Local ZIP SHA-256 `3655dd72d95b64c91b0b39341b27308cead8f5a24402128694032f1390067246` | Same historical evaluator |
| Interactive migration | Actual GCS bundle `b66dcc0d2406e598a2ad0335bda2934ccf80da4c86e7f158e1bfdbee3990bcbe`, generation `1789248886961835`; preflight and child job receipts name this bundle | `0d7b85d886d748f97fa41e4e7e2afcf4ccd2d07bceba01ada5b8c9c306e4e64e` / schema 2 |

The complete original pilot/refinement local ZIP comparison covers 5,275/5,274 entries. Only three common files differ: an unrelated comparator plot script, Iris cluster configuration, and priority documentation. All training/evaluation library source, tokenizer assets, and lock files are byte-identical across the original cohorts. See [config_bundle_comparison.json](config_bundle_comparison.json).

For the interactive bundle, 579,422 bytes were read through four explicit ranges: ZIP tail, central directory, evaluator member header, and the compressed evaluator source. The evaluator member was decompressed, length/CRC checked, and SHA-256 hashed. Its bytes match the current schema-2 evaluator. The complete central-directory inventory covers 4,536 entries. CRC32 plus uncompressed-size comparison finds `lib/levanter/src/levanter/eval.py` as the only changed source file under Levanter, Marin, Haliax, or Fray. Removed library members are tests; no new library source appears. This inventory comparison is weaker than full SHA-256 verification of every common member, but the frozen 27-file guard independently checks hashes for model, optimizer, trainer, data, launch code, lock, and tokenizer assets. See [config_interactive_bundle_comparison.json](config_interactive_bundle_comparison.json) and the [actual interactive evaluator](interactive_source/lib/levanter/src/levanter/eval.py).

## Why the existing checks passed

`launch_starcoder_tpp10.py:44–68` lists the source pins. It includes `main/train_lm.py`, `trainer.py`, `optim/muonh.py`, data and model files, tokenizer assets, and `uv.lock`; it does not include `eval.py`. `verified_training` at lines 82–98 checks that partial file map and four package versions, then writes them to `verified_runtime.json`. Thus an evaluator change can pass the recipe fingerprint and runtime receipt checks.

`collect_results` at lines 229–246 checks the same partial runtime receipt and selects the frozen metric name at the final step. It does not inspect `eval/bpb_schema_version`. The migration receipt's `all_scientific_identities_unchanged=true` therefore establishes unchanged serialized recipe identities but misses this metric-definition drift. The original and interactive evaluators intentionally log under the same BPB key; the schema marker is the distinguishing evidence.

## Target configuration comparison

The fresh W&B `model`, `optimizer`, `trainer`, and `data` subtrees for p50, p55, p60, p65, p70, p80, and p90 were compared recursively against p70. Differences are exactly the seven intended mixture weights, run ID/name, and run-specific checkpoint/tracker paths. No other serialized differences occur. See [config_wandb_target_diffs.json](config_wandb_target_diffs.json).

| Property | Shared target setting |
|---|---|
| Model | Qwen3; 1,024 width, 16 layers, 4,096 intermediate width, 8 attention and KV heads; tied embeddings; QK normalization disabled |
| Parameters | Frozen actual-leaf count 301,241,344 |
| Training budget | 11,491 optimizer steps × 128 sequences × 2,048 tokens = 3,012,296,704 tokens; TPP 9.999612483 |
| Optimizer | MuonH LR 0.02; Adam LR 0.008; momentum 0.95; Nesterov true; beta1 0.9; beta2 0.98; weight decay 0.1; grad clip 1; epsilon 1e-15; Muon epsilon 1e-5 |
| Schedule | Warmup 114 steps; stable phase to step 9,192; cosine decay for 2,299 steps; minimum LR ratio 0; no restart/rewarmup |
| Seed/tokenizer | Trainer seed 20260910; data seed 20260910 in identical launcher source; bundled 32,000-token tokenizer with identical asset pins |
| Training support | Same `starcoder_tpp10/parent/2026.09.09` cache, 92,928 sequences / 190,316,544 tokens; target rows never select a matched-subset cache |
| Training order | Same named component shuffle keys; StarCoder key `[2476089780, 20260910]`; block shuffle Feistel, IO block 256, window 512; mixture block 2,048 |
| Evaluation | Same PALOMA `dolma_100_programing_languages-tpp10/2026.09.09` validation cache; all examples (`max_eval_batches=null`); per-device eval parallelism 32; same data-parallel mesh and v5p-8 resource recipe |
| Runtime pins | JAX/JAXLIB 0.11.1; NumPy 2.3.5; tokenizers 0.22.2; same `uv.lock` |

Model count and budgets come from the frozen design and plans. The W&B extract does not include the runtime parameter-count summary, so this audit does not claim a fresh leaf recount. The previous design-memory estimate 302,023,680 parameters is not the launched value.

The target parent permutation is fixed by seed 20260915 and SHA-256 `11f9991e96ba8f09d9ff50bf060a0ecd4a604d1d81d1ced2505f0cb96264c5e1`. Matched proxy subsets have their own seeds, but `training_step` selects them only for the matched arm; target percent does not select or regenerate support. Preparation verifies source generations, recipe identities, tokenizer metadata, exact token counts, and parent permutation receipts. No new cache payload reads were needed here.

## Resume checks and remaining limits

Saved interactive logs show p55 restored checkpoint step 10,759, p60 step 6,958, p65 step 2,283, and p80 step 728. These are the four target endpoints whose final rows now carry schema 2. The migration inventory verifies preserved checkpoint generations before and after cancellation. The exact immutable training source initializes its fixed data key, restores the trainer state, then calls `train_loader.iter_from_step(state.step)` (`main/train_lm.py:280–285,318–331,531–536`). The loader uses that step to find the batch offset. There is no source evidence of resetting data or optimizer schedules at resume.

This audit did not read multi-gigabyte checkpoint payloads or perform a replay comparison, so bit-identical optimization through preemption is not independently established. Runtime guards cover four library versions and selected source files, not every installed dependency or XLA flag. The source/config/receipt evidence shows no such differential change; the evaluator change is directly verified and sufficient to invalidate BPB comparisons across the two schemas. Further curve claims require one common metric definition. Retained token loss provides an immediate comparable diagnostic without new training.
