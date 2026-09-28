# Completed 1e21 ladder accuracy inventory — 2026-09-14

Read-only audit for the newly authorized accuracy evaluations. No jobs were submitted, no experiment state was changed, and no model weights or evaluation datasets were downloaded outside their region.

## Release now

| Mixture | Fieldbook run | Final checkpoint |
| --- | --- | --- |
| Proportional | `run_01kvvvwj657tsjqbhqpcmkdn8f` | `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/proportional_1e21-2f1a48/hf/step-22056` |
| UniMax-8 | `run_01kvvvwj8n4hza9yt79bg2g0jz` | `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/unimax8_1e21-d685cd/hf/step-22056` |

Both training artifacts are `SUCCESS`; final native checkpoint metadata and metrics are at step 22056. All three safetensor shards named by each HF index exist; weight bytes total 13,532,475,672 per export. Tokenizer files exist. The exact HF paths must be passed with latest-checkpoint discovery disabled. The old June server manifest names intermediate steps 9999/9987 and must not be reused.

Both models have 3,383,104,000 total parameters, 2,726,433,280 nonembedding parameters, 26 Qwen3 layers, hidden dimension 2560, 20 attention/KV heads and a 4096-token context. They use a 128,256-entry Llama tokenizer, not the stock Qwen3 tokenizer. Training used 46,256,881,664 tokens, 22,057 steps, batch 512, sequence 4096, trainer seed 0, and data seeds 660704/660705. The archived training manifest is `reference_outputs/delphi_baseline_mixtures_issue6607_20260623/training_manifest.csv`.

All relevant artifacts are in `marin-us-east5`. Run the parent and child in east5, pin the parent zone us-east5-a, and preserve regional cache/output paths. Current final Uncheatable BPB is 0.7213719487 / 0.7150214911; these are identity checks, not accuracy results.

## Not finished at this audit

The other four intended matched 1e21 endpoints have no final step 22056 export:

- MARINER Uncheatable: `lwspu_u_snc_cap06_1e21_seed666206-4da478`; latest native checkpoint 15000, HF export 10000.
- MARINER suite: `lwspu_t9_snc_cap08_1e21_seed662005-e8e9d7`; latest native checkpoint 5000, no HF export.
- Matched Olmix Uncheatable: `olmixq_u_kl0p05_cap04_1e21_seed666206-54791f`; no saved native checkpoint or HF export yet.
- Matched Olmix suite: `olmixq_t9_kl0p005_cap04_1e21_seed662005-3f95f2`; no saved native checkpoint or HF export yet.

Their executor statuses are RUNNING; this audit does not infer Iris liveness from those markers. Do not substitute historical Olmix policies: those were trained from a different fitting protocol and are slated for replacement in the paper. `launch_inventory.json` gives all six exact roots and run IDs.

## Evaluation coverage and retained evidence

The existing `OLMO_BASE_EASY_OVERLAP_TASKS` in `experiments/evals/olmo_base_easy_overlap.py` supplies eleven task families: MMLU, ARC Easy, ARC Challenge, CommonsenseQA, HellaSwag, WinoGrande, SocialIQA, PIQA, SciQ, LAMBADA OpenAI and MedMCQA. MMLU expands to 57 subjects, for 67 leaf tasks. All are five-shot except zero-shot LAMBADA. Using the standard labeled splits gives 44,248 examples per model. These evaluate choice/continuation accuracy without code execution or generated-answer grading.

The preferred small initial release is one full eleven-family Levanter evaluation per completed model on v6e-8, retaining the complete results object: all numeric metrics, task/subject results, task configs/revisions, document-level samples, likelihoods and choices. Use `log_samples=True`, `sample_log_all=True`, `max_logged_samples_per_task=None`, and `drop_samples_after_metrics=False`; no limit on evaluation examples. These two jobs can be queued concurrently. This is an Easy-overlap accuracy suite, not the full official OLMoBaseEval Easy suite; math/code generation can be added separately if requested.

Current locked lm-eval revision: `d5e3391f22cde186c827674d5c3ec7c5f4fe0cab` (Stanford fork, uv.lock). Save exact runtime/package versions and rendered leaf-task configurations in every result receipt. Report a headline macro only once its definition and complete leaf/group coverage have been checked; do not conflate BPB, choice accuracy and generation accuracy.

## Cache audit and launch pitfalls

`gs://marin-us-east5/raw/eval-datasets/olmo-base-easy-overlap-v1` contains 1,218 objects totaling 880,355,765 bytes. The layout-2 `datasets/` and `hub/` content includes all 67 dataset configurations; MMLU's 57 subject test splits total 14,042 examples. The root manifest omits MMLU from cached_datasets despite its complete nested files. `dataset_config_inventory.json` records all split counts, schemas and exact paths, while `cache_object_inventory.json` pins every object generation/MD5.

- Marin's Levanter evaluator currently ignores `eval_datasets_cache_path`. An explicit regional sync and offline-loading preflight are required; merely passing that argument is not sufficient. `load_eval_datasets_from_gcs` is reusable, but raises no error on failed sync, so verify its return and task readiness.
- Cache syncing belongs on the east5 worker. Set the HF cache paths before library import, load the cached files, then use offline flags so hidden internet fallback does not change task revisions.
- The cached SocialIQA is a legacy scripted dataset; offline task loading must be checked against the locked current datasets/lm-eval environment. Existing cached Arrow data are available if a source-script migration is necessary.
- `LevanterLmEvalEvaluator.evaluate` catches result-upload failures and returns. A launcher must validate the durable GCS results before accepting success or skipping a rerun.
- Use the checkpoint's own config and tokenizer. Do not use default pretrained Qwen3 configuration/tokenizer values.
- Preserve full samples; the older baseline launcher collector discards stderr keys and is insufficient as the only retained record.
- `use_wandb_tracker=False` avoids needing a new W&B publication; Fieldbook and immutable GCS receipts suffice unless parent decides existing authorization covers mirrored metrics.

## Evidence files

`launch_inventory.json` is the compact release inventory. `final_export_inventory.json` contains final config/index/metadata hashes and GCS shard metadata. `live_gcs_root_inventory.json` records the completion boundary. `cache_object_inventory.json`, `dataset_config_inventory.json`, `cache_subdirs.json` and `eval_cache_manifest.json` establish regional data availability. The three `*_fieldbook_status.json` files are the read-only ledger snapshots used before storage inspection.
