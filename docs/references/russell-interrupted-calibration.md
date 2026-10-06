# SFT evaluation after interrupted calibration

Use `experiments.post_training.russell_rsi.launch_interrupted_calibration_sft` for the separate `evaluate-interrupted` stage. The coordinator runs locally in the foreground. It opens the existing CW02 Iris client and waits for the remote workers. It does not submit a CPU coordinator job.

The stage requires a pinned interruption amendment and the original post-SFT configuration. Only the version and the two interruption pin fields can change. The amendment must record `incomplete_infrastructure`, a null signal gate, no RL authorization, no remaining whole-cohort replacement, and no repeated issued sample. The stage verifies all eight terminal evidence pins. It retains the original qualified model identity, SFT export qualification, source replay, bank, panel, and incumbent evidence barriers.

The graph contains SFT coding evaluation, retention evaluation, and the existing selection stage. It contains no calibration or RL execution. Coding uses the same 32 HumanEval+ and 32 MBPP+ tasks. Retention uses the same three tasks. Selection keeps the incumbent scores of 25/32 and 27/32, retention 1/3, and the existing strict coding gain and retention gates.

Use a new version and the `russell-rsi-interrupted-calibration-sft-only-v1` output namespace. Original calibration artifacts and journals stay unchanged. The selection directory contains `post-sft-selection.json` and `calibration-interruption.json`; the latter retains the interruption pin and the undetermined signal status.

Workers use CW02, batch priority, zero failure and preemption retries, and a six-hour deadline. Retention keeps the existing eight-H100, 32-CPU, 512-GB memory, 2-TB disk limits. The coding stage keeps the existing H100x8 serving and Evalchemy worker limits. Evalchemy keeps `max_retries=1`. The pinned custom coding adapters use a separate 900-second transport retry budget. Each case has one scored sample, but connection, payload, timeout, and selected HTTP failures can cause another HTTP request. A timeout does not prove that the server did not generate a response. See the [pinned local adapters](https://github.com/marin-community/evalchemy/blob/022f3ced2ab888eb017183c44b615d4360b0fe75/eval/serve_eval/local_api.py) and [transport request loop](https://github.com/marin-community/evalchemy/blob/022f3ced2ab888eb017183c44b615d4360b0fe75/eval/robust_api.py).

Retention seals an `EvaluationJournal` before model startup. Completed attempts reconstruct without inference. An incomplete reservation or changed binding refuses replay. Coding reserves its whole batch before server startup and retains the existing per-evaluation records. An incomplete coding batch refuses replay; it does not issue a replacement for partial results.

The coordinator needs an approved source review with `status`, `source_path`, and the full `source_head`. Its checkout must be clean. The source guard checks loaded branch package paths and the actual installed SkyRL f124 commit. Set the branch source roots in `PYTHONPATH` and use the validated f124 interpreter; do not use the current primary runtime by default.

```bash
python -m experiments.post_training.russell_rsi.launch_interrupted_calibration_sft \
  --config-uri PINNED_CONFIG_URI --config-sha256 CONFIG_SHA256 \
  --source-review-uri SOURCE_REVIEW_URI --source-review-sha256 SOURCE_REVIEW_SHA256 \
  --stage evaluate-interrupted --version 2026.10.06.7 --max-concurrent 2
```

This prints the plan. Add `--run` only to execute the reviewed request in the foreground. Do not detach the process. The coordinator uses Iris controller proxy routes for readiness, metrics, and evaluator endpoints; it needs no direct access to a private CW02 serving address.

The direct CW02 client translates a matching CW02 federation pin to local placement. It preserves other scheduling constraints and rejects other cluster directives before RPC. Serving and Evalchemy requests without a cluster directive retain their existing route.

Use `--stage retain --version 2026.10.06.8` after a reviewed launch-failure amendment proves that retention had no reservation, submission, or model request. Its separate configuration pins the unchanged v7 configuration and failure amendment. It runs only the three never-issued retention samples and the unchanged preflights. The retention journal binds both pins before model startup. Coding can continue independently.

Use `--stage select --version 2026.10.06.8` after original coding and v8 retention complete. The selection configuration pins the frozen retention configuration, completed coding producer record, and coding journal result. It can also pin completed coding evidence. The validator checks the original producer identity, model, panel, and saved result before adoption. Both stages reconstruct the same retention artifact; coding pins cannot change its fingerprint.

Selection uses the unchanged promotion rule and issues no coding requests. If coding evidence is absent, only the existing extractor reads the saved result archives. It does not restart v7 coding or change the calibration signal. Selection retains both amendment pins in its separate output directory.

Use `--stage replace-coding --version 2026.10.06.9` only after a pinned transport amendment proves zero benchmark generation requests in the failed v7 execution. The amendment must distinguish the startup probe from benchmark requests. This stage keeps the original model, panel order, 32-case limits, and sampler settings. It uses the controller-minted capability URL for the remote evaluator. The evaluator checks `/models` for the exact model ID before it starts generation. An unreachable endpoint or a different model stops the worker.

The shared Evalchemy worker client requires this endpoint and exact-model check for every evaluation. A valid model card without a context length retains the existing context default.

The separate coding configuration pins the original v7 configuration and transport amendment. Its journal binds these pins, source bytes, and the unchanged adapter retry settings before server startup. It schedules coding only. It does not restart the failed v7 journal or v8 retention.

Use `--stage select-replacement --version 2026.10.06.9` after replacement coding and frozen v8 retention complete. The selection configuration pins both producer configurations, the completed coding producer record, and its matching journal result. Selection adopts the completed retention output with its original fingerprint and issues no evaluation requests. Its record labels coding as `replacement_v9` and retention as `frozen_v8`. Calibration remains incomplete, with a null signal gate and no RL authorization.

Use `--stage select-completed --version 2026.10.06.11` to join completed coding v9 and repaired retention v10. Its separate configuration pins the original science configuration, both producer records and launch proofs, the coding reservation and result, all five retention reservation/result pairs, and both retention summaries. Each launch proof binds the producer's actual source commit, installed runtime, source files, configuration, approved source review, and successful artifact-main preflight. Selection checks these frozen producer bytes; it does not substitute the selection checkout's worker hashes.

The join requires both producers to have completed successfully. It checks all three canonical retention grades, recomputes their summary, and retains the original token preflight rule: at least one probe must pass, and neither probe can have a contract failure. Missing results and infrastructure failures cannot count as zero grades. If completed coding evidence is absent, only the existing CPU extractor reads the saved coding archives. The graph contains no serving, evaluator, retention, or training worker.

The existing promotion gate compares the candidate with the update-eight incumbent. Both coding suite scores and retention must be at least the incumbent scores, and one coding suite must improve. The record retains the original-parent comparison separately. `completed-producers.json` records `replacement_v9`, `repaired_v10`, both actual producer identities, and the pinned join input. Calibration remains `incomplete_infrastructure`, the signal gate stays null, and RL remains unauthorized.

Use `--stage extract-completed-coding --version 2026.10.06.11` to extract coding evidence before retention is available. Its input protocol is `russell-rsi-completed-coding-extraction-v1`; it pins only the original source configuration and the same completed coding records. This stage does not read or adopt retention results. It validates the coding producer and journal, then runs only the existing CPU evidence extractor.

Use `--stage analyze-partitioned-coding --version 2026.10.06.14` for the two-part analysis of completed coding evidence. The configuration pins the original v12 input, its terminal failure and object inventory, the oversized request, and a partition manifest. The failed v12 execution issued no generation request. Its router model card omitted the context length. A separate probe of the served model measured 309803 prompt tokens, above the 262144-token context limit.

The stage validates the completed evidence against its original archives and adopts only that evidence. It divides complete failure/test pairs by byte count. Each part retains the shared evaluation context. Two pinned preflights must show identical token vectors from the served tokenizer and chat renderer. Each measured prompt plus 2048 output tokens must fit the served context. The explicit amendment permits two generation requests, each with low reasoning and zero client retries. It preserves the original 576-KiB evidence limit and all failure/test pairs.

A single CW02 CPU worker runs with two CPUs, 8 GB RAM, 16 GB disk, a 20-minute deadline, batch priority, and zero scheduler retries. The coordinator records its worker reservation before RPC. The worker validates its source and installed f124 runtime, then resolves the documented relay namespace for each new request. It saves each raw response before schema validation. An issued request without a saved response cannot run again. Failed artifacts stay unchanged. Private citation review must approve the canonical capability labels before task construction. The graph has no evaluation or training worker.

Use `--stage recover-analysis-response --version 2026.10.06.16` after the first partition saves a complete response that exceeds the private evidence length limit. The amendment pins the failed v14 configuration, terminal state, full object inventory, and four saved partition files. The recovery reconstructs the original producer identity from its recorded source hashes and worker reservation. Current source hashes apply only to the new worker.

The recovery copies the first partition exactly into a new output and issues only the second request. The total remains two generation requests across the original and recovery artifacts. Request schemas and token preflights stay unchanged. The private response parser preserves the full evidence string and all other schema constraints. The 0.7 confidence threshold, four-label limit, and private citation review stay unchanged.

Use `--stage select-recovered-retention --version 2026.10.06.13` only with a completed saved-submission grade recovery. Its input pins the original coding and retention producers, the CPU recovery amendment, manifest, worker, request, launch, terminal record, and grade records. The validator checks the exact saved patch and private verifier, one completed grade call, and zero model or tokenizer calls. It requires the two original valid grades and preserves the original unavailable result and failure summary.

The stage writes a separate derived retention summary and uses the unchanged promotion gate. It adopts completed coding evidence and schedules no evaluation, grading, serving, or training worker. Its selection record retains the original and derived retention producer identities. Calibration remains incomplete, with a null signal gate and no RL authorization.

## Next teacher diversity study

Use `experiments.post_training.russell_rsi.launch_teacher_diversity_study` with protocol `champion-rsi-teacher-eight-family-diversity-v1`. The stages are `collect`, `sft`, `reload`, `calibrate`, `rl`, and `evaluate`. Run the artifact main without `--run` to inspect the graph. Supply `--source-review-uri` and `--source-review-sha256` for the reviewed clean source and installed f124 runtime. Every stage uses the attached local foreground coordinator and its explicit CW02 client; collection does not submit an RL coordinator.

The configuration pins the unchanged original study for lineage, the completed current coding evidence and selection, and a new canonical capability release and review. The selection must retain incumbent update8. Four retained qualified rows stay unchanged. Six additional eligible families must be distinct from those four and from acceptance and final families. The plan records their previous attempts.

Collection permits at most two new trajectories per family, stops at the first four qualified new families, and refuses replay of incomplete reservations. The dataset has eight full rows and four passes: 32 exposures, batch size 8, and four updates. It retains the 16K context, assistant-only loss, 1e-6 learning rate, tokenizer, training template, and update8 initialization.

Post-SFT requires durable evidence for all optimizer steps 0 through 3 and the strict export and serving qualification. It binds all eight calibration samples for each of the 32 to 36 admitted tasks. The existing completeness and signal gates, conditional four-update GRPO trial, and coding and retention barriers stay unchanged. Missing early metrics cannot use the previous export-recovery amendment.

The prospective diversity SFT telemetry stage uses `experiments.post_training.russell_rsi.launch_teacher_diversity_sft`, version `2026.10.06.17`, and stage `sft` or `reload`. Supply the unchanged v15 collection config and the new reviewed source pins. It keeps the original collection producer and writes complete optimizer metric events to the new SFT artifact under `optimizer-telemetry`. Rank 0 writes each event before the metric call returns; storage errors stop training. Events include the run ID, step and metrics, with a content hash in the filename. Default JSON logging has no storage destination.

The post-SFT qualification, calibration, RL, and evaluation stages still require a separate binding to the version 17 producer. Do not launch these stages with a version 15 training handle. Durable events preserve each complete record; a later seal must reject conflicting metric values for the same run and step.
