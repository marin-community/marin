# Luxical Arctic POC

## Scope

- Train a custom Luxical student with
  `Snowflake/snowflake-arctic-embed-m-v2.0` as the only teacher.
- Use a balanced sample of prose, code, math, and different scripts.
- Compare the student with Luxical-One on the failures in issues
  [#6850](https://github.com/marin-community/marin/issues/6850) and
  [#6855](https://github.com/marin-community/marin/issues/6855).
- Use federated Iris for H100 work. Do not select a cluster, region, or zone.
- Use interactive priority.

## Fixed Inputs

- Run prefix: `LUX-ARCTIC`
- Sample root: `s3://marin-us-east-02a/marin/datakit/sample_0.1b_7d7d8fd7`
- Output root:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-poc`
- Teacher revision: `95c2741480856aa9666782eb4afe11959938017f`
- Current Luxical revision:
  `474cfeb959dd473b3d1cd61da630f566037e69e2`
- Seed: `42`

## Acceptance Gates

- The code and non-Latin groups must not have near-zero vector variance.
- No code or script group can put more than 90 percent of its documents in one
  cluster at the selected K.
- The source probe must improve macro-F1 and worst-group recall over
  Luxical-One.
- A separate probe must compare the Nemotron high, medium-high, and medium
  quality tiers when all three sources are in the sample.
- CPU inference must reach at least 70 percent of Luxical-One speed on the same
  documents. The target is at least 85 percent.

## Runs

### LUX-ARCTIC-000: Federated H100 smoke

- State: Succeeded.
- Purpose: Test H100, S3, Arctic, and Luxical package access.
- Resources: One H100, 4 CPUs, 32 GB RAM, 64 GB disk.
- Priority: Interactive.
- First attempt: Failed during environment setup. The sync package was given as
  `marin`, but the package name is `marin-core`. User code did not run.
- Second attempt: Stopped while it was pending. Its workspace snapshot still
  selected Arctic's optional xFormers path, but the environment does not include
  xFormers.
- Third attempt: Uses pinned teacher files and the standard PyTorch attention
  path. It failed because the shared Hugging Face snapshot had a broken link
  for the Arctic configuration source file.
- Fourth attempt: Loads the repository at the exact revision and refreshes the
  three small remote-code files before model load. Arctic loaded, but BF16
  inference caused a CUDA device assertion.
- Fifth attempt: Uses float32 and synchronous CUDA error reporting to test
  whether the fault is specific to the BF16 path. It found the same fault in
  the RoPE lookup.
- Sixth attempt: Supplies explicit contiguous position IDs to avoid the failing
  expanded-buffer CUDA lookup.
- Result: Arctic produced normalized 256-dimensional MRL embeddings on the
  H100. The exact teacher revision and standard attention path loaded.

### LUX-ARCTIC-001: Small end-to-end fit

- State: Succeeded. The first student failed the quality and speed gates.
- Purpose: Test the sample, teacher embedding, feature build, fit, and
  evaluation path.
- Training size: 65,000 documents, with 5,000 from each source.
- Evaluation size: 13,000 documents, with 1,000 from each source.

## Decisions

- 2026-07-31: Use Arctic only. Do not include LFM.
- 2026-07-31: Use the Arctic tokenizer for the first student. This tests a
  multilingual tokenizer without adding a separate tokenizer training step.
- 2026-07-31: Use balanced source and content strata. Do not use a single
  English-weighted sample.

## Submission Commands

```bash
uv run iris --cluster=marin job run --no-wait \
  --job-name lux-arctic-survey --priority interactive \
  --gpu H100 --enable-extra-resources \
  --cpu 4 --memory 32GB --disk 64GB \
  --sync-package marin-core --extra gpu --extra datakit \
  -- python .agents/projects/luxical-arctic-poc/survey.py

uv run iris --cluster=marin job run --no-wait \
  --job-name lux-arctic-train-001 --priority interactive \
  --gpu H100 --enable-extra-resources \
  --cpu 16 --memory 128GB --disk 256GB --timeout 21600 \
  --sync-package marin-core --extra gpu --extra datakit \
  -- python .agents/projects/luxical-arctic-poc/train.py
```

These commands do not set a target cluster, region, or zone.

## Results

### Data survey

- The survey read 1,300 documents, with 100 documents from each of 13 sources.
- No requested source was missing.
- The sample includes LLVM code, function calls, coding-agent traces, math,
  Japanese, Spanish, German, general web text, and three Nemotron quality
  levels.
- Median document size ranged from 424 characters for WikiTeam to 156,559
  characters for CoderForge.
- The largest inspected document had 2,200,253 characters. Training uses fixed
  head, middle, and tail views to limit this size range.

### Teacher-runtime failure

- The first full fit was stopped because its saved teacher vectors were
  constant.
- Arctic returned NaNs in its original threaded embedding path. The 8-bit
  quantizer changed every NaN to byte value 154, which hid the failure as a
  constant-vector collapse.
- The original Luxical wrapper moved token tensors to CUDA in its background
  tokenizer thread. The main thread could start inference before those
  non-blocking copies and the position IDs were ready. Import order changed
  the timing and made one smoke test pass, but it did not remove the race.
- The fixed wrapper tokenizes on the worker thread and does all CUDA work on
  the inference thread. A full-resource check after this fix produced finite
  vectors for a control before and after S3 access, 8 real documents processed
  one at a time, and 128 real documents processed in one batch.
- The fit now checks parameter values, startup vectors, every teacher batch,
  quantized batch diversity, and total teacher variance.

### Teacher validation

- All 65,000 training documents passed the finite and non-constant block
  checks.
- The saved quantized table has 64,992 unique rows out of 65,000.
- All 256 dimensions vary. Quantized values range from 0 through 255.
- The sum of per-dimension quantized variance is 66,402.31.

### Student result

- Training finished 96 steps. Loss changed from 0.00983 to 0.00230.
- Source probe macro-F1 was 0.633, compared with 0.811 for Luxical-One.
- Worst-source recall was 0.121, compared with 0.228 for Luxical-One.
- Nemotron tier macro-F1 was 0.354, compared with 0.520 for Luxical-One.
- Median CPU speed was 3,669 documents per second, compared with 8,127 for
  Luxical-One. The speed ratio was 0.451, so it failed the 0.70 gate.
- CoderForge and massive function-calling each used one cluster. The student
  also had low effective rank across all sources.
- The first student is not suitable for production use.

### Interpretation

- The teacher runtime is no longer the limiting issue.
- The POC used only 65,000 training documents, a 250,000-item vocabulary, a
  `(96, 1024, 1024, 192)` network, and a 2,048-document batch.
- The Luxical-One example uses about 50 million documents, a 2,000,000-item
  vocabulary, a `(96, 3072, 3072, 192)` network, and a 12,288-document batch.
- The next quality test should use the larger contrastive batch and network,
  while it reuses the saved teacher table. The next speed test must separate
  tokenization, bag-of-words, and network time because this smaller network was
  still slower than Luxical-One.

### Proposed scaling ladder

- Treat the 65,000-document run as a pipeline test, not an absolute comparison
  with the 50-million-document Luxical-One training run.
- Use about 0.75 million, 3 million, 12 million, and 48 million documents
  across the full set of about 150 sources.
- Keep one held-out evaluation set, teacher, tokenizer, student architecture,
  batch size, and training schedule fixed across data rungs.
- Run a reference-recipe check on the saved 65,000 teacher vectors before new
  teacher work. Test the 12,288-document contrastive batch and the
  `(96, 3072, 3072, 192)` network from the Luxical-One example.
- Stop the ladder when a rung does not give enough quality improvement to
  justify the next teacher-embedding cost.

### Collapse gates for the ladder

- Require 100 percent finite vectors.
- Require at least 99 percent unique vectors at four decimal places.
- Reject a source when more than 90 percent of its rows are in one of the 40
  common clusters.
- Set minimum per-source total variance and effective-rank ratios against
  Luxical-One on the same evaluation rows.
- Embed the evaluation rows with Arctic. Measure the correlation between
  student and teacher pairwise cosine values.
- Require source/domain macro-F1, worst-source recall, and Nemotron tier
  macro-F1 to meet the fixed rung gates.

## 2026-07-31 source inventory

- Job `/rav/lux-arctic-source-inventory-001` succeeded in 2 minutes and 4.3
  seconds at interactive priority.
- The registry contained 144 active sources.
- Private mirrored artifacts and inspected Parquet schemas were usable for 143
  sources.
- `stack-v3` was the only missing mirrored artifact.
- The result passes the minimum 140-source gate.
- The report is at
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/source_inventory.json`.
- A CPU attempt had no private-mirror credentials. The successful job used one
  federated H100 without a target cluster, region, or zone.
- The fixed evaluation declares four OOD sources before model results:
  `ghalogs/public`, `massive_function_calling`, `molmo2-cap`, and `svg`.
- Code, multilingual, and standard-text sources cannot use the OOD exception.

### Runtime-structure speed check

- The ladder keeps the Luxical-One BERT tokenizer, 2,000,000-item n-gram
  vocabulary, IDF table, and `(96, 3072, 3072, 192)` network shape.
- A fresh random network with this structure was compared with Luxical-One on
  5,000 mixed synthetic code, multilingual, and technical-text documents.
- With eight CPU threads and five timed repetitions, the median rates were
  1,858.59 documents per second for the fresh student and 1,424.67 for
  Luxical-One.
- The structural speed ratio was 1.305. This check passes the 0.70 minimum and
  0.85 target before training.
- This is a structural check on synthetic text. The fixed data evaluation will
  make the final paired speed decision.

## 2026-07-31 source inventory correction

- The first inventory used uncommitted source definitions from the worktree.
  Those definitions selected Stack v1 and a stale output hash.
- The corrected inventory pins registry revision
  `656d77bff319a851cb775e5bef33570ccfd9a9f8`.
- The current registry contains 147 sources. The corrected job found 146
  usable private artifacts.
- Stack v3 is present and readable at output hash `32b6fa6f`.
- The only missing artifact is the current `ghalogs/public` output at hash
  `55a2fec7`.
- Job `/rav/lux-arctic-source-inventory-002` succeeded in 1 minute and 51.8
  seconds. Its report replaced the stale report at the same fixed URL.
- The stale manifest job was stopped before it could publish a final manifest.
  Job `/rav/lux-arctic-manifest-002` builds the replacement from the corrected
  146-source inventory.

## 2026-07-31 corrected manifest

- Job `/rav/lux-arctic-manifest-002` succeeded in 59 minutes and 1.1 seconds
  with no failure or preemption.
- The fixed manifest contains 146 sources: 28 code, 24 multilingual, 3
  predeclared OOD, and 91 standard-text sources.
- It contains exactly 750,000 rows in the small rung, 3,000,000 rows in the
  large rung, and 74,752 held-out evaluation rows.
- The 0.75M rung is an exact subset of the 3M rung by construction.
- Stack v3 output `32b6fa6f` was sampled and written successfully.
- The manifest digest is
  `f32689b85f4c0818d610914135263a5f410d6bd7b1098fb02cca5dee90923ba3`.
- Job `/rav/lux-arctic-manifest-audit-001` performs the independent count,
  nesting, file, and digest audit.

### Manifest audit result

- Job `/rav/lux-arctic-manifest-audit-001` succeeded in 4 minutes and 8.8
  seconds.
- It independently read every source file and passed the digest, file,
  per-source quota, split, exact-count, and nesting checks.
- The audited total is 3,074,752 rows: 3,000,000 training rows and 74,752
  held-out evaluation rows.

## 2026-07-31 fixed data survey

- Job `/rav/lux-arctic-survey-002` succeeded in 3 minutes and 9.2 seconds
  after an interactive H100 placement wait.
- The survey inspected 14,600 fixed documents: 80 random, 10 shortest, and 10
  longest documents from each of 146 sources.
- All inspected documents were non-constant.
- Raw and normalized text were each 99.9863 percent unique.
- MinHash found 0.3082 percent of documents in near-duplicate pairs at the
  fixed 0.80 estimated-Jaccard threshold.
- The private report is at
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v1/survey/report.html`.
- Job `/rav/lux-arctic-teacher-smoke-001` now tests one complete source with
  the pinned Arctic teacher before the full sharded teacher run.

### Teacher smoke startup failure

- Job `/rav/lux-arctic-teacher-smoke-001` failed its startup finite-vector
  check before it read a private document or wrote an artifact.
- The model parameters were finite, but the four startup control embeddings
  were non-finite.
- The earlier successful teacher job used the same pinned Arctic revision,
  controls, Torch, and Transformers versions, but set seed 42 before model
  construction. The new shard loader had omitted that seed.
- The loader now sets the CPU and CUDA seed before construction, forces
  evaluation mode, and approves only the custom code at the exact pinned
  revision.
- Job `/rav/lux-arctic-teacher-smoke-002` tests this correction before full
  teacher work.

### Teacher startup root cause

- The seed correction did not fix the startup failure.
- A blocking CPU diagnostic showed that the pinned Arctic custom model had an
  invalid non-persistent position buffer. It tried to use position
  `1303655952` for an 11-token input. A second normal-loader attempt produced
  another invalid position value.
- Transformers 5.12.1 loads the custom model through a meta-device path that
  does not preserve deterministic non-persistent buffers from construction.
  The affected pinned model defines four such buffers: position IDs, RoPE
  inverse frequencies, and the RoPE cosine and sine caches.
- The wrapper now rebuilds those four buffers from the pinned model config and
  checks them before inference.
- Job `/rav/lux-arctic-diagnostic-003` succeeded. CPU and GPU hidden states
  were finite with zero NaNs and infinities. GPU results were finite with and
  without explicit position IDs. All eight token controls and all three
  multilingual/code/text vectors were finite and distinct.
- Job `/rav/lux-arctic-teacher-smoke-003` now runs one complete source through
  the corrected teacher.
- The durable incident record is https://echo.oa.dev/wiki/49.

### Teacher smoke result

- Job `/rav/lux-arctic-teacher-smoke-003` succeeded on 21,889 real documents.
- It processed three fixed windows per document at about 90 documents per
  second.
- No startup, real-data batch, finite-value, or constant-quantization check
  failed.
- The source output is complete and can be reused by the eight-shard teacher
  run.
- An initial eight-replica job was stopped while it was waiting for all eight
  H100s as one gang. It did not run a teacher task.
- The full teacher is now eight independent interactive jobs with explicit
  shard coordinates. This keeps the same balanced assignment and lets
  federated Iris place each H100 independently.

### Evaluation and provenance checks

- The teacher audit now checks the exact manifest digest, shard coordinates,
  teacher identity, and teacher revision in each shard report.
- Training now requires the complete teacher audit and records the SHA-256
  digest of each student model.
- Evaluation checks that digest after it downloads the student.
- The paired CPU benchmark uses 20,000 rows spread across all fixed evaluation
  data. It alternates baseline and student measurements for five repetitions.
- Code, multilingual, and standard-text macro-F1 deltas are now separate
  required gates. Each must be no more than 0.02 below Luxical-One.
- The evaluation also reports global cluster sizes, effective cluster count,
  source-cluster normalized mutual information, and group-level values. These
  diagnostics cover the catch-all cluster failure in issue 6855. They do not
  add a new pass gate to the agreed goal.

### Teacher preemption

- At 10:51:14 UTC, Kueue preempted teacher shards 4, 6, and 7 to admit another
  workload. The task event is `WorkloadEvictedDueToPreempted`.
- This was one common infrastructure event on three different source types. No
  data or model check failed.
- Iris scheduled one retry for each shard. Shard 7 resumed first. It reused its
  seven atomic source files and restarted only the incomplete eighth source.
  Shards 4 and 6 are waiting for H100 placement.
- The job summaries count the preemption in the failure field even though the
  task events identify a Kueue preemption.
- The retries for shards 4 and 6 stayed scheduling-gated after other shards
  released H100s. Their tasks were still in the build state and had no worker.
  Those two pending jobs were stopped and submitted through federation again
  as `shard-4-r1` and `shard-6-r1`, with two retry attempts each. Both new jobs
  reached the running state in about one minute.

### Teacher audit result

- All eight teacher shards completed.
- Job `/rav/lux-arctic-teacher-audit-001` succeeded.
- The audit verified 146 source files and 3,074,752 rows against manifest
  `f32689b85f4c0818d610914135263a5f410d6bd7b1098fb02cca5dee90923ba3`.
- Every source has 256 `uint8` dimensions, and every dimension varies in every
  source.
- The minimum exact per-source unique fraction is 0.8914066426.
- The audit artifact is
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v1/teacher-arctic-v1/audit.json`.
- Echo milestone: https://echo.oa.dev/logs/1590.

### First 0.75M training attempt

- Job `/rav/lux-arctic-train-750k-001` loaded and aligned all 750,000 rows.
- The first optimization batch failed because Arrow tried to join more than
  the 32-bit string offset limit during shuffled selection.
- The loader now casts the combined text column to Arrow `large_string`.
- A local shuffled-selection check, repository pre-commit checks, and Pyrefly
  passed.
- Fix commit:
  `9c88379885ce2342d02a3a03c71aadd1f2964107`.

### 0.75M training result

- Job `/rav/lux-arctic-train-750k-002` succeeded.
- It trained 750,000 fixed rows for 186 steps and three epochs.
- Loss changed from 0.0055073155 to 0.0029378380.
- Student model SHA-256:
  `7806241aaf7865215d7cc37d5b26e6a596e5dc1050529a6bdde29da131b889a1`.
- Model:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v1/students/750k/luxical-arctic.npz`.
- Training report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v1/students/750k/training.json`.
- Echo milestone: https://echo.oa.dev/logs/1592.

### 0.75M evaluation result

- Job `/rav/lux-arctic-eval-750k-001` succeeded.
- Required gates passed: finite vectors, overall uniqueness, worst-source
  recall delta, and minimum CPU speed.
- Required gates failed: regular-source collapse, Arctic fidelity,
  source macro-F1, code macro-F1, multilingual macro-F1, and standard-text
  macro-F1.
- Student vectors are 100 percent finite and 99.9612 percent unique after
  rounding to four decimal places.
- CPU rates are 8,771.55 documents per second for the student and 8,695.67 for
  Luxical-One. The ratio is 1.00873.
- Source macro-F1 is 0.26189 for the student and 0.62338 for Luxical-One.
- Worst-source recall is 0 for the student and 0.003906 for Luxical-One.
- Arctic cosine Spearman is 0.82682 for the student and 0.86591 for
  Luxical-One.
- Code, multilingual, and standard-text macro-F1 deltas are -0.36074,
  -0.65799, and -0.27374.
- `biocorpus` puts all held-out rows in one cluster. All 143 regular sources
  fail at least one member of the composite cluster, uniqueness, rank, or
  variance check.
- Evaluation report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v1/evaluation/750k/report.html`.

### 3M training result

- Job `/rav/lux-arctic-train-3m-001` succeeded.
- It trained 3,000,000 fixed rows for 735 steps and three epochs.
- Loss changed from 0.0055917976 to 0.0012042067.
- Student model SHA-256:
  `e6a78c93c0ecea83290095acf7cae4a3338754588b2705cdf4ccde41b17cd8f7`.
- Model:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v1/students/3m/luxical-arctic.npz`.
- Training report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v1/students/3m/training.json`.
- Echo milestone: https://echo.oa.dev/logs/1594.

### 3M evaluation result

- Job `/rav/lux-arctic-eval-3m-001` succeeded.
- Eight of ten required gates pass.
- Required failures are regular-source collapse and multilingual macro-F1.
- Student vectors are 100 percent finite and 99.9612 percent unique after
  rounding to four decimal places.
- CPU rates are 8,558.12 documents per second for the student and 8,570.02 for
  Luxical-One. The ratio is 0.99861.
- Source macro-F1 is 0.63005 for the student and 0.62338 for Luxical-One.
- Worst-source recall is 0.007812 for the student and 0.003906 for
  Luxical-One.
- Arctic cosine Spearman is 0.92687 for the student and 0.86591 for
  Luxical-One.
- Code, multilingual, and standard-text macro-F1 deltas are +0.04010,
  -0.08838, and +0.01909.
- `glm-5.2-kernelgym-rollouts` puts all held-out rows in one cluster.
- Eighty-two of 143 regular sources fail at least one composite collapse
  member. The overlapping counts are 49 cluster concentration, 65 rank ratio,
  27 variance ratio, and one uniqueness failure. All three present OOD
  sources also fail.
- Luxical-One also exceeds the 90 percent cluster concentration limit on 43
  regular sources. Thirty-three failures are shared, 16 are student-only, and
  10 are baseline-only. The absolute rule is not a sufficient collapse test.
- Evaluation report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v1/evaluation/3m/report.html`.
- Echo milestone: https://echo.oa.dev/logs/1595.

### Peer review and corrected run

- The required peer review found two method defects.
- The first sampler favored early Parquet shards.
- The source probe scored rows that also trained the probe.
- The first quality results are provisional. They do not support the final
  model decision.
- Commit `aaa870dac` uses uniform global row sampling and a separate probe
  evaluation split.
- The evaluation now uses three clustering seeds and paired source bootstrap
  intervals.
- Teacher Parquet metadata now binds each file to the manifest and teacher
  revision.
- The report records a disposition for all 28 review findings.
- Job `/rav/lux-arctic-manifest-v2-001` was stopped before it wrote an
  artifact. One source had 1,787 Parquet footers and exposed a serial scan.
- Commit `2c099097e` reads source footers through 16 bounded I/O workers.
- Job `/rav/lux-arctic-manifest-v2-002` was stopped before its write phase.
- Commit `f12c1fcc7` also uses the bounded workers for selected-row reads.
- Job `/rav/lux-arctic-manifest-v2-003` reached the write phase. Exact
  independent-row sampling caused one network data read for almost every
  selected shard.
- Commit `ebe09a12e` uses 64 uniform circular row blocks per source. Every row
  has equal marginal probability, while object reads stay bounded.
- Job `/rav/lux-arctic-manifest-v2-004` builds the corrected manifest at
  interactive priority on federated Iris.
- Echo milestone: https://echo.oa.dev/logs/1596.

### Corrected manifest and survey

- Job `/rav/lux-arctic-manifest-v2-004` succeeded.
- The corrected manifest SHA-256 is
  `4aea19379cb6b7414d80f0b72c868f239e9247c05c3a703a26b19a059599f211`.
- It has 146 sources, 750,000 and 3,000,000 nested training rows, and 74,752
  held-out rows.
- The source groups have 28 code, 24 name-matched multilingual, 91 standard,
  and 3 OOD sources.
- Job `/rav/lux-arctic-manifest-audit-v2-001` passed.
- The audit checked 3,074,752 rows from 5,341 selected input files.
- It verified Stack v3 output hash `32b6fa6f`.
- The first corrected survey job failed before data access because it synced
  `marin-core` without the workspace `marin-dupekit` package.
- Job `/rav/lux-arctic-survey-v2-001-r1` synced both packages and succeeded.
- All 14,600 survey documents are non-constant.
- The raw and normalized unique fractions are both 0.9995890411.
- The near-duplicate document fraction is 0.0033561644.
- Survey artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/survey/report.html`.
- Echo milestone: https://echo.oa.dev/logs/1597.

### Corrected teacher recovery and audit

- One H100 teacher task disappeared after it wrote six complete source files.
- Seven replacement H100 jobs stayed scheduling-gated because no H100 capacity
  was available. They were stopped and replaced with seven federated,
  interactive GB200 jobs.
- All seven GB200 jobs succeeded with zero failures and zero preemptions.
- Corrected teacher audit job
  `/rav/lux-arctic-teacher-audit-v2-gb200-001` succeeded.
- The audit verified 146 source files and 3,074,752 rows against manifest
  `4aea19379cb6b7414d80f0b72c868f239e9247c05c3a703a26b19a059599f211`.
- Every source has 256 varying `uint8` dimensions. The minimum exact
  per-source unique fraction is 0.8956096670.
- Audit artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/teacher-arctic-v1/audit.json`.
- The teacher artifacts contain one full H100 shard and six H100 source files
  from a second shard. The remaining teacher files used GB200.
- Incident record: https://echo.oa.dev/wiki/52.
- Echo milestone: https://echo.oa.dev/logs/1625.

### Corrected 0.75M result

- Training job `/rav/lux-arctic-train-v2-gb200-750k-001` succeeded in 6
  minutes 5 seconds.
- It trained 750,000 rows for 186 steps. Loss changed from 0.00550099 to
  0.00310098.
- Model SHA-256:
  `7e5e9202272c27e9c83cc63d048bc4d5ec7f42dd65c3465d3b875af4c902c709`.
- The first evaluation failed when OpenBLAS exceeded its thread-region limit
  on a high-core-count worker.
- Commit `0b82ba40f` applies the fixed eight-thread limit to the complete
  evaluation.
- Replacement evaluation job `/rav/lux-arctic-eval-v2-gb200-750k-r1`
  succeeded in 3 minutes 2 seconds.
- Four of ten required gates pass. CPU speed ratio is 1.00032. Source
  macro-F1 delta is -0.37034. Arctic fidelity delta is -0.05165.
- Code, multilingual, and standard macro-F1 deltas are -0.43098, -0.61964,
  and -0.28845.
- All 143 regular sources fail at least one composite collapse test.
- Evaluation artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/750k/report.json`.

### Corrected 3M result and decision

- Training job `/rav/lux-arctic-train-v2-gb200-3m-001` succeeded in 31
  minutes 28 seconds.
- It trained 3,000,000 rows for 735 steps. Loss changed from 0.00558436 to
  0.00124420.
- Model SHA-256:
  `395aaa10ff2cbabcff18ceabc8a575e1ea4fb49a0ebd64a894581d48f6b76c5a`.
- Evaluation job `/rav/lux-arctic-eval-v2-gb200-3m-001` succeeded in 3
  minutes 9 seconds.
- Eight of ten required gates pass. CPU speed ratio is 0.99470. Source
  macro-F1 delta is +0.00210. Arctic fidelity delta is +0.05824.
- Code, multilingual, and standard macro-F1 deltas are +0.03053, -0.10605,
  and +0.01964.
- The multilingual paired-source bootstrap interval is [-0.21184, -0.01342].
- Ninety-seven of 143 regular sources fail at least one composite collapse
  test. The overlapping counts are 55 cluster, two uniqueness, 88 rank, and
  29 variance failures.
- Luxical-One has 52 regular sources above the same absolute cluster limit.
- The current 3M student is not viable under the fixed gates because the
  multilingual and composite-collapse gates fail.
- Evaluation artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/3m/report.json`.
- All 24 accepted peer-review findings and the one partial disposition were
  checked again after the result update. No accepted item remains open.

### Direct Arctic representation gates

- Commit `d4480e420` adds a direct evaluation of the stored Arctic vectors.
- The commit is based on `origin/main` commit `9ac45e4de`.
- Submission command:

```bash
uv run iris --cluster=marin job run --no-wait \
  --job-name lux-arctic-teacher-gates-v2-gb200-001 \
  --priority interactive --gpu GB200x1 --enable-extra-resources \
  --cpu 16 --memory 128GB --disk 128GB --timeout 3600 \
  --sync-package marin-core --extra cpu --extra datakit \
  -- python .agents/projects/luxical-arctic-poc/evaluate_teacher.py
```

- The command did not set a target cluster, region, or zone.
- Job `/rav/lux-arctic-teacher-gates-v2-gb200-001` succeeded in 2 minutes
  36.91 seconds. It had no failures or preemptions.
- The evaluation used all 74,752 held-out rows.
- Arctic passed six of eight direct gates.
- It failed `regular_source_collapse` and `multilingual_macro_f1`.
- Source macro-F1 was 0.66915, compared with 0.61727 for Luxical-One. The
  delta was +0.05188 with interval [+0.03381, +0.06912].
- Code macro-F1 was 0.79995, compared with 0.68089. The delta was +0.11906
  with interval [+0.08678, +0.15320].
- Multilingual macro-F1 was 0.72857, compared with 0.79561. The delta was
  -0.06704 with interval [-0.14267, +0.00201].
- Standard macro-F1 was 0.63070, compared with 0.56887. The delta was
  +0.06183 with interval [+0.04826, +0.07555].
- Arctic had finite fraction 1.0 and four-decimal unique fraction 0.999759.
- Sixty of 143 regular sources failed the composite collapse gate. The counts
  were 15 of 28 code, 9 of 24 multilingual, and 36 of 91 standard sources.
- The overlapping failure reasons were 59 cluster-concentration failures and
  one uniqueness failure. No source failed the rank or variance checks.
- The minimum Arctic-to-Luxical rank ratio was 1.00858. The minimum variance
  ratio was 1.54833.
- Luxical-One itself has 52 absolute cluster-concentration failures. The
  category counts are 9 code, 17 multilingual, and 26 standard sources.
- Arctic improved the global code distribution. Its largest code cluster
  share was 0.17055, compared with 0.21819 for Luxical-One.
- Arctic code effective cluster count was 11.14955, compared with 10.30169.
- Arctic code source-cluster NMI was 0.52520, compared with 0.42133.
- Interpretation: Arctic does not show the rank, variance, or modality-wide
  code collapse from #6850.
- Arctic still puts more than 90% of each of 15 code sources in one global
  cluster. Luxical-One does this for nine code sources.
- Thus the fixed composite gate fails, but the failure is source concentration
  and not a low-rank or constant-vector collapse.
- JSON artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-arctic-v1/report.json`.
- HTML artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-arctic-v1/report.html`.
- Cross-agent peer review of the evaluator, report logic, and logbook result
  returned no findings.
