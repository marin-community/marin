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

### Arctic Embed Large v2.0 teacher test

- Commit `2dbef9c06` adds the fixed Large teacher test. Commit `81d626a50`
  adds bounded readback output for the saved report.
- The test uses `Snowflake/snowflake-arctic-embed-l-v2.0` at revision
  `ac6544c8a46e00af67e330e85a9028c66b8cfd9a`.
- It keeps the exact 74,752 evaluation rows, source groups, three document
  windows, 512-token window limit, 256-dimensional truncation, quantization,
  probe split, clustering seeds, and gate limits from the Medium test.
- Submission command:

```bash
uv run iris --cluster=marin job run --no-wait \
  --job-name lux-arctic-large-teacher-v2-gb200-001 \
  --priority interactive --gpu GB200x1 --enable-extra-resources \
  --cpu 16 --memory 128GB --disk 128GB --timeout 7200 \
  --sync-package marin-core --extra gpu --extra datakit \
  -- python .agents/projects/luxical-arctic-poc/evaluate_teacher_large.py
```

- The command did not set a target cluster, region, or zone.
- The first submission used the `cpu` package extra and failed before model
  loading because CUDA was not available. It wrote no vectors or reports.
- The corrected job `/rav/lux-arctic-large-teacher-v2-gb200-001` succeeded in
  28 minutes 44.22 seconds. It had no failures or preemptions.
- The embedding phase took 1,635.47 seconds and processed 45.71 documents per
  second. Each document uses three windows. This is not a student-speed test.
- Large passes six of eight direct gates. It fails
  `regular_source_collapse` and `multilingual_macro_f1`.
- Probe macro-F1 results:

| Representation | Overall | Code | Name-matched multilingual | Standard |
| --- | ---: | ---: | ---: | ---: |
| Luxical-One | 0.61727 | 0.68089 | 0.79561 | 0.56887 |
| Arctic Medium v2.0 | 0.66915 | 0.79995 | 0.72857 | 0.63070 |
| Arctic Large v2.0 | 0.66171 | 0.78894 | 0.73634 | 0.62040 |

- Large minus Medium overall macro-F1 is -0.00743 with interval
  [-0.01200, -0.00300].
- Large minus Medium code macro-F1 is -0.01101 with interval
  [-0.01920, -0.00345].
- Large minus Medium multilingual macro-F1 is +0.00777 with interval
  [-0.00817, +0.02441].
- Large minus Medium standard macro-F1 is -0.01030 with interval
  [-0.01519, -0.00566].
- Large trails Luxical-One multilingual macro-F1 by 0.05927. The interval is
  [-0.12500, +0.00438].
- All Large vectors are finite. Exact and four-decimal uniqueness are both
  0.999759.
- Forty-four of 143 regular sources fail the composite rule. The counts are 13
  of 28 code, zero of 24 multilingual, and 31 of 91 standard sources.
- The overlapping failure reasons are 43 cluster-concentration failures and
  one uniqueness failure. No source fails rank or variance checks.
- The minimum Large-to-Luxical rank ratio is 1.02636. The minimum variance
  ratio is 1.51776.
- Large code largest-cluster share is 0.20752, compared with 0.17055 for
  Medium and 0.21819 for Luxical-One.
- Large code effective cluster count is 9.86327, compared with 11.14955 for
  Medium and 10.30169 for Luxical-One.
- Large code source-cluster NMI is 0.50727, compared with 0.52520 for Medium
  and 0.42133 for Luxical-One.
- Interpretation: Large reduces absolute per-source concentration failures
  from 60 to 44. It does not fix the multilingual quality gate. It is lower
  than Medium overall and for code and standard text. Keep Medium rather than
  making Large teacher labels.
- Method limits: source groups use the fixed name rules, and each teacher
  window is limited to 512 tokens. They remain fixed for the controlled model
  comparison.
- JSON artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-arctic-l-v2.0/report.json`.
- HTML artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-arctic-l-v2.0/report.html`.

### Corrected Arctic Embed Large v2.0 teacher test

- Peer review found that the first Large run used a different attention and
  pooling implementation from Medium. The result above is superseded.
- Commit `beb363f2b` makes Large use the pinned Luxical pooling path, eager
  attention, explicit position IDs, float32 inference, and the same token
  staging as Medium.
- Corrected job `/rav/lux-arctic-large-teacher-v2-gb200-002` succeeded in 29
  minutes 13.87 seconds. It had no failures or preemptions.
- The corrected run evaluated all 74,752 held-out rows.
- Probe macro-F1 results:

| Representation | Overall | Code | Name-matched multilingual | Standard |
| --- | ---: | ---: | ---: | ---: |
| Luxical-One | 0.61727 | 0.68089 | 0.79561 | 0.56887 |
| Arctic Medium v2.0 | 0.66915 | 0.79995 | 0.72857 | 0.63070 |
| Arctic Large v2.0 | 0.66810 | 0.79003 | 0.74798 | 0.62683 |

- Large minus Medium overall macro-F1 is -0.00105 with interval
  [-0.00616, +0.00408].
- Large minus Medium code macro-F1 is -0.00992 with interval
  [-0.01922, -0.00150].
- Large minus Medium multilingual macro-F1 is +0.01941 with interval
  [+0.00156, +0.03893].
- Large minus Medium standard macro-F1 is -0.00388 with interval
  [-0.00908, +0.00120].
- Large trails Luxical-One multilingual macro-F1 by 0.04763. The interval is
  [-0.11048, +0.01441].
- Large passes six of eight direct gates. It fails
  `regular_source_collapse` and `multilingual_macro_f1`.
- All Large vectors are finite. Exact and four-decimal uniqueness are both
  0.999759.
- Fifty-one of 143 regular sources fail the composite rule. The counts are 15
  of 28 code, zero of 24 multilingual, and 36 of 91 standard sources.
- The failure reasons are 50 cluster-concentration failures and one uniqueness
  failure. No source fails rank or variance checks.
- The minimum Large-to-Luxical rank ratio is 1.08243. The minimum variance
  ratio is 1.44775.
- Median Large code values are 0.13958 largest-cluster share, 11.49555
  effective clusters, and 0.52480 source-cluster NMI.
- Large reduces Medium's composite source failures from 60 to 51. It does not
  show rank, variance, constant-vector, or modality-wide code collapse.
- The Large embedding phase took 1,490.43 seconds and processed 50.15
  documents per second on one GB200-class worker. The complete source loop
  took 1,659.20 seconds.
- This rate is not student speed. The run did not train a Large-distilled
  student or measure Medium teacher speed.
- Interpretation: keep Medium as the default teacher. Large gives a measured
  multilingual gain and fewer concentration failures, but it gives a measured
  code loss and no overall gain.
- The fixed 512-token windows do not test Large's full context limit.
- The peer-review report records all review findings and dispositions in
  `.agents/projects/luxical-arctic-poc/arctic-large-review.md`.
- JSON artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-arctic-l-v2.0-v2/report.json`.
- HTML artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-arctic-l-v2.0-v2/report.html`.

### FastTransformer student background and speed gate

- Research effort: medium. The question was whether the quality-classifier
  FastTransformer can replace the collapsing Luxical sparse student while it
  keeps Luxical-class document throughput.
- The treatment keeps the Arctic Medium v2.0 teacher, fixed 3M rows, nested
  source-balanced ladder, 256-dimensional output, Gram-KL loss, temperature 3,
  and the 74,752-row holdout. Only the student architecture and input treatment
  change.
- The reused FastTransformer body has mean/max/min local pooling, two layers,
  four heads, and a normalized 256-dimensional output head. The scalar quality
  head and MSE training path are not suitable for embedding distillation.
- Prior quality-classifier work in [PR
  #7191](https://github.com/marin-community/marin/pull/7191) measured high
  accelerator throughput and found that input work can set the end-to-end
  rate. [Issue #6850](https://github.com/marin-community/marin/issues/6850)
  defines the code-collapse problem. [Issue
  #6855](https://github.com/marin-community/marin/issues/6855) tracks this
  Arctic-distillation POC. The Luxical objective comes from
  [training.py](https://github.com/datologyai/luxical/blob/main/luxical/training.py).
- The first treatment used three 160-token `multilingual-e5-small` windows and
  512 model tokens. CPU throughput was 375.05 documents per second, versus
  7,702.66 for stock Luxical. The ratio was 0.0487. Accelerator throughput was
  376.07 documents per second. Equal CPU and accelerator rates identified the
  tokenizer path as the limit.
- The second treatment used `o200k_base` with the same three token windows.
  CPU throughput increased to 3,540.27 documents per second, versus 8,831.43
  for stock Luxical. The ratio was 0.4009. One B200-class worker reached
  4,377.30 documents per second. This result still failed the 0.70 minimum.
- The selected treatment reuses the pinned stock Luxical Rust
  `ArrowTokenizer`. It forms one 256-token input from three 256-character
  document regions. Its compact vocabulary has 30,524 rows. The full model has
  9,653,760 parameters.
- The selected CPU treatment reached 23,228.50 documents per second. Stock
  Luxical reached 8,957.64 documents per second in the paired run. The ratio
  was 2.5931. One B200-class worker reached 89,405.74 documents per second.
  It passes the 0.70 minimum and 0.85 target speed gates.
- Ranked hypothesis: the 64K rung will stay finite and unique because the
  FastTransformer has no sparse 2M-feature bottleneck. Its source probe can
  still fail because the 256-token input has less text than Luxical. Run the
  fixed collapse and quality gates before 750K training.
- Commits `3f806ec2b`, `5e136601a`, and `d62253c91` contain the model, ladder,
  speed gate, and fixed evaluator.

### FastTransformer 64K first training attempt

- Preparation job `/rav/lux-arctic-fast-student-prepare-b200-001` completed
  all 3M fixed training rows in 7 minutes 12.85 seconds. It had no failures or
  preemptions.
- The first 64K training attempt failed before its first update. JAX rejected
  the Boolean mask in the off-diagonal Gram-matrix function during tracing.
- The fix removes each static diagonal with reshape and slice operations. A
  regression test now compiles the complete contrastive loss with `jax.jit`.
- The focused FastTransformer test file and the required file checks pass.
- The corrected Gram-KL run completed one epoch before the collapse gate
  stopped it. Its effective rank was 1.37056, mean cosine was 0.98716, and
  cosine p99 was 0.999866.
- A separate 2,048-row diagnostic measured the untrained student and its
  teacher on the same rows. The untrained student had effective rank 26.89108,
  mean cosine 0.95852, and cosine p99 0.99356. The Arctic teacher had effective
  rank 61.30561, mean cosine 0.27540, and cosine p99 0.93112.
- This result localizes the new collapse to Gram-KL-only optimization of the
  FastTransformer. It does not localize the collapse to Arctic or to the input
  documents.
- The next 64K treatment adds direct cosine alignment with weight 1.0 to the
  unchanged Gram-KL objective. The direct term makes a constant student costly
  for a diverse teacher. The Gram term continues to preserve pairwise
  geometry.
- Hybrid 64K training completed all three epoch audits in 1 minute 11.68
  seconds. It had no failures or preemptions.
- The first fixed-holdout evaluation attempt stopped before it wrote metrics.
  OpenBLAS used the host CPU count and exceeded its compiled thread table. The
  fast-student evaluator now uses the same fixed eight-thread wrapper as the
  existing Luxical evaluator.
- The corrected 64K holdout evaluation completed all 74,752 fixed rows. It
  passed finite, unique, worst-recall, and speed gates. It failed all quality,
  fidelity, and regular-source collapse gates.
- The 64K student had macro-F1 0.20413, Arctic fidelity 0.45959, 143 regular
  collapse failures, minimum rank ratio 0.02236, and minimum variance ratio
  0.01473. Its CPU speed ratio remained 2.59315.
- The 750K training and evaluation jobs both completed without a failure or
  preemption. The student had macro-F1 0.63566, which is 0.01839 above stock
  Luxical. Code was 0.06824 higher and standard text was 0.01227 higher.
- The 750K multilingual macro-F1 was 0.76573, which is 0.02988 below stock and
  fails the allowed 0.02 loss. Arctic fidelity was 0.87323, which is 0.00559
  above stock.
- The 750K student had 66 regular collapse failures. Its minimum rank ratio was
  0.35554 and its minimum variance ratio was 0.31444. Its training audit rank
  was 31.15826, versus 2.72790 at 64K. This scale improvement justifies the 3M
  rung.
- The 3M training and holdout evaluation jobs completed without a failure or
  preemption. The 3M student passes nine of ten gates. It fails only the
  regular-source collapse gate.
- The 3M student had macro-F1 0.65785, which is 0.04058 above stock. Code was
  0.09832 higher, multilingual was 0.01905 lower, and standard text was
  0.03497 higher. Arctic fidelity was 0.89423, which is 0.02658 above stock.
- Fifty-nine regular sources fail the 3M composite rule. The failure reasons
  are 56 concentration, four uniqueness, and one rank, with overlap. No source
  fails the variance ratio gate. The minimum rank ratio is 0.45300 and the
  minimum variance ratio is 0.52589.
- The 3M training audit rank was 44.73097 and its final loss was 0.26548. The
  fixed CPU speed ratio remained 2.59315.

### FastTransformer peer-review correction

- Peer review found that pooled Arctic fidelity can be inflated by separation
  between within-source and across-source pair groups.
- The corrected gate uses within-source Spearman only. All three fixed models
  were reevaluated with unchanged vectors and rows.
- Stock within-source fidelity is 0.82113. The 64K, 750K, and 3M values are
  0.29437, 0.89338, and 0.90901. The corresponding deltas are -0.52676,
  +0.07225, and +0.08788.
- This correction does not change the final gate count. The 3M student still
  passes nine of ten gates and fails only regular-source collapse.
- The peer-review report and dispositions are in
  `.agents/projects/luxical-arctic-poc/fast-student-report.md`.

### FastTransformer failure attribution and source-geometry ladder

- Commit `80563314f` adds the saved-report failure-attribution audit.
- Attribution job `/rav/lux-arctic-attribution-gb200-001` succeeded in 4
  minutes 2.13 seconds. It had no failures or preemptions.
- The teacher has 60 composite failures and the 3M student has 59. Fifty-four
  sources overlap. Five are student-only and six are teacher-only. The Jaccard
  overlap is 0.83077.
- The student-only failures are four multilingual FinePDF sources and one
  standard algorithmic source. No code failure is student-only.
- Student-to-teacher median rank ratios are 0.41618 for code, 0.34775 for
  multilingual data, and 0.45476 for standard data. The corresponding variance
  ratios are 0.64327, 0.49088, and 0.76096.
- Arctic truncates 84,266 of 224,256 teacher windows at 512 tokens. This is
  37.58% of windows and affects at least one window for 45.64% of documents.
- Truncated window fractions are 70.30% for code, 35.15% for multilingual data,
  and 27.16% for standard data. Source truncation has Spearman correlations
  0.01753 and 0.06443 with teacher and student concentration.
- Attribution artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/fast-student/full/3m/attribution.json`.
- Commit `9ec418b6b` adds a same-source cosine-geometry loss. Commit `07619a766`
  adds fixed weights 0.25 and 0.50 after weight 1.0 failed its quality gate.
- The first weight-1.0 submission used the CPU package extra and failed before
  model load. The corrected GPU submission succeeded in 1 minute 23.45 seconds.
- Weight 1.0 reduced regular failures from 66 to 32 and student-only failures
  from 10 to one. Macro-F1 decreased by 0.05279 and multilingual macro-F1
  decreased by 0.07263.
- The 0.25 and 0.50 training jobs and the two holdout evaluation jobs succeeded
  without failures or preemptions.
- Weight 0.25 reduced regular failures to 46 and student-only failures to three.
  Macro-F1 decreased by 0.02523. Code, multilingual, and standard macro-F1
  decreased by 0.02302, 0.02694, and 0.02485.
- Weight 0.50 reduced regular failures to 43 and student-only failures to one.
  Macro-F1 decreased by 0.03883. Code, multilingual, and standard macro-F1
  decreased by 0.03069, 0.04434, and 0.03804.
- The 0.25, 0.50, and 1.0 treatments all fail the fixed -0.02 quality-loss
  limit. Stop this source-geometry treatment and do not confirm it at 3M.
- Comparison artifacts:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/fast-student/full-source-geometry-w0.25/750k/comparison.json`,
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/fast-student/full-source-geometry-w0.5/750k/comparison.json`, and
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/fast-student/full-source-geometry-w1/750k/comparison.json`.

### Alternative teacher audit

- Commit `1eaa537b7` adds the fixed LFM2.5-Embedding-350M and
  Qwen3-Embedding-0.6B audit. Follow-up commits adapt the pinned LFM
  convolution and select supported SDPA kernels.
- Federated interactive H100 jobs `/rav/lux-teacher-lfm25-350m-h100-001` and
  `/rav/lux-teacher-qwen3-06b-h100-001` succeeded. The LFM job took 14 minutes
  19.27 seconds. The Qwen job took 17 minutes 3.03 seconds. Both jobs had zero
  failures and zero preemptions.
- The audit uses the same 74,752 documents and 146 sources as the Arctic and
  student gates. Each candidate uses three windows, 512 tokens per window,
  1,024 output dimensions, BF16 inference, and 8-bit stored vectors.
- LFM has overall macro-F1 0.68401, code 0.79281, multilingual 0.75653, and
  standard 0.65230. It has 47 regular failures. It fails the multilingual
  quality gate against Luxical-One and the zero-failure collapse gate.
- Qwen has overall macro-F1 0.67664, code 0.80067, multilingual 0.81348, and
  standard 0.62159. It has 46 regular failures. It passes all quality, finite,
  and unique gates. It fails only the zero-failure collapse gate.
- Qwen removes 17 Arctic failures and adds three new failures. Forty-three
  failures overlap. Its failures include 15 code, one multilingual, and 30
  standard sources.
- Qwen is the best tested replacement teacher. Before student training, audit
  a 256-dimensional Qwen projection. This keeps the current student output,
  direct cosine term, storage size, and speed target unchanged.
- LFM report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-lfm2.5-embedding-350m/report.json`.
- Qwen report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-qwen3-embedding-0.6b/report.json`.

### Qwen size and dimension ladder

- Commit `453e3be09` adds native 256-dimensional Qwen3 teacher candidates at
  0.6B, 4B, and 8B. The exact 4B and 8B revisions are
  `5cf2132abc99cad020ac570b19d031efec650f2b` and
  `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`.
- The first plan was to truncate the saved 1,024-dimensional document vectors.
  This is not equal to native MRL inference because the artifact stores vectors
  after three-window normalization and pooling. The ladder reruns each model,
  truncates each window before normalization, and then pools the three windows.
- Job `/rav/lux-teacher-qwen3-06b-256-h100-001` used one federated H100 and
  interactive priority. It succeeded in 24 minutes 0.77 seconds with zero
  failures and zero preemptions.
- The 0.6B 256-dimensional teacher has overall macro-F1 0.63626, code 0.76200,
  multilingual 0.77463, and standard 0.58256. It has 39 regular failures: 11
  code, zero multilingual, and 28 standard sources.
- The treatment reduces failures from 46 at 1,024 dimensions to 39. It loses
  0.04039 overall macro-F1. Its multilingual delta from Luxical-One is
  -0.02098, so it misses the fixed quality limit by 0.00098.
- All vectors are finite. Exact and four-decimal uniqueness are 0.99976. The
  failed gates are `regular_source_collapse` and `multilingual_macro_f1`.
- Continue to the 4B 256-dimensional teacher. Run 8B only if 4B passes all
  quality gates and reduces the best Qwen failure count by at least five, to at
  most 34.
- Report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-qwen3-embedding-0.6b-256/report.json`.
- Exact command:

```bash
uv run iris --cluster=marin job run --no-wait \
  --job-name lux-teacher-qwen3-06b-256-h100-001 \
  --priority interactive --gpu H100 --enable-extra-resources \
  --cpu 16 --memory 128GB --disk 128GB --timeout 14400 \
  --sync-package marin-core --extra gpu --extra datakit \
  -- python .agents/projects/luxical-arctic-poc/evaluate_teacher_candidate.py \
  --candidate qwen3-embedding-0.6b-256
```

### Qwen 4B 256-dimensional stop result

- Launch commit: `c8b4f4016`. Candidate code commit: `453e3be09`.
- Job `/rav/lux-teacher-qwen3-4b-256-h100-001` used one federated H100 and
  interactive priority. It succeeded in 50 minutes 3.62 seconds. It had zero
  failures and zero preemptions.
- The 4B teacher has overall macro-F1 0.64300, code 0.75696, multilingual
  0.77509, and standard 0.59347. Its worst-source recall is 0.01953.
- The 4B teacher has 35 regular failures: ten code, zero multilingual, and 25
  standard sources. Thirty-four sources fail concentration, two fail rank, and
  one fails uniqueness, with overlap.
- All 74,752 vectors are finite. Exact and four-decimal uniqueness are 0.99976.
- Against the 0.6B 256-dimensional result, overall macro-F1 increases by
  0.00674 and the failure count decreases from 39 to 35. Code macro-F1
  decreases by 0.00503.
- Against Luxical-One, multilingual macro-F1 is 0.02052 lower. This misses the
  fixed quality limit by 0.00052. The failed gates are
  `regular_source_collapse` and `multilingual_macro_f1`.
- The 4B model processed 29.19 documents per second during teacher inference.
  The 0.6B model processed 94.50 documents per second. The 4B rate is 0.309
  times the 0.6B rate.
- Decision: do not run the 8B model. The fixed condition required all quality
  gates and at most 34 failures. The 4B result has two failed gates and 35
  failures.
- Report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/teacher-qwen3-embedding-4b-256/report.json`.
- Focused report:
  `.agents/projects/luxical-arctic-poc/qwen-teacher-ladder-report.md`.
- Exact command:

```bash
uv run iris --cluster=marin job run --no-wait \
  --job-name lux-teacher-qwen3-4b-256-h100-001 \
  --priority interactive --gpu H100 --enable-extra-resources \
  --cpu 16 --memory 128GB --disk 128GB --timeout 21600 \
  --sync-package marin-core --extra gpu --extra datakit \
  -- python .agents/projects/luxical-arctic-poc/evaluate_teacher_candidate.py \
  --candidate qwen3-embedding-4b-256
```

### Focused Qwen result peer review

- Peer review first read the wrong worktree because `WIGGLE_PROJECT_DIR` kept
  the tab worktree. The corrected command set this variable to the clean
  research worktree and read only the focused Qwen report.
- Accepted: record batch sizes and the MRL compression-fraction confound. The
  teacher-rate ratio is not an isolated size effect.
- Accepted: add exact per-run rates, worst-source recall, gate names, Arctic,
  group denominators, collapse thresholds, and the zero variance-failure count.
- Accepted: replace local Python commands with the original Iris commands.
  Record that saved-vector reuse prevents a new teacher-rate measurement.
- Accepted: add a next test and storage costs. A train-only 256-to-1,024 head
  can keep the deployed student at 256 dimensions. This test has not run.
- Corrected: the holdout has 143 regular sources, not 142. The candidate reports
  contain 28 code, 24 multilingual, and 91 standard sources.
- Retained: the focused report includes the three saved artifacts because a
  reader must be able to validate its numbers without the larger report.
- Retained: the harness keeps the 8B candidate to record the predeclared ladder
  and pinned revision. The report states that the stop rule prevented its run.
- Retained: the report distinguishes teacher rate from student speed because
  CPU student speed is a fixed project gate.

### Qwen cross-dimension student plan

- Base: current `origin/main` at `e096eccf7`. The research branch rebased
  without a conflict before this change.
- Hypothesis: a 256-dimensional FastTransformer can keep the geometry of the
  1,024-dimensional Qwen3-Embedding-0.6B teacher. A train-only linear head maps
  the student vectors to 1,024 dimensions for direct cosine alignment.
- The Gram-KL loss compares pairwise cosine distributions and does not require
  equal vector dimensions. The direct cosine term uses the train-only head.
- Production inference discards the train-only head. The student architecture,
  tokenizer, 256-dimensional output, and paired CPU speed artifact stay fixed.
- The first rung uses the existing source-balanced 750K rows. Eight independent
  federated H100 jobs create the aligned Qwen labels. A separate audit must find
  750,000 aligned, finite, and non-constant labels before training.
- POC gates: all quality deltas are at least -0.02, student-only failures are at
  most five, and each category median rank and variance ratio is at least 0.50.
  Qwen fidelity must not decrease against Luxical-One. CPU speed must be at
  least 0.85 times Luxical-One.
- Continue to 3M only if the 750K rung passes all POC gates.

### Qwen cross-dimension student result

- Commit `33d52e4d6` adds aligned 1,024-dimensional Qwen labels, a train-only
  256-to-1,024 head, the 750K trainer, and the fixed evaluator.
- Eight federated interactive H100 label jobs succeeded. Each job had zero
  failures and zero preemptions. Their durations were 19.55 to 21.34 minutes.
- The label audit found 750,000 aligned rows across 146 sources. All 1,024
  teacher dimensions vary. The minimum source unique fraction is 0.92481.
- The first CPU audit job failed before data checks because its environment had
  no S3 credentials. The H100 audit succeeded in 1 minute 25.54 seconds.
- The 750K training job succeeded in 1 minute 14.22 seconds. Loss decreased
  from 0.97988 to 0.39807. Final effective rank is 25.35137.
- Final student vectors are finite and fully unique at six decimals in the
  training audit. Mean cosine is 0.96642, and cosine p99 is 0.99109.
- The fixed evaluation job succeeded in 8 minutes 13.66 seconds. It used all
  74,752 held-out documents.
- Student macro-F1 is 0.58236. Its delta from Luxical-One is -0.03491. Code,
  multilingual, and standard deltas are +0.02194, -0.03175, and -0.05512.
- The student has 140 regular failures. All 46 Qwen failures remain, and the
  student adds 94 failures.
- Median student-to-Qwen effective-rank ratios are 0.14605 for code, 0.12351
  for multilingual data, and 0.14135 for standard data.
- Median variance ratios are 0.21543 for code, 0.17355 for multilingual data,
  and 0.25840 for standard data. Each fixed limit is 0.50.
- Qwen within-source Spearman is 0.84779. Its delta from Luxical-One is
  +0.11207. The student keeps pair ordering but compresses pair distances.
- The paired CPU speed ratio remains 2.59315. The deployed model has the same
  256-dimensional output and 9,299,200 parameters.
- The trained alignment head condition number is 6.79018, compared with
  2.95179 at initialization. This supports anisotropic amplification through
  the unconstrained head, but it does not prove sole causality.
- Decision: do not scale this treatment to 3M rows. Test an orthonormal-row
  head at 64K before any new 750K run.
- Focused report:
  `.agents/projects/luxical-arctic-poc/qwen-crossdim-student-report.md`.
- Evaluation artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/fast-student/full-qwen3-06b-1024-crossdim/750k/report.json`.

### Qwen cross-dimension peer review

- Accepted: the experiment changed teacher identity and direct-cosine mapping
  together. It does not isolate the cause. Run native 256-dimensional Qwen
  training before an orthonormal-head treatment.
- Accepted: effective-rank ratios are dimension-confounded for a 256d student
  and 1,024d teacher. Keep them as diagnostics. Gate only dimension-free
  variance ratios.
- Accepted: the Arctic comparison has no 750K attribution artifact. Do not
  compare Arctic and Qwen student-to-teacher ratio tables across rungs.
- Accepted: the CPU speed result is inherited from the conservative paired
  speed artifact. The current student did not get a new speed run.
- Accepted: validate Qwen vector metadata, remove legacy gate booleans from the
  report, and guard the empty failure-set result.
- Accepted: validate embedding dimensions before indexing their shapes.
- Accepted: exclude the scale-invariant alignment head from AdamW decay.
- Accepted: use explicit teacher artifact suffixes to prevent output overlap.
- Accepted: replace weak cross-dimension tests with numeric component,
  loss-reduction, and invalid-shape behavior tests.
- Accepted: move the new JSON storage helpers to `ladder_config.py`.
- Retained: require explicit `--teacher` selection. The teacher is a critical
  training input, and this repository does not add compatibility defaults.
- Retained: keep concrete model and array upload helpers. A writer callback
  would add indirection for two different serialization operations.
- The required checks pass after these changes. The peer-review workflow does
  not require a second review loop.

### Qwen cross-dimension report migration

- The corrected evaluator rerun reached all 146 fixed sources. Remote reads
  then slowed to about 30 seconds per Qwen source. The job was stopped before
  report generation because it would only repeat unchanged numerical work.
- A small migration job updated the existing JSON and HTML reports. It renamed
  the category gate, marked rank ratios as diagnostics, removed legacy gate
  fields, and kept all measured values unchanged.
- The first migration failed before data access because the script directory
  was not on the Python module path. The second failed during setup because it
  did not install the required training dependencies. Neither job read or
  wrote the report.
- Job `/rav/lux-qwen-fast-student-report-migrate-h100-004` succeeded in 58.8
  seconds with zero failures and zero preemptions.
- The stored report has no legacy comparison gates. Its category variance gate
  remains false, and the overall POC result remains false.

## 2026-08-02: GLM-5.2 semantic-label pilot

### Hypothesis

GLM-5.2 can create a source-blind semantic taxonomy for 1,000 fixed evaluation documents.
The taxonomy can supply labels for a direct embedding quality test.

### Method

- Select exactly 1,000 evaluation documents across all 146 sources.
- Give six or seven documents to each source through a fixed hash order.
- Do not include source names in a model prompt.
- Ask GLM-5.2 for document facets, a 30 through 50 bucket taxonomy, and final assignments.
- Use one federated eight-GB200 server at interactive priority.
- Give a stratified, blinded result sample and the frozen taxonomy to Claude.

### Gates

- GLM returns 1,000 valid descriptions and 1,000 valid assignments.
- The fallback bucket contains at most 10 percent of the documents.
- Manual review accepts at least 90 percent of the sampled assignments.
- Claude exact primary-bucket agreement is at least 80 percent on 20 blinded documents.
- Claude language and document-type agreement is at least 90 percent.

### Stop conditions

- Stop if invalid JSON or invalid bucket IDs occur after three attempts.
- Stop if GLM cannot start after one corrective change.
- Stop before student work if Claude primary-bucket agreement is less than 70 percent.

### Reproducibility

- Base commit before GLM integration: `4a0abf3c894c87e6fbc9f68fcf41d9852f0ce38c`.
- GLM source branch tip: `075379e6bc3866ea1c14f285ad7e58dba26e0dda`.
- GLM model revision: `ba978f7d347eaf65d22f1a86833408afdb953541`.
- Evaluation manifest: `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/manifest.json`.
- Seed: 42.

### Pilot run 001

- Job `/rav/lux-glm52-semantic-1000-b200-002` reached the GLM endpoint.
- The model cache hit, and all eight ranks loaded the pinned FP8 model.
- The first description batch failed because one JSON response was incomplete after three attempts.
- The child server stopped through the parent cleanup path.
- The corrective change doubles the output-token limit on each JSON retry.

### Pilot run 002

- Job `/rav/lux-glm52-semantic-1000-b200-003` wrote all 1,000 descriptions.
- GLM created 38 buckets, including the required fallback bucket.
- The job wrote 150 assignment checkpoints.
- One response omitted the optional rationale field on three attempts.
- Primary and secondary bucket IDs remained valid.
- The corrective change accepts an empty rationale while retaining all label gates.

### Pilot completion and blinded review

- Job `/rav/lux-glm52-semantic-1000-b200-004` completed all 1,000 descriptions and assignments.
- All 38 buckets received documents. The largest bucket held 13.5 percent, and the five largest held 44.9 percent.
- The effective bucket count was 24.53. `OTHER_UNCLEAR` held 0.3 percent.
- Mean GLM confidence was 0.93751. Confidence is a diagnostic, not an accuracy measure.
- The final resumed attempt took 1,729 seconds. This is not a clean full-run speed measurement.
- Claude classified 20 documents with the frozen vocabulary and without source metadata or GLM labels.
- Exact primary agreement was 50 percent. This fails the 70 percent stop gate for student work.
- Nineteen of 20 GLM and Claude label sets shared at least one bucket.
- Exact language and document-type string gates were invalid. The prompts allowed names, codes, and free-text types.
- Two inspected disagreements showed valid competing primary labels. The main fault is overlapping bucket scope and missing primary-label precedence.
- Do not train the student on these single primary labels. Revise the target as multi-label or hierarchical labels first.
- Report: `.agents/projects/luxical-arctic-poc/glm-semantic-label-report.md`.

### GLM pilot peer review

- The Claude sample was a low-confidence stress sample from 20 of 38 buckets. It was not a representative sample.
- Therefore, its 50 percent primary agreement cannot test the registered 70 percent population gate.
- The review found that long documents use a 6,000-character start, middle, and end view. The stored sample cannot recover the truncation fraction.
- The 30-bucket minimum can cause overlap. Test smaller vocabularies before adding only precedence rules.
- The Claude CLI default model was not recorded. Pin it in the next review.
- Accepted code fixes add reproducible concentration measures and complete secondary-label validation.
- Accepted code fixes also restore a cached taxonomy before candidate work and remove raw review packages from durable job logs.
- The direct task-output stream remains because the local Claude client cannot read the private object store.
- Retry-loop and checkpoint-loop refactors remain separate cleanup. They do not change the stored pilot result.

## 2026-08-02: Semantic embedding screen

### Background research brief

#### Question

Can the saved 3M Fast Transformer form coherent semantic neighborhoods while it keeps its measured CPU speed?

#### Internal evidence

- The 3M Arctic student reached 23,228 documents per second on CPU. Luxical reached 8,958 documents per second.
- The student improved the old source probe over Luxical. Its only old blocker was source concentration.
- Source concentration is not a valid target for this task. One source can contain many unrelated document types.
- Qwen3-Embedding-0.6B gave the best saved code and multilingual probe result among the tested teachers.
- A 750K cross-dimension Qwen student lost variance and quality. It is a useful negative control.
- GLM-5.2 assigned 1,000 fixed documents to 38 semantic buckets. The labels contain useful multi-label overlap.
- The 38-bucket taxonomy has unclear primary-label precedence. It is not sufficient for a production claim.

#### External evidence

- [Qwen3 Embedding](https://arxiv.org/abs/2506.05176) targets multilingual text, code, retrieval, and flexible output dimensions.
- [Gecko](https://arxiv.org/abs/2403.20327) uses LLM-generated pairs and relabeled hard negatives to train a compact embedding model.
- [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368) creates diverse synthetic matching tasks and training triplets.
- [VICReg](https://arxiv.org/abs/2105.04906) adds variance and covariance terms that prevent representation collapse.
- [VCReg](https://arxiv.org/abs/2306.13292) reports better transfer when supervised training controls variance and covariance.

#### Hypothesis

The saved 3M Arctic student keeps useful semantic geometry even though it failed source-concentration gates.

#### Experiment

- Align saved vectors to the 1,000 fixed documents by document hash.
- Do not use source metadata in any quality metric.
- Compare Luxical, the 3M student, the failed Qwen student, Arctic, Qwen, and LFM.
- Measure label agreement within ten nearest neighbors, primary-label nearest-neighbor F1, and cluster NMI.
- Measure finite outputs, unique outputs, effective rank, variance, and pairwise cosine spread.
- Keep a private gallery with representative queries and nearest neighbors for direct inspection.

#### Falsifier

Reject the current student if any semantic metric is more than 0.02 below Qwen on this screen.
Also reject it for non-finite vectors, repeated vectors, low effective rank, or less than 0.8 times Luxical CPU speed.

#### Interpretation limit

A pass only permits the next evaluation stage. It does not establish production quality.
The next stage needs a smaller hierarchy, representative independent review, held-out labels, and a new paired speed test.

### Screen run 001

- Job `/rav/lux-semantic-embedding-screen-b200-001` failed before metric calculation.
- The loader used document hashes as unique row keys. One source contains duplicate document hashes.
- Evaluation rank is the stable unique row key. The correction aligns by evaluation rank and checks the document hash.
- A regression test covers duplicate hashes with different evaluation ranks.

### Cross-source screen

- Job `/rav/lux-semantic-embedding-screen-b200-002` succeeded in 3 minutes 3 seconds.
- The screen excludes all neighbors from the query source. Source identity is not a prediction target.
- The 3M student passed all eight screening gates against the best tested teacher value for each metric.
- Its cross-source label overlap was 0.63020. Arctic reached 0.62770, LFM reached 0.60130, and Qwen reached 0.56330.
- Its cross-source label Jaccard was 0.31825. Arctic reached 0.31236, LFM reached 0.30300, and Qwen reached 0.27284.
- Its cross-source nearest-label macro-F1 was 0.22061. LFM reached 0.22008, Arctic reached 0.21863, and Qwen reached 0.19091.
- Its cluster NMI was 0.47628. LFM reached 0.46923, Arctic reached 0.45611, and Qwen reached 0.44816.
- Its effective-rank fraction was 0.33326. Arctic reached 0.61674, Luxical reached 0.41543, and Qwen reached 0.28118.
- All models returned finite and unique vectors. The student's total normalized variance was 0.83994.
- The inherited paired CPU speed ratio remains 2.59315 against Luxical-One.
- Direct inspection found coherent recipe, literature, code, and molecule neighborhoods.
- It also found weak government-statistics and technical-support neighborhoods. The 38 primary buckets remain too broad and overlapping.
- Artifact: `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/embedding-screen-v2/report.json`.
- This result advances the student to hierarchical-label evaluation. It does not establish production quality.

### Exact trained-artifact CPU speed

- Job `/rav/lux-trained-fast-student-cpu-speed-b200-002` succeeded in 1 minute 55.96 seconds with zero failures and zero preemptions.
- The paired test used 20,000 fixed evaluation documents, five alternating repeats, eight CPU threads, and the CPU JAX backend.
- The exact 3M student reached a median 18,492.61 documents per second. Pinned Luxical-One reached 6,157.05 documents per second.
- The exact student-to-Luxical ratio is 3.00349. It passes the 0.85 target ratio with wide margin.
- The tested model SHA-256 is `8735a4b49de0f7925904b0301516a2c8a5f9651bc2b605e4d27a80bca3f8ac3a`. The tokenizer-map SHA-256 is `50c92752d5a1d408234b8eee58c1c0f6179f603253caabe4fcd3a06f990710f0`.
- Artifact: `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/speed/cpu-trained-full-full-3m.json`.

## 2026-08-02: Hierarchical semantic labels

### Hypothesis

A smaller domain hierarchy will reduce valid primary-label disagreements while it keeps useful semantic detail.

### Method

- Reuse the fixed 1,000 document views and their saved GLM descriptions.
- Create compact and balanced domain hierarchies with 8 through 12 and 12 through 16 non-fallback parents.
- Keep document form as a separate controlled label. Do not mix form with semantic domain.
- Assign every document from its raw source-blind view.
- Give both frozen taxonomies and shared representative documents to a pinned Claude model.

### Gates

- Both hierarchies pass all parent, leaf, fallback, and precedence validation.
- Each hierarchy assigns all 1,000 documents with valid IDs.
- The fallback parent contains at most 5 percent of documents.
- The largest parent contains at most 30 percent of documents.
- At least 80 percent of non-fallback parents and leaves receive documents.
- Claude exact primary-parent agreement is at least 80 percent on 100 representative documents.
- Claude any-parent overlap is at least 90 percent on the same documents.
- Claude exact document-form agreement is at least 85 percent.

### Stop conditions

- Stop one variant after three invalid taxonomy responses.
- Stop assignment after the same schema error repeats after one correction.
- Do not use a hierarchy for production evaluation if Claude primary-parent agreement is below 70 percent.

### Placement correction

- The first CPU coordinator had no private object-store identity and failed before GLM startup.
- A federated one-B200 coordinator reached storage, but the eight-B200 server remained capacity-blocked.
- The coordinator can also leave a four-GPU domain partially occupied.
- A B200 availability constraint without an attached GPU could not federate and created no job.
- The label client now runs inside the GLM head task. This keeps private data and HTTP traffic within the server job.
- Nested GPU jobs do not federate from a CPU-only coordinator. The coordinator still needs a federated B200 placement.
- The first callback run timed out after the default five-minute client wait while the server was still queued.
- The corrected coordinator wait is unbounded inside the two-hour parent timeout.
- The stopped attempts wrote no taxonomy or assignments.
- A scheduler query found broad accelerator saturation. It also confirmed that the nested server could not federate after its one-accelerator parent selected a site.
- The direct server entrypoint submits the two-replica gang through federation. It uses explicit task ports and runs the bounded hierarchy client on rank zero.
- Focused direct-server, lifecycle, hierarchy, review, and metric tests pass.
- The old parent-child run stopped before model startup. Direct job `/rav/lux-glm52-hierarchy-direct-b200-001` entered the federation queue at interactive priority without an idle coordinator accelerator.
- A four-device fit smoke reached model construction with 64 GiB of host memory, then the container was OOM-killed at the first streamed weight load. The tensor and expert-parallel shape is valid, but 64 GiB is not enough host memory.
- A 96 GiB four-device retry reserved accelerator quota but could not find a node with the required host memory. It was stopped because its reservation could prevent the primary eight-device gang from admission.
- The direct eight-device hierarchy run remains the only active accelerator request for this experiment.
- `production-student-report.md` defines the remaining label, 10,000-document, blind-neighborhood, robustness, release, and peer-review gates. The exact CPU speed gate is already complete.

### Production evaluation preparation

- The accepted hierarchy will label exactly 10,000 new held-out documents. Selection starts from a nested 11,000-document source-balanced sample and removes every 1,000-document hierarchy-pilot row by source and evaluation rank.
- The held-out embedding report uses the exact trained-artifact speed result. It gates parent, leaf, and form metrics against the best saved teacher.
- Large label groups need at least 30 documents. Each such group can trail its best teacher by at most 0.03 cross-source nearest-label F1.
- Pairwise geometry uses at most 1,000,000 deterministic document pairs. Exact neighbor search is computed once for each embedding and reused across label levels.
- A 200-query blind review compares the student with the strongest saved teacher. Model names, GLM labels, and source metadata are hidden. Set order is deterministic and randomized for each query.
- Claude will identify the visible query language and whether code is central. Code and non-English subgroup results therefore use document content instead of source names.
- Twenty-six focused hierarchy, server, label, metric, and review tests pass.

### Pre-results peer review

- Rejected the branch-revert blocker. Current `origin/main` at `23d17c62f7fedb20e7f5f8fb21d05689f39711c7` is the exact merge base and an ancestor of this branch. The reviewer compared with an older base.
- Accepted the need to state vector normalization. The production semantic path normalizes every vector before neighbor, pair, rank, variance, and cluster metrics. Older source-provenance metrics are not production gates.
- Accepted the Arctic window-overlap caveat. Exact repeated windows do not change an average. Overlapping medium-document windows can give the middle more weight. Independent labels and blind review gate the effect.
- Accepted the large-rung memory concern. Do not run 10M or 30M with the current materialized loader. Add and test a bounded sharded or streaming loader first.
- The source-inventory provenance limit was already recorded. The unused results collector was already deleted. Source-name categories are not used in the production semantic gates.
- The remaining survey, manifest-helper, old Arctic wrapper, and report-compression findings are outside the production evaluation path. They do not change saved artifacts or current trust gates.

### GLM-5.2 H100 fallback check

- The pinned FP8 checkpoint index reports 755,617,140,416 bytes of model
  weights across 141 shards.
- Eight 80 GB H100 devices provide only 640 GB before runtime memory. They
  cannot hold this checkpoint.
- The model has 64 attention heads and 256 routed experts. The next practical
  tensor-parallel size after eight is 16, not 12.
- A 16-H100 gang is not a smaller placement request than the active
  eight-B200 gang. Keep the B200 request queued unless the scheduler rejects
  it or a different model-serving plan is proven first.

### Bounded fast-student training input

- The optional 10M and 30M ladder now has a disk-backed input layout. It
  writes source-interleaved NumPy memory maps instead of retaining all IDs,
  teacher vectors, and source IDs in host memory.
- Staging interleaves 4,096-row source chunks. Each epoch shuffles 65,536-row
  blocks and shuffles rows inside each block. This keeps contrastive batches
  mixed across sources without random reads across the full dataset.
- The loader rejects a source quota above 262,144 rows. During training, it
  keeps at most one source slice or one epoch block active as array data.
- The original materialized layout remains available and keeps the exact old
  global permutation sequence.
- A canary records measured peak RSS, full batch coverage, source coverage,
  a batch digest, and the calculated memory limits. The live 3M canary is the
  remaining gate before a larger rung is permitted.
- Forty-three focused training, hierarchy, label, metric, and review tests
  pass.
- The first full 3M canary scanned every batch and source and exited cleanly,
  but peak RSS reached 7,777,198,080 bytes. The process retained clean mapped
  pages after staging. This passes the provisional 8 GiB gate but does not
  prove that a 30M run is bounded.
- The corrected loader flushes and releases mapped pages with
  `MADV_DONTNEED` after each staged source and each epoch block. A second 3M
  canary reduced peak RSS to 1,623,203,840 bytes. This is a 79.1 percent
  reduction from the first canary.
- The corrected canary scanned all 3,000,000 rows and saw all 146 sources. All
  finite-value, shape, batch-coverage, source-coverage, layout, and 8 GiB peak
  RSS gates passed.
- Corrected canary artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/loader-canary/prepared-3m-20260802-002.json`.
- The bounded-input gate now passes. A larger rung is permitted only when the
  3M student fails a semantic gate and the required prepared rows and teacher
  vectors exist.

### GLM hierarchy run

- The first direct eight-device attempt loaded the pinned GLM-5.2-FP8 model
  and reached final CUDA-graph autotuning.
- The scheduler preempted the attempt to admit another workload. Iris then
  rescheduled both replicas as one unit. This was not a model, data, or schema
  failure, and the attempt produced no hierarchy labels.
- The replacement attempt started at 10:45 UTC and remains active.
- The replacement worker resolved the prior attempt's still-leased Ray
  endpoint. Its process retried the old head address while the new head waited
  for the missing four devices. The stable endpoint name made automatic
  preemption recovery unsafe.
- The GLM launcher now adds the Iris attempt number to the internal Ray
  endpoint name. Two focused port and retry tests pass.
- The stuck job was stopped. Corrected job
  `/rav/lux-glm52-hierarchy-direct-b200-002` started at interactive priority.
  Direct process inspection confirmed that its worker uses its current head
  address and that both replicas joined the same Ray cluster.
- The corrected job loaded GLM, completed all warmups, and built a valid compact
  taxonomy with 12 parents, 23 leaves, and 13 precedence rules.
- It saved 13 compact assignment checkpoints, or 650 documents, before one
  document returned a primary leaf under the wrong parent on all three
  attempts. This was a model-output validation failure. Saved checkpoints are
  valid and reusable.
- The old retry changed only the seed. The assignment client now appends the
  exact validation error and invalid JSON to the next request. Nine focused
  GLM and hierarchy tests pass.
- Resume job `/rav/lux-glm52-hierarchy-direct-b200-003` was submitted at
  interactive priority. It will restore the taxonomy and first 650 compact
  labels.
- The resume passed the prior invalid leaf-to-parent row and completed all
  1,000 compact labels. The compact pilot has 1.5 percent fallback labels, a
  25 percent largest parent, and full use of all 12 parents and 23 leaves.
- One compact precedence rule names `FORMS_TEMPLATES`, which is not in the
  taxonomy. The saved assignments use only valid IDs, but this taxonomy is not
  eligible for the frozen evaluation until the rule is corrected.
- Hierarchy validation now rejects unknown IDs in precedence rules. A failed
  hierarchy request also gives its exact validation error and JSON to the next
  model attempt. Eight focused hierarchy tests pass.
- The balanced hierarchy request is still active. Its first outputs reached
  the generation limit and did not produce a saved taxonomy.
- The final balanced output had 37 non-fallback parents instead of the allowed
  12 through 16. This was a hierarchy-generation validation failure, not a
  server or data failure.
- Corrected job `/rav/lux-glm52-hierarchy-direct-b200-004` runs only the
  balanced variant. Both four-device replicas were admitted at interactive
  priority. The compact checkpoints remain unchanged.
- The corrective feedback path ran through all three checked requests, but GLM
  returned the same 37-parent hierarchy each time. No balanced taxonomy was
  saved.
- The hierarchy prompt now requests exactly 14 non-fallback parents and 34
  non-fallback leaves. It limits definitions to 20 words, include and exclude
  arrays to two short values, and precedence rules to 12. This removes the
  ambiguous ranges and the long response that caused repeated JSON retries.
- Nine focused hierarchy tests pass. Job
  `/rav/lux-glm52-hierarchy-direct-b200-005` was submitted at interactive
  priority with the exact-size prompt.
- The exact-size prompt produced 15 total parents and 34 leaves in 5,253
  generation tokens. It removed the repeated output-length and 37-parent
  failures.
- The balanced taxonomy is rejected. It mapped code to technical forums, legal
  text to public notices, news to non-fiction books, and API manuals to
  tutorials. It also placed government, health, forms, and events below the
  fallback parent.
- Hierarchy validation now permits only the fallback leaf below the fallback
  parent. Ten focused hierarchy tests pass.
- The balanced run labeled all 1,000 documents and wrote its summary. The head
  task succeeded, but the worker task reported expected Ray shutdown as a
  failure. The worker launcher now treats head shutdown as authoritative. Four
  focused direct-launch tests pass.
- The compact taxonomy was curated into run
  `hierarchy-1000-20260802-002`. The only change removes the invalid
  `FORMS_TEMPLATES` precedence rule. Document form is already a separate label,
  and the domain prompt forbids form-based domains. The original artifact was
  not changed.
- The curated compact pilot retains 1.5 percent fallback use, a 25 percent
  largest parent, and full use of all 12 parents and 23 leaves.
- Claude Opus 5 reviewed 100 representative documents and 50 lowest-confidence
  documents. The representative sample has 81 percent exact parent agreement,
  98 percent any-parent overlap, 76 percent exact leaf agreement, 96 percent
  any-leaf overlap, and 85 percent exact form agreement. It passes the fixed
  parent, overlap, and form gates.
- The lowest-confidence sample has 38 percent exact parent agreement, 74
  percent any-parent overlap, and 58 percent exact form agreement. This is a
  boundary-label failure. Adjudicate the low-confidence tail of the held-out
  set before using it for final embedding metrics.
- The accepted Claude review cost $7.4280115. Its full report SHA-256 is
  `5de28b7ce065cfc44cc65e013949f92968f1b3196e740d93c11e15acf29d6751`.
  Report artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/claude-review-v1/report.json`.
- Held-out job `/rav/lux-glm52-heldout-10k-b200-001` was submitted with the
  curated compact taxonomy for exactly 10,000 new documents.
- The held-out review now fixes the adjudication population before results are
  visible. It selects the lowest-confidence 5 percent, which matches the failed
  50-of-1,000 pilot stress sample. The source-blind exporter has four behavior
  tests. The embedding evaluator can replace only these rows with checked
  Claude labels. It keeps separate raw and adjudicated artifacts.
- The raw and adjudicated student metrics must differ by at most 0.02, and the
  gate decision must not change. This checks that boundary-label noise does not
  control the student decision.

### Held-out hierarchical embedding result

- The fixed GLM hierarchy labeled all 10,000 new held-out documents. The raw
  embedding evaluation job completed without a failure or preemption.
- The 3M Fast Transformer remains finite and non-constant. Its finite fraction
  is 1.0, its four-decimal unique fraction is 0.9997, its effective-rank
  fraction is 0.35219, and its total variance is 0.84226.
- The 3M student fails the fixed semantic release gates. Parent and form
  nearest-primary macro-F1 fail. Leaf cluster NMI fails. Large-group F1 also
  fails at all three hierarchy levels.
- The parent macro-F1 is 0.44498. The best saved teacher value is 0.47415.
  The form macro-F1 is 0.37281. The best saved teacher value is 0.41156.
  The leaf cluster NMI is 0.40660. The best saved teacher value is 0.42961.
- The failures include large code groups. `SOFTWARE_CODE` trails its best
  teacher by 0.04395 F1, and the `CODE` form trails by 0.04100. The result is
  not a code-collapse failure because the student remains diverse, but it is a
  measurable semantic-quality loss.
- This raw result starts the larger student rung. The low-confidence 5 percent
  label adjudication and label-sensitivity check remain active. They cannot
  release the 3M student unless the fixed gate decisions stay stable.
- Raw report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/raw-v1/embedding-screen-v1/report.json`.

### LUX-ARCTIC-30M-001: Final pure-scaling rung

- Hypothesis: A nested 30M rung can recover the semantic quality that the 3M
  and 10M students lost.
- Commit Hash: `31fd0c72736a78861c0dd51719747d447e427e70`.
- The 10M blind review had 84 wins, 10 ties, and 106 losses. Its score was
  0.4450 with a 95 percent interval from 0.3775 through 0.5125.
- The 10M code score was 0.45. The non-English score was 0.32258. The other
  text score was 0.47479. All three scores failed their fixed gates.
- The 3M-to-10M score change was -0.0025. Stop pure scaling when the
  10M-to-30M score change is less than 0.005.
- The 10M student CPU speed was 6,798.62 documents per second. Luxical-One
  speed was 655.95 documents per second in the same forced-CPU job.
- The 30M manifest has 30,000,000 training rows and 74,752 evaluation rows
  from 146 sources. Its SHA-256 is
  `19fe07f483b27d26cbf6402a3ee97c7f90953487378bbf5d37401d0819c5dbf4`.
- The exact audit confirmed that all 10M rows are unchanged and nested in the
  30M rung. Audit artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/expanded-30m/manifest-audit.json`.
- The teacher path reuses the exact 10M vectors. It embeds 20M new rows instead
  of 27M new rows.
- Command:
  `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --gpu H100 --cpu 16 --memory 80GB --disk 200GB --priority interactive --max-retries 1 --timeout 21600 --user rav --job-name lux-arctic-teacher-30m-h100-00 --extra gpu --extra datakit -- python .agents/projects/luxical-arctic-poc/extend_arctic_teacher.py --rung 30m --shard-index 0 --num-shards 32`.
- Result: All 32 independent teacher jobs entered Iris. The first source on
  shard 0 has 169,527 new rows after 10M prefix reuse.
- Interpretation: The input and prefix checks pass. Teacher vector generation
  is active.
- Next action: Audit all 30,074,752 teacher vectors. Then prepare and train the
  30M student.

### LUX-CAPACITY-001: Large student control

- Hypothesis: More student capacity can recover semantic quality at the 3M
  rung.
- Commit Hash: `27e8ad750`.
- Training command:
  `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --gpu H100 --cpu 16 --memory 80GB --disk 200GB --priority interactive --max-retries 1 --timeout 10800 --user rav --job-name lux-fast-student-large-3m-h100-002 --extra gpu --extra datakit -- python .agents/projects/luxical-arctic-poc/train_fast_student.py --rung 3m --config large --treatment baseline --teacher arctic-medium-256 --training-layout staged`.
- The first launch stopped before training. The training CLI did not accept the
  `large` config that the model code supplied.
- A regression test reproduced the fault. The corrected CLI and two focused
  tests passed.
- The corrected job succeeded in 2 minutes 3 seconds. The model has 28,432,896
  parameters.
- The final model SHA-256 is
  `faa28194a890e0e50326fba28e99f1924a07e15cfc66a1164983f3e75db46e56`.
- All audit vectors are finite and unique at six decimal places. The effective
  rank is 48.08.
- CPU command:
  `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --gpu H100 --cpu 16 --memory 80GB --disk 200GB --priority interactive --max-retries 1 --timeout 7200 --user rav --job-name lux-fast-student-large-3m-cpu-speed-h100-001 --extra cpu --extra datakit -- env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python .agents/projects/luxical-arctic-poc/benchmark_trained_fast_student.py --config large --teacher large --rung 3m`.
- The student rate is 3,529.46 documents per second. The paired Luxical-One
  rate is 4,470.38 documents per second.
- The speed ratio is 0.78952. It fails the fixed 0.85 CPU release gate.
- Speed artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/speed/cpu-trained-large-large-3m.json`.
- Interpretation: The large model cannot be the production model without a
  speed change. Its semantic result can show whether capacity limits the small
  student.
- Next action: Run the fixed adjudicated semantic evaluation. Keep the 30M
  teacher jobs active.

### LUX-CAPACITY-002: Large student semantic result

- Command:
  `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --gpu H100 --cpu 16 --memory 80GB --disk 200GB --priority interactive --max-retries 1 --timeout 7200 --user rav --job-name lux-hierarchy-eval-adjudicated-large-3m-h100-001 --extra gpu --extra datakit -- python .agents/projects/luxical-arctic-poc/evaluate_hierarchical_embeddings.py --run-id hierarchy-1000-20260802-002 --variants compact --evaluation-run-id heldout-10000-20260802-001 --student-model fast_arctic_large_3m --student-config large --student-training-name large --student-rung 3m --speed-report-url s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/speed/cpu-trained-large-large-3m.json --adjudication-review-url s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/claude-adjudication-v1/report.json`.
- Job `/rav/lux-hierarchy-eval-adjudicated-large-3m-h100-001` succeeded in 4
  minutes 35 seconds.
- Parent cross-source macro-F1 is 0.43780. Leaf cross-source macro-F1 is
  0.36117. Form cross-source macro-F1 is 0.39163.
- Leaf cluster NMI is 0.42561. The larger model improves some fine-label
  structure but does not pass all semantic gates.
- Large-group gates fail for six parent groups, ten leaf groups, and five form
  groups.
- The failed groups include intellectual property, medical text, narrative,
  technical documents, and structured data. Code does not fail the large-group
  gate.
- The CPU gate also fails because the paired speed ratio is 0.78952.
- Report artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/student-fast_arctic_large_3m/adjudicated-v1/embedding-screen-v1/report.json`.
- Interpretation: More capacity alone does not fix the broad semantic loss.
  Do not spend Claude review cost on this failed control.
- Next action: Complete the fixed 30M rung. If it fails, change the teacher
  windows or training objective.

### Background research brief: teacher and student input mismatch

- Effort: Low.
- Stop rule: Stop after the local code and primary distillation sources give
  one falsifiable test.
- Question: Does the teacher target contain text that the fast student cannot
  read?
- Current code: Arctic reads three windows. Each window has as many as 512
  tokens. The student reads one 256-token row from 768 selected characters.
- Current result: The 28.4M-parameter control fails broad semantic gates. Thus,
  more capacity does not correct the loss.
- Luxical describes the distillation input as the same `X` for the teacher and
  student. See the [Luxical training notes](https://github.com/datologyai/luxical#training).
- DistilCSE reports that objective consistency improves sentence-embedding
  distillation. See [Gao et al.](https://arxiv.org/abs/2112.05638).
- No Echo result described this input mismatch.
- Hypothesis: The student cannot match a target that includes approximately six
  times more tokens. Long documents lose the most information.
- Minimum test: Run Arctic on the exact fast student character view. Compare
  its fixed hierarchical metrics with full-window Arctic.
- Falsifier: Reject the hypothesis when the fast-view teacher does not improve
  the student-target information match, or when its own semantic quality fails.
- Cost: One 10,000-document teacher pass and the existing fixed evaluation.
- Confidence: Exploratory. The token-count difference is direct code evidence,
  but no matched-view result exists yet.
- Next action: Run the fast-view Arctic diagnostic while the 30M teacher jobs
  remain active.

### LUX-INPUT-001: Arctic fast-view diagnostic

- Commit Hash: `a65754d15`.
- Command:
  `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources --gpu H100 --cpu 16 --memory 80GB --disk 200GB --priority interactive --max-retries 1 --timeout 7200 --user rav --job-name lux-arctic-fast-view-eval-h100-001 --extra gpu --extra datakit -- python .agents/projects/luxical-arctic-poc/evaluate_arctic_fast_view.py --run-id hierarchy-1000-20260802-002 --variant compact --evaluation-run-id heldout-10000-20260802-001 --adjudication-review-url s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/claude-adjudication-v1/report.json`.
- The job succeeded in 6 minutes 12 seconds. All 10,000 fast-view vectors were
  finite.
- Parent cross-source macro-F1 changed from 0.46499 to 0.43920. Parent cluster
  NMI changed from 0.38533 to 0.37658.
- Leaf cross-source macro-F1 changed from 0.37307 to 0.35364. Leaf cluster NMI
  changed from 0.42397 to 0.41232.
- Form cross-source macro-F1 stayed at 0.4113. Form cluster NMI changed from
  0.29577 to 0.29961.
- The fast-view teacher fails large-group gates for code, software, research,
  narrative, and technical-document labels.
- Report artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/teacher-diagnostics/arctic-fast-view-v1/report.json`.
- Interpretation: The short teacher target is easier to observe, but its own
  semantics are weaker. Reject it as the sole training target.
- Next action: Keep the full-window teacher. Test a 512-token student because
  the 256-token student has a large CPU speed margin.

### LUX-INPUT-002: 512-token student control

- Commit Hash: `61826a83e`.
- The preparation job wrote exactly 3,000,000 rows from 146 sources. Each row
  has 512 token positions and uses 512 characters from each document region.
- Preparation job: `/rav/lux-fast-student-prepare-context512-3m-h100-001`.
- Prepared manifest:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/prepared-3m-context512/manifest.json`.
- Training job: `/rav/lux-fast-student-context512-3m-h100-001`.
- Training completed 2,199 steps over three epochs. All audited vectors are
  finite and unique. Final effective rank is 46.26582, and total variance is
  0.37537.
- Final model SHA-256:
  `a0fe19c545620f5f49f4e9943c3a39d5e5d33cb8cc3464568d975c7e4bdcc43b`.
- The paired CPU test measured 3,737.09 documents per second for the student
  and 9,254.12 for Luxical. The ratio is 0.40383. It fails the 0.85 speed gate.
- Speed artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/speed/cpu-trained-context512-context512-3m.json`.
- The fixed semantic diagnosis gives parent macro-F1 0.43890, leaf macro-F1
  0.35123, and form macro-F1 0.38207. Parent cluster NMI is 0.39564, leaf NMI
  is 0.40145, and form NMI is 0.28984.
- The original 256-token 3M result had parent macro-F1 0.44498, leaf cluster
  NMI 0.40660, and form macro-F1 0.37281. The longer input improves only form
  F1 by 0.00926. Parent F1 and leaf NMI decrease.
- The longer-input student fails all three semantic release decisions and the
  speed decision. Do not send it to the paid blind review.
- Semantic report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/student-fast_arctic_context512_3m/adjudicated-v1/embedding-screen-v1/report.json`.
- Interpretation: The student input length is not the main quality limit.
  Reject this control and keep the 256-token production shape. Complete the
  fixed 30M rung before a change to the training objective or teacher mix.

### LUX-ARCTIC-30M-002: Training, speed, and visible semantic result

- Commit Hash: `bd1a6719e`.
- Training job: `/rav/lux-fast-student-full-30m-h100-001`.
- Training used 30,000,000 rows from 146 sources for three epochs. It completed
  21,975 updates in 17 minutes 12 seconds.
- Final model SHA-256:
  `981388da726eb2dff8d19dd84fff17749f2b6dd974c93ad223fee581139c9c7f`.
- All training-audit vectors are finite and unique. Final effective rank is
  67.96590, and total variance is 0.42014.
- The paired CPU job measured 6,901.41 documents per second for the student and
  9,225.16 for Luxical-One. The 0.74811 ratio fails the 0.85 speed gate.
- Speed artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/speed/cpu-trained-full-full-30m.json`.
- The fixed 10,000-document evaluation gives parent macro-F1 0.45021, leaf
  macro-F1 0.37371, and form macro-F1 0.40299.
- Parent macro-F1 improves by 0.00318 from the 10M value of 0.44703. It still
  fails the fixed teacher-relative gate.
- Parent, leaf, and form large-group gates all fail. The CPU gate also fails.
- Semantic artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/student-fast_arctic_30m/adjudicated-v1/embedding-screen-v1/report.json`.
- Interpretation: Pure Arctic scaling still does not give a releasable student.
  The fixed 200-query review remains active for the registered scaling stop
  rule.

### Background research brief: relation-only student training

- Effort: Medium.
- Stop rule: Stop when local evidence and primary sources select one cheap test
  that uses saved labels.
- Question: Can the student keep semantic neighborhoods without matching the
  teacher coordinate system?
- Current Marin result: Arctic point alignment improves the student health but
  loses semantic quality. Qwen 1,024-dimensional alignment through a learned
  projection gives 140 source failures at 750K rows.
- Current Marin result: Qwen 1,024-dimensional vectors give the best tested
  Qwen code and multilingual probe values. The 750K saved training labels are
  available.
- [Similarity-Preserving Knowledge Distillation](https://arxiv.org/abs/1907.09682)
  trains a student to keep pair relations without copying teacher coordinates.
- [DistillCSE](https://aclanthology.org/2023.findings-emnlp.547/) reports weak
  standard distillation when teacher contrastive logits have high variance. It
  uses contrastive and multi-teacher controls.
- [Improving Text Embeddings with Large Language Models](https://aclanthology.org/2024.acl-long.642/)
  shows that diverse synthetic pairs and a contrastive loss can train semantic
  embeddings without a point-vector target.
- [TALAS](https://aclanthology.org/2026.acl-long.1509/) reports that strict
  point matching can transmit teacher noise across a large capacity gap.
- Negative result: The current Gram-KL-only FastTransformer collapsed. A new
  relation loss needs a direct vector-spread control.
- Negative result: Same-source geometry loss reduced concentration failures but
  reduced semantic probe quality. Source identity is not a training target.
- Hypothesis: A sharp cross-source neighbor target from Qwen can keep semantic
  pairs without the failed 256-to-1,024 projection.
- Minimum test: Train the 64K and then 750K Qwen rungs. Use four cross-source
  teacher neighbors for each row. Add variance and covariance controls.
- Falsifier: Stop when the 64K model fails vector-health checks, or when the
  750K model does not improve semantic quality over the prior Qwen student.
- Risk: Batch neighbors are approximate. Qwen also has teacher failures. GLM
  semantic pairs remain the next signal when this test fails.
- Confidence: Exploratory. Primary work supports relation-only distillation,
  but no cited paper uses this exact 9.3M-parameter student and corpus.

### LUX-NEIGHBOR-001: Cross-source Qwen neighbor POC

- Hypothesis: A relation-only Qwen objective can remove projection loss and
  keep the FastTransformer vector space non-collapsed.
- Commit Hash: `d8ea6f727`.
- Config: Four Qwen-selected positives for each row, cross-source candidates
  only, temperature 0.1, standard-deviation target 0.04, and covariance weight
  0.1.
- The production shape stays at 9.3M parameters, 256 input tokens, and 256
  output numbers.
- Test result: 23 focused FastTransformer tests pass. The required file checks
  also pass.
- Next action: Run the 64K H100 smoke test. Start the 750K rung only when all
  three epoch audits stay finite, unique, variable, and above rank 2.

### LUX-ARCTIC-30M-003: Hidden review result

- Claude Opus 5 reviewed 200 model-hidden neighborhoods. The reference model
  was Arctic Medium.
- The 30M student had 92 wins, 10 ties, and 98 losses. Its score was 0.4850.
  The paired 95-percent interval was [0.4175, 0.5525].
- The score improved by 0.0400 from the 10M score of 0.4450. This exceeds the
  registered 0.005 scaling threshold.
- The code score was 0.48889 on 45 queries. The non-English score was 0.3750
  on 32 queries. The other-text score was 0.51220 on 123 queries.
- The overall, code, non-English, and other-text interval gates all fail.
- Claude cost was $11.90656. The report SHA-256 is
  `531f28f5ac368831c18bec1941e222c3247dc5ed3658c4fcc59856b7b0bf1ba5`.
- Report artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/student-fast_arctic_30m/adjudicated-v1/blind-neighborhood-review-v1/claude-opus-5-report.json`.
- Interpretation: The 30M rung is not releasable. The hidden score shows that
  Arctic scaling did not saturate at 10M. More scaling remains a fallback, but
  the 30M visible parent, large-group, and CPU gates still fail.

### LUX-NEIGHBOR-002: Qwen neighbor result

- Commit Hash: `d8ea6f727`.
- The 64K smoke job completed all three epochs. Effective rank rose from 3.98
  to 5.47. All vectors were finite and unique.
- The 750K job completed 552 updates in 1 minute 15 seconds. The final model
  SHA-256 is
  `3dfad418733503db251ab4518362cb037cf06b15b55872948c162f816ecda5af`.
- The 750K training audit had effective rank 12.49 and total variance 0.43810.
- The exact student CPU rates were stable from 6,780 through 7,079 documents
  per second. Luxical rates varied from 314 through 4,779 on the same worker.
  The median ratio was 8.60771, but the Luxical variation makes this ratio weak
  evidence.
- On 10,000 held-out documents, the effective-rank fraction was 0.05788. The
  fixed minimum is 0.25.
- Parent macro-F1 was 0.40751. Leaf macro-F1 was 0.30528. Form macro-F1 was
  0.34132. Every global semantic level and every large-group gate failed.
- Semantic artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/student-fast_qwen_neighbor_750k/adjudicated-v1/embedding-screen-v1/report.json`.
- Interpretation: A sharp relation-only teacher loss produces a low-rank space
  and lower semantic quality. Stop this treatment. Do not run a paid review.
- Next action: Keep the 30M Arctic base model. Train a small semantic projection
  on separate GLM hierarchy labels, anchor it to the base geometry, and fold it
  into the existing embedding head.

### LUX-SEMANTIC-PROJECTION-001: Raw GLM projection result

- Commit Hash: `9f095735e`.
- Job: `/rav/lux-semantic-projection-pilot-1k-h100-001`.
- The split dropped exactly 50 low-confidence rows. It used 760 training rows
  and 190 validation rows.
- Parent macro-F1 increased from 0.28632 to 0.32150. Form macro-F1 increased
  from 0.28765 to 0.31939.
- Leaf macro-F1 decreased from 0.20762 to 0.20037. The mean semantic increase
  was 0.01989.
- The effective-rank fraction decreased from 0.31188 to 0.17904. It failed the
  fixed 0.25 gate.
- Total variance was 0.90255. All vectors were finite and unique.
- The minimum folded-head cosine was 0.99983. Thus, head folding kept the
  learned projection without an extra inference operation.
- Report artifact:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/full-glm-semantic-projection/pilot-1k/training.json`.
- Interpretation: The GLM signal improves parent and form semantics, but the
  raw projection compresses too many vector dimensions. Do not use the fixed
  10,000-document set for this failed candidate.
- Next action: Use a fixed identity-mix ladder. Select the best pilot mix that
  passes rank, variance, per-level, and semantic-mean gates.

### LUX-SEMANTIC-PROJECTION-002: Rank-preserving GLM projection result

- Commit Hash: `32db5127c`.
- Pilot job: `/rav/lux-semantic-projection-mix-v2-h100-001`.
- The fixed identity-mix ladder selected alpha 0.6. The validation
  effective-rank fraction was 0.26271, above the 0.25 gate.
- The validation mean semantic gain was 0.01247. Parent and leaf macro-F1
  increased by 0.03535 and 0.00818. Form macro-F1 decreased by 0.00613, within
  the fixed 0.01 limit.
- The minimum folded-head cosine was 0.99999. The final model has 9,299,200
  parameters and no additional inference operation.
- The exact student CPU rates were stable from 6,589 through 6,836 documents
  per second. The median was 6,729.92.
- The Luxical-One rates varied from 367 through 4,735 documents per second.
  Thus, the reported 16.77 paired ratio is not reliable evidence about the
  baseline. The student rate agrees with prior FastTransformer tests.
- On the fixed 10,000-document set, parent macro-F1 increased from 0.45021 to
  0.49406. Leaf macro-F1 increased from 0.37371 to 0.40247. Form macro-F1
  increased from 0.40299 to 0.42825.
- Parent cluster NMI increased from 0.38913 to 0.45331. Leaf cluster NMI
  increased from 0.40530 to 0.46006. Form cluster NMI increased from 0.28708
  to 0.32904.
- Every global semantic and vector-health gate passed. The held-out
  effective-rank fraction was 0.39314, and total variance was 0.87190.
- Twelve large-group gates failed, down from eighteen for the 30M base.
- Parent failures were humanities and culture, and intellectual property.
- Leaf failures were creative narrative, health and medical, humanities and
  social, opinion and commentary, procurement requirements, technical
  documentation, and technical patents.
- Form failures were instruction, unclear text, and structured data.
- Training report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/fast-student/full-glm-semantic-projection/pilot-1k-mix-v2/training.json`.
- Held-out report:
  `s3://marin-us-east-02a/marin/user/rav/luxical-arctic-ladder/manifest-v2/evaluation/semantic-labels/glm-5.2/pilot-1000-20260802-001/hierarchies-v1/hierarchy-1000-20260802-002/compact/heldout-10000-20260802-001/student-fast_glm_projection_pilot_1k_mix_v2/adjudicated-v1/embedding-screen-v1/report.json`.
- Interpretation: GLM supervision corrects the main semantic loss without a
  runtime cost. The 760-row training set is too small for all large groups.
- Next action: Create a separate 50,000-document GLM label set. Train the same
  folded projection method, then run the fixed release gates once.

### Background research brief: semantic fine-tuning fallback

- Effort: Low.
- Stop rule: Stop when one fallback can reuse the 50,000 GLM labels and keep
  the current inference shape.
- Question: Can changes to the full FastTransformer correct subgroup failures
  that a folded output projection cannot correct?
- Current Marin result: The rank-preserving projection improves every global
  semantic metric. It still fails 12 large-group gates.
- Current Marin result: Pure Arctic scaling, a larger student, a longer input,
  and Qwen relation-only training all fail the release gates.
- Repository search found no hierarchy-supervised FastTransformer trainer.
- [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) supports
  the same-label batch objective that the projection already uses.
- [Robust fine-tuning of zero-shot models](https://arxiv.org/abs/2109.01903)
  supports interpolation between the base and fine-tuned weights to keep
  out-of-distribution behavior.
- [Model soups](https://arxiv.org/abs/2203.05482) reports that weight averaging
  can improve fine-tuned models without an inference-time cost. The paper also
  ties this result to models in one low-error basin.
- Caveat: The cited weight-mix results do not prove that this small text encoder
  stays in one low-error basin. The private validation ladder must select the
  mix and reject a low-rank result.
- Hypothesis: Fine-tuning the 30M Arctic checkpoint on parent, leaf, and form
  labels can correct failures inside the encoder. An Arctic-vector anchor and
  a base-to-fine-tuned weight mix can limit geometry loss.
- Minimum experiment: Train three epochs on the same disjoint 50,000 labels.
  Select one of 11 fixed weight mixes on the private 5-percent split.
- Falsifier: Stop this treatment when no mix passes the semantic, rank,
  variance, finite, and unique validation gates.
- Cost: One federated H100 job after the projection release test fails.
- Confidence: Exploratory. The projection result supports the GLM signal, but
  the full-model treatment has not run.

### LUX-SEMANTIC-FINETUNE-001: End-to-end fallback preparation

- Commit Hash: `e8d239438`.
- The trainer starts from the exact 30M Arctic model.
- It drops the least-confident 5 percent of GLM labels.
- It trains parent, leaf, and document-form objectives across different
  sources.
- It anchors each new vector to the original Arctic-student vector.
- It adds the existing vector-spread loss.
- It selects one of 11 base-to-fine-tuned weight mixes on the private split.
- The selected model keeps 9.3 million parameters and adds no inference step.
- Eight focused behavior tests pass in 63.37 seconds.
- Pyrefly reports zero errors. The required pre-commit checks pass.
- Decision: Run this fallback only if the 50,000-label folded projection fails
  a release gate.
