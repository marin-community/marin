# Campaign notepad

2026-10-10 03:00 UTC: Read the full overnight brief. Both allowed H100 clusters returned healthy current controllers. MarinSkyRL main is c96f6d25 and Marin main is c4687932. Earlier Snowball evidence at 40 updates and consumed ages 0–2 does not qualify the new port or settle severe staleness.

The released reference has no dependency lock. Freeze a resolved Python 3.13/CUDA12 JAX environment and record it separately from code. Reference evaluation holds out the first 64 rows after the training-seed shuffle, rather than using the environment's separate eval split. Its terminal step-299 evaluation is before update 300; this timing is now explicit in the protocol.

Initial state at 2026-10-10 03:00 UTC:

| Rung | Status | Evidence needed next |
| --- | --- | --- |
| Authors' reference | Preparing | Frozen environment; bounded real GPU update and retained outputs |
| SkyRL reproduction | Porting | FP32 parity, current-head GPU serving/update qualification |
| Qwen task/recipe transfer | Not started | Qualified reproduction followed by separately frozen changes |
| Snowball controlled mismatch | Not started | New-head five-node qualification and calibration |
| Snowball staleness | Not started | Measured 65-update cadence; frozen substantive main horizon |

The reference TIS and SC pilots each completed two full 64-by-8 updates on eight H100s. SC also exported its HF weights. Both serving captures were nonuniform; the natural top-128 head matched the sampled token probabilities at the exact next-token/prefix positions. These pilots qualify execution, not a learning effect.

Three earlier attempts remain failed engineering attempts: inherited Marin uv constraints broke strict hash installation; the image's old uv could not download a newly pinned Python patch; and the recorder needed explicit JAX output sharding to select two evidence rows. The fixes leave the released model, sampler and loss unchanged. The resolved reference environment is isolated from Marin config and uses the task image's Python 3.13.5.

Initial trajectories differed between runs despite equal seeds. Two additional successful diagnostic pilots established exact initial FP32 weight hash `ab4ac115c005a20165cc303e3b1b5f080e62f6b95d265a27e04f4af1e1439a43` and fixed-delta hash `74713004b950a61a105a6b8ba07b03e78113b08e8c3b0239f50d1e9b37ca1361` in both arms. Equal source, environment, initial weights and noise exclude those as the cause. Distributed BF16 execution and sampling outputs are not bit-identical in these pilots. Retain this observed variability; do not claim identical trajectories or infer a scientific result from two updates.

The current-main core at `117d0598` passed 257 objective checks plus 26 capture-control, replay and mesh checks. The released JAX loss and the production PPO/TIS closure agreed across twelve frozen FP32 cases, including actual +1/-1 group-centered advantages, masks, TIS cap 2 and modeled tails. Maximum gradient error was 2.24e-8. PPO clipping was inactive at p=o. Raw scalar losses have different origins and detached entropy terms; the artifact records both raw losses and their explicit scalar offset.

Countdown membership for seeds 0, 1 and 2 uses the exact original generator and shuffled held-out head. The first export's auxiliary prompt-ID field mistakenly stored BatchEncoding keys. No training used it. Preserve that export, regenerate integer prompt IDs as version two, and audit them against the authors' retained prompts before SkyRL launch. The original prompt text and membership hashes are unchanged.

The eight-run reference main matrix fixes 300 updates: three paired seeds at noise 0.05 plus one no-noise pair. It uses batch priority and does not occupy interactive campaign nodes. No main scientific result is available yet.

## Continuation at 2026-10-10T17:39:52.862506+00:00

| Rung | Current status | Evidence or next gate |
| --- | --- | --- |
| Authors’ reference | Complete | Eight full main runs re-audited; three noisy paired seeds and no-noise sanity pair |
| SkyRL reproduction | Partial | Older pilot audited; current-head paired qualification prepared, main matrix pending |
| Qwen task/recipe transfer | Not started | Reproduction must pass before changing the task and recipe |
| Snowball controlled mismatch | Not started | Five-node batch qualification and separate calibration pending |
| Snowball staleness | Not started | Actual 65-update serving cadence and substantive paired comparisons pending |

Noise0.05 paired mean clean-quality difference remains +46.61 percentage points, with n=3 paired-t 95% interval [-28.02,+121.25] points. SC seed1 remains stalled; the effect estimate is imprecise. Separate seed0 diagnostics give +0.20 points at noise0.01 and +33.59 points at noise0.02. They do not establish precise effects at those noise levels.

The older SkyRL pilot has two applied finite SGD updates with FP32 master weights and gradient buffers, zero momentum, zero decay and zero clipping. Complete stored behavior heads and tokens were audited, as described in the protocol. Exit0 alone was not used to qualify it. The detached native launch explains the missing terminal manifest; retained checkpoint2, resolved launch, raw groups, evaluations and Ray logs provide independent evidence. Needed temporary outputs are being promoted to durable storage.

Current-source CPU checks passed 78 tests after including the production loss module tests, which populate the policy-loss registry. Two isolated config checks had failed earlier because their registry was empty; no production or test tolerance was weakened. The bounded learner evidence and scalar persistence checks passed all 13 replay tests. New paired qualification configs pin published runtime `01944d73c92c493b835b9e3773147e4e72097e8f`.

Current-head paired qualification completed successfully on `01944d73`. Each arm applied two finite full-batch SGD updates with no clipping or skipped updates and checkpoint2. The machine audit checked all 1,024 training responses and all 1,536 evaluation responses per arm, exact reference prompt IDs, actual logical 512-token limits, natural behavior heads, chosen-token probabilities, TIS weights, and clean-evaluation membership and grades. The SC learner sample has nonzero pre-update correction magnitudes (0.03156 and 0.01707 mean absolute token terms). TIS retains the same capture and old/current candidate rescoring while multiplying the correction by zero. These are runtime qualifications, not effect estimates.

The eight SkyRL main configs predeclare 300 updates: noise0.05 paired seeds0,1,2 and one no-noise seed0 pair. Explicit evaluation callbacks give initial weights, every20 completed updates, and the primary 299-update endpoint. Final weights are checkpointed after update300. All main jobs use batch. The sampler model and tokenizer now explicitly name HF revision `c1899de289a04d12100db370d81485cdf75e47ca`, which the current Hub also reports; the trainer was already pinned. Start with one full-horizon job and verify its early real path before admitting the remaining seven.

All eight frozen SkyRL main jobs are running in cw-us-east-02a at batch priority, two nodes each, after the first run produced a finite applied update with the explicit model/tokenizer pin. The source remains01944d73; no main endpoint is available yet. Registry now accounts for33 jobs, including25 finished jobs/31 task attempts. The finished subtotal is54.2556H100 task-allocation hours: reference qualification1.1351, reference main28.4112, reference diagnostics14.9971, SkyRL qualification9.7122. The live matrix is excluded until observed intervals are collected. All old submitted reference arguments match preserved configs exactly; all four old SkyRL resolved launch documents were recovered, confirming each source revision.

The first actual periodic-refresh smoke failed at batch2 on fa4e12ff: the learner's queue-delay assertion still rejected sampler age1. The new source731b85c1 accepts only the exact declared generating version within the current refresh cycle, while retaining zero buffer delay. It passed74 trainer/buffer/context/fixed-noise checks. Retry a1 keeps six updates and refresh period3; this qualifies engineering only.

The first completed SkyRL main run also retained an extra clean evaluation after update300: the every20 interval callback fires at300 even with train-end evaluation disabled. All predeclared evaluations including primary299 are retained; the extra300 evaluation is supplemental and its compute is charged. It does not change the primary endpoint or justify rerunning valid completed training.

Periodic-refresh a1 on731b85c1 completed six nonzero updates and checkpoint6. Retained records match actual sampler versions0 for batches1–3 and3 for batches4–6; ages0/1/2 repeat, with clean current-policy evaluations0..6. This remains engineering evidence, not setting64 science.

The exact-core117d0598 paired runtime checks succeeded on DP8. Source-math reference errors were retained and masked; the accepted batches still had valid natural K128 and finite nonzero gradients. Required read-only review completed without findings through Codex after Claude authentication failed; no login store was changed. Raw reviewer actions and native IDs are retained in this continuation’s native review directory. The core code is unchanged and its draft PR is pending artifact publication.

Before curriculum science, repair the inherited native math adapter rather than excluding legitimate matrices, percentages or time questions. Published experiment158c7f0a passes49 grader tests; new explicit v2 configs supersede unsubmitted pilot-a0 plans and preserve all original source rows. Full reference validation is running on the frozen corrected source. Snowball platform qualification a0 still runs on64ba4b93; a1 will qualify the corrected grading path separately.
