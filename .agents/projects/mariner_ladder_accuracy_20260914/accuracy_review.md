# MARINER ladder accuracy: focused pre-submission review

Reviewed 2026-09-14 UTC. Scope: the planned full accuracy overlap evaluation of the two completed 1e21 checkpoints, Proportional and UniMax-8. No training, submission, model-weight download, or GCS data transfer was performed during this review.

## Verdict

The planned evaluation is suitable to launch after the offline checkpoint-converter correction described below. That correction is present in `evaluate_mariner_ladder_accuracy.py`: `checkpoint_model_config` explicitly constructs a Qwen3 converter with the row checkpoint as both the reference and tokenizer, reconstructs the archived architecture, and preserves those two fields. No remaining source-level blocking issue was found. The worker's regional offline task-loading gate runs before model initialization; its actual execution, then the first inference batches, remain live checks.

## Checkpoints and runtime

`launch_inventory.json` identifies final step-22056 HF exports for Proportional and UniMax-8. The four MARINER/Olmix rows are ineligible because their final exports are absent. This review did not infer completion from an intermediate native checkpoint or executor status.

Using the archived config already saved in the inventory, current Transformers 5.12.1 and Levanter reconstruct the intended Qwen3 geometry: 26 layers, hidden size 2560, intermediate size 10240, 20 query/KV heads, head dimension 128, untied embeddings, no attention bias or sliding window, and context length 4096. The Llama3 rotary configuration retains theta 500000, factor 8, low/high frequency factors 1/4 and original context 8192. Transformers emits a range warning because original context 8192 exceeds the active 4096 limit, but the reconstructed configuration preserves all those values. No weights were loaded for this check.

Current local versions checked: Transformers 5.12.1, Datasets 4.8.5, JAX 0.11.1, lm-eval 0.4.9.1. The launcher freezes runtime versions, lm-eval's VCS revision, source hashes and uv.lock. BF16 parameters/compute, 4096 positions and eight packed sequences per evaluation batch are consistent with the requested v6e-8 configuration. Distributed initialization precedes HF model conversion, and the second trainer initialization disables duplicate distributed initialization.

## Offline converter issue, corrected

The original `HFCheckpointConverter.from_hf` path discovers the model by constructing default converters. Qwen3's default converter eagerly resolves the Qwen/Qwen3-0.6B tokenizer, which need not exist in the fresh offline worker cache. A second such lookup occurred in the standalone harness after `from_hf_config` reset the reference/tokenizer to defaults. The explicit checkpoint-bound converter and preserved config references avoid both lookups. The implementation agent reports that its empty-HF-cache fixture passed under strict offline flags, including the initial converter and the standalone harness's second converter construction; no default Qwen cache was created. This reviewer checked the correction in the source.

The initially documented root command `uv run --extra lm_eval` was also invalid for the root workspace package. The worker now documents the workspace-qualified extra command, and Fray requests its TPU/lm_eval dependency groups.

## Tasks and complete split sizes

The installed lm-eval task definitions were cross-checked against the saved regional `dataset_config_inventory.json`. The expected suite contains 67 leaves and 44,248 evaluated documents:

| Family | Leaves | Evaluation documents |
| --- | ---: | ---: |
| MMLU | 57 | 14,042 |
| ARC-Easy | 1 | 2,376 |
| ARC-Challenge | 1 | 1,172 |
| CommonsenseQA | 1 | 1,221 |
| HellaSwag | 1 | 10,042 |
| WinoGrande | 1 | 1,267 |
| SocialIQA | 1 | 1,954 |
| PIQA | 1 | 1,838 |
| SciQ | 1 | 1,000 |
| LAMBADA OpenAI | 1 | 5,153 |
| MedMCQA | 1 | 4,183 |
| **Total** | **67** | **44,248** |

The MMLU aliases are `mmlu_<subject>_5shot`; the other ten aliases match `OLMO_BASE_EASY_OVERLAP_TASKS`, including `lambada_0shot`. Every task requests five-shot evaluation except zero-shot LAMBADA; no chat template is used. The final-result validation checks the resolved few-shot configuration of every leaf.

This is the complete eleven-family accuracy overlap configured in Marin, not the full 51-head OlmoBaseEval Easy BPB objective. It excludes several code, generative QA and Minerva math tasks. Retain the automatically returned macro/micro aggregates, but do not label the equal-leaf macro as the full suite's accuracy: 57 of its 67 leaves are MMLU subjects. Per-family results permit a subsequent justified reporting choice.

## Offline cache compatibility

Datasets 4.8.5's offline loader explicitly falls back to cached Arrow datasets, so removal of dataset-script execution does not by itself prevent SocialIQA evaluation. A local offline load probe successfully loaded all MMLU subjects and ARC, CSQA, HellaSwag, WinoGrande, SocialIQA and PIQA. It was deliberately stopped when SciQ was missing from the local cache; the regional inventory contains SciQ. No online download was attempted.

The local SocialIQA cache is a 0.0.0 conversion, while the saved regional cache is the older script-built 0.1.0 Arrow directory. Therefore the local probe does not certify that exact regional cache. The worker loads all tasks from the regional cache and requires the complete leaf set before opening model weights. Its cache manifest requires complete HF Hub/modules metadata and no declared failed datasets. First-child logs should verify that this regional gate succeeds and that the loaded total is 44,248 documents.

## Metrics, samples and durable completion

`validate_results` requires finite accuracy in [0,1] for every leaf, matching original/effective split counts, one unique logged `doc_id` per evaluated document, per-request Levanter outputs, and the expected few-shot setting. It rejects a completed run with missing leaves or samples. `log_samples=True` and `SampleLoggingConfig(log_all=True)` preserve all returned documents, responses and per-request details.

The worker stores the full result object compressed, a metric summary, and provenance containing the full plan, checkpoint identity, task counts, runtime and topology. It reads each artifact back to verify its size/SHA-256 before writing SUCCESS, and resumptions revalidate both identity and payload completeness. This preserves the information needed to choose metrics later rather than retaining only selected headline scores.

These source and metadata checks do not establish successful TPU weight loading or inference. The first child remains the test of memory use, archived-weight compatibility and actual benchmark execution.
