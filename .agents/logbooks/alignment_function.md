# Alignment Function: Research Logbook

**Issue:** https://github.com/marin-community/marin/issues/3355
**Branch:** `alignment_function`
**Experiment ID prefix:** `ALIGN`

## Scope
- **Goal:** Build a marin function that, given a pretrained model, teacher model, and specification/constitution, produces a synthetic preference dataset and trains an aligned model via DPO.
- **Primary metric(s):** Specification adherence score (SpecEval), DPO training loss convergence, preference pair quality (chosen vs rejected margin).
- **Constraints:** Must integrate with existing marin Executor framework, preference data format (sharded `.jsonl.gz`), and Levanter DPO training pipeline.

## Stop Criteria
- A working end-to-end pipeline that takes a specification document and produces a DPO-aligned model.
- Validated on at least one specification set with measurable adherence improvement.
- Pipeline is composable with existing marin ExecutorSteps.

## Key References
- Bloom pipeline (external): prompt generation, inference, preference pair building, marin export
- CS229 project: validated pipeline on 46 OpenAI Model Spec statements, 30k+ preference pairs, +1.7 adherence improvement (DPO beta=0.01)
- SpecEval (arXiv:2509.02464): evaluation framework for behavioral specification adherence
- Existing DPO infra: `lib/levanter/src/levanter/main/train_dpo.py`, `experiments/posttrain/preference_datasets.py`

## Existing Infrastructure (Baseline)
- **DPO training:** Fully implemented in Levanter (`train_dpo.py`), supports frozen reference model, beta tuning, validation splits.
- **Preference data format:** `PreferenceTransformAdapter` converts to OpenAI chat messages with chosen/rejected pairs. Sharded to `.jsonl.gz`.
- **Preference datasets:** UltraFeedback, OLMo-2 registered. Download → transform → tokenize pipeline works.
- **What's missing:** No synthetic preference data generation from specifications. No prompt generation, no teacher model inference for chosen/rejected responses, no specification-to-dataset pipeline.

## Architecture Plan

The alignment function needs these new components, each as an ExecutorStep:

1. **PromptGenerationStep** — Given behavioral statements (specification), generate diverse evaluation prompts.
   - Input: specification document (list of behavioral statements)
   - Output: sharded prompts `.jsonl.gz`
   - Technique: axes of variation -> covering arrays -> concretization -> prompt extraction (Bloom pipeline)

2. **ResponseGenerationStep** — Run teacher model on prompts to produce chosen responses; run weaker/adversarial model for rejected responses.
   - Input: prompts, teacher model endpoint, adversarial model endpoint
   - Output: chosen responses `.jsonl.gz`, rejected responses `.jsonl.gz`

3. **PreferencePairConstructionStep** — Pair chosen/rejected responses into preference format.
   - Input: chosen responses, rejected responses
   - Output: preference dataset in marin format (compatible with existing `PreferenceTransformAdapter`)

4. **Integration step** — Wire into `default_dpo()` or a new `default_alignment()` function.

## Experiment Log

### 2026-03-16 — ALIGN-000: Research Thread Kickoff
- **Action:** Created branch `alignment_function`, research logbook, and initial plan.
- **Findings from codebase exploration:**
  - DPO training is production-ready (PR #2460).
  - Preference data pipeline handles download, transform, tokenization.
  - No synthetic data generation exists — this is the gap.
  - The Bloom pipeline (external repo) has all generation stages; integration work is wiring them into marin ExecutorSteps.
- **Next action:** Design the specification data model and prompt generation step. Investigate how to integrate LLM API calls (teacher model inference) as an ExecutorStep.
