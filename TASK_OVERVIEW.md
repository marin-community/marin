# Unified Image-Text Model Scaling Laws

## Objective

Derive scaling laws for unified multimodal models that jointly process and generate both text and images using discrete visual tokens. The goal is to understand how model performance scales across compute budgets, model sizes, data volumes, data mixture ratios, and visual token loss weights — enabling principled decisions for training large-scale unified vision-language models.

---

## Notation

| Symbol | Meaning |
|---|---|
| C | Base compute budget (FLOPs) |
| C' | Extended compute after adding visual tokens (C' > C) |
| N | Model size (parameters) |
| N* | Compute-optimal model size |
| D | Total data size (tokens) |
| D* | Compute-optimal total data size |
| D*_text | Compute-optimal text data size |
| D_U | Understanding data size (image-text pairs for comprehension) |
| D_G | Generation data size (image generation instruction data) |
| r₁ | Text-to-multimodal data ratio |
| r₂ | Understanding-to-generation ratio within multimodal data |
| L_text | Language modeling loss |
| L_U | Understanding validation loss |
| L_G | Generation validation loss |
| w_text | Loss weight on text tokens within multimodal data |
| w_visual | Loss weight on visual tokens within multimodal data |

---

## Scope

This project investigates scaling behavior in the **10¹⁸–10²⁰ FLOPs** compute regime, systematically varying the following dimensions:

| Dimension | Description |
|---|---|
| **Model Size** | Qwen model series |
| **Total Data Size** | Amount of training data consumed |
| **Data Mixture** | Ratio r₁ (text vs. multimodal) and r₂ (understanding vs. generation within multimodal) |
| **Visual Token Loss Weight** | Ratio w_text / w_visual within multimodal data (w_visual/w_text = 0 → understanding only; w_text/w_visual = 0 → generation only; Emu3 baseline uses w_visual = 0.5) |

---

## Architecture

- **Language Model**: 
 Alternative: Qwen3-series (includes QK-Norm for training stability)
- **Visual Tokenizer**: Two candidates under consideration:

| Property | TokLIP-L (recommended) | TokLIP-XL |
|---|---|---|
| Resolution | 384 × 384 | 512 × 512 |
| Tokens per image | 576 | 1024 |
| Codebook size | 16,384 | 262,144 (260K) |
| Pros | Fewer tokens → more efficient, more data under same compute | More expressive, higher resolution, better reconstruction quality |
| Cons | Less expressive, lower resolution | More tokens → higher computational cost |

The model processes both text and images as discrete token sequences in a unified autoregressive framework. Image pixels are encoded into discrete visual tokens via TokLIP, and the language model is trained to predict both text and visual tokens.

---

## Training Data

### Language-Only Data
- **Nemotron-CC** (high-quality): ~0.9T real tokens
  - Only high-quality real data; synthetic data excluded.

### Image-Text Data
| Dataset | Role | Notes |
|---|---|---|
| LLaVA-OV-1.5 Mid-Training 90M | Image-caption pairs for pre-training | Only ~45B tokens when tokenized with TokLIP-L; insufficient alone |
| UCSC-VLAA/Recap-DataComp-1B | Supplementary image-caption data | Added to compensate for limited caption token count |
| LLaVA-OV-1.5 Instruct + FineVision | Image-text interleaved data; image understanding instruction tuning | |
| BLIP3o / BLIP3o-60k | Image generation instruction data | |

### Key Data Mixture Variables
- r₁: Fraction of text-only vs. multimodal data
- r₂: Fraction of understanding vs. generation within multimodal data
- w_text / w_visual: Loss weight ratio on text vs. visual tokens within multimodal sequences

---

## Data Formatting and Interleaving

This section defines how text and visual tokens are organized into training sequences. The design is inspired by the SODA audio scaling law paper (ICML 2026), which demonstrated that utterance-level interleaving of semantic, acoustic, and text tokens — with dual ordering (audio-first and text-first) — is critical for enabling general-purpose cross-modal capabilities in a unified model.

### Audio → Vision Analogy

| Audio (SODA) | Vision (Ours) |
|---|---|
| Semantic tokens (Mimi codebook 1) | Visual tokens (TokLIP — single codebook fusing semantic and fine-grained detail) |
| Acoustic tokens (Mimi codebook 2–8) | N/A (TokLIP is not hierarchical RVQ) |
| Text tokens (transcript) | Text tokens (caption / question / prompt) |
| Utterance (contiguous speech segment) | Image (one image's visual token sequence) |
| Multi-utterance document | Multi-image interleaved document |
| ASR (audio → text) | Image Understanding (image → text) |
| TTS (text → audio) | Image Generation (text → image) |
| Audio continuation | Image completion / unconditional generation |

### Tokenizer and Special Tokens

We use the **Llama 3 tokenizer** (`meta-llama/Meta-Llama-3.1-8B`, vocab size 128,256) as the base text tokenizer. This choice is driven by practical infrastructure considerations: Nemotron-CC data is already tokenized and cached on GCS with the Llama 3 tokenizer, avoiding re-tokenization. The Qwen3 model architecture is paired with the Llama 3 tokenizer, following the established pattern in the Marin codebase (e.g., `exp1395_qwen3_32b.py`).

Llama 3 has 256 reserved special tokens (`<|reserved_special_token_0|>` through `<|reserved_special_token_255|>`, IDs 128,002–128,255). We repurpose two of these as vision sentinel tokens:

| Token | ID | Role |
|---|---|---|
| `<\|end_of_text\|>` | 128,001 | Document separator / EOS. No BOS is prepended. |
| `<\|vision_start\|>` | 128,004 | Start of visual token sequence (repurposed `<\|reserved_special_token_2\|>`) |
| `<\|vision_end\|>` | 128,005 | End of visual token sequence (repurposed `<\|reserved_special_token_3\|>`) |

For the SFT stage, chat turn delimiters (`<|im_start|>`, `<|im_end|>`) can be similarly repurposed from the remaining reserved tokens.

**Vocabulary extension and token ID shifting**: TokLIP codebook indices start from 0, but they must be shifted to avoid collision with the Llama 3 text vocabulary. We define a shift offset equal to the Llama 3 vocab size (128,256). Each TokLIP codebook index `c` is mapped to token ID `c + 128256` in the unified vocabulary. The extended tokenizer is created once via `create_unified_tokenizer()` in `experiments/unified/unified_pretrain.py` and saved to GCS at `gs://marin-vlm/tokenizers/llama3-unified-144k`.

| TokLIP Variant | Codebook Range (original) | Token ID Range (shifted) | Unified Vocab Size |
|---|---|---|---|
| TokLIP-L | 0 – 16,383 | 128,256 – 144,639 | 144,640 |
| TokLIP-XL | 0 – 262,143 | 128,256 – 390,399 | 390,400 |

Visual tokens are denoted V₀, V₁, ..., V₁₆₃₈₃ (TokLIP-L) or V₀, ..., V₂₆₂₁₄₃ (TokLIP-XL), where Vᵢ corresponds to unified token ID `i + 128256`.

**Embedding expansion**: The model's token embedding layer and output (lm_head) projection are sized to the unified vocab size (e.g., 144,640 for TokLIP-L). Levanter determines the model's vocab size from `len(tokenizer)` at model construction time (`train_lm.py:156`). When using cold-start training (from scratch), this is trivially handled by initializing the full vocabulary at model creation time. When using warm-start (from a pre-trained checkpoint), the original text embeddings are preserved and only the appended visual token embeddings are freshly initialized.

### Token Composition Types

Analogous to the SODA paper's three compositions (S-only, S+A, S+A+T), we define three. Note: unlike audio (which supports pure audio continuation), we do not include a visual-only configuration — images are always paired with text in our setting.

**(a) Visual + Text, image-first (understanding-oriented)**
```
<|vision_start|> V₁ V₂ ... V₅₇₆ <|vision_end|> T₁ T₂ ... Tₙ <|endoftext|>
```
Image first, then text. Analogous to SODA's audio-first format (c.2). Natural for understanding tasks: see the image, then predict the description/answer.

**(b) Visual + Text, text-first (generation-oriented)**
```
T₁ T₂ ... Tₙ <|vision_start|> V₁ V₂ ... V₅₇₆ <|vision_end|> <|endoftext|>
```
Text first, then image. Analogous to SODA's text-first format (c.1). Natural for generation tasks: read the prompt, then generate the image.

**(c) Interleaved multi-image**
```
T₁...Tₖ <|vision_start|> V₁...V₅₇₆ <|vision_end|> Tₖ₊₁...Tₘ <|vision_start|> V₁...V₅₇₆ <|vision_end|> Tₘ₊₁... <|endoftext|>
```
Natural document order with images and text interleaved at image boundaries. Analogous to SODA's multi-utterance interleaving.

### Per-Dataset Format Specification

#### Pre-training Stage

| Dataset | Format | Sequence Structure |
|---|---|---|
| Nemotron-CC (text-only) | Text | `T₁ T₂ ... Tₙ <\|endoftext\|>` |
| LLaVA-OV Mid-Training + Recap-DataComp-1B (caption) | Dual-ordering | Image-first: `<\|vision_start\|> V... <\|vision_end\|> caption <\|endoftext\|>` AND Text-first: `caption <\|vision_start\|> V... <\|vision_end\|> <\|endoftext\|>` |
| LLaVA-OV Instruct + FineVision (interleaved) | Interleaved | `T... <\|vision_start\|> V... <\|vision_end\|> T... <\|vision_start\|> V... <\|vision_end\|> T... <\|endoftext\|>` |

#### SFT Stage

**Understanding Instruct (LLaVA-OV Instruct + FineVision)**
```
<|im_start|>user
<|vision_start|> V₁...V₅₇₆ <|vision_end|> What animal is shown?<|im_end|>
<|im_start|>assistant
A golden retriever.<|im_end|><|endoftext|>
```

**Generation Instruct (BLIP3o / BLIP3o-60k)**
```
<|im_start|>user
Generate an image of a sunset over the ocean.<|im_end|>
<|im_start|>assistant
<|vision_start|> V₁...V₅₇₆ <|vision_end|><|im_end|><|endoftext|>
```

### Loss Weight Configuration

For each training sequence, the loss on multimodal data is computed as: L = w_text × L_text_tokens + w_visual × L_visual_tokens.

| Data Type | w_text | w_visual | Rationale |
|---|---|---|---|
| Text-only | 1.0 | — | No visual tokens present |
| Caption dual-ordering (image-first) | 1.0 | w | Understanding-oriented: text is primary prediction target |
| Caption dual-ordering (text-first) | w | 1.0 | Generation-oriented: visual tokens are primary prediction target |
| Understanding instruct | 1.0 | w | Answer text tokens are primary target |
| Generation instruct | w | 1.0 | Visual tokens are primary target |
| Interleaved multi-image | 1.0 | w | Both modalities, with configurable visual weight |

Here w is a tunable hyperparameter (Emu3 uses w = 0.5). This connects directly to RQ4 (Visual Token Loss Weight Scaling). For dual-ordered caption data, the image-first and text-first samples naturally emphasize different modalities via their loss weight configurations.

### Token Composition Ablation (Suggested Early Experiment)

Analogous to SODA's Table 1 ablation (S-only vs. S+A vs. S+A+T), we recommend an early small-scale ablation:

| Config | Format | Expected Validation |
|---|---|---|
| V+T (image-first only) | Only image-first ordering | Understanding-biased: can it generate? |
| V+T (text-first only) | Only text-first ordering | Generation-biased: can it understand? |
| V+T (dual-ordering) | Both orderings | General-purpose: full U + G capability |

SODA found that S+A+T unlocked cross-modal capabilities (ASR/TTS) with minimal degradation on speech metrics. We expect V+T dual-ordering to similarly unlock cross-modal image-text capabilities. The key question is whether dual-ordering provides meaningful gains over single-ordering, or whether a single ordering suffices when combined with dedicated U and G instruction data.

### Key Differences from SODA Audio Setting

1. **No hierarchical codebook**: SODA uses Mimi's 8-layer RVQ (semantic → acoustic) which requires flattening. TokLIP has a single codebook, so no flattening is needed, but there is also no explicit semantic/fine-grained separation.
2. **Token information density**: SODA audio runs at 100 tokens/sec (~25× text token rate). Vision uses 576 tokens/image (TokLIP-L), where each image carries substantially more information than a single audio utterance. This may affect optimal interleaving granularity and data mixture ratios.
3. **Expected text ratio difference**: SODA found 5% text + 95% speech optimal. Vision's optimal ratio is likely very different due to differences in visual token information density and the role of text in visual understanding. This is precisely what RQ2 investigates.

---

## Evaluation

### Text Benchmarks
| Benchmark | Capability |
|---|---|
| HellaSwag | Commonsense reasoning (sentence completion) |
| WinoGrande | Coreference resolution |
| ARC | Science question answering |
| MMLU | Broad academic knowledge |

### Image Understanding Benchmarks
| Benchmark | Capability |
|---|---|
| VQAv2 | General visual question answering |
| TextVQA | OCR-based visual question answering |
| GQA | Compositional visual reasoning |
| ChartQA | Chart understanding |
| AI2D | Diagram understanding |
| MMMU | Multidiscipline multimodal understanding (may be too hard for small models) |

### Image Generation Benchmarks
| Dataset | Metrics |
|---|---|
| ImageNet | FID, CLIP Score |
| COCO-30K | FID, CLIP Score |

---

## Infrastructure Milestones

These milestones establish the technical foundation before running scaling experiments.

### Milestone 1: LLM Scaling Law Foundations
Study existing scaling law methodology (Chinchilla, Hoffmann et al., etc.) to establish the theoretical and practical baseline.

### Milestone 2: Evaluation Infrastructure
Extend the **Marin** evaluation framework to support all text benchmarks (HellaSwag, WinoGrande, ARC, MMLU), image understanding benchmarks (VQAv2, TextVQA, GQA, ChartQA, AI2D, MMMU), and image generation benchmarks (FID and CLIP Score on ImageNet / COCO-30K).

### Milestone 3: Dataset Tokenization
Tokenize the full multimodal training set into discrete tokens: text tokenization (Llama 3 tokenizer, extended with TokLIP visual tokens), image tokenization via TokLIP (384×384 → 576 tokens or 512×512 → 1024 tokens per image). Store in Levanter cache format suitable for efficient data loading with variable mixture ratios. Implementation: `experiments/unified/vlm_tokenize_captions.py`.

### Milestone 4: Stable Multimodal Training
Build a robust training pipeline for joint text–visual token prediction. Incorporate stability techniques from prior work (e.g., Chameleon: z-loss, QK-Norm). Handle variable-length sequences mixing text and visual tokens. Support configurable visual token loss weight. Validate training stability across model sizes (30M → 3.5B).

---

## Research Questions

### RQ1: Does Multimodal Training Affect Language Capability?

**Scientific Question**: Does multimodal training (adding visual data) harm the model's pure language capability? Does a "Multimodal Tax" exist?

**Hypotheses**:
- **H1 (Multimodal Tax)**: Multimodal training has inherent language capability loss, as model capacity is occupied by visual processing.
- **H2 (Grounding Benefit)**: Moderate multimodal training actually enhances language understanding through visual grounding.
- **H3 (Scale Dependent)**: Small models have tax, large models have no tax or even benefit.

**Experimental Design**:

Fix the overall text tokens, additionally add visual tokens.

- **Step 1**: Find the compute-optimal configuration for text-only under each compute budget C. This gives N*, D*_text.
- **Step 2**: Fix D_text = D*_text, add visual tokens. Set up four configurations:

| Config | Training Data | Compute |
|---|---|---|
| A (Text-only baseline) | N*, D*_text | C |
| B (+ Understanding) | Fix D*_text, add D_U | C' = 6 × N × (D*_text + D_U) > C |
| C (+ Generation) | Fix D*_text, add D_G | C' > C |
| D (+ Both) | Fix D*_text, add D_U + D_G | C' > C |

- **Step 3**: Scale up C' and observe the trends of L_text, L_G, and L_U.

**Core Comparisons**: With the same text tokens, does adding visual data cause tax? How does tax scale with text compute or overall compute?

---

### RQ2: Data Mixture Scaling Law

**Scientific Question**: In unified multimodal models, what is the optimal data mixture strategy? How should we balance text vs. multimodal data, and within multimodal data, how should we balance Understanding vs. Generation?

**Experimental Design**:

**Level 1 — Text vs. Multimodal Ratio (r₁)**:
For each compute budget C, fix U:G = 1:1 within multimodal data. Vary text:multimodal ratio r₁ ∈ {9:1, 7:3, 5:5, 3:7, 1:9}. Find optimal r₁* that minimizes combined loss (or maximizes combined benchmark).

**Level 2 — Understanding vs. Generation Ratio (r₂)**:
Using optimal r₁* from Level 1, vary U:G ratio r₂ ∈ {9:1, 7:3, 5:5, 3:7, 1:9} within the multimodal portion. Find optimal r₂* at each compute budget.

---

### RQ3: Do Understanding and Generation Truly Have Mutual Benefits?

**Scientific Question**: Does joint training of Understanding and Generation produce true mutual benefits?

**Hypotheses**:
- **H1 (Mutual Benefits)**: U and G learn shared visual knowledge; joint training is more efficient than independent training.
- **H2 (No Benefits)**: Mutual benefits do not exist; U and G neither help nor interfere with each other significantly.
- **H3 (Interference Dominant)**: In most cases, tasks interfere with each other; joint training is suboptimal.
- **H4 (Conditional Benefits)**: Mutual benefits only exist under specific conditions (e.g., specific data mixture or scale). Benefits may only appear after scaling up beyond a certain threshold.

**Experimental Design**:

Three model configurations for comparison:

| Config | Training Objective |
|---|---|
| A (Unified) | U + G + text simultaneously |
| B (U-only) | Understanding + text only |
| C (G-only) | Generation + text only |

For each compute budget C, first find the optimal configuration (N*, D*), then train all three configs under this optimal setting to ensure fair comparison.

**Mutual Benefit Quantification**: Define MB_U = U_unified − U_only_baseline and MB_G = G_unified − G_only_baseline. If MB_U > 0 and MB_G > 0, true mutual benefits exist. If MB_U < 0 or MB_G < 0, interference exists.

---

### RQ4: Visual Token Loss Weight Scaling

**Scientific Question**: Within multimodal data, how do loss weights for text tokens vs. visual tokens interact with model scale? Can we achieve better U/G trade-offs by adjusting supervision strength rather than data mixture?

**Hypotheses**:
- **H1 (Weight-Scale Interaction)**: Optimal w_text / w_visual ratio within multimodal data is scale-dependent. Small models may need different weighting than large models.
- **H2 (Interference Mitigation)**: Properly tuned loss weights can reduce U-G interference by balancing the gradient contributions from text and visual tokens.

**Experimental Design**:

Within previously explored optimal settings, define the multimodal loss as: L = w_text × L_text_tokens + w_visual × L_visual_tokens.

The extremes are: w_text/w_visual = 0 → optimize for generation only; w_visual/w_text = 0 → optimize for understanding only. Sweep across intermediate ratios at multiple compute budgets to characterize the optimal weighting as a function of scale.

---

### RQ5: Correlation Between Benchmark and Validation Loss

**Scientific Question**: In unified multimodal models, are Understanding/Generation benchmark scores strongly correlated with corresponding validation loss? Is loss a reliable training signal for scaling law fitting?

**Hypotheses**:
- **H1 (Strong Correlation)**: Loss and benchmark are strongly correlated; loss is a reliable proxy for downstream performance.
- **H2 (Task-Dependent)**: Understanding's loss–benchmark correlation is strong; Generation's correlation is weak (because FID and similar metrics don't directly correspond to likelihood).
- **H3 (Scale-Dependent)**: Small-model correlation is weak; large-model correlation is strong.

**Experimental Design**:

For each converged training run, record: L_U (understanding validation loss), L_G (generation validation loss), U_benchmarks (VQA, MMMU, etc.), G_benchmarks (FID, CLIP Score, GenEval). Fit the loss–benchmark relationship using parametric functions. If strong correlation holds, subsequent scaling law experiments can rely on loss alone, making them far cheaper.

---

### RQ6: The Role of Interleaved Data

**Scientific Question**: Does interleaved image-text data exhibit different scaling behavior compared to separate U/G data? How does performance scale with the amount of interleaved data?

**Hypotheses**:
- **H1**: Interleaved data has better scaling efficiency than separate data, with steeper performance improvement per FLOP.
- **H2**: Interleaved data has similar scaling behavior to separate data, but with a constant offset (higher baseline performance).
- **H3**: Interleaved data only shows scaling advantages for specific tasks (multi-image reasoning, story generation), not for general U/G tasks.

**Experimental Design**:

Data Configurations:

| Config | Data Composition |
|---|---|
| A (Pure Separate) | Only (image, caption) pairs |
| B (Mixed 10%) | Separate + 10% interleaved data |
| C (Mixed 30%) | Separate + 30% interleaved data |
| D (Mixed 50%) | Separate + 50% interleaved data |

For each configuration, train models at multiple compute budgets (1e18, 3e18, 1e19, 3e19, 1e20 FLOPs) and fit scaling curves.

---

### RQ7: Multimodal Emergent Abilities Analysis

**Scientific Question**: How do different multimodal capabilities emerge as data/compute scales up? Do Understanding and Generation abilities have different emergence patterns? Are there predictable thresholds for specific capabilities?

**Experimental Design**:

For each compute budget, train the compute-optimal model configuration and collect evaluation scores on all capability benchmarks. Analyze the emergence pattern for each benchmark — identifying whether capabilities emerge gradually (smooth scaling) or abruptly (phase transition) and whether Understanding and Generation capabilities follow different emergence trajectories.

---

## Key References

- Chinchilla scaling laws (Hoffmann et al.)
- Chameleon (Meta): training stability for unified multimodal models
- Emu3: visual token loss weighting
- SODA (ICML 2026): scaling laws for discrete audio models with interleaved semantic, acoustic, and text tokens — primary inspiration for our data formatting and interleaving design
- TokLIP: discrete visual tokenizer
- LLaVA-OV / FineVision: multimodal instruction data
- BLIP3o: image generation instruction data
- Nemotron-CC: high-quality text corpus
- Recap-DataComp-1B: supplementary image-caption data

---

## Summary

This project aims to answer a series of interconnected questions about unified image-text models: whether multimodal training taxes language capability (RQ1), what the optimal data mixture is and how it scales (RQ2), whether understanding and generation truly benefit each other (RQ3), how visual token loss weights interact with scale (RQ4), whether validation loss reliably predicts downstream benchmarks (RQ5), what role interleaved data plays (RQ6), and how capabilities emerge across scale (RQ7). The deliverable is a comprehensive set of scaling laws that enable principled, compute-efficient training of future large-scale multimodal models.

---

## Edit Log

### 2025-02-11: Tokenizer switch from Qwen3 to Llama 3

**Motivation**: Nemotron-CC text data is already tokenized and cached on GCS with the Llama 3 tokenizer (`meta-llama/Meta-Llama-3.1-8B`). Using the Qwen3 tokenizer would require re-tokenizing the entire ~0.9T token Nemotron corpus, which is unnecessary since the Marin codebase already supports pairing Qwen3 model architecture with Llama 3 tokenized data (established pattern in `exp1395_qwen3_32b.py`).

**Changes to TASK_OVERVIEW.md**:

1. **Section "Tokenizer and Special Tokens"**: Replaced Qwen3 tokenizer (vocab 151,936) with Llama 3 tokenizer (vocab 128,256) as the base text tokenizer. Updated all special token IDs:

   | Token | Old ID (Qwen3) | New ID (Llama 3) |
   |---|---|---|
   | `<\|endoftext\|>` / `<\|end_of_text\|>` | 151,643 | 128,001 |
   | `<\|vision_start\|>` | 151,652 | 128,004 (repurposed `reserved_special_token_2`) |
   | `<\|vision_end\|>` | 151,653 | 128,005 (repurposed `reserved_special_token_3`) |

   Removed Qwen3-specific tokens (`<|im_start|>`, `<|im_end|>`, `<|image_pad|>`) from the main table since they are not native to Llama 3. Added note that SFT chat delimiters can be repurposed from remaining Llama 3 reserved tokens.

2. **Vocabulary extension table**: Updated shift offset from 151,936 to 128,256, and unified vocab sizes:

   | TokLIP Variant | Old Unified Vocab | New Unified Vocab |
   |---|---|---|
   | TokLIP-L | 168,320 | 144,640 |
   | TokLIP-XL | 414,080 | 390,400 |

3. **Embedding expansion**: Updated to reference Levanter's `len(tokenizer)` mechanism for determining model vocab size, removing the specific number 151,936.

4. **Milestone 3**: Updated from "Qwen3 tokenizer" to "Llama 3 tokenizer, extended with TokLIP visual tokens". Added reference to implementation file `experiments/unified/vlm_tokenize_captions.py`.

**Corresponding code changes**:

- **`experiments/unified/vlm_tokenize_captions.py`**: Updated constants (`VISUAL_TOKEN_OFFSET` 151,936→128,256; `VISION_START_ID` 151,652→128,004; `VISION_END_ID` 151,653→128,005; `ENDOFTEXT_ID` 151,643→128,001). Changed default tokenizer from `Qwen/Qwen3-0.6B` to the extended Llama 3 tokenizer on GCS.
- **`experiments/unified/unified_pretrain.py`** (new file): Training experiment script implementing `create_unified_tokenizer()` (extends Llama 3 with vision sentinels + 16,384 TokLIP visual tokens), `unified_data_config()` (mixes Nemotron text caches + multimodal Levanter cache), and training configs for Qwen3-0.6B/1.7B/4B model sizes.
