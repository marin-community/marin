# Audio Experiments

Scripts for training discrete audio language models using the Marin framework.

## Training Runs (`exp1699_*.py`)

- `exp1699` refers to the issue created for audio experiments: https://github.com/marin-community/marin/issues/1699
- Qwen3x = Qwen3 extended with `<|begin_of_text|>` and `<|end_of_text|>` (like Llama) to utlize existiing processed data

| Exp# | Script | Description |
|------|--------|-------------|
| Exp1.1 | `exp1699_marin_yodas2.py` | First-Experiment: Train 600M model on 500B Yodas2 tokens (en, es, fr, de, th, hi, ar, zh) |
| Exp1.3 | `exp1699_marin_yodas2_anneal.py` | Annealing training for Exp1.1 with different data mixes for the last 50K steps |
| Exp1.4 | `exp1699_nemotron_sweep.py` | Study the optimal ratio of text-only data (Nemotron) vs speech (YODAS2) at 150M size |
| Exp1.7 | `exp1699_ablate_tokens.py` | Ablation study on token types (semantic-only, semantic+acoustic, semantic+acoustic+text) |
| Exp2.1 | `exp1699_marin_audio_all.py` | Train all model sizes (4B, 1.7B, 600M, 135M) from scratch for 500B tokens |
| Exp2.2 | `exp1699_marin_audio_ws.py` | Train 1.7B and 600M models with warm-start from Qwen3x |
|------|--------|-------------|
| ? | `exp1699_marin_audio_135m_anneal_nemo.py` | Annealing 135M model with different data mixes |
| ? | `exp1699_data_mix_yodas_sweep_150m.py` | Study Yodas data mixtures at 150M size |
| ? | `exp1699_data_mix_150m.py` | Study data mixtures at 150M size |


## IsoFLOP Study (`isoflop_*.py`)

| Exp# | Script | Description |
|------|--------|-------------|
| Exp1.6 | `isoflop_audio_sweep.py` | Generate sweep of model configurations at fixed FLOP budgets |
| Exp1.6 | `isoflop_audio_target.py` | Train specific (budget, model size) targets for curve fitting |

## Fine-Tuning Experiments

| Script | Description |
|--------|-------------|
| `audio_sft_cvss.py` | Voice-preserving speech-to-speech translation fine-tuning using CVSS dataset |

## Data Preparation (`tokenize_*.py`)

| Script | Description |
|--------|-------------|
| `tokenize_yodas.py` | Tokenize YODAS2 dataset (multilingual speech-text) |
| `tokenize_emilia.py` | Tokenize Emilia dataset (speech-text pairs) |
| `tokenize_mls_en.py` | Tokenize Multilingual LibriSpeech (English) |
| `tokenize_nemotron.py` | Tokenize Nemotron-CC (text-only data) |
| `tokenize_finetune.py` | Tokenize fine-tuning datasets (LibriSpeech, CommonVoice, etc.) |
| `tokenize_sft_cvss.py` | Tokenize CVSS instruction dataset for S2ST fine-tuning |
| `tokenize_qwen3x_audio.py` | Tokenize datasets with Qwen3 tokenizer |

## Supporting Modules

| Module | Description |
|--------|-------------|
| `audio_defaults.py` | Audio-specific training defaults and annealing helpers |
| `data_mixes.py` | Data mixture configurations (mix2, mix3, cooldown, etc.) |
| `download_weights.py` | Download warm-start checkpoints |
| `models.py` | Model definitions (Qwen3x base checkpoints) |
| `qwen3.py` | Small Qwen3 model configs (30M–150M) |
| `convert_initial_checkpoints.py` | Convert HuggingFace checkpoints for training |
