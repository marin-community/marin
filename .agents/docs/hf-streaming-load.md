# HuggingFace Streaming Load Implementation Plan

## Problem Statement

TPU VMs have limited disk space (~100GB). When loading HuggingFace models by ID (e.g., `meta-llama/Llama-2-7b`), the current code downloads the model to a local temp directory or HF cache, causing disk space issues on TPUs.

The solution is to prepend `hf://` to HF model IDs, which enables fsspec-based streaming directly from HuggingFace Hub without hitting the local cache.

## Current Flow

```
Model ID (e.g., "meta-llama/Llama-2-7b")
    ↓
is_hf_checkpoint() returns True (no "://" in string)
    ↓
HFCheckpointConverter.load_state_dict()
    ↓
hf_hub_download() → Downloads to ~/.cache/huggingface/
    ↓
Load from local cache (CAUSES DISK SPACE ISSUES)
```

## Target Flow

```
Model ID (e.g., "meta-llama/Llama-2-7b")
    ↓
Convert to "hf://meta-llama/Llama-2-7b"
    ↓
is_hf_checkpoint() returns True (special case for hf://)
    ↓
HFCheckpointConverter.load_state_dict()
    ↓
_load_from_remote() with fsspec → Stream from HF Hub
    ↓
No local cache used!
```

## Key Files

| File | Purpose |
|------|---------|
| `lib/levanter/src/levanter/compat/hf_checkpoints.py` | Core HFCheckpointConverter with `load_state_dict()` and `_load_from_remote()` |
| `lib/marin/src/marin/rl/model_utils.py` | `is_hf_checkpoint()` and `load_model_from_checkpoint()` |
| `lib/marin/src/marin/rl/rl_job.py` | `make_tokenizer()` entry point |

## Implementation Plan (Spiral Approach)

### Stage 1: Basic `hf://` Streaming for Safetensors

**Goal:** Enable streaming model loading via `hf://` prefix for a minimal case.

1. **Modify `hf_checkpoints.py`**
   - Add a helper function `convert_hf_model_id_to_url(model_id: str) -> str` that:
     - Takes a model ID like `meta-llama/Llama-2-7b`
     - Returns `hf://meta-llama/Llama-2-7b`
   - Update `load_state_dict()` to optionally use `hf://` prefix when loading sharded safetensors
   - Add a parameter `use_hf_streaming: bool = False` to control this behavior

2. **Add minimal test**
   - Create test that loads a small HF model (e.g., `hf-internal-testing/tiny-random-gpt2`)
   - Verify it uses `hf://` streaming and doesn't write to HF cache
   - Test file: `lib/levanter/tests/test_hf_streaming.py`

### Stage 2: Extend to RL Model Loading

**Goal:** Integrate `hf://` streaming into the RL model loading pipeline.

1. **Update `model_utils.py`**
   - Modify `is_hf_checkpoint()` to recognize `hf://` as a valid HF checkpoint prefix
   - Add option to `load_model_from_checkpoint()` to use streaming mode

2. **Update `HFCheckpointConverter.load_state_dict()`**
   - When `use_hf_streaming=True` and model ID is a plain HF repo ID:
     - Convert to `hf://` URL
     - Route through `_load_from_remote()` instead of `hf_hub_download()`

3. **Test RL loading path**
   - Add test in `lib/marin/tests/rl/test_hf_streaming_load.py`
   - Verify streaming mode works for RL training worker model loading

### Stage 3: Environment Variable Control

**Goal:** Allow users to enable streaming via environment variable.

1. **Add environment variable support**
   - `MARIN_HF_USE_STREAMING=1` enables streaming by default
   - This provides backward compatibility while allowing easy opt-in on TPUs

2. **Update documentation**
   - Document the new streaming behavior
   - Add guidance for TPU users

### Stage 4: Handle Edge Cases and Tokenizers

**Goal:** Ensure tokenizers and all model formats work with streaming.

1. **Tokenizer loading**
   - Update `make_tokenizer()` in `rl_job.py` to support `hf://` paths
   - May require using `huggingface_hub.hf_hub_download` with fsspec backend

2. **Handle revision/branch support**
   - Ensure `hf://` URLs can include revision info (e.g., `hf://model@revision`)

3. **Comprehensive test coverage**
   - Test sharded models
   - Test models with different file formats
   - Test tokenizers

## Testing Strategy

### Unit Test: `test_hf_streaming_load.py`

```python
def test_load_small_hf_model_streams_from_hf():
    """Verify that loading an HF model by ID streams from hf:// and doesn't hit local cache."""
    # 1. Clear HF cache for target model
    # 2. Load model using hf:// streaming
    # 3. Verify model loaded successfully
    # 4. Verify HF cache directory is empty (no files downloaded)
```

### Test Models

- `hf-internal-testing/tiny-random-gpt2` - Minimal GPT-2 for fast testing
- `sshleifer/tiny-gpt2` - Another small option

## Rollback Plan

If issues arise:
1. Set `MARIN_HF_USE_STREAMING=0` to disable
2. Revert to using `hf_hub_download()` path

## References

- GitHub Issue: https://github.com/marin-community/marin/issues/2247
- PR #2337: Related tokenizer loading fixes
- fsspec HuggingFace support: https://huggingface.co/docs/huggingface_hub/guides/hf_file_system
