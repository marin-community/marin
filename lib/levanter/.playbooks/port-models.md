---
title: Porting a Model to Levanter
description: A guide for porting model architectures to the Levanter ecosystem using Haliax and Equinox.
---

# Playbook: Porting a Model to Levanter

This guide outlines the steps to port a model architecture (e.g. from Hugging Face or another JAX framework) to
the Levanter ecosystem using Haliax and Equinox. This guide is focused on LLMs, but the principles apply to other models as well.
Agents can also refer to [Port-Models.md](../docs/dev/Port-Models.md), which contains a more detailed, human-oriented guide.

In general, pattern match on `llama.py` or `mixtral.py` for examples of how to implement a model.

## Step 1: Write the Config
- Define all model hyperparameters and axis definitions using a `@dataclass`.
- Axes should be properties that reference the appropriate hyperparameter sizes.
- If applicable, inherit from `HFCompatConfig` and implement `from_hf_config()` / `to_hf_config()`.
- Register your config with `LmConfig.register_subclass("your_model_name")`.

## Step 2: Implement the Model
- Build components like MLP, attention, and blocks using `equinox.Module` and `haliax.NamedArray`.
- Use `Stacked` to represent repeated layers.
- Organize into modules like `Embedding`, `Block`, `Transformer`, and `LMHead`.
- Each module should have an `init()` static method.

## Step 3: Add Serialization Support (if needed)
- For identical field names, nothing is required.
- For remapping keys (e.g., `blocks` → `h`), override `_state_dict_key_map()`.
- For nontrivial formats, override `to_state_dict()` and `from_state_dict()`.
- Refer to `ModuleWithStateDictSerialization`.

## Step 4: Write Tests

### Unit Tests
- Test individual components like `Embedding`, `Block`, and `Transformer` and compare to their Hugging Face counterparts. Generally speaking, it is better to test each module.
- Use `chex.assert_trees_all_close` with tolerances for cross-framework comparisons. Generally use 1e-4 for complex modules and 1e-5 for simpler ones. Never relax past these tolerances without discussion.
- Create helpers like `_get_random_inputs()` for reuse. See `test_llama.py` for examples.

### Serialization Tests

- If porting a Hugging Face model, ensure serialization works with `to_state_dict()` and `from_state_dict()`. Write a round-trip test similar to what's in `test_llama.py`.
- Use `to_torch_compatible_state_dict` and load into Hugging Face modules.
- Compare Levanter and HF model outputs on the same input.

## Step 5: Create a "nano" config
- Create a `yaml` training config under `configs/`. Pattern match on llama2_nano.yaml
- Probably you cannot run

## Tips
- Avoid positional tensors: always use named `Axis`.
- Use `rearrange`, `dot`, and `einsum` for shape and contraction logic.
- Use gradient checkpointing policies similar to what's in `llama.py`.
- Haliax means you generally don't need to worry about sharding. For more complex things you may. See `mixtral.py` for an example of an MoE.

## Example Models
- `GPT2Config`, `Gpt2Transformer` in `levanter.models.gpt2`
- `LlamaConfig`, `LlamaTransformer` in `levanter.models.llama`

Llama is probably the most modern and well-structured example, so it is a good reference point.

---

If you find any friction points or helpful patterns, update this playbook and/or Port-Models.md!
