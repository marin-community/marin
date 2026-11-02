"""A tiny Qwen2 policy and GSM8K-format dataset for end-to-end CPU training runs.

The tokenizer is byte-level with ChatML special tokens and a single ``####`` token, so the policy's vocabulary
has a few hundred entries; a production-sized vocabulary makes the output projection dominate every CPU forward
and backward pass. The model is a hand-shaped bigram policy. Zeroed attention and MLP output projections make each
position's hidden state a function of its current token only, and the embedding and LM head encode
a transition table over five answer tokens. The policy emits ``#### 1`` (the GSM8K reward format)
with moderate probability, so GRPO groups have reward variance and RL can raise the success rate.
Training updates every parameter, so the model is free to leave the bigram form.
"""

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

END_OF_TURN = "<|im_end|>"
# ChatML, as Qwen's template renders conversations without tools or a default system prompt.
CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)
GROUND_TRUTH = "1"
# Dataset bins for curriculum sampling, in grade order.
CURRICULUM_BINS = ("g0-even", "g1-odd")
HIDDEN_SIZE = 16
DISALLOWED_LOGIT = -8.0
ALLOWED_LOGIT_OFFSET = 8.0

# Answer tokens in transition-table order.
ANSWER_TOKENS = ("####", " ", "1", "2")
# Row: the current token's state (0 = any other token, then ANSWER_TOKENS). Column: the next token,
# ANSWER_TOKENS followed by end of turn.
TRANSITIONS = (
    (0.6, 0.1, 0.1, 0.1, 0.1),
    (0.1, 0.6, 0.1, 0.1, 0.1),
    (0.1, 0.1, 0.4, 0.3, 0.1),
    (0.1, 0.1, 0.1, 0.1, 0.6),
    (0.2, 0.1, 0.1, 0.1, 0.5),
)


def build_tiny_policy(output_dir: Path) -> Path:
    """Write the tiny policy and its tokenizer as a Hugging Face model directory."""
    tokenizer = _tiny_tokenizer()
    answer_ids = [_single_token_id(tokenizer, text) for text in ANSWER_TOKENS]
    next_ids = [*answer_ids, tokenizer.convert_tokens_to_ids(END_OF_TURN)]

    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=HIDDEN_SIZE,
        intermediate_size=2 * HIDDEN_SIZE,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids(END_OF_TURN),
        torch_dtype="float32",
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.o_proj.weight.zero_()
            layer.mlp.down_proj.weight.zero_()
        embedding = model.model.embed_tokens.weight
        embedding.zero_()
        embedding[:, 0] = 1.0
        for state, token_id in enumerate(answer_ids, start=1):
            embedding[token_id] = 0.0
            embedding[token_id, state] = 1.0
        # RMSNorm scales a one-hot state vector to sqrt(hidden_size).
        state_scale = HIDDEN_SIZE**0.5
        lm_head = model.lm_head.weight
        lm_head.normal_(std=0.02)
        lm_head[:, : len(TRANSITIONS)] = DISALLOWED_LOGIT / state_scale
        transitions = torch.tensor(TRANSITIONS)
        for column, token_id in enumerate(next_ids):
            logits = transitions[:, column].log() + ALLOWED_LOGIT_OFFSET
            lm_head[token_id, : len(TRANSITIONS)] = logits / state_scale

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def write_gsm8k_dataset(path: Path, num_prompts: int, *, max_turns: int) -> Path:
    """Write ``num_prompts`` GSM8K-environment rows whose ground truth is always ``1``.

    Rows alternate between the two ``CURRICULUM_BINS``, so curriculum sampling can read the dataset. With
    ``max_turns`` above one, the rows use the multi-turn GSM8K environment, which asks again after each wrong
    answer until the turns run out.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for index in range(num_prompts):
            grade = index % len(CURRICULUM_BINS)
            row = {
                "prompt": [{"role": "user", "content": f"Question {index}: what is one? End with '#### 1'."}],
                "env_class": "gsm8k" if max_turns == 1 else "gsm8k_multi_turn",
                "reward_spec": {"method": "rule", "ground_truth": GROUND_TRUTH},
                "extra_info": {"data_source": CURRICULUM_BINS[grade], "grade": grade, "max_turns": max_turns},
            }
            handle.write(json.dumps(row) + "\n")
    return path


def _single_token_id(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{text!r} is not a single token: {token_ids}")
    return token_ids[0]


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    """A byte-level tokenizer with ChatML's special tokens and the GSM8K answer marker as one token."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE(vocab={char: index for index, char in enumerate(alphabet)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token=END_OF_TURN,
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"],
    )
    tokenizer.add_tokens([ANSWER_TOKENS[0]])
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer
