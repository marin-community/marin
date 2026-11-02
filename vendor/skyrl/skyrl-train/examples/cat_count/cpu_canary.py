"""Pretrain the tiny policy fixture used by the real-stack CPU CatCount canary."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from skyrl_gym.envs.cat_count.reward import TARGET_WORD
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

PROMPT = f"Reply with the word {TARGET_WORD} exactly {{N}} times, separated by single spaces. Nothing else."
TEMPLATE = PROMPT.replace(f"word {TARGET_WORD}", "word {W}")
OTHER_WORDS = ["dog", "bird", "fox", "cow", "pig", "owl", "bee", "ant"]
SPECIAL = ["<pad>", "<eos>", "<unk>", "<|user|>", "<|assistant|>"]
PROMPT_WORDS = sorted(set(PROMPT.replace("{N}", "").replace(",", "").replace(".", "").split()))
VOCAB = SPECIAL + sorted(set(PROMPT_WORDS) | set(OTHER_WORDS)) + [",", "."] + [str(d) for d in range(10)]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|user|> {{ m['content'] }} {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|> {% endif %}"
)
HELD_OUT_N = [3, 5, 13, 16]
TRAIN_N = [n for n in range(1, 21) if n not in HELD_OUT_N]
MAX_SEQUENCE_LENGTH = 128


def build_tokenizer() -> PreTrainedTokenizerFast:
    word_tokenizer = Tokenizer(models.WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="<unk>"))
    word_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation(), pre_tokenizers.Digits(individual_digits=True)]
    )
    word_tokenizer.decoder = decoders.WordPiece(prefix="##")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=word_tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        additional_special_tokens=["<|user|>", "<|assistant|>"],
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.padding_side = "left"
    return tokenizer


def prompt_text(tok, n: int, word: str = TARGET_WORD) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": TEMPLATE.format(W=word, N=n)}], tokenize=False, add_generation_prompt=True
    )


def pretrain(args) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tok = build_tokenizer()
    config = LlamaConfig(
        vocab_size=len(VOCAB),
        hidden_size=args.width,
        intermediate_size=4 * args.width,
        num_hidden_layers=args.layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=MAX_SEQUENCE_LENGTH,
        bos_token_id=None,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    start = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        batch = []
        for _ in range(32):
            if random.random() < 1 / 9:
                word, n = TARGET_WORD, 1
            else:
                word, n = random.choice(OTHER_WORDS), random.randint(1, 30)
            p = tok(prompt_text(tok, n, word), add_special_tokens=False).input_ids
            c = tok(" ".join([word] * n), add_special_tokens=False).input_ids + [tok.eos_token_id]
            batch.append((p, c))
        length = max(len(p) + len(c) for p, c in batch)
        ids = torch.zeros(len(batch), length, dtype=torch.long)
        labels = torch.full((len(batch), length), -100)
        attn = torch.zeros(len(batch), length, dtype=torch.long)
        for i, (p, c) in enumerate(batch):
            seq = p + c
            ids[i, : len(seq)] = torch.tensor(seq)
            attn[i, : len(seq)] = 1
            labels[i, len(p) : len(seq)] = torch.tensor(c)
        loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(json.dumps({"pretrain_step": step, "loss": round(loss.item(), 4)}), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    params = sum(p.numel() for p in model.parameters())
    print(
        json.dumps(
            {
                "saved": str(out),
                "params": params,
                "vocab": len(VOCAB),
                "final_loss": round(loss.item(), 4),
                "seconds": round(time.time() - start, 1),
            }
        ),
        flush=True,
    )


def main() -> int:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    pretrain(parser.parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
