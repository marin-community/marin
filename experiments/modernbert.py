# Requires transformers>=4.48.0

import ray
import torch
import torch_xla.core.xla_model as xm
import transformers
from torch_xla.experimental.custom_kernel import flash_attention_xla
from transformers import AutoTokenizer, ModernBertForSequenceClassification
from transformers.models.modernbert.modular_modernbert import apply_rotary_pos_emb


def eager_attention_forward(
    module: "transformers.models.modernbert.modeling_modernbert.ModernBertAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: torch.LongTensor | None,
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    output_attentions: bool | None = False,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)

    # position_ids = torch.arange(0, query.shape[2], device=query.device)
    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    # scale = self.head_dim**-0.5
    # attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    # attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=False)
    # attn_output = torch.matmul(attn_weights, value)
    attn_output = flash_attention_xla(query, key, value, False)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bs, -1, dim)
    if output_attentions:
        scale = module.head_dim**-0.5
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale
        attn_weights = attn_weights + attention_mask
        return (attn_output, attn_weights)

    return (attn_output,)
    # attn_outputs = (attn_output, attn_weights)

    # return


def forward(
    self,
    hidden_states: torch.Tensor,
    output_attentions: bool | None = False,
    **kwargs,
) -> torch.Tensor:
    qkv = self.Wqkv(hidden_states)

    bs = hidden_states.shape[0]
    if self.config._attn_implementation == "flash_attention_2":
        qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
    else:
        qkv = qkv.view(bs, -1, 3, self.num_heads, self.head_dim)

    attn_outputs = eager_attention_forward(
        self,
        qkv=qkv,
        rotary_emb=self.rotary_emb,
        local_attention=self.local_attention,
        bs=bs,
        dim=self.all_head_size,
        output_attentions=output_attentions,
        **kwargs,
    )

    hidden_states = attn_outputs[0]
    hidden_states = self.out_drop(self.Wo(hidden_states))

    return (hidden_states,) + attn_outputs[1:]  # add attentions if outputted


@ray.remote(resources={"TPU": 1})
def run_modernbert_attention():
    device = xm.xla_device()
    model = ModernBertForSequenceClassification.from_pretrained(
        "Alibaba-NLP/gte-modernbert-base", reference_compile=False, num_labels=1
    ).to(device)
    model.eval()

    first_layer = model.model.layers[0]

    # x = torch.randn(1, 256, 768).to(device)
    input_texts = ["what is the capital of China?"]

    model_path = "Alibaba-NLP/gte-modernbert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors="pt")
    batch_dict = batch_dict.to(device)
    attention_mask, sliding_window_mask = model.model._update_attention_mask(
        batch_dict["attention_mask"], output_attentions=True
    )

    print(batch_dict)
    embeddings = model.model.embeddings(batch_dict["input_ids"])
    ref_attn_output = first_layer.attn(
        first_layer.attn_norm(embeddings),
        attention_mask=attention_mask,
        sliding_window_mask=sliding_window_mask,
        position_ids=torch.arange(embeddings.shape[1], device=embeddings.device).unsqueeze(0),
        cu_seqlens=None,
        max_seqlen=None,
        output_attentions=False,
    )

    patched_model = ModernBertForSequenceClassification.from_pretrained(
        "Alibaba-NLP/gte-modernbert-base", reference_compile=False, num_labels=1
    ).to(device)
    patched_model.eval()

    # patched_model.model.layers[0].attn.forward = forward
    transformers.models.modernbert.modeling_modernbert.ModernBertAttention.forward = forward

    patched_attn_output = patched_model.model.layers[0].attn(
        patched_model.model.layers[0].attn_norm(embeddings),
        attention_mask=attention_mask,
        sliding_window_mask=sliding_window_mask,
        position_ids=torch.arange(embeddings.shape[1], device=embeddings.device).unsqueeze(0),
        cu_seqlens=None,
        max_seqlen=None,
        output_attentions=False,
    )

    all_close = torch.allclose(ref_attn_output[0], patched_attn_output[0]), "Attention output mismatch"
    if not all_close:
        print(f"Max diff: {torch.max(torch.abs(ref_attn_output[0] - patched_attn_output[0]))}")
        print(f"Mean diff: {torch.mean(torch.abs(ref_attn_output[0] - patched_attn_output[0]))}")

    # print(ref_attn_output[1].shape)
    # print(patched_attn_output[1].shape)

    # print(attn_output[1].shape)


@ray.remote(resources={"TPU": 1})
def run_modernbert():
    device = xm.xla_device()
    input_texts = [
        "what is the capital of China?",
        "how to implement quick sort in python?",
        "Beijing",
        "sorting algorithms",
    ]

    model_path = "Alibaba-NLP/gte-modernbert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # config = ModernBertConfig(reference_compile=False)
    model = ModernBertForSequenceClassification.from_pretrained(model_path, reference_compile=False, num_labels=1).to(
        device
    )

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors="pt")
    print(batch_dict)
    batch_dict = batch_dict.to(device)
    output = model(**batch_dict)
    print(output.logits.shape)
    # embeddings = outputs.last_hidden_state[:, 0]

    # (Optionally) normalize embeddings
    # embeddings = F.normalize(embeddings, p=2, dim=1)
    # scores = (embeddings[:1] @ embeddings[1:].T) * 100
    # print(scores.tolist())


if __name__ == "__main__":
    # transformers.models.modernbert.modeling_modernbert.ModernBertAttention.forward = forward
    ray.get(run_modernbert_attention.remote())

# tensor([ 2.1387,  2.4609, -1.6729])
