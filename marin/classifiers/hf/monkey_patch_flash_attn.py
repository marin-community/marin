import torch
import transformers
from torch_xla.experimental.custom_kernel import flash_attention
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

    scale = module.head_dim**-0.5

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    # upcast attention to fp32
    attn_output = flash_attention(query, key, value, False, None, None, scale, ab=attention_mask)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bs, -1, dim)

    if output_attentions:
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale
        attn_weights = attn_weights + attention_mask
        return (attn_output, attn_weights)

    return (attn_output,)


def forward(
    self,
    hidden_states: torch.Tensor,
    output_attentions: bool | None = False,
    **kwargs,
) -> torch.Tensor:
    qkv = self.Wqkv(hidden_states)

    bs = hidden_states.shape[0]
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


def apply_flash_attn_monkey_patch():
    transformers.models.modernbert.modeling_modernbert.ModernBertAttention.forward = forward
