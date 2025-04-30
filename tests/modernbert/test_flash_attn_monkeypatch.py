# Requires transformers>=4.48.0

import types

import ray
import torch
import torch_xla.core.xla_model as xm
from transformers import AutoTokenizer, ModernBertForSequenceClassification


@ray.remote(resources={"TPU": 1})
def test_modernbert_flash_attn():
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
        batch_dict["attention_mask"], output_attentions=False
    )

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

    sliding_window_ref_attn_output = model.model.layers[1].attn(
        model.model.layers[1].attn_norm(embeddings),
        attention_mask=attention_mask,
        sliding_window_mask=sliding_window_mask,
        position_ids=torch.arange(embeddings.shape[1], device=embeddings.device).unsqueeze(0),
        cu_seqlens=None,
        max_seqlen=None,
        output_attentions=False,
    )
    print(sliding_window_ref_attn_output[0])
    ref_model_output = model(**batch_dict).logits
    print(ref_model_output)

    # transformers.models.modernbert.modeling_modernbert.ModernBertAttention.forward = forward

    patched_model = ModernBertForSequenceClassification.from_pretrained(
        "Alibaba-NLP/gte-modernbert-base", reference_compile=False, num_labels=1
    ).to(device)
    patched_model.eval()

    for layer in patched_model.model.layers:
        layer.attn.forward = types.MethodType(forward, layer.attn)

    patched_model.classifier.bias = model.classifier.bias
    patched_model.classifier.weight = model.classifier.weight
    patched_model.head.dense.weight = model.head.dense.weight
    patched_model.head.norm.weight = model.head.norm.weight

    # patched_model.model.layers[0].attn.forward = forward

    # patched_attn_output = patched_model.model.layers[0].attn(
    #     patched_model.model.layers[0].attn_norm(embeddings),
    #     attention_mask=attention_mask,
    #     sliding_window_mask=sliding_window_mask,
    #     position_ids=torch.arange(embeddings.shape[1], device=embeddings.device).unsqueeze(0),
    #     cu_seqlens=None,
    #     max_seqlen=None,
    #     output_attentions=False,
    # )

    # sliding_window_patched_attn_output = patched_model.model.layers[1].attn(
    #     patched_model.model.layers[1].attn_norm(embeddings),
    #     attention_mask=attention_mask,
    #     sliding_window_mask=sliding_window_mask,
    #     position_ids=torch.arange(embeddings.shape[1], device=embeddings.device).unsqueeze(0),
    #     cu_seqlens=None,
    #     max_seqlen=None,
    #     output_attentions=False,
    # )
    # print(sliding_window_patched_attn_output[0])

    # Start with embeddings
    ref_hidden = embeddings.clone()
    patched_hidden = embeddings.clone()
    for i in range(len(model.model.layers)):
        ref_layer = model.model.layers[i]
        patched_layer = patched_model.model.layers[i]
        ref_hidden = ref_layer(
            ref_hidden,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=torch.arange(embeddings.shape[1], device=embeddings.device).unsqueeze(0),
        )[0]
        patched_hidden = patched_layer(
            patched_hidden,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=torch.arange(embeddings.shape[1], device=embeddings.device).unsqueeze(0),
        )[0]
        if not torch.allclose(ref_hidden, patched_hidden, atol=1e-5):
            print(f"Hidden states differ at layer {i}")
            break

    patched_model_output = patched_model(**batch_dict).logits
    print(patched_model_output)

    # all_close = torch.allclose(ref_attn_output[0], patched_attn_output[0])
    # if not all_close:
    #     print(f"Max diff: {torch.max(torch.abs(ref_attn_output[0] - patched_attn_output[0]))}")
    #     print(f"Mean diff: {torch.mean(torch.abs(ref_attn_output[0] - patched_attn_output[0]))}")
    # else:
    #     print("Attention output matches")

    # all_close = torch.allclose(sliding_window_ref_attn_output[0], sliding_window_patched_attn_output[0])
    # if not all_close:
    #     print(f"Max diff: {torch.max(torch.abs(sliding_window_ref_attn_output[0] - sliding_window_patched_attn_output[0]))}")
    #     print(f"Mean diff: {torch.mean(torch.abs(sliding_window_ref_attn_output[0] - sliding_window_patched_attn_output[0]))}")
    # else:
    #     print("Sliding window attention output matches")

    all_close = torch.allclose(ref_model_output, patched_model_output)
    if not all_close:
        print(f"Max diff: {torch.max(torch.abs(ref_model_output - patched_model_output))}")
        print(f"Mean diff: {torch.mean(torch.abs(ref_model_output - patched_model_output))}")
    else:
        print("Model output matches")

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
