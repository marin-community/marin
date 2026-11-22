from pathlib import Path

from safetensors.torch import load_file, save_file
import torch
from torch import nn

from skyrl_train.inference_engines.vllm.online_eagle_trainer import _candidate_state


class DraftWithVocabMaps(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding.from_pretrained(torch.arange(32, dtype=torch.float32).reshape(8, 4))
        self.fc = nn.Linear(4, 4, bias=False)
        self.register_buffer("t2d", torch.zeros(8, dtype=torch.bool))
        self.register_buffer("d2t", torch.zeros(4, dtype=torch.long))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.fc(self.embed_tokens(input_ids[self.t2d]))
        return logits[:, self.d2t]


def test_candidate_checkpoint_restores_vocabulary_maps_without_target_embedding(tmp_path: Path):
    model = DraftWithVocabMaps()
    with torch.no_grad():
        model.fc.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16)
        model.t2d.copy_(torch.tensor([True, False, True, False, True, False, True, False]))
        model.d2t.copy_(torch.tensor([2, 0, 3, 1]))
    checkpoint = tmp_path / "model.safetensors"
    save_file(_candidate_state(model, serving_dtype=torch.float32), checkpoint)
    saved = load_file(checkpoint)
    assert "embed_tokens.weight" not in saved

    restored = DraftWithVocabMaps()
    restored.load_state_dict(saved, strict=False)
    input_ids = torch.arange(8)
    torch.testing.assert_close(restored(input_ids), model(input_ids), rtol=0, atol=0)
