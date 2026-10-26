"""The online draft publication fence covers the entire checkpoint upload."""

from dataclasses import asdict
import json
from pathlib import Path
import sys

from fsspec.implementations.memory import MemoryFileSystem
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
for source_root in (_REPO_ROOT, _REPO_ROOT / "skyrl-train"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from marinskyrl.speculative_decoding import SpeculatorModelConfig  # noqa: E402
from skyrl_train.draft_trainer import DraftTrainer, read_latest_draft_checkpoint  # noqa: E402
from skyrl_train.io import io  # noqa: E402


@pytest.mark.parametrize("failure", [None, "upload", "truncated"])
def test_publication_preserves_incumbent_until_all_files_arrive(tmp_path, monkeypatch, failure):
    filesystem = MemoryFileSystem()
    root = f"s3://drafts/{tmp_path.name}"
    monkeypatch.setattr(io, "_get_filesystem", lambda _uri: filesystem)
    trainer = DraftTrainer(
        initial_model=SpeculatorModelConfig("/initial", "original-source"),
        checkpoint_root=root,
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "model.safetensors").write_bytes(b"learned draft")
    (candidate / "config.json").write_text("{}")
    # Check that non-serving recovery state crosses the same publication fence.
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.sum().backward()
    optimizer.step()
    torch.save(optimizer.state_dict(), candidate / "trainer_state.pt")
    incumbent = trainer._publish_checkpoint(step=1, revision="draft-step-1", candidate_dir=candidate)
    assert read_latest_draft_checkpoint(root) == incumbent
    assert read_latest_draft_checkpoint(root, source_identity="another-source") is None

    put_file = filesystem.put_file

    def upload(source, destination, **kwargs):
        assert read_latest_draft_checkpoint(root) == incumbent
        if failure == "upload" and source.endswith("trainer_state.pt"):
            raise OSError("lost worker during optimizer upload")
        put_file(source, destination, **kwargs)
        if failure == "truncated" and source.endswith("model.safetensors"):
            filesystem.pipe_file(destination, b"short")

    monkeypatch.setattr(filesystem, "put_file", upload)
    if failure is not None:
        with pytest.raises(OSError):
            trainer._publish_checkpoint(step=2, revision="draft-step-2", candidate_dir=candidate)
        assert read_latest_draft_checkpoint(root) == incumbent
        return

    published = trainer._publish_checkpoint(step=2, revision="draft-step-2", candidate_dir=candidate)
    assert read_latest_draft_checkpoint(root) == published
    assert published.revision == "draft-step-2"
    restored = tmp_path / "new-worker"
    io.download_directory(published.uri, str(restored))
    restored_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1.0)
    restored_optimizer.load_state_dict(torch.load(restored / "trainer_state.pt", weights_only=True))
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    torch.testing.assert_close(
        next(iter(restored_optimizer.state.values()))["exp_avg"],
        optimizer.state[parameter]["exp_avg"],
        rtol=0,
        atol=0,
    )
    assert json.loads(io.read_bytes(root + "/latest.json")) == asdict(published)
    # A later truncated object must also be rejected by a refresh reader.
    filesystem.pipe_file(published.weights_uri, b"short")
    with pytest.raises(ValueError, match="incomplete"):
        read_latest_draft_checkpoint(root)
