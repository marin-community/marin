import fsspec
from fsspec import AbstractFileSystem


def test_discover_latest_hf_checkpoints():
    temp_fs: AbstractFileSystem = fsspec.filesystem("memory")

    # Create a temporary directory
    temp_fs.mkdirs("checkpoints/hf/step-1000")
    temp_fs.mkdirs("checkpoints/hf/step-9999")
    temp_fs.touch("checkpoints/hf/step-1000/config.json")
    temp_fs.touch("checkpoints/hf/step-1000/tokenizer_config.json")
    temp_fs.touch("checkpoints/hf/step-9999/config.json")
    temp_fs.touch("checkpoints/hf/step-9999/tokenizer_config.json")

    # Test the function
    from marin.evaluation.utils import discover_hf_checkpoints

    checkpoints = discover_hf_checkpoints("memory:///")

    assert checkpoints == ["memory:///checkpoints/hf/step-1000", "memory:///checkpoints/hf/step-9999"]
