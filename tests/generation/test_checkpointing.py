"""
Tests for the checkpointing functionality.
"""

import json
import os
import tempfile

import pytest
import ray

from marin.generation.checkpointing import (
    CheckpointConfig,
    CheckpointManager,
    filter_completed_records,
    get_output_completed_ids,
)


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpointing tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return [
        {"id": "doc_1", "text": "This is document 1"},
        {"id": "doc_2", "text": "This is document 2"},
        {"id": "doc_3", "text": "This is document 3"},
        {"id": "doc_4", "text": "This is document 4"},
        {"id": "doc_5", "text": "This is document 5"},
    ]


def test_checkpoint_manager_basic(temp_checkpoint_dir):
    """Test basic checkpoint manager functionality."""
    config = CheckpointConfig(checkpoint_dir=temp_checkpoint_dir)
    manager = CheckpointManager(config)

    # Initially no completed IDs
    completed_ids = manager.get_completed_ids()
    assert len(completed_ids) == 0

    # Save some completed IDs
    test_ids = ["doc_1", "doc_2", "doc_3"]
    manager.save_completed_ids(test_ids)

    # Load them back
    loaded_ids = manager.get_completed_ids()
    assert loaded_ids == set(test_ids)

    # Save more IDs (should append)
    more_ids = ["doc_4", "doc_5"]
    manager.save_completed_ids(more_ids)

    # Load all IDs
    all_loaded_ids = manager.get_completed_ids()
    assert all_loaded_ids == set(test_ids + more_ids)


def test_checkpoint_manager_metadata(temp_checkpoint_dir):
    """Test checkpoint metadata functionality."""
    config = CheckpointConfig(checkpoint_dir=temp_checkpoint_dir)
    manager = CheckpointManager(config)

    # Update metadata
    manager.update_metadata(status="running", progress=0.5)

    # Update more metadata
    manager.update_metadata(status="completed", progress=1.0, total_records=100)

    # Check metadata file exists and contains expected data
    metadata_file = os.path.join(temp_checkpoint_dir, "checkpoint_metadata.json")
    assert os.path.exists(metadata_file)

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    assert metadata["status"] == "completed"
    assert metadata["progress"] == 1.0
    assert metadata["total_records"] == 100


def test_filter_completed_records(sample_data):
    """Test filtering completed records from a dataset."""
    # Create a Ray dataset
    dataset = ray.data.from_items(sample_data)

    # Filter out some completed records
    completed_ids = {"doc_2", "doc_4"}
    filtered_dataset = filter_completed_records(dataset, completed_ids, "id")

    # Check that the correct records remain
    remaining_data = filtered_dataset.take_all()
    remaining_ids = {item["id"] for item in remaining_data}

    expected_remaining = {"doc_1", "doc_3", "doc_5"}
    assert remaining_ids == expected_remaining


def test_get_output_completed_ids(temp_checkpoint_dir, sample_data):
    """Test extracting completed IDs from output files."""
    # Create some mock output files
    output_file1 = os.path.join(temp_checkpoint_dir, "output_1.jsonl.gz")
    output_file2 = os.path.join(temp_checkpoint_dir, "output_2.jsonl")

    # Write some sample data to the files
    import gzip

    with gzip.open(output_file1, "wt") as f:
        for item in sample_data[:3]:  # First 3 items
            f.write(json.dumps(item) + "\n")

    with open(output_file2, "w") as f:
        for item in sample_data[3:]:  # Last 2 items
            f.write(json.dumps(item) + "\n")

    # Extract completed IDs
    completed_ids = get_output_completed_ids(temp_checkpoint_dir, "id")

    # Should find all IDs
    expected_ids = {item["id"] for item in sample_data}
    assert completed_ids == expected_ids


def test_get_output_completed_ids_with_pattern(temp_checkpoint_dir, sample_data):
    """Test extracting completed IDs using file patterns."""
    # Create output files with different extensions
    output_file1 = os.path.join(temp_checkpoint_dir, "output_1.jsonl.gz")
    output_file2 = os.path.join(temp_checkpoint_dir, "other_file.txt")  # Should be ignored

    import gzip

    with gzip.open(output_file1, "wt") as f:
        for item in sample_data[:3]:
            f.write(json.dumps(item) + "\n")

    with open(output_file2, "w") as f:
        f.write("This is not a JSON file")

    # Extract completed IDs
    completed_ids = get_output_completed_ids(temp_checkpoint_dir, "id")

    # Should only find IDs from the JSON file
    expected_ids = {item["id"] for item in sample_data[:3]}
    assert completed_ids == expected_ids


@pytest.mark.skipif(not ray.is_initialized(), reason="Ray not initialized")
def test_checkpoint_actor(temp_checkpoint_dir):
    """Test the checkpoint actor functionality."""
    from marin.generation.checkpointing import CheckpointActor

    config = CheckpointConfig(
        checkpoint_dir=temp_checkpoint_dir,
        batch_size=3,  # Small batch size for testing
    )

    # Create checkpoint actor
    actor = CheckpointActor.remote(config)

    # Add some completed IDs
    ray.get(actor.add_completed_id.remote("doc_1"))
    ray.get(actor.add_completed_id.remote("doc_2"))
    ray.get(actor.add_completed_id.remote("doc_3"))  # Should trigger flush

    # Add more IDs
    ray.get(actor.add_completed_ids.remote(["doc_4", "doc_5"]))

    # Finalize to flush remaining IDs
    ray.get(actor.finalize.remote())

    # Check that IDs were saved
    manager = CheckpointManager(config)
    completed_ids = manager.get_completed_ids()

    expected_ids = {"doc_1", "doc_2", "doc_3", "doc_4", "doc_5"}
    assert completed_ids == expected_ids


def test_malformed_checkpoint_file(temp_checkpoint_dir):
    """Test handling of malformed checkpoint files."""
    config = CheckpointConfig(checkpoint_dir=temp_checkpoint_dir)
    manager = CheckpointManager(config)

    # Create a malformed checkpoint file
    checkpoint_file = os.path.join(temp_checkpoint_dir, "completed_ids.jsonl.gz")
    import gzip

    with gzip.open(checkpoint_file, "wt") as f:
        f.write('{"id": "doc_1"}\n')  # Valid line
        f.write("invalid json line\n")  # Invalid line
        f.write('{"id": "doc_2"}\n')  # Valid line

    # Should handle malformed lines gracefully
    completed_ids = manager.get_completed_ids()
    assert completed_ids == {"doc_1", "doc_2"}


if __name__ == "__main__":
    # Initialize Ray for testing
    if not ray.is_initialized():
        ray.init(local_mode=True)

    pytest.main([__file__])
