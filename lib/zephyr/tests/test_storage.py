# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from zephyr.storage import ChunkWriter, InlineRef, StorageManager, StorageRef
from zephyr.writers import write_vortex_file


def test_inline_ref_load_roundtrip():
    """Test InlineRef iteration roundtrip."""
    data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}, {"id": 3, "name": "Charlie"}]
    ref = InlineRef(data=data)

    assert ref.count == 3
    loaded = list(ref)
    assert loaded == data


def test_storage_ref_load_roundtrip(tmp_path):
    """Test StorageRef iteration roundtrip with Vortex file."""
    data = [{"id": 1, "value": 100}, {"id": 2, "value": 200}, {"id": 3, "value": 300}]
    vortex_path = tmp_path / "test.vortex"

    result = write_vortex_file(data, str(vortex_path))
    ref = StorageRef(path=str(vortex_path), count=result["count"])

    assert ref.count == 3
    loaded = list(ref)
    assert len(loaded) == 3
    assert loaded[0]["id"] == 1
    assert loaded[1]["value"] == 200


def test_chunk_writer_spills_over_threshold(tmp_path):
    """Test ChunkWriter spills to storage when over threshold."""
    spill_path = tmp_path / "chunk.vortex"
    # Use a low threshold to force spill
    writer = ChunkWriter(spill_path=str(spill_path), spill_threshold_bytes=1000)

    # Write enough data to exceed threshold
    items = [{"id": i, "data": "x" * 1000} for i in range(10)]
    for item in items:
        writer.write(item)

    ref = writer.finish()

    # Should be spilled to storage
    assert isinstance(ref, StorageRef)
    assert ref.count == 10
    assert ref.path == str(spill_path)
    assert spill_path.exists()

    # Verify data roundtrip
    loaded = list(ref)
    assert len(loaded) == 10
    assert loaded[0]["id"] == 0
    assert loaded[9]["id"] == 9


def test_storage_manager_path_generation(tmp_path):
    """Test StorageManager generates correct paths."""
    storage = StorageManager(base_path=str(tmp_path))

    # Check job_path
    assert storage.job_path.startswith(str(tmp_path))
    assert "job_" in storage.job_path

    # Check chunk_path format
    chunk_path = storage.chunk_path(shard_idx=0, chunk_idx=0)
    assert "stage_0" in chunk_path
    assert "shard_00000" in chunk_path
    assert "chunk_00000.vortex" in chunk_path

    # Check with different indices
    chunk_path = storage.chunk_path(shard_idx=123, chunk_idx=456)
    assert "shard_00123" in chunk_path
    assert "chunk_00456.vortex" in chunk_path


def test_storage_manager_cleanup_removes_directory(tmp_path):
    """Test StorageManager.cleanup() removes job directory."""
    storage = StorageManager(base_path=str(tmp_path), spill_threshold_bytes=1000)

    # Write some data to create the directory
    items = [{"id": i, "data": "x" * 1000} for i in range(5)]
    writer = storage.create_writer(shard_idx=0, chunk_idx=0)
    for item in items:
        writer.write(item)
    ref = writer.finish()

    assert isinstance(ref, StorageRef)
    job_path = Path(storage.job_path)
    assert job_path.exists()

    # Cleanup should remove the directory
    storage.cleanup()
    assert not job_path.exists()


def test_storage_manager_context_manager_calls_cleanup(tmp_path):
    """Test StorageManager context manager calls cleanup on exit."""
    items = [{"id": i, "data": "x" * 1000} for i in range(5)]
    job_path = None

    with StorageManager(base_path=str(tmp_path), spill_threshold_bytes=1000) as storage:
        writer = storage.create_writer(shard_idx=0, chunk_idx=0)
        for item in items:
            writer.write(item)
        ref = writer.finish()
        assert isinstance(ref, StorageRef)
        job_path = Path(storage.job_path)
        assert job_path.exists()

    # After exiting context, cleanup should have run
    assert not job_path.exists()


def test_storage_ref_with_complex_data(tmp_path):
    """Test StorageRef with complex nested data structures."""
    data = [
        {"id": 1, "nested": {"key": "value"}, "array": [1, 2, 3]},
        {"id": 2, "nested": {"key": "other"}, "array": [4, 5, 6]},
    ]
    vortex_path = tmp_path / "complex.vortex"
    write_vortex_file(data, str(vortex_path))

    ref = StorageRef(path=str(vortex_path), count=2)
    loaded = list(ref)

    assert len(loaded) == 2
    assert loaded[0]["id"] == 1
