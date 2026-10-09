# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
import safetensors.torch
import torch
from marin.merging.arithmetic import MergeMethod, MergeParameters, RamParameters, merge_tensor
from marin.merging.checkpoint import (
    CheckpointReader,
    CheckpointSource,
    ChunkMerge,
    CurvatureMerge,
    RowMerge,
    merge_checkpoint,
)
from marin.merging.curvature import ota_merge_tensor
from marin.merging.geometry import weight_update_gram
from marin.merging.learned import differentiable_weight_blend


@pytest.mark.parametrize("chunk_elements", [1, 2, 10])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_weight_geometry_distinguishes_opposing_updates(chunk_elements, dtype):
    anchor = torch.tensor([1, 2, 0], dtype=dtype)
    donors = [torch.tensor([3, 2, 0], dtype=dtype), torch.tensor([0, 2, 0], dtype=dtype)]
    gram = weight_update_gram(anchor, donors, chunk_elements=chunk_elements)
    expected = torch.tensor(
        [[5, 7, 4, 2, -1], [7, 13, 4, 6, -3], [4, 4, 4, 0, 0], [2, 6, 0, 4, -2], [-1, -3, 0, -2, 1]], dtype=torch.float64
    )
    torch.testing.assert_close(gram, expected, rtol=0, atol=0)


def test_weight_geometry_preserves_small_updates_and_zero_updates():
    anchor = torch.tensor([2**20, 2**20], dtype=torch.float64)
    donor = anchor + torch.tensor([2**-10, -(2**-10)], dtype=torch.float64)
    gram = weight_update_gram(anchor, [donor, anchor], chunk_elements=1)
    assert gram[3, 3].item() == 2**-19
    assert gram[0, 3].item() == 0
    assert torch.count_nonzero(gram[4]).item() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "method,coefficients,expected",
    [
        (MergeMethod.AVERAGE, (0.25, 0.5), [2.5, 1.0, 5.5]),
        (MergeMethod.TASK_ARITHMETIC, (1.0, 1.0), [5.0, 2.0, 7.0]),
        (MergeMethod.TIES, (1.0, 1.0), [3.0, 2.0, 9.0]),
    ],
)
def test_merge_preserves_dtype_and_computes_weighted_updates(dtype, method, coefficients, expected):
    anchor = torch.tensor([1, 2, 3], dtype=dtype)
    donors = [torch.tensor([3, 6, 1], dtype=dtype), torch.tensor([3, -2, 9], dtype=dtype)]
    result = merge_tensor(anchor, donors, MergeParameters(method, coefficients, 1, 1, 42), tensor_name="weight")
    torch.testing.assert_close(result, torch.tensor(expected, dtype=dtype), rtol=0, atol=0)


def test_ties_trimming_and_conflicting_signs_have_hand_computed_result():
    anchor = torch.zeros(4)
    donors = [torch.tensor([8.0, 4.0, -2.0, 1.0]), torch.tensor([-3.0, 6.0, 7.0, -2.0])]
    result = merge_tensor(anchor, donors, MergeParameters(MergeMethod.TIES, (1, 1), 0.5, 1, 42), tensor_name="weight")
    torch.testing.assert_close(result, torch.tensor([8.0, 5.0, 7.0, 0.0]), rtol=0, atol=0)


def test_dare_mask_is_order_independent_and_rescales_survivors():
    anchor = torch.full((256,), 2.0)
    donor = torch.full((256,), 3.0)
    recipe = MergeParameters(MergeMethod.DARE_LINEAR, (1,), 0.5, 1, 42)
    first = merge_tensor(anchor, [donor], recipe, tensor_name="first")
    other = merge_tensor(anchor, [donor], recipe, tensor_name="other")
    repeated = merge_tensor(anchor, [donor], recipe, tensor_name="first")
    assert torch.equal(first, repeated)
    assert not torch.equal(first, other)
    assert set(first.tolist()) == {2.0, 4.0}


@pytest.mark.parametrize("method", [MergeMethod.DARE_LINEAR, MergeMethod.DARE_TIES])
def test_dare_full_density_retains_nonconflicting_donor(method):
    anchor = torch.tensor([1.0, 4.0, 9.0])
    donor = torch.tensor([-1.0, 5.0, 9.0])
    result = merge_tensor(anchor, [donor], MergeParameters(method, (1,), 1, 1, 7), tensor_name="weight")
    torch.testing.assert_close(result, donor, rtol=0, atol=0)


@pytest.mark.parametrize("tensor_coefficients,expected_a", [({}, [3.0, 2.0]), ({"a": (1.0,)}, [5.0, 2.0])])
def test_checkpoint_merge_handles_different_sharding_and_publishes_verified_manifest(
    tmp_path, tensor_coefficients, expected_a
):
    anchor_dir, donor_dir, output = (tmp_path / name for name in ("anchor", "donor", "output"))
    for folder in (anchor_dir, donor_dir):
        folder.mkdir()
        (folder / "config.json").write_text('{"model_type":"fixture"}')
        (folder / "tokenizer_config.json").write_text('{"eos_token":"end"}')
    anchor_weights = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([[3.0]])}
    donor_weights = {"a": torch.tensor([5.0, 6.0]), "b": torch.tensor([[7.0]])}
    safetensors.torch.save_file(anchor_weights, anchor_dir / "all.safetensors")
    (anchor_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "all.safetensors" for name in anchor_weights}})
    )
    for name, tensor in donor_weights.items():
        safetensors.torch.save_file({name: tensor}, donor_dir / f"{name}.safetensors")
    (donor_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: f"{name}.safetensors" for name in donor_weights}})
    )
    parameters = MergeParameters(MergeMethod.AVERAGE, (0.5,), 1, 1, 0)
    merge_checkpoint(
        CheckpointSource(str(anchor_dir), "anchor-revision"),
        [CheckpointSource(str(donor_dir), "donor-revision")],
        str(output),
        parameters,
        code_revision="code-revision",
        preserve_rows={"a": (1,)},
        tensor_coefficients=tensor_coefficients,
        row_overrides={},
    )
    manifest_path = output / "merge-manifest.json"
    completed_manifest = manifest_path.read_bytes()
    manifest_path.unlink()
    with pytest.raises(ValueError, match="Incomplete merged checkpoint"):
        CheckpointReader(CheckpointSource(str(output), "interrupted-export"))
    manifest_path.write_bytes(completed_manifest)
    reader = CheckpointReader(CheckpointSource(str(output), "merged"))
    torch.testing.assert_close(reader.tensor("a"), torch.tensor(expected_a), rtol=0, atol=0)
    torch.testing.assert_close(reader.tensor("b"), torch.tensor([[5.0]]), rtol=0, atol=0)
    manifest = json.loads((output / "merge-manifest.json").read_text())
    for entry in manifest["objects"]:
        data = (output / entry["name"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert len(data) == entry["bytes"]
    assert (output / "tokenizer_config.json").read_bytes() == (anchor_dir / "tokenizer_config.json").read_bytes()
    with pytest.raises(FileExistsError):
        merge_checkpoint(
            CheckpointSource(str(anchor_dir), "anchor-revision"),
            [CheckpointSource(str(donor_dir), "donor-revision")],
            str(output),
            parameters,
            code_revision="code-revision",
            preserve_rows={"a": (1,)},
            tensor_coefficients=tensor_coefficients,
            row_overrides={},
        )


def test_checkpoint_row_merges_preserve_other_experts_and_router(tmp_path):
    names = [f"model.layers.0.mlp.experts.{projection}.weight" for projection in ("gate_proj", "up_proj", "down_proj")]
    router = "model.layers.0.mlp.gate.weight"
    sources = []
    weights = []
    for source_index, offset in enumerate((0, 10, 20)):
        folder = tmp_path / f"source-{source_index}"
        folder.mkdir()
        tensors = {
            name: torch.arange(12).reshape(3, 2, 2).float() + offset + index * 100 for index, name in enumerate(names)
        }
        tensors[router] = torch.arange(9).reshape(3, 3).float() + offset
        safetensors.torch.save_file(tensors, folder / "weights.safetensors")
        (folder / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: "weights.safetensors" for name in tensors}})
        )
        (folder / "config.json").write_text('{"model_type":"fixture"}')
        sources.append(CheckpointSource(str(folder), str(source_index)))
        weights.append(tensors)
    output = tmp_path / "merged"
    manifest = merge_checkpoint(
        sources[0],
        sources[1:],
        str(output),
        MergeParameters(MergeMethod.AVERAGE, (1, 0), 1, 1, 0),
        code_revision="fixture",
        preserve_rows={},
        tensor_coefficients={},
        row_overrides={name: (RowMerge(1, (0, 1)), RowMerge(2, (0.5, 0.5))) for name in names},
    )
    reader = CheckpointReader(CheckpointSource(str(output), "merged"))
    for name in names:
        merged = reader.tensor(name)
        torch.testing.assert_close(merged[0], weights[1][name][0], rtol=0, atol=0)
        torch.testing.assert_close(merged[1], weights[2][name][1], rtol=0, atol=0)
        torch.testing.assert_close(merged[2], weights[1][name][2] + 5, rtol=0, atol=0)
    torch.testing.assert_close(reader.tensor(router), weights[1][router], rtol=0, atol=0)
    saved_manifest = json.loads((output / "merge-manifest.json").read_text())
    assert saved_manifest["row_overrides"][names[0]] == [
        {"row": 1, "coefficients": [0, 1]},
        {"row": 2, "coefficients": [0.5, 0.5]},
    ]
    assert manifest["tensor_count"] == 4


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "method,rescale,expected",
    [
        (MergeMethod.RAM, 0, [12, 14, 10, 10, 16]),
        (MergeMethod.RAM_PLUS_TL, 1, [13, 16, 10, 10, 16]),
    ],
)
def test_ram_preserves_unique_updates_and_averages_shared_updates(dtype, method, rescale, expected):
    anchor = torch.full((5,), 10, dtype=dtype)
    donors = [
        torch.tensor([12, 10, 14, 10.125, 14], dtype=dtype),
        torch.tensor([10, 14, 6, 10, 18], dtype=dtype),
    ]
    parameters = MergeParameters(method, (1, 1), 1, 1, 42, RamParameters(0.25, rescale, 0.5))
    result = merge_tensor(anchor, donors, parameters, tensor_name="weight")
    torch.testing.assert_close(result, torch.tensor(expected, dtype=dtype), rtol=0, atol=0)


def test_curvature_grafting_keeps_small_sensitive_edits_and_weights_all_donors():
    anchor = torch.zeros(2)
    donors = [torch.tensor([1.0, 10.0]), torch.tensor([3.0, 2.0])]
    moments = [torch.tensor([100.0, 0.01]), torch.tensor([4.0, 25.0])]
    result = ota_merge_tensor(anchor, donors, moments, density=0.5, epsilon=1.0)
    # Saliencies are [100, 1] and [36, 100]; each donor retains a different coordinate.
    expected = torch.tensor([11 / 14, 12 / 7.1])
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("block_elements", [1, 3, 100])
def test_learned_blend_gradients_match_direct_weight_arithmetic(block_elements):
    anchor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    donors = (anchor + 2, anchor - 1)
    coefficients = torch.tensor([[0.25, 0.5], [0.75, 0.25]], requires_grad=True)
    inputs = torch.tensor([2.0, -1.0])
    weights = differentiable_weight_blend(anchor, donors, coefficients, block_elements=block_elements)
    prediction = weights @ inputs
    prediction.square().sum().backward()
    # Merged rows are [1, 2] and [4.25, 5.25], giving predictions 0 and 3.25.
    torch.testing.assert_close(prediction, torch.tensor([0.0, 3.25]), rtol=0, atol=0)
    torch.testing.assert_close(coefficients.grad, torch.tensor([[0.0, 0.0], [13.0, -6.5]]), rtol=0, atol=0)


def test_learned_blend_uneven_chunks_use_every_coefficient():
    anchor = torch.zeros(4)
    donor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    coefficients = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    result = differentiable_weight_blend(anchor, (donor,), coefficients, block_elements=2)
    result.sum().backward()
    torch.testing.assert_close(result, torch.tensor([1.0, 4.0, 9.0, 12.0]), rtol=0, atol=0)
    torch.testing.assert_close(coefficients.grad, torch.tensor([[1.0], [2.0], [7.0]]), rtol=0, atol=0)


def test_curvature_full_density_preserves_updates_without_measured_moment():
    result = ota_merge_tensor(
        torch.tensor([10.0]),
        [torch.tensor([12.0]), torch.tensor([16.0])],
        [torch.zeros(1), torch.zeros(1)],
        density=1.0,
        epsilon=1e-6,
    )
    torch.testing.assert_close(result, torch.tensor([14.0]), rtol=0, atol=0)


@pytest.mark.parametrize("method", ["chunks", "curvature"])
def test_calibrated_checkpoint_round_trip_matches_materialized_updates(tmp_path, method):
    sources = []
    for label, values in (
        ("anchor", [1.0, 2.0, 3.0, 4.0]),
        ("donor", [5.0, 6.0, 7.0, 8.0]),
        ("moments", [1.0, 4.0, 9.0, 16.0]),
    ):
        folder = tmp_path / label
        folder.mkdir()
        safetensors.torch.save_file(
            {"weight": torch.tensor(values), "fixed": torch.tensor([9.0])}, folder / "weights.safetensors"
        )
        (folder / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "weights.safetensors", "fixed": "weights.safetensors"}})
        )
        sources.append(CheckpointSource(str(folder), label))
    if method == "chunks":
        calibration = ChunkMerge({"weight": ((0.25,), (0.75,))}, 1, "trained.json", "fixture")
        expected = torch.tensor([2.0, 3.0, 6.0, 4.0])
    else:
        calibration = CurvatureMerge((sources[2],), 0.5, 1e-6)
        expected = torch.tensor([1.0, 2.0, 7.0, 4.0])
    output = tmp_path / "output"
    merge_checkpoint(
        sources[0],
        [sources[1]],
        str(output),
        MergeParameters(MergeMethod.TASK_ARITHMETIC, (0.0,), 1.0, 1.0, 42),
        code_revision="test",
        preserve_rows={"weight": (3,)},
        tensor_coefficients={},
        row_overrides={},
        calibration=calibration,
    )
    reader = CheckpointReader(CheckpointSource(str(output), "output"))
    torch.testing.assert_close(reader.tensor("weight"), expected, rtol=0, atol=0)
    torch.testing.assert_close(reader.tensor("fixed"), torch.tensor([9.0]), rtol=0, atol=0)
    manifest = json.loads((output / "merge-manifest.json").read_text())
    for entry in manifest["objects"]:
        assert hashlib.sha256((output / entry["name"]).read_bytes()).hexdigest() == entry["sha256"]
