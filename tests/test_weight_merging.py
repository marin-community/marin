# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
import safetensors.torch
import torch
from marin.merging.arithmetic import MergeMethod, MergeParameters, merge_tensor
from marin.merging.checkpoint import CheckpointReader, CheckpointSource, merge_checkpoint


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
    )
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
        )
