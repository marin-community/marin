# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Platform-neutral Snowball BF16 export and content manifest."""

import argparse
import dataclasses
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import safetensors
from haliax import Axis
from haliax.partitioning import set_mesh
from huggingface_hub import snapshot_download
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import validate_single_name_config
from rigging.filesystem import StoragePath
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from experiments.grug.moe.model import GrugModelConfig
from tests.cluster.vllm.snowball import (
    SNOWBALL,
    SNOWBALL_NATIVE_GPU,
    SNOWBALL_NATIVE_TPU,
    TOKENIZER_FILE_PATTERNS,
    NativeLevanterCell,
    RepresentativeGolden,
    read_prompt_fixture,
    read_representative_goldens,
)
from tests.cluster.vllm.snowball_checkpoint import (
    VendoredTransformer,
    decode_vendored_config,
    load_checkpoint,
    logical_array_digest,
    prepare_bf16_parameters,
    read_executor_info,
)

EXPORT_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExportFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ExportReport:
    platform: str
    logical_bf16_parameters_sha256: str
    executor_config_sha256: str
    tokenizer: str
    tokenizer_revision: str
    tree_sha256: str
    total_bytes: int
    files: tuple[ExportFile, ...]
    canonical_tree_match: bool
    uploaded_uri: str | None
    schema_version: int = EXPORT_REPORT_SCHEMA_VERSION

    def to_json_bytes(self) -> bytes:
        return (
            json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> "ExportReport":
        raw = json.loads(payload)
        if raw["schema_version"] != EXPORT_REPORT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported export report schema {raw['schema_version']}")
        return cls(
            platform=raw["platform"],
            logical_bf16_parameters_sha256=raw["logical_bf16_parameters_sha256"],
            executor_config_sha256=raw["executor_config_sha256"],
            tokenizer=raw["tokenizer"],
            tokenizer_revision=raw["tokenizer_revision"],
            tree_sha256=raw["tree_sha256"],
            total_bytes=int(raw["total_bytes"]),
            files=tuple(ExportFile(**file) for file in raw["files"]),
            canonical_tree_match=bool(raw["canonical_tree_match"]),
            uploaded_uri=raw["uploaded_uri"],
            schema_version=int(raw["schema_version"]),
        )


def _decode_main_config(model_config: dict[str, Any]) -> GrugModelConfig:
    main_fields = {field.name for field in dataclasses.fields(GrugModelConfig)}
    return draccus.decode(
        GrugModelConfig,
        {name: value for name, value in model_config.items() if name in main_fields},
    )


def _linear_inference_tensor(value: jax.Array) -> jax.Array:
    return jnp.swapaxes(value, -1, -2)


class StackedSnowballExportModel(eqx.Module):
    """Exporter view that materializes only one HF output shard at a time."""

    params: VendoredTransformer
    config: GrugModelConfig = eqx.field(static=True)

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self.config.vocab_size)

    def to_state_dict(self, prefix: str | None = None) -> dict[str, jax.Array]:
        return self._state_dict_subset(None, prefix=prefix)

    def to_state_dict_subset(self, subset: tuple[str, ...]) -> dict[str, jax.Array]:
        return self._state_dict_subset(frozenset(subset), prefix=None)

    def _state_dict_subset(
        self,
        subset: frozenset[str] | None,
        *,
        prefix: str | None,
    ) -> dict[str, jax.Array]:
        params = self.params
        assert params.stacked_blocks is not None
        stacked = cast(Any, params.stacked_blocks.stacked)

        def include(name: str) -> bool:
            return subset is None or name in subset

        def add(output: dict[str, jax.Array], name: str, value: jax.Array) -> None:
            if include(name):
                output[name if prefix is None else f"{prefix}.{name}"] = value

        output: dict[str, jax.Array] = {}
        add(output, "model.embed_tokens.weight", params.token_embed)
        add(output, "model.embed_norm.weight", params.embed_norm.weight)
        add(
            output,
            "model.embed_gated_norm.down_proj.weight",
            _linear_inference_tensor(params.embed_gated_norm.w_down),
        )
        add(
            output,
            "model.embed_gated_norm.up_proj.weight",
            _linear_inference_tensor(params.embed_gated_norm.w_up),
        )
        add(output, "model.norm.weight", params.final_norm.weight)
        add(
            output,
            "model.final_gated_norm.down_proj.weight",
            _linear_inference_tensor(params.final_gated_norm.w_down),
        )
        add(
            output,
            "model.final_gated_norm.up_proj.weight",
            _linear_inference_tensor(params.final_gated_norm.w_up),
        )
        add(output, "lm_head.weight", _linear_inference_tensor(params.output_proj))

        layer_values = {
            "input_layernorm.weight": stacked.rms_attn.weight,
            "attn_gated_norm.down_proj.weight": _linear_inference_tensor(stacked.attn_gated_norm.w_down),
            "attn_gated_norm.up_proj.weight": _linear_inference_tensor(stacked.attn_gated_norm.w_up),
            "self_attn.q_proj.weight": _linear_inference_tensor(stacked.attn.w_q),
            "self_attn.k_proj.weight": _linear_inference_tensor(stacked.attn.w_k),
            "self_attn.v_proj.weight": _linear_inference_tensor(stacked.attn.w_v),
            "self_attn.o_proj.weight": _linear_inference_tensor(stacked.attn.w_o),
            "self_attn.attn_gate.weight": _linear_inference_tensor(stacked.attn.attn_gate),
            "post_attention_layernorm.weight": stacked.rms_mlp.weight,
            "mlp_gated_norm.down_proj.weight": _linear_inference_tensor(stacked.mlp_gated_norm.w_down),
            "mlp_gated_norm.up_proj.weight": _linear_inference_tensor(stacked.mlp_gated_norm.w_up),
            "mlp.router.weight": _linear_inference_tensor(stacked.mlp.router),
            "mlp.router.bias": stacked.mlp.router_bias,
            "mlp.experts.gate_proj.weight": _linear_inference_tensor(stacked.mlp.expert_mlp.w_gate),
            "mlp.experts.up_proj.weight": _linear_inference_tensor(stacked.mlp.expert_mlp.w_up),
            "mlp.experts.down_proj.weight": _linear_inference_tensor(stacked.mlp.expert_mlp.w_down),
        }
        if stacked.shared is not None:
            layer_values.update(
                {
                    "shared_expert.gate_proj.weight": _linear_inference_tensor(stacked.shared.w_gate),
                    "shared_expert.up_proj.weight": _linear_inference_tensor(stacked.shared.w_up),
                    "shared_expert.down_proj.weight": _linear_inference_tensor(stacked.shared.w_down),
                }
            )

        for layer_index in range(self.config.num_layers):
            layer_prefix = f"model.layers.{layer_index}"
            for suffix, stacked_value in layer_values.items():
                name = f"{layer_prefix}.{suffix}"
                if include(name):
                    value = jax.lax.dynamic_index_in_dim(stacked_value, layer_index, keepdims=False)
                    add(output, name, value)
        return output


def assert_vllm_bf16(export_dir: Path, config: GrugModelConfig) -> None:
    exported_config = json.loads((export_dir / "config.json").read_text())
    assert exported_config["architectures"] == ["GrugMoeForCausalLM"]
    assert exported_config["model_type"] == "grug_moe"
    assert exported_config["dtype"] == "bfloat16"
    validate_single_name_config(exported_config, config)

    shard_paths = sorted(export_dir.glob("model-*.safetensors"))
    assert shard_paths, "Export contains no sharded safetensors weights"
    tensor_dtypes: set[str] = set()
    for shard_path in shard_paths:
        with safetensors.safe_open(shard_path, framework="numpy") as tensors:
            tensor_dtypes.update(tensors.get_slice(name).get_dtype() for name in tensors.keys())
    assert tensor_dtypes == {"BF16"}


def tree_manifest(export_dir: Path) -> tuple[str, tuple[ExportFile, ...]]:
    """Hash relative paths plus each file's SHA-256, matching the canonical tree digest."""
    digest = hashlib.sha256()
    files = []
    for path in sorted(path for path in export_dir.rglob("*") if path.is_file()):
        relative_path = path.relative_to(export_dir).as_posix()
        with path.open("rb") as file:
            file_sha256 = hashlib.file_digest(file, "sha256").hexdigest()
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256))
        files.append(ExportFile(path=relative_path, size=path.stat().st_size, sha256=file_sha256))
    return digest.hexdigest(), tuple(files)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_pinned_tokenizer(name: str, revision: str) -> PreTrainedTokenizerBase:
    """Load revision-pinned tokenizer bytes while preserving their canonical Hub identity."""
    tokenizer_dir = snapshot_download(
        name,
        revision=revision,
        allow_patterns=list(TOKENIZER_FILE_PATTERNS),
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    tokenizer.name_or_path = name
    # These are provenance hints emitted by ``save_pretrained``. They describe
    # the canonical source, not how this process accessed the pinned snapshot.
    tokenizer.init_kwargs["is_local"] = False
    tokenizer.init_kwargs["local_files_only"] = False
    return tokenizer


def _remote_files(root: StoragePath) -> dict[str, int]:
    return {path.relative_to(root): path.size() for path in (root / "**").glob() if path.isfile()}


def upload_verified_export(export_dir: Path, report: ExportReport, output_uri: str) -> ExportReport:
    """Upload only a canonical tree, without overwriting an existing prefix."""
    if not report.canonical_tree_match:
        raise ValueError(f"Refusing to upload noncanonical export {report.tree_sha256}")
    output = StoragePath(output_uri)
    expected_files = {file.path: file.size for file in report.files}
    existing_files = _remote_files(output)
    if existing_files:
        if existing_files != expected_files:
            raise FileExistsError(f"Existing export at {output} does not match the verified file manifest")
    else:
        output.upload_from(f"{export_dir}/", recursive=True)
        uploaded_files = _remote_files(output)
        if uploaded_files != expected_files:
            raise OSError(f"Uploaded export manifest mismatch at {output}")
    return dataclasses.replace(report, uploaded_uri=str(output))


def export_snowball_bf16(
    cell: NativeLevanterCell,
    *,
    scratch_root: str,
    report_uri: str,
    output_uri: str | None = None,
    goldens: tuple[RepresentativeGolden, ...] | None = None,
) -> ExportReport:
    """Export one regional checkpoint, persist its report, then optionally publish verified bytes."""
    if jax.default_backend() != cell.location.name:
        raise RuntimeError(f"Expected {cell.location.name} backend, found {jax.default_backend()}")

    goldens = read_representative_goldens() if goldens is None else goldens
    prompt_fixture = read_prompt_fixture(goldens, fixture_uri=cell.location.prompt_fixture_uri)
    executor_info = read_executor_info(cell.location)
    if executor_info["config"]["data"]["tokenizer"] != prompt_fixture.tokenizer:
        raise ValueError("Checkpoint and prompt fixture use different tokenizers")
    vendored_config = decode_vendored_config(executor_info)
    main_config = _decode_main_config(executor_info["config"]["model"])
    tokenizer = load_pinned_tokenizer(prompt_fixture.tokenizer, prompt_fixture.tokenizer_revision)

    mesh = compact_grug_mesh()
    with set_mesh(mesh):
        params, pending_qb_betas = load_checkpoint(
            vendored_config,
            mesh,
            location=cell.location,
            parameter_dtype=jnp.bfloat16 if cell.location.name == "tpu" else None,
        )
        params = prepare_bf16_parameters(params, pending_qb_betas)
        del pending_qb_betas
        parameter_digest = logical_array_digest(params)
        converter = (
            main_config.hf_checkpoint_converter()
            .replaced(tokenizer=tokenizer)
            .with_config_overrides({"dtype": "bfloat16"})
        )
        export_model = StackedSnowballExportModel(params=params, config=main_config)

        with tempfile.TemporaryDirectory(prefix="snowball-bf16-export-", dir=scratch_root) as export_dir_str:
            export_dir = Path(export_dir_str)
            converter.save_pretrained(export_model, export_dir_str, dtype=jnp.bfloat16)
            assert_vllm_bf16(export_dir, main_config)
            tree_sha256, files = tree_manifest(export_dir)
            report = ExportReport(
                platform=cell.location.name,
                logical_bf16_parameters_sha256=parameter_digest,
                executor_config_sha256=_sha256_json(executor_info["config"]),
                tokenizer=prompt_fixture.tokenizer,
                tokenizer_revision=prompt_fixture.tokenizer_revision,
                tree_sha256=tree_sha256,
                total_bytes=sum(file.size for file in files),
                files=files,
                canonical_tree_match=tree_sha256 == SNOWBALL.export_sha256,
                uploaded_uri=None,
            )
            StoragePath(report_uri).write_bytes(report.to_json_bytes())
            if not report.canonical_tree_match:
                raise AssertionError(
                    f"{cell.location.name} export tree {tree_sha256} does not match canonical {SNOWBALL.export_sha256}"
                )
            if output_uri is not None:
                report = upload_verified_export(export_dir, report, output_uri)
                StoragePath(report_uri).write_bytes(report.to_json_bytes())
            return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=("gpu", "tpu"), required=True)
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    cell = SNOWBALL_NATIVE_GPU if args.platform == "gpu" else SNOWBALL_NATIVE_TPU
    export_snowball_bf16(
        cell,
        scratch_root=args.scratch_root,
        report_uri=args.report,
        output_uri=args.output,
    )


if __name__ == "__main__":
    main()
