# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import dataclasses
import itertools
import json

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from jax.sharding import AxisType, Mesh
from levanter.checkpoint import save_checkpoint
from levanter.tensorstore_serialization import TensorStoreWriteConfig

from experiments.grug.moe_hero_ep import averaging
from experiments.grug.moe_hero_ep.averaging import (
    AverageConfig,
    CheckpointPin,
    LrSample,
    ProviderRegionProbe,
    TokenizerPin,
    average_checkpoints,
    cooldown_coefficients,
    read_prepared_provenance,
)
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer, apply_qb_betas
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_hero_ep.weights import restore_weights


class LocalRegionProbe:
    def __init__(self):
        self.worker_identity = {"provider": "local-test"}

    def worker_region(self):
        return "local-test"

    def storage_region(self, path):
        return "local-test"


def test_cooldown_coefficients_match_integrated_constant_direction_updates():
    optimizer = GrugMoeMuonHConfig(learning_rate=0.1, adam_lr=0.02, warmup=0, lr_schedule="linear", min_lr_ratio=0.1)
    schedule = optimizer.lr_scheduler(100)
    adam = optimizer.lr_scheduler(100, override_lr=optimizer.adam_lr)
    steps = (20, 40, 60)
    telemetry = tuple(LrSample(k, float(schedule(k - 1)), float(adam(k - 1))) for k in (20, 40, 50))
    result = cooldown_coefficients(steps, optimizer, 100, telemetry)
    # Every observed increment is exactly the LR sum times a fixed direction.
    snapshots = [3.0]
    for start, end in itertools.pairwise(steps):
        snapshots.append(snapshots[-1] + sum(float(schedule(k - 1)) for k in range(start + 1, end + 1)))
    expected = 3.0 + sum(float(schedule(19)) * (1 - (k - 20) / 40) for k in range(21, 61))
    assert np.dot(result.snapshots, snapshots) == pytest.approx(expected, rel=1e-7)
    shifted = tuple(dataclasses.replace(point, completed_update=point.completed_update + 1) for point in telemetry)
    with pytest.raises(ValueError, match="telemetry"):
        cooldown_coefficients(steps, optimizer, 100, shifted)


@pytest.mark.parametrize(("donors", "config_format"), [(False, "json"), (True, "yaml")])
def test_streamed_native_hero_average_roundtrip_preserves_master_and_latest_router(
    tmp_path, monkeypatch, donors, config_format
):
    model_config = GrugModelConfig(
        vocab_size=128010 if donors else 16,
        hidden_dim=8,
        intermediate_dim=4,
        shared_expert_intermediate_dim=4,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        local_kv_heads=1,
        global_kv_heads=1,
        head_dim=4,
        max_seq_len=8,
        sliding_window=4,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
    )
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1, 1),
        ("replica_dcn", "data", "context", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )
    optimizer = GrugMoeMuonHConfig(learning_rate=0.1, adam_lr=0.02, warmup=0, lr_schedule="linear")
    revision = "a" * 40
    run_config = {
        "model": draccus.encode(model_config),
        "optimizer": draccus.encode(optimizer),
        "trainer": {"trainer": {"num_train_steps": 100}},
        "data": {"tokenizer": "test-tokenizer"},
    }
    run_config = json.loads(json.dumps(run_config))
    config_path = tmp_path / f"config.{config_format}"
    # JSON scientific numbers such as layer_norm_eps=1e-5 must remain numeric.
    config_path.write_text(json.dumps(run_config) if config_format == "json" else yaml.safe_dump(run_config))
    schedule = optimizer.lr_scheduler(100)
    adam = optimizer.lr_scheduler(100, override_lr=optimizer.adam_lr)
    pins, masters = [], []
    with jax.set_mesh(mesh):
        model = Transformer.init(model_config, key=jax.random.key(7))
        for index, step in enumerate((20, 40, 60)):
            # Deliberately distinct BF16 forward params and authoritative FP32 master.
            master = jax.tree.map(lambda value, index=index: value.astype(jnp.float32) + index, model)
            masters.append(master)
            state = {
                "master_params": master,
                "params": jax.tree.map(lambda value: jnp.zeros_like(value, dtype=jnp.bfloat16), master),
                "pending_qb_betas": jnp.asarray([[index + 1.0, -index - 2.0]], dtype=jnp.float32),
            }
            path = str(tmp_path / f"step-{step}")
            save_checkpoint(
                state,
                step=step,
                checkpoint_path=path,
                is_temporary=False,
                write_config=TensorStoreWriteConfig(max_chunk_bytes=65536 if donors else 256),
            )
            metadata = json.loads((tmp_path / f"step-{step}" / "metadata.json").read_text())
            manifest = json.loads((tmp_path / f"step-{step}" / "manifest.json").read_text())
            pins.append(
                CheckpointPin(
                    path,
                    digest(metadata),
                    str(config_path),
                    digest(run_config),
                    revision,
                    digest(manifest),
                    "test fixture",
                )
            )
        if donors:

            class PinnedTokenizer:
                def encode(self, text, *, add_special_tokens):
                    mapping = {
                        "<|start_header_id|>": 128006,
                        " role": 3560,
                        "<|end_header_id|>": 128007,
                        " message": 1984,
                        "<|start_think|>": 128002,
                        " think": 1781,
                        "<|end_think|>": 128003,
                        " answer": 4320,
                        "<|eot_id|>": 128009,
                        "<|end_of_text|>": 128001,
                    }
                    return [mapping[text]]

            monkeypatch.setattr(averaging.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: PinnedTokenizer())
            pins = pins[-1:]
            masters = masters[-1:]
        config = AverageConfig(
            tuple(pins),
            str(tmp_path / "average"),
            str(tmp_path / "scratch"),
            TokenizerPin("test-tokenizer", revision),
            tuple(LrSample(k, float(schedule(k - 1)), float(adam(k - 1))) for k in (20, 40, 60)),
            source_revision="f" * 40,
            forward_context_length=8,
            max_chunk_bytes=262144 if donors else 256,
            initialize_donors=donors,
        )
        metadata = average_checkpoints(config, region_probe=LocalRegionProbe())
        restored = restore_weights(config.output, digest(metadata), model_config, mesh)
        provenance = read_prepared_provenance(config.output, digest(metadata))
        assert provenance["model"] == run_config["model"]
        assert provenance["source_pt_step"] == 60
        assert provenance["candidate"] == ("C" if donors else "B")
        if donors:
            assert provenance["selected_parent"]["metadata_digest"] == pins[0].metadata_digest
        coefficients = provenance["coefficients"]["snapshots"]
        expected = jax.tree.map(
            lambda *leaves: sum(weight * leaf for weight, leaf in zip(coefficients, leaves, strict=True)), *masters
        )
        if donors:
            donor_output = np.array(expected.output_proj)
            for destination, donor in [(128006, 3560), (128007, 1984), (128002, 1781), (128003, 4320), (128009, 128001)]:
                donor_output[:, destination] = np.asarray(expected.output_proj[:, donor])
            expected = eqx.tree_at(lambda model: model.output_proj, expected, donor_output)
        expected = apply_qb_betas(expected, state["pending_qb_betas"])
        for actual, desired in zip(jax.tree.leaves(restored), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(actual, desired, rtol=1e-6, atol=1e-6)
    assert provenance["norms"]["output_proj"]["output_l2"] > 0
    output_manifest = json.loads((tmp_path / "average" / "manifest.json").read_text())
    # Regression: global-shape tiles intersected many native expert chunks,
    # repeatedly decompressing each. Preserve the actual source storage grid.
    source_grids = {
        entry["path"]: entry["chunk_shape"] for entry in manifest["arrays"] if entry["path"].startswith("master_params/")
    }
    output_grids = {
        entry["path"]: entry["chunk_shape"]
        for entry in output_manifest["arrays"]
        if entry["path"].startswith("master_params/")
    }
    assert output_grids == source_grids
    provenance["model"]["qk_mult"] += 0.1
    (tmp_path / "average" / "averaging.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="provenance"):
        read_prepared_provenance(config.output, digest(metadata))
    assert all(region == "local-test" for region in provenance["storage_regions"].values())
    assert not any(
        "opt_state" in entry["path"]
        for entry in json.loads((tmp_path / "average" / "manifest.json").read_text())["arrays"]
    )


def test_region_mismatch_blocks_before_checkpoint_array_access(tmp_path):
    class DifferentRegion(LocalRegionProbe):
        def storage_region(self, path):
            return "other" if path.endswith("scratch") else "local-test"

    # Sources intentionally do not exist: a region failure must precede any checkpoint read.
    pins = tuple(
        CheckpointPin(str(tmp_path / name), "digest", "missing.yaml", "digest", "a" * 40, "digest", "test fixture")
        for name in ("a", "b")
    )
    config = AverageConfig(
        pins,
        str(tmp_path / "output"),
        str(tmp_path / "scratch"),
        TokenizerPin("test", "a" * 40),
        (),
        source_revision="f" * 40,
        forward_context_length=8,
    )
    with pytest.raises(ValueError, match="regions must match"):
        average_checkpoints(config, region_probe=DifferentRegion())
    assert not (tmp_path / "output").exists()


def test_kubernetes_region_uses_authenticated_bound_pod_node_instead_of_task_environment(tmp_path, monkeypatch):
    claims = {"kubernetes.io": {"namespace": "iris", "pod": {"name": "average-pod", "uid": "pod-uid"}}}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    (tmp_path / "token").write_text(f"header.{payload}.signature")
    (tmp_path / "namespace").write_text("iris")
    (tmp_path / "ca.crt").write_bytes(b"test-cluster-ca")

    class MetadataService:
        def __init__(self, *, namespace, timeout):
            assert namespace == "iris"

        def get_json(self, resource, name):
            objects = {
                (averaging.K8sResource.PODS, "average-pod"): {
                    "metadata": {"uid": "pod-uid"},
                    "spec": {"nodeName": "actual-node"},
                },
                (averaging.K8sResource.NODES, "actual-node"): {
                    "metadata": {"labels": {"topology.kubernetes.io/region": "US-EAST-02A"}},
                },
            }
            return objects.get((resource, name))

    monkeypatch.setattr(averaging, "CloudK8sService", MetadataService)
    monkeypatch.setenv("IRIS_WORKER_REGION", "incorrect-region")
    monkeypatch.setenv("IRIS_NODE_NAME", "incorrect-node")
    probe = ProviderRegionProbe(service_account_directory=tmp_path)
    assert probe.worker_region() == "us-east-02a"
    assert probe.worker_identity["node"] == "actual-node"


@pytest.mark.parametrize(
    ("host_ip", "node_access", "known_ca", "expected_region"),
    [
        ("10.184.195.193", True, True, "us-east-02a"),
        ("10.184.195.193", False, True, "us-east-02a"),
        ("10.185.195.193", False, True, None),
        ("10.184.195.193", False, False, None),
    ],
)
def test_kubernetes_region_uses_exclusive_provider_node_range_without_node_read_permission(
    tmp_path,
    monkeypatch,
    host_ip,
    node_access,
    known_ca,
    expected_region,
):
    claims = {"kubernetes.io": {"namespace": "iris", "pod": {"name": "average-pod", "uid": "pod-uid"}}}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    (tmp_path / "token").write_text(f"header.{payload}.signature")
    (tmp_path / "namespace").write_text("iris")
    certificate = b"test-cluster-ca"
    (tmp_path / "ca.crt").write_bytes(certificate)
    # Public provider geography facts are the external metadata boundary. Bind
    # this local fake API's CA to the same real provider-published node range.
    monkeypatch.setattr(
        averaging,
        "_COREWEAVE_NODE_LOCATIONS",
        (
            averaging.PublishedNodeLocation(
                region="us-east-02a",
                topology_region="us-east-02",
                node_cidr="10.184.192.0/18",
                cluster_ca_sha256=averaging.hashlib.sha256(certificate).hexdigest() if known_ca else "unknown-ca",
                source_url="https://docs.coreweave.com/platform/regions/us-east/us-east-02#network-information",
                verified_on="2026-10-19",
            ),
        ),
    )

    class MetadataService:
        def __init__(self, *, namespace, timeout):
            pass

        def get_json(self, resource, name):
            if resource == averaging.K8sResource.PODS:
                return {
                    "metadata": {"uid": "pod-uid"},
                    "spec": {"nodeName": "actual-node"},
                    "status": {"hostIP": host_ip},
                }
            if not node_access:
                raise averaging.KubectlError("Node metadata denied") from averaging.ApiException(status=403)
            return {"metadata": {"labels": {"topology.kubernetes.io/region": "US-EAST-02"}}}

    monkeypatch.setattr(averaging, "CloudK8sService", MetadataService)
    probe = ProviderRegionProbe(service_account_directory=tmp_path)
    if expected_region is None:
        with pytest.raises(ValueError, match="No unambiguous provider-published AZ mapping"):
            probe.worker_region()
        return
    assert probe.worker_region() == expected_region
    assert probe.worker_identity["host_ip"] == host_ip
    assert probe.worker_identity["node_cidr"] == "10.184.192.0/18"
    assert probe.worker_identity["pod_uid"] == "pod-uid"
