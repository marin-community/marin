"""Unit tests for GKEEnvironment GPU and TPU support.

Covers the GPU- and TPU-specific capability flags, the GKE_GPU_TYPE_MAP
and GKE_TPU_TYPE_MAP constants, and pod-spec construction (resource
requests/limits, node selectors, tolerations) when
task_env_config.gpus > 0 or task_env_config.tpu is not None.
"""

import asyncio
import io
import tarfile
import threading
import tracemalloc
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes import client as k8s_client

from harbor.environments import gke as gke_module
from pydantic import ValidationError

from harbor.environments.gke import (
    GKE_GPU_TYPE_MAP,
    GKE_TPU_TYPE_MAP,
    GKEEnvironment,
)
from harbor.models.task.config import EnvironmentConfig, TpuSpec
from harbor.models.trial.paths import TrialPaths


async def _start_and_capture_pod(gke_env):
    """Run GKEEnvironment.start() with all external calls mocked and
    return the V1Pod that was passed to create_namespaced_pod.

    Shared by both the GPU and TPU pod-spec test classes: the harness is
    accelerator-agnostic — what differs between tests is only the
    EnvironmentConfig baked into gke_env.
    """
    captured_pods: list = []

    def capture_create_pod(namespace, body):
        captured_pods.append(body)

    mock_api = MagicMock(spec=k8s_client.CoreV1Api)
    mock_api.create_namespaced_pod.side_effect = capture_create_pod
    mock_api.read_namespaced_pod.return_value = MagicMock(
        status=MagicMock(
            phase="Running",
            container_statuses=[MagicMock(ready=True)],
        )
    )

    gke_env._core_api = mock_api
    gke_env._client_manager = MagicMock()
    gke_env._image_exists = AsyncMock(return_value=True)
    gke_env._wait_for_container_exec_ready = AsyncMock()
    gke_env.exec = AsyncMock(
        return_value=MagicMock(return_code=0, stdout="", stderr="")
    )

    await gke_env.start(force_build=False)
    assert len(captured_pods) == 1
    return captured_pods[0]


def _make_gke_env(temp_dir, dockerfile_content, *, suffix="", **env_config_kwargs):
    """Create a GKEEnvironment with the given Dockerfile and overrides."""
    env_dir = temp_dir / f"environment{suffix}"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(dockerfile_content)

    trial_dir = temp_dir / f"trial{suffix}"
    trial_dir.mkdir(exist_ok=True)
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    defaults: dict = {"cpus": 2, "memory_mb": 4096, "storage_mb": 10240}
    defaults.update(env_config_kwargs)

    return GKEEnvironment(
        environment_dir=env_dir,
        environment_name=f"test-task{suffix}",
        session_id=f"test-task{suffix}__abc123",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(**defaults),
        cluster_name="test-cluster",
        region="us-central1",
        namespace="default",
        registry_location="us-central1",
        registry_name="test-images",
        project_id="test-project",
    )


@pytest.fixture
def gke_env(temp_dir):
    """A minimal GKEEnvironment without GPUs."""
    return _make_gke_env(temp_dir, "FROM ubuntu:24.04\n")


@pytest.fixture
def gke_env_gpu(temp_dir):
    """A GKEEnvironment requesting 1x H100 with a memory limit."""
    env_dir = temp_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM nvidia/cuda:12.4.0-base-ubuntu22.04\n")

    trial_dir = temp_dir / "trial"
    trial_dir.mkdir()
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    return GKEEnvironment(
        environment_dir=env_dir,
        environment_name="gpu-task",
        session_id="gpu-task__xyz789",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(
            cpus=4,
            memory_mb=16384,
            storage_mb=20480,
            gpus=1,
            gpu_types=["H100"],
        ),
        cluster_name="test-cluster",
        region="us-central1",
        namespace="default",
        registry_location="us-central1",
        registry_name="test-images",
        project_id="test-project",
        memory_limit_multiplier=1.0,
    )


@pytest.fixture
def gke_env_multi_gpu(temp_dir):
    """A GKEEnvironment requesting 4x A100s."""
    return _make_gke_env(
        temp_dir,
        "FROM ubuntu:24.04\n",
        suffix="-multi",
        cpus=8,
        memory_mb=65536,
        storage_mb=102400,
        gpus=4,
        gpu_types=["A100"],
    )


class TestGKECapabilitiesGPU:
    """The GKE environment advertises GPU capability."""

    def test_capabilities_gpus_is_true(self, gke_env):
        assert gke_env.capabilities.gpus is True

    def test_gpu_env_config_preserved(self, gke_env_gpu):
        assert gke_env_gpu.task_env_config.gpus == 1
        assert gke_env_gpu.task_env_config.gpu_types == ["H100"]


class TestGKEGPUTypeMap:
    """The GKE_GPU_TYPE_MAP exposes the expected user-friendly aliases."""

    def test_common_gpu_types_mapped(self):
        assert GKE_GPU_TYPE_MAP["t4"] == "nvidia-tesla-t4"
        assert GKE_GPU_TYPE_MAP["l4"] == "nvidia-l4"
        assert GKE_GPU_TYPE_MAP["a100"] == "nvidia-tesla-a100"
        assert GKE_GPU_TYPE_MAP["h100"] == "nvidia-h100-80gb"

    def test_variant_gpu_types_mapped(self):
        # A100 has both 40GB and 80GB SKUs that map to *different* GKE
        # labels, so both aliases need to live in the map.
        assert GKE_GPU_TYPE_MAP["a100-40gb"] == "nvidia-tesla-a100"
        assert GKE_GPU_TYPE_MAP["a100-80gb"] == "nvidia-a100-80gb"

    def test_high_end_gpu_types_mapped(self):
        # H100 Mega, H200, B200, GB200, and RTX PRO 6000 are all
        # currently-listed GKE accelerator SKUs.
        assert GKE_GPU_TYPE_MAP["h100-mega"] == "nvidia-h100-mega-80gb"
        assert GKE_GPU_TYPE_MAP["h200"] == "nvidia-h200-141gb"
        assert GKE_GPU_TYPE_MAP["b200"] == "nvidia-b200"
        assert GKE_GPU_TYPE_MAP["gb200"] == "nvidia-gb200"
        assert GKE_GPU_TYPE_MAP["rtx-pro-6000"] == "nvidia-rtx-pro-6000"

    def test_redundant_long_form_aliases_omitted(self):
        # Where the long-form alias would map to the same GKE label as the
        # bare alias (e.g. 'h100-80gb' == 'h100' → 'nvidia-h100-80gb'), the
        # long form is intentionally NOT in the map — users who really want
        # to type it can pass the canonical GKE label directly via the
        # canonical-label passthrough in _resolve_gpu_accelerator_label.
        assert "h100-80gb" not in GKE_GPU_TYPE_MAP
        assert "h100-mega-80gb" not in GKE_GPU_TYPE_MAP
        assert "h200-141gb" not in GKE_GPU_TYPE_MAP

    def test_modal_only_skus_not_silently_advertised(self):
        # A10 and L40S exist on Modal but not on GKE. They must not appear
        # in the map (and therefore must raise at construction time) so
        # users don't discover the mismatch at pod-scheduling time.
        assert "a10" not in GKE_GPU_TYPE_MAP
        assert "l40s" not in GKE_GPU_TYPE_MAP

    def test_all_keys_are_lowercase(self):
        for key in GKE_GPU_TYPE_MAP:
            assert key == key.lower(), f"Key '{key}' should be lowercase"

    def test_all_values_are_valid_gke_labels(self):
        # Sanity-check: every value should look like a GKE accelerator
        # label (nvidia-* per the official supported list).
        for alias, label in GKE_GPU_TYPE_MAP.items():
            assert label.startswith("nvidia-"), (
                f"Alias '{alias}' maps to '{label}', which doesn't look like "
                "a GKE accelerator label (expected to start with 'nvidia-')."
            )


class TestGKEPodSpecGPU:
    """start() constructs the pod spec correctly for GPU and CPU pods."""

    async def test_no_gpu_pod_spec(self, gke_env):
        """CPU-only pod has no GPU/TPU resources, node selector, or tolerations."""
        pod = await _start_and_capture_pod(gke_env)

        container = pod.spec.containers[0]
        requests = container.resources.requests
        limits = container.resources.limits

        assert "nvidia.com/gpu" not in requests
        assert "google.com/tpu" not in requests
        assert limits is None
        assert pod.spec.node_selector is None
        assert pod.spec.tolerations is None

    async def test_gpu_resource_requests_and_limits(self, gke_env_gpu):
        """GPU pod requests and limits both set nvidia.com/gpu."""
        pod = await _start_and_capture_pod(gke_env_gpu)

        container = pod.spec.containers[0]
        assert container.resources.requests["nvidia.com/gpu"] == "1"
        assert container.resources.limits["nvidia.com/gpu"] == "1"

    async def test_gpu_node_selector(self, gke_env_gpu):
        """GPU pod targets the right accelerator label."""
        pod = await _start_and_capture_pod(gke_env_gpu)

        assert pod.spec.node_selector is not None
        assert (
            pod.spec.node_selector["cloud.google.com/gke-accelerator"]
            == "nvidia-h100-80gb"
        )

    async def test_gpu_tolerations(self, gke_env_gpu):
        """GPU pod gets the standard nvidia.com/gpu NoSchedule toleration."""
        pod = await _start_and_capture_pod(gke_env_gpu)

        assert pod.spec.tolerations is not None
        assert len(pod.spec.tolerations) == 1
        tol = pod.spec.tolerations[0]
        assert tol.key == "nvidia.com/gpu"
        assert tol.operator == "Exists"
        assert tol.effect == "NoSchedule"

    async def test_multi_gpu_count(self, gke_env_multi_gpu):
        """Multi-GPU pod requests the correct count."""
        pod = await _start_and_capture_pod(gke_env_multi_gpu)

        container = pod.spec.containers[0]
        assert container.resources.requests["nvidia.com/gpu"] == "4"
        assert container.resources.limits["nvidia.com/gpu"] == "4"

    async def test_multi_gpu_node_selector_uses_a100(self, gke_env_multi_gpu):
        """Multi-GPU A100 pod targets nvidia-tesla-a100."""
        pod = await _start_and_capture_pod(gke_env_multi_gpu)

        assert (
            pod.spec.node_selector["cloud.google.com/gke-accelerator"]
            == "nvidia-tesla-a100"
        )

    async def test_gpu_memory_limit_still_set(self, gke_env_gpu):
        """memory_limit_multiplier still propagates to the GPU pod's limits."""
        pod = await _start_and_capture_pod(gke_env_gpu)

        container = pod.spec.containers[0]
        assert container.resources.limits["memory"] == "16384Mi"

    async def test_gpu_no_type_specified(self, temp_dir):
        """GPU pod without gpu_types still gets resources + tolerations but no node selector."""
        env = _make_gke_env(
            temp_dir,
            "FROM ubuntu:24.04\n",
            suffix="-notype",
            cpus=2,
            memory_mb=8192,
            storage_mb=10240,
            gpus=1,
        )

        pod = await _start_and_capture_pod(env)

        container = pod.spec.containers[0]
        assert container.resources.requests["nvidia.com/gpu"] == "1"
        assert container.resources.limits["nvidia.com/gpu"] == "1"
        assert pod.spec.node_selector is None
        assert pod.spec.tolerations is not None

    def test_unsupported_gpu_type_raises_error_at_construction(self, temp_dir):
        """An unsupported GPU type fails fast at __init__ — before start() runs
        the (slow, retried) image build pipeline."""
        with pytest.raises(RuntimeError, match="not supported on GKE"):
            _make_gke_env(
                temp_dir,
                "FROM ubuntu:24.04\n",
                suffix="-unknown",
                cpus=2,
                memory_mb=8192,
                storage_mb=10240,
                gpus=1,
                gpu_types=["L40S"],
            )

    def test_unsupported_gpu_type_skips_image_build(self, temp_dir, monkeypatch):
        """Eager validation must short-circuit before _build_and_push_image
        is ever invoked (the original bug: a typo would burn ~40 min of
        Cloud Build before surfacing)."""
        build_calls: list = []

        async def _fake_build(self):
            build_calls.append(self)

        monkeypatch.setattr(
            GKEEnvironment, "_build_and_push_image", _fake_build, raising=True
        )

        with pytest.raises(RuntimeError, match="not supported on GKE"):
            _make_gke_env(
                temp_dir,
                "FROM ubuntu:24.04\n",
                suffix="-no-build",
                cpus=2,
                memory_mb=8192,
                storage_mb=10240,
                gpus=1,
                gpu_types=["definitely-not-a-real-gpu"],
            )

        assert build_calls == [], (
            "Image build was triggered for an invalid GPU type — eager "
            "validation should fail before reaching _build_and_push_image."
        )

    async def test_gpu_type_matching_is_case_insensitive(self, temp_dir):
        """Mixed-case GPU type strings are normalized to the map keys."""
        env = _make_gke_env(
            temp_dir,
            "FROM ubuntu:24.04\n",
            suffix="-case",
            cpus=2,
            memory_mb=8192,
            storage_mb=10240,
            gpus=1,
            gpu_types=["  H100  "],
        )

        pod = await _start_and_capture_pod(env)

        assert (
            pod.spec.node_selector["cloud.google.com/gke-accelerator"]
            == "nvidia-h100-80gb"
        )

    async def test_canonical_gke_label_passthrough_in_pod_spec(self, temp_dir):
        """A canonical GKE label (a map *value*) passes through unchanged
        to the node selector — users can supply 'nvidia-h100-80gb'
        directly instead of going through the 'h100' alias."""
        env = _make_gke_env(
            temp_dir,
            "FROM ubuntu:24.04\n",
            suffix="-canonical",
            cpus=2,
            memory_mb=8192,
            storage_mb=10240,
            gpus=1,
            gpu_types=["nvidia-h100-80gb"],
        )

        pod = await _start_and_capture_pod(env)

        assert (
            pod.spec.node_selector["cloud.google.com/gke-accelerator"]
            == "nvidia-h100-80gb"
        )

    def test_canonical_gke_label_accepted_at_construction(self, temp_dir):
        """Eager __init__ validation accepts canonical labels too — no
        RuntimeError when the user supplies a valid map value directly."""
        env = _make_gke_env(
            temp_dir,
            "FROM ubuntu:24.04\n",
            suffix="-canonical-init",
            cpus=2,
            memory_mb=8192,
            storage_mb=10240,
            gpus=1,
            gpu_types=["nvidia-rtx-pro-6000"],
        )
        assert env.task_env_config.gpu_types == ["nvidia-rtx-pro-6000"]

    async def test_canonical_gke_label_is_case_insensitive(self, temp_dir):
        """Canonical labels also get the lowercased/stripped treatment so
        'NVIDIA-H100-80GB' resolves to 'nvidia-h100-80gb'."""
        env = _make_gke_env(
            temp_dir,
            "FROM ubuntu:24.04\n",
            suffix="-canonical-case",
            cpus=2,
            memory_mb=8192,
            storage_mb=10240,
            gpus=1,
            gpu_types=["  NVIDIA-H100-80GB  "],
        )

        pod = await _start_and_capture_pod(env)

        assert (
            pod.spec.node_selector["cloud.google.com/gke-accelerator"]
            == "nvidia-h100-80gb"
        )


@pytest.fixture
def gke_env_tpu(temp_dir):
    """A GKEEnvironment requesting a v4 TPU slice with topology 2x2x1 (4 chips)."""
    return _make_gke_env(
        temp_dir,
        "FROM ubuntu:24.04\n",
        suffix="-tpu",
        cpus=4,
        memory_mb=16384,
        storage_mb=20480,
        tpu=TpuSpec(type="v4", topology="2x2x1"),
    )


class TestGKECapabilitiesTPU:
    """The GKE environment advertises TPU capability."""

    def test_capabilities_tpus_is_true(self, gke_env):
        assert gke_env.capabilities.tpus is True

    def test_tpu_env_config_preserved(self, gke_env_tpu):
        tpu = gke_env_tpu.task_env_config.tpu
        assert tpu is not None
        assert tpu.type == "v4"
        assert tpu.topology == "2x2x1"
        assert tpu.chip_count == 4


class TestTpuSpec:
    """TpuSpec validates inputs and derives chip_count from topology."""

    def test_basic_2d_topology_chip_count(self):
        assert TpuSpec(type="v6e", topology="2x4").chip_count == 8

    def test_basic_3d_topology_chip_count(self):
        assert TpuSpec(type="v4", topology="2x2x1").chip_count == 4

    def test_single_chip_topology(self):
        assert TpuSpec(type="v5e", topology="1x1").chip_count == 1

    def test_larger_topology_chip_count(self):
        assert TpuSpec(type="v5p", topology="4x4x4").chip_count == 64

    def test_topology_whitespace_is_trimmed(self):
        assert TpuSpec(type="v4", topology="  2x2x1  ").topology == "2x2x1"

    def test_missing_topology_rejected(self):
        # 'topology' is required: omitting it would let GKE pick an implicit
        # default that's not part of any stable contract.
        with pytest.raises(ValidationError):
            TpuSpec.model_validate({"type": "v4"})

    def test_missing_type_rejected(self):
        with pytest.raises(ValidationError):
            TpuSpec.model_validate({"topology": "2x2x1"})

    def test_empty_type_rejected(self):
        with pytest.raises(ValidationError):
            TpuSpec(type="", topology="2x2x1")

    @pytest.mark.parametrize(
        "bad_topology",
        ["", "2", "2x", "x2", "2x2x", "2xx2", "2,2", "2 x 2", "2X2", "a x b"],
    )
    def test_invalid_topology_format_rejected(self, bad_topology):
        with pytest.raises(ValidationError, match="Invalid TPU topology"):
            TpuSpec(type="v4", topology=bad_topology)

    @pytest.mark.parametrize(
        "bad_topology",
        ["0x4", "4x0", "0x0", "2x0x2", "0x2x2", "02x4", "2x04", "2x4x00"],
    )
    def test_zero_or_leading_zero_dimensions_rejected(self, bad_topology):
        # Each dimension must be a *positive* integer. A zero dimension
        # would slip through math.prod as 0 and produce a nonsensical
        # google.com/tpu = "0" pod request that GKE would either fail
        # to schedule or schedule onto a non-TPU node — with no signal
        # back to the bad topology. Leading zeros are caught for the
        # same reason: '02x4' parses to chip_count=8 today but reads
        # like an off-by-one bug in the operator's task.toml, so we
        # require canonical form.
        with pytest.raises(ValidationError, match="Invalid TPU topology"):
            TpuSpec(type="v4", topology=bad_topology)


class TestEnvironmentConfigTPU:
    """EnvironmentConfig accepts an optional single TpuSpec."""

    def test_no_tpu_by_default(self):
        cfg = EnvironmentConfig()
        assert cfg.tpu is None

    def test_single_spec_round_trips(self):
        cfg = EnvironmentConfig(tpu=TpuSpec(type="v4", topology="2x2x1"))
        assert cfg.tpu is not None
        assert cfg.tpu.type == "v4"
        assert cfg.tpu.topology == "2x2x1"
        assert cfg.tpu.chip_count == 4

    def test_tpu_spec_constructible_from_dict(self):
        # Mirrors how the spec lands at runtime: parsed from a
        # [environment.tpu] sub-table in task.toml. Use model_validate
        # so the test exercises the same code path that TOML parsing
        # takes.
        cfg = EnvironmentConfig.model_validate(
            {"tpu": {"type": "v6e", "topology": "2x4"}}
        )
        assert cfg.tpu is not None
        assert cfg.tpu.chip_count == 8

    def test_list_payload_rejected(self):
        # Defensive regression: TOML's [[environment.tpus]] (array of
        # tables) used to be the accepted shape. After collapsing to a
        # single TpuSpec we want loud failure rather than silently
        # taking the first entry.
        with pytest.raises(ValidationError):
            EnvironmentConfig.model_validate(
                {"tpu": [{"type": "v6e", "topology": "2x4"}]}
            )


class TestGKETPUTypeMap:
    """The GKE_TPU_TYPE_MAP exposes the expected user-friendly aliases."""

    def test_short_family_aliases(self):
        assert GKE_TPU_TYPE_MAP["v3"] == "tpu-v3-slice"
        assert GKE_TPU_TYPE_MAP["v3-device"] == "tpu-v3-device"
        assert GKE_TPU_TYPE_MAP["v4"] == "tpu-v4-podslice"
        assert GKE_TPU_TYPE_MAP["v5e"] == "tpu-v5-lite-podslice"
        assert GKE_TPU_TYPE_MAP["v5p"] == "tpu-v5p-slice"
        assert GKE_TPU_TYPE_MAP["v6e"] == "tpu-v6e-slice"
        assert GKE_TPU_TYPE_MAP["v7"] == "tpu7x"

    def test_marketing_name_aliases(self):
        assert GKE_TPU_TYPE_MAP["trillium"] == "tpu-v6e-slice"
        assert GKE_TPU_TYPE_MAP["ironwood"] == "tpu7x"

    def test_canonical_labels_present_as_values(self):
        # Canonical GKE labels are not keys in the map (the map is pure
        # aliases) but they are values, so the start() validation can
        # accept a canonical label directly via a values() lookup.
        for label in [
            "tpu-v3-slice",
            "tpu-v3-device",
            "tpu-v4-podslice",
            "tpu-v5-lite-podslice",
            "tpu-v5p-slice",
            "tpu-v6e-slice",
            "tpu7x",
        ]:
            assert label in GKE_TPU_TYPE_MAP.values()
            assert label not in GKE_TPU_TYPE_MAP

    def test_all_keys_are_lowercase(self):
        for key in GKE_TPU_TYPE_MAP:
            assert key == key.lower(), f"Key '{key}' should be lowercase"


class TestGKEPodSpecTPU:
    """start() constructs the pod spec correctly for TPU pods."""

    async def test_tpu_resource_requests_and_limits(self, gke_env_tpu):
        """TPU pod requests and limits both set google.com/tpu."""
        pod = await _start_and_capture_pod(gke_env_tpu)

        container = pod.spec.containers[0]
        assert container.resources.requests["google.com/tpu"] == "4"
        assert container.resources.limits["google.com/tpu"] == "4"

    async def test_tpu_node_selectors(self, gke_env_tpu):
        """TPU pod sets both accelerator and topology node selectors."""
        pod = await _start_and_capture_pod(gke_env_tpu)

        assert pod.spec.node_selector is not None
        assert (
            pod.spec.node_selector["cloud.google.com/gke-tpu-accelerator"]
            == "tpu-v4-podslice"
        )
        assert pod.spec.node_selector["cloud.google.com/gke-tpu-topology"] == "2x2x1"

    async def test_tpu_tolerations(self, gke_env_tpu):
        """TPU pod gets the standard google.com/tpu NoSchedule toleration."""
        pod = await _start_and_capture_pod(gke_env_tpu)

        assert pod.spec.tolerations is not None
        assert len(pod.spec.tolerations) == 1
        tol = pod.spec.tolerations[0]
        assert tol.key == "google.com/tpu"
        assert tol.operator == "Exists"
        assert tol.effect == "NoSchedule"

    async def test_tpu_pod_has_no_gpu_resources(self, gke_env_tpu):
        """TPU pod does not request GPU resources."""
        pod = await _start_and_capture_pod(gke_env_tpu)

        container = pod.spec.containers[0]
        assert "nvidia.com/gpu" not in container.resources.requests
        assert "nvidia.com/gpu" not in (container.resources.limits or {})

    async def test_tpu_canonical_label_passthrough(self, temp_dir):
        """Canonical GKE TPU label (e.g. 'tpu-v6e-slice') passes through unchanged.

        Also exercises chip-count derivation: topology '2x4' → 8 chips.
        """
        env = _make_gke_env(
            temp_dir,
            "FROM ubuntu:24.04\n",
            suffix="-tpu-canonical",
            cpus=2,
            memory_mb=8192,
            storage_mb=10240,
            tpu=TpuSpec(type="tpu-v6e-slice", topology="2x4"),
        )

        pod = await _start_and_capture_pod(env)

        container = pod.spec.containers[0]
        assert container.resources.requests["google.com/tpu"] == "8"
        assert container.resources.limits["google.com/tpu"] == "8"
        assert (
            pod.spec.node_selector["cloud.google.com/gke-tpu-accelerator"]
            == "tpu-v6e-slice"
        )
        assert pod.spec.node_selector["cloud.google.com/gke-tpu-topology"] == "2x4"

    async def test_tpu_canonical_label_that_is_only_a_value(self, temp_dir):
        """A canonical label like 'tpu7x' (not a key in the map) is still accepted via values() lookup."""
        env = _make_gke_env(
            temp_dir,
            "FROM ubuntu:24.04\n",
            suffix="-tpu-only-value",
            cpus=2,
            memory_mb=8192,
            storage_mb=10240,
            tpu=TpuSpec(type="tpu7x", topology="2x2"),
        )

        pod = await _start_and_capture_pod(env)

        assert pod.spec.node_selector["cloud.google.com/gke-tpu-accelerator"] == "tpu7x"
        assert pod.spec.node_selector["cloud.google.com/gke-tpu-topology"] == "2x2"

    async def test_tpu_chip_count_derived_from_topology(self, temp_dir):
        """google.com/tpu request/limit must equal product(topology) — there
        is no independent chip-count input, only the topology."""
        env = _make_gke_env(
            temp_dir,
            "FROM ubuntu:24.04\n",
            suffix="-tpu-chips",
            cpus=2,
            memory_mb=8192,
            storage_mb=10240,
            tpu=TpuSpec(type="v5p", topology="4x4x4"),
        )

        pod = await _start_and_capture_pod(env)

        container = pod.spec.containers[0]
        assert container.resources.requests["google.com/tpu"] == "64"
        assert container.resources.limits["google.com/tpu"] == "64"

    def test_unsupported_tpu_type_raises_error_at_construction(self, temp_dir):
        """An unsupported TPU type fails fast at __init__ — before start() runs
        the (slow, retried) image build pipeline."""
        with pytest.raises(RuntimeError, match="not supported on GKE"):
            _make_gke_env(
                temp_dir,
                "FROM ubuntu:24.04\n",
                suffix="-tpu-unknown",
                cpus=2,
                memory_mb=8192,
                storage_mb=10240,
                tpu=TpuSpec(type="tpu-v99-future", topology="2x2"),
            )

    def test_unsupported_tpu_type_skips_image_build(self, temp_dir, monkeypatch):
        """Eager validation must short-circuit before _build_and_push_image
        is ever invoked (symmetric with the GPU branch's regression test)."""
        build_calls: list = []

        async def _fake_build(self):
            build_calls.append(self)

        monkeypatch.setattr(
            GKEEnvironment, "_build_and_push_image", _fake_build, raising=True
        )

        with pytest.raises(RuntimeError, match="not supported on GKE"):
            _make_gke_env(
                temp_dir,
                "FROM ubuntu:24.04\n",
                suffix="-tpu-no-build",
                cpus=2,
                memory_mb=8192,
                storage_mb=10240,
                tpu=TpuSpec(type="definitely-not-a-real-tpu", topology="2x2"),
            )

        assert build_calls == [], (
            "Image build was triggered for an invalid TPU type — eager "
            "validation should fail before reaching _build_and_push_image."
        )

    async def test_tpu_type_matching_is_case_insensitive(self, temp_dir):
        """Mixed-case TPU type strings are normalized to the map keys."""
        env = _make_gke_env(
            temp_dir,
            "FROM ubuntu:24.04\n",
            suffix="-tpu-case",
            cpus=2,
            memory_mb=8192,
            storage_mb=10240,
            tpu=TpuSpec(type="  V4  ", topology="2x2x1"),
        )

        pod = await _start_and_capture_pod(env)

        assert (
            pod.spec.node_selector["cloud.google.com/gke-tpu-accelerator"]
            == "tpu-v4-podslice"
        )


class TestGKEAcceleratorMutualExclusion:
    """A single GKE pod can only target one accelerator family via
    nodeSelector (cloud.google.com/gke-accelerator vs
    cloud.google.com/gke-tpu-accelerator). Requesting both would
    produce a pod that can never be scheduled — eager validation must
    catch this at construction time."""

    def test_gpu_and_tpu_together_rejected_at_construction(self, temp_dir):
        with pytest.raises(RuntimeError, match="one accelerator family per pod"):
            _make_gke_env(
                temp_dir,
                "FROM ubuntu:24.04\n",
                suffix="-mutex",
                cpus=4,
                memory_mb=16384,
                storage_mb=20480,
                gpus=1,
                gpu_types=["h100"],
                tpu=TpuSpec(type="v6e", topology="2x4"),
            )

    def test_gpu_without_type_still_conflicts_with_tpu(self, temp_dir):
        """Conflict is about the resource request (gpus > 0), not about
        whether a specific GPU type was named — a 'gpu_types is None'
        run still has the same nodeSelector clash."""
        with pytest.raises(RuntimeError, match="one accelerator family per pod"):
            _make_gke_env(
                temp_dir,
                "FROM ubuntu:24.04\n",
                suffix="-mutex-untyped",
                cpus=4,
                memory_mb=16384,
                storage_mb=20480,
                gpus=1,
                tpu=TpuSpec(type="v4", topology="2x2x1"),
            )

    def test_mutex_check_skips_image_build(self, temp_dir, monkeypatch):
        """Like the unsupported-type checks, the mutex check must short-
        circuit before any image build kicks off."""
        build_calls: list = []

        async def _fake_build(self):
            build_calls.append(self)

        monkeypatch.setattr(
            GKEEnvironment, "_build_and_push_image", _fake_build, raising=True
        )

        with pytest.raises(RuntimeError, match="one accelerator family per pod"):
            _make_gke_env(
                temp_dir,
                "FROM ubuntu:24.04\n",
                suffix="-mutex-no-build",
                cpus=2,
                memory_mb=8192,
                storage_mb=10240,
                gpus=1,
                gpu_types=["t4"],
                tpu=TpuSpec(type="v4", topology="2x2x1"),
            )

        assert build_calls == [], (
            "Image build was triggered for a GPU+TPU conflict — eager "
            "validation should fail before reaching _build_and_push_image."
        )

    def test_gpu_only_still_allowed(self, temp_dir):
        """Sanity check: the mutex guard must not over-fire on the
        common single-accelerator case."""
        env = _make_gke_env(
            temp_dir,
            "FROM ubuntu:24.04\n",
            suffix="-mutex-gpu-only",
            cpus=2,
            memory_mb=8192,
            storage_mb=10240,
            gpus=1,
            gpu_types=["h100"],
        )
        assert env.task_env_config.gpus == 1
        assert env.task_env_config.tpu is None

    def test_tpu_only_still_allowed(self, temp_dir):
        env = _make_gke_env(
            temp_dir,
            "FROM ubuntu:24.04\n",
            suffix="-mutex-tpu-only",
            cpus=2,
            memory_mb=8192,
            storage_mb=10240,
            tpu=TpuSpec(type="v6e", topology="2x4"),
        )
        assert env._effective_gpus == 0
        assert env.task_env_config.tpu is not None


class _FakeExecResponse:
    """Minimal stand-in for the WSClient that stream() returns.

    Implements only what GKEEnvironment.exec() touches.
    """

    returncode = 0

    def __init__(self, stdout_chunks=(), stderr_chunks=()):
        self.closed = False
        self.stdout_chunks = list(stdout_chunks)
        self.stderr_chunks = list(stderr_chunks)

    def is_open(self):
        return bool(self.stdout_chunks or self.stderr_chunks)

    def update(self, timeout):
        pass

    def peek_stdout(self):
        return bool(self.stdout_chunks)

    def read_stdout(self):
        return self.stdout_chunks.pop(0)

    def peek_stderr(self):
        return bool(self.stderr_chunks)

    def read_stderr(self):
        return self.stderr_chunks.pop(0)

    def run_forever(self, timeout=0):
        pass

    def close(self):
        self.closed = True


class _BlockingTransferResponse(_FakeExecResponse):
    def __init__(self, loop, started, release, stdout_chunks=()):
        super().__init__(stdout_chunks=stdout_chunks)
        self._loop = loop
        self._started = started
        self._release = release
        self._blocked = False

    def _block_once(self):
        if self._blocked:
            return
        self._blocked = True
        self._loop.call_soon_threadsafe(self._started.set)
        if not self._release.wait(timeout=2):
            raise TimeoutError("event loop could not release websocket operation")

    def update(self, timeout):
        self._block_once()

    def write_stdin(self, data):
        self._block_once()


def _make_client_env(*, core_api=None, client_manager=None) -> GKEEnvironment:
    """Build a GKEEnvironment with only the attributes the client paths read.

    __init__ wants task files and resource config that are irrelevant to client
    ownership, so bypass it.
    """
    env = object.__new__(GKEEnvironment)
    env.cluster_name = "test-cluster"
    env.region = "us-central1"
    env.project_id = "test-project"
    env.namespace = "default"
    env.pod_name = "test-pod"
    env._client_manager = client_manager
    env._core_api = core_api
    env.default_user = None
    env._persistent_env = {}
    env.task_env_config = SimpleNamespace(workdir=None)
    env.logger = MagicMock()
    return env


class TestGKEExecReaderLifecycle:
    @pytest.mark.asyncio
    async def test_high_concurrency_exec_readers_all_start(self):
        configuration = k8s_client.Configuration()
        rest_client = k8s_client.ApiClient(configuration)
        env = _make_client_env(
            core_api=k8s_client.CoreV1Api(rest_client), client_manager=MagicMock()
        )

        concurrency = 40
        responses = [MagicMock(returncode=0) for _ in range(concurrency)]
        responses_lock = threading.Lock()
        readers = threading.Barrier(concurrency, timeout=5)

        def fake_stream(*args, **kwargs):
            with responses_lock:
                return responses.pop()

        def read_output(_self: GKEEnvironment, _response):
            readers.wait()
            return "", ""

        try:
            with (
                patch.object(gke_module, "stream", side_effect=fake_stream),
                patch.object(GKEEnvironment, "_read_exec_output", read_output),
            ):
                async with asyncio.timeout(10):
                    async with asyncio.TaskGroup() as task_group:
                        tasks = [
                            task_group.create_task(env.exec(f"command-{index}"))
                            for index in range(concurrency)
                        ]
                    results = [task.result() for task in tasks]
        finally:
            rest_client.close()

        assert [result.return_code for result in results] == [0] * concurrency

    @pytest.mark.parametrize("termination", ["timeout", "cancellation"])
    @pytest.mark.asyncio
    async def test_stuck_reader_does_not_delay_termination(self, termination):
        configuration = k8s_client.Configuration()
        rest_client = k8s_client.ApiClient(configuration)
        env = _make_client_env(
            core_api=k8s_client.CoreV1Api(rest_client), client_manager=MagicMock()
        )
        response = MagicMock(returncode=0)
        loop = asyncio.get_running_loop()
        reader_started = asyncio.Event()
        release_reader = threading.Event()
        reader_finished = asyncio.Event()
        close_count = 0

        def read_output(_self: GKEEnvironment, _response):
            loop.call_soon_threadsafe(reader_started.set)
            release_reader.wait(timeout=5)
            loop.call_soon_threadsafe(reader_finished.set)
            return "", ""

        def close_response():
            nonlocal close_count
            close_count += 1

        response.sock.abort.side_effect = OSError("abort unavailable")
        response.sock.sock.shutdown.return_value = None
        response.close.side_effect = close_response

        try:
            with (
                patch.object(gke_module, "stream", return_value=response),
                patch.object(GKEEnvironment, "_read_exec_output", read_output),
            ):
                if termination == "timeout":
                    task = asyncio.create_task(env.exec("sleep 600", timeout_sec=1))
                    await asyncio.wait_for(reader_started.wait(), timeout=1)
                    result = await asyncio.wait_for(asyncio.shield(task), timeout=2)
                    assert result.return_code == 124
                else:
                    task = asyncio.create_task(env.exec("sleep 600"))
                    await asyncio.wait_for(reader_started.wait(), timeout=1)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(asyncio.shield(task), timeout=1)

                assert not reader_finished.is_set()
                assert response.close.called
        finally:
            release_reader.set()
            await asyncio.wait_for(reader_finished.wait(), timeout=1)
            rest_client.close()

        assert close_count == 1

    @pytest.mark.asyncio
    async def test_external_cancellation_wins_during_timeout_cleanup(self):
        configuration = k8s_client.Configuration()
        rest_client = k8s_client.ApiClient(configuration)
        env = _make_client_env(
            core_api=k8s_client.CoreV1Api(rest_client), client_manager=MagicMock()
        )
        response = MagicMock(returncode=0)
        loop = asyncio.get_running_loop()
        reader_started = asyncio.Event()
        abort_started = asyncio.Event()
        release_reader = threading.Event()
        reader_finished = asyncio.Event()

        def read_output(_self: GKEEnvironment, _response):
            loop.call_soon_threadsafe(reader_started.set)
            release_reader.wait(timeout=5)
            loop.call_soon_threadsafe(reader_finished.set)
            return "", ""

        response.sock.abort.side_effect = abort_started.set

        try:
            with (
                patch.object(gke_module, "stream", return_value=response),
                patch.object(GKEEnvironment, "_read_exec_output", read_output),
            ):
                task = asyncio.create_task(env.exec("sleep 600", timeout_sec=1))
                await asyncio.wait_for(reader_started.wait(), timeout=1)
                await asyncio.wait_for(abort_started.wait(), timeout=2)
                task.cancel()

                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(task), timeout=1)
        finally:
            release_reader.set()
            await asyncio.wait_for(reader_finished.wait(), timeout=1)
            rest_client.close()

    @pytest.mark.asyncio
    async def test_cancelled_handshake_closes_late_response_and_client(self):
        configuration = k8s_client.Configuration()
        rest_client = k8s_client.ApiClient(configuration)
        env = _make_client_env(
            core_api=k8s_client.CoreV1Api(rest_client), client_manager=MagicMock()
        )
        response = MagicMock(returncode=0)
        loop = asyncio.get_running_loop()
        handshake_started = asyncio.Event()
        release_handshake = threading.Event()
        response_closed = asyncio.Event()
        client_closed = asyncio.Event()
        stream_client = None

        def fake_stream(api_method, *args, **kwargs):
            nonlocal stream_client
            stream_client = api_method.__self__.api_client
            loop.call_soon_threadsafe(handshake_started.set)
            release_handshake.wait(timeout=5)
            return response

        def close_client(client):
            if client is stream_client:
                loop.call_soon_threadsafe(client_closed.set)

        response.close.side_effect = lambda: loop.call_soon_threadsafe(
            response_closed.set
        )

        try:
            with (
                patch.object(gke_module, "stream", side_effect=fake_stream),
                patch.object(
                    k8s_client.ApiClient,
                    "close",
                    autospec=True,
                    side_effect=close_client,
                ),
            ):
                task = asyncio.create_task(env.exec("true"))
                await asyncio.wait_for(handshake_started.wait(), timeout=1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

                assert not response_closed.is_set()
                assert not client_closed.is_set()
                release_handshake.set()
                await asyncio.wait_for(response_closed.wait(), timeout=1)
                await asyncio.wait_for(client_closed.wait(), timeout=1)
        finally:
            release_handshake.set()
            rest_client.close()


class TestGKETransferStreams:
    @pytest.mark.parametrize(
        "operation",
        ["upload_file", "upload_dir", "download_file", "download_dir"],
    )
    @pytest.mark.asyncio
    async def test_transfer_socket_io_is_concurrent_and_client_isolated(
        self, operation, tmp_path
    ):
        configuration = k8s_client.Configuration()
        rest_client = k8s_client.ApiClient(configuration)
        env = _make_client_env(
            core_api=k8s_client.CoreV1Api(rest_client), client_manager=MagicMock()
        )
        loop = asyncio.get_running_loop()
        socket_operation_started = asyncio.Event()
        release_socket_operation = threading.Event()

        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            member = tarfile.TarInfo("tmp/source.txt")
            member.size = len(b"downloaded")
            tar.addfile(member, io.BytesIO(b"downloaded"))

        response = _BlockingTransferResponse(
            loop,
            socket_operation_started,
            release_socket_operation,
            stdout_chunks=[archive.getvalue()]
            if operation.startswith("download")
            else [],
        )
        stream_client = None
        closed_clients = []

        def fake_stream(api_method, *args, **kwargs):
            nonlocal stream_client
            stream_client = api_method.__self__.api_client
            return response

        def close_client(client):
            closed_clients.append(client)

        source_file = tmp_path / "source.txt"
        source_file.write_text("uploaded")
        source_dir = tmp_path / "source-dir"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("uploaded")
        target_file = tmp_path / "target.txt"
        target_dir = tmp_path / "target-dir"

        if operation == "upload_file":
            transfer = env.upload_file(source_file, "/tmp/target.txt")
        elif operation == "upload_dir":
            transfer = env.upload_dir(source_dir, "/tmp/target-dir")
        elif operation == "download_file":
            transfer = env.download_file("/tmp/source.txt", target_file)
        else:
            transfer = env.download_dir("/tmp", target_dir)

        try:
            with (
                patch.object(gke_module, "stream", side_effect=fake_stream),
                patch.object(env, "_wait_for_container_exec_ready", new=AsyncMock()),
                patch.object(
                    env,
                    "exec",
                    new=AsyncMock(return_value=MagicMock(return_code=0)),
                ),
                patch.object(
                    k8s_client.ApiClient,
                    "close",
                    autospec=True,
                    side_effect=close_client,
                ),
            ):
                task = asyncio.create_task(transfer)
                await asyncio.wait_for(socket_operation_started.wait(), timeout=1)
                release_socket_operation.set()
                await asyncio.wait_for(task, timeout=1)

                assert stream_client is not rest_client
                assert stream_client in closed_clients
                assert response.closed
        finally:
            release_socket_operation.set()
            rest_client.close()

        if operation == "download_file":
            assert target_file.read_bytes() == b"downloaded"
        elif operation == "download_dir":
            assert (target_dir / "tmp" / "source.txt").read_bytes() == b"downloaded"


class TestGKEClientIsolation:
    """Ownership of the mutable Kubernetes clients.

    kubernetes.stream.stream() selects its WebSocket transport by replacing
    ApiClient.request on the client it is handed, so a client shared between
    exec and ordinary REST calls carries that mutation to every other caller.
    """

    @pytest.mark.asyncio
    async def test_ensure_client_creates_private_api_client_per_environment(self):
        configuration = k8s_client.Configuration()
        manager_client = k8s_client.ApiClient(configuration)
        shared_api = k8s_client.CoreV1Api(manager_client)

        manager = MagicMock()
        manager.get_client = AsyncMock(return_value=shared_api)

        env_a = _make_client_env(client_manager=manager)
        env_b = _make_client_env(client_manager=manager)

        await env_a._ensure_client()
        await env_b._ensure_client()

        assert env_a._api.api_client is not shared_api.api_client
        assert env_b._api.api_client is not shared_api.api_client
        assert env_a._api.api_client is not env_b._api.api_client

        # The manager still owns authentication; only the client is private.
        assert env_a._api.api_client.configuration is configuration
        assert env_b._api.api_client.configuration is configuration

    @pytest.mark.asyncio
    async def test_stop_closes_environment_owned_api_client(self):
        configuration = k8s_client.Configuration()
        rest_client = k8s_client.ApiClient(configuration)
        manager = MagicMock()
        manager.release_client = AsyncMock()
        env = _make_client_env(
            core_api=k8s_client.CoreV1Api(rest_client), client_manager=manager
        )

        with patch.object(rest_client, "close") as close_client:
            await env.stop(delete=False)

        close_client.assert_called_once_with()
        manager.release_client.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_exec_uses_temporary_exec_only_api_client(self):
        configuration = k8s_client.Configuration()
        rest_client = k8s_client.ApiClient(configuration)
        env = _make_client_env(core_api=k8s_client.CoreV1Api(rest_client))

        captured = {}
        response = _FakeExecResponse()

        def fake_stream(api_method, *args, **kwargs):
            captured["client"] = api_method.__self__.api_client
            return response

        with patch.object(gke_module, "stream", side_effect=fake_stream):
            with patch.object(
                k8s_client.ApiClient, "close", autospec=True
            ) as mock_close:
                result = await env.exec("true")

        assert result.return_code == 0
        assert captured["client"] is not rest_client
        assert captured["client"].configuration is configuration
        assert response.closed
        assert mock_close.call_count == 1
        assert mock_close.call_args.args[0] is captured["client"]

    @pytest.mark.asyncio
    async def test_exec_stream_mutation_does_not_affect_rest_client(self):
        """The regression: REST transport must survive an in-flight exec.

        The handshake runs in a dedicated thread, so it is held open with
        threading events rather than sleeps.
        """
        configuration = k8s_client.Configuration()
        rest_client = k8s_client.ApiClient(configuration)
        env = _make_client_env(core_api=k8s_client.CoreV1Api(rest_client))

        rest_request = MagicMock(name="rest_request")
        websocket_request = MagicMock(name="websocket_request")
        setattr(env._api.api_client, "request", rest_request)

        entered_stream = threading.Event()
        release_stream = threading.Event()

        def fake_stream(api_method, *args, **kwargs):
            stream_client = api_method.__self__.api_client
            previous_request = stream_client.request
            stream_client.request = websocket_request
            entered_stream.set()
            assert release_stream.wait(timeout=5)
            stream_client.request = previous_request
            return _FakeExecResponse()

        with patch.object(gke_module, "stream", side_effect=fake_stream):
            exec_task = asyncio.create_task(env.exec("true"))
            assert await asyncio.to_thread(entered_stream.wait, 5)
            try:
                assert env._api.api_client.request is rest_request
            finally:
                release_stream.set()
                await exec_task

    @pytest.mark.asyncio
    async def test_concurrent_exec_calls_use_distinct_api_clients(self):
        configuration = k8s_client.Configuration()
        rest_client = k8s_client.ApiClient(configuration)
        env = _make_client_env(core_api=k8s_client.CoreV1Api(rest_client))

        both_active = threading.Barrier(2, timeout=5)
        captured: list = []
        lock = threading.Lock()

        def fake_stream(api_method, *args, **kwargs):
            with lock:
                captured.append(api_method.__self__.api_client)
            both_active.wait()
            return _FakeExecResponse()

        with patch.object(gke_module, "stream", side_effect=fake_stream):
            async with asyncio.TaskGroup() as task_group:
                first = task_group.create_task(env.exec("true"))
                second = task_group.create_task(env.exec("true"))

        assert first.result().return_code == 0
        assert second.result().return_code == 0

        assert len(captured) == 2
        assert captured[0] is not captured[1]
        assert all(client is not rest_client for client in captured)

    @pytest.mark.asyncio
    async def test_download_dir_uses_temporary_client_and_bounded_memory(
        self, tmp_path
    ):
        configuration = k8s_client.Configuration()
        rest_client = k8s_client.ApiClient(configuration)
        env = _make_client_env(core_api=k8s_client.CoreV1Api(rest_client))

        payload_size = 8 * 1024 * 1024
        archive_buffer = io.BytesIO()
        with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
            member = tarfile.TarInfo("large.log")
            member.size = payload_size
            archive.addfile(member, io.BytesIO(b"x" * payload_size))
        archive_data = archive_buffer.getvalue()
        chunks = [
            archive_data[offset : offset + 64 * 1024]
            for offset in range(0, len(archive_data), 64 * 1024)
        ]
        response = _FakeExecResponse(stdout_chunks=chunks)
        captured = {}

        def fake_stream(api_method, *args, **kwargs):
            captured["client"] = api_method.__self__.api_client
            return response

        tracemalloc.start()
        try:
            with patch.object(gke_module, "stream", side_effect=fake_stream):
                with patch.object(
                    k8s_client.ApiClient, "close", autospec=True
                ) as mock_close:
                    await env.download_dir("/logs/agent", tmp_path)
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert (tmp_path / "large.log").stat().st_size == payload_size
        assert peak_bytes < payload_size
        assert captured["client"] is not rest_client
        assert response.closed
        assert mock_close.call_count == 1
        assert mock_close.call_args.args[0] is captured["client"]
