# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `finelog.deploy.config`."""

import json
import textwrap
from pathlib import Path

import pytest
from finelog.deploy.config import (
    Deployment,
    FinelogConfig,
    ForwardingConfig,
    GcpDeployment,
    JwtAuthLayer,
    K8sDeployment,
    _bundled_config_dir,
    auth_policy_json,
    derive_endpoint_uri,
    load_finelog_config,
)


def _write_config(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body))


def test_load_config_from_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "test.yaml"
    _write_config(
        cfg_path,
        """
        name: finelog-test
        port: 10001
        image: ghcr.io/test/finelog:latest
        remote_log_dir: gs://bucket/test
        deployment:
          gcp:
            project: my-proj
            zone: us-central1-a
        """,
    )
    cfg = load_finelog_config(str(cfg_path))
    assert cfg.name == "finelog-test"
    assert cfg.port == 10001
    assert cfg.image == "ghcr.io/test/finelog:latest"
    assert cfg.remote_log_dir == "gs://bucket/test"
    assert cfg.deployment.gcp is not None
    assert cfg.deployment.gcp.project == "my-proj"
    assert cfg.deployment.gcp.zone == "us-central1-a"
    assert cfg.deployment.gcp.machine_type == "n2-standard-4"  # default
    assert cfg.deployment.k8s is None


def test_load_config_from_repo_marin() -> None:
    cfg = load_finelog_config("marin")
    assert cfg.name == "finelog-marin"
    assert cfg.port == 10001
    assert cfg.image == "ghcr.io/marin-community/finelog:latest"
    assert cfg.remote_log_dir == "gs://marin-us-central2/finelog/marin"
    assert cfg.deployment.gcp is not None
    assert cfg.deployment.gcp.project == "hai-gcp-models"
    assert cfg.deployment.gcp.zone == "us-central1-a"


def test_load_config_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Point the user-config dir at an empty tmpdir to keep the searched-paths
    # output deterministic and rule out accidental hits.
    monkeypatch.setattr("finelog.deploy.config.USER_CONFIG_DIR", tmp_path / "no-such-dir")
    with pytest.raises(FileNotFoundError) as exc:
        load_finelog_config("definitely-not-a-real-config-name-xyz")
    msg = str(exc.value)
    assert "definitely-not-a-real-config-name-xyz" in msg
    assert "searched" in msg


def test_load_config_neither_deployment_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    _write_config(
        cfg_path,
        """
        name: finelog-bad
        port: 10001
        image: ghcr.io/test/finelog:latest
        deployment: {}
        """,
    )
    with pytest.raises(ValueError, match="exactly one of"):
        load_finelog_config(str(cfg_path))


def test_load_config_both_deployments_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "both.yaml"
    _write_config(
        cfg_path,
        """
        name: finelog-both
        port: 10001
        image: ghcr.io/test/finelog:latest
        deployment:
          gcp:
            project: p
            zone: z
          k8s:
            namespace: ns
        """,
    )
    with pytest.raises(ValueError, match="exactly one of"):
        load_finelog_config(str(cfg_path))


def test_derive_endpoint_uri_gcp() -> None:
    cfg = FinelogConfig(
        name="finelog-x",
        port=10001,
        image="ghcr.io/x/finelog:latest",
        remote_log_dir="",
        deployment=Deployment(gcp=GcpDeployment(project="proj", zone="us-central1-a")),
    )
    uri, metadata = derive_endpoint_uri(cfg)
    assert uri == "gcp://finelog-x"
    assert metadata == {"project": "proj", "zone": "us-central1-a", "port": "10001"}


def test_derive_endpoint_uri_k8s() -> None:
    cfg = FinelogConfig(
        name="finelog-x",
        port=10001,
        image="ghcr.io/x/finelog:latest",
        remote_log_dir="",
        deployment=Deployment(k8s=K8sDeployment(namespace="iris")),
    )
    uri, metadata = derive_endpoint_uri(cfg)
    assert uri == "k8s://finelog-x.iris"
    assert metadata == {"port": "10001"}


def test_auth_layers_serialize_to_finelog_policy_json(tmp_path: Path) -> None:
    """An ordered cidr+jwt `auth:` list serializes to the exact `FINELOG_AUTH_POLICY`
    JSON the finelog Rust server parses — order preserved, snake_case tags, and the
    per-cluster `public_keys` list (two entries model a rotation overlap)."""
    cfg_path = tmp_path / "authed.yaml"
    _write_config(
        cfg_path,
        """
        name: finelog-authed
        port: 10001
        image: ghcr.io/test/finelog:latest
        remote_log_dir: gs://bucket/x
        deployment:
          gcp:
            project: p
            zone: us-central1-a
        auth:
          - type: cidr
            cidrs: [10.0.0.0/8, "::1/128"]
          - type: jwt
            keys:
              - cluster: marin
                public_keys: [ed25519-pub-marin-current, ed25519-pub-marin-previous]
        """,
    )
    cfg = load_finelog_config(str(cfg_path))
    assert json.loads(auth_policy_json(cfg.auth)) == [
        {"type": "cidr", "cidrs": ["10.0.0.0/8", "::1/128"]},
        {
            "type": "jwt",
            "keys": [{"cluster": "marin", "public_keys": ["ed25519-pub-marin-current", "ed25519-pub-marin-previous"]}],
        },
    ]


def test_auth_unknown_layer_type_rejected(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    _write_config(
        cfg_path,
        """
        name: finelog-bad
        port: 10001
        image: ghcr.io/test/finelog:latest
        remote_log_dir: gs://bucket/x
        deployment:
          gcp:
            project: p
            zone: us-central1-a
        auth:
          - type: mtls
        """,
    )
    with pytest.raises(ValueError, match="unknown type 'mtls'"):
        load_finelog_config(str(cfg_path))


# -- forwarding ----------------------------------------------------------------

_FORWARDING_YAML = """
    name: finelog-cw
    port: 10001
    image: ghcr.io/test/finelog:latest
    remote_log_dir: s3://bucket/x
    deployment:
      k8s:
        namespace: iris
    forwarding:
      target: https://finelog.oa.dev
      cluster: cw-rno2a
      signing_key: gcp-secret://projects/p/secrets/s/versions/1
    """


def test_load_config_parses_forwarding(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cw.yaml"
    _write_config(cfg_path, _FORWARDING_YAML)
    forwarding = load_finelog_config(str(cfg_path)).forwarding
    assert forwarding is not None
    assert forwarding.target == "https://finelog.oa.dev"
    assert forwarding.cluster == "cw-rno2a"
    # The reference, not the key: a config file holds no secret material.
    assert forwarding.signing_key == ("gcp-secret://projects/p/secrets/s/versions/1",)


def test_forwarding_env_json_omits_the_signing_key(tmp_path: Path) -> None:
    """`FINELOG_FORWARDING` inlines into a plaintext manifest, so it must carry only the
    hub's address and this sender's cluster — never the credential that reaches it."""
    cfg_path = tmp_path / "cw.yaml"
    _write_config(cfg_path, _FORWARDING_YAML)
    forwarding = load_finelog_config(str(cfg_path)).forwarding
    assert forwarding is not None
    assert json.loads(forwarding.to_env_json()) == {
        "target": "https://finelog.oa.dev",
        "cluster": "cw-rno2a",
    }


def test_forwarding_without_a_signing_key_is_rejected(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cw.yaml"
    _write_config(
        cfg_path,
        """
        name: finelog-cw
        port: 10001
        image: ghcr.io/test/finelog:latest
        remote_log_dir: s3://bucket/x
        deployment:
          k8s:
            namespace: iris
        forwarding:
          target: https://finelog.oa.dev
          cluster: cw-rno2a
        """,
    )
    with pytest.raises(ValueError, match="forwarding is missing signing_key"):
        load_finelog_config(str(cfg_path))


def test_forwarding_target_must_be_https() -> None:
    """The bearer is a hub credential; plaintext http would ship it in the clear."""
    with pytest.raises(ValueError, match="must be an https:// url"):
        ForwardingConfig(target="http://finelog.oa.dev", cluster="cw", signing_key=("env:K",))


def test_forwarding_cluster_must_be_named() -> None:
    """The hub records rows under the cluster this sender's key authenticates. An empty
    name would make every sender's rows indistinguishable from the hub's own."""
    with pytest.raises(ValueError, match="must name the origin cluster"):
        ForwardingConfig(target="https://finelog.oa.dev", cluster="", signing_key=("env:K",))


def _bundled_configs() -> dict[str, FinelogConfig]:
    """Every config shipped in `lib/finelog/config/`, keyed by name.

    Discovered rather than listed, so a config added without a matching hub entry fails
    the checks below instead of being silently skipped.
    """
    names = sorted(p.stem for p in _bundled_config_dir().glob("*.yaml"))
    assert names, "no bundled finelog configs found"
    return {name: load_finelog_config(name) for name in names}


def test_every_bundled_sender_names_a_cluster_some_bundled_hub_trusts() -> None:
    """A sender whose cluster appears in no hub's `jwt` layer can never authenticate.

    The hub stamps each row with the cluster its trusted key maps to, and rejects a push
    naming any other, so `forwarding.cluster` here and `keys[].cluster` there must agree
    exactly — a typo is a silent, total forwarding outage. This pairs the two by name
    across the bundled configs; it cannot check that the key halves match (the private
    half lives in Secret Manager) nor that the sender points at that particular hub.
    """
    configs = _bundled_configs()
    trusted = {
        entry.cluster
        for cfg in configs.values()
        for layer in cfg.auth
        if isinstance(layer, JwtAuthLayer)
        for entry in layer.keys
    }
    senders = {name: cfg.forwarding.cluster for name, cfg in configs.items() if cfg.forwarding}
    assert senders, "expected at least one bundled config to forward"
    for name, cluster in senders.items():
        assert cluster in trusted, f"{name}: forwards as {cluster!r}, which no bundled hub's jwt layer trusts"


def test_bundled_forwarding_configs_deploy_on_k8s() -> None:
    """The gcp backend refuses forwarding — it can only reach the server through
    world-readable startup-script metadata, which is no place for a signing key."""
    for name, cfg in _bundled_configs().items():
        if cfg.forwarding is not None:
            assert cfg.deployment.k8s is not None, f"{name}: forwards but deploys on gcp"
