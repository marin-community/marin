import pytest

from capability_pipeline import sandbox_provider


def test_default_provider_is_daytona_and_claims_daytona(monkeypatch):
    monkeypatch.delenv("CAPABILITY_SANDBOX_PROVIDER", raising=False)
    assert sandbox_provider.provider() == "daytona"
    assert sandbox_provider.adapter_id() == "taskcompendium-daytona"
    assert sandbox_provider.isolation_id() == "daytona-network-block-all"


def test_silo_claims_silo_never_daytona(monkeypatch):
    monkeypatch.setenv("CAPABILITY_SANDBOX_PROVIDER", "silo")
    assert sandbox_provider.adapter_id() == "taskcompendium-silo"
    assert sandbox_provider.isolation_id() == "silo-netns-none"
    assert "daytona" not in sandbox_provider.isolation_id()


def test_checkers_accept_either_proven_mechanism():
    assert sandbox_provider.NETWORK_BLOCKED_ISOLATION == {
        "daytona-network-block-all", "silo-netns-none"}
    assert sandbox_provider.KNOWN_ADAPTERS == {"taskcompendium-daytona", "taskcompendium-silo"}


def test_credentials_follow_the_selected_provider(monkeypatch):
    for key in ("DAYTONA_API_KEY", "SILO_API_TOKEN", "SILO_BROKER_RESOLVE_URL", "SILO_BROKER_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CAPABILITY_SANDBOX_PROVIDER", "silo")
    monkeypatch.setenv("DAYTONA_API_KEY", "k")
    # a Daytona key must not satisfy a silo run: nothing falls back silently
    assert not sandbox_provider.credentials_present()
    monkeypatch.setenv("SILO_API_TOKEN", "t")
    assert not sandbox_provider.credentials_present()
    monkeypatch.setenv("SILO_BROKER_RESOLVE_URL", "http://broker")
    assert sandbox_provider.credentials_present()
    monkeypatch.setenv("CAPABILITY_SANDBOX_PROVIDER", "daytona")
    monkeypatch.delenv("DAYTONA_API_KEY")
    assert not sandbox_provider.credentials_present()


def test_unknown_provider_is_refused(monkeypatch):
    monkeypatch.setenv("CAPABILITY_SANDBOX_PROVIDER", "docker")
    with pytest.raises(RuntimeError, match="must be one of"):
        sandbox_provider.provider()
