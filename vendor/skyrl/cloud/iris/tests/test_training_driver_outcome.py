import json
import signal

from omegaconf import OmegaConf

from cloud.iris.training_driver import LocalRLConfig, LocalRLRunner
from marinskyrl.environment_contract import DEBUG_ARTIFACT_DIR_ENV


class _AbortedProcess:
    pid = 321

    def wait(self) -> int:
        return -signal.SIGABRT


def test_training_driver_preserves_child_signal_outcome(tmp_path, monkeypatch) -> None:
    (tmp_path / "skyrl-train").mkdir()
    artifact_root = tmp_path / "debug"
    monkeypatch.setenv("RAY_ADDRESS", "ray://controller")
    monkeypatch.setenv("SKYRL_HOME", str(tmp_path))
    monkeypatch.setenv(DEBUG_ARTIFACT_DIR_ENV, str(artifact_root))
    monkeypatch.setattr("cloud.iris.training_driver.subprocess.Popen", lambda *_args, **_kwargs: _AbortedProcess())
    runner = LocalRLRunner(
        LocalRLConfig(
            job_name="signal-test",
            model_path="org/model",
        )
    )
    launch_config = OmegaConf.create({"runtime": {"entrypoint": "skyrl_train.entrypoints.main_base"}, "skyrl": {}})

    exit_code = runner._run_skyrl(launch_config)

    assert exit_code == 128 + signal.SIGABRT
    receipts = list((artifact_root / "outcomes").glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["raw_returncode"] == -signal.SIGABRT
    assert receipt["signal_name"] == "SIGABRT"
    assert receipt["metadata"]["entrypoint"] == "skyrl_train.entrypoints.main_base"
