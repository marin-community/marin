from queue import Queue

from omegaconf import OmegaConf

from skyrl_train.entrypoints.evaluation_progress import forward_evaluation_progress, wait_for_evaluation
from skyrl_train.utils.progress import configure_progress, tqdm


def test_existing_worker_progress_reaches_driver_before_results(monkeypatch, capsys):
    progress = Queue()
    configure_progress(
        OmegaConf.create(
            {"mode": "logging", "min_interval_seconds": 0, "heartbeat_seconds": 60, "percent_step": 5, "count_step": 1000}
        )
    )
    with forward_evaluation_progress(progress):
        with tqdm(total=2, desc="Evaluation Progress") as bar:
            bar.update(1)
            bar.update(1)
    result = object()
    ready = iter([([], [result]), ([result], [])])
    monkeypatch.setattr("skyrl_train.entrypoints.evaluation_progress.ray.wait", lambda *args, **kwargs: next(ready))
    monkeypatch.setattr("skyrl_train.entrypoints.evaluation_progress.ray.get", lambda ref: {"scored": 2})
    assert wait_for_evaluation(result, progress) == {"scored": 2}
    output = capsys.readouterr().out
    assert "Evaluation Progress: 0/2" in output
    assert "Evaluation Progress: 2/2" in output
    assert progress.empty()
