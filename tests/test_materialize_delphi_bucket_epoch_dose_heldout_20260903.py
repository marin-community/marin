# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_bucket_epoch_dose_heldout_20260903 as materialize,
)


def _run(state: str) -> SimpleNamespace:
    summary = {materialize.heldout.UNCHEATABLE_AGGREGATE: 1.0}
    summary.update({component: 1.0 for component in materialize.heldout.UNCHEATABLE_COMPONENTS})
    return SimpleNamespace(id="run-id", state=state, summary=summary, config={})


def test_preempted_run_prefers_exact_final_step_over_complete_summary(monkeypatch) -> None:
    persisted_values = np.full(len(materialize.heldout.UNCHEATABLE_COMPONENTS), 2.0)
    monkeypatch.setattr(
        materialize,
        "_persisted_uncheatable",
        lambda _run, _step: (2.0, persisted_values, "gs://marin-us-east5/final.jsonl"),
    )

    observed = materialize._uncheatable_candidate(_run("crashed"), expected_step=3006)

    assert observed is not None
    assert observed["aggregate"] == 2.0
    assert observed["provenance"] == "gcs_exact_final_step"
    np.testing.assert_array_equal(observed["values"], persisted_values)


def test_preempted_run_rejects_summary_without_exact_final_step(monkeypatch) -> None:
    monkeypatch.setattr(materialize, "_persisted_uncheatable", lambda _run, _step: None)

    assert materialize._uncheatable_candidate(_run("crashed"), expected_step=3006) is None


def test_finished_run_uses_validated_summary_without_persisted_read(monkeypatch) -> None:
    def unexpected_read(_run, _step):
        raise AssertionError("finished run should not read persisted metrics")

    monkeypatch.setattr(materialize, "_persisted_uncheatable", unexpected_read)

    observed = materialize._uncheatable_candidate(_run("finished"), expected_step=3006)

    assert observed is not None
    assert observed["aggregate"] == 1.0
    assert observed["provenance"] == "wandb_summary_validated"
