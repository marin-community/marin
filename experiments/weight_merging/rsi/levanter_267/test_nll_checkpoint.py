# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Choice-level interruption/restore through the exact pinned FineStore manager."""

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from experiments.weight_merging.rsi.levanter_267.checkpoint_entrypoint import (
    NLL_NATIVE_SHA256,
    install_nll_checkpointing,
)


def test_nll_choices_commit_then_restore_without_reordering(tmp_path, monkeypatch):
    spec = importlib.util.find_spec("eval")
    if spec is None:
        pytest.skip("Requires the pinned Evalchemy metadata package")
    root = Path(spec.origin).parent
    source = (root / "resume/lm_eval_native.py").read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == NLL_NATIVE_SHA256
    if importlib.util.find_spec("lm_eval") is None:
        utils = ModuleType("lm_eval.utils")
        utils.handle_non_serializable = lambda value: (_ for _ in ()).throw(AssertionError("unused formatter"))
        monkeypatch.setitem(sys.modules, "lm_eval", ModuleType("lm_eval"))
        monkeypatch.setitem(sys.modules, "lm_eval.utils", utils)
    resume = pytest.importorskip("eval.contracts.finestore_resume")
    manifest = pytest.importorskip("eval.contracts.sample_manifest")
    fingerprint = pytest.importorskip("eval.resume.fingerprint")
    # Only optional, unused LM construction imports are stubbed. Execute the
    # entire exact pinned module, install the real hook and call its real wrapper.
    lm_module = ModuleType("lm_eval.api.model")
    lm_module.LM = object
    monkeypatch.setitem(sys.modules, "lm_eval.api", ModuleType("lm_eval.api"))
    monkeypatch.setitem(sys.modules, "lm_eval.api.model", lm_module)
    robust = ModuleType("eval.robust_api")
    robust.configure_generation_overrides = lambda *args, **kwargs: None
    robust.parse_generation_overrides = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "eval.robust_api", robust)
    native_spec = importlib.util.spec_from_file_location("eval.resume.lm_eval_native", root / "resume/lm_eval_native.py")
    native = importlib.util.module_from_spec(native_spec)
    monkeypatch.setitem(sys.modules, native_spec.name, native)
    native_spec.loader.exec_module(native)
    install_nll_checkpointing()
    wrapper = native._make_resume_caching_lm_cls()
    requests = [
        SimpleNamespace(
            doc_id=i // 3,
            idx=i % 3,
            task_name="piqa",
            repeats=1,
            args=(f"context-{i // 3}", f"continuation-{i % 3}"),
            ordinal=i,
        )
        for i in range(600)
    ]
    original_args = [request.args for request in requests]
    fp = fingerprint.RunFingerprint(inputs=dict(task_name="piqa", max_model_len=16320))

    def manager(path):
        return resume.FineStoreResumeManager(str(path), "piqa_0shot", "piqa", fp)

    class Model:
        def __init__(self, fail=False, bad=False):
            self.calls = []
            self.fail = fail
            self.bad = bad

        def loglikelihood(self, batch):
            self.calls.append(list(batch))
            if self.fail and len(self.calls) == 2:
                raise RuntimeError("forced second-chunk interruption")
            outputs = [(-float(request.ordinal + 1), bool(request.idx % 2)) for request in batch]
            if self.bad:
                outputs[-1] = (float("nan"), False)
            return outputs

    def owner(store, model):
        return wrapper(model, store, "piqa", sample_manifest=manifest.SampleManifest("piqa"))

    first = Model(fail=True)
    with pytest.raises(RuntimeError, match="second-chunk"):
        owner(manager(tmp_path), first).loglikelihood(requests)
    restored = manager(tmp_path)
    assert len(restored.restore()) == 256
    # Three choices/doc means the first committed chunk ends inside doc85.
    assert len({payload["sample"]["source_id"] for payload in restored.restore().values()}) == 86
    resumed = Model()
    live = owner(restored, resumed)
    outputs = live.loglikelihood(requests)
    assert [request.ordinal for batch in resumed.calls for request in batch] == list(range(256, 600))
    assert outputs == [(-float(i + 1), bool((i % 3) % 2)) for i in range(600)]
    assert [request.args for request in requests] == original_args
    assert len(manager(tmp_path).restore()) == 600
    live._sample_manifest.validate_generated()
    assert live._sample_manifest.generated_sample_count == 200

    invalid_path = tmp_path / "invalid"
    with pytest.raises(RuntimeError, match="Invalid loglikelihood"):
        owner(manager(invalid_path), Model(bad=True)).loglikelihood(requests)
    assert manager(invalid_path).restore() == {}

    corrupted = manager(tmp_path / "corrupted")
    key, payload = next(iter(manager(tmp_path).restore().items()))
    corrupted.record(dict(key), dict(payload, output=[float("nan"), False]))
    corrupted.finalize()
    untouched = Model()
    with pytest.raises(RuntimeError, match="Invalid loglikelihood"):
        owner(manager(tmp_path / "corrupted"), untouched).loglikelihood(requests)
    assert untouched.calls == []
    changed = list(requests)
    changed[0] = SimpleNamespace(**{**vars(requests[0]), "args": ("changed-context", "changed-choice")})
    with pytest.raises(ValueError, match="identities differ"):
        owner(manager(tmp_path), Model()).loglikelihood(changed)
