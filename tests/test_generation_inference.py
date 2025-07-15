import ray

import marin.generation.inference as inference


def test_run_inference_passes_max_doc_length(monkeypatch):
    ray.init(ignore_reinit_error=True)
    recorded = {}
    ds = ray.data.from_items([{"text": "hello"}])

    monkeypatch.setattr(inference, "set_ray_data_config", lambda config: None)
    monkeypatch.setattr(inference, "ray_resources_kwarg", lambda config: {})
    monkeypatch.setattr(ray.data, "read_json", lambda *a, **kw: ds)

    def fake_vLLMTextGeneration(**kwargs):
        recorded["constructor_max_doc_length"] = kwargs.get("max_doc_length")

        class Dummy:
            def __call__(self, batch):
                return batch

        return Dummy()

    monkeypatch.setattr(inference, "vLLMTextGeneration", fake_vLLMTextGeneration)

    def fake_map_batches(self, fn, *, concurrency=None, batch_size=None, fn_constructor_kwargs=None, **kw):
        recorded["max_doc_length"] = fn_constructor_kwargs.get("max_doc_length")
        fn(**fn_constructor_kwargs)  # instantiate to record
        return self

    monkeypatch.setattr(ray.data.Dataset, "map_batches", fake_map_batches, raising=False)

    def fake_write_json(self, *a, **k):
        recorded["write"] = True
        return None

    monkeypatch.setattr(ray.data.Dataset, "write_json", fake_write_json, raising=False)

    config = inference.TextGenerationInferenceConfig(
        input_path="in",
        output_path="out",
        model_name="model",
        engine_kwargs={},
        generation_kwargs={},
        template="{example}",
        max_doc_length=5,
    )

    inference.run_inference._function(config)

    assert recorded["max_doc_length"] == 5
    assert recorded["constructor_max_doc_length"] == 5
    assert recorded["write"] is True
    ray.shutdown()
