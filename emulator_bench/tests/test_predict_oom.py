import torch

from emulator_bench import predict


class FakeModel:
    """Runs out of memory on batches above `max_batch` and on inputs listed in `too_big`."""

    def __init__(self, max_batch, too_big=()):
        self.max_batch, self.too_big, self.calls = max_batch, set(too_big), []

    def __call__(self, inputs, num_results):
        self.calls.append(list(inputs))
        if len(inputs) > self.max_batch or self.too_big & set(inputs):
            raise torch.OutOfMemoryError("CUDA out of memory")
        return [[f"pred:{x}"] for x in inputs]


def test_batch_fits():
    model = FakeModel(max_batch=4)
    preds, skipped = predict.predict_batch(model, ["a", "b"], ["a", "b"], 50)
    assert preds == [["pred:a"], ["pred:b"]] and skipped == frozenset()
    assert model.calls == [["a", "b"]]


def test_oom_batch_is_retried_one_by_one(monkeypatch):
    monkeypatch.setattr(predict, "OOM_WAIT_S", 0)
    model = FakeModel(max_batch=1, too_big={"c"})
    preds, skipped = predict.predict_batch(model, ["a", "b", "c"], ["a", "b", "c"], 50)
    assert preds == [["pred:a"], ["pred:b"], []]
    assert skipped == frozenset({"c"})
    assert model.calls.count(["c"]) == predict.OOM_RETRIES + 1
