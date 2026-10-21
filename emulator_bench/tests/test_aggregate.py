import json

import pytest

from emulator_bench.aggregate import topk_table
from emulator_bench.common import DONE_MARKER


def _write_run(root, name, **result):
    run = root / "m" / name / "g" / "s" / "f0"
    run.mkdir(parents=True)
    base = {
        "model": "m",
        "direction": name,
        "split_group": "g",
        "split_name": "s",
        "fold": 0,
        "n_multi_product": 0,
        "n_multi_tie": 0,
    }
    (run / "result.json").write_text(json.dumps(base | result))
    (run / "predictions.jsonl").write_text("{}\n" * result["n"])
    (run / DONE_MARKER).touch()


def test_dropped_rows_count_as_misses(tmp_path):
    _write_run(tmp_path, "a", n=3, n_dropped=1, top1=0.5, top10=0.8, mrr=0.6)
    _write_run(tmp_path, "b", n=4, n_dropped=0, top1=0.5, top10=0.8, mrr=0.6)
    a, b = topk_table(tmp_path).set_index("direction").loc[["a", "b"]].itertuples()
    assert a.n == 4
    assert (a.top1, a.top10, a.mrr) == pytest.approx((0.375, 0.6, 0.45))
    assert a.n_dropped == 1
    assert b.n == 4
    assert (b.top1, b.top10, b.mrr) == pytest.approx((0.5, 0.8, 0.6))
