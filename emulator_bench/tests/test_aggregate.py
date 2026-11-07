import json

import pytest

from emulator_bench.aggregate import seeds_table, topk_table
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
    a, b = topk_table([tmp_path]).set_index("direction").loc[["a", "b"]].itertuples()
    assert a.n == 4
    assert (a.top1, a.top10, a.mrr) == pytest.approx((0.375, 0.6, 0.45))
    assert a.n_dropped == 1
    assert b.n == 4
    assert (b.top1, b.top10, b.mrr) == pytest.approx((0.5, 0.8, 0.6))


def test_seeds_table(tmp_path):
    for seed, top1 in ((0, 0.2), (1, 0.4), (2, 0.6)):
        _write_run(
            tmp_path / f"seed{seed}",
            "retro",
            n=5,
            n_dropped=0,
            top1=top1,
            top10=0.8,
            mrr=0.5,
            seed=seed,
        )
    seeds = seeds_table(topk_table([tmp_path / f"seed{s}" for s in range(3)]))
    [row] = seeds.itertuples()
    assert row.seeds == "0 1 2" and row.n == 5
    assert (row.top1_mean, row.top1_std, row.top10_std) == pytest.approx((0.4, 0.2, 0.0))
    _write_run(tmp_path / "seed3", "retro", n=4, n_dropped=0, top1=0.1, top10=0.8, mrr=0.5, seed=3)
    with pytest.raises(RuntimeError):
        seeds_table(topk_table([tmp_path / f"seed{s}" for s in range(4)]))


def test_headline_metrics_are_stereo_agnostic(tmp_path):
    """The headline columns come from rescore.json's no-stereo scores; stereo_* keeps the rest."""
    import json

    from emulator_bench.rescore import RESCORE_FILE

    _write_run(tmp_path / "retro", "m", n=4, n_dropped=0, top1=0.25, top10=0.5, top50=0.5, mrr=0.3)
    run = tmp_path / "retro" / "m" / "m" / "g" / "s" / "f0"
    stereo = {"top1": 0.25, "top10": 0.5, "top50": 0.5, "mrr": 0.3}
    free = {"top1": 0.5, "top10": 0.75, "top50": 0.75, "mrr": 0.6}
    (run / RESCORE_FILE).write_text(
        json.dumps(
            {
                "seed": 0,
                "all": {"stereo": {"per_reaction": stereo}},
                "common": {
                    "n_reactions": 4,
                    "n_products": 3,
                    "stereo": {"per_reaction": stereo, "multi_reference": stereo},
                    "no_stereo": {"per_reaction": free, "multi_reference": free},
                },
            }
        )
    )
    [row] = topk_table([tmp_path / "retro"]).itertuples()
    assert (row.top1, row.top10) == pytest.approx((0.5, 0.75))  # stereo ignored
    assert (row.stereo_top1, row.stereo_top10) == pytest.approx((0.25, 0.5))
    assert (row.all_top1, row.n_products) == pytest.approx((0.25, 3))
