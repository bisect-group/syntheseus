import pytest

from emulator_bench.rescore import rescore_records


def _record(product, predictions, target):
    preds = [{"reactants": [{"smiles": s} for s in p.split(".")]} for p in predictions]
    return {
        "input": product,
        "predictions": preds,
        "ground_truth_correct": [
            sorted(p.split(".")) == sorted(target.split(".")) for p in predictions
        ],
    }


def test_variants():
    # row 0: hit at rank 2; row 1: hit at rank 3, rank 2 once the two stereoisomers collapse;
    # row 2: hit at rank 1. Rows 1 and 2 share a product with different targets.
    rows = [
        ("CCO", ["CC=O", "CCOC(C)=O"], "CCOC(C)=O"),
        ("C[C@H](N)O", ["C[C@@H](N)O.O", "C[C@H](N)O.O", "CC(N)=O"], "CC(N)=O"),
        ("C[C@H](N)O", ["C[C@@H](N)O.O", "C[C@H](N)O.O", "CC(N)=O"], "C[C@@H](N)O.O"),
    ]
    records = [_record(p, preds, t) for p, preds, t in rows]
    out = rescore_records(records, [t for *_, t in rows], num_top=50)
    assert out["n_reactions"] == 3 and out["n_products"] == 2
    exact = out["stereo"]["per_reaction"]
    assert exact["top1"] == pytest.approx(1 / 3)  # row 2
    assert exact["top10"] == pytest.approx(1.0)
    assert exact["mrr"] == pytest.approx((1 / 2 + 1 / 3 + 1) / 3)
    free = out["no_stereo"]["per_reaction"]
    # without stereo, row 1's first two predictions collapse, so CC(N)=O moves to rank 2
    assert free["mrr"] == pytest.approx((1 / 2 + 1 / 2 + 1) / 3)
    multi = out["stereo"]["multi_reference"]
    assert multi["top1"] == pytest.approx(1 / 2) and multi["mrr"] == pytest.approx((1 / 2 + 1) / 2)
    assert "top50" in exact


def test_self_check_catches_a_mismatch():
    record = _record("CCO", ["CC=O"], "CC=O")
    record["ground_truth_correct"] = [False]
    with pytest.raises(RuntimeError):
        rescore_records([record], ["CC=O"], num_top=50)


def test_forward_scores_the_predicted_product():
    """Forward reads the same rows the other way: the product side is scored against the product."""
    record = {
        "input": "CCO.CC(=O)O",
        "predictions": [
            {"reactants": [{"smiles": "CCO"}], "products": [{"smiles": "CC=O"}]},
            {"reactants": [{"smiles": "CCO"}], "products": [{"smiles": "CCOC(C)=O"}]},
        ],
        "ground_truth_correct": [False, True],
    }
    out = rescore_records([record], ["CCOC(C)=O"], num_top=50, side="products")
    exact = out["stereo"]["per_reaction"]
    assert exact["top1"] == 0.0 and exact["top10"] == 1.0
    # scoring the reactant side instead would miss, which is what retro does
    other = rescore_records(
        [{**record, "ground_truth_correct": [True, True]}], ["CCO"], num_top=50, side="reactants"
    )
    assert other["stereo"]["per_reaction"]["top1"] == 1.0
