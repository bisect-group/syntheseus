import pytest

from emulator_bench.finetune_data import mapped_subreaction
from emulator_bench.retro_targets import Prepared, load_cofactors, prepare_row, target_of

NADPH = "NC(=O)C1=CN(C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(OP(=O)(O)O)C3O)C(O)C2O)C=CC1"
NADP = "NC(=O)c1ccc[n+](C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(OP(=O)(O)O)C3O)C(O)C2O)c1"


@pytest.fixture(scope="module")
def cof():
    return load_cofactors()


def _prep(rxn, cof) -> Prepared:
    prep = prepare_row(rxn, rxn, cof)
    assert isinstance(prep, Prepared)
    return prep


def test_subreaction_keeps_only_the_curated_task(cof):
    # ketone reduction: NADPH is the hydride donor and must not appear in the training reaction
    rxn = f"CC(=O)c1ccccc1.{NADPH}>>CC(O)c1ccccc1.{NADP}"
    prep = _prep(rxn, cof)
    mapped = (
        f"[CH3:1][C:2](=[O:3])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1.{NADPH}"
        f">>[CH3:1][CH:2]([OH:3])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1.{NADP}"
    )
    mapping = {"method": "full", "mapped_rxn": mapped, "confidence": 0.9}
    target = target_of(prep, mapping, cof)
    sub = mapped_subreaction(prep, mapping, target.reactants)
    left, right = sub.split(">>")
    assert "." not in left  # only the ketone, not NADPH
    assert ":2]" in left and ":2]" in right  # shared atoms stay mapped


def test_atoms_from_removed_reactants_are_unmapped(cof):
    # monooxygenase: the product oxygen comes from O2, which is not in the target
    rxn = "c1ccccc1.O=O>>Oc1ccccc1"
    prep = _prep(rxn, cof)
    mapped = (
        "[cH:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1.[O:1]=[O:8]"
        ">>[OH:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1"
    )
    mapping = {"method": "full", "mapped_rxn": mapped, "confidence": 0.5}
    target = target_of(prep, mapping, cof)
    sub = mapped_subreaction(prep, mapping, target.reactants)
    left, right = sub.split(">>")
    assert "O=O" not in left
    assert ":1]" not in right  # the new oxygen carries no map number
    assert ":2]" in left and ":2]" in right


def test_no_mapping_gives_none(cof):
    prep = _prep("CC(=O)c1ccccc1>>CC(O)c1ccccc1", cof)
    assert mapped_subreaction(prep, None, ("CC(=O)c1ccccc1",)) is None
    bad = {"method": "failed", "mapped_rxn": "", "confidence": float("nan")}
    assert mapped_subreaction(prep, bad, ("CC(=O)c1ccccc1",)) is None


def test_no_shared_atoms_gives_none(cof):
    # the mapper linked no product atom to the target reactant: nothing to learn from
    prep = _prep("CCO>>CC(=O)O", cof)
    mapped = "[CH3:1][CH2:2][OH:3]>>[CH3:4][C:5](=[O:6])[OH:7]"
    mapping = {"method": "full", "mapped_rxn": mapped, "confidence": 0.1}
    assert mapped_subreaction(prep, mapping, ("CCO",)) is None


def test_unwrap_numpy_reprs(tmp_path):
    from emulator_bench.localretro_preprocess import unwrap_numpy_reprs

    path = tmp_path / "labeled_data.csv"
    path.write_text('Labels\n"[(np.int64(3), np.int64(7))]"\n"[(1, 2)]"\n')
    assert unwrap_numpy_reprs(tmp_path) == [path]
    assert path.read_text() == 'Labels\n"[(3, 7)]"\n"[(1, 2)]"\n'
    assert unwrap_numpy_reprs(tmp_path) == []  # already plain


def test_fully_map_products_numbers_new_atoms():
    from emulator_bench.finetune_data import fully_map_products

    # the product oxygen came from O2, which the target does not contain, so it carries no map
    mapped = "[cH:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1>>[OH][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1"
    out = fully_map_products(mapped)
    from rdkit import Chem

    product = Chem.MolFromSmiles(out.split(">>")[1], sanitize=False)
    assert all(a.GetAtomMapNum() for a in product.GetAtoms())  # RootAligned requires this
    assert out.split(">>")[0] == mapped.split(">>")[0]  # reactants untouched
    # the new number does not collide with a reactant map number
    assert 8 in {a.GetAtomMapNum() for a in product.GetAtoms()}


def test_write_megan_csv_uses_an_integer_class(tmp_path):
    """MEGAN's featuriser puts the reaction class in a sparse matrix, so it cannot be a string."""
    import pandas as pd

    from emulator_bench.finetune_megan import NO_REACTION_CLASS, write_megan_csv

    src = tmp_path / "raw_train.csv"
    pd.DataFrame(
        {"id": ["a"], "class": ["UNK"], "reactants>reagents>production": ["[CH4:1]>>[CH4:1]"]}
    ).to_csv(src, index=False)
    dest = tmp_path / "out.csv"
    write_megan_csv(src, dest)
    table = pd.read_csv(dest)
    assert table["class"].tolist() == [NO_REACTION_CLASS]
    assert table["class"].dtype.kind in "iu"
    assert table["reactants>reagents>production"].tolist() == ["[CH4:1]>>[CH4:1]"]


def test_forward_is_the_reverse_of_retro(tmp_path, monkeypatch):
    """Both directions read the same curated rows, so forward is retro with the roles swapped."""
    import json

    import pandas as pd

    from emulator_bench import common

    curated = tmp_path / "curated"
    curated.mkdir()
    pd.DataFrame(
        {
            "rxn_key": ["k1"],
            "reactants": ["CCO.CC(=O)O"],
            "product": ["CCOC(C)=O"],
            "minor_products": [""],
            "tie": [False],
            "n_merged": [1],
        }
    ).to_parquet(curated / "test.parquet", index=False)
    pd.DataFrame({"rxn_key": [], "rxn_smiles": [], "reason": [], "detail": []}).to_parquet(
        curated / "test_dropped.parquet", index=False
    )
    split = common.SplitFile("g", "all", curated)
    monkeypatch.setattr(common, "retro_file", lambda s, f, **k: curated / f"{f}.parquet")

    rows = {}
    for direction in ("retro", "forward"):
        path, counts = common.write_jsonl(tmp_path / "out", split, "test", direction)
        rows[direction] = json.loads(path.read_text().strip())
        assert counts["n_kept"] == 1
    assert rows["retro"] == rows["forward"]  # one file, the model picks the direction
    assert [m["smiles"] for m in rows["retro"]["reactants"]] == ["CCO", "CC(=O)O"]
    assert [m["smiles"] for m in rows["retro"]["products"]] == ["CCOC(C)=O"]
