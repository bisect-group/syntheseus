import sys
import types

import pytest

from emulator_bench.run_model import megan_class
from syntheseus.interface.molecule import Molecule


def test_oversize_molecule_gets_no_model_input_and_others_stay(monkeypatch):
    # MEGAN's own `src` package is only importable from a loaded model; stand in for its H fix.
    for name in ("src", "src.feat"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    utils = types.ModuleType("src.feat.utils")
    utils.fix_explicit_hs = lambda mol: mol
    monkeypatch.setitem(sys.modules, "src.feat.utils", utils)

    model = object.__new__(megan_class())
    model.n_max_atoms = 4
    model.skipped = set()
    batch = model._mols_to_batch([Molecule("CCO"), Molecule("CCCCCC"), Molecule("CC")])
    assert [m is not None for m in batch] == [True, False, True]
    assert model.skipped == {"CCCCCC"}


def test_size_mismatch_in_batch_is_retried_one_by_one(monkeypatch):
    from syntheseus.reaction_prediction.inference import megan

    def fake(self, inputs, num_results):
        if len(inputs) > 1 and any(m.smiles == "CCC" for m in inputs) or inputs[0].smiles == "CCC":
            raise RuntimeError("The size of tensor a (201) must match the size of tensor b (204)")
        return [[m.smiles] for m in inputs]

    monkeypatch.setattr(megan.MEGANModel, "_get_reactions", fake)
    model = object.__new__(megan_class())
    model.skipped = set()
    out = model._get_reactions([Molecule("CC"), Molecule("CCC"), Molecule("CCO")], 10)
    assert out == [["CC"], [], ["CCO"]]
    assert model.skipped == {"CCC"}


def test_other_runtime_errors_are_raised(monkeypatch):
    from syntheseus.reaction_prediction.inference import megan

    def fake(self, inputs, num_results):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(megan.MEGANModel, "_get_reactions", fake)
    model = object.__new__(megan_class())
    with pytest.raises(RuntimeError, match="CUDA"):
        model._get_reactions([Molecule("CC")], 10)
