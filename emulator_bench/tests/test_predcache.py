import pytest

from emulator_bench.predcache import CachedModel, cache_dir, input_of, save_batch
from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction


def prediction(product: str, reactants: str) -> SingleProductReaction:
    return SingleProductReaction(
        product=Molecule(product), reactants=Bag([Molecule(s) for s in reactants.split(".")])
    )


def test_cache_dir_depends_on_top_n_and_model(tmp_path):
    a = cache_dir(tmp_path, "LocalRetro", "retro", 10)
    assert a != cache_dir(tmp_path, "LocalRetro", "retro", 100)
    assert a != cache_dir(tmp_path, "MEGAN", "retro", 10)


def test_input_key_is_canonical_and_order_free():
    class Sample:
        reactants = Bag([Molecule("OC"), Molecule("CC")])
        products = Bag([Molecule("C(C)O")])

    assert input_of(Sample, "forward")[1] == input_of(Sample, "forward")[1] == "CC.CO"
    assert input_of(Sample, "retro")[1] == "CCO"


def test_cached_model_expands_duplicates_with_fresh_objects(tmp_path):
    save_batch(tmp_path, ["CCO", "CC"], [[prediction("CCO", "CC.O")], []], seconds=1.0)
    model = CachedModel(tmp_path, "retro", 10)
    first, second, third = model([Molecule("OCC"), Molecule("CC"), Molecule("OCC")], num_results=10)
    assert len(first) == len(third) == 1 and second == []
    first[0].metadata["ground_truth_match"] = True
    assert "ground_truth_match" not in third[0].metadata


def test_missing_input_stops(tmp_path):
    save_batch(tmp_path, ["CCO"], [[]], seconds=0.0)
    with pytest.raises(SystemExit):
        CachedModel(tmp_path, "retro", 10)([Molecule("CC")], num_results=10)


def test_skipped_keys_are_cached_and_counted(tmp_path):
    save_batch(tmp_path, ["CCO", "CC"], [[prediction("CCO", "CC.O")], []], 1.0, frozenset({"CC"}))
    model = CachedModel(tmp_path, "retro", 10)
    model([Molecule("CC"), Molecule("OCC"), Molecule("CC")], num_results=10)
    assert model.skipped == {"CC"} and model.n_skipped == 2
