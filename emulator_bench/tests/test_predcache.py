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


def test_cache_dir_depends_on_seed(tmp_path):
    seed0 = cache_dir(tmp_path, "RootAligned", "retro", 50)
    assert seed0 == cache_dir(tmp_path, "RootAligned", "retro", 50, seed=0)
    assert not seed0.name.endswith("_seed0")  # seed 0 keeps the original folder
    assert cache_dir(tmp_path, "RootAligned", "retro", 50, seed=1).name == f"{seed0.name}_seed1"


def test_input_seed():
    from emulator_bench.run_model import input_seed

    assert input_seed("CCO") == input_seed("CCO", 0)  # seed 0: the original CRC32 of the SMILES
    assert len({input_seed("CCO", s) for s in (0, 1, 2)}) == 3


def test_cache_dir_tag_separates_finetuned_weights(tmp_path):
    base = cache_dir(tmp_path, "LocalRetro", "retro", 50)
    tagged = cache_dir(tmp_path, "LocalRetro", "retro", 50, tag="finetuned_random")
    assert base != tagged
    assert tagged.name == f"{base.name}_finetuned_random"
    # tag and seed compose, and the tag comes first so seed stays the suffix
    both = cache_dir(tmp_path, "LocalRetro", "retro", 50, seed=1, tag="finetuned_random")
    assert both.name == f"{base.name}_finetuned_random_seed1"


def test_unique_inputs_can_be_restricted_to_one_split(monkeypatch, tmp_path):
    """A fine-tuned run only needs its own split's inputs, not every split file on disk."""
    from dataclasses import dataclass

    from emulator_bench import predcache

    @dataclass
    class FakeSplit:
        group: str
        name: str

    seen = []

    def fake_open_dataset(out_root, split, fold, model, direction):
        from syntheseus.reaction_prediction.data.dataset import DataFold

        seen.append(f"{split.group}/{split.name}")
        return {DataFold(fold): []}, {}

    monkeypatch.setattr(
        predcache, "discover_splits", lambda: [FakeSplit("a", "all"), FakeSplit("b", "all")]
    )
    monkeypatch.setattr(predcache, "open_dataset", fake_open_dataset)
    predcache.unique_inputs(tmp_path, "LocalRetro", "retro", ["test"], splits=["a"])
    assert seen == ["a/all"]
    seen.clear()
    predcache.unique_inputs(tmp_path, "LocalRetro", "retro", ["test"])
    assert seen == ["a/all", "b/all"]
