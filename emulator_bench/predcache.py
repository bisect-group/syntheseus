"""Prediction cache: each unique model input is predicted once and reused by every split row."""

from __future__ import annotations

import hashlib
import json
import pickle
from itertools import islice
from pathlib import Path

from emulator_bench.common import discover_splits, load_hparams, open_dataset


def cache_dir(out_root: Path, model: str, direction: str, num_top: int) -> Path:
    """One folder per model, direction, top-N and checkpoint plus model settings."""
    hp = load_hparams()["models"][model][direction]
    ident = json.dumps({k: hp[k] for k in ("checkpoint", "model_kwargs")}, sort_keys=True)
    digest = hashlib.sha256(ident.encode()).hexdigest()[:8]
    return out_root / f"cache/predictions_v2/{direction}/{model}/top{num_top}_{digest}"


def input_of(sample, direction: str):
    """The model input of a sample and its key. The key is the canonical SMILES (bag: sorted, joined)."""
    from syntheseus.interface.molecule import molecule_bag_to_smiles

    if direction == "forward":
        return sample.reactants, molecule_bag_to_smiles(sample.reactants)
    [product] = sample.products
    return product, product.smiles


def unique_inputs(out_root: Path, model: str, direction: str, folds, limit=None) -> dict:
    """Unique inputs over all split files, sorted by key. `limit` keeps the first N rows per file."""
    from syntheseus.reaction_prediction.data.dataset import DataFold

    found = {}
    for split in discover_splits():
        for fold in folds:
            dataset, _ = open_dataset(out_root, split, fold, model, direction)
            for sample in islice(dataset[DataFold(fold)], limit):
                obj, key = input_of(sample, direction)
                found[key] = obj
    return dict(sorted(found.items()))


def batch_path(folder: Path, keys: list[str]) -> Path:
    return folder / f"batch_{hashlib.sha256(chr(10).join(keys).encode()).hexdigest()[:16]}.pkl"


def save_batch(
    folder: Path,
    keys: list[str],
    predictions,
    seconds: float,
    skipped: frozenset[str] = frozenset(),
) -> None:
    """`skipped` are the keys the model refused (no prediction, e.g. MEGAN above its atom limit)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = batch_path(folder, keys)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(
        pickle.dumps(
            {
                "seconds": seconds,
                "skipped": sorted(skipped),
                "predictions": {k: pickle.dumps(list(p)) for k, p in zip(keys, predictions)},
            }
        )
    )
    tmp.replace(path)


def load_cache(folder: Path) -> tuple[dict[str, bytes], float, set[str]]:
    """All cached predictions (key -> pickled list), the summed batch seconds and the skipped keys."""
    cached, seconds, skipped = {}, 0.0, set()
    for path in sorted(folder.glob("batch_*.pkl")):
        batch = pickle.loads(path.read_bytes())
        cached.update(batch["predictions"])
        seconds += batch["seconds"]
        skipped.update(batch.get("skipped", ()))
    return cached, seconds, skipped


class CachedModel:
    """Serves cached predictions to syntheseus `compute_metrics`. Each call returns fresh objects."""

    def __init__(self, folder: Path, direction: str, num_top: int):
        self.cached, _, self.skipped = load_cache(folder)
        self.n_skipped = 0  # rows served without a prediction because the model skipped the input
        self.direction = direction
        self.num_top = num_top

    def is_forward(self) -> bool:
        return self.direction == "forward"

    def get_parameters(self):
        return []

    def get_model_info(self) -> dict:
        return {}

    def __call__(self, inputs, num_results: int):
        assert num_results == self.num_top
        keys = [self._key(i) for i in inputs]
        self.n_skipped += sum(k in self.skipped for k in keys)
        try:
            return [pickle.loads(self.cached[k]) for k in keys]
        except KeyError as e:
            raise SystemExit(f"No cached prediction for {e}. Run emulator_bench.predict first.")

    def _key(self, obj) -> str:
        from syntheseus.interface.molecule import Molecule, molecule_bag_to_smiles

        return obj.smiles if isinstance(obj, Molecule) else molecule_bag_to_smiles(obj)
