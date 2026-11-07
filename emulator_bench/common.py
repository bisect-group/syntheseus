"""Paths, split discovery and JSONL conversion shared by the wrapper scripts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EMULATOR_ROOT = Path(__file__).resolve().parents[3]
# Checkouts differ in whether the dataset folder sits under data/processed/datasets/ or data/processed/.
DATA_DIR = next(
    (
        p
        for p in (
            EMULATOR_ROOT / "data/processed/datasets/reaction_outcome_dataset",
            EMULATOR_ROOT / "data/processed/reaction_outcome_dataset",
        )
        if p.exists()
    ),
    EMULATOR_ROOT / "data/processed/datasets/reaction_outcome_dataset",
)
# Curated retro split files (emulator_bench.retro_targets), same layout as DATA_DIR.
RETRO_DIR = DATA_DIR.with_name("reaction_outcome_dataset_retro")
OUT_ROOT = EMULATOR_ROOT / "outputs/baselines/syntheseus"
FOLDS = ("train", "val", "test")
DONE_MARKER = "DONE"
TOP_KS = (1, 10, 50)  # reported top-k, up to the run's num_top_results
CACHE_VERSION = 1
# Validity and JSONL caches. 2: the target is the largest product. 3: retro reads the curated
# split files (main product that is not a cofactor, atom-mapped reactants).
TARGET_VERSION = 3

# Retrosynthesis models with their syntheseus default (USPTO-50K) checkpoints.
RETRO_MODELS = (
    "Chemformer",
    "GLN",
    "Graph2Edits",
    "LocalRetro",
    "MEGAN",
    "MHNreact",
    "RetroKNN",
    "RootAligned",
)
FORWARD_MODELS = ("Chemformer", "MEGAN", "RootAligned")
HPARAMS_PATH = Path(__file__).with_name("default_hparams.json")


@dataclass(frozen=True)
class SplitFile:
    group: str  # e.g. random_splits_grouped_rxn_key or drfp_tanimoto_splits
    name: str  # threshold_* or "all"
    folder: Path

    def fold_path(self, fold: str) -> Path:
        return self.folder / f"{fold}.parquet"


def discover_splits(data_dir: Path = DATA_DIR) -> list[SplitFile]:
    """Find every split folder on disk. Threshold folders are never hardcoded."""
    splits: list[SplitFile] = []
    for group in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if all((group / f"{f}.parquet").exists() for f in FOLDS):
            splits.append(SplitFile(group.name, "all", group))
        for sub in sorted(group.glob("threshold_*")):
            if all((sub / f"{f}.parquet").exists() for f in FOLDS):
                splits.append(SplitFile(group.name, sub.name, sub))
    return splits


def run_dir(out_root: Path, direction: str, model: str, split: SplitFile, fold: str) -> Path:
    return out_root / direction / model / split.group / split.name / fold


def drfp_threshold(data_dir: Path = DATA_DIR) -> float:
    """The DRFP split threshold, read from the threshold folder name on disk."""
    names = sorted((data_dir / "drfp_tanimoto_splits").glob("threshold_*"))
    if len(names) != 1:
        raise RuntimeError(f"Expected one DRFP threshold folder, found {[n.name for n in names]}")
    return float(names[0].name.removeprefix("threshold_"))


def load_hparams() -> dict:
    return json.loads(HPARAMS_PATH.read_text())


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _smiles_list(smiles: str) -> list[str]:
    return smiles.split(".")


def split_reactions(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Unique reactions (rxn_key, reactants, products) over every split file on disk."""
    frames = [
        pd.read_parquet(s.fold_path(f), columns=["rxn_key", "reactants", "products"])
        for s in discover_splits(data_dir)
        for f in FOLDS
    ]
    return pd.concat(frames).drop_duplicates("rxn_key").reset_index(drop=True)


def largest_product(smiles: str) -> tuple[str, int, bool]:
    """Product with the most heavy atoms. Ties: longer canonical SMILES, then lexicographic order.

    Returns (product SMILES, number of products, whether the rule hit a tie).
    """
    from rdkit import Chem

    products = _smiles_list(smiles)
    ranked = []
    for s in products:
        mol = Chem.MolFromSmiles(s)
        ranked.append(((mol.GetNumHeavyAtoms(), len(Chem.MolToSmiles(mol)), s), s))
    ranked.sort(key=lambda t: (-t[0][0], -t[0][1], t[0][2]))
    tie = len(ranked) > 1 and ranked[0][0][:2] == ranked[1][0][:2]
    return ranked[0][1], len(products), tie


def validity_table(out_root: Path, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Per split-file reaction: whether RDKit parses every molecule, plus the largest-product target. Cached."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    df = split_reactions(data_dir)
    key = hashlib.sha256("\n".join(df["rxn_key"]).encode()).hexdigest()[:16]
    path = out_root / f"cache/validity_v{TARGET_VERSION}/validity_{key}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    def ok(smiles: str) -> bool:
        return bool(smiles) and all(Chem.MolFromSmiles(s) is not None for s in _smiles_list(smiles))

    valid = [ok(r) and ok(p) for r, p in zip(df["reactants"], df["products"])]
    main = [largest_product(p) if v else ("", 0, False) for p, v in zip(df["products"], valid)]
    table = pd.DataFrame(
        {
            "rxn_key": df["rxn_key"],
            "valid": valid,
            "main_product": [m[0] for m in main],
            "n_products": [m[1] for m in main],
            "tie": [m[2] for m in main],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    table.to_parquet(tmp)
    tmp.replace(path)
    return table


def retro_file(split: SplitFile, fold: str, retro_dir: Path = RETRO_DIR) -> Path:
    """Curated retro split file of one split folder and fold (see emulator_bench.retro_targets)."""
    folder = (
        retro_dir / split.group if split.name == "all" else retro_dir / split.group / split.name
    )
    return folder / f"{fold}.parquet"


def write_curated_jsonl(
    out_root: Path, split: SplitFile, fold: str, direction: str, extra_filter=None
) -> tuple[Path, dict]:
    """Write the JSONL of one split file from its curated retro file. Returns (path, counts).

    Both directions read the same curated rows, because forward synthesis is exactly the reverse of
    the retro task: the curated reactant set and the main product swap roles. A row holds both, and
    syntheseus takes the input from the model's direction, so one file serves both.

    `extra_filter(df) -> bool mask` drops rows a model's own preprocessing cannot take. Rows the
    curation removed (`<fold>_dropped.parquet`) are not part of the benchmark; they are reported in
    the counts but not scored.
    """
    src = retro_file(split, fold)
    if not src.exists():
        raise SystemExit(
            f"Missing {src}. Build the curated retro files first (emulator_bench.retro_targets)."
        )
    df = pd.read_parquet(src)
    dropped = pd.read_parquet(src.with_name(f"{fold}_dropped.parquet"))
    kept = df if extra_filter is None else df[np.asarray(extra_filter(df))]
    tag = "" if extra_filter is None else f"_{extra_filter.__name__}"
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
    path = (
        out_root / f"cache/jsonl_v{TARGET_VERSION}/{direction}{tag}"
        f"/{split.group}/{split.name}/{fold}_{digest}.jsonl"
    )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = (
            json.dumps(
                {
                    "rxn_key": k,
                    "reactants": [{"smiles": s} for s in _smiles_list(r)],
                    "products": [{"smiles": p}],
                }
            )
            for k, r, p in zip(kept["rxn_key"], kept["reactants"], kept["product"])
        )
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(path)
    counts = {
        "n_kept": len(kept),
        "n_dropped": len(df) - len(kept),
        "n_multi_product": int((kept["minor_products"] != "").sum()),
        "n_multi_tie": int(kept["tie"].sum()),
        "n_curation_dropped": len(dropped),
        "n_curation_merged": int((kept["n_merged"] - 1).sum()),
    }
    return path, counts


def write_jsonl(
    out_root: Path, split: SplitFile, fold: str, direction: str, extra_filter=None
) -> tuple[Path, dict]:
    """Write the syntheseus JSONL for one split file. Returns (path, counts).

    Both directions read the curated retro files, so forward synthesis is scored on exactly the
    reverse of the retro task: the same rows, with the curated reactant set and the main product
    swapping roles. `extra_filter(df) -> bool mask` drops rows the original model cannot take.
    """
    return write_curated_jsonl(out_root, split, fold, direction, extra_filter)


def rsmiles_forward_ok(df):
    """Rows the original R-SMILES forward preprocessing keeps (generate_RtoP_data.py `multi_process`).

    The curated frame names the single target `product`; the raw frame used `products`.
    """
    from rdkit import Chem

    def keep(reactants: str, product: str) -> bool:
        return (
            Chem.MolFromSmiles(reactants).GetNumAtoms() >= 5  # small_r
            and Chem.MolFromSmiles(product).GetNumAtoms() > 1  # small_p, on the main product
        )

    products = df["product"] if "product" in df else df["products"]
    return [keep(r, p) for r, p in zip(df["reactants"], products)]


def open_dataset(out_root: Path, split: SplitFile, fold: str, model: str, direction: str):
    """Syntheseus dataset of one split file (rows the model can take). Returns (dataset, counts)."""
    from syntheseus.reaction_prediction.data.dataset import DiskReactionDataset
    from syntheseus.reaction_prediction.data.reaction_sample import ReactionSample

    extra = rsmiles_forward_ok if (model, direction) == ("RootAligned", "forward") else None
    jsonl, counts = write_jsonl(out_root, split, fold, direction, extra)
    data_dir = jsonl.with_suffix(".data")
    data_dir.mkdir(exist_ok=True)
    link = data_dir / f"{fold}.jsonl"
    if not link.exists():
        link.symlink_to(jsonl)
    return DiskReactionDataset(data_dir, sample_cls=ReactionSample), counts
