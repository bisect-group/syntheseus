"""Training data for fine-tuning, built from the curated retro files and their atom maps.

Fine-tuning must learn exactly the task the benchmark scores: the curated main product in, the
curated reactant target out (emulator_bench.retro_targets). This module rebuilds, per curated row,
an atom-mapped sub-reaction `target reactants >> main product` out of the RXNMapper mapping of the
full reaction, which the template-based trainers need.

Atom maps are kept only for atoms that appear on both sides of the sub-reaction. A product atom
whose source was removed from the target (the oxygen of an oxygenation, which comes from O2) is
left unmapped, which is how a new atom is written in a training reaction.

Rows without a usable mapping cannot be used for template extraction and are dropped from the
training data; they stay in the test set, where the benchmark scores them as misses.

    $PY -m emulator_bench.finetune_data --model LocalRetro --splits random_splits_grouped_rxn_key
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from emulator_bench.common import (
    OUT_ROOT,
    SplitFile,
    atomic_write_text,
    discover_splits,
    retro_file,
)
from emulator_bench.retro_targets import (
    Prepared,
    load_cofactors,
    map_paths,
    prepare_all,
    prepare_row,
    requests_of,
    target_of,
)

FINETUNE_DIR = "finetune"


def _fragment_maps(fragment: str) -> set[int]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(fragment, sanitize=False)
    if mol is None:
        return set()
    return {a.GetAtomMapNum() for a in mol.GetAtoms()} - {0}


def _keep_maps(fragment: str, keep: set[int]) -> str | None:
    """The fragment with every atom map not in `keep` cleared."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(fragment, sanitize=False)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() not in keep:
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def mapped_subreaction(
    prep: Prepared, mapping: dict | None, target_reactants: tuple[str, ...]
) -> str | None:
    """`reactants>>product` of the curated task with consistent atom maps, or None if unavailable.

    Only the target reactants are on the left and only the main product on the right, so the
    mapped reaction is the task the benchmark scores, not the full enzymatic reaction.
    """
    if mapping is None or mapping["method"] not in ("full", "main"):
        return None
    left = mapping["mapped_rxn"].split(">>")[0].split(".")
    right = mapping["mapped_rxn"].split(">>")[1].split(".")
    if len(left) != len(prep.map_reactants):
        return None
    product_index = prep.main_fragment if mapping["method"] == "full" else 0
    if product_index >= len(right):
        return None
    wanted = set(target_reactants)
    frags = [
        f
        for f, i in zip(left, prep.map_reactants)
        if i is not None and prep.reactants[i].smiles in wanted
    ]
    if not frags:
        return None
    product = right[product_index]
    shared = _fragment_maps(product) & {m for f in frags for m in _fragment_maps(f)}
    if not shared:
        return None  # no atom of the product comes from the target: no template to extract
    kept = [_keep_maps(f, shared) for f in frags]
    mapped_product = _keep_maps(product, shared)
    if mapped_product is None or any(f is None for f in kept):
        return None
    return f"{'.'.join(kept)}>>{mapped_product}"


def subreactions(split: SplitFile, fold: str, prepared: dict, maps: dict, cof) -> pd.DataFrame:
    """Curated rows of one split file with their mapped sub-reaction (None when unavailable)."""
    df = pd.read_parquet(retro_file(split, fold))
    out = []
    for row in df.itertuples():
        prep = prepared.get(row.rxn_key)
        if prep is None or isinstance(prep, str):
            prep = prepare_row(row.rxn_key, row.rxn_smiles, cof)
        if isinstance(prep, str):
            out.append(None)
            continue
        mapping = maps.get(prep.full)
        target = target_of(prep, mapping, cof)
        out.append(mapped_subreaction(prep, mapping, target.reactants))
    df["mapped_rxn"] = out
    return df


def write_localretro(frames: dict[str, pd.DataFrame], out_dir: Path, dataset: str) -> dict:
    """LocalRetro's layout: <out_dir>/data/<dataset>/raw_{train,val,test}.csv, atom-mapped."""
    data_dir = out_dir / "data" / dataset
    data_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for fold, df in frames.items():
        usable = df[df["mapped_rxn"].notna()]
        table = pd.DataFrame(
            {
                "id": usable["rxn_key"].to_numpy(),
                "class": "UNK",
                "reactants>reagents>production": usable["mapped_rxn"].to_numpy(),
            }
        )
        # LocalRetro reads raw_val.csv for validation and raw_test.csv for testing.
        table.to_csv(data_dir / f"raw_{fold}.csv", index=False)
        counts[fold] = {"rows": len(df), "usable": len(usable)}
    return counts


def write_chemformer(frames: dict[str, pd.DataFrame], out_dir: Path, dataset: str) -> dict:
    """Chemformer's `uspto_50` pickle: RDKit molecules plus a `set` column.

    A sequence model needs no atom maps, so every curated row is used, not only the mapped ones.
    """
    from rdkit import Chem

    out_dir.mkdir(parents=True, exist_ok=True)
    fold_name = {"train": "train", "val": "valid", "test": "test"}
    parts, counts = [], {}
    for fold, df in frames.items():
        part = pd.DataFrame(
            {
                "reactants_mol": [Chem.MolFromSmiles(s) for s in df["reactants"]],
                "products_mol": [Chem.MolFromSmiles(s) for s in df["product"]],
                "reaction_type": "UNK",
                "set": fold_name[fold],
                "rxn_key": df["rxn_key"].to_numpy(),
            }
        )
        usable = part[part["reactants_mol"].notna() & part["products_mol"].notna()]
        counts[fold] = {"rows": len(df), "usable": len(usable)}
        parts.append(usable)
    table = pd.concat(parts, ignore_index=True)
    table.to_pickle(out_dir / f"{dataset}.pickle")
    return counts


def fully_map_products(mapped_rxn: str) -> str | None:
    """Give every product atom a map number, numbering the new ones past the existing maximum.

    RootAligned rejects a reaction unless every product atom carries a map number: it picks the
    root atom of the product by map number. Atoms whose source was removed from the target carry
    none (see `mapped_subreaction`), so they are numbered here. The numbers are fresh, so they
    match nothing on the reactant side and only serve as root labels.
    """
    from rdkit import Chem

    left, right = mapped_rxn.split(">>")
    product = Chem.MolFromSmiles(right, sanitize=False)
    if product is None:
        return None
    used = {int(n) for n in re.findall(r"(?<=:)\d+", mapped_rxn)}
    nxt = max(used, default=0) + 1
    for atom in product.GetAtoms():
        if not atom.GetAtomMapNum():
            atom.SetAtomMapNum(nxt)
            nxt += 1
    return f"{left}>>{Chem.MolToSmiles(product)}"


def write_rootaligned(frames: dict[str, pd.DataFrame], out_dir: Path, dataset: str) -> dict:
    """RootAligned reads LocalRetro's CSV layout, but needs every product atom mapped."""
    mapped = {}
    for fold, df in frames.items():
        part = df.copy()
        part["mapped_rxn"] = [
            fully_map_products(r) if isinstance(r, str) else None for r in part["mapped_rxn"]
        ]
        mapped[fold] = part
    return write_localretro(mapped, out_dir, dataset)


WRITERS = {
    "LocalRetro": write_localretro,
    "Chemformer": write_chemformer,
    "RootAligned": write_rootaligned,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", default="LocalRetro", choices=sorted(WRITERS))
    p.add_argument("--skip_maps", action="store_true", help="Sequence models need no atom maps")
    p.add_argument("--splits", nargs="*", default=None, help="Split group or group/name")
    p.add_argument("--folds", nargs="+", default=["train", "val", "test"])
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    args = p.parse_args()
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    cof = load_cofactors()
    if args.skip_maps:
        prepared_all, maps = {}, {}
    else:
        prepared_all = prepare_all(args.folds, cof)
        _, maps_path = map_paths(args.out_root, requests_of(prepared_all))
        if not maps_path.exists():
            raise SystemExit(
                f"Missing {maps_path}. Run retro_targets prepare and the mapper first."
            )
        maps = pd.read_parquet(maps_path).set_index("full").to_dict("index")

    splits = [
        s
        for s in discover_splits()
        if not args.splits or f"{s.group}/{s.name}" in args.splits or s.group in args.splits
    ]
    summary = {}
    for split in splits:
        prepared = {}
        for fold in args.folds:
            prepared.update(prepared_all.get((split.group, split.name, fold), {}))
        if args.skip_maps:
            frames = {f: pd.read_parquet(retro_file(split, f)) for f in args.folds}
        else:
            frames = {f: subreactions(split, f, prepared, maps, cof) for f in args.folds}
        dataset = f"{split.group}_{split.name}"
        out_dir = args.out_root / FINETUNE_DIR / args.model / dataset
        counts = WRITERS[args.model](frames, out_dir, dataset)
        summary[dataset] = counts
        print(f"{dataset}: {counts}", flush=True)
    atomic_write_text(
        args.out_root / FINETUNE_DIR / args.model / "data_summary.json",
        json.dumps(summary, indent=2) + "\n",
    )


if __name__ == "__main__":
    main()
