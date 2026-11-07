"""Common retro test set: the curated rows that every model can take, in every seed.

All models are scored on the same rows. A curated row is left out for every model when its main
product or its target is beyond what one model can take:

- Any model: an input it skipped at run time in any seed (the prediction cache's `skipped`): MEGAN
  beam search crashes, or an input that did not fit in GPU memory even alone (predict.py).
- MEGAN: a product with more than `n_max_atoms` heavy atoms (default_hparams.json, 200).
- Chemformer: a product or target longer than max_seq_len = 512 tokens with the begin and end tokens
  (checkpoint hyper-parameters). The wrapper truncates longer inputs, and the decoder stops at 512.
- RootAligned: a target longer than max_length = 500 tokens (translate options in the checkpoint).

An empty prediction list (e.g. no template applies) is a miss, not a reason to leave a row out.
Writes `<fold>_common.parquet` (the kept rows, with `row`: their index in `<fold>.parquet`) and
`<fold>_common_excluded.parquet` (left-out rows and why) next to each curated file. Run after the
predictions of every model and seed exist:

    $PY -m emulator_bench.common_set --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from emulator_bench.common import (
    OUT_ROOT,
    RETRO_DIR,
    RETRO_MODELS,
    atomic_write_text,
    discover_splits,
    load_hparams,
    retro_file,
)
from emulator_bench.predcache import cache_dir, load_cache

CHEMFORMER_MAX_SEQ_LEN = 512  # Chemformer_backward/last.ckpt hyper_parameters["max_seq_len"]
ROOT_ALIGNED_MAX_LENGTH = 500  # RootAligned_backward translate yml "max_length"
SMILES_TOKEN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)


def n_tokens(smiles: str) -> int:
    return len(SMILES_TOKEN.findall(smiles))


def canonical(smiles: str) -> str:
    """The model input key: syntheseus canonical SMILES."""
    from syntheseus.interface.molecule import Molecule

    return Molecule(smiles).smiles


def heavy_atoms(smiles: str) -> int:
    from rdkit import Chem

    return Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()


def exclusion_reasons(
    product: str, reactants: str, skipped: dict[tuple[str, int], set[str]]
) -> list[str]:
    """Why one curated row is left out of the common set (empty: it stays).

    `skipped` maps (model, seed) to the inputs that model skipped at run time.
    """
    reasons = []
    n_max_atoms = load_hparams()["models"]["MEGAN"]["retro"]["model_kwargs"]["n_max_atoms"]
    if (n := heavy_atoms(product)) > n_max_atoms:
        reasons.append(f"MEGAN: product has {n} > {n_max_atoms} heavy atoms")
    key = canonical(product)
    for (model, seed), keys in sorted(skipped.items()):
        if key in keys:
            reasons.append(f"{model}: skipped at run time (seed {seed})")
    for what, smiles in (("product", product), ("target", reactants)):
        if (n := n_tokens(smiles) + 2) > CHEMFORMER_MAX_SEQ_LEN:
            reasons.append(f"Chemformer: {what} has {n} > {CHEMFORMER_MAX_SEQ_LEN} tokens")
    if (n := n_tokens(reactants)) > ROOT_ALIGNED_MAX_LENGTH:
        reasons.append(f"RootAligned: target has {n} > {ROOT_ALIGNED_MAX_LENGTH} tokens")
    return reasons


def check_predictions(out_root: Path, keys: set[str], seeds, models=RETRO_MODELS) -> dict:
    """Skipped inputs per (model, seed); fails if any model lacks a prediction in any seed."""
    num_top = load_hparams()["num_top_results"]["retro"]
    found = {}
    for seed in seeds:
        for model in models:
            cached, _, skipped = load_cache(cache_dir(out_root, model, "retro", num_top, seed))
            missing = keys - set(cached)
            if missing:
                raise SystemExit(
                    f"{model} seed {seed}: {len(missing)} inputs have no prediction yet "
                    f"(e.g. {next(iter(missing))}). Run all predictions first."
                )
            found[(model, seed)] = skipped & keys
    return found


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--folds", nargs="+", default=["test"])
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    p.add_argument(
        "--limit", type=int, default=None, help="Check predictions of the first N rows only (tests)"
    )
    args = p.parse_args()
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    files = [(s, f) for s in discover_splits() for f in args.folds]
    curated = {(s.group, s.name, f): pd.read_parquet(retro_file(s, f)) for s, f in files}
    keys = {canonical(x) for df in curated.values() for x in df["product"].iloc[: args.limit]}
    skipped = check_predictions(args.out_root, keys, args.seeds)

    summary = {"seeds": args.seeds, "files": {}}
    for split, fold in files:
        df = curated[(split.group, split.name, fold)]
        reasons = [exclusion_reasons(p, r, skipped) for p, r in zip(df["product"], df["reactants"])]
        keep = [not r for r in reasons]
        kept = df[keep].copy()
        kept.insert(0, "row", [i for i, k in enumerate(keep) if k])
        kept["n_references"] = kept["product"].map(kept["product"].value_counts())
        excluded = df[[not k for k in keep]][["rxn_key", "product", "reactants"]].copy()
        excluded["reasons"] = ["; ".join(r) for r in reasons if r]
        path = retro_file(split, fold)
        kept.to_parquet(path.with_name(f"{fold}_common.parquet"), index=False)
        excluded.to_parquet(path.with_name(f"{fold}_common_excluded.parquet"), index=False)
        # rows per model: a row a model skipped in several seeds counts once
        by_model = Counter(m for r in reasons for m in {x.split(":")[0] for x in r})
        summary["files"][f"{split.group}/{split.name}/{fold}"] = {
            "curated_rows": len(df),
            "common_rows": len(kept),
            "common_products": int(kept["product"].nunique()),
            "excluded_rows_by_model": dict(by_model),
        }
        print(
            f"{split.group}/{split.name} {fold}: {len(kept)} of {len(df)} rows, left out {dict(by_model)}"
        )
    atomic_write_text(RETRO_DIR / "common_set.json", json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
