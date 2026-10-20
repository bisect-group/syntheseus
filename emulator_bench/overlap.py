"""Training-corpus overlap audit: DRFP similarity and exact reaction matches per split file.

Fingerprints follow the Leak-CURBER DRFP splitter (src/data/components/embedders/drfp.py and
src/data/components/splitters/reaction_tanimoto.py): DRFP on the `rxn_key` string with
radius 3, min radius 0, 2048 folded bits, no rings, no hydrogens. Max Tanimoto is the float32 matrix
product of the splitter, with a float64 quotient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from emulator_bench.common import (
    CACHE_VERSION,
    DATA_DIR,
    FOLDS,
    OUT_ROOT,
    atomic_write_text,
    discover_splits,
    drfp_threshold,
    split_reactions,
)

DRFP = dict(n_folded_length=2048, min_radius=0, radius=3, rings=False, include_hydrogens=False)
QUERY_BLOCK = 64
ENCODE_CHUNK = 2000
WORKERS = min(16, len(os.sched_getaffinity(0)))


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _encode_chunk(reactions: list[str]) -> np.ndarray:
    from drfp import DrfpEncoder

    return np.asarray(
        DrfpEncoder.encode(reactions, show_progress_bar=False, **DRFP), dtype=np.uint8
    )


def encode_drfp(reactions: list[str]) -> np.ndarray:
    """Binary DRFP matrix (n, 2048) as uint8. Each reaction is encoded on its own, so chunking changes nothing."""
    chunks = [reactions[i : i + ENCODE_CHUNK] for i in range(0, len(reactions), ENCODE_CHUNK)]
    with Pool(WORKERS) as pool:
        return np.concatenate(pool.map(_encode_chunk, chunks))


def cached_fingerprints(cache_dir: Path, name: str, reactions: list[str]) -> np.ndarray:
    """Packed-bit cache. The key covers the DRFP settings and the reaction list."""
    key = hashlib.sha256(json.dumps([DRFP, reactions]).encode()).hexdigest()[:16]
    path = cache_dir / f"drfp_v{CACHE_VERSION}" / f"{name}_{key}.npy"
    if path.exists():
        return np.unpackbits(np.load(path), axis=1)[:, : DRFP["n_folded_length"]]
    fps = encode_drfp(reactions)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, np.packbits(fps, axis=1))
    tmp.replace(path)
    return fps


def max_tanimoto(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Max Tanimoto of each query to the corpus, as float64 (same arithmetic as the splitter)."""
    corpus_f = (corpus > 0).astype(np.float32)
    corpus_sums = corpus_f.sum(axis=1)
    out = np.zeros(len(queries), dtype=np.float64)

    def block_max(start: int) -> None:
        block = (queries[start : start + QUERY_BLOCK] > 0).astype(np.float32)
        inter = block @ corpus_f.T
        union = block.sum(axis=1)[:, None] + corpus_sums[None, :] - inter
        sims = np.divide(
            inter, union, out=np.zeros(inter.shape, np.float64), where=union > 0, dtype=np.float64
        )
        out[start : start + len(block)] = sims.max(axis=1)

    with threadpool_limits(1), ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(block_max, range(0, len(queries), QUERY_BLOCK)))
    return out


def canonical_reaction(reactants: str, products: str) -> str:
    """Order-free reaction key: sorted canonical molecules, stereo kept, atom maps removed."""
    from rdkit import Chem

    def side(smiles: str) -> str:
        mols = []
        for s in smiles.split("."):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                return ""
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            mols.append(Chem.MolToSmiles(mol))
        return ".".join(sorted(mols))

    left, right = side(reactants), side(products)
    return f"{left}>>{right}" if left and right else ""


def _canonical_pair(pair: tuple[str, str]) -> str:
    return canonical_reaction(*pair)


def canonical_reactions(reactants, products) -> list[str]:
    with Pool(WORKERS) as pool:
        return pool.map(_canonical_pair, list(zip(reactants, products)), chunksize=500)


def corpus_reactions(path: Path) -> pd.DataFrame:
    """Corpus parquet with columns reactants and products (atom maps removed)."""
    return pd.read_parquet(path)


def audit(
    out_root: Path, corpus_id: str, threshold: float, data_dir: Path = DATA_DIR
) -> list[dict]:
    corpus = corpus_reactions(out_root / "corpora" / f"{corpus_id}.parquet")
    corpus_keys = [f"{r}>>{p}" for r, p in zip(corpus["reactants"], corpus["products"])]
    corpus_fps = cached_fingerprints(out_root / "cache", corpus_id, corpus_keys)
    corpus_exact = set(canonical_reactions(corpus["reactants"], corpus["products"]))
    log(f"{corpus_id}: {len(corpus)} corpus reactions, canonical set built")

    dataset = split_reactions(data_dir)
    fps = cached_fingerprints(out_root / "cache", "dataset", dataset["rxn_key"].tolist())
    log(f"{corpus_id}: fingerprints ready ({len(fps)} dataset reactions)")
    sims = pd.Series(max_tanimoto(fps, corpus_fps), index=dataset["rxn_key"])
    log(f"{corpus_id}: max similarity done")
    zero_fp = pd.Series(fps.sum(axis=1) == 0, index=dataset["rxn_key"])
    left_right = dataset["rxn_key"].str.split(">>", n=1, expand=True)
    exact_key = pd.Series(
        [c in corpus_exact for c in canonical_reactions(left_right[0], left_right[1])],
        index=dataset["rxn_key"],
    )
    exact_full = pd.Series(
        [c in corpus_exact for c in canonical_reactions(dataset["reactants"], dataset["products"])],
        index=dataset["rxn_key"],
    )

    rows = []
    for split in discover_splits(data_dir):
        for fold in FOLDS:
            keys = pd.read_parquet(split.fold_path(fold), columns=["rxn_key"])["rxn_key"]
            frame = pd.DataFrame(
                {
                    "rxn_key": keys.to_numpy(),
                    "max_drfp_tanimoto": sims.loc[keys].to_numpy(),
                    "exact_match_rxn_key": exact_key.loc[keys].to_numpy(),
                    "exact_match_full_reaction": exact_full.loc[keys].to_numpy(),
                }
            )
            path = out_root / "overlap" / corpus_id / split.group / split.name / f"{fold}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path.with_name(path.name + ".tmp"))
            path.with_name(path.name + ".tmp").replace(path)
            n = len(frame)
            rows.append(
                {
                    "corpus": corpus_id,
                    "n_corpus": len(corpus),
                    "split_group": split.group,
                    "split_name": split.name,
                    "fold": fold,
                    "n": n,
                    "threshold": threshold,
                    "pct_ge_threshold": 100
                    * float((frame["max_drfp_tanimoto"] >= threshold).mean()),
                    "mean_sim": float(frame["max_drfp_tanimoto"].mean()),
                    "median_sim": float(frame["max_drfp_tanimoto"].median()),
                    "n_zero_fingerprint": int(zero_fp.loc[keys].sum()),
                    "exact_rxn_key": int(frame["exact_match_rxn_key"].sum()),
                    "pct_exact_rxn_key": 100 * float(frame["exact_match_rxn_key"].mean()),
                    "exact_full_reaction": int(frame["exact_match_full_reaction"].sum()),
                    "pct_exact_full_reaction": 100
                    * float(frame["exact_match_full_reaction"].mean()),
                }
            )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    p.add_argument(
        "--corpora", nargs="*", help="Corpus ids (default: every parquet in OUT/corpora)"
    )
    args = p.parse_args()
    threshold = drfp_threshold()
    ids = args.corpora or sorted(f.stem for f in (args.out_root / "corpora").glob("*.parquet"))
    rows = []
    for corpus_id in ids:
        rows += audit(args.out_root, corpus_id, threshold)
    table = pd.DataFrame(rows)
    atomic_write_text(args.out_root / "summary_overlap_by_corpus.csv", table.to_csv(index=False))
    print(table.to_string())


if __name__ == "__main__":
    main()
