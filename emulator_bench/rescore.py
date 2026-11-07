"""Extra retro metrics from the saved predictions of every finished run (writes rescore.json per run).

For each run: top-k (TOP_KS up to the run's num_top_results) and MRR in four variants.

- per_reaction: each benchmark row has its own reactant target (as syntheseus compute_metrics).
- multi_reference: one unit per unique main product; a prediction is correct if it matches the
  target of any row of the split file with that product. Rows that share a product get the same
  predictions, so the first row's list is used.
- stereo: exact match of the canonical SMILES bags, as syntheseus Reaction equality.
- no_stereo: stereo is removed from predictions and targets first. The ranked list is then
  deduplicated (first occurrence kept), so stereoisomers of one prediction count once.

Every variant is computed on all curated rows ("all") and on the common test set ("common", the
rows every model can take in every seed; emulator_bench.common_set). The stereo per_reaction numbers
on all rows must equal result.json (syntheseus compute_metrics), otherwise the run fails the check.
Run from baselines/syntheseus after common_set: $PY -m emulator_bench.rescore (all seed folders).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from emulator_bench.common import (
    DONE_MARKER,
    OUT_ROOT,
    TOP_KS,
    atomic_write_text,
    discover_splits,
    retro_file,
)

RESCORE_FILE = "rescore.json"


@lru_cache(maxsize=None)
def canonical(smiles: str, stereo: bool) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    if not stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def bag(smiles: list[str], stereo: bool) -> tuple[str, ...]:
    """Order-free multiset of molecules, as a sorted tuple."""
    return tuple(sorted(canonical(s, stereo) for s in smiles))


def ranked(predictions: list[list[str]], stereo: bool) -> list[tuple[str, ...]]:
    """Predicted reactant bags in rank order; without stereo, repeats are removed."""
    bags = [bag(p, stereo) for p in predictions]
    return bags if stereo else list(dict.fromkeys(bags))


def first_hit(bags: list[tuple[str, ...]], targets: set[tuple[str, ...]]) -> int | None:
    return next((i for i, b in enumerate(bags) if b in targets), None)


def summarize(ranks: list[int | None], num_top: int) -> dict:
    n = len(ranks)
    out = {
        f"top{k}": sum(r is not None and r < k for r in ranks) / n for k in TOP_KS if k <= num_top
    }
    out["mrr"] = sum(1 / (r + 1) for r in ranks if r is not None) / n
    return out


def rescore_records(
    records: list[dict], targets: list[str], num_top: int, side: str = "reactants"
) -> dict:
    """Metrics of one run. `records` are predictions.jsonl rows, `targets` the '.'-joined molecules.

    `side` is the half of each predicted reaction being scored: the reactants for retro, the
    products for forward.
    """
    preds = [[[m["smiles"] for m in p[side]] for p in r.get("predictions", [])] for r in records]
    products = [r["input"] for r in records]
    out: dict = {"n_reactions": len(records), "n_products": len(set(products))}
    for stereo in (True, False):
        gold = [bag(t.split("."), stereo) for t in targets]
        refs: dict[str, set] = {}
        for product, g in zip(products, gold):
            refs.setdefault(product, set()).add(g)
        lists = [ranked(p, stereo) for p in preds]
        per_reaction = [first_hit(b, {g}) for b, g in zip(lists, gold)]
        first_row = {}
        for i, product in enumerate(products):
            first_row.setdefault(product, i)
        multi = [first_hit(lists[i], refs[product]) for product, i in first_row.items()]
        out["stereo" if stereo else "no_stereo"] = {
            "per_reaction": summarize(per_reaction, num_top),
            "multi_reference": summarize(multi, num_top),
        }
        if stereo:  # self-check against syntheseus: same first correct rank on every row
            syn = [
                next((i for i, c in enumerate(r["ground_truth_correct"]) if c), None)
                for r in records
            ]
            if syn != per_reaction:
                bad = next(i for i, (a, b) in enumerate(zip(syn, per_reaction)) if a != b)
                raise RuntimeError(
                    f"row {bad}: syntheseus rank {syn[bad]}, rescore {per_reaction[bad]}"
                )
    return out


def runs_roots(out_root: Path) -> list[Path]:
    """Run folders of every seed: out_root (seed 0) and out_root/seed<N>."""
    return [out_root] + sorted(p for p in out_root.glob("seed*") if p.is_dir())


def jsonl_mask(curated, model: str, direction: str):
    """Which curated rows were written to the JSONL this run read.

    `common.open_dataset` applies a model's own preprocessing filter, which removes rows from
    anywhere in the file, so the predictions line up with this mask rather than with the file.
    """
    from emulator_bench.common import rsmiles_forward_ok

    if (model, direction) == ("RootAligned", "forward"):
        return rsmiles_forward_ok(curated)
    return [True] * len(curated)


def rescore_run(run: Path) -> Path:
    result = json.loads((run / "result.json").read_text())
    split = next(
        s
        for s in discover_splits()
        if s.group == result["split_group"] and s.name == result["split_name"]
    )
    direction = result.get("direction", "retro")
    # retro predicts the reactants of the curated product; forward predicts the product
    side = "reactants" if direction == "retro" else "products"
    target_column = "reactants" if direction == "retro" else "product"
    curated = pd.read_parquet(retro_file(split, result["fold"]), columns=["product", "reactants"])
    with open(run / "predictions.jsonl") as f:
        records = [json.loads(line) for line in f]
    if len(records) != result["n"]:
        raise RuntimeError(f"{run}: {len(records)} predictions, expected {result['n']}")

    # A model's own preprocessing filter drops rows from anywhere in the file, so the predictions
    # line up with the rows that were written, not with the first N curated rows.
    kept = np.flatnonzero(np.asarray(jsonl_mask(curated, result["model"], direction)))
    kept = kept[: len(records)]
    if len(kept) != len(records):
        raise RuntimeError(f"{run}: {len(records)} predictions but {len(kept)} rows were written")
    curated = curated.iloc[kept]
    if direction == "retro":  # the retro input is the curated product; a forward input is a bag
        for i, (r, product) in enumerate(zip(records, curated["product"])):
            if canonical(r["input"], True) != canonical(product, True):
                raise RuntimeError(
                    f"{run}: row {i} input {r['input']} is not curated product {product}"
                )
    num_top = result["num_top_results"]
    out = {"seed": result.get("seed", 0), "direction": direction}
    out["all"] = rescore_records(records, list(curated[target_column]), num_top, side)
    for k, v in out["all"]["stereo"]["per_reaction"].items():
        if abs(v - result[k]) > 1e-9:
            raise RuntimeError(f"{run}: {k} {v} differs from result.json {result[k]}")
    common_path = retro_file(split, result["fold"]).with_name(f"{result['fold']}_common.parquet")
    if not common_path.exists():
        raise SystemExit(f"Missing {common_path}. Run emulator_bench.common_set first.")
    common = set(pd.read_parquet(common_path, columns=["row"])["row"])
    rows = [i for i, row in enumerate(kept) if row in common]
    out["common"] = rescore_records(
        [records[i] for i in rows],
        [curated[target_column].iloc[i] for i in rows],
        num_top,
        side,
    )
    path = run / RESCORE_FILE
    atomic_write_text(path, json.dumps(out, indent=2) + "\n")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    p.add_argument(
        "--runs_root",
        type=Path,
        nargs="+",
        default=None,
        help="Default: out_root and every out_root/seed<N>",
    )
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--force", action="store_true", help="Recompute existing rescore.json files")
    args = p.parse_args()
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    runs = [
        d.parent
        for root in args.runs_root or runs_roots(args.out_root)
        for d in sorted(root.glob(f"*/*/*/*/*/{DONE_MARKER}"))
        if args.force or not (d.parent / RESCORE_FILE).exists()
    ]
    with ProcessPoolExecutor(args.procs) as pool:
        for path in pool.map(rescore_run, runs):
            print(path, flush=True)
    print(f"{len(runs)} runs rescored")


if __name__ == "__main__":
    main()
